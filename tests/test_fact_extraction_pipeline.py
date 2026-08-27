from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from hashlib import sha256
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from material_graph.knowledge.extraction import (
    EvidenceFactExtractionPipeline,
    FactExtractionCheckpoint,
    FactExtractionCheckpointConflict,
    FactExtractionError,
    FactExtractionPolicy,
    FactExtractor,
    FactExtractorProviderError,
    InMemoryFactExtractionCheckpointRepository,
    build_fact_extraction_idempotency_key,
    export_extractor_payload_json_schema,
)
from material_graph.knowledge.facts import InMemoryGlobalKnowledgeGraphWriter
from material_graph.knowledge.facts import ExtractionProvenance
from material_graph.knowledge.models import EvidenceFragment, SourceLocator


FRAGMENT_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SOURCE_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
GENERATION_ID = "fact-extractor-generation-2026-07-27"


def _provenance(*, generation_id: str = GENERATION_ID) -> ExtractionProvenance:
    return ExtractionProvenance(
        extractor_name="structured-material-fact-extractor",
        extractor_version="1.0.0",
        generation_id=generation_id,
        model_name="provider-neutral-test-model",
        model_version="2026-07-27",
    )


def _fragment(
    *,
    text: str = (
        "Composite MX-17 showed a tensile strength of 125 MPa at 23 degC "
        "under ISO 527 testing after solution casting."
    ),
    content_sha256: str | None = None,
) -> EvidenceFragment:
    return EvidenceFragment(
        fragment_id=FRAGMENT_ID,
        source_id=SOURCE_ID,
        text=text,
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path="private/nas/patents/material-paper.pdf",
            page=7,
            section="Mechanical properties",
            block_index=3,
        ),
        content_sha256=content_sha256,
        retention_reason="supports active material evidence gap",
        parser_name="mineru",
        parser_version="2.5.4",
        embedding_generation_id="qwen3-embedding-8b-1024-v1",
        metadata={
            "session_token": "must-not-cross-extractor-boundary",
            "complete_parser_output": "must-not-be-checkpointed",
        },
    )


def _provider_payload(*, value: float = 125.0) -> dict[str, Any]:
    material = {
        "entity_type": "material",
        "canonical_name": "Composite MX-17",
        "aliases": ["MX17"],
        "identifiers": {"laboratory_id": "MX-17"},
    }
    custom_process_window = {
        "entity_type": "fiber_process_window",
        "canonical_name": "MX-17 solution-casting window",
        "aliases": [],
        "identifiers": {},
    }
    return {
        "entities": [material, custom_process_window],
        "relations": [
            {
                "subject": material,
                "predicate": "processed within",
                "object": custom_process_window,
                "test_conditions": [],
                "process_conditions": [
                    {
                        "process_step": "solution casting",
                        "parameter": "drying temperature",
                        "value": 80.0,
                        "unit": "degC",
                    }
                ],
                "evidence_role": "supports",
                "confidence": 0.88,
                "evidence_quality": "high",
                "assertion_status": "affirmed",
            }
        ],
        "observations": [
            {
                "subject": material,
                "subject_role": "material",
                "property_name": "tensile strength",
                "value": value,
                "unit": "MPa",
                "test_method": "ISO 527",
                "test_conditions": [{"name": "temperature", "value": 23.0, "unit": "degC"}],
                "process_conditions": [],
                "evidence_role": "supports",
                "confidence": 0.93,
                "evidence_quality": "high",
                "assertion_status": "affirmed",
            }
        ],
    }


class FakeExtractor:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[object] = []

    async def extract(self, request: object) -> object:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("fake extractor called more times than expected")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(request)
        return outcome


class NeverExtractor:
    calls = 0

    async def extract(self, request: object) -> object:
        del request
        self.calls += 1
        raise AssertionError("provider must not be called while resuming completed work")


def _pipeline(
    extractor: object,
    *,
    repository: InMemoryFactExtractionCheckpointRepository | None = None,
    max_attempts: int = 3,
    max_fragment_bytes: int = 131_072,
    sleeps: list[float] | None = None,
) -> EvidenceFactExtractionPipeline:
    async def record_sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return EvidenceFactExtractionPipeline(
        extractor=extractor,  # type: ignore[arg-type]
        extraction=_provenance(),
        checkpoints=repository or InMemoryFactExtractionCheckpointRepository(),
        policy=FactExtractionPolicy(
            max_attempts=max_attempts,
            retry_base_seconds=0,
            retry_max_seconds=0,
            max_fragment_bytes=max_fragment_bytes,
        ),
        sleep=record_sleep,
    )


def test_success_builds_pending_review_batch_without_graph_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_write(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("extraction must not write the global graph")

    monkeypatch.setattr(InMemoryGlobalKnowledgeGraphWriter, "write_batch", forbidden_write)
    fragment = _fragment()
    provider = FakeExtractor(json.dumps(_provider_payload()))

    result = asyncio.run(_pipeline(provider).extract(fragment))

    assert result.review_status == "pending_review"
    assert result.resumed is False
    assert result.provider_calls == 1
    assert result.checkpoint.status == "completed"
    assert result.batch.batch_id == result.checkpoint.batch.batch_id
    assert result.batch.extraction == _provenance()
    assert result.batch.entities[1].entity_type == "fiber_process_window"
    observation = result.batch.observations[0]
    assert observation.value == pytest.approx(125.0)
    assert observation.unit == "MPa"
    assert observation.test_method == "ISO 527"
    assert observation.test_conditions[0].name == "temperature"
    link = observation.evidence[0]
    assert link.fragment_id == FRAGMENT_ID
    assert link.source_id == SOURCE_ID
    assert link.locator.page == 7
    assert link.locator.section == "Mechanical properties"
    assert link.locator.block_index == 3
    assert link.locator.relative_path == f"fragments/{FRAGMENT_ID}"
    assert link.public_source_uri == fragment.locator.to_public_uri(SOURCE_ID)

    request = provider.requests[0]
    rendered_request = request.model_dump_json()  # type: ignore[union-attr]
    assert fragment.text in rendered_request
    assert "private/nas" not in rendered_request
    assert "material-paper.pdf" not in rendered_request
    assert "session_token" not in rendered_request
    assert "complete_parser_output" not in rendered_request

    rendered_checkpoint = result.checkpoint.model_dump_json()
    assert fragment.text not in rendered_checkpoint
    assert "private/nas" not in rendered_checkpoint
    assert "material-paper.pdf" not in rendered_checkpoint
    assert "must-not-cross-extractor-boundary" not in rendered_checkpoint


def test_protocol_schema_and_idempotency_are_provider_neutral_and_deterministic() -> None:
    provider = FakeExtractor(_provider_payload())
    assert isinstance(provider, FactExtractor)

    schema = export_extractor_payload_json_schema()
    rendered = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    assert schema["title"] == "ExtractorFactPayload"
    assert "polyimide" not in rendered.casefold()
    assert "pet/pa6" not in rendered.casefold()
    assert "raw_pdf" not in rendered
    assert "complete_parser_output" not in rendered
    assert '"evidence"' not in rendered
    assert '"extraction"' not in rendered

    key = build_fact_extraction_idempotency_key(FRAGMENT_ID, GENERATION_ID)
    repeated = build_fact_extraction_idempotency_key(FRAGMENT_ID, GENERATION_ID)
    changed = build_fact_extraction_idempotency_key(FRAGMENT_ID, "different-generation")
    result = asyncio.run(_pipeline(provider).extract(_fragment()))
    assert key == repeated == result.batch.idempotency_key
    assert changed != key


def test_retryable_provider_failure_is_bounded_and_completed_checkpoint_resumes() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    sleeps: list[float] = []
    secret = "PROVIDER-DETAIL-MUST-NOT-LEAK"
    provider = FakeExtractor(RuntimeError(secret), _provider_payload())

    first = asyncio.run(
        _pipeline(provider, repository=repository, sleeps=sleeps).extract(_fragment())
    )
    never = NeverExtractor()
    second = asyncio.run(_pipeline(never, repository=repository).extract(_fragment()))

    assert first.checkpoint.attempts == 2
    assert first.provider_calls == 2
    assert sleeps == [0]
    assert second.resumed is True
    assert second.provider_calls == 0
    assert second.batch == first.batch
    assert never.calls == 0
    assert secret not in second.checkpoint.model_dump_json()


def test_completed_task_state_can_seed_an_empty_checkpoint_repository() -> None:
    first = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    new_repository = InMemoryFactExtractionCheckpointRepository()
    never = NeverExtractor()

    resumed = asyncio.run(
        _pipeline(never, repository=new_repository).extract(
            _fragment(), resume_from=first.checkpoint
        )
    )

    assert resumed.resumed is True
    assert resumed.batch == first.batch
    stored = asyncio.run(new_repository.load(first.checkpoint.idempotency_key))
    assert stored == first.checkpoint


def test_provider_declared_permanent_error_is_not_retried_or_leaked() -> None:
    provider = FakeExtractor(FactExtractorProviderError(retryable=False))

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(provider).extract(_fragment()))

    assert str(captured.value) == "extraction.provider_rejected"
    assert captured.value.code == "extraction.provider_rejected"
    assert captured.value.retryable is False
    assert captured.value.attempts == 1
    assert len(provider.requests) == 1


def test_retry_exhaustion_is_terminal_and_does_not_call_provider_on_resume() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    marker = "do-not-echo-provider-stacktrace"
    provider = FakeExtractor(RuntimeError(marker), RuntimeError(marker))

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(provider, repository=repository, max_attempts=2).extract(_fragment()))

    assert str(captured.value) == "extraction.provider_unavailable"
    assert marker not in str(captured.value)
    assert captured.value.retryable is False
    assert captured.value.attempts == 2

    never = NeverExtractor()
    with pytest.raises(FactExtractionError) as resumed:
        asyncio.run(_pipeline(never, repository=repository).extract(_fragment()))
    assert str(resumed.value) == "extraction.provider_unavailable"
    assert never.calls == 0


def test_invalid_json_and_schema_mismatch_can_retry_without_persisting_raw_output() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    marker = "SENSITIVE-INVALID-OUTPUT"
    provider = FakeExtractor(
        json.dumps({**_provider_payload(), "unexpected_raw_field": marker}),
        _provider_payload(),
    )

    result = asyncio.run(_pipeline(provider, repository=repository).extract(_fragment()))

    assert result.checkpoint.attempts == 2
    assert marker not in result.checkpoint.model_dump_json()
    assert result.review_status == "pending_review"


@pytest.mark.parametrize(
    "raw_output",
    [
        b"%PDF-1.7 not a structured fact payload",
        {"raw_pdf": "JVBERi0xLjQK", **_provider_payload()},
        {"api_key": "redacted-provider-value", **_provider_payload()},
        {**_provider_payload(), "full_text": "complete MinerU markdown output"},
    ],
)
def test_raw_document_or_credential_fields_fail_closed_without_retry(raw_output: object) -> None:
    provider = FakeExtractor(raw_output, _provider_payload())

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(provider).extract(_fragment()))

    assert str(captured.value) == "extraction.unsafe_provider_output"
    assert captured.value.retryable is False
    assert len(provider.requests) == 1
    assert "redacted-provider-value" not in str(captured.value)
    assert "MinerU" not in str(captured.value)


def test_fragment_echo_in_provider_output_is_rejected_without_echoing_it() -> None:
    fragment = _fragment(text="X" * 300)
    payload = _provider_payload()
    payload["entities"][0]["canonical_name"] = fragment.text
    payload["relations"][0]["subject"] = payload["entities"][0]
    payload["observations"][0]["subject"] = payload["entities"][0]

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(FakeExtractor(payload)).extract(fragment))

    assert str(captured.value) == "extraction.unsafe_provider_output"
    assert fragment.text not in str(captured.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["observations"][0].pop("unit"),
        lambda payload: payload["observations"][0].pop("test_method"),
        lambda payload: payload["observations"][0].update({"test_conditions": []}),
        lambda payload: payload["observations"][0].update({"value": float("inf")}),
    ],
)
def test_numeric_properties_require_unit_method_finite_value_and_condition(mutate: object) -> None:
    payload = deepcopy(_provider_payload())
    mutate(payload)  # type: ignore[operator]

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(FakeExtractor(payload), max_attempts=1).extract(_fragment()))

    assert str(captured.value) == "extraction.invalid_provider_output"
    assert captured.value.retryable is False


def test_declared_fragment_hash_mismatch_and_missing_anchor_fail_before_provider() -> None:
    provider = NeverExtractor()
    bad_hash = _fragment(content_sha256="0" * 64)
    with pytest.raises(FactExtractionError, match="extraction.invalid_fragment"):
        asyncio.run(_pipeline(provider).extract(bad_hash))

    no_anchor = _fragment().model_copy(
        update={
            "locator": SourceLocator(
                root_id="document_data_1",
                relative_path="private/nas/no-anchor.pdf",
            )
        }
    )
    with pytest.raises(FactExtractionError, match="extraction.invalid_fragment"):
        asyncio.run(_pipeline(provider).extract(no_anchor))
    assert provider.calls == 0


def test_fragment_size_is_bounded_before_provider_call() -> None:
    provider = NeverExtractor()

    with pytest.raises(FactExtractionError, match="extraction.fragment_too_large"):
        asyncio.run(
            _pipeline(provider, max_fragment_bytes=16).extract(_fragment(text="large fragment" * 5))
        )
    assert provider.calls == 0


def test_fragment_mutation_during_provider_call_fails_closed() -> None:
    fragment = _fragment()

    def mutate_fragment(_request: object) -> object:
        fragment.text = "content changed while the provider was running"
        fragment.content_sha256 = sha256(fragment.text.encode("utf-8")).hexdigest()
        return _provider_payload()

    with pytest.raises(FactExtractionError) as captured:
        asyncio.run(_pipeline(FakeExtractor(mutate_fragment)).extract(fragment))

    assert str(captured.value) == "extraction.content_drift"
    assert captured.value.retryable is False


def test_completed_checkpoint_rejects_changed_fragment_content_before_provider() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    first = asyncio.run(
        _pipeline(FakeExtractor(_provider_payload()), repository=repository).extract(_fragment())
    )
    changed = _fragment(text="the same fragment identifier now points to changed retained evidence")
    never = NeverExtractor()

    with pytest.raises(FactExtractionError, match="extraction.content_drift"):
        asyncio.run(_pipeline(never, repository=repository).extract(changed))

    assert first.checkpoint.status == "completed"
    assert never.calls == 0


def test_duplicate_json_keys_non_json_values_and_oversized_output_are_rejected() -> None:
    duplicate = '{"entities":[],"entities":[],"relations":[],"observations":[]}'
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    oversized = json.dumps({"padding": "x" * 300_000})

    for output in (duplicate, cyclic, oversized):
        with pytest.raises(FactExtractionError, match="extraction.invalid_provider_output"):
            asyncio.run(_pipeline(FakeExtractor(output), max_attempts=1).extract(_fragment()))


def test_checkpoint_repository_is_defensive_idempotent_and_conflict_safe() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    repository = InMemoryFactExtractionCheckpointRepository()
    saved = asyncio.run(repository.save(result.checkpoint))
    repeated = asyncio.run(repository.save(result.checkpoint))
    assert saved == repeated == result.checkpoint

    changed_payload = result.checkpoint.model_dump(mode="python")
    changed_payload["request_fingerprint"] = "0" * 64
    changed_payload["fragment_content_sha256"] = "0" * 64
    changed = FactExtractionCheckpoint.model_validate(changed_payload)
    with pytest.raises(FactExtractionCheckpointConflict) as captured:
        asyncio.run(repository.save(changed))
    assert str(captured.value) == "extraction.checkpoint_conflict"

    loaded = asyncio.run(repository.load(result.checkpoint.idempotency_key))
    assert loaded == result.checkpoint
    assert loaded is not result.checkpoint


def test_checkpoint_repository_rejects_fact_content_drift_for_the_same_key() -> None:
    baseline = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    drifted = asyncio.run(
        _pipeline(FakeExtractor(_provider_payload(value=126.0))).extract(_fragment())
    )
    repository = InMemoryFactExtractionCheckpointRepository()
    asyncio.run(repository.save(baseline.checkpoint))

    with pytest.raises(FactExtractionCheckpointConflict) as captured:
        asyncio.run(repository.save(drifted.checkpoint))

    assert str(captured.value) == "extraction.checkpoint_conflict"
    persisted = asyncio.run(repository.load(baseline.checkpoint.idempotency_key))
    assert persisted == baseline.checkpoint


def test_checkpoint_and_policy_validation_are_strict_and_hide_input_values() -> None:
    with pytest.raises(ValidationError):
        FactExtractionPolicy(max_attempts=0)
    with pytest.raises(ValidationError):
        FactExtractionPolicy(max_response_bytes=100)
    with pytest.raises(ValidationError):
        FactExtractionCheckpoint.model_validate(
            {
                "idempotency_key": "not-a-key",
                "fragment_id": FRAGMENT_ID,
                "source_id": SOURCE_ID,
                "fragment_content_sha256": "0" * 64,
                "request_fingerprint": "0" * 64,
                "extraction": _provenance().model_dump(mode="json"),
                "status": "completed",
                "attempts": 1,
                "batch": None,
                "last_error_code": None,
                "raw_provider_output": "SECRET-CHECKPOINT-PAYLOAD",
            }
        )


def test_idempotency_and_policy_reject_invalid_configuration() -> None:
    with pytest.raises(TypeError):
        build_fact_extraction_idempotency_key("not-a-uuid", GENERATION_ID)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_fact_extraction_idempotency_key(FRAGMENT_ID, " ")
    with pytest.raises(ValueError):
        build_fact_extraction_idempotency_key(FRAGMENT_ID, "x" * 301)
    with pytest.raises(ValidationError):
        FactExtractionPolicy(retry_base_seconds=2, retry_max_seconds=1)


def test_provider_payload_requires_at_least_one_fact() -> None:
    payload = _provider_payload()
    payload["relations"] = []
    payload["observations"] = []

    with pytest.raises(FactExtractionError, match="extraction.invalid_provider_output"):
        asyncio.run(_pipeline(FakeExtractor(payload), max_attempts=1).extract(_fragment()))


def test_checkpoint_model_rejects_inconsistent_lifecycle_and_batch_bindings() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    baseline = result.checkpoint.model_dump(mode="python")

    wrong_key = deepcopy(baseline)
    wrong_key["idempotency_key"] = "fact-batch-idempotency:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="checkpoint_idempotency_mismatch"):
        FactExtractionCheckpoint.model_validate(wrong_key)

    no_batch = deepcopy(baseline)
    no_batch["batch"] = None
    with pytest.raises(ValidationError, match="completed_checkpoint_requires_batch"):
        FactExtractionCheckpoint.model_validate(no_batch)

    other_fragment = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    batch_mismatch = deepcopy(baseline)
    batch_mismatch["fragment_id"] = other_fragment
    batch_mismatch["idempotency_key"] = build_fact_extraction_idempotency_key(
        other_fragment, GENERATION_ID
    )
    with pytest.raises(ValidationError, match="checkpoint_batch_mismatch"):
        FactExtractionCheckpoint.model_validate(batch_mismatch)

    wrong_path = deepcopy(baseline)
    for fact in (*wrong_path["batch"]["relations"], *wrong_path["batch"]["observations"]):
        fact["evidence"][0]["locator"]["relative_path"] = "private/path.pdf"
    with pytest.raises(ValidationError, match="checkpoint_evidence_mismatch"):
        FactExtractionCheckpoint.model_validate(wrong_path)

    unfinished_with_batch = deepcopy(baseline)
    unfinished_with_batch.update({"status": "running", "last_error_code": None})
    with pytest.raises(ValidationError, match="unfinished_checkpoint_cannot_store_batch"):
        FactExtractionCheckpoint.model_validate(unfinished_with_batch)

    retry_without_error = deepcopy(baseline)
    retry_without_error.update({"status": "retry_wait", "batch": None})
    with pytest.raises(ValidationError, match="failed_checkpoint_requires_error_code"):
        FactExtractionCheckpoint.model_validate(retry_without_error)

    running_with_error = deepcopy(baseline)
    running_with_error.update(
        {
            "status": "running",
            "batch": None,
            "last_error_code": "extraction.provider_unavailable",
        }
    )
    with pytest.raises(ValidationError, match="running_checkpoint_cannot_store_error"):
        FactExtractionCheckpoint.model_validate(running_with_error)


def test_checkpoint_rejects_multiple_evidence_links_per_extracted_fact() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    payload = result.checkpoint.model_dump(mode="python")
    relation = payload["batch"]["relations"][0]
    second_link = deepcopy(relation["evidence"][0])
    second_link.pop("evidence_ref_id")
    second_link["locator"]["block_index"] = 4
    relation["evidence"] = (*relation["evidence"], second_link)
    relation.pop("relation_id")
    payload["batch"].pop("batch_id")

    with pytest.raises(ValidationError, match="checkpoint_evidence_mismatch"):
        FactExtractionCheckpoint.model_validate(payload)


def test_checkpoint_repository_rejects_invalid_types_and_state_regression() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    with pytest.raises(ValueError):
        asyncio.run(repository.load("bad-key"))
    with pytest.raises(TypeError):
        asyncio.run(repository.save(object()))  # type: ignore[arg-type]

    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    completed = result.checkpoint
    asyncio.run(repository.save(completed))
    running_payload = completed.model_dump(mode="python")
    running_payload.update({"status": "running", "batch": None, "last_error_code": None})
    running = FactExtractionCheckpoint.model_validate(running_payload)
    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(repository.save(running))

    second_repository = InMemoryFactExtractionCheckpointRepository()
    attempt_two = running.model_copy(update={"attempts": 2})
    asyncio.run(second_repository.save(attempt_two))
    attempt_one = running.model_copy(update={"attempts": 1})
    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(second_repository.save(attempt_one))


def test_result_contract_rejects_noncompleted_or_false_resume_claims() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    running_payload = result.checkpoint.model_dump(mode="python")
    running_payload.update({"status": "running", "batch": None, "last_error_code": None})
    running = FactExtractionCheckpoint.model_validate(running_payload)

    with pytest.raises(ValidationError, match="extraction_result_checkpoint_mismatch"):
        result.model_copy(update={"checkpoint": running}).__class__.model_validate(
            {**result.model_dump(mode="python"), "checkpoint": running}
        )
    with pytest.raises(ValidationError, match="resumed_result_cannot_claim_provider_calls"):
        result.__class__.model_validate(
            {**result.model_dump(mode="python"), "resumed": True, "provider_calls": 1}
        )


def test_non_fragment_and_non_provenance_inputs_are_rejected() -> None:
    with pytest.raises(TypeError):
        EvidenceFactExtractionPipeline(
            extractor=NeverExtractor(),
            extraction=object(),  # type: ignore[arg-type]
            checkpoints=InMemoryFactExtractionCheckpointRepository(),
        )
    with pytest.raises(FactExtractionError, match="extraction.invalid_fragment"):
        asyncio.run(_pipeline(NeverExtractor()).extract(object()))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({1: "non-string-key"}, "extraction.invalid_provider_output"),
        ({"entities": object()}, "extraction.invalid_provider_output"),
        ('{"entities":NaN,"relations":[],"observations":[]}', "extraction.invalid_provider_output"),
        ("[]", "extraction.invalid_provider_output"),
        (b"\xff\xfe", "extraction.invalid_provider_output"),
        (
            {"canonical_name": "Bearer abcdefghijklmnopqrstuvwxyz"},
            "extraction.unsafe_provider_output",
        ),
    ],
)
def test_provider_json_tree_and_value_guards(output: object, expected: str) -> None:
    with pytest.raises(FactExtractionError, match=expected):
        asyncio.run(_pipeline(FakeExtractor(output), max_attempts=1).extract(_fragment()))


def test_deep_and_oversized_mapping_or_byte_payloads_are_bounded() -> None:
    deep: dict[str, object] = {"leaf": "value"}
    for _ in range(30):
        deep = {"nested": deep}
    oversized_mapping = {"padding": "x" * 300_000}
    oversized_bytes = b" " * 300_000

    for output in (deep, oversized_mapping, oversized_bytes):
        with pytest.raises(FactExtractionError, match="extraction.invalid_provider_output"):
            asyncio.run(_pipeline(FakeExtractor(output), max_attempts=1).extract(_fragment()))


def test_valid_utf8_json_bytes_are_supported() -> None:
    output = json.dumps(_provider_payload()).encode("utf-8")

    result = asyncio.run(_pipeline(FakeExtractor(output)).extract(_fragment()))

    assert result.review_status == "pending_review"
    assert result.batch.observations[0].value == pytest.approx(125)


def test_final_fact_contract_gate_rejects_overclaimed_unknown_test_context() -> None:
    payload = _provider_payload()
    payload["observations"][0]["test_method"] = "unknown"

    with pytest.raises(FactExtractionError, match="extraction.invalid_provider_output"):
        asyncio.run(_pipeline(FakeExtractor(payload), max_attempts=1).extract(_fragment()))


def test_resume_rejects_wrong_type_generation_and_divergent_task_state() -> None:
    repository = InMemoryFactExtractionCheckpointRepository()
    completed = asyncio.run(
        _pipeline(FakeExtractor(_provider_payload()), repository=repository).extract(_fragment())
    )

    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(
            _pipeline(NeverExtractor(), repository=repository).extract(
                _fragment(),
                resume_from=object(),  # type: ignore[arg-type]
            )
        )

    different_generation = EvidenceFactExtractionPipeline(
        extractor=NeverExtractor(),
        extraction=_provenance(generation_id="different-generation"),
        checkpoints=InMemoryFactExtractionCheckpointRepository(),
        policy=FactExtractionPolicy(retry_base_seconds=0, retry_max_seconds=0),
    )
    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(different_generation.extract(_fragment(), resume_from=completed.checkpoint))

    running_payload = completed.checkpoint.model_dump(mode="python")
    running_payload.update({"status": "running", "batch": None, "last_error_code": None})
    running = FactExtractionCheckpoint.model_validate(running_payload)
    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(
            _pipeline(NeverExtractor(), repository=repository).extract(
                _fragment(), resume_from=running
            )
        )


def test_running_checkpoint_at_attempt_limit_becomes_terminal_without_provider() -> None:
    completed = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    payload = completed.checkpoint.model_dump(mode="python")
    payload.update({"status": "running", "attempts": 3, "batch": None, "last_error_code": None})
    running = FactExtractionCheckpoint.model_validate(payload)
    repository = InMemoryFactExtractionCheckpointRepository()
    asyncio.run(repository.save(running))
    never = NeverExtractor()

    with pytest.raises(FactExtractionError, match="extraction.retry_exhausted") as captured:
        asyncio.run(_pipeline(never, repository=repository, max_attempts=3).extract(_fragment()))

    assert captured.value.attempts == 3
    assert never.calls == 0
    terminal = asyncio.run(repository.load(running.idempotency_key))
    assert terminal is not None
    assert terminal.status == "failed_permanent"


def test_resume_detects_evidence_anchor_drift_inside_completed_batch() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    payload = result.checkpoint.model_dump(mode="python")
    batch = payload["batch"]
    for fact in (*batch["relations"], *batch["observations"]):
        fact.pop("relation_id", None)
        fact.pop("observation_id", None)
        fact["evidence"][0].pop("evidence_ref_id", None)
        fact["evidence"][0]["locator"]["block_index"] = 4
    batch.pop("batch_id", None)
    drifted = FactExtractionCheckpoint.model_validate(payload)

    with pytest.raises(FactExtractionError, match="extraction.content_drift"):
        asyncio.run(_pipeline(NeverExtractor()).extract(_fragment(), resume_from=drifted))


def test_invalid_stable_error_code_in_terminal_checkpoint_fails_closed() -> None:
    result = asyncio.run(_pipeline(FakeExtractor(_provider_payload())).extract(_fragment()))
    payload = result.checkpoint.model_dump(mode="python")
    payload.update(
        {
            "status": "failed_permanent",
            "batch": None,
            "last_error_code": "extraction.unknown_failure",
        }
    )
    terminal = FactExtractionCheckpoint.model_validate(payload)
    repository = InMemoryFactExtractionCheckpointRepository()
    asyncio.run(repository.save(terminal))

    with pytest.raises(FactExtractionCheckpointConflict):
        asyncio.run(_pipeline(NeverExtractor(), repository=repository).extract(_fragment()))


def test_invalid_fragment_mutation_during_call_is_reported_as_content_drift() -> None:
    fragment = _fragment()

    def corrupt_hash(_request: object) -> object:
        fragment.content_sha256 = "0" * 64
        return _provider_payload()

    with pytest.raises(FactExtractionError, match="extraction.content_drift"):
        asyncio.run(_pipeline(FakeExtractor(corrupt_hash)).extract(fragment))
