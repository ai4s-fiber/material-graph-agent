from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Sequence
from uuid import uuid4

import httpx
import pytest

from material_graph.knowledge.lightrag_client import (
    InMemoryLightRAGSourceMappingStore,
    LightRAGClient,
    LightRAGForbiddenOperation,
    LightRAGPollingTimeout,
    LightRAGProtocolError,
    LightRAGRequestError,
    LightRAGSourceMappingConflict,
    build_lightrag_insert_idempotency_key,
)
from material_graph.knowledge.lightrag_models import (
    LightRAGFixedTokenParams,
    LightRAGInsertResult,
    LightRAGSourceMapping,
    LightRAGTextRequest,
    LightRAGTextsRequest,
    LightRAGTrackStatus,
)
from material_graph.knowledge.models import EvidenceFragment, SourceLocator


def _fragment(
    text: str = "含氟结构在 1 MHz 测试条件下降低聚酰亚胺介电常数。",
    *,
    metadata: dict[str, object] | None = None,
    embedding_generation_id: str = "fixture-embedding-generation-v1",
) -> EvidenceFragment:
    return EvidenceFragment(
        source_id=uuid4(),
        text=text,
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path="信智学院文献数据/polymer/paper.pdf",
            page=7,
            section="Results",
            block_index=3,
        ),
        retention_reason="supports:dielectric_constant",
        supported_entity_ids=["entity:fluorinated_group"],
        supported_relation_ids=["relation:decreases_dielectric_constant"],
        parser_name="mineru",
        parser_version="3.4.4",
        embedding_generation_id=embedding_generation_id,
        metadata=metadata or {},
    )


def _track_payload(
    track_id: str,
    status: str,
    basenames: Sequence[str],
) -> dict[str, object]:
    documents = [
        {
            "id": f"doc-{index}",
            "status": status,
            "file_path": basename,
            "error_msg": "provider failed" if status == "failed" else None,
        }
        for index, basename in enumerate(basenames, start=1)
    ]
    return {
        "track_id": track_id,
        "documents": documents,
        "total_count": len(documents),
        "status_summary": {status: len(documents)},
    }


def _run(awaitable: Awaitable[object]) -> object:
    return asyncio.run(awaitable)


def test_source_mapping_is_deterministic_filename_safe_and_complete() -> None:
    fragment = _fragment()

    first = LightRAGSourceMapping.from_fragment(fragment)
    second = LightRAGSourceMapping.from_fragment(fragment)

    assert first == second
    assert re.fullmatch(r"mg_[0-9a-f]{32}_[0-9a-f]{32}_[0-9a-f]{16}\.txt", first.basename)
    assert first.basename == first.basename.rsplit("/", maxsplit=1)[-1]
    assert first.locator == fragment.locator
    assert first.logical_source_uri == fragment.locator.to_public_uri(fragment.source_id)
    assert first.content_sha256 == fragment.content_sha256
    assert "信智学院文献数据" not in first.logical_source_uri
    assert "api_key" not in first.model_dump(mode="json")


def test_mapping_store_is_idempotent_and_returns_defensive_copies() -> None:
    async def scenario() -> None:
        store = InMemoryLightRAGSourceMappingStore()
        mapping = LightRAGSourceMapping.from_fragment(_fragment())

        await store.persist_many([mapping])
        await store.persist_many([mapping])
        loaded = await store.get(mapping.basename)

        assert loaded == mapping
        assert loaded is not mapping
        assert len(await store.list_all()) == 1

    _run(scenario())


def test_mapping_is_persisted_before_text_submission_and_track_is_polled() -> None:
    async def scenario() -> None:
        fragment = _fragment()
        mapping = LightRAGSourceMapping.from_fragment(fragment)
        events: list[str] = []
        status_calls = 0
        api_key = "runtime-only-credential"

        class RecordingStore(InMemoryLightRAGSourceMappingStore):
            async def persist_many(self, mappings: Sequence[LightRAGSourceMapping]) -> None:
                events.append("mapping-persisted")
                await super().persist_many(mappings)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            assert request.headers["x-api-key"] == api_key
            if request.url.path == "/documents/text":
                assert request.headers["idempotency-key"] == (
                    build_lightrag_insert_idempotency_key([mapping])
                )
                events.append("text-submitted")
                payload = json.loads(request.content)
                assert payload == {
                    "text": fragment.text,
                    "file_source": mapping.basename,
                    "chunking": {
                        "strategy": "fixed_token",
                        "params": {
                            "chunk_token_size": 1200,
                            "chunk_overlap_token_size": 100,
                        },
                    },
                }
                return httpx.Response(
                    200,
                    json={"status": "success", "message": "accepted", "track_id": "insert_1"},
                )
            assert request.url.path == "/documents/track_status/insert_1"
            status_calls += 1
            status = "processing" if status_calls == 1 else "processed"
            return httpx.Response(200, json=_track_payload("insert_1", status, [mapping.basename]))

        async def no_wait(_: float) -> None:
            return None

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key=api_key,
            mapping_store=RecordingStore(),
            transport=httpx.MockTransport(handler),
            sleep=no_wait,
            poll_interval_seconds=0,
        ) as client:
            result = await client.insert_retained_fragment(fragment)
            rendered = repr(client)

        assert events[:2] == ["mapping-persisted", "text-submitted"]
        assert status_calls == 2
        assert result.outcome == "processed"
        assert result.track_status is not None
        assert result.track_status.is_terminal is True
        assert api_key not in rendered
        assert api_key not in result.model_dump_json()

    _run(scenario())


def test_batch_uses_texts_and_persists_every_mapping_first() -> None:
    async def scenario() -> None:
        fragments = [_fragment("evidence one"), _fragment("evidence two")]
        expected = [LightRAGSourceMapping.from_fragment(item) for item in fragments]
        persisted: list[LightRAGSourceMapping] = []

        class RecordingStore(InMemoryLightRAGSourceMappingStore):
            async def persist_many(self, mappings: Sequence[LightRAGSourceMapping]) -> None:
                await super().persist_many(mappings)
                persisted.extend(mappings)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/documents/texts":
                assert persisted == expected
                assert request.headers["idempotency-key"] == (
                    build_lightrag_insert_idempotency_key(expected)
                )
                payload = json.loads(request.content)
                assert payload["texts"] == [item.text for item in fragments]
                assert payload["file_sources"] == [item.basename for item in expected]
                return httpx.Response(
                    200,
                    json={"status": "success", "message": "accepted", "track_id": "insert_many"},
                )
            return httpx.Response(
                200,
                json=_track_payload(
                    "insert_many",
                    "processed",
                    [item.basename for item in expected],
                ),
            )

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=RecordingStore(),
            transport=httpx.MockTransport(handler),
            poll_interval_seconds=0,
        ) as client:
            result = await client.insert_retained_fragments(fragments)

        assert result.outcome == "processed"
        assert result.mappings == expected

    _run(scenario())


def test_409_is_an_idempotency_conflict_and_is_not_retried() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                409,
                json={"detail": "Document storage already contains it; token=runtime-secret"},
            )

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=5,
        ) as client:
            result = await client.insert_retained_fragment(_fragment())

        assert calls == 1
        assert result.outcome == "idempotent_conflict"
        assert result.track_id is None
        assert result.track_status is None
        assert result.message == "idempotent_conflict"
        assert "runtime-secret" not in result.model_dump_json()

    _run(scenario())


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
def test_non_retryable_client_errors_fail_once(status_code: int) -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                status_code,
                json={"detail": "rejected; password=runtime-secret"},
            )

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=4,
        ) as client:
            with pytest.raises(LightRAGRequestError) as raised:
                await client.insert_retained_fragment(_fragment())

        assert calls == 1
        assert "runtime-secret" not in str(raised.value)
        assert raised.value.status_code == status_code

    _run(scenario())


def test_429_honors_retry_after_and_5xx_retries_are_bounded() -> None:
    async def scenario() -> None:
        responses = [
            httpx.Response(429, headers={"Retry-After": "2"}, json={"detail": "slow down"}),
            httpx.Response(503, json={"detail": "temporarily unavailable"}),
            httpx.Response(
                200,
                json={"status": "success", "message": "accepted", "track_id": "insert_retry"},
            ),
        ]
        sleeps: list[float] = []
        post_calls = 0
        fragment = _fragment()
        mapping = LightRAGSourceMapping.from_fragment(fragment)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal post_calls
            if request.url.path == "/documents/text":
                response = responses[post_calls]
                post_calls += 1
                return response
            return httpx.Response(
                200,
                json=_track_payload("insert_retry", "processed", [mapping.basename]),
            )

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=2,
            retry_backoff_seconds=0.25,
            sleep=record_sleep,
            poll_interval_seconds=0,
        ) as client:
            result = await client.insert_retained_fragment(fragment)

        assert result.outcome == "processed"
        assert post_calls == 3
        assert sleeps == [2.0, 0.5]

    _run(scenario())


def test_exhausted_5xx_retry_raises_without_leaking_credential() -> None:
    async def scenario() -> None:
        calls = 0
        api_key = "must-never-appear"

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, json={"detail": f"failed {api_key}"})

        async def no_wait(_: float) -> None:
            return None

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key=api_key,
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=2,
            sleep=no_wait,
        ) as client:
            with pytest.raises(LightRAGRequestError) as raised:
                await client.insert_retained_fragment(_fragment())

        assert calls == 3
        assert api_key not in str(raised.value)

    _run(scenario())


def test_failed_track_is_returned_as_terminal_result() -> None:
    async def scenario() -> None:
        fragment = _fragment()
        mapping = LightRAGSourceMapping.from_fragment(fragment)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/documents/text":
                return httpx.Response(
                    200,
                    json={"status": "success", "message": "accepted", "track_id": "insert_fail"},
                )
            return httpx.Response(
                200,
                json=_track_payload("insert_fail", "failed", [mapping.basename]),
            )

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            poll_interval_seconds=0,
        ) as client:
            result = await client.insert_retained_fragment(fragment)

        assert result.outcome == "failed"
        assert result.track_status is not None
        assert result.track_status.has_failures is True
        assert result.track_status.documents[0].error_msg == "processing_failed"
        assert "provider failed" not in result.model_dump_json()

    _run(scenario())


def test_polling_has_a_finite_attempt_budget() -> None:
    async def scenario() -> None:
        fragment = _fragment()
        mapping = LightRAGSourceMapping.from_fragment(fragment)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/documents/text":
                return httpx.Response(
                    200,
                    json={"status": "success", "message": "accepted", "track_id": "insert_slow"},
                )
            return httpx.Response(
                200,
                json=_track_payload("insert_slow", "processing", [mapping.basename]),
            )

        async def no_wait(_: float) -> None:
            return None

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_poll_attempts=2,
            poll_interval_seconds=0,
            sleep=no_wait,
        ) as client:
            with pytest.raises(LightRAGPollingTimeout, match="insert_slow"):
                await client.insert_retained_fragment(fragment)

    _run(scenario())


def test_outstanding_limit_is_held_until_track_reaches_terminal_state() -> None:
    async def scenario() -> None:
        first = _fragment("first evidence")
        second = _fragment("second evidence")
        first_mapping = LightRAGSourceMapping.from_fragment(first)
        second_mapping = LightRAGSourceMapping.from_fragment(second)
        first_pending = asyncio.Event()
        sequence: list[str] = []
        first_poll_count = 0
        post_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal first_poll_count, post_count
            if request.url.path == "/documents/text":
                post_count += 1
                track_id = f"insert_{post_count}"
                sequence.append(f"post:{track_id}")
                return httpx.Response(
                    200,
                    json={"status": "success", "message": "accepted", "track_id": track_id},
                )
            if request.url.path.endswith("insert_1"):
                first_poll_count += 1
                status = "processing" if first_poll_count == 1 else "processed"
                sequence.append(f"poll:insert_1:{status}")
                if status == "processing":
                    first_pending.set()
                return httpx.Response(
                    200,
                    json=_track_payload("insert_1", status, [first_mapping.basename]),
                )
            sequence.append("poll:insert_2:processed")
            return httpx.Response(
                200,
                json=_track_payload("insert_2", "processed", [second_mapping.basename]),
            )

        async def yield_control(_: float) -> None:
            await asyncio.sleep(0)

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            outstanding_track_limit=1,
            poll_interval_seconds=0,
            sleep=yield_control,
        ) as client:
            first_task = asyncio.create_task(client.insert_retained_fragment(first))
            await first_pending.wait()
            second_task = asyncio.create_task(client.insert_retained_fragment(second))
            await asyncio.gather(first_task, second_task)

        assert sequence.index("post:insert_2") > sequence.index("poll:insert_1:processed")

    _run(scenario())


@pytest.mark.parametrize(
    "invalid_input",
    [
        "raw text is not retained evidence",
        b"%PDF-1.7 raw bytes",
    ],
)
def test_client_rejects_non_evidence_input(invalid_input: object) -> None:
    async def scenario() -> None:
        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        ) as client:
            with pytest.raises(TypeError, match="EvidenceFragment"):
                await client.insert_retained_fragment(invalid_input)  # type: ignore[arg-type]

    _run(scenario())


@pytest.mark.parametrize(
    "fragment",
    [
        _fragment("%PDF-1.7 raw PDF body"),
        _fragment("selected text", metadata={"complete_mineru_output": {"pages": []}}),
        _fragment("x" * 65_537),
    ],
)
def test_client_rejects_raw_or_oversized_parser_payload(fragment: EvidenceFragment) -> None:
    async def scenario() -> None:
        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        ) as client:
            with pytest.raises(ValueError, match="retained evidence"):
                await client.insert_retained_fragment(fragment)

    _run(scenario())


def test_upload_and_scan_routes_are_rejected_before_transport() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={})

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
        ) as client:
            for forbidden_path in ("/documents/upload", "/documents/scan"):
                with pytest.raises(LightRAGForbiddenOperation):
                    await client._request_json("POST", forbidden_path, payload={})

        assert calls == 0

    _run(scenario())


def test_mixed_embedding_generations_are_not_batched() -> None:
    async def scenario() -> None:
        fragments = [
            _fragment("first", embedding_generation_id="generation-a"),
            _fragment("second", embedding_generation_id="generation-b"),
        ]
        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        ) as client:
            with pytest.raises(ValueError, match="embedding generation"):
                await client.insert_retained_fragments(fragments)

    _run(scenario())


def test_mapping_store_rejects_ambiguous_transactions() -> None:
    async def scenario() -> None:
        store = InMemoryLightRAGSourceMappingStore()
        mapping = LightRAGSourceMapping.from_fragment(_fragment())

        assert await store.get("missing.txt") is None
        with pytest.raises(LightRAGSourceMappingConflict, match="duplicate basename"):
            await store.persist_many([mapping, mapping])

        await store.persist_many([mapping])
        changed_generation = mapping.model_copy(
            update={"embedding_generation_id": "different-generation"}
        )
        with pytest.raises(LightRAGSourceMappingConflict, match="different provenance"):
            await store.persist_many([changed_generation])

    _run(scenario())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": " "}, "api_key"),
        ({"base_url": "lightrag-without-scheme"}, "base_url"),
        ({"outstanding_track_limit": 0}, "limits"),
        ({"max_retries": -1}, "retry and polling"),
    ],
)
def test_client_constructor_rejects_unsafe_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "base_url": "http://lightrag:9621",
        "api_key": "secret",
        "mapping_store": InMemoryLightRAGSourceMappingStore(),
        "transport": httpx.MockTransport(lambda _: httpx.Response(200, json={})),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        LightRAGClient(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("client_overrides", "fragments", "message"),
    [
        ({}, [], "at least one"),
        ({"max_batch_fragments": 1}, [_fragment("one"), _fragment("two")], "fragment limit"),
        ({"max_batch_chars": 5}, [_fragment("sixsix")], "character limit"),
        ({}, [_fragment(" padded evidence ")], "normalized"),
    ],
)
def test_fragment_admission_limits_reject_unretained_shapes(
    client_overrides: dict[str, int],
    fragments: list[EvidenceFragment],
    message: str,
) -> None:
    async def scenario() -> None:
        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
            **client_overrides,
        ) as client:
            with pytest.raises(ValueError, match=message):
                await client.insert_retained_fragments(fragments)

    _run(scenario())


def test_duplicate_fragment_identity_and_nested_binary_metadata_are_rejected() -> None:
    async def scenario() -> None:
        duplicate = _fragment("one retained block")
        binary_metadata = _fragment(
            "another retained block",
            metadata={"nested": [{"payload": b"raw-parser-artifact"}]},
        )
        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        ) as client:
            with pytest.raises(ValueError, match="unique identities"):
                await client.insert_retained_fragments([duplicate, duplicate])
            with pytest.raises(ValueError, match="retained evidence"):
                await client.insert_retained_fragment(binary_metadata)

    _run(scenario())


@pytest.mark.parametrize(
    "response_payload",
    [
        {},
        {"status": "failure", "message": "not accepted", "track_id": "insert_rejected"},
    ],
)
def test_invalid_or_rejected_insert_acceptance_is_a_protocol_error(
    response_payload: dict[str, object],
) -> None:
    async def scenario() -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_payload)

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(LightRAGProtocolError):
                await client.insert_retained_fragment(_fragment())

    _run(scenario())


def test_terminal_track_must_match_persisted_source_mappings() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/documents/text":
                return httpx.Response(
                    200,
                    json={
                        "status": "success",
                        "message": "accepted",
                        "track_id": "insert_wrong_source",
                    },
                )
            return httpx.Response(
                200,
                json=_track_payload("insert_wrong_source", "processed", ["unmapped.txt"]),
            )

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            poll_interval_seconds=0,
        ) as client:
            with pytest.raises(LightRAGProtocolError, match="source mappings"):
                await client.insert_retained_fragment(_fragment())

    _run(scenario())


def test_track_id_and_track_response_are_strictly_validated() -> None:
    async def scenario() -> None:
        responses = [
            httpx.Response(200, json={"track_id": "bad"}),
            httpx.Response(
                200,
                json={
                    "track_id": "different_track",
                    "documents": [],
                    "total_count": 0,
                    "status_summary": {},
                },
            ),
        ]
        calls = 0

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_poll_attempts=1,
        ) as client:
            with pytest.raises(ValueError, match="track_id"):
                await client.wait_for_track("../unsafe")
            with pytest.raises(LightRAGProtocolError, match="track status"):
                await client.wait_for_track("expected_track")
            with pytest.raises(LightRAGProtocolError, match="different track_id"):
                await client.wait_for_track("expected_track")

    _run(scenario())


def test_transport_errors_retry_then_fail_with_secret_free_error() -> None:
    async def scenario() -> None:
        calls = 0
        sleeps: list[float] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("network unavailable", request=request)

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=1,
            retry_backoff_seconds=0.25,
            sleep=record_sleep,
        ) as client:
            with pytest.raises(LightRAGRequestError) as raised:
                await client.insert_retained_fragment(_fragment())

        assert calls == 2
        assert sleeps == [0.25]
        assert raised.value.status_code is None

    _run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
def test_non_object_json_responses_are_rejected(response: httpx.Response) -> None:
    async def scenario() -> None:
        async def handler(_: httpx.Request) -> httpx.Response:
            return response

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(LightRAGProtocolError):
                await client._request_json("POST", "/documents/text", payload={})

    _run(scenario())


def test_model_validators_reject_inconsistent_mapping_and_request_contracts() -> None:
    first = LightRAGSourceMapping.from_fragment(_fragment("first"))
    second = LightRAGSourceMapping.from_fragment(_fragment("second"))

    bad_basename = first.model_dump()
    bad_basename["basename"] = f"mg_{first.source_id.hex}_{first.fragment_id.hex}_{'0' * 16}.txt"
    with pytest.raises(ValueError, match="basename"):
        LightRAGSourceMapping.model_validate(bad_basename)

    bad_uri = first.model_dump()
    bad_uri["logical_source_uri"] = f"source://document_data_1/{uuid4()}"
    with pytest.raises(ValueError, match="logical_source_uri"):
        LightRAGSourceMapping.model_validate(bad_uri)

    with pytest.raises(ValueError, match="overlap"):
        LightRAGFixedTokenParams(chunk_token_size=100, chunk_overlap_token_size=100)
    with pytest.raises(ValueError, match="blank"):
        LightRAGTextRequest(text=" ", file_source=first.basename)
    with pytest.raises(ValueError, match="blank"):
        LightRAGTextsRequest(
            texts=["ok", " "],
            file_sources=[first.basename, second.basename],
        )
    with pytest.raises(ValueError, match="unique"):
        LightRAGTextsRequest(
            texts=["one", "two"],
            file_sources=[first.basename, first.basename],
        )
    with pytest.raises(ValueError, match="every"):
        LightRAGTextsRequest(texts=["one", "two"], file_sources=[first.basename])


def test_track_and_insert_result_models_reject_inconsistent_terminal_state() -> None:
    mapping = LightRAGSourceMapping.from_fragment(_fragment())
    track = LightRAGTrackStatus.model_validate(
        _track_payload("insert_model", "PROCESSED", [mapping.basename])
    )
    assert track.is_terminal is True

    bad_count = _track_payload("insert_model", "processed", [mapping.basename])
    bad_count["total_count"] = 2
    with pytest.raises(ValueError, match="total_count"):
        LightRAGTrackStatus.model_validate(bad_count)

    with pytest.raises(ValueError, match="cannot claim"):
        LightRAGInsertResult(
            outcome="idempotent_conflict",
            mappings=[mapping],
            track_id="insert_model",
        )
    with pytest.raises(ValueError, match="requires track"):
        LightRAGInsertResult(outcome="processed", mappings=[mapping])
    with pytest.raises(ValueError, match="IDs"):
        LightRAGInsertResult(
            outcome="processed",
            mappings=[mapping],
            track_id="other_track",
            track_status=track,
        )
    with pytest.raises(ValueError, match="outcome"):
        LightRAGInsertResult(
            outcome="failed",
            mappings=[mapping],
            track_id="insert_model",
            track_status=track,
        )
    with pytest.raises(ValueError, match="stable safe status"):
        LightRAGInsertResult(
            outcome="idempotent_conflict",
            mappings=[mapping],
            message="provider detail with token",
        )


def test_non_json_error_bodies_do_not_disable_retry_or_status_classification() -> None:
    async def scenario() -> None:
        responses = [
            httpx.Response(503, json=["temporary", "failure"]),
            httpx.Response(401, content=b"unauthorized"),
        ]
        calls = 0
        sleeps: list[float] = []

        async def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        async with LightRAGClient(
            base_url="http://lightrag:9621",
            api_key="secret",
            mapping_store=InMemoryLightRAGSourceMappingStore(),
            transport=httpx.MockTransport(handler),
            max_retries=1,
            retry_backoff_seconds=0.25,
            sleep=record_sleep,
        ) as client:
            with pytest.raises(LightRAGRequestError) as raised:
                await client._request_json("POST", "/documents/text", payload={})

        assert calls == 2
        assert sleeps == [0.25]
        assert raised.value.status_code == 401
        assert raised.value.detail == "HTTP 401"

    _run(scenario())
