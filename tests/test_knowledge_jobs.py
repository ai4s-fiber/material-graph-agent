from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from material_graph.knowledge.extraction import FactExtractionError
from material_graph.knowledge.jobs import (
    InMemoryKnowledgeJobRepository,
    KnowledgeJobExecutionError,
    KnowledgeJobExecutor,
    KnowledgeJobLease,
    KnowledgeJobLeaseLost,
    KnowledgeJobRecord,
    KnowledgeJobResult,
    KnowledgeJobStatus,
    MetadataOnlyJobPayload,
    SingleDocumentJobPayload,
    build_knowledge_job_idempotency_key,
)
from material_graph.knowledge.models import SelectionDecision, SourceLocator
from material_graph.knowledge.service import (
    CanaryResult,
    CanarySource,
    CanaryStage,
    CanaryStatus,
    MetadataCanaryRequest,
    OperatorApproval,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 27, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def metadata_payload(run_id: str = "metadata-run") -> MetadataOnlyJobPayload:
    return MetadataOnlyJobPayload(
        request=MetadataCanaryRequest(
            run_id=run_id,
            root_id="document_data_1",
            slice_id="literature",
            manifest_path="private/manifests/catalog.jsonl",
            manifest_format="jsonl",
        )
    )


def selected_source(*, selected: bool = True, path: str = "private/literature/paper.pdf") -> CanarySource:
    source_id = UUID("11111111-1111-4111-8111-111111111111")
    return CanarySource(
        decision=SelectionDecision(
            source_id=source_id,
            selected=selected,
            reason_code="active_evidence_gap" if selected else "budget_deferred",
            policy_version="selection-v1",
        ),
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path=path,
        ),
        slice_id="literature",
        source_version_key="source-version-v1:" + "a" * 64,
    )


def single_document_payload() -> SingleDocumentJobPayload:
    source = selected_source()
    return SingleDocumentJobPayload(
        run_id="single-run",
        source=source,
        approval=OperatorApproval(
            approval_id="approval:single-run:single_pdf",
            run_id="single-run",
            stage=CanaryStage.SINGLE_PDF,
            approved=True,
            source_ids=(source.source_id,),
        ),
    )


def metadata_result(run_id: str = "metadata-run") -> KnowledgeJobResult:
    return KnowledgeJobResult(
        job_type="metadata_only",
        run_id=run_id,
        code="metadata_completed",
        metadata_records=3,
    )


def test_in_memory_queue_is_idempotent_and_fences_expired_workers() -> None:
    async def exercise() -> None:
        clock = Clock()
        repository = InMemoryKnowledgeJobRepository(clock=clock)
        payload = metadata_payload()
        first = await repository.enqueue(payload)
        repeated = await repository.enqueue(payload)

        assert first == repeated
        assert first.idempotency_key == build_knowledge_job_idempotency_key(payload)

        stale = await repository.claim("worker-a", lease_seconds=5)
        assert stale is not None and stale.lease_token == 1
        clock.advance(6)
        current = await repository.claim("worker-b", lease_seconds=5)
        assert current is not None and current.lease_token == 2

        with pytest.raises(KnowledgeJobLeaseLost):
            await repository.complete(stale, metadata_result())
        completed = await repository.complete(current, metadata_result())
        repeated_completion = await repository.complete(current, metadata_result())
        assert completed == repeated_completion
        assert completed.status is KnowledgeJobStatus.SUCCEEDED

    asyncio.run(exercise())


def test_in_memory_queue_retries_then_succeeds() -> None:
    async def exercise() -> None:
        clock = Clock()
        repository = InMemoryKnowledgeJobRepository(clock=clock)
        record = await repository.enqueue(metadata_payload(), max_attempts=2)
        first = await repository.claim("worker-a", lease_seconds=30)
        assert first is not None
        failed = await repository.fail(
            first,
            code="provider.rate_limited",
            retryable=True,
            retry_delay_seconds=7,
        )
        repeated_failure = await repository.fail(
            first,
            code="provider.rate_limited",
            retryable=True,
            retry_delay_seconds=7,
        )
        assert failed == repeated_failure
        assert failed.status is KnowledgeJobStatus.RETRY_WAIT
        assert await repository.claim("worker-b", lease_seconds=30) is None

        clock.advance(7)
        second = await repository.claim("worker-b", lease_seconds=30)
        assert second is not None and second.record.attempt == 2
        completed = await repository.complete(second, metadata_result())
        assert completed.job_id == record.job_id
        assert completed.status is KnowledgeJobStatus.SUCCEEDED

    asyncio.run(exercise())


def test_executor_maps_metadata_canary_to_safe_summary() -> None:
    class Canaries:
        async def run_metadata(self, request: MetadataCanaryRequest) -> CanaryResult:
            return CanaryResult(
                run_id=request.run_id,
                stage=CanaryStage.METADATA_ONLY,
                status=CanaryStatus.SUCCEEDED,
                code="metadata_completed",
                attempt=1,
                metadata_records=17,
                next_stage_allowed=True,
            )

    executor = KnowledgeJobExecutor(
        canaries=Canaries(),  # type: ignore[arg-type]
        evidence=object(),  # type: ignore[arg-type]
        fact_extraction=object(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    result = asyncio.run(executor.execute(metadata_payload()))
    assert result.metadata_records == 17
    assert result.review_status == "not_applicable"


def test_single_document_executor_only_returns_pending_fact_batches() -> None:
    payload = single_document_payload()

    class Canaries:
        def __init__(self) -> None:
            self.registered = []

        async def register_selection(self, source: CanarySource) -> None:
            self.registered.append(source.source_id)

        async def run_single_pdf(self, **_: object) -> CanaryResult:
            return CanaryResult(
                run_id=payload.run_id,
                stage=CanaryStage.SINGLE_PDF,
                status=CanaryStatus.SUCCEEDED,
                code="single_pdf_completed",
                attempt=1,
                source_ids=(payload.source.source_id,),
                attempted_count=1,
                completed_count=1,
                evidence_count=2,
                next_stage_allowed=True,
            )

    class Evidence:
        async def list_for_source(self, *_: object, **__: object) -> list[object]:
            return [object(), object()]

    class Extraction:
        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, _fragment: object) -> object:
            self.calls += 1
            suffix = "a" if self.calls == 1 else "b"
            return SimpleNamespace(
                review_status="pending_review",
                batch=SimpleNamespace(batch_id="fact-batch:v1:" + suffix * 64),
            )

    canaries = Canaries()
    extraction = Extraction()
    executor = KnowledgeJobExecutor(
        canaries=canaries,  # type: ignore[arg-type]
        evidence=Evidence(),  # type: ignore[arg-type]
        fact_extraction=extraction,  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    result = asyncio.run(executor.execute(payload))

    assert canaries.registered == [payload.source.source_id]
    assert extraction.calls == 2
    assert result.evidence_count == 2
    assert result.review_status == "pending_review"
    assert len(result.pending_fact_batch_ids) == 2
    assert not hasattr(executor, "graph_writer")


def test_job_contracts_reject_invalid_scope_and_state() -> None:
    source = selected_source()
    approval = OperatorApproval(
        approval_id="approval:single-run:single_pdf",
        run_id="single-run",
        stage=CanaryStage.SINGLE_PDF,
        approved=True,
        source_ids=(source.source_id,),
    )
    with pytest.raises(ValueError, match="single_document_requires_selected_source"):
        SingleDocumentJobPayload(
            run_id="single-run",
            source=selected_source(selected=False),
            approval=approval,
        )
    with pytest.raises(ValueError, match="single_document_requires_pdf"):
        SingleDocumentJobPayload(
            run_id="single-run",
            source=selected_source(path="private/literature/paper.txt"),
            approval=approval,
        )
    with pytest.raises(ValueError, match="single_document_approval_mismatch"):
        SingleDocumentJobPayload(
            run_id="other-run",
            source=source,
            approval=approval,
        )
    with pytest.raises(ValueError, match="metadata_job_cannot_contain_facts"):
        KnowledgeJobResult(
            job_type="metadata_only",
            run_id="metadata-run",
            code="metadata_completed",
            pending_fact_batch_ids=("batch",),
        )
    with pytest.raises(ValueError, match="single_document_facts_must_remain_pending"):
        KnowledgeJobResult(
            job_type="single_document",
            run_id="single-run",
            code="single_pdf_completed",
        )

    base = {
        "job_id": UUID("22222222-2222-4222-8222-222222222222"),
        "idempotency_key": build_knowledge_job_idempotency_key(metadata_payload()),
        "payload": metadata_payload(),
        "status": KnowledgeJobStatus.QUEUED,
        "attempt": 0,
        "max_attempts": 2,
        "available_at": datetime(2026, 7, 27, tzinfo=UTC),
    }
    invalid_records = [
        ({**base, "available_at": datetime(2026, 7, 27)}, "knowledge_job_datetime"),
        ({**base, "attempt": 2, "max_attempts": 1}, "attempt_exceeds_maximum"),
        ({**base, "attempt": 1}, "queued_knowledge_job"),
        (
            {
                **base,
                "status": KnowledgeJobStatus.RETRY_WAIT,
                "attempt": 2,
                "max_attempts": 2,
                "last_error_code": "provider.failure",
            },
            "retrying_knowledge_job",
        ),
        ({**base, "lease_owner": "worker-a"}, "knowledge_job_lease_incomplete"),
        ({**base, "status": KnowledgeJobStatus.RUNNING}, "running_knowledge_job"),
        (
            {
                **base,
                "lease_owner": "worker-a",
                "lease_until": datetime(2026, 7, 27, 1, tzinfo=UTC),
            },
            "non_running_knowledge_job",
        ),
        ({**base, "status": KnowledgeJobStatus.SUCCEEDED}, "succeeded_knowledge_job"),
        ({**base, "result": metadata_result()}, "unfinished_knowledge_job"),
        ({**base, "status": KnowledgeJobStatus.RETRY_WAIT}, "failed_knowledge_job"),
        ({**base, "last_error_code": "provider.failure"}, "active_knowledge_job"),
    ]
    for candidate, message in invalid_records:
        with pytest.raises(ValueError, match=message):
            KnowledgeJobRecord.model_validate(candidate)


def test_repository_exhaustion_and_argument_guards() -> None:
    async def exercise() -> None:
        clock = Clock()
        repository = InMemoryKnowledgeJobRepository(clock=clock)
        queued = await repository.enqueue(metadata_payload(), max_attempts=1)
        lease = await repository.claim("worker-a", lease_seconds=1)
        assert lease is not None
        with pytest.raises(ValueError, match="knowledge_job_lease_mismatch"):
            KnowledgeJobLease(
                record=lease.record,
                worker_id="worker-b",
                lease_token=lease.lease_token,
                lease_until=lease.lease_until,
            )
        with pytest.raises(ValueError, match="invalid worker lease request"):
            await repository.claim("invalid worker!", lease_seconds=1)
        with pytest.raises(ValueError, match="lease_seconds must be positive"):
            await repository.renew(lease, lease_seconds=0)
        with pytest.raises(ValueError, match="retry delay cannot be negative"):
            await repository.fail(
                lease,
                code="provider.failure",
                retryable=True,
                retry_delay_seconds=-1,
            )

        clock.advance(2)
        assert await repository.claim("worker-b", lease_seconds=1) is None
        exhausted = await repository.get(queued.job_id)
        assert exhausted.status is KnowledgeJobStatus.FAILED_PERMANENT
        assert exhausted.last_error_code == "worker.lease_expired"

    asyncio.run(exercise())


def test_executor_fails_closed_on_invalid_review_or_dependency() -> None:
    payload = single_document_payload()

    class Canaries:
        async def register_selection(self, _source: CanarySource) -> None:
            return None

        async def run_single_pdf(self, **_: object) -> CanaryResult:
            return CanaryResult(
                run_id=payload.run_id,
                stage=CanaryStage.SINGLE_PDF,
                status=CanaryStatus.SUCCEEDED,
                code="single_pdf_completed",
                attempt=1,
                source_ids=(payload.source.source_id,),
                attempted_count=1,
                completed_count=1,
                evidence_count=1,
            )

    class Evidence:
        async def list_for_source(self, *_: object, **__: object) -> list[object]:
            return [object()]

    class InvalidExtraction:
        async def extract(self, _fragment: object) -> object:
            return SimpleNamespace(
                review_status="approved",
                batch=SimpleNamespace(batch_id=None),
            )

    with pytest.raises(ValueError, match="embedding_generation_id is required"):
        KnowledgeJobExecutor(
            canaries=Canaries(),  # type: ignore[arg-type]
            evidence=Evidence(),  # type: ignore[arg-type]
            fact_extraction=InvalidExtraction(),  # type: ignore[arg-type]
            embedding_generation_id=" ",
        )
    invalid = KnowledgeJobExecutor(
        canaries=Canaries(),  # type: ignore[arg-type]
        evidence=Evidence(),  # type: ignore[arg-type]
        fact_extraction=InvalidExtraction(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    with pytest.raises(KnowledgeJobExecutionError, match="fact_review_boundary_invalid"):
        asyncio.run(invalid.execute(payload))

    class BrokenCanaries(Canaries):
        async def register_selection(self, _source: CanarySource) -> None:
            raise RuntimeError("provider secret")

    broken = KnowledgeJobExecutor(
        canaries=BrokenCanaries(),  # type: ignore[arg-type]
        evidence=Evidence(),  # type: ignore[arg-type]
        fact_extraction=InvalidExtraction(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    with pytest.raises(KnowledgeJobExecutionError, match="worker.dependency_failure"):
        asyncio.run(broken.execute(payload))

    class FailedExtraction:
        async def extract(self, _fragment: object) -> object:
            raise FactExtractionError(
                "extraction.provider_unavailable",
                retryable=True,
            )

    failed = KnowledgeJobExecutor(
        canaries=Canaries(),  # type: ignore[arg-type]
        evidence=Evidence(),  # type: ignore[arg-type]
        fact_extraction=FailedExtraction(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    with pytest.raises(KnowledgeJobExecutionError) as raised:
        asyncio.run(failed.execute(payload))
    assert raised.value.code == "extraction.provider_unavailable"
    assert raised.value.retryable is True


def test_executor_preserves_retryability_for_blocked_canary() -> None:
    class BlockedCanaries:
        async def run_metadata(self, request: MetadataCanaryRequest) -> CanaryResult:
            return CanaryResult(
                run_id=request.run_id,
                stage=CanaryStage.METADATA_ONLY,
                status=CanaryStatus.BLOCKED,
                code="metadata_admission_denied",
                attempt=1,
            )

    executor = KnowledgeJobExecutor(
        canaries=BlockedCanaries(),  # type: ignore[arg-type]
        evidence=object(),  # type: ignore[arg-type]
        fact_extraction=object(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    with pytest.raises(KnowledgeJobExecutionError) as raised:
        asyncio.run(executor.execute(metadata_payload()))
    assert raised.value.code == "metadata_admission_denied"
    assert raised.value.retryable is True

    class BrokenCanaries:
        async def run_metadata(self, _request: MetadataCanaryRequest) -> CanaryResult:
            raise RuntimeError("provider credential detail")

    broken = KnowledgeJobExecutor(
        canaries=BrokenCanaries(),  # type: ignore[arg-type]
        evidence=object(),  # type: ignore[arg-type]
        fact_extraction=object(),  # type: ignore[arg-type]
        embedding_generation_id="embedding-v1",
    )
    with pytest.raises(KnowledgeJobExecutionError) as dependency:
        asyncio.run(broken.execute(metadata_payload()))
    assert dependency.value.code == "worker.dependency_failure"
