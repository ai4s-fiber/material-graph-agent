from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from uuid import UUID

import pytest

from material_graph.knowledge.bindings import ProviderBindings
from material_graph.knowledge.concurrency import AdmissionSnapshot, GlobalAdmissionController
from material_graph.knowledge.ingestion import (
    FailureDisposition,
    InMemoryCheckpointRepository,
    IngestionPipelineError,
    IngestionResult,
)
from material_graph.knowledge.manifest import (
    MetadataCursor,
    MetadataStreamError,
    MetadataStreamResult,
)
from material_graph.knowledge.models import SelectionDecision, SourceLocator
from material_graph.knowledge.policy import CorpusPolicy
from material_graph.knowledge.processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    SourceLifecycleStatus,
)
from material_graph.knowledge.service import (
    CanarySource,
    CanaryStage,
    CanaryStageRecord,
    CanaryStatus,
    InMemoryCanaryRunRepository,
    KnowledgeCanaryError,
    KnowledgeCanaryPolicy,
    KnowledgeCanaryService,
    MetadataCanaryRequest,
    OperatorApproval,
)


CONFIG_ROOT = Path(__file__).parents[1] / "config" / "knowledge"


class FakeMetadataRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.wait: asyncio.Event | None = None

    async def ingest(
        self,
        *,
        root_id: str,
        slice_id: str,
        manifest_path: str,
        manifest_format: str,
    ) -> MetadataStreamResult:
        self.calls += 1
        if self.wait is not None:
            await self.wait.wait()
        if self.error is not None:
            raise self.error
        cursor = MetadataCursor(
            root_id=root_id,
            slice_id=slice_id,
            manifest_path=manifest_path,
            manifest_format=manifest_format,  # type: ignore[arg-type]
            manifest_version_key="source-version-v1:" + "a" * 64,
            next_byte_offset=128,
            records_committed=3,
        )
        return MetadataStreamResult(
            cursor=cursor,
            records_seen=3,
            records_created=2,
            records_updated=1,
            bounded_digest_required=1,
        )


class FakeBodyRunner:
    def __init__(self) -> None:
        self.calls: list[UUID] = []
        self.indexed: set[UUID] = set()
        self.fail_source: UUID | None = None
        self.fail_unknown = False
        self.wait: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.active_spool = 0

    async def ingest(
        self,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        source_version_key: str,
        embedding_generation_id: str,
    ) -> IngestionResult:
        del source_locator, slice_id, source_version_key, embedding_generation_id
        self.calls.append(decision.source_id)
        self.active_spool += 1
        self.started.set()
        try:
            if self.wait is not None:
                await self.wait.wait()
            if self.fail_unknown:
                raise RuntimeError("provider=https://secret.invalid token=must-not-leak")
            if decision.source_id == self.fail_source:
                raise IngestionPipelineError(
                    disposition=FailureDisposition.RETRYABLE,
                    stage=IngestionStage.SPOOL,
                    category="spool.capacity",
                )
            resumed = decision.source_id in self.indexed
            self.indexed.add(decision.source_id)
            checkpoint = ProcessingCheckpoint(
                source_id=decision.source_id,
                lifecycle_status=SourceLifecycleStatus.EVIDENCE_RETAINED,
                stage=IngestionStage.INDEX,
                job_status=IngestionJobStatus.SUCCEEDED,
                idempotency_key=f"fake:{decision.source_id.hex}",
                selection=decision,
            )
            return IngestionResult(
                outcome="indexed",
                checkpoint=checkpoint,
                evidence_count=2,
                index_outcome="processed",
                resumed=resumed,
            )
        finally:
            self.active_spool -= 1


class FakeAdmissionSnapshots:
    def __init__(self, *snapshots: AdmissionSnapshot) -> None:
        self.snapshots = deque(snapshots or (AdmissionSnapshot(),))
        self.calls = 0
        self.error: Exception | None = None

    async def snapshot(self) -> AdmissionSnapshot:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if len(self.snapshots) > 1:
            return self.snapshots.popleft()
        return self.snapshots[0]


class CancelAfterCalls:
    def __init__(self, body: FakeBodyRunner, count: int) -> None:
        self.body = body
        self.count = count

    def is_cancelled(self) -> bool:
        return len(self.body.calls) >= self.count


class ToggleCheckpointRepository(InMemoryCheckpointRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_load = False
        self.fail_save = False

    async def load(self, idempotency_key: str) -> ProcessingCheckpoint | None:
        if self.fail_load:
            raise RuntimeError("checkpoint provider detail")
        return await super().load(idempotency_key)

    async def save(self, checkpoint: ProcessingCheckpoint) -> None:
        if self.fail_save:
            raise RuntimeError("checkpoint provider detail")
        await super().save(checkpoint)


class ToggleRunRepository(InMemoryCanaryRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_load_stage: CanaryStage | None = None
        self.fail_begin: Exception | None = None
        self.fail_finish = False

    async def load(self, run_id: str, stage: CanaryStage) -> CanaryStageRecord | None:
        if stage is self.fail_load_stage:
            raise RuntimeError("run store provider detail")
        return await super().load(run_id, stage)

    async def begin(self, **kwargs: object) -> CanaryStageRecord:
        if self.fail_begin is not None:
            raise self.fail_begin
        return await super().begin(**kwargs)  # type: ignore[arg-type]

    async def finish(self, record: CanaryStageRecord) -> None:
        if self.fail_finish:
            raise RuntimeError("run store provider detail")
        await super().finish(record)


def _source(seed: str = "1", *, selected: bool = True, path: str | None = None) -> CanarySource:
    source_id = UUID(seed.rjust(32, "0"))
    return CanarySource(
        decision=SelectionDecision(
            source_id=source_id,
            selected=selected,
            reason_code="active_evidence_gap" if selected else "budget_deferred",
            policy_version="selection-v1",
        ),
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path=path or f"private/literature/{seed}.pdf",
        ),
        slice_id="literature",
        source_version_key="source-version-v1:" + seed[-1] * 64,
    )


def _metadata_request(run_id: str = "canary-run") -> MetadataCanaryRequest:
    return MetadataCanaryRequest(
        run_id=run_id,
        root_id="document_data_1",
        slice_id="literature",
        manifest_path="private/manifests/catalog.jsonl",
        manifest_format="jsonl",
    )


def _approval(
    run_id: str, stage: CanaryStage, sources: tuple[CanarySource, ...]
) -> OperatorApproval:
    return OperatorApproval(
        approval_id=f"approval:{run_id}:{stage.value}",
        run_id=run_id,
        stage=stage,  # type: ignore[arg-type]
        approved=True,
        source_ids=tuple(source.source_id for source in sources),
    )


def _service(
    *,
    max_stage: CanaryStage = CanaryStage.SMALL_BATCH,
    metadata: FakeMetadataRunner | None = None,
    body: FakeBodyRunner | None = None,
    snapshots: FakeAdmissionSnapshots | None = None,
    checkpoints: InMemoryCheckpointRepository | None = None,
    runs: InMemoryCanaryRunRepository | None = None,
    max_batch: int = 8,
) -> tuple[
    KnowledgeCanaryService,
    FakeMetadataRunner,
    FakeBodyRunner,
    InMemoryCheckpointRepository,
    InMemoryCanaryRunRepository,
]:
    metadata_runner = metadata or FakeMetadataRunner()
    body_runner = body or FakeBodyRunner()
    checkpoint_repository = checkpoints or InMemoryCheckpointRepository()
    run_repository = runs or InMemoryCanaryRunRepository()
    service = KnowledgeCanaryService(
        metadata=metadata_runner,
        ingestion=body_runner,
        checkpoints=checkpoint_repository,
        runs=run_repository,
        admission=GlobalAdmissionController(),
        admission_snapshots=snapshots or FakeAdmissionSnapshots(),
        corpus_policy=CorpusPolicy.load(CONFIG_ROOT / "corpus-policy.v1.json"),
        bindings=ProviderBindings.load(
            embedding_path=CONFIG_ROOT / "embedding-binding.v1.json",
            reranker_path=CONFIG_ROOT / "reranker-binding.v1.json",
        ),
        policy=KnowledgeCanaryPolicy(
            max_enabled_stage=max_stage,
            max_small_batch_sources=max_batch,
        ),
    )
    return (
        service,
        metadata_runner,
        body_runner,
        checkpoint_repository,
        run_repository,
    )


@pytest.mark.asyncio
async def test_default_policy_runs_metadata_only_and_never_auto_starts_body() -> None:
    service, metadata, body, _, _ = _service(max_stage=CanaryStage.METADATA_ONLY)
    source = _source()

    metadata_result = await service.run_metadata(_metadata_request())
    await service.register_selection(source)
    body_result = await service.run_single_pdf(
        run_id="canary-run",
        source=source,
        approval=_approval("canary-run", CanaryStage.SINGLE_PDF, (source,)),
    )

    assert metadata_result.status is CanaryStatus.SUCCEEDED
    assert metadata_result.next_stage_allowed is False
    assert body_result.code == "canary_stage_disabled"
    assert metadata.calls == 1
    assert body.calls == []


@pytest.mark.asyncio
async def test_metadata_success_is_idempotent_and_result_never_exposes_locator() -> None:
    service, metadata, _, _, _ = _service()
    request = _metadata_request("metadata-safe")

    first = await service.run_metadata(request)
    second = await service.run_metadata(request)
    serialized = second.model_dump_json()

    assert first.metadata_records == 3
    assert first.next_stage_allowed is True
    assert second.idempotent is True
    assert metadata.calls == 1
    assert request.manifest_path not in serialized
    assert "https://" not in serialized
    assert "token" not in serialized.casefold()


@pytest.mark.asyncio
async def test_metadata_failure_blocks_single_pdf_until_retry_succeeds() -> None:
    metadata = FakeMetadataRunner()
    metadata.error = MetadataStreamError(
        "metadata.provider.stream_failed",
        failure_class="provider",
        retryable=True,
    )
    service, _, body, _, _ = _service(metadata=metadata)
    source = _source()
    await service.register_selection(source)

    failed = await service.run_metadata(_metadata_request("retry-run"))
    blocked = await service.run_single_pdf(
        run_id="retry-run",
        source=source,
        approval=_approval("retry-run", CanaryStage.SINGLE_PDF, (source,)),
    )

    assert failed.code == "metadata.provider.stream_failed"
    assert blocked.code == "canary_prerequisite_not_satisfied"
    assert body.calls == []

    metadata.error = None
    recovered = await service.run_metadata(_metadata_request("retry-run"))
    single = await service.run_single_pdf(
        run_id="retry-run",
        source=source,
        approval=_approval("retry-run", CanaryStage.SINGLE_PDF, (source,)),
    )
    assert recovered.status is CanaryStatus.SUCCEEDED
    assert single.status is CanaryStatus.SUCCEEDED
    assert single.resumed is True


@pytest.mark.asyncio
async def test_body_requires_durable_selected_true_and_exact_operator_approval() -> None:
    service, _, body, _, _ = _service()
    source = _source()
    await service.run_metadata(_metadata_request("gates"))

    missing = await service.run_single_pdf(
        run_id="gates",
        source=source,
        approval=_approval("gates", CanaryStage.SINGLE_PDF, (source,)),
    )
    assert missing.code == "persisted_selection_required"

    with pytest.raises(KnowledgeCanaryError, match="selected_decision_required"):
        await service.register_selection(_source("2", selected=False))

    receipt = await service.register_selection(source)
    repeated = await service.register_selection(source)
    wrong_source = _source("2")
    invalid = await service.run_single_pdf(
        run_id="gates",
        source=source,
        approval=_approval("gates", CanaryStage.SINGLE_PDF, (wrong_source,)),
    )

    assert receipt.persisted is True
    assert repeated.code == "selection_already_persisted"
    assert invalid.code == "operator_approval_invalid"
    assert body.calls == []


@pytest.mark.asyncio
async def test_body_admission_checks_spool_and_disk_before_opening_source() -> None:
    snapshots = FakeAdmissionSnapshots(
        AdmissionSnapshot(),
        AdmissionSnapshot(filesystem_used_ratio=0.75),
    )
    service, _, body, _, _ = _service(snapshots=snapshots)
    source = _source()
    await service.run_metadata(_metadata_request("pressure"))
    await service.register_selection(source)

    result = await service.run_single_pdf(
        run_id="pressure",
        source=source,
        approval=_approval("pressure", CanaryStage.SINGLE_PDF, (source,)),
    )

    assert result.status is CanaryStatus.BLOCKED
    assert result.code == "body_admission_denied"
    assert body.calls == []
    assert body.active_spool == 0


@pytest.mark.asyncio
async def test_single_pdf_success_unlocks_but_does_not_start_small_batch() -> None:
    service, _, body, _, _ = _service()
    source = _source()
    await service.run_metadata(_metadata_request("single"))
    await service.register_selection(source)

    first = await service.run_single_pdf(
        run_id="single",
        source=source,
        approval=_approval("single", CanaryStage.SINGLE_PDF, (source,)),
    )
    second = await service.run_single_pdf(
        run_id="single",
        source=source,
        approval=_approval("single", CanaryStage.SINGLE_PDF, (source,)),
    )

    assert first.status is CanaryStatus.SUCCEEDED
    assert first.next_stage_allowed is True
    assert first.completed_count == 1
    assert first.evidence_count == 2
    assert second.idempotent is True
    assert body.calls == [source.source_id]


@pytest.mark.asyncio
async def test_small_batch_is_bounded_unique_and_requires_single_success() -> None:
    service, _, body, _, _ = _service(max_batch=2)
    first, second, third = _source("1"), _source("2"), _source("3")
    await service.run_metadata(_metadata_request("batch-gates"))

    too_large = await service.run_small_batch(
        run_id="batch-gates",
        sources=(first, second, third),
        approval=_approval("batch-gates", CanaryStage.SMALL_BATCH, (first, second, third)),
    )
    duplicate = await service.run_small_batch(
        run_id="batch-gates",
        sources=(first, first),
        approval=_approval("batch-gates", CanaryStage.SMALL_BATCH, (first,)),
    )
    prerequisite = await service.run_small_batch(
        run_id="batch-gates",
        sources=(first, second),
        approval=_approval("batch-gates", CanaryStage.SMALL_BATCH, (first, second)),
    )

    assert too_large.code == "small_batch_size_invalid"
    assert duplicate.code == "small_batch_duplicate_source"
    assert prerequisite.code == "canary_prerequisite_not_satisfied"
    assert body.calls == []


@pytest.mark.asyncio
async def test_small_batch_cancellation_resumes_idempotently_and_cleans_spool() -> None:
    service, _, body, _, _ = _service()
    sources = (_source("1"), _source("2"), _source("3"))
    run_id = "cancel-resume"
    await service.run_metadata(_metadata_request(run_id))
    for source in sources:
        await service.register_selection(source)
    await service.run_single_pdf(
        run_id=run_id,
        source=sources[0],
        approval=_approval(run_id, CanaryStage.SINGLE_PDF, (sources[0],)),
    )

    cancelled = await service.run_small_batch(
        run_id=run_id,
        sources=sources,
        approval=_approval(run_id, CanaryStage.SMALL_BATCH, sources),
        cancellation=CancelAfterCalls(body, 2),
    )
    resumed = await service.run_small_batch(
        run_id=run_id,
        sources=sources,
        approval=_approval(run_id, CanaryStage.SMALL_BATCH, sources),
    )

    assert cancelled.status is CanaryStatus.CANCELLED
    assert cancelled.completed_count == 1
    assert resumed.status is CanaryStatus.SUCCEEDED
    assert resumed.resumed is True
    assert resumed.completed_count == 3
    assert body.active_spool == 0
    assert body.indexed == {source.source_id for source in sources}


@pytest.mark.asyncio
async def test_body_failure_is_sanitized_cleans_spool_and_blocks_next_stage() -> None:
    body = FakeBodyRunner()
    source = _source()
    body.fail_source = source.source_id
    service, _, _, _, _ = _service(body=body)
    await service.run_metadata(_metadata_request("body-failure"))
    await service.register_selection(source)

    failed = await service.run_single_pdf(
        run_id="body-failure",
        source=source,
        approval=_approval("body-failure", CanaryStage.SINGLE_PDF, (source,)),
    )
    second = _source("2")
    batch = await service.run_small_batch(
        run_id="body-failure",
        sources=(source, second),
        approval=_approval("body-failure", CanaryStage.SMALL_BATCH, (source, second)),
    )

    assert failed.status is CanaryStatus.FAILED
    assert failed.code == "spool.capacity"
    assert body.active_spool == 0
    assert batch.code == "canary_prerequisite_not_satisfied"


@pytest.mark.asyncio
async def test_unknown_provider_failure_never_crosses_result_boundary() -> None:
    body = FakeBodyRunner()
    body.fail_unknown = True
    service, _, _, _, _ = _service(body=body)
    source = _source()
    await service.run_metadata(_metadata_request("sanitize"))
    await service.register_selection(source)

    result = await service.run_single_pdf(
        run_id="sanitize",
        source=source,
        approval=_approval("sanitize", CanaryStage.SINGLE_PDF, (source,)),
    )
    serialized = result.model_dump_json()

    assert result.code == "canary_dependency_failure"
    assert "secret.invalid" not in serialized
    assert "must-not-leak" not in serialized
    assert source.locator.relative_path not in serialized


@pytest.mark.asyncio
async def test_external_cancellation_records_state_and_always_cleans_spool() -> None:
    body = FakeBodyRunner()
    body.wait = asyncio.Event()
    service, _, _, _, runs = _service(body=body)
    source = _source()
    run_id = "external-cancel"
    await service.run_metadata(_metadata_request(run_id))
    await service.register_selection(source)

    task = asyncio.create_task(
        service.run_single_pdf(
            run_id=run_id,
            source=source,
            approval=_approval(run_id, CanaryStage.SINGLE_PDF, (source,)),
        )
    )
    await body.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = await runs.load(run_id, CanaryStage.SINGLE_PDF)
    assert record is not None
    assert record.status is CanaryStatus.CANCELLED
    assert body.active_spool == 0


@pytest.mark.asyncio
async def test_persisted_selection_identity_mismatch_fails_closed() -> None:
    service, _, body, _, _ = _service()
    source = _source()
    await service.run_metadata(_metadata_request("identity"))
    await service.register_selection(source)
    changed = _source(path="different/private/path.pdf")

    result = await service.run_single_pdf(
        run_id="identity",
        source=changed,
        approval=_approval("identity", CanaryStage.SINGLE_PDF, (changed,)),
    )

    assert result.code == "persisted_selection_mismatch"
    assert body.calls == []


@pytest.mark.asyncio
async def test_run_repository_claims_are_atomic_and_finish_is_fail_closed() -> None:
    runs = InMemoryCanaryRunRepository()
    first = await runs.begin(
        run_id="atomic",
        stage=CanaryStage.METADATA_ONLY,
        request_fingerprint="a" * 64,
        source_ids=(),
        approval_id=None,
    )
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_running"):
        await runs.begin(
            run_id="atomic",
            stage=CanaryStage.METADATA_ONLY,
            request_fingerprint="a" * 64,
            source_ids=(),
            approval_id=None,
        )
    with pytest.raises(ValueError, match="running"):
        await runs.finish(first)

    finished = CanaryStageRecord.model_validate(
        {**first.model_dump(mode="python"), "status": "succeeded", "code": "done"}
    )
    await runs.finish(finished)
    await runs.finish(finished)

    stale = finished.model_copy(update={"attempt": 99, "status": CanaryStatus.FAILED})
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        await runs.finish(stale)

    conflicting = finished.model_copy(update={"code": "different"})
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        await runs.finish(conflicting)

    with pytest.raises(KnowledgeCanaryError, match="canary_stage_identity_mismatch"):
        await runs.begin(
            run_id="atomic",
            stage=CanaryStage.METADATA_ONLY,
            request_fingerprint="b" * 64,
            source_ids=(),
            approval_id=None,
        )


def test_canary_dtos_reject_ambiguous_approval_and_invalid_stage_state() -> None:
    source_id = _source().source_id
    with pytest.raises(ValueError, match="unique"):
        OperatorApproval(
            approval_id="approval:duplicate",
            run_id="dto",
            stage=CanaryStage.SINGLE_PDF,
            approved=True,
            source_ids=(source_id, source_id),
        )
    base = {
        "run_id": "dto",
        "stage": CanaryStage.METADATA_ONLY,
        "status": CanaryStatus.FAILED,
        "attempt": 1,
        "code": "failed",
        "request_fingerprint": "a" * 64,
    }
    with pytest.raises(ValueError, match="completed_count"):
        CanaryStageRecord(**base, attempted_count=0, completed_count=1)
    with pytest.raises(ValueError, match="metadata canary"):
        CanaryStageRecord(**base, source_ids=(source_id,), approval_id="approval:bad")
    with pytest.raises(ValueError, match="body canary"):
        CanaryStageRecord(**{**base, "stage": CanaryStage.SINGLE_PDF})


@pytest.mark.asyncio
async def test_checkpoint_failures_are_stable_and_never_open_body() -> None:
    checkpoints = ToggleCheckpointRepository()
    service, _, body, _, _ = _service(checkpoints=checkpoints)
    source = _source()

    checkpoints.fail_load = True
    with pytest.raises(KnowledgeCanaryError, match="checkpoint_read_failed"):
        await service.register_selection(source)
    checkpoints.fail_load = False
    checkpoints.fail_save = True
    with pytest.raises(KnowledgeCanaryError, match="checkpoint_write_failed"):
        await service.register_selection(source)

    checkpoints.fail_save = False
    await service.register_selection(source)
    await service.run_metadata(_metadata_request("checkpoint-body"))
    checkpoints.fail_load = True
    result = await service.run_single_pdf(
        run_id="checkpoint-body",
        source=source,
        approval=_approval("checkpoint-body", CanaryStage.SINGLE_PDF, (source,)),
    )
    assert result.code == "checkpoint_read_failed"
    assert body.calls == []


@pytest.mark.asyncio
async def test_metadata_admission_unknown_root_and_cancellation_fail_closed() -> None:
    denied, _, _, _, _ = _service(
        snapshots=FakeAdmissionSnapshots(AdmissionSnapshot(filesystem_used_ratio=0.8))
    )
    denied_result = await denied.run_metadata(_metadata_request("metadata-denied"))
    unknown_request = _metadata_request("unknown-root").model_copy(
        update={"root_id": "unknown_root"}
    )
    unknown_result = await denied.run_metadata(unknown_request)

    assert denied_result.code == "metadata_admission_denied"
    assert unknown_result.code == "corpus_root_not_approved"

    metadata = FakeMetadataRunner()
    metadata.wait = asyncio.Event()
    service, _, _, _, runs = _service(metadata=metadata)
    task = asyncio.create_task(service.run_metadata(_metadata_request("metadata-cancel")))
    while metadata.calls == 0:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    record = await runs.load("metadata-cancel", CanaryStage.METADATA_ONLY)
    assert record is not None and record.status is CanaryStatus.CANCELLED


@pytest.mark.asyncio
async def test_request_identity_non_pdf_and_invalid_run_id_are_rejected() -> None:
    service, metadata, body, _, _ = _service()
    request = _metadata_request("identity-fingerprint")
    assert (await service.run_metadata(request)).status is CanaryStatus.SUCCEEDED
    changed = request.model_copy(update={"manifest_path": "private/manifests/other.jsonl"})
    mismatch = await service.run_metadata(changed)
    assert mismatch.code == "canary_stage_identity_mismatch"
    assert metadata.calls == 1

    non_pdf = _source(path="private/literature/not-a-pdf.txt")
    await service.register_selection(non_pdf)
    blocked = await service.run_single_pdf(
        run_id="identity-fingerprint",
        source=non_pdf,
        approval=_approval(
            "identity-fingerprint",
            CanaryStage.SINGLE_PDF,
            (non_pdf,),
        ),
    )
    assert blocked.code == "body_source_not_pdf"
    assert body.calls == []

    with pytest.raises(KnowledgeCanaryError, match="canary_run_id_invalid"):
        await service.run_single_pdf(
            run_id="unsafe/run",
            source=non_pdf,
            approval=_approval("identity-fingerprint", CanaryStage.SINGLE_PDF, (non_pdf,)),
        )


@pytest.mark.asyncio
async def test_run_store_and_admission_failures_return_only_stable_codes() -> None:
    runs = ToggleRunRepository()
    service, _, _, _, _ = _service(runs=runs)
    runs.fail_load_stage = CanaryStage.METADATA_ONLY
    read_failed = await service.run_metadata(_metadata_request("run-read"))
    assert read_failed.code == "canary_state_read_failed"

    runs.fail_load_stage = None
    runs.fail_begin = KnowledgeCanaryError("canary_stage_running")
    begin_blocked = await service.run_metadata(_metadata_request("run-begin-known"))
    assert begin_blocked.code == "canary_stage_running"

    runs.fail_begin = RuntimeError("provider detail")
    begin_failed = await service.run_metadata(_metadata_request("run-begin-unknown"))
    assert begin_failed.code == "canary_state_write_failed"

    runs.fail_begin = None
    runs.fail_finish = True
    finish_failed = await service.run_metadata(_metadata_request("run-finish"))
    assert finish_failed.code == "canary_state_write_failed"

    snapshots = FakeAdmissionSnapshots()
    snapshots.error = RuntimeError("disk probe provider detail")
    admission_service, _, _, _, _ = _service(snapshots=snapshots)
    admission_failed = await admission_service.run_metadata(_metadata_request("probe-fail"))
    assert admission_failed.code == "admission_probe_failed"


@pytest.mark.asyncio
async def test_prerequisite_state_failure_is_sanitized() -> None:
    runs = ToggleRunRepository()
    service, _, body, _, _ = _service(runs=runs)
    source = _source()
    await service.register_selection(source)
    runs.fail_load_stage = CanaryStage.METADATA_ONLY

    result = await service.run_single_pdf(
        run_id="prerequisite-store",
        source=source,
        approval=_approval("prerequisite-store", CanaryStage.SINGLE_PDF, (source,)),
    )

    assert result.code == "canary_state_read_failed"
    assert body.calls == []
    await service._require_prerequisite("unused", CanaryStage.METADATA_ONLY)
