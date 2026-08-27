from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from material_graph.knowledge.ingestion import (
    FailureDisposition,
    InMemoryCheckpointRepository,
    InMemoryEvidenceRepository,
    IngestionCancelledError,
    IngestionPipelineError,
    KnowledgeIngestionPipeline,
    build_ingestion_idempotency_key,
)
from material_graph.knowledge.lightrag_client import (
    LightRAGError,
    LightRAGPollingTimeout,
    LightRAGProtocolError,
    LightRAGRequestError,
)
from material_graph.knowledge.mineru_client import (
    MinerUBlock,
    MinerUError,
    MinerUParseResult,
)
from material_graph.knowledge.models import (
    EvidenceFragment,
    SelectionDecision,
    SourceLocator,
)
from material_graph.knowledge.policy import SpoolPolicy
from material_graph.knowledge.processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    SourceLifecycleStatus,
)
from material_graph.knowledge.remote_reader import (
    DirectoryCursor,
    RemoteEntry,
    RemoteSourceReader,
    RemoteStat,
)
from material_graph.knowledge.retention import (
    BlockEvidenceAssessment,
    EvidenceSelector,
)
from material_graph.knowledge.spool import (
    CapacitySnapshot,
    SpoolCapacityError,
    SpoolError,
    SpoolIntegrityError,
    SpoolManager,
)


GENERATION = "qwen3-embedding-8b:1024:v1"
VERSION_A = "source-version-v1:" + "a" * 64


class FakeReader(RemoteSourceReader):
    def __init__(self, body: bytes = b"selected-pdf") -> None:
        self.body = body
        self.stat_calls = 0
        self.open_calls = 0
        self.closed = False
        self.stat_override: RemoteStat | None = None
        self.error: Exception | None = None

    def iter_entries(
        self,
        root_id: str,
        slice_id: str,
        *,
        cursor: DirectoryCursor | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[RemoteEntry]:
        del root_id, slice_id, cursor, page_size

        async def empty() -> AsyncIterator[RemoteEntry]:
            if False:  # pragma: no cover
                yield RemoteEntry(
                    root_id="unused",
                    slice_id="unused",
                    relative_path="unused.pdf",
                    name="unused.pdf",
                    is_dir=False,
                )

        return empty()

    async def stat(self, root_id: str, slice_id: str, relative_path: str) -> RemoteStat:
        self.stat_calls += 1
        if self.error is not None:
            raise self.error
        return self.stat_override or RemoteStat(
            root_id=root_id,
            slice_id=slice_id,
            relative_path=relative_path,
            is_dir=False,
            byte_size=len(self.body),
            modified_at=123,
        )

    def open_stream(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
        *,
        offset: int = 0,
        expected_size: int | None = None,
        expected_mtime: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        del root_id, slice_id, relative_path, offset, expected_size, expected_mtime, chunk_size
        self.open_calls += 1

        async def stream() -> AsyncIterator[bytes]:
            midpoint = max(1, len(self.body) // 2)
            yield self.body[:midpoint]
            yield self.body[midpoint:]

        return stream()

    async def close(self) -> None:
        self.closed = True


class FakeMinerU:
    def __init__(self) -> None:
        self.calls = 0
        self.keys: list[str] = []
        self.error: BaseException | None = None

    async def parse(
        self,
        source_path: str | Path,
        *,
        file_name: str,
        idempotency_key: str,
        output_dir: str | Path,
    ) -> MinerUParseResult:
        self.calls += 1
        self.keys.append(idempotency_key)
        assert Path(source_path).is_file()
        assert file_name == "paper.pdf"
        output = Path(output_dir)
        (output / "complete-output.json").write_text("transient parser result", encoding="utf-8")
        if self.error is not None:
            raise self.error
        return MinerUParseResult(
            batch_id="batch-1",
            task_id="task-1",
            filename=file_name,
            parser_version="3.4.4",
            model_version="vlm",
            blocks=[
                MinerUBlock(
                    block_type="text",
                    text="Polyimide evidence with dielectric constant below three.",
                    page=2,
                    block_index=4,
                    section="Results",
                )
            ],
        )


class FakeAssessor:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls = 0

    async def assess(
        self,
        parsed: MinerUParseResult,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
    ) -> Sequence[BlockEvidenceAssessment]:
        del decision, source_locator
        self.calls += 1
        return [
            BlockEvidenceAssessment(
                block_index=parsed.blocks[0].block_index,
                accepted=self.accepted,
                confidence=0.95 if self.accepted else 0.1,
                retention_reason="supports_gap" if self.accepted else "no_support",
                evidence_gap_ids=["gap-1"],
            )
        ]


class FakeLightRAG:
    def __init__(self, outcome: str = "processed") -> None:
        self.outcome = outcome
        self.calls = 0
        self.fragments: list[EvidenceFragment] = []
        self.error: Exception | None = None

    async def insert_retained_fragments(
        self,
        fragments: Sequence[EvidenceFragment],
    ) -> SimpleNamespace:
        self.calls += 1
        self.fragments = [fragment.model_copy(deep=True) for fragment in fragments]
        if self.error is not None:
            raise self.error
        return SimpleNamespace(outcome=self.outcome)


class FailingCheckpointRepository:
    def __init__(self, *, fail_load: bool = False) -> None:
        self.fail_load = fail_load

    async def load(self, idempotency_key: str) -> ProcessingCheckpoint | None:
        del idempotency_key
        if self.fail_load:
            raise RuntimeError("secret checkpoint endpoint")
        return None

    async def save(self, checkpoint: ProcessingCheckpoint) -> None:
        del checkpoint
        raise RuntimeError("secret checkpoint endpoint")


def _capacity() -> CapacitySnapshot:
    return CapacitySnapshot(
        filesystem_total_bytes=10_000,
        filesystem_used_bytes=1_000,
        filesystem_free_bytes=9_000,
        derived_active_bytes=100,
        derived_projected_bytes=100,
        minimum_free_bytes=1_000,
        hard_stop_free_bytes=500,
        derived_target_bytes=8_000,
        derived_hard_cap_bytes=9_000,
        filesystem_alert_ratio=0.8,
        filesystem_stop_ratio=0.9,
    )


def _spool(tmp_path: Path) -> SpoolManager:
    return SpoolManager(
        tmp_path,
        SpoolPolicy(
            max_total_bytes=1_024,
            max_object_bytes=512,
            max_active_objects=2,
            abandoned_ttl_seconds=60,
        ),
        capacity_probe=_capacity,
    )


def _decision(source_id: UUID, *, selected: bool = True) -> SelectionDecision:
    return SelectionDecision(
        source_id=source_id,
        selected=selected,
        reason_code="active_evidence_gap" if selected else "budget_deferred",
        policy_version="selection-v1",
    )


def _locator() -> SourceLocator:
    return SourceLocator(
        root_id="document_data_1",
        relative_path="polymers/paper.pdf",
    )


def _pipeline(
    tmp_path: Path,
    *,
    reader: FakeReader | None = None,
    mineru: FakeMinerU | None = None,
    assessor: FakeAssessor | None = None,
    lightrag: FakeLightRAG | None = None,
    checkpoints: InMemoryCheckpointRepository | None = None,
    evidence: InMemoryEvidenceRepository | None = None,
) -> tuple[
    KnowledgeIngestionPipeline,
    FakeReader,
    FakeMinerU,
    FakeAssessor,
    FakeLightRAG,
    InMemoryCheckpointRepository,
    InMemoryEvidenceRepository,
]:
    resolved_reader = reader or FakeReader()
    resolved_mineru = mineru or FakeMinerU()
    resolved_assessor = assessor or FakeAssessor()
    resolved_lightrag = lightrag or FakeLightRAG()
    resolved_checkpoints = checkpoints or InMemoryCheckpointRepository()
    resolved_evidence = evidence or InMemoryEvidenceRepository()
    pipeline = KnowledgeIngestionPipeline(
        reader=resolved_reader,
        spool=_spool(tmp_path),
        mineru=resolved_mineru,  # type: ignore[arg-type]
        evidence_selector=EvidenceSelector(),
        assessment_provider=resolved_assessor,
        lightrag=resolved_lightrag,  # type: ignore[arg-type]
        checkpoints=resolved_checkpoints,
        evidence=resolved_evidence,
    )
    return (
        pipeline,
        resolved_reader,
        resolved_mineru,
        resolved_assessor,
        resolved_lightrag,
        resolved_checkpoints,
        resolved_evidence,
    )


async def _ingest(
    pipeline: KnowledgeIngestionPipeline,
    decision: SelectionDecision,
    *,
    source_version_key: str = VERSION_A,
    embedding_generation_id: str = GENERATION,
):
    return await pipeline.ingest(
        decision=decision,
        source_locator=_locator(),
        slice_id="literature",
        source_version_key=source_version_key,
        embedding_generation_id=embedding_generation_id,
    )


def _key(
    source_id: UUID,
    *,
    source_version_key: str = VERSION_A,
    embedding_generation_id: str = GENERATION,
) -> str:
    return build_ingestion_idempotency_key(
        source_id,
        source_version_key=source_version_key,
        embedding_generation_id=embedding_generation_id,
    )


def _resume_checkpoint(
    source_id: UUID,
    decision: SelectionDecision,
    *,
    stage: IngestionStage,
    index_completed: bool = False,
) -> ProcessingCheckpoint:
    metadata: dict[str, object] = {
        "root_id": "document_data_1",
        "slice_id": "literature",
        "relative_path": "polymers/paper.pdf",
        "embedding_generation_id": GENERATION,
        "source_version_fingerprint": sha256(VERSION_A.encode("utf-8")).hexdigest(),
        "parse_completed": True,
        "evidence_persisted": True,
        "fragment_count": 1,
    }
    if index_completed:
        metadata.update(index_completed=True, index_outcome="processed")
    return ProcessingCheckpoint(
        source_id=source_id,
        lifecycle_status=SourceLifecycleStatus.PARSE_ELIGIBLE,
        stage=stage,
        job_status=IngestionJobStatus.RUNNING,
        idempotency_key=_key(source_id),
        selection=decision,
        metadata=metadata,
    )


def _fragment(source_id: UUID) -> EvidenceFragment:
    return EvidenceFragment(
        source_id=source_id,
        text="Durable evidence fragment retained before process restart.",
        locator=_locator().model_copy(update={"page": 3, "block_index": 8}),
        retention_reason="supports_gap",
        parser_name="mineru",
        parser_version="3.4.4",
        embedding_generation_id=GENERATION,
    )


@pytest.mark.asyncio
async def test_rejected_decision_never_opens_or_processes_remote_body(tmp_path: Path) -> None:
    pipeline, reader, mineru, assessor, lightrag, checkpoints, _ = _pipeline(tmp_path)
    source_id = uuid4()

    result = await _ingest(pipeline, _decision(source_id, selected=False))

    assert result.outcome == "skipped"
    assert result.checkpoint.stage is IngestionStage.SELECT
    assert result.checkpoint.job_status is IngestionJobStatus.SUCCEEDED
    assert reader.stat_calls == reader.open_calls == 0
    assert mineru.calls == assessor.calls == lightrag.calls == 0
    assert list(tmp_path.iterdir()) == []
    history = await checkpoints.history(source_id)
    assert history == []
    assert result.checkpoint.lifecycle_status is SourceLifecycleStatus.METADATA_INDEXED


@pytest.mark.asyncio
async def test_rejected_then_selected_same_source_can_parse_normally(tmp_path: Path) -> None:
    pipeline, reader, mineru, _, lightrag, checkpoints, _ = _pipeline(tmp_path)
    source_id = uuid4()

    await _ingest(pipeline, _decision(source_id, selected=False))
    result = await _ingest(pipeline, _decision(source_id, selected=True))

    assert result.outcome == "indexed"
    assert reader.stat_calls == reader.open_calls == mineru.calls == lightrag.calls == 1
    assert await checkpoints.load(_key(source_id)) is not None


@pytest.mark.asyncio
async def test_version_and_embedding_generation_have_independent_runs(tmp_path: Path) -> None:
    pipeline, _, mineru, _, lightrag, checkpoints, evidence = _pipeline(tmp_path)
    source_id = uuid4()
    decision = _decision(source_id)
    version_b = "source-version-v1:" + "b" * 64
    generation_b = "qwen3-embedding-8b:1024:v2"

    first = await _ingest(pipeline, decision)
    repeated = await _ingest(pipeline, decision)
    second_version = await _ingest(pipeline, decision, source_version_key=version_b)
    second_generation = await _ingest(
        pipeline,
        decision,
        embedding_generation_id=generation_b,
    )

    assert first.resumed is False
    assert repeated.resumed is True
    assert second_version.resumed is False
    assert second_generation.resumed is False
    assert mineru.calls == lightrag.calls == 3
    keys = {
        _key(source_id),
        _key(source_id, source_version_key=version_b),
        _key(source_id, embedding_generation_id=generation_b),
    }
    assert len(keys) == 3
    loaded = [await checkpoints.load(key) for key in keys]
    assert all(item is not None for item in loaded)
    for key in keys:
        assert len(await evidence.list_for_source(source_id, idempotency_key=key)) == 1


@pytest.mark.asyncio
async def test_selected_pipeline_persists_each_boundary_and_only_fragments(tmp_path: Path) -> None:
    pipeline, reader, mineru, assessor, lightrag, checkpoints, evidence = _pipeline(tmp_path)
    source_id = uuid4()
    decision = _decision(source_id)

    result = await _ingest(pipeline, decision)

    assert result.outcome == "indexed"
    assert result.evidence_count == 1
    assert result.index_outcome == "processed"
    assert result.checkpoint.lifecycle_status is SourceLifecycleStatus.EVIDENCE_RETAINED
    assert result.checkpoint.job_status is IngestionJobStatus.SUCCEEDED
    assert reader.stat_calls == reader.open_calls == mineru.calls == assessor.calls == 1
    assert lightrag.calls == 1
    assert len(lightrag.fragments) == 1
    assert isinstance(lightrag.fragments[0], EvidenceFragment)
    assert "complete" not in str(lightrag.fragments[0].metadata).casefold()
    assert list(tmp_path.iterdir()) == []

    history = await checkpoints.history(source_id)
    stages = [item.stage for item in history]
    assert stages[:5] == [
        IngestionStage.SELECT,
        IngestionStage.SPOOL,
        IngestionStage.PARSE,
        IngestionStage.RETAIN,
        IngestionStage.INDEX,
    ]
    key = _key(source_id)
    assert mineru.keys == [key]
    stored = await evidence.list_for_source(source_id, idempotency_key=key)
    assert stored == lightrag.fragments
    assert stored[0].fragment_id == lightrag.fragments[0].fragment_id


@pytest.mark.asyncio
async def test_no_value_parse_reaches_terminal_without_lightrag(tmp_path: Path) -> None:
    assessor = FakeAssessor(accepted=False)
    pipeline, _, mineru, _, lightrag, _, evidence = _pipeline(
        tmp_path,
        assessor=assessor,
    )
    source_id = uuid4()

    result = await _ingest(pipeline, _decision(source_id))

    assert result.outcome == "parsed_no_value"
    assert result.index_outcome == "no_evidence"
    assert result.evidence_count == 0
    assert result.checkpoint.stage is IngestionStage.INDEX
    assert result.checkpoint.lifecycle_status is SourceLifecycleStatus.PARSED_NO_VALUE
    assert mineru.calls == 1
    assert lightrag.calls == 0
    assert (
        await evidence.list_for_source(
            source_id,
            idempotency_key=_key(source_id),
        )
        == []
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_parser_failure_is_sanitized_classified_and_cleans_spool(tmp_path: Path) -> None:
    mineru = FakeMinerU()
    mineru.error = MinerUError(
        "provider leaked signed URL and credential",
        category="parse_failed",
        retryable=False,
    )
    pipeline, _, _, _, lightrag, checkpoints, _ = _pipeline(tmp_path, mineru=mineru)
    source_id = uuid4()

    with pytest.raises(IngestionPipelineError) as raised:
        await _ingest(pipeline, _decision(source_id))

    assert raised.value.disposition is FailureDisposition.PERMANENT
    assert raised.value.category == "mineru.parse_failed"
    assert "signed" not in str(raised.value)
    assert "credential" not in str(raised.value)
    checkpoint = await checkpoints.load(_key(source_id))
    assert checkpoint is not None
    assert checkpoint.job_status is IngestionJobStatus.FAILED_PERMANENT
    assert checkpoint.lifecycle_status is SourceLifecycleStatus.FAILED_PERMANENT
    assert checkpoint.last_error_category == "mineru.parse_failed"
    assert lightrag.calls == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cancellation_is_recorded_and_cleans_all_transient_files(tmp_path: Path) -> None:
    mineru = FakeMinerU()
    mineru.error = asyncio.CancelledError()
    pipeline, _, _, _, _, checkpoints, _ = _pipeline(tmp_path, mineru=mineru)
    source_id = uuid4()

    with pytest.raises(IngestionCancelledError) as raised:
        await _ingest(pipeline, _decision(source_id))

    assert raised.value.disposition is FailureDisposition.CANCELLED
    checkpoint = await checkpoints.load(_key(source_id))
    assert checkpoint is not None
    assert checkpoint.job_status is IngestionJobStatus.CANCELLED
    assert checkpoint.last_error_category == "cancelled"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_completed_run_is_terminal_and_never_repeats_parse_or_index(tmp_path: Path) -> None:
    checkpoints = InMemoryCheckpointRepository()
    evidence = InMemoryEvidenceRepository()
    first, _, first_mineru, _, first_lightrag, _, _ = _pipeline(
        tmp_path,
        checkpoints=checkpoints,
        evidence=evidence,
    )
    source_id = uuid4()
    decision = _decision(source_id)
    await _ingest(first, decision)

    second, reader, mineru, assessor, lightrag, _, _ = _pipeline(
        tmp_path,
        checkpoints=checkpoints,
        evidence=evidence,
    )
    result = await _ingest(second, decision)

    assert result.resumed is True
    assert result.outcome == "indexed"
    assert first_mineru.calls == first_lightrag.calls == 1
    assert reader.stat_calls == reader.open_calls == 0
    assert mineru.calls == assessor.calls == lightrag.calls == 0


@pytest.mark.asyncio
async def test_retain_checkpoint_resumes_from_durable_fragments_without_parse(
    tmp_path: Path,
) -> None:
    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    evidence = InMemoryEvidenceRepository()
    checkpoint = _resume_checkpoint(source_id, decision, stage=IngestionStage.RETAIN)
    fragment = _fragment(source_id)
    await checkpoints.save(checkpoint)
    await evidence.persist_many(
        source_id,
        [fragment],
        idempotency_key=checkpoint.idempotency_key,
    )
    lightrag = FakeLightRAG(outcome="idempotent_conflict")
    pipeline, reader, mineru, assessor, _, _, _ = _pipeline(
        tmp_path,
        lightrag=lightrag,
        checkpoints=checkpoints,
        evidence=evidence,
    )

    result = await _ingest(pipeline, decision)

    assert result.outcome == "indexed"
    assert result.index_outcome == "idempotent_conflict"
    assert result.resumed is True
    assert reader.stat_calls == reader.open_calls == mineru.calls == assessor.calls == 0
    assert lightrag.calls == 1
    assert lightrag.fragments == [fragment]


@pytest.mark.asyncio
async def test_index_completion_marker_finalizes_without_reindexing(tmp_path: Path) -> None:
    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    await checkpoints.save(
        _resume_checkpoint(
            source_id,
            decision,
            stage=IngestionStage.INDEX,
            index_completed=True,
        )
    )
    pipeline, reader, mineru, assessor, lightrag, _, _ = _pipeline(
        tmp_path,
        checkpoints=checkpoints,
    )

    result = await _ingest(pipeline, decision)

    assert result.checkpoint.job_status is IngestionJobStatus.SUCCEEDED
    assert result.checkpoint.lifecycle_status is SourceLifecycleStatus.EVIDENCE_RETAINED
    assert reader.stat_calls == reader.open_calls == mineru.calls == assessor.calls == 0
    assert lightrag.calls == 0


@pytest.mark.asyncio
async def test_retryable_index_error_resumes_index_only_and_hides_provider_detail(
    tmp_path: Path,
) -> None:
    checkpoints = InMemoryCheckpointRepository()
    evidence = InMemoryEvidenceRepository()
    failing_lightrag = FakeLightRAG()
    failing_lightrag.error = LightRAGRequestError(
        path="/documents/texts",
        status_code=503,
        detail="secret provider response",
    )
    first, _, first_mineru, _, _, _, _ = _pipeline(
        tmp_path,
        lightrag=failing_lightrag,
        checkpoints=checkpoints,
        evidence=evidence,
    )
    source_id = uuid4()
    decision = _decision(source_id)

    with pytest.raises(IngestionPipelineError) as raised:
        await _ingest(first, decision)
    assert raised.value.disposition is FailureDisposition.RETRYABLE
    assert raised.value.category == "lightrag.request"
    assert "secret" not in str(raised.value)

    retry_lightrag = FakeLightRAG(outcome="idempotent_conflict")
    second, reader, second_mineru, assessor, _, _, _ = _pipeline(
        tmp_path,
        lightrag=retry_lightrag,
        checkpoints=checkpoints,
        evidence=evidence,
    )
    result = await _ingest(second, decision)

    assert result.index_outcome == "idempotent_conflict"
    assert result.checkpoint.attempt == 1
    assert first_mineru.calls == 1
    assert reader.stat_calls == reader.open_calls == second_mineru.calls == assessor.calls == 0
    assert retry_lightrag.calls == 1


@pytest.mark.parametrize("bad_value", ["", "   "])
def test_idempotency_key_requires_version_and_generation(bad_value: str) -> None:
    with pytest.raises(ValueError, match="source version"):
        build_ingestion_idempotency_key(
            uuid4(),
            source_version_key=bad_value,
            embedding_generation_id=GENERATION,
        )


@pytest.mark.asyncio
async def test_checkpoint_read_and_write_failures_are_sanitized(tmp_path: Path) -> None:
    source_id = uuid4()
    for repository, category in (
        (FailingCheckpointRepository(fail_load=True), "checkpoint_read_failed"),
        (FailingCheckpointRepository(), "checkpoint_write_failed"),
    ):
        pipeline, *_ = _pipeline(
            tmp_path,
            checkpoints=repository,  # type: ignore[arg-type]
        )
        with pytest.raises(IngestionPipelineError) as raised:
            await _ingest(pipeline, _decision(source_id))
        assert raised.value.disposition is FailureDisposition.RETRYABLE
        assert raised.value.category == category
        assert "secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_rejected_source_bypasses_even_a_failing_checkpoint_repository(
    tmp_path: Path,
) -> None:
    pipeline, *_ = _pipeline(
        tmp_path,
        checkpoints=FailingCheckpointRepository(fail_load=True),  # type: ignore[arg-type]
    )

    result = await _ingest(pipeline, _decision(uuid4(), selected=False))

    assert result.outcome == "skipped"


@pytest.mark.asyncio
async def test_invalid_request_and_checkpoint_identity_fail_before_remote_access(
    tmp_path: Path,
) -> None:
    pipeline, reader, *_ = _pipeline(tmp_path)
    with pytest.raises(IngestionPipelineError) as invalid:
        await _ingest(
            pipeline,
            _decision(uuid4()),
            embedding_generation_id=" ",
        )
    assert invalid.value.category == "invalid_ingestion_request"

    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    mismatched = _resume_checkpoint(source_id, decision, stage=IngestionStage.RETAIN)
    mismatched.metadata["relative_path"] = "different/paper.pdf"
    await checkpoints.save(mismatched)
    resumed, resumed_reader, *_ = _pipeline(tmp_path / "mismatch", checkpoints=checkpoints)
    with pytest.raises(IngestionPipelineError) as identity:
        await _ingest(resumed, decision)
    assert identity.value.category == "checkpoint_identity_mismatch"
    assert reader.stat_calls == resumed_reader.stat_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_status", "lifecycle", "error_type"),
    [
        (
            IngestionJobStatus.FAILED_PERMANENT,
            SourceLifecycleStatus.FAILED_PERMANENT,
            IngestionPipelineError,
        ),
        (
            IngestionJobStatus.CANCELLED,
            SourceLifecycleStatus.PARSE_ELIGIBLE,
            IngestionCancelledError,
        ),
    ],
)
async def test_terminal_checkpoint_is_not_automatically_restarted(
    tmp_path: Path,
    job_status: IngestionJobStatus,
    lifecycle: SourceLifecycleStatus,
    error_type: type[BaseException],
) -> None:
    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    terminal = _resume_checkpoint(source_id, decision, stage=IngestionStage.INDEX).model_copy(
        update={
            "job_status": job_status,
            "lifecycle_status": lifecycle,
            "last_error_category": "terminal_fixture",
        }
    )
    await checkpoints.save(terminal)
    pipeline, reader, mineru, assessor, lightrag, _, _ = _pipeline(
        tmp_path,
        checkpoints=checkpoints,
    )

    with pytest.raises(error_type):
        await _ingest(pipeline, decision)

    assert reader.stat_calls == reader.open_calls == mineru.calls == assessor.calls == 0
    assert lightrag.calls == 0


@pytest.mark.asyncio
async def test_queued_retain_checkpoint_resumes_without_remote_parse(tmp_path: Path) -> None:
    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    evidence = InMemoryEvidenceRepository()
    queued = _resume_checkpoint(source_id, decision, stage=IngestionStage.RETAIN).model_copy(
        update={"job_status": IngestionJobStatus.QUEUED}
    )
    await checkpoints.save(queued)
    await evidence.persist_many(
        source_id, [_fragment(source_id)], idempotency_key=queued.idempotency_key
    )
    pipeline, reader, mineru, _, lightrag, _, _ = _pipeline(
        tmp_path,
        checkpoints=checkpoints,
        evidence=evidence,
    )

    result = await _ingest(pipeline, decision)

    assert result.outcome == "indexed"
    assert reader.stat_calls == reader.open_calls == mineru.calls == 0
    assert lightrag.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["invalid", "mismatch"])
async def test_remote_stat_validation_is_permanent_before_stream(
    tmp_path: Path,
    kind: str,
) -> None:
    reader = FakeReader()
    if kind == "invalid":
        reader.stat_override = RemoteStat(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="polymers/paper.pdf",
            is_dir=False,
            byte_size=0,
        )
        expected = "invalid_remote_source"
    else:
        reader.stat_override = RemoteStat(
            root_id="other_root",
            slice_id="literature",
            relative_path="polymers/paper.pdf",
            is_dir=False,
            byte_size=12,
        )
        expected = "remote_identity_mismatch"
    pipeline, _, mineru, _, lightrag, _, _ = _pipeline(tmp_path, reader=reader)

    with pytest.raises(IngestionPipelineError) as raised:
        await _ingest(pipeline, _decision(uuid4()))

    assert raised.value.disposition is FailureDisposition.PERMANENT
    assert raised.value.category == expected
    assert reader.open_calls == mineru.calls == lightrag.calls == 0


@pytest.mark.asyncio
async def test_durable_fragment_mismatch_and_failed_index_are_permanent(tmp_path: Path) -> None:
    source_id = uuid4()
    decision = _decision(source_id)
    checkpoints = InMemoryCheckpointRepository()
    checkpoint = _resume_checkpoint(source_id, decision, stage=IngestionStage.RETAIN)
    await checkpoints.save(checkpoint)
    pipeline, *_ = _pipeline(tmp_path, checkpoints=checkpoints)

    with pytest.raises(IngestionPipelineError) as missing:
        await _ingest(pipeline, decision)
    assert missing.value.category == "durable_evidence_mismatch"

    fresh, *_ = _pipeline(tmp_path / "failed-index", lightrag=FakeLightRAG("failed"))
    with pytest.raises(IngestionPipelineError) as failed:
        await _ingest(fresh, _decision(uuid4()))
    assert failed.value.category == "lightrag_document_failed"


@pytest.mark.parametrize(
    ("error", "disposition", "category"),
    [
        (SpoolCapacityError("full"), FailureDisposition.RETRYABLE, "spool.capacity"),
        (SpoolIntegrityError("changed"), FailureDisposition.RETRYABLE, "spool.integrity"),
        (SpoolError("spool"), FailureDisposition.RETRYABLE, "spool.failure"),
        (
            LightRAGRequestError(path="/documents/text", status_code=400, detail="bad"),
            FailureDisposition.PERMANENT,
            "lightrag.request",
        ),
        (LightRAGPollingTimeout("late"), FailureDisposition.RETRYABLE, "lightrag.poll_timeout"),
        (LightRAGProtocolError("bad"), FailureDisposition.PERMANENT, "lightrag.protocol"),
        (LightRAGError("down"), FailureDisposition.RETRYABLE, "lightrag.failure"),
        (ValueError("bad"), FailureDisposition.PERMANENT, "invalid_pipeline_data"),
        (OSError("io"), FailureDisposition.RETRYABLE, "dependency_io"),
        (RuntimeError("unknown"), FailureDisposition.RETRYABLE, "dependency_failure"),
    ],
)
def test_dependency_failure_classification(
    error: Exception,
    disposition: FailureDisposition,
    category: str,
) -> None:
    assert KnowledgeIngestionPipeline._classify(error) == (disposition, category)


@pytest.mark.asyncio
async def test_in_memory_repositories_enforce_identity_and_keep_first_metadata() -> None:
    source_id = uuid4()
    key = _key(source_id)
    checkpoints = InMemoryCheckpointRepository()
    checkpoint = _resume_checkpoint(source_id, _decision(source_id), stage=IngestionStage.RETAIN)
    await checkpoints.save(checkpoint)
    other_source_id = uuid4()
    conflicting = checkpoint.model_copy(
        update={
            "source_id": other_source_id,
            "selection": _decision(other_source_id),
        }
    )
    with pytest.raises(ValueError, match="source identity"):
        await checkpoints.save(conflicting)

    evidence = InMemoryEvidenceRepository()
    fragment = _fragment(source_id)
    await evidence.persist_many(source_id, [fragment], idempotency_key=key)
    retried = fragment.model_copy(update={"metadata": {"retry_batch": "second"}})
    await evidence.persist_many(source_id, [retried], idempotency_key=key)
    stored = await evidence.list_for_source(source_id, idempotency_key=key)
    assert stored[0].metadata == fragment.metadata
    with pytest.raises(ValueError, match="idempotency"):
        await evidence.persist_many(source_id, [], idempotency_key=" ")
    with pytest.raises(TypeError, match="EvidenceFragment"):
        await evidence.persist_many(source_id, [object()], idempotency_key=key)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="source_id"):
        await evidence.persist_many(uuid4(), [fragment], idempotency_key=key)
