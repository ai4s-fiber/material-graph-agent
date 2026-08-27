"""Fail-closed production assembly for staged knowledge-ingestion canaries.

The service owns no provider connection and implements no parsing or indexing.
It composes the existing metadata and selected-body pipelines through injected
interfaces while enforcing durable selection, operator approval, resource
admission, stage ordering, idempotency, and bounded batch size.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bindings import ProviderBindings
from .concurrency import AdmissionSnapshot, GlobalAdmissionController
from .ingestion import (
    CheckpointRepository,
    IngestionPipelineError,
    IngestionResult,
    KnowledgeIngestionPipeline,
    build_ingestion_idempotency_key,
)
from .manifest import (
    ManifestFormat,
    MetadataManifestIngestor,
    MetadataStreamError,
    MetadataStreamResult,
)
from .models import SelectionDecision, SourceLocator
from .policy import CorpusPolicy
from .processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    SourceLifecycleStatus,
)
from .remote_reader import normalize_identifier


_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_SAFE_APPROVAL_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_SOURCE_VERSION_KEY = re.compile(r"^source-version-v1:[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class CanaryStage(StrEnum):
    METADATA_ONLY = "metadata_only"
    SINGLE_PDF = "single_pdf"
    SMALL_BATCH = "small_batch"


class CanaryStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


_STAGE_RANK = {
    CanaryStage.METADATA_ONLY: 0,
    CanaryStage.SINGLE_PDF: 1,
    CanaryStage.SMALL_BATCH: 2,
}
_PREREQUISITE = {
    CanaryStage.SINGLE_PDF: CanaryStage.METADATA_ONLY,
    CanaryStage.SMALL_BATCH: CanaryStage.SINGLE_PDF,
}


class KnowledgeCanaryError(RuntimeError):
    """Credential-free service error containing only a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code if _SAFE_CODE.fullmatch(code) else "canary_dependency_failure"
        super().__init__(self.code)


class KnowledgeCanaryPolicy(BaseModel):
    """Operator-controlled rollout ceiling; metadata-only is the safe default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_enabled_stage: CanaryStage = CanaryStage.METADATA_ONLY
    max_small_batch_sources: int = Field(default=8, ge=2, le=64)


class MetadataCanaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=_SAFE_RUN_ID.pattern)
    root_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    slice_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    manifest_path: str = Field(min_length=1)
    manifest_format: ManifestFormat


class CanarySource(BaseModel):
    """Internal source identity used for selection and body ingestion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: SelectionDecision
    locator: SourceLocator
    slice_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_version_key: str = Field(pattern=_SOURCE_VERSION_KEY.pattern)

    @model_validator(mode="after")
    def validate_source_identity(self) -> "CanarySource":
        normalize_identifier(self.slice_id, field="slice_id")
        if self.decision.source_id is None:  # pragma: no cover - pydantic invariant
            raise ValueError("source decision is required")
        return self

    @property
    def source_id(self) -> UUID:
        return self.decision.source_id


class OperatorApproval(BaseModel):
    """Explicit, narrowly-scoped approval for exactly one body stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str = Field(pattern=_SAFE_APPROVAL_ID.pattern)
    run_id: str = Field(pattern=_SAFE_RUN_ID.pattern)
    stage: Literal[CanaryStage.SINGLE_PDF, CanaryStage.SMALL_BATCH]
    approved: Literal[True]
    source_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_sources(self) -> "OperatorApproval":
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("approval source_ids must be unique")
        return self


class SelectionReceipt(BaseModel):
    """Safe acknowledgement that a selected decision is durable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=512)
    persisted: bool
    code: Literal["selection_persisted", "selection_already_persisted"]


class CanaryStageRecord(BaseModel):
    """Durable, provider-free canary state used for retry and resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=_SAFE_RUN_ID.pattern)
    stage: CanaryStage
    status: CanaryStatus
    attempt: int = Field(ge=1)
    code: str = Field(pattern=_SAFE_CODE.pattern)
    request_fingerprint: str = Field(pattern=_FINGERPRINT.pattern)
    source_ids: tuple[UUID, ...] = ()
    approval_id: str | None = Field(default=None, pattern=_SAFE_APPROVAL_ID.pattern)
    attempted_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    metadata_records: int = Field(default=0, ge=0)
    resumed: bool = False

    @model_validator(mode="after")
    def validate_counts_and_stage(self) -> "CanaryStageRecord":
        if self.completed_count > self.attempted_count:
            raise ValueError("completed_count cannot exceed attempted_count")
        if self.stage is CanaryStage.METADATA_ONLY:
            if self.source_ids or self.approval_id is not None:
                raise ValueError("metadata canary cannot contain body approval state")
        elif self.approval_id is None:
            raise ValueError("body canary requires an approval_id")
        return self


class CanaryResult(BaseModel):
    """Public-safe result; never contains a remote path or provider detail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=_SAFE_RUN_ID.pattern)
    stage: CanaryStage
    status: CanaryStatus
    code: str = Field(pattern=_SAFE_CODE.pattern)
    attempt: int = Field(ge=0)
    source_ids: tuple[UUID, ...] = ()
    attempted_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    metadata_records: int = Field(default=0, ge=0)
    resumed: bool = False
    idempotent: bool = False
    next_stage_allowed: bool = False


class CanaryRunRepository(Protocol):
    """Durable stage repository; production implementations must claim atomically."""

    async def load(self, run_id: str, stage: CanaryStage) -> CanaryStageRecord | None: ...

    async def begin(
        self,
        *,
        run_id: str,
        stage: CanaryStage,
        request_fingerprint: str,
        source_ids: tuple[UUID, ...],
        approval_id: str | None,
    ) -> CanaryStageRecord: ...

    async def finish(self, record: CanaryStageRecord) -> None: ...


class AdmissionSnapshotProvider(Protocol):
    async def snapshot(self) -> AdmissionSnapshot: ...


class CancellationProbe(Protocol):
    def is_cancelled(self) -> bool: ...


class MetadataCanaryRunner(Protocol):
    async def ingest(
        self,
        *,
        root_id: str,
        slice_id: str,
        manifest_path: str,
        manifest_format: ManifestFormat,
    ) -> MetadataStreamResult: ...


class BodyCanaryRunner(Protocol):
    async def ingest(
        self,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        source_version_key: str,
        embedding_generation_id: str,
    ) -> IngestionResult: ...


class InMemoryCanaryRunRepository:
    """Atomic in-process fake for tests; inject a durable store in production."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, CanaryStage], CanaryStageRecord] = {}
        self._lock = asyncio.Lock()

    async def load(self, run_id: str, stage: CanaryStage) -> CanaryStageRecord | None:
        async with self._lock:
            record = self._records.get((run_id, stage))
            return None if record is None else record.model_copy(deep=True)

    async def begin(
        self,
        *,
        run_id: str,
        stage: CanaryStage,
        request_fingerprint: str,
        source_ids: tuple[UUID, ...],
        approval_id: str | None,
    ) -> CanaryStageRecord:
        async with self._lock:
            key = (run_id, stage)
            existing = self._records.get(key)
            if existing is not None and (
                existing.request_fingerprint != request_fingerprint
                or existing.source_ids != source_ids
                or existing.approval_id != approval_id
            ):
                raise KnowledgeCanaryError("canary_stage_identity_mismatch")
            if existing is not None and existing.status is CanaryStatus.RUNNING:
                raise KnowledgeCanaryError("canary_stage_running")
            attempt = 1 if existing is None else existing.attempt + 1
            record = CanaryStageRecord(
                run_id=run_id,
                stage=stage,
                status=CanaryStatus.RUNNING,
                attempt=attempt,
                code="canary_running",
                request_fingerprint=request_fingerprint,
                source_ids=source_ids,
                approval_id=approval_id,
                resumed=existing is not None,
            )
            self._records[key] = record
            return record.model_copy(deep=True)

    async def finish(self, record: CanaryStageRecord) -> None:
        if record.status is CanaryStatus.RUNNING:
            raise ValueError("cannot finish a running canary record")
        async with self._lock:
            key = (record.run_id, record.stage)
            existing = self._records.get(key)
            if existing is None or existing.attempt != record.attempt:
                raise KnowledgeCanaryError("canary_stage_claim_lost")
            if existing.status is not CanaryStatus.RUNNING:
                if existing == record:
                    return
                raise KnowledgeCanaryError("canary_stage_claim_lost")
            self._records[key] = record.model_copy(deep=True)


class KnowledgeCanaryService:
    """Coordinate metadata, single-source, and bounded-batch canaries."""

    def __init__(
        self,
        *,
        metadata: MetadataCanaryRunner | MetadataManifestIngestor,
        ingestion: BodyCanaryRunner | KnowledgeIngestionPipeline,
        checkpoints: CheckpointRepository,
        runs: CanaryRunRepository,
        admission: GlobalAdmissionController,
        admission_snapshots: AdmissionSnapshotProvider,
        corpus_policy: CorpusPolicy,
        bindings: ProviderBindings,
        policy: KnowledgeCanaryPolicy | None = None,
    ) -> None:
        self._metadata = metadata
        self._ingestion = ingestion
        self._checkpoints = checkpoints
        self._runs = runs
        self._admission = admission
        self._admission_snapshots = admission_snapshots
        self._corpus_policy = corpus_policy
        self._bindings = bindings
        self.policy = policy or KnowledgeCanaryPolicy()

    async def register_selection(self, source: CanarySource) -> SelectionReceipt:
        """Persist ``selected=true`` before any method may open source bytes."""

        if not source.decision.selected:
            raise KnowledgeCanaryError("selected_decision_required")
        self._require_known_root(source.locator.root_id)
        key = self._idempotency_key(source)
        expected_metadata = self._checkpoint_metadata(source)
        try:
            existing = await self._checkpoints.load(key)
        except Exception:
            raise KnowledgeCanaryError("checkpoint_read_failed") from None

        persisted = existing is None
        if existing is None:
            checkpoint = ProcessingCheckpoint(
                source_id=source.source_id,
                lifecycle_status=SourceLifecycleStatus.PARSE_ELIGIBLE,
                stage=IngestionStage.SELECT,
                job_status=IngestionJobStatus.QUEUED,
                idempotency_key=key,
                selection=source.decision,
                metadata=expected_metadata,
            )
            try:
                await self._checkpoints.save(checkpoint)
                existing = await self._checkpoints.load(key)
            except Exception:
                raise KnowledgeCanaryError("checkpoint_write_failed") from None

        self._validate_persisted_selection(existing, source, key, expected_metadata)
        return SelectionReceipt(
            source_id=source.source_id,
            idempotency_key=key,
            persisted=persisted,
            code="selection_persisted" if persisted else "selection_already_persisted",
        )

    async def run_metadata(self, request: MetadataCanaryRequest) -> CanaryResult:
        fingerprint = self._fingerprint(
            CanaryStage.METADATA_ONLY.value,
            request.root_id,
            request.slice_id,
            request.manifest_path,
            request.manifest_format,
        )
        cached = await self._cached_success(
            request.run_id,
            CanaryStage.METADATA_ONLY,
            request_fingerprint=fingerprint,
        )
        if cached is not None:
            return cached
        started = await self._begin_or_block(
            run_id=request.run_id,
            stage=CanaryStage.METADATA_ONLY,
            request_fingerprint=fingerprint,
            source_ids=(),
            approval_id=None,
        )
        if isinstance(started, CanaryResult):
            return started
        record = started
        try:
            self._require_known_root(request.root_id)
            if not await self._admitted(body=False):
                return await self._finish(record, CanaryStatus.BLOCKED, "metadata_admission_denied")
            result = await self._metadata.ingest(
                root_id=request.root_id,
                slice_id=request.slice_id,
                manifest_path=request.manifest_path,
                manifest_format=request.manifest_format,
            )
            if not isinstance(result, MetadataStreamResult):
                raise KnowledgeCanaryError("metadata_result_invalid")
            records_seen = result.records_seen
            metadata_records = result.records_created + result.records_updated
        except asyncio.CancelledError:
            await self._finish_cancelled(record)
            raise
        except Exception as error:
            return await self._finish(record, self._error_status(error), self._error_code(error))

        return await self._finish(
            record,
            CanaryStatus.SUCCEEDED,
            "metadata_canary_succeeded",
            attempted_count=records_seen,
            completed_count=records_seen,
            metadata_records=metadata_records,
        )

    async def run_single_pdf(
        self,
        *,
        run_id: str,
        source: CanarySource,
        approval: OperatorApproval,
        cancellation: CancellationProbe | None = None,
    ) -> CanaryResult:
        return await self._run_body_stage(
            run_id=run_id,
            stage=CanaryStage.SINGLE_PDF,
            sources=(source,),
            approval=approval,
            cancellation=cancellation,
        )

    async def run_small_batch(
        self,
        *,
        run_id: str,
        sources: Sequence[CanarySource],
        approval: OperatorApproval,
        cancellation: CancellationProbe | None = None,
    ) -> CanaryResult:
        candidates = tuple(sources)
        if len(candidates) < 2 or len(candidates) > self.policy.max_small_batch_sources:
            return self._ephemeral_result(
                run_id,
                CanaryStage.SMALL_BATCH,
                CanaryStatus.BLOCKED,
                "small_batch_size_invalid",
            )
        source_ids = tuple(source.source_id for source in candidates)
        if len(source_ids) != len(set(source_ids)):
            return self._ephemeral_result(
                run_id,
                CanaryStage.SMALL_BATCH,
                CanaryStatus.BLOCKED,
                "small_batch_duplicate_source",
            )
        return await self._run_body_stage(
            run_id=run_id,
            stage=CanaryStage.SMALL_BATCH,
            sources=candidates,
            approval=approval,
            cancellation=cancellation,
        )

    async def _run_body_stage(
        self,
        *,
        run_id: str,
        stage: CanaryStage,
        sources: tuple[CanarySource, ...],
        approval: OperatorApproval,
        cancellation: CancellationProbe | None,
    ) -> CanaryResult:
        if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise KnowledgeCanaryError("canary_run_id_invalid")
        source_ids = tuple(source.source_id for source in sources)
        fingerprint = self._fingerprint(
            stage.value,
            approval.approval_id,
            *(
                value
                for source in sources
                for value in (
                    source.source_id.hex,
                    source.locator.root_id,
                    source.locator.relative_path,
                    source.slice_id,
                    source.source_version_key,
                )
            ),
        )
        cached = await self._cached_success(
            run_id,
            stage,
            request_fingerprint=fingerprint,
        )
        if cached is not None:
            return cached
        started = await self._begin_or_block(
            run_id=run_id,
            stage=stage,
            request_fingerprint=fingerprint,
            source_ids=source_ids,
            approval_id=approval.approval_id,
        )
        if isinstance(started, CanaryResult):
            return started
        record = started
        completed = 0
        evidence_count = 0
        resumed = record.resumed
        try:
            self._require_stage_enabled(stage)
            await self._require_prerequisite(run_id, stage)
            self._validate_approval(approval, run_id, stage, source_ids)
            for source in sources:
                if cancellation is not None and cancellation.is_cancelled():
                    return await self._finish(
                        record,
                        CanaryStatus.CANCELLED,
                        "canary_cancelled",
                        attempted_count=len(sources),
                        completed_count=completed,
                        evidence_count=evidence_count,
                        resumed=resumed,
                    )
                if not source.locator.relative_path.casefold().endswith(".pdf"):
                    raise KnowledgeCanaryError("body_source_not_pdf")
                self._require_known_root(source.locator.root_id)
                if not await self._admitted(body=True):
                    return await self._finish(
                        record,
                        CanaryStatus.BLOCKED,
                        "body_admission_denied",
                        attempted_count=len(sources),
                        completed_count=completed,
                        evidence_count=evidence_count,
                        resumed=resumed,
                    )
                checkpoint = await self._load_persisted_selection(source)
                result = await self._ingestion.ingest(
                    decision=checkpoint.selection,  # type: ignore[arg-type]
                    source_locator=source.locator,
                    slice_id=source.slice_id,
                    source_version_key=source.source_version_key,
                    embedding_generation_id=self._bindings.embedding.generation_id,
                )
                completed += 1
                evidence_count += result.evidence_count
                resumed = resumed or result.resumed
        except asyncio.CancelledError:
            await self._finish_cancelled(
                record,
                attempted_count=len(sources),
                completed_count=completed,
                evidence_count=evidence_count,
                resumed=resumed,
            )
            raise
        except Exception as error:
            return await self._finish(
                record,
                self._error_status(error),
                self._error_code(error),
                attempted_count=len(sources),
                completed_count=completed,
                evidence_count=evidence_count,
                resumed=resumed,
            )
        return await self._finish(
            record,
            CanaryStatus.SUCCEEDED,
            f"{stage.value}_canary_succeeded",
            attempted_count=len(sources),
            completed_count=completed,
            evidence_count=evidence_count,
            resumed=resumed,
        )

    def _require_known_root(self, root_id: str) -> None:
        try:
            self._corpus_policy.source(root_id)
        except Exception:
            raise KnowledgeCanaryError("corpus_root_not_approved") from None

    def _require_stage_enabled(self, stage: CanaryStage) -> None:
        if _STAGE_RANK[stage] > _STAGE_RANK[self.policy.max_enabled_stage]:
            raise KnowledgeCanaryError("canary_stage_disabled")

    async def _require_prerequisite(self, run_id: str, stage: CanaryStage) -> None:
        prerequisite = _PREREQUISITE.get(stage)
        if prerequisite is None:
            return
        try:
            record = await self._runs.load(run_id, prerequisite)
        except Exception:
            raise KnowledgeCanaryError("canary_state_read_failed") from None
        if record is not None and not isinstance(record, CanaryStageRecord):
            raise KnowledgeCanaryError("canary_state_invalid")
        if record is None or record.status is not CanaryStatus.SUCCEEDED:
            raise KnowledgeCanaryError("canary_prerequisite_not_satisfied")

    @staticmethod
    def _validate_approval(
        approval: OperatorApproval,
        run_id: str,
        stage: CanaryStage,
        source_ids: tuple[UUID, ...],
    ) -> None:
        if (
            approval.run_id != run_id
            or approval.stage is not stage
            or set(approval.source_ids) != set(source_ids)
        ):
            raise KnowledgeCanaryError("operator_approval_invalid")

    async def _admitted(self, *, body: bool) -> bool:
        try:
            snapshot = await self._admission_snapshots.snapshot()
            decision = self._admission.decide(snapshot)
        except Exception:
            raise KnowledgeCanaryError("admission_probe_failed") from None
        return decision.allow_body if body else decision.allow_metadata

    def _idempotency_key(self, source: CanarySource) -> str:
        return build_ingestion_idempotency_key(
            source.source_id,
            source_version_key=source.source_version_key,
            embedding_generation_id=self._bindings.embedding.generation_id,
        )

    def _checkpoint_metadata(self, source: CanarySource) -> dict[str, str]:
        return {
            "root_id": source.locator.root_id,
            "slice_id": source.slice_id,
            "relative_path": source.locator.relative_path,
            "embedding_generation_id": self._bindings.embedding.generation_id,
            "source_version_fingerprint": sha256(
                source.source_version_key.encode("utf-8")
            ).hexdigest(),
        }

    @staticmethod
    def _validate_persisted_selection(
        checkpoint: ProcessingCheckpoint | None,
        source: CanarySource,
        key: str,
        expected_metadata: dict[str, str],
    ) -> None:
        if (
            not isinstance(checkpoint, ProcessingCheckpoint)
            or checkpoint.source_id != source.source_id
            or checkpoint.idempotency_key != key
            or checkpoint.selection is None
            or not checkpoint.selection.selected
            or checkpoint.selection != source.decision
            or any(
                checkpoint.metadata.get(name) != value for name, value in expected_metadata.items()
            )
        ):
            raise KnowledgeCanaryError("persisted_selection_mismatch")

    async def _load_persisted_selection(self, source: CanarySource) -> ProcessingCheckpoint:
        key = self._idempotency_key(source)
        try:
            checkpoint = await self._checkpoints.load(key)
        except Exception:
            raise KnowledgeCanaryError("checkpoint_read_failed") from None
        if checkpoint is None:
            raise KnowledgeCanaryError("persisted_selection_required")
        self._validate_persisted_selection(
            checkpoint,
            source,
            key,
            self._checkpoint_metadata(source),
        )
        return checkpoint

    async def _cached_success(
        self,
        run_id: str,
        stage: CanaryStage,
        *,
        request_fingerprint: str,
    ) -> CanaryResult | None:
        try:
            record = await self._runs.load(run_id, stage)
        except Exception:
            return self._ephemeral_result(
                run_id,
                stage,
                CanaryStatus.BLOCKED,
                "canary_state_read_failed",
            )
        if record is not None and (
            not isinstance(record, CanaryStageRecord)
            or record.run_id != run_id
            or record.stage is not stage
        ):
            return self._ephemeral_result(
                run_id,
                stage,
                CanaryStatus.FAILED,
                "canary_state_invalid",
            )
        if record is None or record.status is not CanaryStatus.SUCCEEDED:
            return None
        if record.request_fingerprint != request_fingerprint:
            return self._ephemeral_result(
                run_id,
                stage,
                CanaryStatus.FAILED,
                "canary_stage_identity_mismatch",
            )
        return self._to_result(record, idempotent=True)

    async def _begin_or_block(
        self,
        *,
        run_id: str,
        stage: CanaryStage,
        request_fingerprint: str,
        source_ids: tuple[UUID, ...],
        approval_id: str | None,
    ) -> CanaryStageRecord | CanaryResult:
        try:
            record = await self._runs.begin(
                run_id=run_id,
                stage=stage,
                request_fingerprint=request_fingerprint,
                source_ids=source_ids,
                approval_id=approval_id,
            )
        except KnowledgeCanaryError as error:
            return self._ephemeral_result(run_id, stage, CanaryStatus.BLOCKED, error.code)
        except Exception:
            return self._ephemeral_result(
                run_id,
                stage,
                CanaryStatus.BLOCKED,
                "canary_state_write_failed",
            )
        if (
            not isinstance(record, CanaryStageRecord)
            or record.run_id != run_id
            or record.stage is not stage
            or record.request_fingerprint != request_fingerprint
        ):
            return self._ephemeral_result(
                run_id,
                stage,
                CanaryStatus.BLOCKED,
                "canary_state_invalid",
            )
        return record

    async def _finish(
        self,
        record: CanaryStageRecord,
        status: CanaryStatus,
        code: str,
        *,
        attempted_count: int = 0,
        completed_count: int = 0,
        evidence_count: int = 0,
        metadata_records: int = 0,
        resumed: bool | None = None,
    ) -> CanaryResult:
        payload = record.model_dump(mode="python")
        payload.update(
            status=status,
            code=code if _SAFE_CODE.fullmatch(code) else "canary_dependency_failure",
            attempted_count=attempted_count,
            completed_count=completed_count,
            evidence_count=evidence_count,
            metadata_records=metadata_records,
            resumed=record.resumed if resumed is None else resumed,
        )
        finished = CanaryStageRecord.model_validate(payload)
        try:
            await self._runs.finish(finished)
        except Exception:
            return self._ephemeral_result(
                record.run_id,
                record.stage,
                CanaryStatus.BLOCKED,
                "canary_state_write_failed",
            )
        return self._to_result(finished)

    async def _finish_cancelled(
        self,
        record: CanaryStageRecord,
        *,
        attempted_count: int = 0,
        completed_count: int = 0,
        evidence_count: int = 0,
        resumed: bool | None = None,
    ) -> None:
        try:
            await self._finish(
                record,
                CanaryStatus.CANCELLED,
                "canary_cancelled",
                attempted_count=attempted_count,
                completed_count=completed_count,
                evidence_count=evidence_count,
                resumed=resumed,
            )
        except Exception:  # pragma: no cover - cancellation must remain primary
            pass

    def _to_result(self, record: CanaryStageRecord, *, idempotent: bool = False) -> CanaryResult:
        return CanaryResult(
            run_id=record.run_id,
            stage=record.stage,
            status=record.status,
            code=record.code,
            attempt=record.attempt,
            source_ids=record.source_ids,
            attempted_count=record.attempted_count,
            completed_count=record.completed_count,
            evidence_count=record.evidence_count,
            metadata_records=record.metadata_records,
            resumed=record.resumed,
            idempotent=idempotent,
            next_stage_allowed=(
                record.status is CanaryStatus.SUCCEEDED
                and record.stage is not CanaryStage.SMALL_BATCH
                and _STAGE_RANK[record.stage] < _STAGE_RANK[self.policy.max_enabled_stage]
            ),
        )

    @staticmethod
    def _ephemeral_result(
        run_id: str,
        stage: CanaryStage,
        status: CanaryStatus,
        code: str,
    ) -> CanaryResult:
        safe_code = code if _SAFE_CODE.fullmatch(code) else "canary_dependency_failure"
        return CanaryResult(
            run_id=run_id,
            stage=stage,
            status=status,
            code=safe_code,
            attempt=0,
        )

    @staticmethod
    def _error_code(error: Exception) -> str:
        if isinstance(error, KnowledgeCanaryError):
            return error.code
        if isinstance(error, MetadataStreamError):
            return error.code if _SAFE_CODE.fullmatch(error.code) else "metadata_canary_failed"
        if isinstance(error, IngestionPipelineError):
            return error.category if _SAFE_CODE.fullmatch(error.category) else "body_canary_failed"
        return "canary_dependency_failure"

    @staticmethod
    def _error_status(error: Exception) -> CanaryStatus:
        if isinstance(error, KnowledgeCanaryError) and error.code in {
            "body_source_not_pdf",
            "canary_prerequisite_not_satisfied",
            "canary_stage_disabled",
            "corpus_root_not_approved",
            "operator_approval_invalid",
            "persisted_selection_required",
        }:
            return CanaryStatus.BLOCKED
        return CanaryStatus.FAILED

    @staticmethod
    def _fingerprint(*values: str) -> str:
        return sha256("\x00".join(values).encode("utf-8")).hexdigest()


__all__ = [
    "AdmissionSnapshotProvider",
    "BodyCanaryRunner",
    "CancellationProbe",
    "CanaryResult",
    "CanaryRunRepository",
    "CanarySource",
    "CanaryStage",
    "CanaryStageRecord",
    "CanaryStatus",
    "InMemoryCanaryRunRepository",
    "KnowledgeCanaryError",
    "KnowledgeCanaryPolicy",
    "KnowledgeCanaryService",
    "MetadataCanaryRequest",
    "MetadataCanaryRunner",
    "OperatorApproval",
    "SelectionReceipt",
]
