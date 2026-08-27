"""Durable, typed background jobs for knowledge ingestion.

The queue stores only bounded request metadata and safe summaries. Source
bytes, provider responses and credentials never cross this boundary.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
import json
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .age_writer import (
    KnowledgeGraphApprovalError,
    KnowledgeGraphPersistenceError,
    UnsafeKnowledgeGraphPayload,
)
from .extraction import EvidenceFactExtractionPipeline, FactExtractionError
from .facts import FactBatch, GlobalKnowledgeGraphWriter, KnowledgeGraphConflict
from .ingestion import EvidenceRepository, build_ingestion_idempotency_key
from .service import (
    CanarySource,
    CanaryStatus,
    KnowledgeCanaryService,
    MetadataCanaryRequest,
    OperatorApproval,
)


_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_SAFE_WORKER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_JOB_KEY = re.compile(r"^knowledge-job:v1:[0-9a-f]{64}$")
_RETRYABLE_CANARY_CODES = frozenset(
    {
        "admission_probe_failed",
        "body_admission_denied",
        "canary_state_read_failed",
        "canary_state_write_failed",
        "checkpoint_read_failed",
        "checkpoint_write_failed",
        "metadata_admission_denied",
        "metadata.catalog.upsert_failed",
        "metadata.cursor.load_failed",
        "metadata.cursor.save_failed",
        "metadata.provider.stat_failed",
        "metadata.provider.stream_failed",
    }
)


class KnowledgeJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED_PERMANENT = "failed_permanent"
    CANCELLED = "cancelled"


class MetadataOnlyJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    job_type: Literal["metadata_only"] = "metadata_only"
    request: MetadataCanaryRequest


class SingleDocumentJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    job_type: Literal["single_document"] = "single_document"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    source: CanarySource
    approval: OperatorApproval

    @model_validator(mode="after")
    def validate_scope(self) -> "SingleDocumentJobPayload":
        if not self.source.decision.selected:
            raise ValueError("single_document_requires_selected_source")
        if not self.source.locator.relative_path.casefold().endswith(".pdf"):
            raise ValueError("single_document_requires_pdf")
        if (
            self.approval.run_id != self.run_id
            or self.approval.stage.value != "single_pdf"
            or self.approval.source_ids != (self.source.source_id,)
        ):
            raise ValueError("single_document_approval_mismatch")
        return self


class GraphWriteJobPayload(BaseModel):
    """Content-bound outbox message emitted only by an approved fact review."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    job_type: Literal["graph_write"] = "graph_write"
    batch_id: str = Field(pattern=r"^fact-batch:v1:[0-9a-f]{64}$")
    fact_batch_idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    projection_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_digest: str = Field(pattern=r"^graph-approval:v1:[0-9a-f]{64}$")


KnowledgeJobPayload = Annotated[
    MetadataOnlyJobPayload | SingleDocumentJobPayload | GraphWriteJobPayload,
    Field(discriminator="job_type"),
]
_PAYLOAD_ADAPTER = TypeAdapter(KnowledgeJobPayload)


class KnowledgeJobResult(BaseModel):
    """Public-safe completion summary; extracted facts remain pending review."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    job_type: Literal["metadata_only", "single_document", "graph_write"]
    run_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    code: str = Field(pattern=_SAFE_CODE.pattern)
    metadata_records: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    pending_fact_batch_ids: tuple[str, ...] = ()
    review_status: Literal["not_applicable", "pending_review"] = "not_applicable"
    batch_id: str | None = Field(default=None, pattern=r"^fact-batch:v1:[0-9a-f]{64}$")
    fact_batch_idempotency_key: str | None = Field(
        default=None,
        pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$",
    )
    graph_write_status: Literal["written", "already_present"] | None = None
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_review_boundary(self) -> "KnowledgeJobResult":
        if self.job_type == "metadata_only":
            if (
                self.run_id is None
                or self.pending_fact_batch_ids
                or self.review_status != "not_applicable"
                or self.batch_id is not None
                or self.fact_batch_idempotency_key is not None
                or self.graph_write_status is not None
                or self.node_count
                or self.edge_count
            ):
                raise ValueError("metadata_job_cannot_contain_facts")
        elif self.job_type == "single_document":
            if (
                self.run_id is None
                or self.review_status != "pending_review"
                or self.batch_id is not None
                or self.fact_batch_idempotency_key is not None
                or self.graph_write_status is not None
                or self.node_count
                or self.edge_count
            ):
                raise ValueError("single_document_facts_must_remain_pending")
        elif (
            self.run_id is not None
            or self.metadata_records
            or self.evidence_count
            or self.pending_fact_batch_ids
            or self.review_status != "not_applicable"
            or self.batch_id is None
            or self.fact_batch_idempotency_key is None
            or self.graph_write_status is None
        ):
            raise ValueError("graph_write_result_invalid")
        return self


class KnowledgeJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    job_id: UUID
    idempotency_key: str = Field(pattern=_JOB_KEY.pattern)
    payload: KnowledgeJobPayload
    status: KnowledgeJobStatus
    attempt: int = Field(ge=0, le=8)
    max_attempts: int = Field(ge=1, le=8)
    available_at: datetime
    lease_owner: str | None = Field(default=None, pattern=_SAFE_WORKER.pattern)
    lease_token: int = Field(default=0, ge=0)
    lease_until: datetime | None = None
    result: KnowledgeJobResult | None = None
    last_error_code: str | None = Field(default=None, pattern=_SAFE_CODE.pattern)

    @field_validator("available_at", "lease_until")
    @classmethod
    def require_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("knowledge_job_datetime_must_be_aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_state(self) -> "KnowledgeJobRecord":
        if self.attempt > self.max_attempts:
            raise ValueError("knowledge_job_attempt_exceeds_maximum")
        if self.status is KnowledgeJobStatus.QUEUED and self.attempt != 0:
            raise ValueError("queued_knowledge_job_cannot_have_attempts")
        if self.status is KnowledgeJobStatus.RETRY_WAIT and self.attempt >= self.max_attempts:
            raise ValueError("retrying_knowledge_job_requires_remaining_attempt")
        leased = self.lease_owner is not None and self.lease_until is not None
        if (self.lease_owner is None) != (self.lease_until is None):
            raise ValueError("knowledge_job_lease_incomplete")
        if self.status is KnowledgeJobStatus.RUNNING:
            if not leased or self.lease_token < 1 or self.attempt < 1:
                raise ValueError("running_knowledge_job_requires_lease")
        elif leased:
            raise ValueError("non_running_knowledge_job_cannot_hold_lease")
        if self.status is KnowledgeJobStatus.SUCCEEDED:
            if self.result is None or self.last_error_code is not None:
                raise ValueError("succeeded_knowledge_job_requires_result")
        elif self.result is not None:
            raise ValueError("unfinished_knowledge_job_cannot_store_result")
        if self.status in {
            KnowledgeJobStatus.RETRY_WAIT,
            KnowledgeJobStatus.FAILED_PERMANENT,
        }:
            if self.last_error_code is None:
                raise ValueError("failed_knowledge_job_requires_error")
        elif self.last_error_code is not None:
            raise ValueError("active_knowledge_job_cannot_store_error")
        return self


class KnowledgeJobLease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record: KnowledgeJobRecord
    worker_id: str = Field(pattern=_SAFE_WORKER.pattern)
    lease_token: int = Field(ge=1)
    lease_until: datetime

    @model_validator(mode="after")
    def bind_record(self) -> "KnowledgeJobLease":
        if (
            self.record.status is not KnowledgeJobStatus.RUNNING
            or self.record.lease_owner != self.worker_id
            or self.record.lease_token != self.lease_token
            or self.record.lease_until != self.lease_until
        ):
            raise ValueError("knowledge_job_lease_mismatch")
        return self


class KnowledgeJobExecutionError(RuntimeError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        self.code = code if _SAFE_CODE.fullmatch(code) else "worker.dependency_failure"
        self.retryable = retryable
        super().__init__(self.code)


class KnowledgeJobLeaseLost(RuntimeError):
    def __init__(self) -> None:
        super().__init__("worker.lease_lost")


class KnowledgeJobRepository(Protocol):
    async def enqueue(
        self, payload: KnowledgeJobPayload, *, max_attempts: int = 4
    ) -> KnowledgeJobRecord: ...

    async def claim(self, worker_id: str, *, lease_seconds: int) -> KnowledgeJobLease | None: ...

    async def renew(self, lease: KnowledgeJobLease, *, lease_seconds: int) -> KnowledgeJobLease: ...

    async def complete(
        self, lease: KnowledgeJobLease, result: KnowledgeJobResult
    ) -> KnowledgeJobRecord: ...

    async def fail(
        self,
        lease: KnowledgeJobLease,
        *,
        code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> KnowledgeJobRecord: ...

    async def get(self, job_id: UUID) -> KnowledgeJobRecord: ...


def canonical_job_payload(payload: KnowledgeJobPayload) -> str:
    validated = _PAYLOAD_ADAPTER.validate_python(payload)
    return json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_knowledge_job_idempotency_key(payload: KnowledgeJobPayload) -> str:
    return "knowledge-job:v1:" + sha256(canonical_job_payload(payload).encode("utf-8")).hexdigest()


class InMemoryKnowledgeJobRepository:
    """Production-equivalent in-memory repository for worker unit tests."""

    def __init__(self, *, clock=lambda: datetime.now(UTC)) -> None:
        self._clock = clock
        self._items: dict[UUID, KnowledgeJobRecord] = {}
        self._by_key: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self, payload: KnowledgeJobPayload, *, max_attempts: int = 4
    ) -> KnowledgeJobRecord:
        validated = _PAYLOAD_ADAPTER.validate_python(payload)
        if isinstance(validated, GraphWriteJobPayload):
            raise ValueError("graph_write_jobs_require_review_outbox")
        key = build_knowledge_job_idempotency_key(validated)
        async with self._lock:
            existing_id = self._by_key.get(key)
            if existing_id is not None:
                return self._items[existing_id].model_copy(deep=True)
            record = KnowledgeJobRecord(
                job_id=uuid4(),
                idempotency_key=key,
                payload=validated,
                status=KnowledgeJobStatus.QUEUED,
                attempt=0,
                max_attempts=max_attempts,
                available_at=self._clock(),
            )
            self._items[record.job_id] = record
            self._by_key[key] = record.job_id
            return record.model_copy(deep=True)

    async def get(self, job_id: UUID) -> KnowledgeJobRecord:
        async with self._lock:
            return self._items[job_id].model_copy(deep=True)

    async def claim(self, worker_id: str, *, lease_seconds: int) -> KnowledgeJobLease | None:
        if _SAFE_WORKER.fullmatch(worker_id) is None or lease_seconds < 1:
            raise ValueError("invalid worker lease request")
        async with self._lock:
            now = self._clock()
            for job_id in sorted(self._items, key=lambda value: value.hex):
                existing = self._items[job_id]
                expired = (
                    existing.status is KnowledgeJobStatus.RUNNING
                    and existing.lease_until is not None
                    and existing.lease_until <= now
                )
                eligible = existing.status in {
                    KnowledgeJobStatus.QUEUED,
                    KnowledgeJobStatus.RETRY_WAIT,
                }
                if not ((eligible and existing.available_at <= now) or expired):
                    continue
                if existing.attempt >= existing.max_attempts:
                    if expired:
                        self._items[job_id] = existing.model_copy(
                            update={
                                "status": KnowledgeJobStatus.FAILED_PERMANENT,
                                "lease_owner": None,
                                "lease_until": None,
                                "last_error_code": "worker.lease_expired",
                            }
                        )
                    continue
                until = now + timedelta(seconds=lease_seconds)
                running = existing.model_copy(
                    update={
                        "status": KnowledgeJobStatus.RUNNING,
                        "attempt": existing.attempt + 1,
                        "lease_owner": worker_id,
                        "lease_token": existing.lease_token + 1,
                        "lease_until": until,
                        "last_error_code": None,
                    }
                )
                self._items[job_id] = running
                return KnowledgeJobLease(
                    record=running,
                    worker_id=worker_id,
                    lease_token=running.lease_token,
                    lease_until=until,
                )
        return None

    def _leased(self, lease: KnowledgeJobLease) -> KnowledgeJobRecord:
        existing = self._items.get(lease.record.job_id)
        if (
            existing is None
            or existing.status is not KnowledgeJobStatus.RUNNING
            or existing.lease_owner != lease.worker_id
            or existing.lease_token != lease.lease_token
            or existing.lease_until is None
            or existing.lease_until <= self._clock()
        ):
            raise KnowledgeJobLeaseLost()
        return existing

    async def renew(self, lease: KnowledgeJobLease, *, lease_seconds: int) -> KnowledgeJobLease:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        async with self._lock:
            existing = self._leased(lease)
            until = self._clock() + timedelta(seconds=lease_seconds)
            renewed = existing.model_copy(update={"lease_until": until})
            self._items[existing.job_id] = renewed
            return KnowledgeJobLease(
                record=renewed,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                lease_until=until,
            )

    async def complete(
        self, lease: KnowledgeJobLease, result: KnowledgeJobResult
    ) -> KnowledgeJobRecord:
        validated = KnowledgeJobResult.model_validate(result.model_dump(mode="python"))
        async with self._lock:
            existing = self._items.get(lease.record.job_id)
            if existing is not None and existing.status is KnowledgeJobStatus.SUCCEEDED:
                if existing.result == validated and existing.lease_token == lease.lease_token:
                    return existing.model_copy(deep=True)
                raise KnowledgeJobLeaseLost()
            existing = self._leased(lease)
            completed = existing.model_copy(
                update={
                    "status": KnowledgeJobStatus.SUCCEEDED,
                    "lease_owner": None,
                    "lease_until": None,
                    "result": validated,
                }
            )
            self._items[existing.job_id] = completed
            return completed.model_copy(deep=True)

    async def fail(
        self,
        lease: KnowledgeJobLease,
        *,
        code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> KnowledgeJobRecord:
        safe_code = code if _SAFE_CODE.fullmatch(code) else "worker.dependency_failure"
        if retry_delay_seconds < 0:
            raise ValueError("retry delay cannot be negative")
        async with self._lock:
            existing = self._items.get(lease.record.job_id)
            if (
                existing is not None
                and existing.lease_token == lease.lease_token
                and existing.status
                in {KnowledgeJobStatus.RETRY_WAIT, KnowledgeJobStatus.FAILED_PERMANENT}
                and existing.last_error_code == safe_code
            ):
                return existing.model_copy(deep=True)
            existing = self._leased(lease)
            will_retry = retryable and existing.attempt < existing.max_attempts
            failed = existing.model_copy(
                update={
                    "status": (
                        KnowledgeJobStatus.RETRY_WAIT
                        if will_retry
                        else KnowledgeJobStatus.FAILED_PERMANENT
                    ),
                    "available_at": self._clock()
                    + timedelta(seconds=retry_delay_seconds if will_retry else 0),
                    "lease_owner": None,
                    "lease_until": None,
                    "last_error_code": safe_code,
                }
            )
            self._items[existing.job_id] = failed
            return failed.model_copy(deep=True)


class KnowledgeJobExecutor:
    """Execute canary-backed jobs and leave extracted facts pending review."""

    def __init__(
        self,
        *,
        canaries: KnowledgeCanaryService,
        evidence: EvidenceRepository,
        fact_extraction: EvidenceFactExtractionPipeline,
        embedding_generation_id: str,
    ) -> None:
        if not embedding_generation_id.strip():
            raise ValueError("embedding_generation_id is required")
        self._canaries = canaries
        self._evidence = evidence
        self._fact_extraction = fact_extraction
        self._embedding_generation_id = embedding_generation_id

    async def execute(self, payload: KnowledgeJobPayload) -> KnowledgeJobResult:
        validated = _PAYLOAD_ADAPTER.validate_python(payload)
        if isinstance(validated, MetadataOnlyJobPayload):
            try:
                result = await self._canaries.run_metadata(validated.request)
                self._require_success(result.status, result.code)
            except KnowledgeJobExecutionError:
                raise
            except Exception:
                raise KnowledgeJobExecutionError(
                    "worker.dependency_failure",
                    retryable=True,
                ) from None
            return KnowledgeJobResult(
                job_type="metadata_only",
                run_id=validated.request.run_id,
                code=result.code,
                metadata_records=result.metadata_records,
            )

        try:
            await self._canaries.register_selection(validated.source)
            result = await self._canaries.run_single_pdf(
                run_id=validated.run_id,
                source=validated.source,
                approval=validated.approval,
            )
            self._require_success(result.status, result.code)
            key = build_ingestion_idempotency_key(
                validated.source.source_id,
                source_version_key=validated.source.source_version_key,
                embedding_generation_id=self._embedding_generation_id,
            )
            fragments = await self._evidence.list_for_source(
                validated.source.source_id,
                idempotency_key=key,
            )
            batch_ids: list[str] = []
            for fragment in fragments:
                extracted = await self._fact_extraction.extract(fragment)
                if extracted.review_status != "pending_review" or extracted.batch.batch_id is None:
                    raise KnowledgeJobExecutionError(
                        "worker.fact_review_boundary_invalid",
                        retryable=False,
                    )
                batch_ids.append(extracted.batch.batch_id)
        except KnowledgeJobExecutionError:
            raise
        except FactExtractionError as error:
            raise KnowledgeJobExecutionError(error.code, retryable=error.retryable) from None
        except Exception:
            raise KnowledgeJobExecutionError("worker.dependency_failure", retryable=True) from None

        return KnowledgeJobResult(
            job_type="single_document",
            run_id=validated.run_id,
            code=result.code,
            evidence_count=result.evidence_count,
            pending_fact_batch_ids=tuple(sorted(set(batch_ids))),
            review_status="pending_review",
        )

    @staticmethod
    def _require_success(status: CanaryStatus, code: str) -> None:
        if status is CanaryStatus.SUCCEEDED:
            return
        raise KnowledgeJobExecutionError(code, retryable=code in _RETRYABLE_CANARY_CODES)


class ApprovedFactBatchError(RuntimeError):
    """Stable loader failure raised before a graph writer can see a batch."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        self.code = code if _SAFE_CODE.fullmatch(code) else "graph_write.review_invalid"
        self.retryable = retryable
        super().__init__(self.code)


class ApprovedFactBatchRepository(Protocol):
    async def load_approved_batch(self, payload: GraphWriteJobPayload) -> FactBatch: ...


class GraphWriteJobExecutor:
    """Run one independently leased, approval-gated global graph write."""

    def __init__(
        self,
        *,
        approved_batches: ApprovedFactBatchRepository,
        graph_writer: GlobalKnowledgeGraphWriter,
    ) -> None:
        self._approved_batches = approved_batches
        self._graph_writer = graph_writer

    async def execute(self, payload: GraphWriteJobPayload) -> KnowledgeJobResult:
        validated = GraphWriteJobPayload.model_validate(payload.model_dump(mode="python"))
        try:
            batch = await self._approved_batches.load_approved_batch(validated)
            written = await self._graph_writer.write_batch(batch)
        except ApprovedFactBatchError as error:
            raise KnowledgeJobExecutionError(error.code, retryable=error.retryable) from None
        except KnowledgeGraphApprovalError as error:
            raise KnowledgeJobExecutionError(
                error.code,
                retryable=error.code == "knowledge_graph_approval_unavailable",
            ) from None
        except KnowledgeGraphPersistenceError:
            raise KnowledgeJobExecutionError(
                "graph_write.persistence_failed",
                retryable=True,
            ) from None
        except KnowledgeGraphConflict:
            raise KnowledgeJobExecutionError(
                "graph_write.conflict",
                retryable=False,
            ) from None
        except UnsafeKnowledgeGraphPayload:
            raise KnowledgeJobExecutionError(
                "graph_write.unsafe_payload",
                retryable=False,
            ) from None
        except Exception:
            raise KnowledgeJobExecutionError(
                "graph_write.dependency_failure",
                retryable=True,
            ) from None

        if (
            written.batch_id != validated.batch_id
            or written.idempotency_key != validated.fact_batch_idempotency_key
        ):
            raise KnowledgeJobExecutionError(
                "graph_write.result_mismatch",
                retryable=False,
            )
        return KnowledgeJobResult(
            job_type="graph_write",
            code="graph_write_completed",
            batch_id=written.batch_id,
            fact_batch_idempotency_key=written.idempotency_key,
            graph_write_status=written.status,
            node_count=written.node_count,
            edge_count=written.edge_count,
        )


class RoutedKnowledgeJobExecutor:
    """Dispatch ingestion and approved graph writes without merging their boundaries."""

    def __init__(
        self,
        *,
        ingestion: KnowledgeJobExecutor,
        graph_write: GraphWriteJobExecutor,
    ) -> None:
        self._ingestion = ingestion
        self._graph_write = graph_write

    async def execute(self, payload: KnowledgeJobPayload) -> KnowledgeJobResult:
        validated = _PAYLOAD_ADAPTER.validate_python(payload)
        if isinstance(validated, GraphWriteJobPayload):
            return await self._graph_write.execute(validated)
        return await self._ingestion.execute(validated)


__all__ = [
    "ApprovedFactBatchError",
    "ApprovedFactBatchRepository",
    "GraphWriteJobExecutor",
    "GraphWriteJobPayload",
    "InMemoryKnowledgeJobRepository",
    "KnowledgeJobExecutionError",
    "KnowledgeJobExecutor",
    "KnowledgeJobLease",
    "KnowledgeJobLeaseLost",
    "KnowledgeJobPayload",
    "KnowledgeJobRecord",
    "KnowledgeJobRepository",
    "KnowledgeJobResult",
    "KnowledgeJobStatus",
    "MetadataOnlyJobPayload",
    "RoutedKnowledgeJobExecutor",
    "SingleDocumentJobPayload",
    "build_knowledge_job_idempotency_key",
    "canonical_job_payload",
]
