"""Human-reviewed fact batches and transactional graph-write outbox.

The PostgreSQL repository is the sole production enqueue boundary for
``graph_write`` jobs.  It restores a completed extraction checkpoint, binds a
terminal decision to the exact AGE-safe projection, and commits the review,
outbox row, and audit event in one transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import re
from typing import Any, Literal, Protocol, runtime_checkable
import unicodedata
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .age_writer import (
    GraphWriteApproval,
    GraphWriteApprovalRequest,
    build_graph_write_approval_request,
)
from .facts import FactBatch, _canonical_json
from .jobs import (
    ApprovedFactBatchError,
    GraphWriteJobPayload,
    build_knowledge_job_idempotency_key,
    canonical_job_payload,
)
from .postgres import AsyncConnectionPool
from .postgres_pipeline_state import (
    PipelineStatePersistenceError,
    restore_redacted_fact_batch,
)


FactReviewDecision = Literal["approve", "reject"]
FactReviewStatus = Literal["pending", "approved", "rejected"]
GraphWriteAuditEvent = Literal[
    "review_approved",
    "review_rejected",
    "claimed",
    "succeeded",
    "retry_scheduled",
    "failed_permanent",
]

_BATCH_ID = re.compile(r"^fact-batch:v1:[0-9a-f]{64}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_OUTBOX_NAMESPACE = UUID("1f9fcae4-f09a-5f7b-a955-daa8564218be")
_MAX_JOB_PAYLOAD_BYTES = 256 * 1024


class FactReviewNotFound(LookupError):
    def __init__(self) -> None:
        super().__init__("fact_review_batch_not_found")


class FactReviewConflict(RuntimeError):
    def __init__(self) -> None:
        super().__init__("fact_review_conflict")


class FactReviewPersistenceError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("fact_review_persistence_failed")


def _clean_text(value: str, *, maximum: int) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if (
        not normalized
        or len(normalized) > maximum
        or any(unicodedata.category(character) == "Cc" for character in normalized)
    ):
        raise ValueError("invalid_fact_review_text")
    return normalized


class FactReviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    decision: FactReviewDecision
    reviewer: str = Field(min_length=1, max_length=200)
    comment: str = Field(default="", max_length=2000)

    @field_validator("reviewer")
    @classmethod
    def normalize_reviewer(cls, value: str) -> str:
        return _clean_text(value, maximum=200)

    @field_validator("comment")
    @classmethod
    def normalize_comment(cls, value: str) -> str:
        if not value.strip():
            return ""
        return _clean_text(value, maximum=2000)


class FactReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_id: str = Field(pattern=_BATCH_ID.pattern)
    fact_batch_idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    projection_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: FactReviewStatus
    job_id: UUID | None = None
    approval_digest: str | None = Field(
        default=None,
        pattern=r"^graph-approval:v1:[0-9a-f]{64}$",
    )
    reviewer_generation_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    audit_generation_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approval_expires_at: datetime | None = None
    reviewed_at: datetime | None = None

    @field_validator("approval_expires_at", "reviewed_at")
    @classmethod
    def require_aware_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fact_review_time_must_be_aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "FactReviewRecord":
        receipt_fields = (
            self.approval_digest,
            self.reviewer_generation_digest,
            self.audit_generation_digest,
            self.approval_expires_at,
            self.reviewed_at,
        )
        if self.status == "pending":
            if self.job_id is not None or any(value is not None for value in receipt_fields):
                raise ValueError("pending_fact_review_cannot_have_terminal_state")
        elif any(value is None for value in receipt_fields):
            raise ValueError("terminal_fact_review_requires_receipt")
        elif self.status == "approved" and self.job_id is None:
            raise ValueError("approved_fact_review_requires_job")
        elif self.status == "rejected" and self.job_id is not None:
            raise ValueError("rejected_fact_review_cannot_have_job")
        return self


@runtime_checkable
class FactReviewRepository(Protocol):
    async def get_review(self, batch_id: str) -> FactReviewRecord: ...

    async def decide(
        self,
        batch_id: str,
        command: FactReviewCommand,
    ) -> FactReviewRecord: ...


_SELECT_BATCH = """
SELECT fragment_id, batch
FROM knowledge_fact_extraction_checkpoints
WHERE status = 'completed' AND batch ->> 'batch_id' = %s
"""
_SELECT_BATCH_FOR_UPDATE = _SELECT_BATCH + " FOR UPDATE"
_SELECT_REVIEW = """
SELECT batch_id, fact_batch_idempotency_key, projection_digest, status,
       job_id, approval_digest, reviewer_generation_digest,
       audit_generation_digest, approval_expires_at, reviewed_at
FROM knowledge_fact_reviews
WHERE batch_id = %s
"""
_SELECT_REVIEW_FOR_UPDATE = _SELECT_REVIEW + " FOR UPDATE"
_INSERT_JOB = """
INSERT INTO knowledge_worker_jobs(
    job_id, idempotency_key, job_type, payload, status, attempt,
    max_attempts, available_at, lease_token
)
VALUES (%s, %s, 'graph_write', %s::jsonb, 'queued', 0, %s, now(), 0)
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING job_id
"""
_SELECT_JOB_BY_KEY = """
SELECT job_id, job_type, payload
FROM knowledge_worker_jobs
WHERE idempotency_key = %s
"""
_INSERT_REVIEW = """
INSERT INTO knowledge_fact_reviews(
    batch_id, fact_batch_idempotency_key, projection_digest, status,
    job_id, approval_digest, reviewer_generation_digest,
    audit_generation_digest, approval_expires_at, reviewed_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING batch_id, fact_batch_idempotency_key, projection_digest, status,
          job_id, approval_digest, reviewer_generation_digest,
          audit_generation_digest, approval_expires_at, reviewed_at
"""
_INSERT_AUDIT = """
INSERT INTO knowledge_graph_write_audit(
    event_key, batch_id, job_id, event_type, attempt, lease_token, code,
    approval_digest, reviewer_generation_digest, audit_generation_digest
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (event_key) DO NOTHING
"""


def build_graph_write_audit_key(
    *,
    batch_id: str,
    event_type: GraphWriteAuditEvent,
    job_id: UUID | None,
    attempt: int,
    lease_token: int,
    code: str,
) -> str:
    if _BATCH_ID.fullmatch(batch_id) is None or _SAFE_CODE.fullmatch(code) is None:
        raise ValueError("invalid_graph_write_audit_identity")
    digest = sha256(
        _canonical_json(
            {
                "batch_id": batch_id,
                "event_type": event_type,
                "job_id": None if job_id is None else str(job_id),
                "attempt": attempt,
                "lease_token": lease_token,
                "code": code,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"graph-write-audit:v1:{digest}"


def _row(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactReviewPersistenceError()
    return value


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise FactReviewPersistenceError() from None
    if not isinstance(value, Mapping):
        raise FactReviewPersistenceError()
    return {str(key): item for key, item in value.items()}


def _batch_from_row(value: object) -> FactBatch:
    row = _row(value)
    try:
        fragment_id = UUID(str(row.get("fragment_id")))
        return restore_redacted_fact_batch(row.get("batch"), fragment_id)
    except (PipelineStatePersistenceError, ValidationError, TypeError, ValueError):
        raise FactReviewPersistenceError() from None


def _record_from_row(value: object) -> FactReviewRecord:
    row = _row(value)
    try:
        return FactReviewRecord(
            batch_id=row.get("batch_id"),
            fact_batch_idempotency_key=row.get("fact_batch_idempotency_key"),
            projection_digest=row.get("projection_digest"),
            status=row.get("status"),
            job_id=row.get("job_id"),
            approval_digest=row.get("approval_digest"),
            reviewer_generation_digest=row.get("reviewer_generation_digest"),
            audit_generation_digest=row.get("audit_generation_digest"),
            approval_expires_at=row.get("approval_expires_at"),
            reviewed_at=row.get("reviewed_at"),
        )
    except (ValidationError, TypeError, ValueError):
        raise FactReviewPersistenceError() from None


def _pending(batch: FactBatch) -> FactReviewRecord:
    request = build_graph_write_approval_request(batch)
    return FactReviewRecord(
        batch_id=request.batch_id,
        fact_batch_idempotency_key=request.idempotency_key,
        projection_digest=request.projection_digest,
        status="pending",
    )


def _review_digests(batch_id: str, command: FactReviewCommand) -> tuple[str, str]:
    reviewer = sha256(command.reviewer.casefold().encode("utf-8")).hexdigest()
    audit = sha256(
        _canonical_json(
            {
                "batch_id": batch_id,
                "decision": command.decision,
                "reviewer_generation_digest": reviewer,
                "comment": command.comment,
            }
        ).encode("utf-8")
    ).hexdigest()
    return reviewer, audit


class PostgresFactReviewRepository:
    """Immutable human decisions plus an atomic durable graph-write outbox."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        clock: Callable[[], datetime] | None = None,
        approval_ttl: timedelta = timedelta(days=30),
        graph_write_max_attempts: int = 4,
    ) -> None:
        if approval_ttl <= timedelta(0) or approval_ttl > timedelta(days=365):
            raise ValueError("approval_ttl_out_of_range")
        if isinstance(graph_write_max_attempts, bool) or not 1 <= graph_write_max_attempts <= 8:
            raise ValueError("graph_write_max_attempts_out_of_range")
        self._pool = pool
        self._clock = clock or (lambda: datetime.now(UTC))
        self._approval_ttl = approval_ttl
        self._graph_write_max_attempts = graph_write_max_attempts

    async def get_review(self, batch_id: str) -> FactReviewRecord:
        if _BATCH_ID.fullmatch(batch_id) is None:
            raise ValueError("invalid_fact_batch_id")
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_SELECT_BATCH, (batch_id,))
                    raw_batch = await cursor.fetchone()
                    if raw_batch is None:
                        raise FactReviewNotFound()
                    batch = _batch_from_row(raw_batch)
                    cursor = await connection.execute(_SELECT_REVIEW, (batch_id,))
                    raw_review = await cursor.fetchone()
                    if raw_review is None:
                        return _pending(batch)
                    record = _record_from_row(raw_review)
                    self._validate_record_binding(record, batch)
                    return record
        except (FactReviewNotFound, FactReviewPersistenceError):
            raise
        except Exception:
            raise FactReviewPersistenceError() from None

    async def decide(
        self,
        batch_id: str,
        command: FactReviewCommand,
    ) -> FactReviewRecord:
        if _BATCH_ID.fullmatch(batch_id) is None:
            raise ValueError("invalid_fact_batch_id")
        decision = FactReviewCommand.model_validate(command.model_dump(mode="python"))
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_SELECT_BATCH_FOR_UPDATE, (batch_id,))
                    raw_batch = await cursor.fetchone()
                    if raw_batch is None:
                        raise FactReviewNotFound()
                    batch = _batch_from_row(raw_batch)
                    request = build_graph_write_approval_request(batch)
                    cursor = await connection.execute(_SELECT_REVIEW_FOR_UPDATE, (batch_id,))
                    existing_raw = await cursor.fetchone()
                    if existing_raw is not None:
                        existing = _record_from_row(existing_raw)
                        self._validate_record_binding(existing, batch)
                        expected = "approved" if decision.decision == "approve" else "rejected"
                        if existing.status != expected:
                            raise FactReviewConflict()
                        return existing

                    now = self._clock()
                    if now.tzinfo is None or now.utcoffset() is None:
                        raise FactReviewPersistenceError()
                    now = now.astimezone(UTC)
                    reviewer_digest, audit_digest = _review_digests(batch_id, decision)
                    approval = GraphWriteApproval(
                        batch_id=request.batch_id,
                        idempotency_key=request.idempotency_key,
                        projection_digest=request.projection_digest,
                        approved=decision.decision == "approve",
                        reviewer_generation_digest=reviewer_digest,
                        audit_generation_digest=audit_digest,
                        expires_at=now + self._approval_ttl,
                    )
                    job_id: UUID | None = None
                    if approval.approved:
                        payload = GraphWriteJobPayload(
                            batch_id=request.batch_id,
                            fact_batch_idempotency_key=request.idempotency_key,
                            projection_digest=request.projection_digest,
                            approval_digest=str(approval.approval_digest),
                        )
                        job_id = await self._insert_outbox(connection, payload)

                    status: Literal["approved", "rejected"] = (
                        "approved" if approval.approved else "rejected"
                    )
                    cursor = await connection.execute(
                        _INSERT_REVIEW,
                        (
                            request.batch_id,
                            request.idempotency_key,
                            request.projection_digest,
                            status,
                            job_id,
                            approval.approval_digest,
                            reviewer_digest,
                            audit_digest,
                            approval.expires_at,
                            now,
                        ),
                    )
                    inserted = await cursor.fetchone()
                    if inserted is None:
                        raise FactReviewPersistenceError()
                    record = _record_from_row(inserted)
                    expected_record = FactReviewRecord(
                        batch_id=request.batch_id,
                        fact_batch_idempotency_key=request.idempotency_key,
                        projection_digest=request.projection_digest,
                        status=status,
                        job_id=job_id,
                        approval_digest=approval.approval_digest,
                        reviewer_generation_digest=reviewer_digest,
                        audit_generation_digest=audit_digest,
                        approval_expires_at=approval.expires_at,
                        reviewed_at=now,
                    )
                    if record != expected_record:
                        raise FactReviewPersistenceError()
                    await self._insert_review_audit(connection, record)
                    return record
        except (FactReviewConflict, FactReviewNotFound, FactReviewPersistenceError):
            raise
        except Exception:
            raise FactReviewPersistenceError() from None

    async def get_approval(
        self,
        request: GraphWriteApprovalRequest,
    ) -> GraphWriteApproval | None:
        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(_SELECT_REVIEW, (request.batch_id,))
                raw = await cursor.fetchone()
                if raw is None:
                    return None
                record = _record_from_row(raw)
                if (
                    record.fact_batch_idempotency_key != request.idempotency_key
                    or record.projection_digest != request.projection_digest
                    or record.status == "pending"
                    or record.approval_digest is None
                    or record.reviewer_generation_digest is None
                    or record.audit_generation_digest is None
                    or record.approval_expires_at is None
                ):
                    return None
                return GraphWriteApproval(
                    batch_id=record.batch_id,
                    idempotency_key=record.fact_batch_idempotency_key,
                    projection_digest=record.projection_digest,
                    approved=record.status == "approved",
                    reviewer_generation_digest=record.reviewer_generation_digest,
                    audit_generation_digest=record.audit_generation_digest,
                    expires_at=record.approval_expires_at,
                    approval_digest=record.approval_digest,
                )
        except FactReviewPersistenceError:
            raise
        except Exception:
            raise FactReviewPersistenceError() from None

    async def load_approved_batch(self, payload: GraphWriteJobPayload) -> FactBatch:
        validated = GraphWriteJobPayload.model_validate(payload.model_dump(mode="python"))
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _SELECT_BATCH_FOR_UPDATE,
                        (validated.batch_id,),
                    )
                    raw_batch = await cursor.fetchone()
                    if raw_batch is None:
                        raise ApprovedFactBatchError(
                            "graph_write.batch_missing",
                            retryable=False,
                        )
                    batch = _batch_from_row(raw_batch)
                    cursor = await connection.execute(
                        _SELECT_REVIEW_FOR_UPDATE,
                        (validated.batch_id,),
                    )
                    raw_review = await cursor.fetchone()
                    if raw_review is None:
                        raise ApprovedFactBatchError(
                            "graph_write.review_pending",
                            retryable=False,
                        )
                    review = _record_from_row(raw_review)
                    request = build_graph_write_approval_request(batch)
                    if review.status == "rejected":
                        raise ApprovedFactBatchError(
                            "graph_write.review_rejected",
                            retryable=False,
                        )
                    if review.status != "approved":
                        raise ApprovedFactBatchError(
                            "graph_write.review_pending",
                            retryable=False,
                        )
                    if (
                        request.batch_id != validated.batch_id
                        or request.idempotency_key != validated.fact_batch_idempotency_key
                        or request.projection_digest != validated.projection_digest
                        or review.fact_batch_idempotency_key != validated.fact_batch_idempotency_key
                        or review.projection_digest != validated.projection_digest
                        or review.approval_digest != validated.approval_digest
                    ):
                        raise ApprovedFactBatchError(
                            "graph_write.review_mismatch",
                            retryable=False,
                        )
                    return batch
        except ApprovedFactBatchError:
            raise
        except (FactReviewPersistenceError, PipelineStatePersistenceError):
            raise ApprovedFactBatchError(
                "graph_write.review_unavailable",
                retryable=True,
            ) from None
        except Exception:
            raise ApprovedFactBatchError(
                "graph_write.review_unavailable",
                retryable=True,
            ) from None

    async def _insert_outbox(self, connection: Any, payload: GraphWriteJobPayload) -> UUID:
        key = build_knowledge_job_idempotency_key(payload)
        encoded = canonical_job_payload(payload)
        if len(encoded.encode("utf-8")) > _MAX_JOB_PAYLOAD_BYTES:
            raise FactReviewPersistenceError()
        job_id = uuid5(_OUTBOX_NAMESPACE, key)
        cursor = await connection.execute(
            _INSERT_JOB,
            (job_id, key, encoded, self._graph_write_max_attempts),
        )
        inserted = await cursor.fetchone()
        if inserted is not None:
            return UUID(str(_row(inserted).get("job_id")))
        cursor = await connection.execute(_SELECT_JOB_BY_KEY, (key,))
        raw = await cursor.fetchone()
        if raw is None:
            raise FactReviewPersistenceError()
        row = _row(raw)
        if row.get("job_type") != "graph_write" or _json_object(
            row.get("payload")
        ) != payload.model_dump(mode="json"):
            raise FactReviewPersistenceError()
        return UUID(str(row.get("job_id")))

    async def _insert_review_audit(self, connection: Any, record: FactReviewRecord) -> None:
        event: GraphWriteAuditEvent = (
            "review_approved" if record.status == "approved" else "review_rejected"
        )
        code = record.status
        await connection.execute(
            _INSERT_AUDIT,
            (
                build_graph_write_audit_key(
                    batch_id=record.batch_id,
                    event_type=event,
                    job_id=record.job_id,
                    attempt=0,
                    lease_token=0,
                    code=code,
                ),
                record.batch_id,
                record.job_id,
                event,
                0,
                0,
                code,
                record.approval_digest,
                record.reviewer_generation_digest,
                record.audit_generation_digest,
            ),
        )

    @staticmethod
    def _validate_record_binding(record: FactReviewRecord, batch: FactBatch) -> None:
        request = build_graph_write_approval_request(batch)
        if (
            record.batch_id != request.batch_id
            or record.fact_batch_idempotency_key != request.idempotency_key
            or record.projection_digest != request.projection_digest
        ):
            raise FactReviewPersistenceError()


__all__ = [
    "FactReviewCommand",
    "FactReviewConflict",
    "FactReviewDecision",
    "FactReviewNotFound",
    "FactReviewPersistenceError",
    "FactReviewRecord",
    "FactReviewRepository",
    "FactReviewStatus",
    "GraphWriteAuditEvent",
    "PostgresFactReviewRepository",
    "build_graph_write_audit_key",
]
