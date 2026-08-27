"""PostgreSQL-backed durable queue for knowledge ingestion workers.

Claims use ``FOR UPDATE SKIP LOCKED`` so multiple worker processes can share
one queue. Every state-changing lease operation is fenced by the immutable
``job_id + lease_owner + lease_token`` tuple and an unexpired running lease.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any
from uuid import UUID, uuid4

from pydantic import TypeAdapter

from .jobs import (
    GraphWriteJobPayload,
    KnowledgeJobLease,
    KnowledgeJobLeaseLost,
    KnowledgeJobPayload,
    KnowledgeJobRecord,
    KnowledgeJobResult,
    KnowledgeJobStatus,
    build_knowledge_job_idempotency_key,
)
from .postgres import AsyncConnectionPool
from .reviewed_graph import GraphWriteAuditEvent, build_graph_write_audit_key


_SAFE_WORKER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SAFE_CODE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_PAYLOAD_ADAPTER = TypeAdapter(KnowledgeJobPayload)
_MAX_PAYLOAD_BYTES = 256 * 1024
_MAX_RESULT_BYTES = 64 * 1024

_RETURNING = """
RETURNING job_id, idempotency_key, job_type, payload, status, attempt,
          max_attempts, available_at, lease_owner, lease_token, lease_until,
          result, last_error_code
"""

_CLAIM_RETURNING = """
RETURNING jobs.job_id, jobs.idempotency_key, jobs.job_type, jobs.payload,
          jobs.status, jobs.attempt, jobs.max_attempts, jobs.available_at,
          jobs.lease_owner, jobs.lease_token, jobs.lease_until, jobs.result,
          jobs.last_error_code
"""

_INSERT_JOB = f"""
INSERT INTO knowledge_worker_jobs(
    job_id, idempotency_key, job_type, payload, status, attempt,
    max_attempts, available_at, lease_token
) VALUES (%s, %s, %s, %s::jsonb, 'queued', 0, %s, now(), 0)
ON CONFLICT (idempotency_key) DO NOTHING
{_RETURNING}
"""

_SELECT_BY_KEY = """
SELECT job_id, idempotency_key, job_type, payload, status, attempt,
       max_attempts, available_at, lease_owner, lease_token, lease_until,
       result, last_error_code
FROM knowledge_worker_jobs
WHERE idempotency_key = %s
"""

_SELECT_BY_ID = """
SELECT job_id, idempotency_key, job_type, payload, status, attempt,
       max_attempts, available_at, lease_owner, lease_token, lease_until,
       result, last_error_code
FROM knowledge_worker_jobs
WHERE job_id = %s
"""

_CLAIM_JOB = f"""
WITH expired_exhausted AS (
    UPDATE knowledge_worker_jobs
       SET status = 'failed_permanent',
           lease_owner = NULL,
           lease_until = NULL,
           last_error_code = 'worker.lease_expired',
           completed_at = now(),
           updated_at = now()
     WHERE status = 'running'
       AND lease_until <= now()
       AND attempt >= max_attempts
    RETURNING job_id
), candidate AS (
    SELECT job_id
      FROM knowledge_worker_jobs
     WHERE attempt < max_attempts
       AND (
           (status IN ('queued', 'retry_wait') AND available_at <= now())
           OR (status = 'running' AND lease_until <= now())
       )
     ORDER BY
       CASE WHEN status = 'running' THEN lease_until ELSE available_at END,
       created_at,
       job_id
     FOR UPDATE SKIP LOCKED
     LIMIT 1
)
UPDATE knowledge_worker_jobs AS jobs
   SET status = 'running',
       attempt = jobs.attempt + 1,
       lease_owner = %s,
       lease_token = jobs.lease_token + 1,
       lease_until = now() + (%s * INTERVAL '1 second'),
       result = NULL,
       last_error_code = NULL,
       completed_at = NULL,
       updated_at = now()
 FROM candidate
 WHERE jobs.job_id = candidate.job_id
{_CLAIM_RETURNING}
"""

_RENEW_JOB = f"""
UPDATE knowledge_worker_jobs
   SET lease_until = now() + (%s * INTERVAL '1 second'),
       updated_at = now()
 WHERE job_id = %s
   AND status = 'running'
   AND lease_owner = %s
   AND lease_token = %s
   AND lease_until > now()
{_RETURNING}
"""

_COMPLETE_JOB = f"""
UPDATE knowledge_worker_jobs
   SET status = 'succeeded',
       result = %s::jsonb,
       lease_owner = NULL,
       lease_until = NULL,
       last_error_code = NULL,
       completed_at = now(),
       updated_at = now()
 WHERE job_id = %s
   AND status = 'running'
   AND lease_owner = %s
   AND lease_token = %s
   AND lease_until > now()
{_RETURNING}
"""

_FAIL_JOB = f"""
UPDATE knowledge_worker_jobs
   SET status = %s,
       available_at = now() + (%s * INTERVAL '1 second'),
       lease_owner = NULL,
       lease_until = NULL,
       result = NULL,
       last_error_code = %s,
       completed_at = CASE WHEN %s = 'failed_permanent' THEN now() ELSE NULL END,
       updated_at = now()
 WHERE job_id = %s
   AND status = 'running'
   AND lease_owner = %s
   AND lease_token = %s
   AND lease_until > now()
{_RETURNING}
"""

_INSERT_GRAPH_WRITE_AUDIT = """
INSERT INTO knowledge_graph_write_audit(
    event_key, batch_id, job_id, event_type, attempt, lease_token, code,
    approval_digest, reviewer_generation_digest, audit_generation_digest
)
SELECT %s, review.batch_id, %s, %s, %s, %s, %s,
       review.approval_digest, review.reviewer_generation_digest,
       review.audit_generation_digest
FROM knowledge_fact_reviews AS review
WHERE review.batch_id = %s
  AND review.job_id = %s
  AND review.status = 'approved'
ON CONFLICT (event_key) DO NOTHING
RETURNING audit_id
"""


class KnowledgeJobPersistenceError(RuntimeError):
    """Credential-free durable queue failure."""

    def __init__(self) -> None:
        super().__init__("knowledge_job_persistence_failed")


def _bounded_json(value: Mapping[str, Any], *, maximum: int) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError("knowledge_job_payload_too_large")
    return encoded


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("knowledge_job_json_object_required")
    return {str(key): item for key, item in value.items()}


def _record_from_row(row: Mapping[str, Any]) -> KnowledgeJobRecord:
    payload = _PAYLOAD_ADAPTER.validate_python(_json_object(row["payload"]))
    if payload.job_type != str(row["job_type"]):
        raise ValueError("knowledge_job_type_mismatch")
    result_value = row.get("result")
    result = None
    if result_value is not None:
        result = KnowledgeJobResult.model_validate(_json_object(result_value))
    return KnowledgeJobRecord(
        job_id=UUID(str(row["job_id"])),
        idempotency_key=str(row["idempotency_key"]),
        payload=payload,
        status=KnowledgeJobStatus(str(row["status"])),
        attempt=int(row["attempt"]),
        max_attempts=int(row["max_attempts"]),
        available_at=row["available_at"],
        lease_owner=None if row.get("lease_owner") is None else str(row["lease_owner"]),
        lease_token=int(row["lease_token"]),
        lease_until=row.get("lease_until"),
        result=result,
        last_error_code=(
            None if row.get("last_error_code") is None else str(row["last_error_code"])
        ),
    )


def _validate_worker(worker_id: str, lease_seconds: int) -> None:
    if _SAFE_WORKER.fullmatch(worker_id) is None:
        raise ValueError("invalid worker id")
    if isinstance(lease_seconds, bool) or lease_seconds < 1:
        raise ValueError("lease_seconds must be positive")


async def _audit_graph_write_transition(
    connection: Any,
    record: KnowledgeJobRecord,
    *,
    event_type: GraphWriteAuditEvent,
    code: str,
) -> None:
    payload = record.payload
    if not isinstance(payload, GraphWriteJobPayload):
        return
    event_key = build_graph_write_audit_key(
        batch_id=payload.batch_id,
        event_type=event_type,
        job_id=record.job_id,
        attempt=record.attempt,
        lease_token=record.lease_token,
        code=code,
    )
    cursor = await connection.execute(
        _INSERT_GRAPH_WRITE_AUDIT,
        (
            event_key,
            record.job_id,
            event_type,
            record.attempt,
            record.lease_token,
            code,
            payload.batch_id,
            record.job_id,
        ),
    )
    inserted = await cursor.fetchone()
    if inserted is None:
        # A replay is valid only when the deterministic audit row already exists.
        cursor = await connection.execute(
            "SELECT event_key FROM knowledge_graph_write_audit WHERE event_key = %s",
            (event_key,),
        )
        if await cursor.fetchone() is None:
            raise KnowledgeJobPersistenceError()


class PostgresKnowledgeJobRepository:
    """Durable knowledge job repository with lease fencing and retries."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def enqueue(
        self,
        payload: KnowledgeJobPayload,
        *,
        max_attempts: int = 4,
    ) -> KnowledgeJobRecord:
        validated = _PAYLOAD_ADAPTER.validate_python(payload)
        if isinstance(validated, GraphWriteJobPayload):
            raise ValueError("graph_write_jobs_require_review_outbox")
        if isinstance(max_attempts, bool) or not 1 <= max_attempts <= 8:
            raise ValueError("max_attempts must be between 1 and 8")
        key = build_knowledge_job_idempotency_key(validated)
        encoded = _bounded_json(validated.model_dump(mode="json"), maximum=_MAX_PAYLOAD_BYTES)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _INSERT_JOB,
                        (uuid4(), key, validated.job_type, encoded, max_attempts),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        cursor = await connection.execute(_SELECT_BY_KEY, (key,))
                        row = await cursor.fetchone()
                    if row is None:
                        raise KnowledgeJobPersistenceError()
                    return _record_from_row(row)
        except KnowledgeJobPersistenceError:
            raise
        except Exception:
            raise KnowledgeJobPersistenceError() from None

    async def get(self, job_id: UUID) -> KnowledgeJobRecord:
        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(_SELECT_BY_ID, (job_id,))
                row = await cursor.fetchone()
                if row is None:
                    raise KeyError(job_id)
                return _record_from_row(row)
        except KeyError:
            raise
        except Exception:
            raise KnowledgeJobPersistenceError() from None

    async def claim(self, worker_id: str, *, lease_seconds: int) -> KnowledgeJobLease | None:
        _validate_worker(worker_id, lease_seconds)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_CLAIM_JOB, (worker_id, lease_seconds))
                    row = await cursor.fetchone()
                    if row is None:
                        return None
                    record = _record_from_row(row)
                    await _audit_graph_write_transition(
                        connection,
                        record,
                        event_type="claimed",
                        code="claimed",
                    )
                    return KnowledgeJobLease(
                        record=record,
                        worker_id=worker_id,
                        lease_token=record.lease_token,
                        lease_until=record.lease_until,
                    )
        except Exception as error:
            if isinstance(error, KnowledgeJobLeaseLost):
                raise
            raise KnowledgeJobPersistenceError() from None

    async def renew(
        self,
        lease: KnowledgeJobLease,
        *,
        lease_seconds: int,
    ) -> KnowledgeJobLease:
        _validate_worker(lease.worker_id, lease_seconds)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _RENEW_JOB,
                        (
                            lease_seconds,
                            lease.record.job_id,
                            lease.worker_id,
                            lease.lease_token,
                        ),
                    )
                    row = await cursor.fetchone()
                    if row is None:
                        raise KnowledgeJobLeaseLost()
                    record = _record_from_row(row)
                    return KnowledgeJobLease(
                        record=record,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_until=record.lease_until,
                    )
        except KnowledgeJobLeaseLost:
            raise
        except Exception:
            raise KnowledgeJobPersistenceError() from None

    async def complete(
        self,
        lease: KnowledgeJobLease,
        result: KnowledgeJobResult,
    ) -> KnowledgeJobRecord:
        validated = KnowledgeJobResult.model_validate(result.model_dump(mode="python"))
        encoded = _bounded_json(validated.model_dump(mode="json"), maximum=_MAX_RESULT_BYTES)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _COMPLETE_JOB,
                        (
                            encoded,
                            lease.record.job_id,
                            lease.worker_id,
                            lease.lease_token,
                        ),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        completed = _record_from_row(row)
                        await _audit_graph_write_transition(
                            connection,
                            completed,
                            event_type="succeeded",
                            code=validated.code,
                        )
                        return completed
                    cursor = await connection.execute(_SELECT_BY_ID, (lease.record.job_id,))
                    row = await cursor.fetchone()
                    if row is None:
                        raise KnowledgeJobLeaseLost()
                    existing = _record_from_row(row)
                    if (
                        existing.status is KnowledgeJobStatus.SUCCEEDED
                        and existing.lease_token == lease.lease_token
                        and existing.result == validated
                    ):
                        await _audit_graph_write_transition(
                            connection,
                            existing,
                            event_type="succeeded",
                            code=validated.code,
                        )
                        return existing
                    raise KnowledgeJobLeaseLost()
        except KnowledgeJobLeaseLost:
            raise
        except Exception:
            raise KnowledgeJobPersistenceError() from None

    async def fail(
        self,
        lease: KnowledgeJobLease,
        *,
        code: str,
        retryable: bool,
        retry_delay_seconds: int,
    ) -> KnowledgeJobRecord:
        safe_code = code if _SAFE_CODE.fullmatch(code) else "worker.dependency_failure"
        if isinstance(retry_delay_seconds, bool) or retry_delay_seconds < 0:
            raise ValueError("retry delay cannot be negative")
        will_retry = retryable and lease.record.attempt < lease.record.max_attempts
        status = (
            KnowledgeJobStatus.RETRY_WAIT if will_retry else KnowledgeJobStatus.FAILED_PERMANENT
        )
        delay = retry_delay_seconds if will_retry else 0
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _FAIL_JOB,
                        (
                            status.value,
                            delay,
                            safe_code,
                            status.value,
                            lease.record.job_id,
                            lease.worker_id,
                            lease.lease_token,
                        ),
                    )
                    row = await cursor.fetchone()
                    if row is not None:
                        failed = _record_from_row(row)
                        await _audit_graph_write_transition(
                            connection,
                            failed,
                            event_type=(
                                "retry_scheduled"
                                if status is KnowledgeJobStatus.RETRY_WAIT
                                else "failed_permanent"
                            ),
                            code=safe_code,
                        )
                        return failed
                    cursor = await connection.execute(_SELECT_BY_ID, (lease.record.job_id,))
                    row = await cursor.fetchone()
                    if row is None:
                        raise KnowledgeJobLeaseLost()
                    existing = _record_from_row(row)
                    if (
                        existing.status is status
                        and existing.lease_token == lease.lease_token
                        and existing.last_error_code == safe_code
                    ):
                        await _audit_graph_write_transition(
                            connection,
                            existing,
                            event_type=(
                                "retry_scheduled"
                                if status is KnowledgeJobStatus.RETRY_WAIT
                                else "failed_permanent"
                            ),
                            code=safe_code,
                        )
                        return existing
                    raise KnowledgeJobLeaseLost()
        except KnowledgeJobLeaseLost:
            raise
        except Exception:
            raise KnowledgeJobPersistenceError() from None


__all__ = [
    "KnowledgeJobPersistenceError",
    "PostgresKnowledgeJobRepository",
]
