"""Psycopg 3 repositories for extraction checkpoints and canary stage claims.

Both adapters revalidate typed state before opening a connection, retain no
source body or provider response, and serialize competing claims with
transaction-scoped advisory locks plus row locks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from .extraction import (
    FactExtractionCheckpoint,
    FactExtractionCheckpointConflict,
    _checkpoint_identity,
    _valid_transition,
)
from .facts import FactBatch
from .postgres import AsyncConnectionPool, UnsafeDurablePayload, _safe_json_value
from .service import (
    CanaryStage,
    CanaryStageRecord,
    CanaryStatus,
    KnowledgeCanaryError,
    _SAFE_RUN_ID,
)


PIPELINE_STATE_SCHEMA_VERSION = "knowledge_0002"
_MAX_EXTRACTION_BYTES = 65_536
_MAX_BATCH_BYTES = 4_194_304
_MAX_CANARY_ATTEMPT = 2_147_483_647
_FACT_KEY = re.compile(r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
_INTERNAL_LOCATION = re.compile(
    r"(?:quickconnect|(?:^|[\s\"'(])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/]"
    r"|/volume\d+/|(?:smb|nfs|file|nas)://))",
    re.IGNORECASE,
)
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "absolute_path",
        "complete_mineru_output",
        "complete_parser_output",
        "document_bytes",
        "document_text",
        "extractor_response",
        "fragment_text",
        "full_document",
        "full_markdown",
        "full_text",
        "local_path",
        "manifest_path",
        "mineru_output",
        "nas_path",
        "original_pdf",
        "parser_output",
        "pdf",
        "pdf_bytes",
        "provider_output",
        "raw_document",
        "raw_markdown",
        "raw_pdf",
        "raw_provider_output",
        "relative_path",
        "response_body",
        "source_bytes",
        "source_path",
        "source_text",
    }
)


class PipelineStatePersistenceError(RuntimeError):
    """Stable, value-free PostgreSQL state persistence failure."""


class UnsafePipelineStatePayload(ValueError):
    """Unsafe data attempted to cross the durable pipeline-state boundary."""


_LOCK_KEYS = """
SELECT pg_advisory_xact_lock(hashtextextended(lock_key, 0))
FROM unnest(%s::text[]) AS locks(lock_key)
ORDER BY lock_key
"""

_CHECKPOINT_COLUMNS = """
idempotency_key, fragment_id, source_id, fragment_content_sha256,
request_fingerprint, extraction, status, attempts, batch, last_error_code
""".strip()
_SELECT_CHECKPOINT = f"""
SELECT {_CHECKPOINT_COLUMNS}
FROM knowledge_fact_extraction_checkpoints
WHERE idempotency_key = %s
FOR UPDATE
"""
_LOAD_CHECKPOINT = f"""
SELECT {_CHECKPOINT_COLUMNS}
FROM knowledge_fact_extraction_checkpoints
WHERE idempotency_key = %s
"""
_INSERT_CHECKPOINT = f"""
INSERT INTO knowledge_fact_extraction_checkpoints ({_CHECKPOINT_COLUMNS})
VALUES (%s, %s, %s, %s, %s, (%s)::jsonb, %s, %s, (%s)::jsonb, %s)
RETURNING {_CHECKPOINT_COLUMNS}
"""
_UPDATE_CHECKPOINT = f"""
UPDATE knowledge_fact_extraction_checkpoints SET
    fragment_id = %s,
    source_id = %s,
    fragment_content_sha256 = %s,
    request_fingerprint = %s,
    extraction = (%s)::jsonb,
    status = %s,
    attempts = %s,
    batch = (%s)::jsonb,
    last_error_code = %s,
    updated_at = now()
WHERE idempotency_key = %s AND status = %s AND attempts = %s
RETURNING {_CHECKPOINT_COLUMNS}
"""

_CANARY_COLUMNS = """
run_id, stage, status, attempt, code, request_fingerprint, source_ids,
approval_id, attempted_count, completed_count, evidence_count,
metadata_records, resumed
""".strip()
_SELECT_CANARY = f"""
SELECT {_CANARY_COLUMNS}
FROM knowledge_canary_runs
WHERE run_id = %s AND stage = %s
FOR UPDATE
"""
_LOAD_CANARY = f"""
SELECT {_CANARY_COLUMNS}
FROM knowledge_canary_runs
WHERE run_id = %s AND stage = %s
"""
_INSERT_CANARY = f"""
INSERT INTO knowledge_canary_runs ({_CANARY_COLUMNS})
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING {_CANARY_COLUMNS}
"""
_RESTART_CANARY = f"""
UPDATE knowledge_canary_runs SET
    status = %s,
    attempt = %s,
    code = %s,
    request_fingerprint = %s,
    source_ids = %s,
    approval_id = %s,
    attempted_count = 0,
    completed_count = 0,
    evidence_count = 0,
    metadata_records = 0,
    resumed = true,
    updated_at = now()
WHERE run_id = %s AND stage = %s AND status = %s AND attempt = %s
RETURNING {_CANARY_COLUMNS}
"""
_FINISH_CANARY = f"""
UPDATE knowledge_canary_runs SET
    status = %s,
    code = %s,
    attempted_count = %s,
    completed_count = %s,
    evidence_count = %s,
    metadata_records = %s,
    resumed = %s,
    updated_at = now()
WHERE run_id = %s
  AND stage = %s
  AND status = 'running'
  AND attempt = %s
  AND request_fingerprint = %s
  AND source_ids = %s
  AND approval_id IS NOT DISTINCT FROM %s
RETURNING {_CANARY_COLUMNS}
"""


def _reject_unsafe_state(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")
            normalized = key.strip().casefold().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_STATE_KEYS:
                raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")
            _reject_unsafe_state(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _reject_unsafe_state(nested)
        return
    if isinstance(value, str) and _INTERNAL_LOCATION.search(value):
        raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")


def _safe_json_object(value: object, *, max_bytes: int) -> dict[str, object]:
    try:
        if not isinstance(value, Mapping):
            raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")
        _reject_unsafe_state(value)
        safe = _safe_json_value(value)
        if not isinstance(safe, dict):  # pragma: no cover - guarded above
            raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")
        rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered.encode("utf-8")) > max_bytes:
            raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload")
        return safe
    except (UnsafeDurablePayload, TypeError, ValueError):
        raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload") from None


def _json_parameter(value: object, *, max_bytes: int) -> str:
    safe = _safe_json_object(value, max_bytes=max_bytes)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_from_row(value: object, *, max_bytes: int) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None
    try:
        return _safe_json_object(value, max_bytes=max_bytes)
    except UnsafePipelineStatePayload:
        raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


def _row(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineStatePersistenceError("pipeline_state_persistence_error")
    return value


def _validated_checkpoint(value: object) -> FactExtractionCheckpoint:
    if not isinstance(value, FactExtractionCheckpoint):
        raise TypeError("checkpoint must be a FactExtractionCheckpoint")
    try:
        return FactExtractionCheckpoint.model_validate(value.model_dump(mode="python"))
    except (ValidationError, TypeError, ValueError):
        raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload") from None


def _redacted_batch(batch: FactBatch, fragment_id: UUID) -> dict[str, object]:
    payload = batch.model_dump(mode="json")
    expected_path = f"fragments/{fragment_id}"
    try:
        for collection in ("relations", "observations"):
            for fact in payload[collection]:
                for link in fact["evidence"]:
                    locator = link["locator"]
                    if locator.pop("relative_path") != expected_path:
                        raise ValueError("unexpected evidence path")
    except (KeyError, TypeError, ValueError):
        raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload") from None
    return _safe_json_object(payload, max_bytes=_MAX_BATCH_BYTES)


def _restored_batch(value: object, fragment_id: UUID) -> FactBatch:
    payload = deepcopy(_json_from_row(value, max_bytes=_MAX_BATCH_BYTES))
    relative_path = f"fragments/{fragment_id}"
    try:
        for collection in ("relations", "observations"):
            for fact in payload[collection]:
                for link in fact["evidence"]:
                    link["locator"]["relative_path"] = relative_path
        return FactBatch.model_validate(payload)
    except (KeyError, TypeError, ValueError, ValidationError):
        raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


def restore_redacted_fact_batch(value: object, fragment_id: UUID) -> FactBatch:
    """Restore the synthetic fragment locator used by durable extraction state."""

    return _restored_batch(value, fragment_id)


def _checkpoint_parameters(checkpoint: FactExtractionCheckpoint) -> tuple[object, ...]:
    batch_parameter: str | None = None
    if checkpoint.batch is not None:
        batch_parameter = _json_parameter(
            _redacted_batch(checkpoint.batch, checkpoint.fragment_id),
            max_bytes=_MAX_BATCH_BYTES,
        )
    return (
        checkpoint.idempotency_key,
        checkpoint.fragment_id,
        checkpoint.source_id,
        checkpoint.fragment_content_sha256,
        checkpoint.request_fingerprint,
        _json_parameter(
            checkpoint.extraction.model_dump(mode="json"),
            max_bytes=_MAX_EXTRACTION_BYTES,
        ),
        checkpoint.status,
        checkpoint.attempts,
        batch_parameter,
        checkpoint.last_error_code,
    )


def _checkpoint_from_row(value: object) -> FactExtractionCheckpoint:
    row = _row(value)
    fragment_id = row.get("fragment_id")
    try:
        resolved_fragment_id = UUID(str(fragment_id))
        raw_batch = row.get("batch")
        payload = {
            "idempotency_key": row.get("idempotency_key"),
            "fragment_id": resolved_fragment_id,
            "source_id": row.get("source_id"),
            "fragment_content_sha256": row.get("fragment_content_sha256"),
            "request_fingerprint": row.get("request_fingerprint"),
            "extraction": _json_from_row(
                row.get("extraction"),
                max_bytes=_MAX_EXTRACTION_BYTES,
            ),
            "status": row.get("status"),
            "attempts": row.get("attempts"),
            "batch": None
            if raw_batch is None
            else _restored_batch(raw_batch, resolved_fragment_id),
            "last_error_code": row.get("last_error_code"),
        }
        return FactExtractionCheckpoint.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


def _checkpoint_update_parameters(
    candidate: FactExtractionCheckpoint,
    existing: FactExtractionCheckpoint,
) -> tuple[object, ...]:
    parameters = _checkpoint_parameters(candidate)
    return (*parameters[1:], parameters[0], existing.status, existing.attempts)


def _validate_checkpoint_key(idempotency_key: str) -> None:
    if not isinstance(idempotency_key, str) or _FACT_KEY.fullmatch(idempotency_key) is None:
        raise ValueError("invalid extraction checkpoint key")


def _validate_run_key(run_id: str, stage: CanaryStage) -> CanaryStage:
    if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid canary run key")
    try:
        resolved = CanaryStage(stage)
        _safe_json_object({"run_id": run_id, "stage": resolved.value}, max_bytes=1024)
        return resolved
    except (ValueError, UnsafePipelineStatePayload):
        raise ValueError("invalid canary run key") from None


def _validated_claim(
    *,
    run_id: str,
    stage: CanaryStage,
    request_fingerprint: str,
    source_ids: tuple[UUID, ...],
    approval_id: str | None,
) -> CanaryStageRecord:
    resolved_stage = _validate_run_key(run_id, stage)
    try:
        record = CanaryStageRecord(
            run_id=run_id,
            stage=resolved_stage,
            status=CanaryStatus.RUNNING,
            attempt=1,
            code="canary_running",
            request_fingerprint=request_fingerprint,
            source_ids=source_ids,
            approval_id=approval_id,
        )
        _safe_json_object(
            {
                "run_id": record.run_id,
                "approval_id": record.approval_id,
                "code": record.code,
            },
            max_bytes=2048,
        )
        return record
    except (ValidationError, TypeError, ValueError, UnsafePipelineStatePayload):
        raise ValueError("invalid canary stage claim") from None


def _validated_canary_record(value: object) -> CanaryStageRecord:
    if not isinstance(value, CanaryStageRecord):
        raise TypeError("record must be a CanaryStageRecord")
    try:
        record = CanaryStageRecord.model_validate(value.model_dump(mode="python"))
        _safe_json_object(
            {
                "run_id": record.run_id,
                "approval_id": record.approval_id,
                "code": record.code,
            },
            max_bytes=2048,
        )
        return record
    except (ValidationError, TypeError, ValueError, UnsafePipelineStatePayload):
        raise UnsafePipelineStatePayload("unsafe_pipeline_state_payload") from None


def _canary_parameters(record: CanaryStageRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.stage.value,
        record.status.value,
        record.attempt,
        record.code,
        record.request_fingerprint,
        list(record.source_ids),
        record.approval_id,
        record.attempted_count,
        record.completed_count,
        record.evidence_count,
        record.metadata_records,
        record.resumed,
    )


def _canary_from_row(value: object) -> CanaryStageRecord:
    row = _row(value)
    try:
        return _validated_canary_record(
            CanaryStageRecord(
                run_id=row.get("run_id"),
                stage=row.get("stage"),
                status=row.get("status"),
                attempt=row.get("attempt"),
                code=row.get("code"),
                request_fingerprint=row.get("request_fingerprint"),
                source_ids=tuple(row.get("source_ids") or ()),
                approval_id=row.get("approval_id"),
                attempted_count=row.get("attempted_count"),
                completed_count=row.get("completed_count"),
                evidence_count=row.get("evidence_count"),
                metadata_records=row.get("metadata_records"),
                resumed=row.get("resumed"),
            )
        )
    except (ValidationError, TypeError, ValueError, UnsafePipelineStatePayload):
        raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


def _canary_identity(record: CanaryStageRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.stage,
        record.request_fingerprint,
        record.source_ids,
        record.approval_id,
    )


class PostgresFactExtractionCheckpointRepository:
    """Atomic PostgreSQL implementation of extraction checkpoint recovery."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def load(self, idempotency_key: str) -> FactExtractionCheckpoint | None:
        _validate_checkpoint_key(idempotency_key)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(_LOAD_CHECKPOINT, (idempotency_key,))
                    raw = await cursor.fetchone()
                    return None if raw is None else _checkpoint_from_row(raw)
        except PipelineStatePersistenceError:
            raise
        except Exception:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None

    async def save(self, checkpoint: FactExtractionCheckpoint) -> FactExtractionCheckpoint:
        candidate = _validated_checkpoint(checkpoint)
        parameters = _checkpoint_parameters(candidate)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(
                        _LOCK_KEYS,
                        ([f"fact-extraction:{candidate.idempotency_key}"],),
                    )
                    cursor = await connection.execute(
                        _SELECT_CHECKPOINT,
                        (candidate.idempotency_key,),
                    )
                    raw = await cursor.fetchone()
                    if raw is None:
                        cursor = await connection.execute(_INSERT_CHECKPOINT, parameters)
                        inserted = await cursor.fetchone()
                        if inserted is None:
                            raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                        return _checkpoint_from_row(inserted)

                    existing = _checkpoint_from_row(raw)
                    if _checkpoint_identity(existing) != _checkpoint_identity(candidate):
                        raise FactExtractionCheckpointConflict()
                    if not _valid_transition(existing, candidate):
                        raise FactExtractionCheckpointConflict()
                    if existing == candidate:
                        return _checkpoint_from_row(raw)

                    cursor = await connection.execute(
                        _UPDATE_CHECKPOINT,
                        _checkpoint_update_parameters(candidate, existing),
                    )
                    updated = await cursor.fetchone()
                    if updated is None:
                        raise FactExtractionCheckpointConflict()
                    persisted = _checkpoint_from_row(updated)
                    if persisted != candidate:
                        raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                    return persisted
        except (FactExtractionCheckpointConflict, PipelineStatePersistenceError):
            raise
        except Exception:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


class PostgresCanaryRunRepository:
    """Atomic PostgreSQL claim store for staged production canaries."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def load(self, run_id: str, stage: CanaryStage) -> CanaryStageRecord | None:
        resolved_stage = _validate_run_key(run_id, stage)
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    cursor = await connection.execute(
                        _LOAD_CANARY,
                        (run_id, resolved_stage.value),
                    )
                    raw = await cursor.fetchone()
                    return None if raw is None else _canary_from_row(raw)
        except PipelineStatePersistenceError:
            raise
        except Exception:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None

    async def begin(
        self,
        *,
        run_id: str,
        stage: CanaryStage,
        request_fingerprint: str,
        source_ids: tuple[UUID, ...],
        approval_id: str | None,
    ) -> CanaryStageRecord:
        template = _validated_claim(
            run_id=run_id,
            stage=stage,
            request_fingerprint=request_fingerprint,
            source_ids=source_ids,
            approval_id=approval_id,
        )
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(
                        _LOCK_KEYS,
                        ([f"canary:{template.run_id}:{template.stage.value}"],),
                    )
                    cursor = await connection.execute(
                        _SELECT_CANARY,
                        (template.run_id, template.stage.value),
                    )
                    raw = await cursor.fetchone()
                    if raw is None:
                        cursor = await connection.execute(
                            _INSERT_CANARY,
                            _canary_parameters(template),
                        )
                        inserted = await cursor.fetchone()
                        if inserted is None:
                            raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                        persisted = _canary_from_row(inserted)
                        if persisted != template:
                            raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                        return persisted

                    existing = _canary_from_row(raw)
                    if _canary_identity(existing) != _canary_identity(template):
                        raise KnowledgeCanaryError("canary_stage_identity_mismatch")
                    if existing.status is CanaryStatus.RUNNING:
                        raise KnowledgeCanaryError("canary_stage_running")
                    if existing.attempt >= _MAX_CANARY_ATTEMPT:
                        raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                    restarted = template.model_copy(
                        update={"attempt": existing.attempt + 1, "resumed": True},
                        deep=True,
                    )
                    cursor = await connection.execute(
                        _RESTART_CANARY,
                        (
                            restarted.status.value,
                            restarted.attempt,
                            restarted.code,
                            restarted.request_fingerprint,
                            list(restarted.source_ids),
                            restarted.approval_id,
                            restarted.run_id,
                            restarted.stage.value,
                            existing.status.value,
                            existing.attempt,
                        ),
                    )
                    updated = await cursor.fetchone()
                    if updated is None:
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    persisted = _canary_from_row(updated)
                    if persisted != restarted:
                        raise PipelineStatePersistenceError("pipeline_state_persistence_error")
                    return persisted
        except (KnowledgeCanaryError, PipelineStatePersistenceError):
            raise
        except Exception:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None

    async def finish(self, record: CanaryStageRecord) -> None:
        candidate = _validated_canary_record(record)
        if candidate.status is CanaryStatus.RUNNING:
            raise ValueError("cannot finish a running canary record")
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(
                        _LOCK_KEYS,
                        ([f"canary:{candidate.run_id}:{candidate.stage.value}"],),
                    )
                    cursor = await connection.execute(
                        _SELECT_CANARY,
                        (candidate.run_id, candidate.stage.value),
                    )
                    raw = await cursor.fetchone()
                    if raw is None:
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    existing = _canary_from_row(raw)
                    if existing.attempt != candidate.attempt:
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    if existing.status is not CanaryStatus.RUNNING:
                        if existing == candidate:
                            return
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    if _canary_identity(existing) != _canary_identity(candidate):
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    cursor = await connection.execute(
                        _FINISH_CANARY,
                        (
                            candidate.status.value,
                            candidate.code,
                            candidate.attempted_count,
                            candidate.completed_count,
                            candidate.evidence_count,
                            candidate.metadata_records,
                            candidate.resumed,
                            candidate.run_id,
                            candidate.stage.value,
                            candidate.attempt,
                            candidate.request_fingerprint,
                            list(candidate.source_ids),
                            candidate.approval_id,
                        ),
                    )
                    updated = await cursor.fetchone()
                    if updated is None:
                        raise KnowledgeCanaryError("canary_stage_claim_lost")
                    if _canary_from_row(updated) != candidate:
                        raise PipelineStatePersistenceError("pipeline_state_persistence_error")
        except (KnowledgeCanaryError, PipelineStatePersistenceError):
            raise
        except Exception:
            raise PipelineStatePersistenceError("pipeline_state_persistence_error") from None


__all__ = [
    "PIPELINE_STATE_SCHEMA_VERSION",
    "PipelineStatePersistenceError",
    "PostgresCanaryRunRepository",
    "PostgresFactExtractionCheckpointRepository",
    "restore_redacted_fact_batch",
    "UnsafePipelineStatePayload",
]
