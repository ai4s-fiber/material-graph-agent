"""Psycopg 3 persistence adapters for the durable knowledge boundary.

The driver is imported only by the pool factories. Production code can use
psycopg pools while tests inject compatible recording pools without requiring
a local PostgreSQL server.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from datetime import datetime
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from .catalog import (
    CatalogWriteResult,
    RelationType,
    SourceRelation,
    _canonical_remote_modified_at,
    _is_excluded_process_data,
    _merge_records,
    build_source_version_key,
    choose_canonical_source,
    normalize_doi,
)
from .lightrag_client import LightRAGSourceMappingConflict
from .lightrag_models import LightRAGSourceMapping
from .models import EvidenceFragment, SourceCatalogRecord
from .processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    SourceLifecycleStatus,
)

KNOWLEDGE_SCHEMA_VERSION = "knowledge_0001"
_MAX_JSONB_BYTES = 262_144
_MAX_JSON_DEPTH = 16
_MAX_EVIDENCE_CHARS = 65_536
_RAW_KEYS = frozenset(
    {
        "complete_mineru_output",
        "complete_parser_output",
        "full_document",
        "full_document_text",
        "mineru_json",
        "mineru_markdown",
        "original_pdf",
        "parser_output",
        "pdf_bytes",
        "raw_document",
        "raw_pdf",
        "source_bytes",
    }
)
_SENSITIVE_MARKERS = frozenset(
    {
        "address",
        "api_key",
        "credential",
        "endpoint",
        "host",
        "password",
        "secret",
        "session",
        "token",
        "username",
    }
)
_EXCEPTION_KEYS = frozenset({"exception", "error_detail", "stack_trace", "traceback"})
_TRANSPORT_KEYS = frozenset(
    {
        "cookie",
        "cookies",
        "device_id",
        "did",
        "quickconnect_did",
        "sid",
        "syno_token",
        "synotoken",
        "transport",
    }
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_API_KEY = re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{16,}", re.IGNORECASE)
_INGESTION_KEY = re.compile(r"^knowledge-ingestion:v2:(?P<source>[0-9a-f]{32}):[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_JOBS = frozenset(
    {
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED_PERMANENT,
        IngestionJobStatus.CANCELLED,
    }
)
_TERMINAL_LIFECYCLES = frozenset(
    {
        SourceLifecycleStatus.DEDUPLICATED,
        SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
        SourceLifecycleStatus.EVIDENCE_RETAINED,
        SourceLifecycleStatus.PARSED_NO_VALUE,
        SourceLifecycleStatus.FAILED_PERMANENT,
    }
)
_STAGE_RANK = {stage: index for index, stage in enumerate(IngestionStage)}


class KnowledgePersistenceError(RuntimeError):
    """Stable, value-free persistence failure."""


class KnowledgeRecordConflict(KnowledgePersistenceError):
    """An immutable durable identity was reused for different data."""


class CheckpointRegressionError(KnowledgeRecordConflict):
    """A stale worker attempted to rewind durable progress."""


class UnsafeDurablePayload(ValueError):
    """A payload is unsafe for the durable boundary."""


class _SyncCursor(Protocol):
    def fetchone(self) -> Mapping[str, Any] | None: ...

    def fetchall(self) -> Sequence[Mapping[str, Any]]: ...


class _SyncConnection(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> _SyncCursor: ...

    def transaction(self) -> AbstractContextManager[Any]: ...


class SyncConnectionPool(Protocol):
    def connection(self) -> AbstractContextManager[_SyncConnection]: ...


class _AsyncCursor(Protocol):
    async def fetchone(self) -> Mapping[str, Any] | None: ...

    async def fetchall(self) -> Sequence[Mapping[str, Any]]: ...


class _AsyncConnection(Protocol):
    async def execute(self, query: str, params: Sequence[Any] | None = None) -> _AsyncCursor: ...

    def transaction(self) -> AbstractAsyncContextManager[Any]: ...


class AsyncConnectionPool(Protocol):
    def connection(self) -> AbstractAsyncContextManager[_AsyncConnection]: ...


def create_psycopg_sync_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 4
) -> SyncConnectionPool:
    """Create a closed psycopg 3 sync pool using mapping rows."""

    _validate_pool_settings(dsn, min_size=min_size, max_size=max_size)
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError:
        raise RuntimeError("install psycopg[binary,pool]>=3.2,<4 for PostgreSQL storage") from None
    return ConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=False,
    )


def create_psycopg_async_pool(
    dsn: str, *, min_size: int = 1, max_size: int = 16
) -> AsyncConnectionPool:
    """Create a closed psycopg 3 async pool using mapping rows."""

    _validate_pool_settings(dsn, min_size=min_size, max_size=max_size)
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError:
        raise RuntimeError("install psycopg[binary,pool]>=3.2,<4 for PostgreSQL storage") from None
    return AsyncConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=False,
    )


def _validate_pool_settings(dsn: str, *, min_size: int, max_size: int) -> None:
    if not isinstance(dsn, str) or not dsn.strip():
        raise ValueError("PostgreSQL DSN is required")
    if min_size < 0 or max_size <= 0 or min_size > max_size:
        raise ValueError("invalid PostgreSQL pool size")


def _key_is_forbidden(key: str) -> bool:
    normalized = key.strip().casefold().replace("-", "_")
    if normalized in _RAW_KEYS or normalized in _EXCEPTION_KEYS or normalized in _TRANSPORT_KEYS:
        return True
    return any(
        normalized == marker
        or normalized.startswith(f"{marker}_")
        or normalized.endswith(f"_{marker}")
        for marker in _SENSITIVE_MARKERS
    )


def _contains_secret(value: str) -> bool:
    return bool(_JWT.search(value) or _API_KEY.search(value) or _BEARER.search(value))


def _reject_secret_text(*values: object) -> None:
    if any(isinstance(value, str) and _contains_secret(value) for value in values):
        raise UnsafeDurablePayload("credential-like values are forbidden in durable fields")


def _safe_json_value(value: object, *, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise UnsafeDurablePayload("durable JSON exceeds the nesting limit")
    if isinstance(value, BaseException):
        raise UnsafeDurablePayload("exception objects are forbidden in durable JSON")
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsafeDurablePayload("non-finite numbers are forbidden in durable JSON")
        return value
    if isinstance(value, str):
        if value.lstrip().startswith("%PDF-"):
            raise UnsafeDurablePayload("raw PDF data is forbidden in durable JSON")
        if _contains_secret(value):
            raise UnsafeDurablePayload("credential-like values are forbidden in durable JSON")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key.strip():
                raise UnsafeDurablePayload("durable JSON keys must be non-empty strings")
            if _key_is_forbidden(key):
                raise UnsafeDurablePayload("forbidden field at the durable JSON boundary")
            result[key] = _safe_json_value(nested, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, depth=depth + 1) for item in value]
    raise UnsafeDurablePayload("durable JSON contains an unsupported value type")


def _validated_json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise UnsafeDurablePayload(f"{field} must be a JSON object")
    safe = _safe_json_value(value)
    assert isinstance(safe, dict)
    encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_JSONB_BYTES:
        raise UnsafeDurablePayload(f"{field} exceeds the durable JSON size limit")
    return safe


def _json_parameter(value: object, *, field: str) -> str:
    safe = _validated_json_object(value, field=field)
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object_from_row(value: object, *, field: str) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise KnowledgePersistenceError("stored durable JSON is invalid") from None
    try:
        return _validated_json_object(value, field=field)
    except UnsafeDurablePayload:
        raise KnowledgePersistenceError("stored durable JSON violates the contract") from None


def _row(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgePersistenceError("PostgreSQL row_factory must return mapping rows")
    return value


def _model(factory: Any, payload: object, *, label: str) -> Any:
    try:
        return factory(payload)
    except Exception:
        raise KnowledgePersistenceError(f"stored {label} violates the contract") from None


_SOURCE_COLUMN_NAMES = (
    "source_id",
    "root_id",
    "relative_path",
    "source_version_key",
    "source_kind",
    "display_title",
    "status",
    "directory_year",
    "normalized_doi",
    "application_number",
    "publication_number",
    "grant_number",
    "legal_status",
    "sha256",
    "byte_size",
    "material_category",
    "knowledge_domain",
    "locator",
    "metadata",
    "canonical_source_id",
)
_SOURCE_COLUMNS = ", ".join(_SOURCE_COLUMN_NAMES)
_CANONICAL_SOURCE_COLUMNS = ", ".join(f"canonical.{column}" for column in _SOURCE_COLUMN_NAMES)
_SELECT_SOURCE_BY_PATH = f"""
SELECT {_SOURCE_COLUMNS}
FROM knowledge_sources
WHERE root_id = %s AND relative_path = %s
FOR UPDATE
"""
_SELECT_SOURCE_BY_ID = f"""
SELECT {_SOURCE_COLUMNS}
FROM knowledge_sources
WHERE source_id = %s
"""
_UPSERT_SOURCE = f"""
INSERT INTO knowledge_sources ({_SOURCE_COLUMNS}) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, (%s)::jsonb, (%s)::jsonb, %s
)
ON CONFLICT (root_id, relative_path) DO UPDATE SET
    source_version_key = EXCLUDED.source_version_key,
    source_kind = EXCLUDED.source_kind,
    display_title = EXCLUDED.display_title,
    status = EXCLUDED.status,
    directory_year = EXCLUDED.directory_year,
    normalized_doi = EXCLUDED.normalized_doi,
    application_number = EXCLUDED.application_number,
    publication_number = EXCLUDED.publication_number,
    grant_number = EXCLUDED.grant_number,
    legal_status = EXCLUDED.legal_status,
    sha256 = EXCLUDED.sha256,
    byte_size = EXCLUDED.byte_size,
    material_category = EXCLUDED.material_category,
    knowledge_domain = EXCLUDED.knowledge_domain,
    locator = EXCLUDED.locator,
    metadata = EXCLUDED.metadata,
    canonical_source_id = EXCLUDED.canonical_source_id,
    updated_at = now()
RETURNING {_SOURCE_COLUMNS}
"""
_ADVISORY_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"


def _source_from_row(raw: object) -> SourceCatalogRecord:
    row = _row(raw)
    payload = {
        "source_id": row.get("source_id"),
        "locator": _json_object_from_row(row.get("locator"), field="source.locator"),
        "source_kind": row.get("source_kind"),
        "display_title": row.get("display_title"),
        "status": row.get("status"),
        "directory_year": row.get("directory_year"),
        "normalized_doi": row.get("normalized_doi"),
        "application_number": row.get("application_number"),
        "publication_number": row.get("publication_number"),
        "grant_number": row.get("grant_number"),
        "legal_status": row.get("legal_status"),
        "sha256": row.get("sha256"),
        "byte_size": row.get("byte_size"),
        "material_category": row.get("material_category"),
        "knowledge_domain": row.get("knowledge_domain"),
        "canonical_source_id": row.get("canonical_source_id"),
        "metadata": _json_object_from_row(row.get("metadata"), field="source.metadata"),
    }
    return _model(SourceCatalogRecord.model_validate, payload, label="source")


def _prepare_source(
    record: SourceCatalogRecord,
    remote_modified_at: datetime | str | int | float | None,
) -> tuple[SourceCatalogRecord, str]:
    if not isinstance(record, SourceCatalogRecord):
        raise TypeError("catalog repository accepts SourceCatalogRecord instances only")
    if record.model_extra:
        raise UnsafeDurablePayload("untyped source fields are forbidden in PostgreSQL storage")
    record = SourceCatalogRecord.model_validate(record.model_dump(mode="python"))
    _reject_secret_text(
        record.locator.root_id,
        record.locator.relative_path,
        record.display_title,
        record.application_number,
        record.publication_number,
        record.grant_number,
        record.legal_status,
        record.material_category,
        record.knowledge_domain,
    )
    version_key = build_source_version_key(
        locator=record.locator,
        byte_size=record.byte_size,
        remote_modified_at=remote_modified_at,
    )
    metadata = dict(record.metadata)
    metadata["source_version_key"] = version_key
    canonical_mtime = _canonical_remote_modified_at(remote_modified_at)
    if canonical_mtime is not None:
        metadata["remote_modified_at"] = canonical_mtime
    status = record.status
    if _is_excluded_process_data(record.locator):
        status = "excluded_process_data"
        metadata["exclusion_reason"] = "process_data_never_open"
    return (
        record.model_copy(
            update={
                "normalized_doi": normalize_doi(record.normalized_doi),
                "status": status,
                "metadata": _validated_json_object(metadata, field="source.metadata"),
                "canonical_source_id": None,
            },
            deep=True,
        ),
        version_key,
    )


def _source_parameters(record: SourceCatalogRecord, version_key: str) -> tuple[object, ...]:
    return (
        record.source_id,
        record.locator.root_id,
        record.locator.relative_path,
        version_key,
        record.source_kind,
        record.display_title,
        record.status,
        record.directory_year,
        record.normalized_doi,
        record.application_number,
        record.publication_number,
        record.grant_number,
        record.legal_status,
        record.sha256,
        record.byte_size,
        record.material_category,
        record.knowledge_domain,
        _json_parameter(record.locator.model_dump(mode="json"), field="source.locator"),
        _json_parameter(record.metadata, field="source.metadata"),
        record.canonical_source_id,
    )


class PostgresSourceCatalogRepository:
    """Synchronous PostgreSQL implementation of ``SourceCatalogRepository``."""

    def __init__(self, pool: SyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def upsert(
        self,
        record: SourceCatalogRecord,
        *,
        remote_modified_at: datetime | str | int | float | None = None,
    ) -> CatalogWriteResult:
        prepared, version_key = _prepare_source(record, remote_modified_at)
        path_lock = (
            f"knowledge-source:path:{prepared.locator.root_id}:{prepared.locator.relative_path}"
        )
        with self._pool.connection() as connection:
            with connection.transaction():
                connection.execute(_ADVISORY_LOCK, (path_lock,))
                raw = connection.execute(
                    _SELECT_SOURCE_BY_PATH,
                    (prepared.locator.root_id, prepared.locator.relative_path),
                ).fetchone()
                existing = None if raw is None else _source_from_row(raw)
                created = existing is None
                previous_sha = None if existing is None else existing.sha256
                previous_doi = None if existing is None else existing.normalized_doi
                stored = prepared if existing is None else _merge_records(existing, prepared)
                version_key = str(stored.metadata["source_version_key"])

                identities = {
                    f"knowledge-source:sha:{value}"
                    for value in (previous_sha, stored.sha256)
                    if value
                }
                identities.update(
                    f"knowledge-source:doi:{value}"
                    for value in (previous_doi, stored.normalized_doi)
                    if value
                )
                for identity in sorted(identities):
                    connection.execute(_ADVISORY_LOCK, (identity,))

                raw = connection.execute(
                    _UPSERT_SOURCE, _source_parameters(stored, version_key)
                ).fetchone()
                if raw is None:
                    raise KnowledgePersistenceError("catalog upsert returned no durable source")
                durable = _source_from_row(raw)
                affected_dois = {value for value in (previous_doi, durable.normalized_doi) if value}
                for digest in sorted({value for value in (previous_sha, durable.sha256) if value}):
                    affected_dois.update(self._reconcile_sha(connection, digest))
                for doi in sorted(affected_dois):
                    self._reconcile_doi(connection, doi)
                raw = connection.execute(_SELECT_SOURCE_BY_ID, (durable.source_id,)).fetchone()
                if raw is None:
                    raise KnowledgePersistenceError("catalog source disappeared during transaction")
                durable = _source_from_row(raw)
        return CatalogWriteResult(record=durable, created=created, source_version_key=version_key)

    @staticmethod
    def _reconcile_sha(connection: _SyncConnection, digest: str) -> set[str]:
        rows = connection.execute(
            f"""
            SELECT {_SOURCE_COLUMNS} FROM knowledge_sources
            WHERE sha256 = %s ORDER BY source_id FOR UPDATE
            """,
            (digest,),
        ).fetchall()
        records = [_source_from_row(item) for item in rows]
        if not records:
            return set()
        canonical = choose_canonical_source(records)
        connection.execute(
            """
            UPDATE knowledge_sources
            SET canonical_source_id = CASE WHEN source_id = %s THEN NULL ELSE %s END,
                updated_at = now()
            WHERE sha256 = %s
            """,
            (canonical.source_id, canonical.source_id, digest),
        )
        return {record.normalized_doi for record in records if record.normalized_doi}

    @staticmethod
    def _reconcile_doi(connection: _SyncConnection, doi: str) -> None:
        connection.execute(
            """
            DELETE FROM knowledge_source_relations
            WHERE relation_type = 'IS_VERSION_OF' AND normalized_doi = %s
            """,
            (doi,),
        )
        rows = connection.execute(
            f"""
            SELECT {_CANONICAL_SOURCE_COLUMNS}
            FROM knowledge_sources AS member
            JOIN knowledge_sources AS canonical
              ON canonical.source_id = COALESCE(member.canonical_source_id, member.source_id)
            WHERE member.normalized_doi = %s AND member.sha256 IS NOT NULL
            ORDER BY canonical.source_id
            """,
            (doi,),
        ).fetchall()
        representatives: dict[UUID, SourceCatalogRecord] = {}
        for raw in rows:
            record = _source_from_row(raw)
            representatives[record.source_id] = record
        versions = list(representatives.values())
        if len(versions) < 2:
            return
        canonical = choose_canonical_source(versions)
        for version in sorted(versions, key=lambda item: item.source_id.hex):
            if version.source_id == canonical.source_id:
                continue
            connection.execute(
                """
                INSERT INTO knowledge_source_relations (
                    relation_type, source_id, target_source_id, normalized_doi, reason
                ) VALUES ('IS_VERSION_OF', %s, %s, %s, %s)
                ON CONFLICT (relation_type, source_id, target_source_id) DO UPDATE SET
                    normalized_doi = EXCLUDED.normalized_doi,
                    reason = EXCLUDED.reason
                """,
                (
                    version.source_id,
                    canonical.source_id,
                    doi,
                    "same_normalized_doi_with_different_sha256",
                ),
            )

    def get(self, source_id: UUID) -> SourceCatalogRecord:
        with self._pool.connection() as connection:
            raw = connection.execute(_SELECT_SOURCE_BY_ID, (source_id,)).fetchone()
        if raw is None:
            raise KeyError(f"unknown source: {source_id}")
        return _source_from_row(raw)

    def canonical_for(self, source_id: UUID) -> SourceCatalogRecord:
        record = self.get(source_id)
        return (
            record if record.canonical_source_id is None else self.get(record.canonical_source_id)
        )

    def relations(self, relation_type: RelationType | None = None) -> list[SourceRelation]:
        params: tuple[object, ...] = ()
        predicate = ""
        if relation_type is not None:
            if relation_type not in {"DUPLICATE_OF", "IS_VERSION_OF"}:
                raise ValueError("invalid source relation type")
            predicate = "WHERE relation_type = %s"
            params = (relation_type,)
        with self._pool.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT relation_type, source_id, target_source_id, normalized_doi, reason
                FROM knowledge_source_relations {predicate}
                ORDER BY relation_type, source_id, target_source_id
                """,
                params,
            ).fetchall()
        return [
            _model(SourceRelation.model_validate, dict(_row(item)), label="source relation")
            for item in rows
        ]


_CHECKPOINT_COLUMNS = """
idempotency_key, source_id, source_version_fingerprint, embedding_generation_id,
lifecycle_status, stage, job_status, attempt, selection, cursor, metadata,
last_error_category
""".strip()
_SELECT_CHECKPOINT = f"""
SELECT {_CHECKPOINT_COLUMNS}
FROM knowledge_ingestion_checkpoints
WHERE idempotency_key = %s
"""
_SELECT_CHECKPOINT_FOR_UPDATE = _SELECT_CHECKPOINT + " FOR UPDATE"
_UPSERT_CHECKPOINT = f"""
INSERT INTO knowledge_ingestion_checkpoints ({_CHECKPOINT_COLUMNS}) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    (%s)::jsonb, (%s)::jsonb, (%s)::jsonb, %s
)
ON CONFLICT (idempotency_key) DO UPDATE SET
    lifecycle_status = EXCLUDED.lifecycle_status,
    stage = EXCLUDED.stage,
    job_status = EXCLUDED.job_status,
    attempt = EXCLUDED.attempt,
    selection = EXCLUDED.selection,
    cursor = EXCLUDED.cursor,
    metadata = EXCLUDED.metadata,
    last_error_category = EXCLUDED.last_error_category,
    updated_at = now()
WHERE knowledge_ingestion_checkpoints.source_id = EXCLUDED.source_id
  AND knowledge_ingestion_checkpoints.source_version_fingerprint =
      EXCLUDED.source_version_fingerprint
  AND knowledge_ingestion_checkpoints.embedding_generation_id =
      EXCLUDED.embedding_generation_id
  AND CASE knowledge_ingestion_checkpoints.stage
      WHEN 'catalog' THEN 0 WHEN 'hash' THEN 1 WHEN 'select' THEN 2
      WHEN 'spool' THEN 3 WHEN 'parse' THEN 4 WHEN 'retain' THEN 5
      WHEN 'index' THEN 6 END
      <= CASE EXCLUDED.stage
      WHEN 'catalog' THEN 0 WHEN 'hash' THEN 1 WHEN 'select' THEN 2
      WHEN 'spool' THEN 3 WHEN 'parse' THEN 4 WHEN 'retain' THEN 5
      WHEN 'index' THEN 6 END
  AND knowledge_ingestion_checkpoints.attempt <= EXCLUDED.attempt
  AND (
      knowledge_ingestion_checkpoints.lifecycle_status NOT IN
          ('deduplicated', 'excluded_process_data', 'evidence_retained',
           'parsed_no_value', 'failed_permanent')
      OR knowledge_ingestion_checkpoints.lifecycle_status = EXCLUDED.lifecycle_status
  )
  AND (
      knowledge_ingestion_checkpoints.selection IS NULL
      OR knowledge_ingestion_checkpoints.selection = EXCLUDED.selection
  )
  AND (
      knowledge_ingestion_checkpoints.job_status NOT IN
          ('succeeded', 'failed_permanent', 'cancelled')
      OR knowledge_ingestion_checkpoints.job_status = EXCLUDED.job_status
  )
RETURNING idempotency_key
"""


def _checkpoint_identity(checkpoint: ProcessingCheckpoint) -> tuple[str, str]:
    match = _INGESTION_KEY.fullmatch(checkpoint.idempotency_key)
    if match is None or match.group("source") != checkpoint.source_id.hex:
        raise UnsafeDurablePayload("checkpoint idempotency key is invalid")
    fingerprint = checkpoint.metadata.get("source_version_fingerprint")
    generation = checkpoint.metadata.get("embedding_generation_id")
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise UnsafeDurablePayload("checkpoint source version fingerprint is invalid")
    if not isinstance(generation, str) or not generation.strip():
        raise UnsafeDurablePayload("checkpoint embedding generation is invalid")
    _reject_secret_text(generation, checkpoint.last_error_category)
    return fingerprint, generation


def _checkpoint_from_row(raw: object) -> ProcessingCheckpoint:
    row = _row(raw)
    selection = row.get("selection")
    if selection is not None:
        selection = _json_object_from_row(selection, field="checkpoint.selection")
    payload = {
        "source_id": row.get("source_id"),
        "lifecycle_status": row.get("lifecycle_status"),
        "stage": row.get("stage"),
        "job_status": row.get("job_status"),
        "attempt": row.get("attempt"),
        "idempotency_key": row.get("idempotency_key"),
        "selection": selection,
        "cursor": _json_object_from_row(row.get("cursor"), field="checkpoint.cursor"),
        "metadata": _json_object_from_row(row.get("metadata"), field="checkpoint.metadata"),
        "last_error_category": row.get("last_error_category"),
    }
    checkpoint = _model(ProcessingCheckpoint.model_validate, payload, label="checkpoint")
    _checkpoint_identity(checkpoint)
    return checkpoint


def _checkpoint_parameters(checkpoint: ProcessingCheckpoint) -> tuple[object, ...]:
    fingerprint, generation = _checkpoint_identity(checkpoint)
    selection = None
    if checkpoint.selection is not None:
        selection = _json_parameter(
            checkpoint.selection.model_dump(mode="json"), field="checkpoint.selection"
        )
    return (
        checkpoint.idempotency_key,
        checkpoint.source_id,
        fingerprint,
        generation,
        checkpoint.lifecycle_status,
        checkpoint.stage,
        checkpoint.job_status,
        checkpoint.attempt,
        selection,
        _json_parameter(checkpoint.cursor, field="checkpoint.cursor"),
        _json_parameter(checkpoint.metadata, field="checkpoint.metadata"),
        checkpoint.last_error_category,
    )


def _assert_checkpoint_progress(
    existing: ProcessingCheckpoint, candidate: ProcessingCheckpoint
) -> None:
    if existing.source_id != candidate.source_id:
        raise KnowledgeRecordConflict("checkpoint source identity conflict")
    if _checkpoint_identity(existing) != _checkpoint_identity(candidate):
        raise KnowledgeRecordConflict("checkpoint ingestion identity conflict")
    if _STAGE_RANK[candidate.stage] < _STAGE_RANK[existing.stage]:
        raise CheckpointRegressionError("checkpoint stage regression is forbidden")
    if candidate.attempt < existing.attempt:
        raise CheckpointRegressionError("checkpoint attempt regression is forbidden")
    if existing.job_status in _TERMINAL_JOBS and candidate.job_status != existing.job_status:
        raise CheckpointRegressionError("terminal checkpoint job status is immutable")
    if (
        existing.lifecycle_status in _TERMINAL_LIFECYCLES
        and candidate.lifecycle_status != existing.lifecycle_status
    ):
        raise CheckpointRegressionError("terminal checkpoint lifecycle is immutable")
    if existing.selection is not None and candidate.selection != existing.selection:
        raise KnowledgeRecordConflict("checkpoint selection is immutable")


class PostgresCheckpointRepository:
    """Async PostgreSQL checkpoint repository with monotonic writes."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def load(self, idempotency_key: str) -> ProcessingCheckpoint | None:
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("checkpoint idempotency key is required")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(_SELECT_CHECKPOINT, (idempotency_key,))
            raw = await cursor.fetchone()
        return None if raw is None else _checkpoint_from_row(raw)

    async def save(self, checkpoint: ProcessingCheckpoint) -> None:
        if not isinstance(checkpoint, ProcessingCheckpoint):
            raise TypeError("checkpoint repository accepts ProcessingCheckpoint instances only")
        candidate = ProcessingCheckpoint.model_validate(checkpoint.model_dump(mode="python"))
        params = _checkpoint_parameters(candidate)
        async with self._pool.connection() as connection:
            async with connection.transaction():
                cursor = await connection.execute(
                    _SELECT_CHECKPOINT_FOR_UPDATE, (candidate.idempotency_key,)
                )
                raw = await cursor.fetchone()
                if raw is not None:
                    _assert_checkpoint_progress(_checkpoint_from_row(raw), candidate)
                cursor = await connection.execute(_UPSERT_CHECKPOINT, params)
                if await cursor.fetchone() is None:
                    raise CheckpointRegressionError("checkpoint write was rejected as stale")


_EVIDENCE_COLUMNS = """
fragment_id, source_id, idempotency_key, text, locator, content_sha256,
retention_reason, supported_entity_ids, supported_relation_ids, parser_name,
parser_version, embedding_generation_id, metadata
""".strip()
_INSERT_EVIDENCE = f"""
INSERT INTO knowledge_evidence_fragments ({_EVIDENCE_COLUMNS}) VALUES (
    %s, %s, %s, %s, (%s)::jsonb, %s, %s, %s, %s, %s, %s, %s, (%s)::jsonb
)
ON CONFLICT (fragment_id) DO UPDATE SET fragment_id = EXCLUDED.fragment_id
WHERE knowledge_evidence_fragments.source_id = EXCLUDED.source_id
  AND knowledge_evidence_fragments.idempotency_key = EXCLUDED.idempotency_key
  AND knowledge_evidence_fragments.text = EXCLUDED.text
  AND knowledge_evidence_fragments.locator = EXCLUDED.locator
  AND knowledge_evidence_fragments.content_sha256 = EXCLUDED.content_sha256
  AND knowledge_evidence_fragments.parser_name = EXCLUDED.parser_name
  AND knowledge_evidence_fragments.parser_version = EXCLUDED.parser_version
  AND knowledge_evidence_fragments.embedding_generation_id =
      EXCLUDED.embedding_generation_id
RETURNING fragment_id
"""


def _validate_ingestion_key(source_id: UUID, idempotency_key: str) -> None:
    if not isinstance(idempotency_key, str):
        raise TypeError("evidence idempotency key must be a string")
    match = _INGESTION_KEY.fullmatch(idempotency_key)
    if match is None or match.group("source") != source_id.hex:
        raise ValueError("evidence idempotency key does not match the source")


def _evidence_identity_matches(left: EvidenceFragment, right: EvidenceFragment) -> bool:
    return (
        left.fragment_id == right.fragment_id
        and left.source_id == right.source_id
        and left.text == right.text
        and left.locator == right.locator
        and left.content_sha256 == right.content_sha256
        and left.parser_name == right.parser_name
        and left.parser_version == right.parser_version
        and left.embedding_generation_id == right.embedding_generation_id
    )


def _expected_fragment_id(fragment: EvidenceFragment, idempotency_key: str) -> UUID:
    locator = fragment.locator
    identity = "|".join(
        (
            idempotency_key,
            fragment.content_sha256 or "",
            str(locator.page or 0),
            str(locator.block_index or 0),
            locator.section or "",
        )
    )
    return uuid5(NAMESPACE_URL, identity)


def _validate_fragment(
    fragment: EvidenceFragment, *, source_id: UUID, idempotency_key: str
) -> EvidenceFragment:
    if not isinstance(fragment, EvidenceFragment):
        raise TypeError("evidence repository accepts EvidenceFragment instances only")
    fragment = EvidenceFragment.model_validate(fragment.model_dump(mode="python"))
    if fragment.source_id != source_id:
        raise ValueError("evidence source_id does not match repository key")
    if fragment.fragment_id != _expected_fragment_id(fragment, idempotency_key):
        raise ValueError("evidence fragment_id is not deterministic for the ingestion identity")
    if len(fragment.text) > _MAX_EVIDENCE_CHARS:
        raise UnsafeDurablePayload("retained evidence exceeds the durable fragment size limit")
    if fragment.text.lstrip().startswith("%PDF-"):
        raise UnsafeDurablePayload("raw PDF data is forbidden in retained evidence")
    if _contains_secret(fragment.text):
        raise UnsafeDurablePayload("credential-like data is forbidden in retained evidence")
    _reject_secret_text(
        fragment.retention_reason,
        fragment.parser_name,
        fragment.parser_version,
        fragment.embedding_generation_id,
        *fragment.supported_entity_ids,
        *fragment.supported_relation_ids,
    )
    _validated_json_object(fragment.metadata, field="evidence.metadata")
    return fragment


def _evidence_parameters(fragment: EvidenceFragment, idempotency_key: str) -> tuple[object, ...]:
    return (
        fragment.fragment_id,
        fragment.source_id,
        idempotency_key,
        fragment.text,
        _json_parameter(fragment.locator.model_dump(mode="json"), field="evidence.locator"),
        fragment.content_sha256,
        fragment.retention_reason,
        list(fragment.supported_entity_ids),
        list(fragment.supported_relation_ids),
        fragment.parser_name,
        fragment.parser_version,
        fragment.embedding_generation_id,
        _json_parameter(fragment.metadata, field="evidence.metadata"),
    )


def _evidence_from_row(raw: object) -> EvidenceFragment:
    row = _row(raw)
    payload = {
        "fragment_id": row.get("fragment_id"),
        "source_id": row.get("source_id"),
        "text": row.get("text"),
        "locator": _json_object_from_row(row.get("locator"), field="evidence.locator"),
        "content_sha256": row.get("content_sha256"),
        "retention_reason": row.get("retention_reason"),
        "supported_entity_ids": row.get("supported_entity_ids") or [],
        "supported_relation_ids": row.get("supported_relation_ids") or [],
        "parser_name": row.get("parser_name"),
        "parser_version": row.get("parser_version"),
        "embedding_generation_id": row.get("embedding_generation_id"),
        "metadata": _json_object_from_row(row.get("metadata"), field="evidence.metadata"),
    }
    return _model(EvidenceFragment.model_validate, payload, label="evidence")


class PostgresEvidenceRepository:
    """Async evidence-only store with immutable deterministic fragment IDs."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def persist_many(
        self,
        source_id: UUID,
        fragments: Sequence[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> None:
        _validate_ingestion_key(source_id, idempotency_key)
        candidates: dict[UUID, EvidenceFragment] = {}
        for fragment in fragments:
            fragment = _validate_fragment(
                fragment, source_id=source_id, idempotency_key=idempotency_key
            )
            existing = candidates.get(fragment.fragment_id)
            if existing is not None and not _evidence_identity_matches(existing, fragment):
                raise KnowledgeRecordConflict("conflicting evidence fragment identity")
            candidates.setdefault(fragment.fragment_id, fragment)
        if not candidates:
            return
        async with self._pool.connection() as connection:
            async with connection.transaction():
                for fragment_id in sorted(candidates, key=lambda item: item.hex):
                    cursor = await connection.execute(
                        _INSERT_EVIDENCE,
                        _evidence_parameters(candidates[fragment_id], idempotency_key),
                    )
                    if await cursor.fetchone() is None:
                        raise KnowledgeRecordConflict("conflicting durable evidence fragment")

    async def list_for_source(
        self, source_id: UUID, *, idempotency_key: str
    ) -> list[EvidenceFragment]:
        _validate_ingestion_key(source_id, idempotency_key)
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS} FROM knowledge_evidence_fragments
                WHERE source_id = %s AND idempotency_key = %s ORDER BY fragment_id
                """,
                (source_id, idempotency_key),
            )
            rows = await cursor.fetchall()
        return [_evidence_from_row(item) for item in rows]


_MAPPING_COLUMNS = """
basename, fragment_id, source_id, locator, logical_source_uri,
content_sha256, embedding_generation_id
""".strip()
_INSERT_MAPPING = f"""
INSERT INTO knowledge_lightrag_source_mappings ({_MAPPING_COLUMNS})
VALUES (%s, %s, %s, (%s)::jsonb, %s, %s, %s)
ON CONFLICT DO NOTHING
RETURNING basename
"""


def _mapping_parameters(mapping: LightRAGSourceMapping) -> tuple[object, ...]:
    return (
        mapping.basename,
        mapping.fragment_id,
        mapping.source_id,
        _json_parameter(mapping.locator.model_dump(mode="json"), field="mapping.locator"),
        mapping.logical_source_uri,
        mapping.content_sha256,
        mapping.embedding_generation_id,
    )


def _mapping_from_row(raw: object) -> LightRAGSourceMapping:
    row = _row(raw)
    payload = {
        "basename": row.get("basename"),
        "fragment_id": row.get("fragment_id"),
        "source_id": row.get("source_id"),
        "locator": _json_object_from_row(row.get("locator"), field="mapping.locator"),
        "logical_source_uri": row.get("logical_source_uri"),
        "content_sha256": row.get("content_sha256"),
        "embedding_generation_id": row.get("embedding_generation_id"),
    }
    return _model(
        LightRAGSourceMapping.model_validate,
        payload,
        label="LightRAG source mapping",
    )


class PostgresLightRAGSourceMappingReader:
    """Synchronous lookup boundary used by online LightRAG retrieval."""

    def __init__(self, pool: SyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def get(self, basename: str) -> LightRAGSourceMapping | None:
        if not isinstance(basename, str) or not basename.strip():
            raise ValueError("LightRAG basename is required")
        normalized = basename.strip()
        if normalized != basename:
            raise ValueError("LightRAG basename must be normalized")
        with self._pool.connection() as connection:
            raw = connection.execute(
                f"""
                SELECT {_MAPPING_COLUMNS}
                FROM knowledge_lightrag_source_mappings WHERE basename = %s
                """,
                (normalized,),
            ).fetchone()
        return None if raw is None else _mapping_from_row(raw)


class PostgresLightRAGSourceMappingStore:
    """Atomic provenance mapping persisted before LightRAG submission."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def persist_many(self, mappings: Sequence[LightRAGSourceMapping]) -> None:
        candidates: dict[str, LightRAGSourceMapping] = {}
        for mapping in mappings:
            if not isinstance(mapping, LightRAGSourceMapping):
                raise TypeError("mapping store accepts LightRAGSourceMapping instances only")
            mapping = LightRAGSourceMapping.model_validate(mapping.model_dump(mode="python"))
            _reject_secret_text(
                mapping.basename,
                mapping.logical_source_uri,
                mapping.embedding_generation_id,
                mapping.locator.root_id,
                mapping.locator.relative_path,
            )
            existing = candidates.get(mapping.basename)
            if existing is not None and existing != mapping:
                raise LightRAGSourceMappingConflict("duplicate basename in mapping transaction")
            candidates[mapping.basename] = mapping
        if not candidates:
            return
        async with self._pool.connection() as connection:
            async with connection.transaction():
                for basename in sorted(candidates):
                    mapping = candidates[basename]
                    cursor = await connection.execute(_INSERT_MAPPING, _mapping_parameters(mapping))
                    if await cursor.fetchone() is not None:
                        continue
                    cursor = await connection.execute(
                        f"""
                        SELECT {_MAPPING_COLUMNS}
                        FROM knowledge_lightrag_source_mappings
                        WHERE basename = %s OR fragment_id = %s
                        ORDER BY basename FOR UPDATE
                        """,
                        (mapping.basename, mapping.fragment_id),
                    )
                    rows = await cursor.fetchall()
                    if not any(_mapping_from_row(item) == mapping for item in rows):
                        raise LightRAGSourceMappingConflict(
                            "LightRAG provenance identity is already bound differently"
                        )

    async def get(self, basename: str) -> LightRAGSourceMapping | None:
        if not isinstance(basename, str) or not basename.strip():
            raise ValueError("LightRAG basename is required")
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT {_MAPPING_COLUMNS}
                FROM knowledge_lightrag_source_mappings WHERE basename = %s
                """,
                (basename,),
            )
            raw = await cursor.fetchone()
        return None if raw is None else _mapping_from_row(raw)


__all__ = [
    "AsyncConnectionPool",
    "CheckpointRegressionError",
    "KNOWLEDGE_SCHEMA_VERSION",
    "KnowledgePersistenceError",
    "KnowledgeRecordConflict",
    "PostgresCheckpointRepository",
    "PostgresEvidenceRepository",
    "PostgresLightRAGSourceMappingReader",
    "PostgresLightRAGSourceMappingStore",
    "PostgresSourceCatalogRepository",
    "SyncConnectionPool",
    "UnsafeDurablePayload",
    "create_psycopg_async_pool",
    "create_psycopg_sync_pool",
]
