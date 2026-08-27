"""PostgreSQL-backed metadata manifest cursors.

The repository is synchronous because ``MetadataManifestIngestor`` commits a
cursor immediately after each catalog write.  Advisory transaction locks make
the missing-row insert path safe across workers, while immutable manifest
identity and monotonic progress checks prevent stale workers from rewinding a
stream.
"""

from __future__ import annotations

from .manifest import MetadataCursor, MetadataCursorKey
from .postgres import (
    CheckpointRegressionError,
    KnowledgePersistenceError,
    KnowledgeRecordConflict,
    SyncConnectionPool,
    UnsafeDurablePayload,
    _json_object_from_row,
    _json_parameter,
    _row,
)


_CURSOR_COLUMNS = """
root_id, slice_id, manifest_path, manifest_format, manifest_version_key,
next_byte_offset, records_committed, checkpoint
""".strip()

_SELECT_CURSOR = f"""
SELECT {_CURSOR_COLUMNS}
FROM knowledge_metadata_cursors
WHERE root_id = %s
  AND slice_id = %s
  AND manifest_path = %s
  AND manifest_format = %s
"""

_SELECT_CURSOR_FOR_UPDATE = _SELECT_CURSOR + " FOR UPDATE"

_INSERT_CURSOR = f"""
INSERT INTO knowledge_metadata_cursors (
    root_id, slice_id, manifest_path, manifest_format, manifest_version_key,
    next_byte_offset, records_committed, checkpoint
) VALUES (%s, %s, %s, %s, %s, %s, %s, (%s)::jsonb)
ON CONFLICT (root_id, slice_id, manifest_path, manifest_format) DO NOTHING
RETURNING {_CURSOR_COLUMNS}
"""

_UPDATE_CURSOR = f"""
UPDATE knowledge_metadata_cursors
SET manifest_version_key = %s,
    next_byte_offset = %s,
    records_committed = %s,
    checkpoint = (%s)::jsonb,
    updated_at = now()
WHERE root_id = %s
  AND slice_id = %s
  AND manifest_path = %s
  AND manifest_format = %s
  AND manifest_version_key = %s
  AND next_byte_offset <= %s
  AND records_committed <= %s
RETURNING {_CURSOR_COLUMNS}
"""

_LOCK_CURSOR = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"


class MetadataCursorConflict(KnowledgeRecordConflict):
    """The immutable manifest identity was reused for different content."""

    def __init__(self) -> None:
        super().__init__("metadata_cursor_conflict")


class MetadataCursorRegressionError(CheckpointRegressionError):
    """A stale worker attempted to rewind durable metadata progress."""

    def __init__(self) -> None:
        super().__init__("metadata_cursor_regression")


def _key_parameters(key: MetadataCursorKey) -> tuple[str, str, str, str]:
    return (key.root_id, key.slice_id, key.manifest_path, key.manifest_format)


def _validated_cursor(value: MetadataCursor) -> tuple[MetadataCursor, str]:
    if not isinstance(value, MetadataCursor):
        raise TypeError("metadata cursor repository accepts MetadataCursor instances only")
    candidate = MetadataCursor.from_checkpoint(value.to_checkpoint())
    checkpoint = _json_parameter(
        candidate.to_checkpoint(),
        field="metadata cursor checkpoint",
    )
    return candidate, checkpoint


def _cursor_from_row(raw: object) -> MetadataCursor:
    row = _row(raw)
    checkpoint = _json_object_from_row(
        row.get("checkpoint"),
        field="metadata cursor checkpoint",
    )
    try:
        cursor = MetadataCursor.from_checkpoint(checkpoint)
    except Exception:
        raise KnowledgePersistenceError("stored metadata cursor violates the contract") from None

    expected_text = {
        "root_id": cursor.root_id,
        "slice_id": cursor.slice_id,
        "manifest_path": cursor.manifest_path,
        "manifest_format": cursor.manifest_format,
        "manifest_version_key": cursor.manifest_version_key,
    }
    if any(row.get(name) != expected for name, expected in expected_text.items()):
        raise KnowledgePersistenceError("stored metadata cursor identity is inconsistent")
    for name, expected in (
        ("next_byte_offset", cursor.next_byte_offset),
        ("records_committed", cursor.records_committed),
    ):
        stored = row.get(name)
        if isinstance(stored, bool) or not isinstance(stored, int) or stored != expected:
            raise KnowledgePersistenceError("stored metadata cursor progress is inconsistent")
    return cursor


def _assert_progress(existing: MetadataCursor, candidate: MetadataCursor) -> None:
    if existing.key != candidate.key:
        raise MetadataCursorConflict()
    if existing.manifest_version_key != candidate.manifest_version_key:
        raise MetadataCursorConflict()
    if (
        candidate.next_byte_offset < existing.next_byte_offset
        or candidate.records_committed < existing.records_committed
    ):
        raise MetadataCursorRegressionError()
    if (
        candidate.next_byte_offset == existing.next_byte_offset
        and candidate.records_committed != existing.records_committed
    ):
        raise MetadataCursorConflict()
    if existing.csv_fieldnames is not None and candidate.csv_fieldnames != existing.csv_fieldnames:
        raise MetadataCursorConflict()


def _lock_identity(key: MetadataCursorKey) -> str:
    return "\x1f".join(("metadata-cursor-v1", *_key_parameters(key)))


def _insert_parameters(cursor: MetadataCursor, checkpoint: str) -> tuple[object, ...]:
    return (
        *_key_parameters(cursor.key),
        cursor.manifest_version_key,
        cursor.next_byte_offset,
        cursor.records_committed,
        checkpoint,
    )


def _update_parameters(cursor: MetadataCursor, checkpoint: str) -> tuple[object, ...]:
    return (
        cursor.manifest_version_key,
        cursor.next_byte_offset,
        cursor.records_committed,
        checkpoint,
        *_key_parameters(cursor.key),
        cursor.manifest_version_key,
        cursor.next_byte_offset,
        cursor.records_committed,
    )


class PostgresMetadataCursorRepository:
    """Durable cursor repository with exact identity and monotonic updates."""

    def __init__(self, pool: SyncConnectionPool) -> None:
        self._pool = pool

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def load(self, key: MetadataCursorKey) -> MetadataCursor | None:
        if not isinstance(key, MetadataCursorKey):
            raise TypeError("metadata cursor key must be a MetadataCursorKey")
        try:
            with self._pool.connection() as connection:
                raw = connection.execute(_SELECT_CURSOR, _key_parameters(key)).fetchone()
            return None if raw is None else _cursor_from_row(raw)
        except (KnowledgePersistenceError, UnsafeDurablePayload):
            raise
        except Exception:
            raise KnowledgePersistenceError("metadata_cursor_persistence_failed") from None

    def save(self, cursor: MetadataCursor) -> None:
        candidate, checkpoint = _validated_cursor(cursor)
        try:
            with self._pool.connection() as connection:
                with connection.transaction():
                    connection.execute(_LOCK_CURSOR, (_lock_identity(candidate.key),))
                    raw = connection.execute(
                        _SELECT_CURSOR_FOR_UPDATE,
                        _key_parameters(candidate.key),
                    ).fetchone()
                    if raw is None:
                        inserted = connection.execute(
                            _INSERT_CURSOR,
                            _insert_parameters(candidate, checkpoint),
                        ).fetchone()
                        if inserted is None:
                            raise MetadataCursorConflict()
                        if _cursor_from_row(inserted) != candidate:
                            raise KnowledgePersistenceError(
                                "stored metadata cursor differs from the candidate"
                            )
                        return

                    existing = _cursor_from_row(raw)
                    if existing == candidate:
                        return
                    _assert_progress(existing, candidate)
                    updated = connection.execute(
                        _UPDATE_CURSOR,
                        _update_parameters(candidate, checkpoint),
                    ).fetchone()
                    if updated is None:
                        raise MetadataCursorRegressionError()
                    if _cursor_from_row(updated) != candidate:
                        raise KnowledgePersistenceError(
                            "stored metadata cursor differs from the candidate"
                        )
        except (
            MetadataCursorConflict,
            MetadataCursorRegressionError,
            KnowledgePersistenceError,
            UnsafeDurablePayload,
        ):
            raise
        except Exception:
            raise KnowledgePersistenceError("metadata_cursor_persistence_failed") from None


__all__ = [
    "MetadataCursorConflict",
    "MetadataCursorRegressionError",
    "PostgresMetadataCursorRepository",
]
