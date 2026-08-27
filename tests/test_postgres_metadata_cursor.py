from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest

from material_graph.knowledge.manifest import MetadataCursor, MetadataCursorKey
from material_graph.knowledge.postgres import KnowledgePersistenceError, UnsafeDurablePayload
from material_graph.knowledge.postgres_manifest import (
    MetadataCursorConflict,
    MetadataCursorRegressionError,
    PostgresMetadataCursorRepository,
)


ROOT = Path(__file__).parents[1]
VERSION_A = "source-version-v1:" + "a" * 64
VERSION_B = "source-version-v1:" + "b" * 64


@dataclass(frozen=True)
class Statement:
    sql: str
    params: tuple[object, ...]


class Script:
    def __init__(self) -> None:
        self.responses: dict[str, deque[list[Mapping[str, Any]]]] = defaultdict(deque)
        self.failures: dict[str, BaseException] = {}

    def add(self, needle: str, *responses: Sequence[Mapping[str, Any]]) -> None:
        for response in responses:
            self.responses[needle].append(list(response))

    def fail(self, needle: str, error: BaseException) -> None:
        self.failures[needle] = error

    def take(self, sql: str) -> list[Mapping[str, Any]]:
        compact = " ".join(sql.split())
        for needle, error in self.failures.items():
            if needle in compact:
                raise error
        for needle, responses in self.responses.items():
            if needle in compact and responses:
                return responses.popleft()
        return []


class Cursor:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def fetchone(self) -> Mapping[str, Any] | None:
        return None if not self.rows else self.rows[0]

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class RecordingConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> Cursor:
        self.statements.append(Statement(sql, tuple(params or ())))
        return Cursor(self.script.take(sql))

    @contextmanager
    def transaction(self):
        self.transactions += 1
        try:
            yield self
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1


class RecordingPool:
    def __init__(self, connection: RecordingConnection) -> None:
        self.value = connection
        self.connections = 0
        self.dsn = "postgresql://secret-user:secret-password@internal/db"

    @contextmanager
    def connection(self):
        self.connections += 1
        yield self.value

    def __repr__(self) -> str:
        return f"RecordingPool({self.dsn!r})"


def metadata_cursor(
    *,
    version: str = VERSION_A,
    offset: int = 0,
    records: int = 0,
    fieldnames: tuple[str, ...] | None = None,
    manifest_format: str = "jsonl",
) -> MetadataCursor:
    return MetadataCursor(
        root_id="document_data_1",
        slice_id="literature",
        manifest_path=(
            "private/manifests/catalog.csv"
            if manifest_format == "csv"
            else "private/manifests/catalog.jsonl"
        ),
        manifest_format=manifest_format,  # type: ignore[arg-type]
        manifest_version_key=version,
        next_byte_offset=offset,
        records_committed=records,
        csv_fieldnames=fieldnames,
    )


def row(value: MetadataCursor, *, json_string: bool = False) -> dict[str, object]:
    checkpoint: object = value.to_checkpoint()
    if json_string:
        checkpoint = json.dumps(
            checkpoint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return {
        "root_id": value.root_id,
        "slice_id": value.slice_id,
        "manifest_path": value.manifest_path,
        "manifest_format": value.manifest_format,
        "manifest_version_key": value.manifest_version_key,
        "next_byte_offset": value.next_byte_offset,
        "records_committed": value.records_committed,
        "checkpoint": checkpoint,
    }


def _repository(script: Script) -> tuple[PostgresMetadataCursorRepository, RecordingConnection]:
    connection = RecordingConnection(script)
    return PostgresMetadataCursorRepository(RecordingPool(connection)), connection


def test_insert_load_and_missing_lookup_use_exact_logical_identity() -> None:
    value = metadata_cursor(offset=128, records=3)
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [], [row(value, json_string=True)], [])
    script.add("INSERT INTO knowledge_metadata_cursors", [row(value)])
    repository, connection = _repository(script)

    repository.save(value)
    assert repository.load(value.key) == value
    assert (
        repository.load(
            MetadataCursorKey(
                root_id="document_data_1",
                slice_id="literature",
                manifest_path="private/manifests/missing.jsonl",
                manifest_format="jsonl",
            )
        )
        is None
    )

    insert = next(
        statement
        for statement in connection.statements
        if "INSERT INTO knowledge_metadata_cursors" in statement.sql
    )
    assert insert.params[:4] == (
        value.root_id,
        value.slice_id,
        value.manifest_path,
        value.manifest_format,
    )
    assert json.loads(str(insert.params[-1])) == value.to_checkpoint()
    assert connection.transactions == 1
    assert connection.commits == 1
    assert repr(repository) == "PostgresMetadataCursorRepository()"
    assert "secret-password" not in repr(repository)


def test_monotonic_update_and_exact_replay_are_idempotent() -> None:
    existing = metadata_cursor(offset=128, records=3)
    advanced = metadata_cursor(offset=256, records=5)
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [row(existing)], [row(advanced)])
    script.add("UPDATE knowledge_metadata_cursors", [row(advanced)])
    repository, connection = _repository(script)

    repository.save(advanced)
    repository.save(advanced)

    updates = [
        statement
        for statement in connection.statements
        if "UPDATE knowledge_metadata_cursors" in statement.sql
    ]
    assert len(updates) == 1
    assert connection.transactions == 2
    assert connection.commits == 2


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [
        (
            metadata_cursor(offset=128, records=3),
            metadata_cursor(version=VERSION_B, offset=256, records=4),
        ),
        (
            metadata_cursor(
                offset=128,
                records=3,
                fieldnames=("title", "doi"),
                manifest_format="csv",
            ),
            metadata_cursor(
                offset=256,
                records=4,
                fieldnames=("title", "sha256"),
                manifest_format="csv",
            ),
        ),
        (metadata_cursor(offset=128, records=3), metadata_cursor(offset=128, records=4)),
    ],
)
def test_immutable_version_and_csv_schema_conflicts_are_rejected(
    existing: MetadataCursor,
    candidate: MetadataCursor,
) -> None:
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [row(existing)])
    repository, connection = _repository(script)

    with pytest.raises(MetadataCursorConflict, match="metadata_cursor_conflict"):
        repository.save(candidate)

    assert not any(
        "UPDATE knowledge_metadata_cursors" in statement.sql for statement in connection.statements
    )
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    "candidate",
    [
        metadata_cursor(offset=127, records=3),
        metadata_cursor(offset=256, records=2),
    ],
)
def test_progress_regression_is_rejected(candidate: MetadataCursor) -> None:
    existing = metadata_cursor(offset=128, records=3)
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [row(existing)])
    repository, connection = _repository(script)

    with pytest.raises(MetadataCursorRegressionError, match="metadata_cursor_regression"):
        repository.save(candidate)

    assert connection.rollbacks == 1


def test_csv_header_can_be_durably_established_once() -> None:
    existing = metadata_cursor(manifest_format="csv")
    advanced = metadata_cursor(
        offset=64,
        fieldnames=("title", "doi"),
        manifest_format="csv",
    )
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [row(existing)])
    script.add("UPDATE knowledge_metadata_cursors", [row(advanced)])
    repository, _ = _repository(script)

    repository.save(advanced)


def test_cursor_json_is_bounded_before_opening_a_connection() -> None:
    oversized = metadata_cursor(
        offset=1,
        fieldnames=("x" * (262_144 + 1),),
        manifest_format="csv",
    )
    script = Script()
    repository, _ = _repository(script)

    with pytest.raises(UnsafeDurablePayload, match="size limit"):
        repository.save(oversized)

    assert repository._pool.connections == 0  # type: ignore[attr-defined]


def test_invalid_stored_checkpoint_fails_closed_and_backend_errors_are_sanitized() -> None:
    value = metadata_cursor(offset=128, records=3)
    invalid = row(value)
    invalid["checkpoint"] = {**value.to_checkpoint(), "records_committed": 4}
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [invalid])
    repository, _ = _repository(script)

    with pytest.raises(KnowledgePersistenceError, match="stored metadata cursor"):
        repository.load(value.key)

    failing = Script()
    failing.fail("FROM knowledge_metadata_cursors", RuntimeError("password=do-not-leak"))
    repository, _ = _repository(failing)
    with pytest.raises(KnowledgePersistenceError) as captured:
        repository.load(value.key)
    assert str(captured.value) == "metadata_cursor_persistence_failed"


@pytest.mark.parametrize("operation", ["load", "save"])
def test_repository_rejects_wrong_contract_types(operation: str) -> None:
    repository, _ = _repository(Script())

    with pytest.raises(TypeError):
        if operation == "load":
            repository.load(object())  # type: ignore[arg-type]
        else:
            repository.save(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("corruption", ["checkpoint", "identity", "progress"])
def test_stored_cursor_corruption_is_rejected(corruption: str) -> None:
    value = metadata_cursor(offset=128, records=3)
    invalid = row(value)
    if corruption == "checkpoint":
        invalid["checkpoint"] = {"schema_version": 1}
    elif corruption == "identity":
        invalid["manifest_path"] = "other/catalog.jsonl"
    else:
        invalid["next_byte_offset"] = True
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [invalid])
    repository, _ = _repository(script)

    with pytest.raises(KnowledgePersistenceError, match="stored metadata cursor"):
        repository.load(value.key)


def test_row_for_a_different_logical_identity_is_rejected_during_save() -> None:
    candidate = metadata_cursor(offset=256, records=4)
    other = MetadataCursor(
        **{
            **candidate.to_checkpoint(),
            "manifest_path": "private/manifests/other.jsonl",
        }
    )
    script = Script()
    script.add("FROM knowledge_metadata_cursors", [row(other)])
    repository, _ = _repository(script)

    with pytest.raises(MetadataCursorConflict):
        repository.save(candidate)


@pytest.mark.parametrize("operation", ["insert", "update"])
def test_database_write_rejection_is_fail_closed(operation: str) -> None:
    candidate = metadata_cursor(offset=256, records=4)
    script = Script()
    if operation == "insert":
        script.add("FROM knowledge_metadata_cursors", [])
        script.add("INSERT INTO knowledge_metadata_cursors", [])
        expected = MetadataCursorConflict
    else:
        script.add("FROM knowledge_metadata_cursors", [row(metadata_cursor(offset=128, records=3))])
        script.add("UPDATE knowledge_metadata_cursors", [])
        expected = MetadataCursorRegressionError
    repository, _ = _repository(script)

    with pytest.raises(expected):
        repository.save(candidate)


@pytest.mark.parametrize("operation", ["insert", "update"])
def test_database_write_returning_a_different_cursor_is_rejected(operation: str) -> None:
    candidate = metadata_cursor(offset=256, records=4)
    different = metadata_cursor(offset=257, records=4)
    script = Script()
    if operation == "insert":
        script.add("FROM knowledge_metadata_cursors", [])
        script.add("INSERT INTO knowledge_metadata_cursors", [row(different)])
    else:
        script.add("FROM knowledge_metadata_cursors", [row(metadata_cursor(offset=128, records=3))])
        script.add("UPDATE knowledge_metadata_cursors", [row(different)])
    repository, _ = _repository(script)

    with pytest.raises(KnowledgePersistenceError, match="differs from the candidate"):
        repository.save(candidate)


def test_save_backend_error_is_sanitized() -> None:
    script = Script()
    script.fail("pg_advisory_xact_lock", RuntimeError("password=do-not-leak"))
    repository, _ = _repository(script)

    with pytest.raises(KnowledgePersistenceError) as captured:
        repository.save(metadata_cursor())
    assert str(captured.value) == "metadata_cursor_persistence_failed"


def test_migration_owns_cursor_table_and_runs_before_age() -> None:
    migration = (ROOT / "migrations/knowledge_0004.sql").read_text(encoding="utf-8")
    rollback = (ROOT / "migrations/knowledge_0004.down.sql").read_text(encoding="utf-8")
    migrate = (ROOT / "deploy/scripts/migrate.sh").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS knowledge_metadata_cursors" in migration
    assert "PRIMARY KEY (root_id, slice_id, manifest_path, manifest_format)" in migration
    assert "octet_length(checkpoint::text) <= 262144" in migration
    assert "knowledge_0004" in migration
    assert "DROP TABLE IF EXISTS knowledge_metadata_cursors" in rollback
    assert "DELETE FROM schema_migrations WHERE version = 'knowledge_0004'" in rollback
    assert migrate.index("knowledge_0003.sql") < migrate.index("knowledge_0004.sql")
    assert migrate.index("knowledge_0004.sql") < migrate.index("age_0001.sql")
