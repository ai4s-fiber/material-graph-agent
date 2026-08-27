from __future__ import annotations

import json
import sys
import types
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest

from material_graph.knowledge import postgres as pg
from material_graph.knowledge.catalog import build_source_version_key
from material_graph.knowledge.ingestion import build_ingestion_idempotency_key
from material_graph.knowledge.lightrag_client import LightRAGSourceMappingConflict
from material_graph.knowledge.lightrag_models import LightRAGSourceMapping
from material_graph.knowledge.models import (
    EvidenceFragment,
    SelectionDecision,
    SourceCatalogRecord,
    SourceLocator,
)
from material_graph.knowledge.processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    SourceLifecycleStatus,
)

ROOT = Path(__file__).parents[1]
SOURCE_ID = UUID(int=1)
OTHER_SOURCE_ID = UUID(int=2)
GENERATION = "qwen3-embedding-8b:1024:v1"
SOURCE_VERSION = "source-version-v1:" + "a" * 64
VERSION_FINGERPRINT = sha256(SOURCE_VERSION.encode()).hexdigest()


@dataclass(frozen=True)
class Statement:
    sql: str
    params: tuple[object, ...]


class Script:
    def __init__(self) -> None:
        self.responses: dict[str, deque[list[Mapping[str, Any]]]] = defaultdict(deque)

    def add(self, needle: str, *responses: Sequence[Mapping[str, Any]]) -> None:
        for response in responses:
            self.responses[needle].append(list(response))

    def take(self, sql: str) -> list[Mapping[str, Any]]:
        compact = " ".join(sql.split())
        for needle, queued in self.responses.items():
            if needle in compact and queued:
                return queued.popleft()
        return []


class SyncCursor:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def fetchone(self) -> Mapping[str, Any] | None:
        return None if not self.rows else self.rows[0]

    def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class RecordingSyncConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0

    def execute(self, sql: str, params: Sequence[object] | None = None) -> SyncCursor:
        self.statements.append(Statement(sql, tuple(params or ())))
        return SyncCursor(self.script.take(sql))

    @contextmanager
    def transaction(self):
        self.transactions += 1
        yield self


class SyncPool:
    def __init__(self, connection: RecordingSyncConnection) -> None:
        self.value = connection
        self.connections = 0

    @contextmanager
    def connection(self):
        self.connections += 1
        yield self.value


class AsyncCursor:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    async def fetchone(self) -> Mapping[str, Any] | None:
        return None if not self.rows else self.rows[0]

    async def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class RecordingAsyncConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0

    async def execute(self, sql: str, params: Sequence[object] | None = None) -> AsyncCursor:
        self.statements.append(Statement(sql, tuple(params or ())))
        return AsyncCursor(self.script.take(sql))

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        yield self


class AsyncPool:
    def __init__(self, connection: RecordingAsyncConnection) -> None:
        self.value = connection
        self.connections = 0

    @asynccontextmanager
    async def connection(self):
        self.connections += 1
        yield self.value


def source(
    *,
    source_id: UUID = SOURCE_ID,
    root_id: str = "document_data_1",
    path: str = "papers/material.pdf",
    doi: str | None = None,
    digest: str | None = None,
    metadata: dict[str, object] | None = None,
    canonical_source_id: UUID | None = None,
) -> SourceCatalogRecord:
    return SourceCatalogRecord(
        source_id=source_id,
        locator=SourceLocator(root_id=root_id, relative_path=path),
        source_kind="literature",
        display_title="材料证据",
        normalized_doi=doi,
        sha256=digest,
        byte_size=128,
        metadata=metadata or {},
        canonical_source_id=canonical_source_id,
    )


def source_row(record: SourceCatalogRecord, *, version: str = SOURCE_VERSION) -> dict[str, object]:
    metadata = dict(record.metadata)
    metadata.setdefault("source_version_key", version)
    return {
        "source_id": record.source_id,
        "root_id": record.locator.root_id,
        "relative_path": record.locator.relative_path,
        "source_version_key": version,
        "source_kind": record.source_kind,
        "display_title": record.display_title,
        "status": record.status,
        "directory_year": record.directory_year,
        "normalized_doi": record.normalized_doi,
        "application_number": record.application_number,
        "publication_number": record.publication_number,
        "grant_number": record.grant_number,
        "legal_status": record.legal_status,
        "sha256": record.sha256,
        "byte_size": record.byte_size,
        "material_category": record.material_category,
        "knowledge_domain": record.knowledge_domain,
        "locator": record.locator.model_dump(mode="json"),
        "metadata": metadata,
        "canonical_source_id": record.canonical_source_id,
    }


def checkpoint(
    *,
    source_id: UUID = SOURCE_ID,
    stage: IngestionStage = IngestionStage.SELECT,
    job_status: IngestionJobStatus = IngestionJobStatus.RUNNING,
    lifecycle: SourceLifecycleStatus = SourceLifecycleStatus.PARSE_ELIGIBLE,
    attempt: int = 0,
    generation: str = GENERATION,
    fingerprint: str = VERSION_FINGERPRINT,
    selection: SelectionDecision | None = None,
) -> ProcessingCheckpoint:
    key = build_ingestion_idempotency_key(
        source_id,
        source_version_key=SOURCE_VERSION,
        embedding_generation_id=GENERATION,
    )
    resolved_selection = selection
    if resolved_selection is None:
        resolved_selection = SelectionDecision(
            source_id=source_id,
            selected=True,
            reason_code="approved_curation",
            policy_version="policy-v1",
        )
    return ProcessingCheckpoint(
        source_id=source_id,
        lifecycle_status=lifecycle,
        stage=stage,
        job_status=job_status,
        attempt=attempt,
        idempotency_key=key,
        selection=resolved_selection,
        cursor={"offset": 1},
        metadata={
            "source_version_fingerprint": fingerprint,
            "embedding_generation_id": generation,
            "root_id": "document_data_1",
        },
    )


def checkpoint_row(value: ProcessingCheckpoint) -> dict[str, object]:
    return {
        "idempotency_key": value.idempotency_key,
        "source_id": value.source_id,
        "source_version_fingerprint": value.metadata["source_version_fingerprint"],
        "embedding_generation_id": value.metadata["embedding_generation_id"],
        "lifecycle_status": value.lifecycle_status,
        "stage": value.stage,
        "job_status": value.job_status,
        "attempt": value.attempt,
        "selection": None if value.selection is None else value.selection.model_dump(mode="json"),
        "cursor": value.cursor,
        "metadata": value.metadata,
        "last_error_category": value.last_error_category,
    }


def ingestion_key(source_id: UUID = SOURCE_ID) -> str:
    return build_ingestion_idempotency_key(
        source_id,
        source_version_key=SOURCE_VERSION,
        embedding_generation_id=GENERATION,
    )


def fragment(*, text: str = "Tg 为 315 °C。") -> EvidenceFragment:
    locator = SourceLocator(
        root_id="document_data_1",
        relative_path="papers/material.pdf",
        page=3,
        section="Results",
        block_index=4,
    )
    value = EvidenceFragment(
        source_id=SOURCE_ID,
        text=text,
        locator=locator,
        retention_reason="measured_property",
        supported_entity_ids=["material:pi"],
        supported_relation_ids=["relation:tg"],
        parser_name="mineru",
        parser_version="v4",
        embedding_generation_id=GENERATION,
        metadata={"block_type": "text"},
    )
    identity = "|".join(
        (
            ingestion_key(),
            value.content_sha256 or "",
            "3",
            "4",
            "Results",
        )
    )
    return value.model_copy(update={"fragment_id": uuid5(NAMESPACE_URL, identity)})


def evidence_row(value: EvidenceFragment) -> dict[str, object]:
    return {
        "fragment_id": value.fragment_id,
        "source_id": value.source_id,
        "idempotency_key": ingestion_key(),
        "text": value.text,
        "locator": value.locator.model_dump(mode="json"),
        "content_sha256": value.content_sha256,
        "retention_reason": value.retention_reason,
        "supported_entity_ids": value.supported_entity_ids,
        "supported_relation_ids": value.supported_relation_ids,
        "parser_name": value.parser_name,
        "parser_version": value.parser_version,
        "embedding_generation_id": value.embedding_generation_id,
        "metadata": value.metadata,
    }


def mapping(value: EvidenceFragment | None = None) -> LightRAGSourceMapping:
    return LightRAGSourceMapping.from_fragment(value or fragment())


def mapping_row(value: LightRAGSourceMapping) -> dict[str, object]:
    return value.model_dump(mode="python")


def test_migration_has_independent_version_and_exact_identity_constraints() -> None:
    sql = (ROOT / "migrations" / "knowledge_0001.sql").read_text(encoding="utf-8")
    assert "VALUES ('knowledge_0001')" in sql
    for table in (
        "knowledge_sources",
        "knowledge_source_relations",
        "knowledge_ingestion_checkpoints",
        "knowledge_evidence_fragments",
        "knowledge_lightrag_source_mappings",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "UNIQUE (root_id, relative_path)" in sql
    assert "knowledge_sources_sha_dedupe_idx" in sql
    assert "knowledge_sources_doi_sha_version_idx" in sql
    assert "UNIQUE (source_id, source_version_fingerprint, embedding_generation_id)" in sql
    assert "fragment_id uuid PRIMARY KEY" in sql
    assert "basename text PRIMARY KEY" in sql
    assert "UNIQUE (sha256)" not in sql
    assert "SET LOCAL search_path = public" in sql
    assert "char_length(text) <= 65536" in sql


def test_migration_is_idempotent_reversible_and_down_is_not_forward_executable() -> None:
    forward = (ROOT / "migrations" / "knowledge_0001.sql").read_text(encoding="utf-8")
    down = (ROOT / "migrations" / "knowledge_0001.down.sql").read_text(encoding="utf-8")
    assert "ON CONFLICT (version) DO NOTHING" in forward
    assert "DROP TABLE" not in forward
    assert "Never run automatically" in down
    assert down.index("knowledge_lightrag_source_mappings") < down.index("knowledge_sources")
    assert "DELETE FROM schema_migrations WHERE version = 'knowledge_0001'" in down


def test_migration_enforces_structured_json_and_evidence_only_storage() -> None:
    sql = (ROOT / "migrations" / "knowledge_0001.sql").read_text(encoding="utf-8")
    assert sql.count("jsonb_typeof") >= 7
    assert "octet_length(metadata::text) <= 262144" in sql
    assert "left(ltrim(text), 5) <> '%PDF-'" in sql
    assert "FOREIGN KEY (idempotency_key, source_id)" in sql
    assert "FOREIGN KEY (fragment_id, source_id, content_sha256" in sql
    assert "ON DELETE CASCADE" not in sql.split("knowledge_ingestion_checkpoints", 1)[1]


@pytest.mark.parametrize(
    ("dsn", "minimum", "maximum"),
    [("", 1, 2), ("postgresql://db/test", -1, 2), ("postgresql://db/test", 3, 2)],
)
def test_pool_factories_reject_invalid_settings(dsn: str, minimum: int, maximum: int) -> None:
    with pytest.raises(ValueError):
        pg.create_psycopg_sync_pool(dsn, min_size=minimum, max_size=maximum)


def test_pool_factories_fail_with_safe_dependency_message_when_driver_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.setitem(sys.modules, "psycopg_pool", None)

    with pytest.raises(RuntimeError, match=r"psycopg\[binary,pool\]") as error:
        pg.create_psycopg_async_pool("postgresql://db/test")
    assert "postgresql://" not in str(error.value)


def test_pool_factories_use_closed_psycopg_mapping_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = types.ModuleType("psycopg.rows")
    rows.dict_row = object()  # type: ignore[attr-defined]
    psycopg = types.ModuleType("psycopg")
    psycopg.__path__ = []  # type: ignore[attr-defined]
    pool_module = types.ModuleType("psycopg_pool")

    class FakeSyncPool:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeAsyncPool(FakeSyncPool):
        pass

    pool_module.ConnectionPool = FakeSyncPool  # type: ignore[attr-defined]
    pool_module.AsyncConnectionPool = FakeAsyncPool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows)
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_module)

    sync_pool = pg.create_psycopg_sync_pool("postgresql://db/test", max_size=3)
    async_pool = pg.create_psycopg_async_pool("postgresql://db/test", max_size=7)
    assert sync_pool.kwargs["open"] is False  # type: ignore[attr-defined]
    assert sync_pool.kwargs["max_size"] == 3  # type: ignore[attr-defined]
    assert async_pool.kwargs["open"] is False  # type: ignore[attr-defined]
    assert async_pool.kwargs["max_size"] == 7  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "value"},
        {"complete_mineru_output": {}},
        {"transport": {}},
        {"did": "device"},
        {"exception": RuntimeError("failure")},
        {"value": RuntimeError("failure")},
        {"value": b"binary"},
        {"value": float("nan")},
        {"value": "%PDF-1.7"},
        {"value": "Bearer " + "x" * 20},
        {1: "not-a-string-key"},
        {"": "empty-key"},
    ],
)
def test_json_boundary_rejects_raw_sensitive_or_unstructured_values(
    payload: dict[object, object],
) -> None:
    with pytest.raises(pg.UnsafeDurablePayload) as error:
        pg._validated_json_object(payload, field="test")
    assert "Bearer" not in str(error.value)


def test_json_boundary_rejects_depth_size_and_non_object_values() -> None:
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(18):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    with pytest.raises(pg.UnsafeDurablePayload, match="nesting"):
        pg._validated_json_object(nested, field="test")
    with pytest.raises(pg.UnsafeDurablePayload, match="size"):
        pg._validated_json_object({"text": "x" * 262_145}, field="test")
    with pytest.raises(pg.UnsafeDurablePayload, match="JSON object"):
        pg._validated_json_object([], field="test")


def test_json_parameters_are_canonical_and_stored_rows_fail_closed() -> None:
    assert pg._json_parameter({"b": 2, "a": [True, None]}, field="test") == (
        '{"a":[true,null],"b":2}'
    )
    assert pg._json_object_from_row('{"safe":1}', field="test") == {"safe": 1}
    with pytest.raises(pg.KnowledgePersistenceError, match="invalid"):
        pg._json_object_from_row("{", field="test")
    with pytest.raises(pg.KnowledgePersistenceError, match="violates"):
        pg._json_object_from_row({"session": "forbidden"}, field="test")
    with pytest.raises(pg.KnowledgePersistenceError, match="row_factory"):
        pg._row(("tuple",))
    with pytest.raises(pg.KnowledgePersistenceError, match="stored demo"):
        pg._model(lambda _: 1 / 0, {}, label="demo")


def test_catalog_new_source_uses_transaction_and_explicit_path_conflict() -> None:
    record = source()
    version = build_source_version_key(
        locator=record.locator,
        byte_size=record.byte_size,
        remote_modified_at="2026-07-01T00:00:00Z",
    )
    durable = record.model_copy(
        update={
            "metadata": {
                "source_version_key": version,
                "remote_modified_at": "2026-07-01T00:00:00+00:00",
            }
        }
    )
    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [])
    script.add("INSERT INTO knowledge_sources", [source_row(durable, version=version)])
    script.add("WHERE source_id = %s", [source_row(durable, version=version)])
    connection = RecordingSyncConnection(script)
    repository = pg.PostgresSourceCatalogRepository(SyncPool(connection))

    result = repository.upsert(record, remote_modified_at="2026-07-01T00:00:00Z")

    assert result.created is True
    assert result.source_version_key == version
    assert result.record.metadata["remote_modified_at"] == "2026-07-01T00:00:00+00:00"
    assert connection.transactions == 1
    insert = next(
        item for item in connection.statements if "INSERT INTO knowledge_sources" in item.sql
    )
    assert "ON CONFLICT (root_id, relative_path) DO UPDATE" in insert.sql
    assert all("password" not in str(value).casefold() for value in insert.params)
    assert repr(repository) == "PostgresSourceCatalogRepository()"


def test_catalog_existing_path_merges_typed_metadata_without_changing_source_id() -> None:
    existing = source(metadata={"authors": ["A"]})
    incoming = source(source_id=UUID(int=99), metadata={"year": 2025})
    version = build_source_version_key(
        locator=incoming.locator,
        byte_size=incoming.byte_size,
        remote_modified_at=None,
    )
    merged = existing.model_copy(
        update={"metadata": {"authors": ["A"], "year": 2025, "source_version_key": version}}
    )
    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [source_row(existing)])
    script.add("INSERT INTO knowledge_sources", [source_row(merged, version=version)])
    script.add("WHERE source_id = %s", [source_row(merged, version=version)])
    repository = pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(script)))

    result = repository.upsert(incoming)

    assert result.created is False
    assert result.record.source_id == existing.source_id
    assert result.record.metadata["authors"] == ["A"]
    assert result.record.metadata["year"] == 2025


def test_catalog_reconciles_sha_canonical_and_doi_versions_in_one_transaction() -> None:
    digest = "a" * 64
    doi = "10.1000/material"
    incoming = source(doi="DOI:10.1000/Material", digest=digest)
    version = build_source_version_key(
        locator=incoming.locator, byte_size=incoming.byte_size, remote_modified_at=None
    )
    durable = incoming.model_copy(
        update={
            "normalized_doi": doi,
            "metadata": {"source_version_key": version},
        }
    )
    preferred_duplicate = source(
        source_id=OTHER_SOURCE_ID,
        path="papers/material-copy.pdf",
        doi=doi,
        digest=digest,
        metadata={"authors": ["A"], "source_version_key": SOURCE_VERSION},
    )
    other_version = source(
        source_id=UUID(int=3),
        path="papers/material-v2.pdf",
        doi=doi,
        digest="b" * 64,
        metadata={"authors": ["A"], "abstract": "rich", "source_version_key": SOURCE_VERSION},
    )
    refreshed = durable.model_copy(update={"canonical_source_id": OTHER_SOURCE_ID})
    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [])
    script.add("INSERT INTO knowledge_sources", [source_row(durable, version=version)])
    script.add(
        "WHERE sha256 = %s ORDER BY source_id FOR UPDATE",
        [source_row(durable, version=version), source_row(preferred_duplicate)],
    )
    script.add(
        "FROM knowledge_sources AS member",
        [source_row(preferred_duplicate), source_row(other_version)],
    )
    script.add("WHERE source_id = %s", [source_row(refreshed, version=version)])
    connection = RecordingSyncConnection(script)
    repository = pg.PostgresSourceCatalogRepository(SyncPool(connection))

    result = repository.upsert(incoming)

    assert result.record.canonical_source_id == OTHER_SOURCE_ID
    sql = "\n".join(item.sql for item in connection.statements)
    assert "SET canonical_source_id = CASE" in sql
    assert "INSERT INTO knowledge_source_relations" in sql
    assert "ON CONFLICT (relation_type, source_id, target_source_id)" in sql
    assert sum("pg_advisory_xact_lock" in item.sql for item in connection.statements) == 3


def test_catalog_process_data_is_excluded_without_body_or_transport_state() -> None:
    record = source(root_id="data_2", path="process_data/mineru/output.json")
    version = build_source_version_key(
        locator=record.locator, byte_size=record.byte_size, remote_modified_at=None
    )
    durable = record.model_copy(
        update={
            "status": "excluded_process_data",
            "metadata": {
                "source_version_key": version,
                "exclusion_reason": "process_data_never_open",
            },
        }
    )
    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [])
    script.add("INSERT INTO knowledge_sources", [source_row(durable, version=version)])
    script.add("WHERE source_id = %s", [source_row(durable, version=version)])
    connection = RecordingSyncConnection(script)

    result = pg.PostgresSourceCatalogRepository(SyncPool(connection)).upsert(record)

    assert result.record.status == "excluded_process_data"
    flattened = " ".join(str(value) for item in connection.statements for value in item.params)
    for forbidden in ("transport", "session", "quickconnect_did", "synotoken"):
        assert forbidden not in flattened.casefold()


@pytest.mark.parametrize(
    "bad_record",
    [
        source(metadata={"transport": {"sid": "value"}}),
        SourceCatalogRecord(
            locator=SourceLocator(root_id="document_data_1", relative_path="x.pdf"),
            display_title="x",
            unexpected="value",
        ),
    ],
)
def test_catalog_rejects_unsafe_or_untyped_metadata_before_database_io(
    bad_record: SourceCatalogRecord,
) -> None:
    pool = SyncPool(RecordingSyncConnection(Script()))
    with pytest.raises(pg.UnsafeDurablePayload):
        pg.PostgresSourceCatalogRepository(pool).upsert(bad_record)
    assert pool.connections == 0


@pytest.mark.parametrize(
    "metadata",
    [
        {"complete_mineru_output": {}},
        {"blob": b"binary"},
        {"token": "value"},
        {"session": {"sid": "value"}},
        {"did": "value"},
    ],
)
def test_catalog_sensitive_metadata_never_reaches_sql_parameters(
    metadata: dict[str, object],
) -> None:
    pool = SyncPool(RecordingSyncConnection(Script()))
    with pytest.raises(pg.UnsafeDurablePayload):
        pg.PostgresSourceCatalogRepository(pool).upsert(source(metadata=metadata))
    assert pool.connections == 0
    assert pool.value.statements == []


def test_catalog_credential_like_typed_field_is_rejected_before_io() -> None:
    pool = SyncPool(RecordingSyncConnection(Script()))
    record = source().model_copy(update={"display_title": "Bearer " + "x" * 20})
    with pytest.raises(pg.UnsafeDurablePayload, match="credential-like"):
        pg.PostgresSourceCatalogRepository(pool).upsert(record)
    assert pool.connections == 0


def test_catalog_rejects_wrong_type_and_missing_returning_rows() -> None:
    repository = pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(Script())))
    with pytest.raises(TypeError):
        repository.upsert(object())  # type: ignore[arg-type]

    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [])
    script.add("INSERT INTO knowledge_sources", [])
    with pytest.raises(pg.KnowledgePersistenceError, match="upsert returned"):
        pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(script))).upsert(
            source()
        )


def test_catalog_detects_disappearing_row_and_empty_reconciliation_groups() -> None:
    digest = "c" * 64
    record = source(digest=digest)
    version = build_source_version_key(
        locator=record.locator, byte_size=record.byte_size, remote_modified_at=None
    )
    durable = record.model_copy(update={"metadata": {"source_version_key": version}})
    script = Script()
    script.add("WHERE root_id = %s AND relative_path = %s", [])
    script.add("INSERT INTO knowledge_sources", [source_row(durable, version=version)])
    script.add("WHERE sha256 = %s ORDER BY source_id FOR UPDATE", [])
    script.add("WHERE source_id = %s", [])
    with pytest.raises(pg.KnowledgePersistenceError, match="disappeared"):
        pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(script))).upsert(record)


def test_catalog_get_canonical_and_relations_reads() -> None:
    canonical = source(source_id=OTHER_SOURCE_ID, path="canonical.pdf")
    duplicate = source(canonical_source_id=OTHER_SOURCE_ID)
    relation = {
        "relation_type": "IS_VERSION_OF",
        "source_id": SOURCE_ID,
        "target_source_id": OTHER_SOURCE_ID,
        "normalized_doi": "10.1000/a",
        "reason": "same_normalized_doi_with_different_sha256",
    }
    script = Script()
    script.add(
        "WHERE source_id = %s",
        [source_row(duplicate)],
        [source_row(duplicate)],
        [source_row(canonical)],
    )
    script.add("FROM knowledge_source_relations", [relation], [relation])
    repository = pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(script)))

    assert repository.get(SOURCE_ID).source_id == SOURCE_ID
    assert repository.canonical_for(SOURCE_ID).source_id == OTHER_SOURCE_ID
    assert repository.relations()[0].normalized_doi == "10.1000/a"
    assert repository.relations("IS_VERSION_OF")[0].target_source_id == OTHER_SOURCE_ID
    with pytest.raises(ValueError, match="relation type"):
        repository.relations("BAD")  # type: ignore[arg-type]


def test_catalog_get_unknown_and_invalid_relation_row_fail_closed() -> None:
    repository = pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(Script())))
    with pytest.raises(KeyError, match="unknown source"):
        repository.get(uuid4())
    script = Script()
    script.add("FROM knowledge_source_relations", [{"relation_type": "bad"}])
    with pytest.raises(pg.KnowledgePersistenceError, match="source relation"):
        pg.PostgresSourceCatalogRepository(SyncPool(RecordingSyncConnection(script))).relations()


@pytest.mark.asyncio
async def test_checkpoint_save_and_load_use_transactional_monotonic_upsert() -> None:
    value = checkpoint()
    script = Script()
    script.add("WHERE idempotency_key = %s FOR UPDATE", [])
    script.add(
        "INSERT INTO knowledge_ingestion_checkpoints", [{"idempotency_key": value.idempotency_key}]
    )
    connection = RecordingAsyncConnection(script)
    repository = pg.PostgresCheckpointRepository(AsyncPool(connection))

    await repository.save(value)

    assert connection.transactions == 1
    insert = next(
        item
        for item in connection.statements
        if "INSERT INTO knowledge_ingestion_checkpoints" in item.sql
    )
    assert "ON CONFLICT (idempotency_key) DO UPDATE" in insert.sql
    assert "knowledge_ingestion_checkpoints.stage" in insert.sql
    assert "knowledge_ingestion_checkpoints.attempt <= EXCLUDED.attempt" in insert.sql
    assert VERSION_FINGERPRINT in insert.params
    assert repr(repository) == "PostgresCheckpointRepository()"

    load_script = Script()
    load_script.add("WHERE idempotency_key = %s", [checkpoint_row(value)])
    loaded = await pg.PostgresCheckpointRepository(
        AsyncPool(RecordingAsyncConnection(load_script))
    ).load(value.idempotency_key)
    assert loaded == value


@pytest.mark.asyncio
async def test_checkpoint_load_none_and_blank_key() -> None:
    repository = pg.PostgresCheckpointRepository(AsyncPool(RecordingAsyncConnection(Script())))
    assert await repository.load("missing") is None
    with pytest.raises(ValueError, match="required"):
        await repository.load(" ")


@pytest.mark.asyncio
async def test_checkpoint_forward_progress_is_accepted() -> None:
    old = checkpoint(stage=IngestionStage.SELECT)
    new = checkpoint(stage=IngestionStage.SPOOL)
    script = Script()
    script.add("WHERE idempotency_key = %s FOR UPDATE", [checkpoint_row(old)])
    script.add(
        "INSERT INTO knowledge_ingestion_checkpoints", [{"idempotency_key": new.idempotency_key}]
    )
    connection = RecordingAsyncConnection(script)

    await pg.PostgresCheckpointRepository(AsyncPool(connection)).save(new)

    assert connection.transactions == 1


@pytest.mark.parametrize(
    ("existing", "candidate", "message"),
    [
        (checkpoint(), checkpoint(source_id=OTHER_SOURCE_ID), "source identity"),
        (
            checkpoint(),
            checkpoint().model_copy(
                update={
                    "metadata": {
                        **checkpoint().metadata,
                        "embedding_generation_id": "other-generation",
                    }
                }
            ),
            "ingestion identity",
        ),
        (
            checkpoint(stage=IngestionStage.PARSE),
            checkpoint(stage=IngestionStage.SPOOL),
            "stage regression",
        ),
        (checkpoint(attempt=2), checkpoint(attempt=1), "attempt regression"),
        (
            checkpoint(
                stage=IngestionStage.INDEX,
                job_status=IngestionJobStatus.SUCCEEDED,
                lifecycle=SourceLifecycleStatus.EVIDENCE_RETAINED,
            ),
            checkpoint(
                stage=IngestionStage.INDEX,
                job_status=IngestionJobStatus.RUNNING,
                lifecycle=SourceLifecycleStatus.EVIDENCE_RETAINED,
            ),
            "terminal checkpoint job",
        ),
        (
            checkpoint(lifecycle=SourceLifecycleStatus.PARSED_NO_VALUE),
            checkpoint(lifecycle=SourceLifecycleStatus.FAILED_PERMANENT),
            "terminal checkpoint lifecycle",
        ),
        (
            checkpoint(),
            checkpoint(
                selection=SelectionDecision(
                    source_id=SOURCE_ID,
                    selected=True,
                    reason_code="task_semantic_match",
                    policy_version="policy-v1",
                )
            ),
            "selection is immutable",
        ),
    ],
)
def test_checkpoint_progress_guard_rejects_conflicts_and_regressions(
    existing: ProcessingCheckpoint,
    candidate: ProcessingCheckpoint,
    message: str,
) -> None:
    with pytest.raises(pg.KnowledgeRecordConflict, match=message):
        pg._assert_checkpoint_progress(existing, candidate)


@pytest.mark.asyncio
async def test_checkpoint_database_guard_rejection_is_reported_as_stale() -> None:
    value = checkpoint()
    script = Script()
    script.add("WHERE idempotency_key = %s FOR UPDATE", [])
    script.add("INSERT INTO knowledge_ingestion_checkpoints", [])
    with pytest.raises(pg.CheckpointRegressionError, match="rejected as stale"):
        await pg.PostgresCheckpointRepository(AsyncPool(RecordingAsyncConnection(script))).save(
            value
        )


@pytest.mark.asyncio
async def test_checkpoint_repository_rejects_invalid_contract_before_io() -> None:
    repository = pg.PostgresCheckpointRepository(AsyncPool(RecordingAsyncConnection(Script())))
    with pytest.raises(TypeError):
        await repository.save(object())  # type: ignore[arg-type]
    with pytest.raises(pg.UnsafeDurablePayload, match="idempotency"):
        await repository.save(checkpoint().model_copy(update={"idempotency_key": "wrong"}))
    with pytest.raises(pg.UnsafeDurablePayload, match="fingerprint"):
        await repository.save(checkpoint(fingerprint="wrong"))
    with pytest.raises(pg.UnsafeDurablePayload, match="generation is invalid"):
        await repository.save(checkpoint(generation=" "))
    with pytest.raises(pg.UnsafeDurablePayload, match="credential-like"):
        await repository.save(checkpoint(generation="Bearer " + "x" * 20))
    unsafe_error = checkpoint().model_copy(update={"last_error_category": "Bearer " + "x" * 20})
    with pytest.raises(pg.UnsafeDurablePayload, match="credential-like"):
        await repository.save(unsafe_error)


@pytest.mark.asyncio
async def test_checkpoint_stored_invalid_json_or_model_fails_closed() -> None:
    value = checkpoint()
    invalid_json = checkpoint_row(value)
    invalid_json["metadata"] = "{"
    script = Script()
    script.add("WHERE idempotency_key = %s", [invalid_json])
    with pytest.raises(pg.KnowledgePersistenceError, match="JSON"):
        await pg.PostgresCheckpointRepository(AsyncPool(RecordingAsyncConnection(script))).load(
            value.idempotency_key
        )

    invalid_model = checkpoint_row(value)
    invalid_model["stage"] = "unknown"
    script = Script()
    script.add("WHERE idempotency_key = %s", [invalid_model])
    with pytest.raises(pg.KnowledgePersistenceError, match="stored checkpoint"):
        await pg.PostgresCheckpointRepository(AsyncPool(RecordingAsyncConnection(script))).load(
            value.idempotency_key
        )


@pytest.mark.asyncio
async def test_evidence_persist_is_transactional_and_first_write_wins() -> None:
    first = fragment()
    retry = first.model_copy(update={"metadata": {"assessment_confidence": 0.9}})
    script = Script()
    script.add("INSERT INTO knowledge_evidence_fragments", [{"fragment_id": first.fragment_id}])
    connection = RecordingAsyncConnection(script)
    repository = pg.PostgresEvidenceRepository(AsyncPool(connection))

    await repository.persist_many(
        SOURCE_ID,
        [first, retry],
        idempotency_key=ingestion_key(),
    )

    assert connection.transactions == 1
    inserts = [
        item
        for item in connection.statements
        if "INSERT INTO knowledge_evidence_fragments" in item.sql
    ]
    assert len(inserts) == 1
    assert "ON CONFLICT (fragment_id) DO UPDATE" in inserts[0].sql
    assert "metadata" not in inserts[0].sql.split("WHERE", 1)[1]
    assert json.loads(inserts[0].params[-1]) == {"block_type": "text"}
    assert repr(repository) == "PostgresEvidenceRepository()"


@pytest.mark.asyncio
async def test_evidence_empty_batch_does_not_open_database_connection() -> None:
    pool = AsyncPool(RecordingAsyncConnection(Script()))
    await pg.PostgresEvidenceRepository(pool).persist_many(
        SOURCE_ID, [], idempotency_key=ingestion_key()
    )
    assert pool.connections == 0


@pytest.mark.asyncio
async def test_evidence_database_identity_conflict_is_atomic() -> None:
    value = fragment()
    script = Script()
    script.add("INSERT INTO knowledge_evidence_fragments", [])
    connection = RecordingAsyncConnection(script)
    with pytest.raises(pg.KnowledgeRecordConflict, match="durable evidence"):
        await pg.PostgresEvidenceRepository(AsyncPool(connection)).persist_many(
            SOURCE_ID, [value], idempotency_key=ingestion_key()
        )
    assert connection.transactions == 1


@pytest.mark.asyncio
async def test_evidence_list_reconstructs_only_validated_fragments() -> None:
    value = fragment()
    script = Script()
    script.add("FROM knowledge_evidence_fragments", [evidence_row(value)])
    repository = pg.PostgresEvidenceRepository(AsyncPool(RecordingAsyncConnection(script)))

    assert await repository.list_for_source(SOURCE_ID, idempotency_key=ingestion_key()) == [value]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["bad_key_type", "bad_key_source", "bad_fragment_type", "bad_source", "random_id"],
)
async def test_evidence_rejects_invalid_repository_identity(case: str) -> None:
    repository = pg.PostgresEvidenceRepository(AsyncPool(RecordingAsyncConnection(Script())))
    value: Any = fragment()
    source_id = SOURCE_ID
    key: Any = ingestion_key()
    if case == "bad_key_type":
        key = 1
    elif case == "bad_key_source":
        key = ingestion_key(OTHER_SOURCE_ID)
    elif case == "bad_fragment_type":
        value = object()
    elif case == "bad_source":
        source_id = OTHER_SOURCE_ID
        key = ingestion_key(OTHER_SOURCE_ID)
    elif case == "random_id":
        value = value.model_copy(update={"fragment_id": uuid4()})
    with pytest.raises((TypeError, ValueError)):
        await repository.persist_many(source_id, [value], idempotency_key=key)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["pdf", "credential", "metadata", "oversized", "field"])
async def test_evidence_rejects_raw_pdf_credentials_and_parser_payloads(case: str) -> None:
    value = fragment()
    if case == "pdf":
        value = fragment(text="%PDF-1.7 raw body")
    elif case == "credential":
        value = fragment(text="Bearer " + "x" * 20)
    elif case == "metadata":
        value = value.model_copy(update={"metadata": {"complete_parser_output": {}}})
    elif case == "oversized":
        value = fragment(text="x" * 65_537)
    else:
        value = value.model_copy(update={"parser_version": "Bearer " + "x" * 20})
    pool = AsyncPool(RecordingAsyncConnection(Script()))
    with pytest.raises(pg.UnsafeDurablePayload):
        await pg.PostgresEvidenceRepository(pool).persist_many(
            SOURCE_ID, [value], idempotency_key=ingestion_key()
        )
    assert pool.connections == 0


@pytest.mark.asyncio
async def test_evidence_duplicate_fragment_with_different_parser_is_rejected() -> None:
    first = fragment()
    conflicting = first.model_copy(update={"parser_version": "v5"})
    with pytest.raises(pg.KnowledgeRecordConflict, match="fragment identity"):
        await pg.PostgresEvidenceRepository(
            AsyncPool(RecordingAsyncConnection(Script()))
        ).persist_many(SOURCE_ID, [first, conflicting], idempotency_key=ingestion_key())


@pytest.mark.asyncio
async def test_evidence_invalid_stored_row_fails_closed() -> None:
    value = evidence_row(fragment())
    value["metadata"] = {"source_bytes": "forbidden"}
    script = Script()
    script.add("FROM knowledge_evidence_fragments", [value])
    with pytest.raises(pg.KnowledgePersistenceError, match="durable JSON"):
        await pg.PostgresEvidenceRepository(
            AsyncPool(RecordingAsyncConnection(script))
        ).list_for_source(SOURCE_ID, idempotency_key=ingestion_key())


@pytest.mark.asyncio
async def test_mapping_persist_is_transactional_and_explicitly_idempotent() -> None:
    value = mapping()
    script = Script()
    script.add("INSERT INTO knowledge_lightrag_source_mappings", [{"basename": value.basename}])
    connection = RecordingAsyncConnection(script)
    repository = pg.PostgresLightRAGSourceMappingStore(AsyncPool(connection))

    await repository.persist_many([value])

    assert connection.transactions == 1
    insert = connection.statements[0]
    assert "ON CONFLICT DO NOTHING" in insert.sql
    assert insert.params[0] == value.basename
    assert repr(repository) == "PostgresLightRAGSourceMappingStore()"


@pytest.mark.asyncio
async def test_mapping_existing_identical_provenance_is_idempotent() -> None:
    value = mapping()
    script = Script()
    script.add("INSERT INTO knowledge_lightrag_source_mappings", [])
    script.add("WHERE basename = %s OR fragment_id = %s", [mapping_row(value)])
    await pg.PostgresLightRAGSourceMappingStore(
        AsyncPool(RecordingAsyncConnection(script))
    ).persist_many([value])


@pytest.mark.asyncio
async def test_mapping_conflicting_basename_or_fragment_is_rejected() -> None:
    value = mapping()
    conflicting = value.model_copy(update={"embedding_generation_id": "other-generation"})
    with pytest.raises(LightRAGSourceMappingConflict, match="duplicate basename"):
        await pg.PostgresLightRAGSourceMappingStore(
            AsyncPool(RecordingAsyncConnection(Script()))
        ).persist_many([value, conflicting])

    script = Script()
    script.add("INSERT INTO knowledge_lightrag_source_mappings", [])
    script.add("WHERE basename = %s OR fragment_id = %s", [mapping_row(conflicting)])
    with pytest.raises(LightRAGSourceMappingConflict, match="bound differently"):
        await pg.PostgresLightRAGSourceMappingStore(
            AsyncPool(RecordingAsyncConnection(script))
        ).persist_many([value])


@pytest.mark.asyncio
async def test_mapping_empty_wrong_type_and_invalid_copy_are_rejected_before_io() -> None:
    pool = AsyncPool(RecordingAsyncConnection(Script()))
    repository = pg.PostgresLightRAGSourceMappingStore(pool)
    await repository.persist_many([])
    assert pool.connections == 0
    with pytest.raises(TypeError):
        await repository.persist_many([object()])  # type: ignore[list-item]
    invalid = mapping().model_copy(update={"logical_source_uri": "https://private"})
    with pytest.raises(ValueError):
        await repository.persist_many([invalid])
    unsafe = mapping().model_copy(update={"embedding_generation_id": "Bearer " + "x" * 20})
    with pytest.raises(pg.UnsafeDurablePayload, match="credential-like"):
        await repository.persist_many([unsafe])


@pytest.mark.asyncio
async def test_mapping_get_found_none_blank_and_invalid_storage() -> None:
    value = mapping()
    script = Script()
    script.add("WHERE basename = %s", [mapping_row(value)], [])
    repository = pg.PostgresLightRAGSourceMappingStore(AsyncPool(RecordingAsyncConnection(script)))
    assert await repository.get(value.basename) == value
    assert await repository.get("mg_" + "0" * 32 + "_" + "0" * 32 + "_" + "0" * 16 + ".txt") is None
    with pytest.raises(ValueError, match="basename"):
        await repository.get(" ")

    invalid = mapping_row(value)
    invalid["locator"] = "{"
    script = Script()
    script.add("WHERE basename = %s", [invalid])
    with pytest.raises(pg.KnowledgePersistenceError, match="JSON"):
        await pg.PostgresLightRAGSourceMappingStore(
            AsyncPool(RecordingAsyncConnection(script))
        ).get(value.basename)
