from __future__ import annotations

import base64
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pytest

from material_graph.knowledge.bindings import EmbeddingBinding
from material_graph.knowledge.textbook_embedding_bundle import (
    TextbookEmbeddingArchiveSettings,
    build_textbook_embedding_archive,
)
from material_graph.knowledge.textbook_precomputed_lightrag import (
    PrecomputedEmbeddingLookup,
    TextbookPrecomputedImportError,
    TextbookPrecomputedImportSettings,
    _ChunkStore,
    _RelationshipRepairMaterial,
    _canonical_relationship_id,
    _repair_postgres_relationship_vectors,
    _relationship_collision_plan,
    _relationship_replacements,
    _rewrite_relationship_vdb,
    _safe_relationship_id,
    import_textbook_precomputed_lightrag,
)
from material_graph.knowledge.lightrag_runtime import workspace_for_generation


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "config/knowledge/embedding-binding.v1.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _deployment_bundle(
    tmp_path: Path,
    binding: EmbeddingBinding,
    *,
    chunk_contents: list[str] | None = None,
    entity_name: str = "PET",
    entity_description: str = "聚酯",
    relationship_description: str = "牵伸促使PET形成取向结构",
) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    chunks = [
        {
            "content": content,
            "source_id": f"fragment-{index}",
            "file_path": f"mg_fragment_{index}.txt",
            "chunk_order_index": index - 1,
        }
        for index, content in enumerate(
            chunk_contents or ["PET 经牵伸形成取向结构。"],
            start=1,
        )
    ]
    entities = [
        {
            "entity_name": entity_name,
            "entity_type": "Material",
            "description": entity_description,
            "source_id": "fragment-1",
            "file_path": "mg_fragment_1.txt",
        }
    ]
    relationships = [
        {
            "src_id": entity_name,
            "tgt_id": "取向结构",
            "description": relationship_description,
            "keywords": "材料科学,教材知识,Material-Structure",
            "weight": 1.0,
            "source_id": "fragment-1",
            "file_path": "mg_fragment_1.txt",
        }
    ]
    artifacts = {
        "custom_kg_chunks": bundle / "custom-kg-chunks.jsonl",
        "custom_kg_entities": bundle / "custom-kg-entities.jsonl",
        "custom_kg_relationships": bundle / "custom-kg-relationships.jsonl",
        "embedding_binding": bundle / "embedding-binding.json",
    }
    _write_jsonl(artifacts["custom_kg_chunks"], chunks)
    _write_jsonl(artifacts["custom_kg_entities"], entities)
    _write_jsonl(artifacts["custom_kg_relationships"], relationships)
    artifacts["embedding_binding"].write_text(
        binding.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "embedding": {
            "generation_id": binding.generation_id,
            "provider": binding.provider,
            "model": binding.model,
            "dimensions": binding.dimensions,
        },
        "counts": {
            "custom_kg_chunks": len(chunks),
            "custom_kg_entities": len(entities),
            "custom_kg_relationships": len(relationships),
        },
        "artifacts": {
            name: {
                "path": path.name,
                "sha256": _digest(path),
                "bytes": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


async def _archive(
    tmp_path: Path,
    binding: EmbeddingBinding,
    bundle: Path,
) -> Path:
    archive = tmp_path / "archive"

    async def embedder(
        active_binding: EmbeddingBinding,
        api_key: str,
        texts: list[str],
    ) -> np.ndarray:
        values = np.zeros((len(texts), binding.dimensions), dtype=np.float32)
        for index in range(len(texts)):
            values[index, index % binding.dimensions] = 1
        return values

    await build_textbook_embedding_archive(
        TextbookEmbeddingArchiveSettings(
            bundle_dir=bundle,
            output_dir=archive,
        ),
        binding,
        "test-secret",
        embedder=embedder,
    )
    return archive


@pytest.mark.asyncio
async def test_precomputed_lookup_returns_rows_and_rejects_query_context(
    tmp_path: Path,
) -> None:
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _deployment_bundle(tmp_path, binding)
    archive = await _archive(tmp_path, binding, bundle)

    with PrecomputedEmbeddingLookup.open(archive) as lookup:
        values = await lookup(
            ["PET 经牵伸形成取向结构。", "PET\n聚酯"],
            context="document",
        )
        assert values.shape == (2, binding.dimensions)
        assert values.dtype == np.float32
        with pytest.raises(TextbookPrecomputedImportError, match="query"):
            await lookup(["PET"], context="query")
        with pytest.raises(TextbookPrecomputedImportError, match="missing"):
            await lookup(["archive does not contain this"], context="document")


class _FakeRag:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.initialized = False
        self.finalized = False

    async def initialize_storages(self) -> None:
        self.initialized = True

    async def finalize_storages(self) -> None:
        self.finalized = True

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str | None = None,
    ) -> None:
        self.calls.append(custom_kg)


@pytest.mark.asyncio
async def test_precomputed_import_uses_bounded_source_complete_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "material_graph.knowledge.textbook_precomputed_lightrag._lightrag_text_sanitizers",
        lambda: (lambda value: value, lambda value: value),
    )
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _deployment_bundle(tmp_path, binding)
    archive = await _archive(tmp_path, binding, bundle)
    rag = _FakeRag()
    factory_calls: list[tuple[str, str]] = []

    def factory(
        active_binding: EmbeddingBinding,
        workspace: str,
        lookup: PrecomputedEmbeddingLookup,
        working_dir: Path,
    ) -> _FakeRag:
        factory_calls.append((active_binding.generation_id, workspace))
        return rag

    settings = TextbookPrecomputedImportSettings(
        bundle_dir=bundle,
        archive_dir=archive,
        working_dir=tmp_path / "working",
        state_path=tmp_path / "import-state.json",
        batch_size=1,
    )
    summary = await import_textbook_precomputed_lightrag(
        settings,
        rag_factory=factory,
    )
    calls_after_first_run = len(rag.calls)
    second = await import_textbook_precomputed_lightrag(
        settings,
        rag_factory=factory,
    )

    assert summary.status == second.status == "completed"
    assert summary.chunks == summary.entities == summary.relationships == 1
    assert rag.initialized and rag.finalized
    assert len(rag.calls) == calls_after_first_run == 3
    assert factory_calls == [
        (
            binding.generation_id,
            workspace_for_generation(binding.generation_id),
        )
    ]
    for call in rag.calls:
        assert len(call["chunks"]) <= 1
        assert len(call["entities"]) <= 1
        assert len(call["relationships"]) <= 1
        if call["entities"]:
            assert {row["source_id"] for row in call["entities"]} <= {
                row["source_id"] for row in call["chunks"]
            }
        if call["relationships"]:
            assert {row["source_id"] for row in call["relationships"]} <= {
                row["source_id"] for row in call["chunks"]
            }


@pytest.mark.asyncio
async def test_default_importer_replays_archive_into_local_lightrag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lightrag")
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _deployment_bundle(
        tmp_path,
        binding,
        chunk_contents=["  PET &amp; PA6\x01  "],
        entity_name="\x7fPET",
        entity_description="聚\x0c酯",
        relationship_description="牵伸促使PET\x08形成取向结构",
    )
    archive = await _archive(tmp_path, binding, bundle)
    for name, value in {
        "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
        "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
        "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
    }.items():
        monkeypatch.setenv(name, value)
    working_dir = tmp_path / "working"
    state_path = tmp_path / "state.json"

    summary = await import_textbook_precomputed_lightrag(
        TextbookPrecomputedImportSettings(
            bundle_dir=bundle,
            archive_dir=archive,
            working_dir=working_dir,
            state_path=state_path,
            batch_size=1,
        )
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert summary.status == state["status"] == "completed"
    assert state["phase"] == "completed"
    assert any(path.stat().st_size > 0 for path in working_dir.iterdir())


@pytest.mark.asyncio
async def test_precomputed_import_rejects_sanitized_chunk_hash_collision(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lightrag")
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _deployment_bundle(
        tmp_path,
        binding,
        chunk_contents=["PET &amp; PA6", "PET & PA6"],
    )
    archive = await _archive(tmp_path, binding, bundle)

    with pytest.raises(TextbookPrecomputedImportError, match="alias collision"):
        await import_textbook_precomputed_lightrag(
            TextbookPrecomputedImportSettings(
                bundle_dir=bundle,
                archive_dir=archive,
                working_dir=tmp_path / "working",
                state_path=tmp_path / "state.json",
                batch_size=1,
            ),
            rag_factory=lambda *args: _FakeRag(),
        )


def _relationship(src_id: str, tgt_id: str) -> dict[str, Any]:
    vector = base64.b64encode(
        zlib.compress(np.asarray([1.0, 0.5], dtype=np.float16).tobytes())
    ).decode("ascii")
    return {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "source_id": "fragment-1",
        "content": f"keywords\t{src_id}\n{tgt_id}\ndescription",
        "file_path": "mg_fragment_1.txt",
        "vector": vector,
    }


def test_relationship_collision_plan_detects_canonical_concat_collision() -> None:
    rows = [_relationship("a", "bc"), _relationship("ab", "c")]

    plan = _relationship_collision_plan(rows)

    collision_id = _canonical_relationship_id("a", "bc")
    assert collision_id == _canonical_relationship_id("ab", "c")
    assert plan.concat_collision_ids == frozenset({collision_id})
    assert plan.reverse_delete_collision_ids == frozenset()
    assert plan.dangerous_canonical_ids == frozenset({collision_id})
    assert _safe_relationship_id("a", "bc") != _safe_relationship_id("ab", "c")


def test_relationship_collision_plan_detects_reverse_legacy_delete_collision() -> None:
    rows = [_relationship("d", "ef"), _relationship("e", "fd")]

    plan = _relationship_collision_plan(rows)

    target_id = _canonical_relationship_id("e", "fd")
    assert plan.concat_collision_ids == frozenset()
    assert plan.reverse_delete_collision_ids == frozenset({target_id})
    assert plan.dangerous_canonical_ids == frozenset({target_id})


def test_relationship_vdb_rewrite_is_collision_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    rows = [
        _relationship("a", "bc"),
        _relationship("ab", "c"),
        _relationship("d", "ef"),
        _relationship("e", "fd"),
    ]
    plan = _relationship_collision_plan(rows)
    collision_id = _canonical_relationship_id("a", "bc")
    unaffected_id = _canonical_relationship_id("d", "ef")
    vdb_path = tmp_path / "vdb_relationships.json"
    vdb_path.write_text(
        json.dumps(
            {
                "embedding_dim": 2,
                "data": [
                    {"__id__": collision_id, **rows[0]},
                    {"__id__": unaffected_id, **rows[2]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    replacements = [
        {"__id__": _safe_relationship_id(row["src_id"], row["tgt_id"]), **row}
        for row in rows
        if _canonical_relationship_id(row["src_id"], row["tgt_id"]) in plan.dangerous_canonical_ids
    ]

    first = _rewrite_relationship_vdb(
        vdb_path,
        dimensions=2,
        expected_count=4,
        dangerous_ids=plan.dangerous_canonical_ids,
        replacements=replacements,
    )
    second = _rewrite_relationship_vdb(
        vdb_path,
        dimensions=2,
        expected_count=4,
        dangerous_ids=plan.dangerous_canonical_ids,
        replacements=replacements,
    )
    payload = json.loads(vdb_path.read_text(encoding="utf-8"))
    ids = [row["__id__"] for row in payload["data"]]
    matrix = np.frombuffer(base64.b64decode(payload["matrix"]), dtype=np.float32)
    matrix = matrix.reshape(-1, 2)
    nano_vectordb = pytest.importorskip("nano_vectordb")
    runtime_vdb = nano_vectordb.NanoVectorDB(
        2,
        storage_file=str(vdb_path),
    )

    assert first.deleted_dangerous_count == 1
    assert first.inserted_safe_count == 3
    assert second.deleted_dangerous_count == 0
    assert second.inserted_safe_count == 0
    assert len(ids) == len(set(ids)) == 4
    assert matrix.shape == (4, 2)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert len(runtime_vdb) == 4
    assert runtime_vdb._NanoVectorDB__storage["matrix"].shape == (4, 2)
    assert unaffected_id in ids
    assert not plan.dangerous_canonical_ids.intersection(ids)


class _FakePGTransaction:
    def __init__(self, connection: "_FakePGConnection", isolation: str) -> None:
        self.connection = connection
        self.isolation = isolation
        self.snapshot: dict[str, dict[str, Any]] = {}

    async def __aenter__(self) -> "_FakePGTransaction":
        self.connection.transaction_isolations.append(self.isolation)
        self.snapshot = {
            row_id: {
                **row,
                "chunk_ids": list(row.get("chunk_ids") or []),
                "content_vector": (
                    None
                    if row.get("content_vector") is None
                    else np.asarray(
                        row.get("content_vector"),
                        dtype=np.float32,
                    ).copy()
                ),
            }
            for row_id, row in self.connection.rows.items()
        }
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> bool:
        if exc_type is not None:
            self.connection.rows = self.snapshot
            self.connection.rollbacks += 1
        else:
            self.connection.commits += 1
        return False


class _FakePGConnection:
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.transaction_isolations: list[str] = []
        self.commits = 0
        self.rollbacks = 0
        self.advisory_locks: list[str] = []
        self.sql_statements: list[str] = []

    def transaction(self, *, isolation: str) -> _FakePGTransaction:
        return _FakePGTransaction(self, isolation)

    async def fetchval(self, sql: str, *arguments: Any) -> int | None:
        self.sql_statements.append(sql)
        if "pg_advisory_xact_lock" in sql:
            self.advisory_locks.append(str(arguments[0]))
            return None
        if "COUNT(*)" not in sql:
            raise AssertionError(f"unexpected scalar SQL: {sql}")
        if "id = ANY" in sql:
            requested = set(arguments[1])
            return sum(row_id in requested for row_id in self.rows)
        return len(self.rows)

    async def execute(self, sql: str, *arguments: Any) -> str:
        self.sql_statements.append(sql)
        if "LOCK TABLE" in sql:
            return "LOCK TABLE"
        if "DELETE FROM" not in sql:
            raise AssertionError(f"unexpected execute SQL: {sql}")
        requested = set(arguments[1])
        removed = 0
        for row_id in list(self.rows):
            if row_id in requested:
                self.rows.pop(row_id)
                removed += 1
        return f"DELETE {removed}"

    async def executemany(
        self,
        sql: str,
        values: list[tuple[Any, ...]],
    ) -> None:
        self.sql_statements.append(sql)
        assert "ON CONFLICT (workspace, id) DO UPDATE" in sql
        for (
            _workspace,
            row_id,
            source_id,
            target_id,
            content,
            vector,
            chunk_ids,
            file_path,
        ) in values:
            self.rows[row_id] = {
                "id": row_id,
                "source_id": source_id,
                "target_id": target_id,
                "content": content,
                "content_vector": np.asarray(vector, dtype=np.float32).copy(),
                "chunk_ids": list(chunk_ids),
                "file_path": file_path,
            }

    async def fetchrow(self, sql: str, *arguments: Any) -> dict[str, int]:
        self.sql_statements.append(sql)
        assert "COUNT(DISTINCT (source_id, target_id))" in sql
        dimensions = int(arguments[1])
        return {
            "total_count": len(self.rows),
            "distinct_id_count": len(set(self.rows)),
            "distinct_endpoint_count": len(
                {(str(row["source_id"]), str(row["target_id"])) for row in self.rows.values()}
            ),
            "null_vector_count": sum(
                row.get("content_vector") is None for row in self.rows.values()
            ),
            "wrong_dimension_count": sum(
                row.get("content_vector") is not None
                and np.asarray(row["content_vector"]).size != dimensions
                for row in self.rows.values()
            ),
        }

    async def fetch(self, sql: str, *arguments: Any) -> list[dict[str, Any]]:
        self.sql_statements.append(sql)
        assert "content_vector IS NOT NULL AS has_vector" in sql
        requested = set(arguments[1])
        return [
            {
                **row,
                "has_vector": row.get("content_vector") is not None,
                "vector_dimensions": (
                    np.asarray(row["content_vector"]).size
                    if row.get("content_vector") is not None
                    else None
                ),
            }
            for row_id, row in self.rows.items()
            if row_id in requested
        ]


class _FakePGDatabase:
    def __init__(self, connection: _FakePGConnection) -> None:
        self.connection = connection
        self.retry_labels: list[str] = []

    async def _run_with_retry(self, callback: Any, *, timing_label: str) -> None:
        self.retry_labels.append(timing_label)
        await callback(self.connection)


class _FakePGStorage:
    def __init__(
        self,
        connection: _FakePGConnection,
        *,
        table_name: str = "LIGHTRAG_VDB_RELATION_embedding_3_2d",
        workspace: str = "glm-embedding-3-1024-halfvec-v1",
    ) -> None:
        self.table_name = table_name
        self.workspace = workspace
        self.db = _FakePGDatabase(connection)
        self.flush_calls = 0

    async def index_done_callback(self) -> None:
        self.flush_calls += 1


def _postgres_repair_fixture() -> tuple[
    _RelationshipRepairMaterial,
    _FakePGConnection,
]:
    rows = [
        _relationship("a", "bc"),
        _relationship("ab", "c"),
        _relationship("d", "ef"),
        _relationship("e", "fd"),
    ]
    plan = _relationship_collision_plan(rows)
    replacements = tuple(
        {
            "__id__": _safe_relationship_id(row["src_id"], row["tgt_id"]),
            **row,
        }
        for row in rows
        if _canonical_relationship_id(row["src_id"], row["tgt_id"]) in plan.dangerous_canonical_ids
    )
    collision_id = _canonical_relationship_id("a", "bc")
    unaffected_id = _canonical_relationship_id("d", "ef")
    initial = {
        collision_id: {
            "id": collision_id,
            "source_id": "old",
            "target_id": "old",
            "content": "old",
            "content_vector": np.asarray([1.0, 0.0], dtype=np.float32),
            "chunk_ids": ["old"],
            "file_path": "old",
        },
        unaffected_id: {
            "id": unaffected_id,
            "source_id": "d",
            "target_id": "ef",
            "content": "unaffected",
            "content_vector": np.asarray([1.0, 0.0], dtype=np.float32),
            "chunk_ids": ["fragment-1"],
            "file_path": "mg_fragment_1.txt",
        },
    }
    return (
        _RelationshipRepairMaterial(
            plan=plan,
            affected_count=len(replacements),
            replacements=replacements,
        ),
        _FakePGConnection(initial),
    )


@pytest.mark.asyncio
async def test_postgres_relationship_repair_is_transactional_and_idempotent() -> None:
    material, connection = _postgres_repair_fixture()
    storage = _FakePGStorage(connection)

    first = await _repair_postgres_relationship_vectors(
        storage,
        generation_id="glm-embedding-3-1024-halfvec-v1",
        model="embedding-3",
        dimensions=2,
        expected_count=4,
        material=material,
    )
    second = await _repair_postgres_relationship_vectors(
        storage,
        generation_id="glm-embedding-3-1024-halfvec-v1",
        model="embedding-3",
        dimensions=2,
        expected_count=4,
        material=material,
    )

    assert first["backend"] == "PGVectorStorage"
    assert first["table_name"] == "LIGHTRAG_VDB_RELATION_embedding_3_2d"
    assert first["rows_before"] == 2
    assert first["rows_after"] == 4
    assert first["deleted_dangerous_count"] == 1
    assert first["upserted_safe_count"] == first["safe_rows_verified"] == 3
    assert first["graph_mutated"] is False
    assert second["rows_before"] == second["rows_after"] == 4
    assert second["deleted_dangerous_count"] == 0
    assert len(connection.rows) == 4
    assert not material.plan.dangerous_canonical_ids.intersection(connection.rows)
    repaired_id = _safe_relationship_id("a", "bc")
    assert np.allclose(
        connection.rows[repaired_id]["content_vector"],
        np.asarray([1.0, 0.5], dtype=np.float32),
    )
    assert connection.transaction_isolations == ["serializable", "serializable"]
    assert connection.commits == 2
    assert connection.rollbacks == 0
    assert len(connection.advisory_locks) == 2
    assert all(
        "LIGHTRAG_VDB_RELATION_embedding_3_2d" in sql
        for sql in connection.sql_statements
        if any(token in sql for token in ("LOCK TABLE", "FROM LIGHTRAG", "INTO LIGHTRAG"))
    )
    assert any("LOCK TABLE" in sql for sql in connection.sql_statements)


@pytest.mark.asyncio
async def test_postgres_relationship_repair_rolls_back_on_strict_count_failure() -> None:
    material, connection = _postgres_repair_fixture()
    before = set(connection.rows)

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="count mismatch",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(connection),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=5,
            material=material,
        )

    assert set(connection.rows) == before
    assert connection.commits == 0
    assert connection.rollbacks == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_vector",
    [
        None,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    ],
)
async def test_postgres_relationship_repair_rolls_back_on_invalid_stored_vector(
    invalid_vector: np.ndarray | None,
) -> None:
    material, connection = _postgres_repair_fixture()
    unaffected_id = _canonical_relationship_id("d", "ef")
    connection.rows[unaffected_id]["content_vector"] = invalid_vector
    before = set(connection.rows)

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="count mismatch",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(connection),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=4,
            material=material,
        )

    assert set(connection.rows) == before
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_postgres_relationship_repair_rolls_back_on_endpoint_collision() -> None:
    material, connection = _postgres_repair_fixture()
    unaffected_id = _canonical_relationship_id("d", "ef")
    connection.rows[unaffected_id]["source_id"] = "a"
    connection.rows[unaffected_id]["target_id"] = "bc"
    before = set(connection.rows)

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="count mismatch",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(connection),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=4,
            material=material,
        )

    assert set(connection.rows) == before
    assert connection.rollbacks == 1


@pytest.mark.asyncio
async def test_postgres_relationship_repair_rejects_nan_before_transaction() -> None:
    material, connection = _postgres_repair_fixture()
    invalid_records = [dict(row) for row in material.replacements]
    invalid_records[0]["vector"] = base64.b64encode(
        zlib.compress(np.asarray([np.nan, 0.5], dtype=np.float16).tobytes())
    ).decode("ascii")

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="matrix is invalid",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(connection),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=4,
            material=_RelationshipRepairMaterial(
                plan=material.plan,
                affected_count=material.affected_count,
                replacements=tuple(invalid_records),
            ),
        )

    assert connection.transaction_isolations == []


@pytest.mark.asyncio
async def test_relationship_replacement_alias_miss_fails_before_pg_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunk_path,
        [
            {
                "content": "source evidence",
                "source_id": "fragment-1",
                "file_path": "mg_fragment_1.txt",
                "chunk_order_index": 0,
            }
        ],
    )
    raw = {
        "src_id": "a",
        "tgt_id": "bc",
        "description": "description",
        "keywords": "keywords",
        "source_id": "fragment-1",
        "file_path": "mg_fragment_1.txt",
    }

    class _MissingLookup:
        def __enter__(self) -> "_MissingLookup":
            return self

        def __exit__(self, *args: Any) -> None:
            del args

        async def __call__(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            del texts, kwargs
            raise TextbookPrecomputedImportError(
                "precomputed embedding archive is missing requested content"
            )

    monkeypatch.setattr(
        PrecomputedEmbeddingLookup,
        "open",
        lambda *args, **kwargs: _MissingLookup(),
    )
    chunk_store = _ChunkStore(chunk_path)
    try:
        with pytest.raises(
            TextbookPrecomputedImportError,
            match="missing requested content",
        ):
            await _relationship_replacements(
                [(raw, raw)],
                archive_dir=tmp_path,
                chunk_store=chunk_store,
                dimensions=2,
                sanitize_chunk_text=lambda value: value,
                sanitize_graph_text=lambda value: value,
            )
    finally:
        chunk_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table_name",
    [
        "LIGHTRAG_VDB_RELATION",
        "LIGHTRAG_VDB_RELATION_embedding_3_1024d",
        "LIGHTRAG_VDB_RELATION; DROP TABLE material_graph",
    ],
)
async def test_postgres_relationship_repair_rejects_untrusted_table_name(
    table_name: str,
) -> None:
    material, connection = _postgres_repair_fixture()

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="storage is unavailable",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(
                connection,
                table_name=table_name,
            ),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=4,
            material=material,
        )

    assert connection.transaction_isolations == []


@pytest.mark.asyncio
async def test_postgres_relationship_repair_rejects_wrong_workspace() -> None:
    material, connection = _postgres_repair_fixture()

    with pytest.raises(
        TextbookPrecomputedImportError,
        match="storage is unavailable",
    ):
        await _repair_postgres_relationship_vectors(
            _FakePGStorage(connection, workspace="default"),
            generation_id="glm-embedding-3-1024-halfvec-v1",
            model="embedding-3",
            dimensions=2,
            expected_count=4,
            material=material,
        )

    assert connection.transaction_isolations == []


@pytest.mark.asyncio
async def test_postgres_relationship_repair_accepts_production_model_table() -> None:
    material, connection = _postgres_repair_fixture()
    for row in connection.rows.values():
        vector = np.zeros(1024, dtype=np.float32)
        vector[0] = 1.0
        row["content_vector"] = vector
    material_without_collisions = _RelationshipRepairMaterial(
        plan=_relationship_collision_plan([]),
        affected_count=0,
        replacements=(),
    )

    summary = await _repair_postgres_relationship_vectors(
        _FakePGStorage(
            connection,
            table_name="LIGHTRAG_VDB_RELATION_embedding_3_1024d",
        ),
        generation_id="glm-embedding-3-1024-halfvec-v1",
        model="embedding-3",
        dimensions=1024,
        expected_count=2,
        material=material_without_collisions,
    )

    assert summary["table_name"] == "LIGHTRAG_VDB_RELATION_embedding_3_1024d"
    assert summary["workspace"] == "glm-embedding-3-1024-halfvec-v1"
    assert summary["rows_after"] == 2
    assert summary["generation_id"] == "glm-embedding-3-1024-halfvec-v1"
    assert summary["dimensions"] == 1024
    assert material.affected_count == 3


@pytest.mark.asyncio
async def test_completed_state_without_local_vdb_runs_postgres_repair_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "material_graph.knowledge.textbook_precomputed_lightrag._lightrag_text_sanitizers",
        lambda: (lambda value: value, lambda value: value),
    )
    binding = EmbeddingBinding.model_validate_json(PRIMARY.read_text(encoding="utf-8"))
    bundle = _deployment_bundle(tmp_path, binding)
    archive = await _archive(tmp_path, binding, bundle)
    settings = TextbookPrecomputedImportSettings(
        bundle_dir=bundle,
        archive_dir=archive,
        working_dir=tmp_path / "server-working",
        state_path=tmp_path / "server-state.json",
        batch_size=1,
    )
    monkeypatch.delenv("LIGHTRAG_VECTOR_STORAGE", raising=False)
    await import_textbook_precomputed_lightrag(
        settings,
        rag_factory=lambda *args: _FakeRag(),
    )
    old_state = json.loads(settings.state_path.read_text(encoding="utf-8"))
    assert old_state["phase"] == "completed"
    assert "relationship_vdb_repair" not in old_state
    assert not list(settings.working_dir.rglob("vdb_relationships.json"))

    relation_id = _canonical_relationship_id("PET", "取向结构")
    vector = np.zeros(binding.dimensions, dtype=np.float32)
    vector[0] = 1.0
    connection = _FakePGConnection(
        {
            relation_id: {
                "id": relation_id,
                "source_id": "PET",
                "target_id": "取向结构",
                "content": "existing precomputed relationship",
                "content_vector": vector,
                "chunk_ids": ["chunk-existing"],
                "file_path": "mg_fragment_1.txt",
            }
        }
    )
    storage = _FakePGStorage(
        connection,
        table_name="LIGHTRAG_VDB_RELATION_embedding_3_1024d",
    )
    provider_calls = 0

    class _CompletedPGRag:
        def __init__(self) -> None:
            self.relationships_vdb = storage
            self.initialized = 0
            self.finalized = 0
            self.insert_calls = 0

        async def initialize_storages(self) -> None:
            self.initialized += 1

        async def finalize_storages(self) -> None:
            self.finalized += 1

        async def ainsert_custom_kg(
            self,
            custom_kg: dict[str, Any],
            full_doc_id: str | None = None,
        ) -> None:
            del custom_kg, full_doc_id
            self.insert_calls += 1

    rag = _CompletedPGRag()

    def pg_factory(*args: Any) -> _CompletedPGRag:
        nonlocal provider_calls
        del args
        provider_calls += 0
        return rag

    monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "PGVectorStorage")
    summary = await import_textbook_precomputed_lightrag(
        settings,
        rag_factory=pg_factory,
    )
    repaired_state = json.loads(settings.state_path.read_text(encoding="utf-8"))

    assert summary.status == "completed"
    assert repaired_state["relationship_vdb_repair"]["backend"] == "PGVectorStorage"
    assert repaired_state["relationship_vdb_repair"]["rows_after"] == 1
    assert repaired_state["relationship_vdb_repair"]["graph_mutated"] is False
    assert rag.initialized == rag.finalized == 1
    assert rag.insert_calls == 0
    assert storage.flush_calls == 1
    assert provider_calls == 0
    assert not list(settings.working_dir.rglob("vdb_relationships.json"))
