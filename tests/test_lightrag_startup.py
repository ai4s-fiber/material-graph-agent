from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

import pytest

from material_graph.knowledge.bindings import EmbeddingBinding
from material_graph.knowledge.lightrag_runtime import (
    LightRAGPostgresSnapshot,
    LightRAGRuntimeConfigurationError,
    LightRAGStartupValidationError,
)
from material_graph.knowledge.lightrag_startup import (
    load_runtime_contract,
    make_embedding_probe,
    make_postgres_probe,
    validate_live_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_BINDING = ROOT / "config/knowledge/embedding-binding.v1.json"
RERANKER_BINDING = ROOT / "config/knowledge/reranker-binding.v1.json"
VECTOR_TABLES = (
    "lightrag_vdb_chunks_embedding_3_1024d",
    "lightrag_vdb_entity_embedding_3_1024d",
    "lightrag_vdb_relation_embedding_3_1024d",
)


def _runtime_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for raw_line in (
        (ROOT / "deploy/config/ingestion-runtime.env").read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            environment[key] = value
    environment.update(
        {
            "POSTGRES_HOST": "postgres",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "material_graph",
            "POSTGRES_DATABASE": "material_graph",
            "POSTGRES_PASSWORD": "fixture-database-credential",
            "EMBEDDING_BINDING_API_KEY": "fixture-provider-credential",
        }
    )
    return environment


def _run(awaitable: Awaitable[Any]) -> Any:
    return asyncio.run(awaitable)


def test_static_runtime_contract_accepts_exact_release_binding() -> None:
    runtime = load_runtime_contract(
        _runtime_environment(),
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    assert runtime.embedding.model == "embedding-3"
    assert runtime.embedding.dimensions == 1024
    assert runtime.embedding.postgres_vector_index_type == "HNSW_HALFVEC"
    assert runtime.storage.workspace == "glm-embedding-3-1024-halfvec-v1"
    assert runtime.storage.postgres_workspace == runtime.storage.workspace
    assert runtime.storage.vector_storage == "PGVectorStorage"


def test_static_runtime_contract_sanitizes_unreadable_binding_files(tmp_path: Path) -> None:
    with pytest.raises(LightRAGRuntimeConfigurationError) as raised:
        load_runtime_contract(
            _runtime_environment(),
            embedding_path=tmp_path / "missing-embedding.json",
            reranker_path=tmp_path / "missing-reranker.json",
        )

    assert raised.value.code == "provider_bindings_invalid"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_DIM", "1536"),
        ("POSTGRES_VECTOR_INDEX_TYPE", "HNSW"),
        ("WORKSPACE", "shared-default"),
        ("POSTGRES_WORKSPACE", "glm_embedding_3_1024_halfvec_v1"),
        ("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        ("RERANK_MODEL", "different-reranker"),
    ],
)
def test_static_runtime_contract_rejects_binding_drift_without_echoing_value(
    name: str,
    value: str,
) -> None:
    environment = _runtime_environment()
    environment[name] = value

    with pytest.raises(LightRAGRuntimeConfigurationError) as raised:
        load_runtime_contract(
            environment,
            embedding_path=EMBEDDING_BINDING,
            reranker_path=RERANKER_BINDING,
        )

    assert raised.value.code in {
        "native_binding_mismatch",
        "runtime_contract_invalid",
        "storage_contract_invalid",
    }
    assert value not in str(raised.value)


class _CatalogConnection:
    def __init__(
        self,
        *,
        vector_format: str = "halfvec(1024)",
        index_method: str = "hnsw",
        operator_class: str = "halfvec_cosine_ops",
        index_present: bool = True,
        omit_last_schema: bool = False,
        unexpected_index: bool = False,
        pgvector_version: str | None = "0.8.0",
        current_schema: str | None = "material_graph",
        table_schema: str = "material_graph",
        duplicate_schema: str | None = None,
        workspace_present: bool = True,
        index_schema: str | None = None,
    ) -> None:
        self.vector_format = vector_format
        self.index_method = index_method
        self.operator_class = operator_class
        self.index_present = index_present
        self.omit_last_schema = omit_last_schema
        self.unexpected_index = unexpected_index
        self.pgvector_version = pgvector_version
        self.current_schema = current_schema
        self.table_schema = table_schema
        self.duplicate_schema = duplicate_schema
        self.workspace_present = workspace_present
        self.index_schema = index_schema or table_schema
        self.closed = False

    async def fetchval(self, query: str) -> str | None:
        if "pg_extension" in query:
            return self.pgvector_version
        assert "current_schema()" in query
        return self.current_schema

    async def fetch(self, query: str, table_names: list[str]) -> list[dict[str, object]]:
        assert tuple(table_names) == VECTOR_TABLES
        if "format_type" in query:
            rows = [
                {
                    "schema_name": self.table_schema,
                    "table_name": name,
                    "vector_format": self.vector_format,
                    "workspace_present": self.workspace_present,
                }
                for name in (table_names[:-1] if self.omit_last_schema else table_names)
            ]
            if self.duplicate_schema is not None:
                rows.extend(
                    {
                        "schema_name": self.duplicate_schema,
                        "table_name": name,
                        "vector_format": self.vector_format,
                        "workspace_present": self.workspace_present,
                    }
                    for name in table_names
                )
            return rows
        assert "pg_opclass" in query
        rows = [
            {
                "schema_name": self.index_schema,
                "table_name": name,
                "index_method": self.index_method,
                "operator_class": self.operator_class,
                "index_present": self.index_present,
            }
            for name in table_names
        ]
        if self.unexpected_index:
            rows.append({**rows[0], "table_name": "unexpected_table"})
        return rows

    async def close(self) -> None:
        self.closed = True


class _AsyncpgRecordLike:
    """Match asyncpg.Record's indexed access without registering as Mapping."""

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def __getitem__(self, name: str) -> object:
        return self._values[name]


class _AsyncpgRecordConnection(_CatalogConnection):
    async def fetch(
        self,
        query: str,
        table_names: list[str],
    ) -> list[_AsyncpgRecordLike]:
        rows = await super().fetch(query, table_names)
        return [_AsyncpgRecordLike(row) for row in rows]


def test_real_postgres_probe_checks_every_upstream_vector_table() -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )
    connection = _CatalogConnection()
    connect_arguments: dict[str, object] = {}

    async def connect(**kwargs: object) -> _CatalogConnection:
        connect_arguments.update(kwargs)
        return connection

    probe = make_postgres_probe(
        environment,
        runtime.embedding,
        connect=connect,
        suffix_resolver=lambda _: "embedding_3_1024d",
    )
    snapshot = _run(probe(runtime.storage.workspace))

    assert snapshot == LightRAGPostgresSnapshot(
        workspace=runtime.storage.workspace,
        pgvector_version="0.8.0",
        vector_type="halfvec",
        vector_dimensions=1024,
        operator_class="halfvec_cosine_ops",
        index_method="hnsw",
        index_present=True,
    )
    assert connect_arguments["host"] == "postgres"
    assert connect_arguments["password"] == environment["POSTGRES_PASSWORD"]
    assert connection.closed is True


def test_real_postgres_probe_accepts_asyncpg_record_index_access() -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )
    connection = _AsyncpgRecordConnection()

    async def connect(**_: object) -> _AsyncpgRecordConnection:
        return connection

    probe = make_postgres_probe(
        environment,
        runtime.embedding,
        connect=connect,
        suffix_resolver=lambda _: "embedding_3_1024d",
    )

    snapshot = _run(probe(runtime.storage.workspace))

    assert snapshot.workspace == runtime.storage.workspace
    assert snapshot.vector_dimensions == 1024
    assert snapshot.index_present is True
    assert connection.closed is True


@pytest.mark.parametrize(
    "connection",
    [
        _CatalogConnection(current_schema=None),
        _CatalogConnection(current_schema="public"),
        _CatalogConnection(duplicate_schema="public"),
        _CatalogConnection(workspace_present=False),
        _CatalogConnection(index_schema="public"),
    ],
)
def test_postgres_probe_rejects_wrong_or_ambiguous_schema_identity(
    connection: _CatalogConnection,
) -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    async def connect(**_: object) -> _CatalogConnection:
        return connection

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        return [0.1] * binding.dimensions

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(
            validate_live_runtime(
                runtime,
                environment,
                postgres_probe=make_postgres_probe(
                    environment,
                    runtime.embedding,
                    connect=connect,
                    suffix_resolver=lambda _: "embedding_3_1024d",
                ),
                embedding_probe=embedding_probe,
            )
        )

    assert raised.value.code == "postgres_probe_unavailable"
    assert connection.closed is True


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda environment: environment.pop("POSTGRES_PASSWORD"),
            "live_probe_configuration_missing",
        ),
        (
            lambda environment: environment.__setitem__("POSTGRES_PORT", "not-a-port"),
            "live_probe_configuration_invalid",
        ),
        (
            lambda environment: environment.__setitem__("POSTGRES_PORT", "70000"),
            "live_probe_configuration_invalid",
        ),
    ],
)
def test_postgres_probe_rejects_invalid_connection_configuration(
    mutate: Any,
    code: str,
) -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )
    mutate(environment)

    with pytest.raises(LightRAGRuntimeConfigurationError) as raised:
        make_postgres_probe(
            environment,
            runtime.embedding,
            connect=lambda **_: pytest.fail("connection must not be attempted"),
        )

    assert raised.value.code == code


def test_postgres_probe_rejects_unsafe_table_suffix() -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    with pytest.raises(LightRAGRuntimeConfigurationError) as raised:
        make_postgres_probe(
            environment,
            runtime.embedding,
            suffix_resolver=lambda _: "../unsafe",
        )

    assert raised.value.code == "vector_table_suffix_invalid"


@pytest.mark.parametrize(
    ("connection", "code"),
    [
        (_CatalogConnection(vector_format="vector(1024)"), "postgres_probe_unavailable"),
        (_CatalogConnection(vector_format="halfvec(1536)"), "postgres_probe_unavailable"),
        (_CatalogConnection(index_method="ivfflat"), "postgres_probe_unavailable"),
        (
            _CatalogConnection(operator_class="vector_cosine_ops"),
            "postgres_probe_unavailable",
        ),
        (_CatalogConnection(index_present=False), "postgres_probe_unavailable"),
        (_CatalogConnection(omit_last_schema=True), "postgres_probe_unavailable"),
        (_CatalogConnection(unexpected_index=True), "postgres_probe_unavailable"),
        (_CatalogConnection(pgvector_version=None), "postgres_probe_unavailable"),
    ],
)
def test_live_schema_drift_is_a_fail_closed_validator_error(
    connection: _CatalogConnection,
    code: str,
) -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    async def connect(**_: object) -> _CatalogConnection:
        return connection

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        return [0.1] * binding.dimensions

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(
            validate_live_runtime(
                runtime,
                environment,
                postgres_probe=make_postgres_probe(
                    environment,
                    runtime.embedding,
                    connect=connect,
                    suffix_resolver=lambda _: "embedding_3_1024d",
                ),
                embedding_probe=embedding_probe,
            )
        )

    assert raised.value.code == code


def test_official_embedding_probe_uses_frozen_model_dimension_and_query_prefix() -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )
    captured: dict[str, object] = {}

    async def call(**kwargs: object) -> list[list[float]]:
        captured.update(kwargs)
        return [[0.125] * runtime.embedding.dimensions]

    vector = _run(make_embedding_probe(environment, call=call)(runtime.embedding))

    assert len(vector) == 1024
    assert captured["model"] == runtime.embedding.model
    assert captured["embedding_dim"] == runtime.embedding.dimensions
    assert captured["base_url"] == runtime.embedding.base_url
    assert captured["api_key"] == environment["EMBEDDING_BINDING_API_KEY"]
    assert captured["context"] == "query"
    assert captured["query_prefix"] == runtime.embedding.query_prefix
    assert captured["document_prefix"] == ""


def test_embedding_probe_rejects_invalid_batch_shape() -> None:
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    async def call(**_: object) -> list[list[float]]:
        return []

    with pytest.raises(RuntimeError, match="invalid batch"):
        _run(make_embedding_probe(environment, call=call)(runtime.embedding))


def test_connection_failure_is_sanitized_and_blocks_live_validation() -> None:
    secret = "database-exception-sensitive-value"
    environment = _runtime_environment()
    runtime = load_runtime_contract(
        environment,
        embedding_path=EMBEDDING_BINDING,
        reranker_path=RERANKER_BINDING,
    )

    async def connect(**_: object) -> _CatalogConnection:
        raise RuntimeError(secret)

    async def embedding_probe(binding: EmbeddingBinding) -> list[float]:
        return [0.1] * binding.dimensions

    with pytest.raises(LightRAGStartupValidationError) as raised:
        _run(
            validate_live_runtime(
                runtime,
                environment,
                postgres_probe=make_postgres_probe(
                    environment,
                    runtime.embedding,
                    connect=connect,
                    suffix_resolver=lambda _: "embedding_3_1024d",
                ),
                embedding_probe=embedding_probe,
            )
        )

    assert raised.value.code == "postgres_probe_unavailable"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None
