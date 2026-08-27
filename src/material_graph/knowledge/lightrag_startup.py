"""Production probes that bind LightRAG startup to the frozen release contract."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .bindings import EmbeddingBinding, ProviderBindings
from .lightrag_runtime import (
    EmbeddingProbe,
    LightRAGPostgresSnapshot,
    LightRAGRuntimeConfig,
    LightRAGRuntimeConfigurationError,
    LightRAGStartupReport,
    LightRAGStartupValidator,
    PostgresProbe,
)


AsyncpgConnect = Callable[..., Awaitable[Any]]
EmbeddingCall = Callable[..., Awaitable[Any]]
SuffixResolver = Callable[[EmbeddingBinding], str]

_VECTOR_TABLE_BASES = (
    "lightrag_vdb_chunks",
    "lightrag_vdb_entity",
    "lightrag_vdb_relation",
)
_VECTOR_FORMAT = re.compile(r"^halfvec\((\d+)\)$", re.IGNORECASE)
_POSTGRES_VERSION_QUERY = "SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'vector'"
_CURRENT_SCHEMA_QUERY = "SELECT current_schema()"
_VECTOR_SCHEMA_QUERY = """
SELECT table_namespace.nspname AS schema_name,
       table_class.relname AS table_name,
       pg_catalog.format_type(vector_attribute.atttypid, vector_attribute.atttypmod)
           AS vector_format,
       workspace_attribute.attname IS NOT NULL AS workspace_present
FROM pg_catalog.pg_class AS table_class
JOIN pg_catalog.pg_namespace AS table_namespace
  ON table_namespace.oid = table_class.relnamespace
LEFT JOIN pg_catalog.pg_attribute AS vector_attribute
  ON vector_attribute.attrelid = table_class.oid
 AND vector_attribute.attname = 'content_vector'
 AND NOT vector_attribute.attisdropped
LEFT JOIN pg_catalog.pg_attribute AS workspace_attribute
  ON workspace_attribute.attrelid = table_class.oid
 AND workspace_attribute.attname = 'workspace'
 AND NOT workspace_attribute.attisdropped
WHERE table_class.relkind IN ('r', 'p')
  AND table_class.relname = ANY($1::text[])
"""
_VECTOR_INDEX_QUERY = """
SELECT table_namespace.nspname AS schema_name,
       table_class.relname AS table_name,
       access_method.amname AS index_method,
       operator_class.opcname AS operator_class,
       (index_catalog.indisvalid AND index_catalog.indisready AND index_catalog.indislive)
           AS index_present
FROM pg_catalog.pg_class AS table_class
JOIN pg_catalog.pg_namespace AS table_namespace
  ON table_namespace.oid = table_class.relnamespace
JOIN pg_catalog.pg_attribute AS vector_attribute
  ON vector_attribute.attrelid = table_class.oid
 AND vector_attribute.attname = 'content_vector'
 AND NOT vector_attribute.attisdropped
JOIN pg_catalog.pg_index AS index_catalog
  ON index_catalog.indrelid = table_class.oid
 AND vector_attribute.attnum = ANY(index_catalog.indkey::smallint[])
JOIN pg_catalog.pg_class AS index_class
  ON index_class.oid = index_catalog.indexrelid
JOIN pg_catalog.pg_am AS access_method
  ON access_method.oid = index_class.relam
JOIN pg_catalog.pg_opclass AS operator_class
  ON operator_class.oid = ANY(index_catalog.indclass::oid[])
WHERE table_class.relname = ANY($1::text[])
"""


def _format_number(value: float | int) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _expected_reranker_environment(bindings: ProviderBindings) -> dict[str, str]:
    reranker = bindings.reranker
    return {
        "MAX_ASYNC_RERANK": str(reranker.max_async),
        "MIN_RERANK_SCORE": str(reranker.minimum_score),
        "RERANK_BINDING": reranker.binding,
        "RERANK_BINDING_HOST": reranker.endpoint,
        "RERANK_MODEL": reranker.model,
        "RERANK_TIMEOUT": _format_number(reranker.timeout_seconds),
    }


def load_runtime_contract(
    environment: Mapping[str, str],
    *,
    embedding_path: str | Path,
    reranker_path: str | Path,
) -> LightRAGRuntimeConfig:
    """Load image-owned bindings and reject any native runtime drift."""

    try:
        bindings = ProviderBindings.load(
            embedding_path=embedding_path,
            reranker_path=reranker_path,
        )
    except Exception:
        raise LightRAGRuntimeConfigurationError("provider_bindings_invalid") from None

    runtime = LightRAGRuntimeConfig.from_environment(
        embedding=bindings.embedding,
        environment=environment,
    )
    expected_reranker = _expected_reranker_environment(bindings)
    if any(environment.get(name) != value for name, value in expected_reranker.items()):
        raise LightRAGRuntimeConfigurationError("native_binding_mismatch")
    return runtime


def _official_suffix(binding: EmbeddingBinding) -> str:
    """Reuse the pinned LightRAG table-suffix implementation."""

    from lightrag.base import BaseVectorStorage

    probe = SimpleNamespace(
        embedding_func=SimpleNamespace(
            model_name=binding.model,
            embedding_dim=binding.dimensions,
        )
    )
    suffix = BaseVectorStorage._generate_collection_suffix(probe)
    if not suffix or not re.fullmatch(r"[a-z0-9_]+", suffix):
        raise RuntimeError("invalid LightRAG vector table suffix")
    return suffix


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not value:
        raise LightRAGRuntimeConfigurationError("live_probe_configuration_missing")
    return value


def _postgres_connection_arguments(environment: Mapping[str, str]) -> dict[str, object]:
    try:
        port = int(_required(environment, "POSTGRES_PORT"))
    except ValueError:
        raise LightRAGRuntimeConfigurationError("live_probe_configuration_invalid") from None
    if not 1 <= port <= 65535:
        raise LightRAGRuntimeConfigurationError("live_probe_configuration_invalid")
    return {
        "host": _required(environment, "POSTGRES_HOST"),
        "port": port,
        "user": _required(environment, "POSTGRES_USER"),
        "password": _required(environment, "POSTGRES_PASSWORD"),
        "database": _required(environment, "POSTGRES_DATABASE"),
        "timeout": 5,
        "command_timeout": 10,
        "server_settings": {"application_name": "material_graph_lightrag_startup"},
    }


def _row_value(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError):
        return getattr(row, name)


def make_postgres_probe(
    environment: Mapping[str, str],
    binding: EmbeddingBinding,
    *,
    connect: AsyncpgConnect | None = None,
    suffix_resolver: SuffixResolver = _official_suffix,
) -> PostgresProbe:
    """Create a credential-safe pg_catalog probe for all LightRAG vector tables."""

    arguments = _postgres_connection_arguments(environment)
    suffix = suffix_resolver(binding)
    if not re.fullmatch(r"[a-z0-9_]+", suffix):
        raise LightRAGRuntimeConfigurationError("vector_table_suffix_invalid")
    table_names = tuple(f"{base}_{suffix}" for base in _VECTOR_TABLE_BASES)

    async def postgres_probe(workspace: str) -> LightRAGPostgresSnapshot:
        selected_connect = connect
        if selected_connect is None:
            import asyncpg

            selected_connect = asyncpg.connect

        connection = await selected_connect(**arguments)
        try:
            pgvector_version = await connection.fetchval(_POSTGRES_VERSION_QUERY)
            current_schema = await connection.fetchval(_CURRENT_SCHEMA_QUERY)
            schema_rows = await connection.fetch(_VECTOR_SCHEMA_QUERY, list(table_names))
            index_rows = await connection.fetch(_VECTOR_INDEX_QUERY, list(table_names))
        finally:
            await connection.close()

        if not current_schema:
            raise RuntimeError("PostgreSQL current schema is unavailable")
        schema_name = str(current_schema)
        expected_table_identities = {(schema_name, table_name) for table_name in table_names}
        schema_identities = {
            (
                str(_row_value(row, "schema_name")),
                str(_row_value(row, "table_name")),
            )
            for row in schema_rows
        }
        if (
            len(schema_rows) != len(table_names)
            or schema_identities != expected_table_identities
            or any(not bool(_row_value(row, "workspace_present")) for row in schema_rows)
        ):
            raise RuntimeError("LightRAG vector schema is incomplete")
        schema = {
            str(_row_value(row, "table_name")): str(_row_value(row, "vector_format"))
            for row in schema_rows
        }
        dimensions: set[int] = set()
        for table_name in table_names:
            match = _VECTOR_FORMAT.fullmatch(schema[table_name])
            if match is None:
                raise RuntimeError("LightRAG vector type does not match")
            dimensions.add(int(match.group(1)))
        if dimensions != {binding.dimensions}:
            raise RuntimeError("LightRAG vector dimensions do not match")

        indexes_by_table: dict[str, list[Any]] = {name: [] for name in table_names}
        for row in index_rows:
            row_schema = str(_row_value(row, "schema_name"))
            table_name = str(_row_value(row, "table_name"))
            if row_schema != schema_name or table_name not in indexes_by_table:
                raise RuntimeError("unexpected LightRAG vector index")
            indexes_by_table[table_name].append(row)
        for table_name in table_names:
            rows = indexes_by_table[table_name]
            if not rows or any(
                str(_row_value(row, "index_method")).casefold() != "hnsw"
                or str(_row_value(row, "operator_class")).casefold() != "halfvec_cosine_ops"
                or not bool(_row_value(row, "index_present"))
                for row in rows
            ):
                raise RuntimeError("LightRAG vector index does not match")

        if not pgvector_version:
            raise RuntimeError("pgvector extension is unavailable")
        return LightRAGPostgresSnapshot(
            workspace=workspace,
            pgvector_version=str(pgvector_version),
            vector_type="halfvec",
            vector_dimensions=binding.dimensions,
            operator_class="halfvec_cosine_ops",
            index_method="hnsw",
            index_present=True,
        )

    return postgres_probe


def make_embedding_probe(
    environment: Mapping[str, str],
    *,
    call: EmbeddingCall | None = None,
) -> EmbeddingProbe:
    """Create a canary using the same pinned upstream embedding implementation."""

    api_key = _required(environment, "EMBEDDING_BINDING_API_KEY")

    async def embedding_probe(binding: EmbeddingBinding) -> Sequence[float]:
        selected_call = call
        if selected_call is None:
            from lightrag.llm.openai import openai_embed

            selected_call = openai_embed.func
        result = await selected_call(
            texts=["material-science evidence retrieval startup canary"],
            model=binding.model,
            base_url=binding.base_url,
            api_key=api_key,
            embedding_dim=binding.dimensions if binding.send_dimensions else None,
            max_token_size=binding.max_input_tokens,
            client_configs={"timeout": binding.timeout_seconds},
            context="query",
            query_prefix=binding.query_prefix,
            document_prefix="",
        )
        if len(result) != 1:
            raise RuntimeError("embedding canary returned an invalid batch")
        return list(result[0])

    return embedding_probe


async def validate_live_runtime(
    runtime: LightRAGRuntimeConfig,
    environment: Mapping[str, str],
    *,
    postgres_probe: PostgresProbe | None = None,
    embedding_probe: EmbeddingProbe | None = None,
) -> LightRAGStartupReport:
    """Run the existing fail-closed validator against production dependencies."""

    validator = LightRAGStartupValidator(
        postgres_probe=postgres_probe or make_postgres_probe(environment, runtime.embedding),
        embedding_probe=embedding_probe or make_embedding_probe(environment),
    )
    return await validator.validate(runtime)
