"""Run pinned LightRAG PostgreSQL storage without runtime DDL authority.

LightRAG 1.5.4 normally creates and migrates tables, vector indexes and Apache
AGE graphs during every process start.  Production candidate structures are
instead prepared by a one-shot bootstrap job.  This launcher replaces only
those initialization hooks with fail-closed catalog validation; all ordinary
read/write storage methods remain the pinned upstream implementation.
"""

from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


LIGHTRAG_RUNTIME_ROLE = "material_graph_lightrag"
_SHARED_TABLES = (
    "lightrag_doc_full",
    "lightrag_doc_chunks",
    "lightrag_llm_cache",
    "lightrag_doc_status",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_entity_chunks",
    "lightrag_relation_chunks",
)
_VECTOR_TABLE_BASES = (
    "lightrag_vdb_chunks",
    "lightrag_vdb_entity",
    "lightrag_vdb_relation",
)
_AGE_TABLES = ("_ag_label_vertex", "_ag_label_edge", "base", "DIRECTED")
_GENERATION_BINDINGS = {
    "glm-embedding-3-1024-halfvec-v1": ("embedding-3", 1024),
    "qwen3-embedding-4b-2560-bf16-v1": ("qwen3-embedding-4b", 2560),
}
_SAFE_IDENTITY = re.compile(r"^[A-Za-z0-9_-]+$")
_PATCH_MARKER = "_material_graph_least_privilege_v1"


class LightRAGLeastPrivilegeError(RuntimeError):
    """The runtime identity or bootstrap-created storage contract is invalid."""


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise LightRAGLeastPrivilegeError("least_privilege_configuration_missing")
    return value


def _runtime_contract(environment: Mapping[str, str]) -> dict[str, object]:
    generation = _required(environment, "MATERIAL_GRAPH_LIGHTRAG_GENERATION_ID")
    try:
        expected_model, expected_dimensions = _GENERATION_BINDINGS[generation]
    except KeyError:
        raise LightRAGLeastPrivilegeError(
            "least_privilege_generation_not_allowlisted"
        ) from None
    workspace = _required(environment, "WORKSPACE")
    postgres_workspace = _required(environment, "POSTGRES_WORKSPACE")
    model = _required(environment, "EMBEDDING_MODEL")
    try:
        dimensions = int(_required(environment, "EMBEDDING_DIM"))
    except ValueError:
        raise LightRAGLeastPrivilegeError("least_privilege_binding_mismatch") from None
    if (
        workspace != generation
        or postgres_workspace != generation
        or model.casefold() != expected_model
        or dimensions != expected_dimensions
        or not _SAFE_IDENTITY.fullmatch(workspace)
    ):
        raise LightRAGLeastPrivilegeError("least_privilege_binding_mismatch")
    safe_workspace = re.sub(r"[^A-Za-z0-9_]", "_", workspace)
    safe_model = re.sub(r"[^a-z0-9_]", "_", model.casefold())
    vector_suffix = f"{safe_model}_{dimensions}d"
    return {
        "age_schema": f"{safe_workspace}_chunk_entity_relation",
        "dimensions": dimensions,
        "generation": generation,
        "material_graph_tables": _SHARED_TABLES
        + tuple(f"{base}_{vector_suffix}" for base in _VECTOR_TABLE_BASES),
        "vector_tables": tuple(f"{base}_{vector_suffix}" for base in _VECTOR_TABLE_BASES),
        "workspace": workspace,
    }


async def _validate_existing_tables(db: Any, contract: Mapping[str, object]) -> None:
    material_graph_tables = list(contract["material_graph_tables"])
    row = await db.query(
        """
        SELECT
          current_user = 'material_graph_lightrag' AS role_matches,
          NOT roles.rolsuper AS not_superuser,
          NOT roles.rolbypassrls AS cannot_bypass_rls,
          NOT roles.rolcreaterole AS cannot_create_role,
          NOT roles.rolcreatedb AS cannot_create_database,
          NOT roles.rolinherit AS cannot_inherit,
          current_setting('row_security') = 'on' AS row_security_on,
          NOT has_database_privilege(current_user, current_database(), 'CREATE')
              AS cannot_create_schema,
          to_regnamespace('material_graph') IS NOT NULL AS material_graph_exists,
          (
              SELECT count(*)
              FROM pg_class AS relations
              JOIN pg_namespace AS schemas ON schemas.oid = relations.relnamespace
              WHERE schemas.nspname = 'material_graph'
                AND relations.relkind IN ('r', 'p')
                AND relations.relname = ANY($1::text[])
                AND relations.relowner <> roles.oid
          ) = $2 AS tables_match
        FROM pg_roles AS roles
        WHERE roles.rolname = current_user
        """,
        [material_graph_tables, len(material_graph_tables)],
    )
    expected = {
        "role_matches",
        "not_superuser",
        "cannot_bypass_rls",
        "cannot_create_role",
        "cannot_create_database",
        "cannot_inherit",
        "row_security_on",
        "cannot_create_schema",
        "material_graph_exists",
        "tables_match",
    }
    if not isinstance(row, dict) or set(row) != expected or not all(row.values()):
        raise LightRAGLeastPrivilegeError("least_privilege_table_contract_invalid")


async def _validate_existing_vector_table(
    db: Any,
    table_name: str,
    workspace: str,
    embedding_dim: int,
    contract: Mapping[str, object],
) -> None:
    normalized_table = table_name.casefold()
    if (
        normalized_table not in contract["vector_tables"]
        or workspace != contract["workspace"]
        or embedding_dim != contract["dimensions"]
    ):
        raise LightRAGLeastPrivilegeError("least_privilege_vector_identity_invalid")
    row = await db.query(
        """
        SELECT
          pg_catalog.format_type(vector_attribute.atttypid, vector_attribute.atttypmod)
              = format('halfvec(%s)', $2::integer) AS dimensions_match,
          workspace_attribute.attname IS NOT NULL AS workspace_present,
          relation_owner.rolname <> current_user AS runtime_is_not_owner,
          EXISTS (
              SELECT 1
              FROM pg_index AS indexes
              JOIN pg_class AS index_class ON index_class.oid = indexes.indexrelid
              JOIN pg_am AS access_method ON access_method.oid = index_class.relam
              JOIN pg_opclass AS operator_class
                ON operator_class.oid = ANY(indexes.indclass::oid[])
              WHERE indexes.indrelid = table_class.oid
                AND vector_attribute.attnum = ANY(indexes.indkey::smallint[])
                AND access_method.amname = 'hnsw'
                AND operator_class.opcname = 'halfvec_cosine_ops'
                AND indexes.indisvalid
                AND indexes.indisready
                AND indexes.indislive
          ) AS index_present
        FROM pg_class AS table_class
        JOIN pg_namespace AS table_namespace
          ON table_namespace.oid = table_class.relnamespace
        JOIN pg_roles AS relation_owner ON relation_owner.oid = table_class.relowner
        LEFT JOIN pg_attribute AS vector_attribute
          ON vector_attribute.attrelid = table_class.oid
         AND vector_attribute.attname = 'content_vector'
         AND NOT vector_attribute.attisdropped
        LEFT JOIN pg_attribute AS workspace_attribute
          ON workspace_attribute.attrelid = table_class.oid
         AND workspace_attribute.attname = 'workspace'
         AND NOT workspace_attribute.attisdropped
        WHERE table_namespace.nspname = 'material_graph'
          AND table_class.relname = $1
          AND table_class.relkind IN ('r', 'p')
        """,
        [normalized_table, embedding_dim],
    )
    expected = {
        "dimensions_match",
        "workspace_present",
        "runtime_is_not_owner",
        "index_present",
    }
    if not isinstance(row, dict) or set(row) != expected or not all(row.values()):
        raise LightRAGLeastPrivilegeError("least_privilege_vector_contract_invalid")


async def _validate_existing_age_graph(
    connection: Any,
    graph_name: str,
    contract: Mapping[str, object],
) -> None:
    if graph_name != contract["age_schema"]:
        raise LightRAGLeastPrivilegeError("least_privilege_graph_identity_invalid")
    row = await connection.fetchrow(
        """
        SELECT
          to_regnamespace($1) IS NOT NULL AS schema_exists,
          NOT has_schema_privilege(current_user, $1, 'CREATE') AS cannot_create,
          schema_owner.rolname <> current_user AS runtime_is_not_owner,
          (
              SELECT count(*)
              FROM pg_class AS relations
              JOIN pg_namespace AS schemas ON schemas.oid = relations.relnamespace
              WHERE schemas.nspname = $1
                AND relations.relkind IN ('r', 'p')
                AND relations.relname = ANY($2::text[])
          ) = $3 AS labels_exist
        FROM pg_namespace AS graph_schema
        JOIN pg_roles AS schema_owner ON schema_owner.oid = graph_schema.nspowner
        WHERE graph_schema.nspname = $1
        """,
        graph_name,
        list(_AGE_TABLES),
        len(_AGE_TABLES),
    )
    if row is None or not all(bool(row[name]) for name in (
        "schema_exists",
        "cannot_create",
        "runtime_is_not_owner",
        "labels_exist",
    )):
        raise LightRAGLeastPrivilegeError("least_privilege_graph_contract_invalid")


def install_runtime_guard(
    environment: Mapping[str, str] | None = None,
    *,
    postgres_module: ModuleType | None = None,
) -> None:
    """Replace pinned LightRAG 1.5.4 initialization DDL with validation."""

    resolved = os.environ if environment is None else environment
    if _required(resolved, "POSTGRES_USER") != LIGHTRAG_RUNTIME_ROLE:
        raise LightRAGLeastPrivilegeError("least_privilege_role_mismatch")
    contract = _runtime_contract(resolved)
    if postgres_module is None:
        from lightrag.kg import postgres_impl as postgres_module

    if getattr(postgres_module, _PATCH_MARKER, False):
        return

    async def check_tables(db: Any) -> None:
        await _validate_existing_tables(db, contract)

    async def setup_table(
        db: Any,
        table_name: str,
        workspace: str,
        embedding_dim: int,
        legacy_table_name: str,
        base_table: str,
    ) -> None:
        del legacy_table_name, base_table
        await _validate_existing_vector_table(
            db,
            table_name,
            workspace,
            embedding_dim,
            contract,
        )

    async def configure_age(connection: Any, graph_name: str) -> None:
        await _validate_existing_age_graph(connection, graph_name, contract)
        await connection.execute('SET search_path = ag_catalog, "$user", public')

    async def initialize_graph(storage: Any) -> None:
        async with postgres_module.get_data_init_lock():
            if storage.db is None:
                storage.db = await postgres_module.ClientManager.get_client(
                    vector_storage=storage.global_config.get("vector_storage")
                )
            if storage.db.workspace:
                storage.workspace = storage.db.workspace
            elif not getattr(storage, "workspace", ""):
                storage.workspace = "default"
            storage.graph_name = storage._get_workspace_graph_name()

            async def validate(connection: Any) -> None:
                await _validate_existing_age_graph(connection, storage.graph_name, contract)

            await storage.db._run_with_retry(validate)

    postgres_module.PostgreSQLDB.check_tables = check_tables
    postgres_module.PGVectorStorage.setup_table = staticmethod(setup_table)
    postgres_module.PostgreSQLDB.configure_age = staticmethod(configure_age)
    postgres_module.PGGraphStorage.initialize = initialize_graph
    setattr(postgres_module, _PATCH_MARKER, True)


def main(arguments: Sequence[str] | None = None) -> int:
    """Install the guard, then execute the pinned upstream LightRAG server."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.parse_known_args(list(arguments) if arguments is not None else None)
    install_runtime_guard()
    forwarded = list(sys.argv[1:] if arguments is None else arguments)
    sys.argv = ["lightrag.api.lightrag_server", *forwarded]
    runpy.run_module("lightrag.api.lightrag_server", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
