from __future__ import annotations

from contextlib import asynccontextmanager
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_runtime_guard() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "src/material_graph/knowledge/lightrag_least_privilege.py"
    )
    spec = importlib.util.spec_from_file_location(
        "material_graph_lightrag_least_privilege_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


least_privilege = _load_runtime_guard()


GENERATION = "qwen3-embedding-4b-2560-bf16-v1"


def _environment() -> dict[str, str]:
    return {
        "MATERIAL_GRAPH_LIGHTRAG_GENERATION_ID": GENERATION,
        "POSTGRES_USER": "material_graph_lightrag",
        "WORKSPACE": GENERATION,
        "POSTGRES_WORKSPACE": GENERATION,
        "EMBEDDING_MODEL": "qwen3-embedding-4b",
        "EMBEDDING_DIM": "2560",
    }


class _FakeDb:
    def __init__(self, row: dict[str, bool]) -> None:
        self.row = row
        self.queries: list[tuple[str, list[object]]] = []
        self.workspace = GENERATION
        self.connections: list[object] = []

    async def query(self, query: str, params: list[object]) -> dict[str, bool]:
        self.queries.append((query, params))
        return self.row

    async def _run_with_retry(self, callback: object) -> None:
        connection = _FakeAgeConnection()
        self.connections.append(connection)
        await callback(connection)  # type: ignore[operator]


class _FakeAgeConnection:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid
        self.executed: list[str] = []

    async def fetchrow(self, *_: object) -> dict[str, bool] | None:
        if not self.valid:
            return None
        return {
            "schema_exists": True,
            "cannot_create": True,
            "runtime_is_not_owner": True,
            "labels_exist": True,
        }

    async def execute(self, query: str) -> None:
        self.executed.append(query)


def test_runtime_contract_derives_exact_pinned_tables_and_age_schema() -> None:
    contract = least_privilege._runtime_contract(_environment())
    assert contract["age_schema"] == (
        "qwen3_embedding_4b_2560_bf16_v1_chunk_entity_relation"
    )
    assert contract["dimensions"] == 2560
    assert contract["vector_tables"] == (
        "lightrag_vdb_chunks_qwen3_embedding_4b_2560d",
        "lightrag_vdb_entity_qwen3_embedding_4b_2560d",
        "lightrag_vdb_relation_qwen3_embedding_4b_2560d",
    )
    assert len(contract["material_graph_tables"]) == 11


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MATERIAL_GRAPH_LIGHTRAG_GENERATION_ID", "future-generation", "not_allowlisted"),
        ("WORKSPACE", "other", "binding_mismatch"),
        ("POSTGRES_WORKSPACE", "other", "binding_mismatch"),
        ("EMBEDDING_MODEL", "other", "binding_mismatch"),
        ("EMBEDDING_DIM", "1024", "binding_mismatch"),
    ],
)
def test_runtime_contract_rejects_generation_or_binding_drift(
    name: str,
    value: str,
    message: str,
) -> None:
    environment = _environment()
    environment[name] = value
    with pytest.raises(least_privilege.LightRAGLeastPrivilegeError, match=message):
        least_privilege._runtime_contract(environment)


def test_runtime_contract_rejects_missing_and_non_numeric_binding() -> None:
    environment = _environment()
    environment.pop("WORKSPACE")
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="configuration_missing",
    ):
        least_privilege._runtime_contract(environment)
    environment = _environment()
    environment["EMBEDDING_DIM"] = "not-a-number"
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="binding_mismatch",
    ):
        least_privilege._runtime_contract(environment)


@pytest.mark.asyncio
async def test_existing_table_and_vector_contracts_are_read_only_and_fail_closed() -> None:
    contract = least_privilege._runtime_contract(_environment())
    table_row = {
        "role_matches": True,
        "not_superuser": True,
        "cannot_bypass_rls": True,
        "cannot_create_role": True,
        "cannot_create_database": True,
        "cannot_inherit": True,
        "row_security_on": True,
        "cannot_create_schema": True,
        "material_graph_exists": True,
        "tables_match": True,
    }
    table_db = _FakeDb(table_row)
    await least_privilege._validate_existing_tables(table_db, contract)
    assert len(table_db.queries) == 1
    assert "CREATE " not in table_db.queries[0][0].upper()
    assert "ALTER " not in table_db.queries[0][0].upper()
    assert "schemas.nspname = 'material_graph'" in table_db.queries[0][0]

    broken = _FakeDb({**table_row, "tables_match": False})
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="table_contract_invalid",
    ):
        await least_privilege._validate_existing_tables(broken, contract)

    vector_db = _FakeDb(
        {
            "dimensions_match": True,
            "workspace_present": True,
            "runtime_is_not_owner": True,
            "index_present": True,
        }
    )
    await least_privilege._validate_existing_vector_table(
        vector_db,
        "LIGHTRAG_VDB_CHUNKS_QWEN3_EMBEDDING_4B_2560D",
        GENERATION,
        2560,
        contract,
    )
    assert "table_namespace.nspname = 'material_graph'" in vector_db.queries[0][0]
    vector_db.row["index_present"] = False
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="vector_contract_invalid",
    ):
        await least_privilege._validate_existing_vector_table(
            vector_db,
            "lightrag_vdb_chunks_qwen3_embedding_4b_2560d",
            GENERATION,
            2560,
            contract,
        )
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="vector_identity_invalid",
    ):
        await least_privilege._validate_existing_vector_table(
            vector_db,
            "lightrag_vdb_chunks_embedding_3_1024d",
            GENERATION,
            2560,
            contract,
        )


@pytest.mark.asyncio
async def test_existing_age_graph_validation_never_runs_structure_ddl() -> None:
    contract = least_privilege._runtime_contract(_environment())
    connection = _FakeAgeConnection()
    await least_privilege._validate_existing_age_graph(
        connection,
        str(contract["age_schema"]),
        contract,
    )
    assert connection.executed == []
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="graph_identity_invalid",
    ):
        await least_privilege._validate_existing_age_graph(
            connection,
            "other_graph",
            contract,
        )
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="graph_contract_invalid",
    ):
        await least_privilege._validate_existing_age_graph(
            _FakeAgeConnection(valid=False),
            str(contract["age_schema"]),
            contract,
        )


@pytest.mark.asyncio
async def test_installed_guard_replaces_every_upstream_initialization_ddl_hook() -> None:
    contract = least_privilege._runtime_contract(_environment())
    table_row = {
        "role_matches": True,
        "not_superuser": True,
        "cannot_bypass_rls": True,
        "cannot_create_role": True,
        "cannot_create_database": True,
        "cannot_inherit": True,
        "row_security_on": True,
        "cannot_create_schema": True,
        "material_graph_exists": True,
        "tables_match": True,
    }
    class PostgreSQLDB(_FakeDb):
        async def check_tables(self) -> None:
            raise AssertionError("upstream DDL hook was not replaced")

        @staticmethod
        async def configure_age(*_: object) -> None:
            raise AssertionError("upstream create_graph hook was not replaced")

    db = PostgreSQLDB(table_row)

    class PGVectorStorage:
        @staticmethod
        async def setup_table(*_: object) -> None:
            raise AssertionError("upstream vector DDL hook was not replaced")

    class PGGraphStorage:
        def __init__(self) -> None:
            self.db = None
            self.workspace = GENERATION
            self.global_config = {"vector_storage": "PGVectorStorage"}
            self.graph_name = ""

        def _get_workspace_graph_name(self) -> str:
            return str(contract["age_schema"])

        async def initialize(self) -> None:
            raise AssertionError("upstream graph DDL hook was not replaced")

    class ClientManager:
        @staticmethod
        async def get_client(**_: object) -> _FakeDb:
            return db

    @asynccontextmanager
    async def init_lock():
        yield

    module = SimpleNamespace(
        PostgreSQLDB=PostgreSQLDB,
        PGVectorStorage=PGVectorStorage,
        PGGraphStorage=PGGraphStorage,
        ClientManager=ClientManager,
        get_data_init_lock=init_lock,
    )
    least_privilege.install_runtime_guard(
        _environment(),
        postgres_module=module,  # type: ignore[arg-type]
    )
    least_privilege.install_runtime_guard(
        _environment(),
        postgres_module=module,  # type: ignore[arg-type]
    )
    await db.check_tables()
    assert db.queries

    vector_db = _FakeDb(
        {
            "dimensions_match": True,
            "workspace_present": True,
            "runtime_is_not_owner": True,
            "index_present": True,
        }
    )
    await PGVectorStorage.setup_table(
        vector_db,
        str(contract["vector_tables"][0]),
        GENERATION,
        2560,
        "legacy",
        "base",
    )

    await PostgreSQLDB.configure_age(
        _FakeAgeConnection(),
        str(contract["age_schema"]),
    )
    graph = PGGraphStorage()
    await graph.initialize()
    assert graph.db is db
    assert graph.graph_name == contract["age_schema"]
    assert db.connections
    source = inspect.getsource(least_privilege.install_runtime_guard).upper()
    assert "CREATE TABLE" not in source
    assert "CREATE INDEX" not in source
    assert "CREATE_GRAPH" not in source
    assert "ALTER TABLE" not in source


def test_runtime_guard_rejects_any_role_other_than_dedicated_lightrag() -> None:
    environment = _environment()
    environment["POSTGRES_USER"] = "material_graph"
    with pytest.raises(
        least_privilege.LightRAGLeastPrivilegeError,
        match="role_mismatch",
    ):
        least_privilege.install_runtime_guard(environment, postgres_module=SimpleNamespace())


def test_runtime_guard_can_load_the_pinned_upstream_module() -> None:
    from lightrag.kg import postgres_impl

    original = (
        postgres_impl.PostgreSQLDB.__dict__["check_tables"],
        postgres_impl.PGVectorStorage.__dict__["setup_table"],
        postgres_impl.PostgreSQLDB.__dict__["configure_age"],
        postgres_impl.PGGraphStorage.__dict__["initialize"],
    )
    try:
        least_privilege.install_runtime_guard(_environment())
        assert getattr(postgres_impl, least_privilege._PATCH_MARKER) is True
    finally:
        (
            postgres_impl.PostgreSQLDB.check_tables,
            postgres_impl.PGVectorStorage.setup_table,
            postgres_impl.PostgreSQLDB.configure_age,
            postgres_impl.PGGraphStorage.initialize,
        ) = original
        delattr(postgres_impl, least_privilege._PATCH_MARKER)
