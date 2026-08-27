"""Explicitly enabled smoke against a disposable PostgreSQL + Apache AGE database."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest

from material_graph.knowledge.age_writer import (
    GraphWriteApproval,
    GraphWriteApprovalRequest,
    KnowledgeGraphPersistenceError,
    PostgresAGEGlobalKnowledgeGraphWriter,
)
from material_graph.knowledge.facts import (
    EntityRef,
    EvidenceLink,
    ExtractionProvenance,
    FactBatch,
    KnowledgeGraphConflict,
    RelationAssertion,
    project_fact_batch,
)
from material_graph.knowledge.models import SourceLocator


ROOT = Path(__file__).parents[2]
FORWARD_MIGRATIONS = (
    "knowledge_0001.sql",
    "knowledge_0002.sql",
    "knowledge_0003.sql",
    "knowledge_0004.sql",
    "knowledge_0005.sql",
    "provider_0001.sql",
    "graph_admission_0001.sql",
    "age_0001.sql",
)
ENABLE_ENV = "MATERIAL_GRAPH_RUN_AGE_SMOKE"
DSN_ENV = "MATERIAL_GRAPH_AGE_DSN"
RUN_ID_ENV = "MATERIAL_GRAPH_AGE_SMOKE_RUN_ID"
SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
REVIEW_DIGEST = "1" * 64
AUDIT_DIGEST = "2" * 64

pytestmark = pytest.mark.skipif(
    os.environ.get(ENABLE_ENV) != "1",
    reason=f"set {ENABLE_ENV}=1 to run the disposable AGE integration smoke",
)


class ApprovalRepository:
    """Content-bound approval fake used only inside the disposable smoke database."""

    async def get_approval(self, request: GraphWriteApprovalRequest) -> GraphWriteApproval:
        return GraphWriteApproval(
            **request.model_dump(mode="python"),
            approved=True,
            reviewer_generation_digest=REVIEW_DIGEST,
            audit_generation_digest=AUDIT_DIGEST,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )


def _source_uuid(nonce: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"age-smoke:{nonce}:source")


def _fragment_uuid(nonce: str, variant: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"age-smoke:{nonce}:fragment:{variant}")


def _batch(
    nonce: str,
    variant: str,
    *,
    canonical_name: str | None = None,
    enriched_alias: bool = False,
    atomic: bool = False,
) -> FactBatch:
    prefix = "Atomic" if atomic else "Material"
    material_name = canonical_name or f"AGE Smoke {prefix} {nonce}"
    aliases = (f"AS-{nonce[:8]}", material_name)
    if enriched_alias:
        aliases = (*aliases, f"AGE Smoke Alias {nonce[:8]}")
    material = EntityRef(
        entity_type="material",
        canonical_name=material_name,
        aliases=aliases,
        identifiers={"smoke_registry": f"material-{nonce}-{prefix.casefold()}"},
    )
    additive = EntityRef(
        entity_type="component",
        canonical_name=f"AGE Smoke Additive {nonce}",
        identifiers={"smoke_registry": f"additive-{nonce}-{prefix.casefold()}"},
    )
    provenance = ExtractionProvenance(
        extractor_name="age-smoke",
        extractor_version="1.0",
        generation_id=f"age-smoke:{nonce}:{variant}",
        model_name="deterministic-smoke",
        model_version="1",
    )
    fragment_id = _fragment_uuid(nonce, variant)
    evidence = EvidenceLink(
        fragment_id=fragment_id,
        source_id=_source_uuid(nonce),
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path=f"integration/{nonce}.pdf",
            page=1,
            section="AGE smoke",
        ),
        role="supports",
    )
    relation = RelationAssertion(
        subject=additive,
        predicate="improves",
        object=material,
        evidence=(evidence,),
        confidence=0.9,
        evidence_quality="high",
        assertion_status="affirmed",
        extraction=provenance,
    )
    return FactBatch(
        evidence_fragment_id=fragment_id,
        extraction=provenance,
        entities=(material, additive),
        relations=(relation,),
    )


def _material(batch: FactBatch) -> EntityRef:
    return next(entity for entity in batch.entities if entity.entity_type == "material")


def _scalar(row: Mapping[str, Any] | None, field: str = "value") -> Any:
    assert row is not None, "age_smoke_missing_row"
    return row[field]


async def _apply_forward_migrations(dsn: str, psycopg: Any) -> None:
    try:
        connection = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            for filename in FORWARD_MIGRATIONS:
                sql = (ROOT / "migrations" / filename).read_text(encoding="utf-8")
                await connection.execute(sql)
    except Exception:
        raise AssertionError("age_smoke_forward_migration_failed") from None


async def _fetchone(
    pool: Any,
    sql: str,
    params: Sequence[object] = (),
) -> Mapping[str, Any] | None:
    async with pool.connection() as connection:
        cursor = await connection.execute(sql, tuple(params))
        row = await cursor.fetchone()
        assert row is None or isinstance(row, Mapping), "age_smoke_invalid_database_row"
        return row


async def _count(pool: Any, table: str) -> int:
    allowed = {
        "knowledge_graph_batches",
        "knowledge_graph_edges",
        "knowledge_graph_nodes",
    }
    assert table in allowed, "age_smoke_table_not_allowed"
    row = await _fetchone(pool, f"SELECT count(*) AS value FROM public.{table}")
    return int(_scalar(row))


def _agtype_int(value: object) -> int:
    return int(str(value).strip('"'))


async def _age_counts(pool: Any) -> tuple[int, int]:
    async with pool.connection() as connection:
        async with connection.transaction():
            await connection.execute("LOAD 'age'")
            await connection.execute("SET LOCAL search_path = ag_catalog, public")
            node_cursor = await connection.execute(
                """
                SELECT * FROM ag_catalog.cypher(
                    'material_graph',
                    $cypher$ MATCH (node:KnowledgeNode) RETURN count(node) $cypher$
                ) AS (value ag_catalog.agtype)
                """
            )
            edge_cursor = await connection.execute(
                """
                SELECT * FROM ag_catalog.cypher(
                    'material_graph',
                    $cypher$ MATCH ()-[edge:KNOWLEDGE_EDGE]->() RETURN count(edge) $cypher$
                ) AS (value ag_catalog.agtype)
                """
            )
            nodes = _agtype_int(_scalar(await node_cursor.fetchone()))
            edges = _agtype_int(_scalar(await edge_cursor.fetchone()))
            return nodes, edges


async def _install_rollback_guard(pool: Any, idempotency_key: str) -> None:
    async with pool.connection() as connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS public.age_smoke_rejections (
                    idempotency_key text PRIMARY KEY
                )
                """
            )
            await connection.execute(
                """
                CREATE OR REPLACE FUNCTION public.age_smoke_reject_graph_batch()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $function$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM public.age_smoke_rejections
                        WHERE idempotency_key = NEW.idempotency_key
                    ) THEN
                        RAISE EXCEPTION 'age_smoke_forced_rollback';
                    END IF;
                    RETURN NEW;
                END
                $function$
                """
            )
            await connection.execute(
                "DROP TRIGGER IF EXISTS age_smoke_reject_graph_batch "
                "ON public.knowledge_graph_batches"
            )
            await connection.execute(
                """
                CREATE TRIGGER age_smoke_reject_graph_batch
                BEFORE INSERT ON public.knowledge_graph_batches
                FOR EACH ROW EXECUTE FUNCTION public.age_smoke_reject_graph_batch()
                """
            )
            await connection.execute(
                """
                INSERT INTO public.age_smoke_rejections(idempotency_key)
                VALUES (%s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (idempotency_key,),
            )


async def _remove_rollback_guard(pool: Any) -> None:
    async with pool.connection() as connection:
        async with connection.transaction():
            await connection.execute(
                "DROP TRIGGER IF EXISTS age_smoke_reject_graph_batch "
                "ON public.knowledge_graph_batches"
            )
            await connection.execute(
                "DROP FUNCTION IF EXISTS public.age_smoke_reject_graph_batch()"
            )
            await connection.execute("DROP TABLE IF EXISTS public.age_smoke_rejections")


async def _run_smoke(dsn: str, nonce: str) -> None:
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError:
        raise AssertionError("age_smoke_database_extra_missing") from None

    await _apply_forward_migrations(dsn, psycopg)
    pool = AsyncConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"row_factory": dict_row},
    )
    await pool.open()
    await pool.wait()
    try:
        expected_versions = [
            filename.removesuffix(".sql")
            for filename in FORWARD_MIGRATIONS
            if filename != "age_0001.sql"
        ]
        versions = await _fetchone(
            pool,
            """
            SELECT count(*) AS value
            FROM public.schema_migrations
            WHERE version = ANY(%s::text[])
            """,
            (expected_versions,),
        )
        assert int(_scalar(versions)) == len(expected_versions), "age_smoke_migrations_not_applied"

        writer = PostgresAGEGlobalKnowledgeGraphWriter(pool, ApprovalRepository())
        original = _batch(nonce, "original")
        before_batches = await _count(pool, "knowledge_graph_batches")
        first = await writer.write_batch(original)
        assert first.status == "written"
        assert await _count(pool, "knowledge_graph_batches") == before_batches + 1

        before_replay = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        replay = await writer.write_batch(original)
        after_replay = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        assert replay.status == "already_present"
        assert after_replay == before_replay, "age_smoke_idempotent_replay_changed_state"

        enriched = _batch(nonce, "aliases", enriched_alias=True)
        alias_result = await writer.write_batch(enriched)
        assert alias_result.status == "written"
        material_id = _material(original).entity_id
        alias_row = await _fetchone(
            pool,
            "SELECT properties AS value FROM public.knowledge_graph_nodes WHERE node_id = %s",
            (material_id,),
        )
        properties = _scalar(alias_row)
        assert isinstance(properties, Mapping), "age_smoke_invalid_alias_properties"
        aliases = properties.get("aliases")
        assert isinstance(aliases, list), "age_smoke_invalid_aliases"
        assert f"AGE Smoke Alias {nonce[:8]}" in aliases

        conflict = _batch(
            nonce,
            "conflict",
            canonical_name=f"AGE SMOKE MATERIAL {nonce}",
        )
        conflict_before = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        with pytest.raises(KnowledgeGraphConflict, match="knowledge_graph_conflict"):
            await writer.write_batch(conflict)
        conflict_after = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        assert conflict_after == conflict_before, "age_smoke_conflict_was_not_atomic"

        atomic = _batch(nonce, "atomic", atomic=True)
        atomic_projection = project_fact_batch(atomic)
        atomic_node_ids = [node.node_id for node in atomic_projection.nodes]
        atomic_before = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        await _install_rollback_guard(pool, atomic.idempotency_key)  # type: ignore[arg-type]
        try:
            with pytest.raises(
                KnowledgeGraphPersistenceError,
                match="knowledge_graph_persistence_error",
            ):
                await writer.write_batch(atomic)
        finally:
            await _remove_rollback_guard(pool)
        atomic_after = (
            await _count(pool, "knowledge_graph_batches"),
            await _count(pool, "knowledge_graph_nodes"),
            await _count(pool, "knowledge_graph_edges"),
            await _age_counts(pool),
        )
        mirror_atomic = await _fetchone(
            pool,
            "SELECT count(*) AS value FROM public.knowledge_graph_nodes "
            "WHERE node_id = ANY(%s::text[])",
            (atomic_node_ids,),
        )
        assert atomic_after == atomic_before, "age_smoke_transaction_was_not_atomic"
        assert int(_scalar(mirror_atomic)) == 0, "age_smoke_atomic_nodes_survived_rollback"
    finally:
        await pool.close()


def test_real_age_writer_smoke() -> None:
    dsn = os.environ.get(DSN_ENV, "").strip()
    if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
        pytest.fail("age_smoke_dsn_missing_or_invalid", pytrace=False)
    configured_run_id = os.environ.get(RUN_ID_ENV, "").strip()
    if configured_run_id and SAFE_RUN_ID.fullmatch(configured_run_id) is None:
        pytest.fail("age_smoke_run_id_invalid", pytrace=False)
    nonce = configured_run_id or uuid4().hex
    asyncio.run(_run_smoke(dsn, nonce))
