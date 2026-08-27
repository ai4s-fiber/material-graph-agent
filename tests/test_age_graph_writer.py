from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from material_graph.knowledge import age_writer as age
from material_graph.knowledge.facts import (
    EntityRef,
    EvidenceLink,
    ExtractionProvenance,
    FactBatch,
    GlobalKnowledgeGraphWriter,
    KnowledgeGraphConflict,
    PropertyGraphEdge,
    PropertyGraphNode,
    PropertyGraphProjection,
    RelationAssertion,
    project_fact_batch,
)
from material_graph.knowledge.models import SourceLocator


ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)
SOURCE_ID = UUID("11111111-1111-4111-8111-111111111111")
FRAGMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
OTHER_FRAGMENT_ID = UUID("33333333-3333-4333-8333-333333333333")
REVIEW_DIGEST = "1" * 64
AUDIT_DIGEST = "2" * 64


@dataclass(frozen=True)
class Statement:
    sql: str
    params: tuple[object, ...]


class Script:
    def __init__(self) -> None:
        self.responses: dict[str, deque[list[object]]] = defaultdict(deque)
        self.failures: dict[str, BaseException] = {}

    def add(self, needle: str, *responses: Sequence[object]) -> None:
        for response in responses:
            self.responses[needle].append(list(response))

    def fail(self, needle: str, error: BaseException) -> None:
        self.failures[needle] = error

    def take(self, sql: str) -> list[object]:
        compact = " ".join(sql.split())
        for needle, error in self.failures.items():
            if needle in compact:
                raise error
        for needle, queued in self.responses.items():
            if needle in compact and queued:
                return queued.popleft()
        return []


class AsyncCursor:
    def __init__(self, rows: Sequence[object]) -> None:
        self.rows = list(rows)

    async def fetchone(self) -> object | None:
        return None if not self.rows else self.rows[0]

    async def fetchall(self) -> list[object]:
        return list(self.rows)


class RecordingConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    async def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> AsyncCursor:
        self.statements.append(Statement(sql=sql, params=tuple(params or ())))
        return AsyncCursor(self.script.take(sql))

    @asynccontextmanager
    async def transaction(self):
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
        self.dsn = "postgresql://secret-user:secret-password@db/internal"

    @asynccontextmanager
    async def connection(self):
        self.connections += 1
        yield self.value

    def __repr__(self) -> str:
        return f"RecordingPool(dsn={self.dsn!r})"


class ApprovalGate:
    def __init__(self, mode: str = "approved") -> None:
        self.mode = mode
        self.calls: list[age.GraphWriteApprovalRequest] = []

    async def get_approval(
        self,
        request: age.GraphWriteApprovalRequest,
    ) -> age.GraphWriteApproval | Mapping[str, object] | None:
        self.calls.append(request)
        if self.mode == "missing":
            return None
        if self.mode == "unavailable":
            raise RuntimeError("provider included secret reviewer identity")
        if self.mode == "stable_error":
            raise age.KnowledgeGraphApprovalError("knowledge_graph_approval_missing")
        approved = self.mode != "rejected"
        expires_at = NOW + timedelta(hours=1)
        if self.mode == "expired":
            expires_at = NOW - timedelta(seconds=1)
        projection_digest = request.projection_digest
        if self.mode == "mismatch":
            projection_digest = "0" * 64
        payload: dict[str, object] = {
            "batch_id": request.batch_id,
            "idempotency_key": request.idempotency_key,
            "projection_digest": projection_digest,
            "approved": approved,
            "reviewer_generation_digest": REVIEW_DIGEST,
            "audit_generation_digest": AUDIT_DIGEST,
            "expires_at": expires_at,
        }
        if self.mode == "invalid":
            payload["reviewer"] = "raw-person@example.invalid"
            return payload
        return age.GraphWriteApproval.model_validate(payload)


def extractor(generation: str = "extractor-generation-1") -> ExtractionProvenance:
    return ExtractionProvenance(
        extractor_name="materials-fact-extractor",
        extractor_version="1.0",
        generation_id=generation,
        model_name="reasoning-model",
        model_version="2026-07-27",
    )


def evidence(fragment_id: UUID = FRAGMENT_ID, *, page: int = 3) -> EvidenceLink:
    return EvidenceLink(
        fragment_id=fragment_id,
        source_id=SOURCE_ID,
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path="private/nas/material.pdf",
            page=page,
            section="Results",
        ),
        role="supports",
    )


def entities(*, enriched: bool = False) -> tuple[EntityRef, EntityRef]:
    aliases = ("PI-X", "Polymer X")
    if enriched:
        aliases = (*aliases, "Laboratory alias")
    material = EntityRef(
        entity_type="material",
        canonical_name="Polymer X",
        aliases=aliases,
        identifiers={"registry": "MX-001"},
    )
    additive = EntityRef(
        entity_type="component",
        canonical_name="Additive Y",
        identifiers={"registry": "AY-001"},
    )
    return material, additive


def batch(
    *,
    fragment_id: UUID = FRAGMENT_ID,
    generation: str = "extractor-generation-1",
    predicate: str = "improves",
    page: int = 3,
    enriched: bool = False,
) -> FactBatch:
    material, additive = entities(enriched=enriched)
    provenance = extractor(generation)
    relation = RelationAssertion(
        subject=additive,
        predicate=predicate,
        object=material,
        evidence=(evidence(fragment_id, page=page),),
        confidence=0.86,
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


def projection(value: FactBatch) -> PropertyGraphProjection:
    return age._validate_projection(project_fact_batch(value))


def digest(value: FactBatch) -> str:
    return age._projection_digest(projection(value))


def node_row(node: PropertyGraphNode, *, strings: bool = False) -> dict[str, object]:
    labels: object = list(node.labels)
    properties: object = node.properties
    if strings:
        labels = json.dumps(labels)
        properties = json.dumps(properties)
    return {"node_id": node.node_id, "labels": labels, "properties": properties}


def edge_row(edge: PropertyGraphEdge) -> dict[str, object]:
    return {
        "edge_id": edge.edge_id,
        "edge_type": edge.edge_type,
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "properties": edge.properties,
    }


def configured_writer(
    *,
    script: Script | None = None,
    gate: ApprovalGate | None = None,
) -> tuple[
    age.PostgresAGEGlobalKnowledgeGraphWriter,
    RecordingPool,
    RecordingConnection,
    Script,
    ApprovalGate,
]:
    resolved_script = script or Script()
    connection = RecordingConnection(resolved_script)
    pool = RecordingPool(connection)
    resolved_gate = gate or ApprovalGate()
    writer = age.PostgresAGEGlobalKnowledgeGraphWriter(
        pool,
        resolved_gate,
        clock=lambda: NOW,
    )
    return writer, pool, connection, resolved_script, resolved_gate


def configure_new_batch(
    script: Script, *, node_rows: Sequence[object] = (), edge_rows: Sequence[object] = ()
) -> None:
    script.add("FROM public.knowledge_graph_batches", [])
    script.add("FROM public.knowledge_graph_nodes", node_rows)
    script.add("FROM public.knowledge_graph_edges", edge_rows)
    script.add("INSERT INTO public.knowledge_graph_batches", [{"idempotency_key": "written"}])


def run_write(
    writer: age.PostgresAGEGlobalKnowledgeGraphWriter,
    value: FactBatch,
):
    return asyncio.run(writer.write_batch(value))


def test_age_migration_is_independent_idempotent_and_keeps_shared_graph_on_down() -> None:
    forward = (ROOT / "migrations" / "age_0001.sql").read_text(encoding="utf-8")
    down = (ROOT / "migrations" / "age_0001.down.sql").read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS age" in forward
    assert "LOAD 'age'" in forward
    assert "ag_catalog.create_graph('material_graph')" in forward
    assert "VALUES ('age_0001')" in forward
    assert "ON CONFLICT (version) DO NOTHING" in forward
    assert "CREATE TABLE IF NOT EXISTS knowledge_graph_batches" in forward
    assert "CREATE TABLE IF NOT EXISTS knowledge_graph_nodes" in forward
    assert "CREATE TABLE IF NOT EXISTS knowledge_graph_edges" in forward
    assert "reviewer_generation_digest" in forward
    assert "approval_status = 'approved'" in forward
    assert "Never run automatically" in down
    assert "drop_graph" not in down.casefold()
    assert down.index("knowledge_graph_edges") < down.index("knowledge_graph_nodes")
    assert "DELETE FROM schema_migrations WHERE version = 'age_0001'" in down


def test_approval_receipt_is_content_bound_and_rejects_tampering_or_naive_time() -> None:
    value = batch()
    request = age.GraphWriteApprovalRequest(
        batch_id=value.batch_id,
        idempotency_key=value.idempotency_key,
        projection_digest=digest(value),
    )
    receipt = age.GraphWriteApproval(
        **request.model_dump(),
        approved=True,
        reviewer_generation_digest=REVIEW_DIGEST,
        audit_generation_digest=AUDIT_DIGEST,
        expires_at=NOW + timedelta(hours=1),
    )
    assert receipt.approval_digest is not None
    with pytest.raises(ValidationError, match="approval_digest_mismatch"):
        receipt.model_copy(update={"approval_digest": "graph-approval:v1:" + "0" * 64}).model_dump()
        age.GraphWriteApproval.model_validate(
            {**receipt.model_dump(), "approval_digest": "graph-approval:v1:" + "0" * 64}
        )
    with pytest.raises(ValidationError, match="timezone_aware"):
        age.GraphWriteApproval(
            **request.model_dump(),
            approved=True,
            reviewer_generation_digest=REVIEW_DIGEST,
            audit_generation_digest=AUDIT_DIGEST,
            expires_at=datetime(2026, 7, 27, 9, 0),
        )


def test_new_batch_writes_mirror_age_and_approval_ledger_in_one_transaction() -> None:
    value = batch()
    writer, pool, connection, script, gate = configured_writer()
    configure_new_batch(script)

    result = run_write(writer, value)

    assert result.status == "written"
    assert result.node_count == len(projection(value).nodes)
    assert result.edge_count == len(projection(value).edges)
    assert pool.connections == 1
    assert connection.transactions == connection.commits == 1
    assert connection.rollbacks == 0
    assert len(gate.calls) == 1
    compact_sql = [" ".join(statement.sql.split()) for statement in connection.statements]
    assert "LOAD 'age'" in compact_sql
    assert any("pg_advisory_xact_lock" in statement for statement in compact_sql)
    assert any("jsonb_array_elements" in statement for statement in compact_sql)
    assert any("UNWIND $nodes AS item" in statement for statement in compact_sql)
    assert any("UNWIND $edges AS item" in statement for statement in compact_sql)
    assert all("private/nas" not in statement for statement in compact_sql)
    assert all("Polymer X" not in statement for statement in compact_sql)

    age_node_statement = next(
        item for item in connection.statements if "UNWIND $nodes AS item" in item.sql
    )
    age_edge_statement = next(
        item for item in connection.statements if "UNWIND $edges AS item" in item.sql
    )
    assert ":KnowledgeNode" in age_node_statement.sql
    assert ":KNOWLEDGE_EDGE" in age_edge_statement.sql
    assert "MaterialEntity" not in age_node_statement.sql
    assert "ASSERTION_SUBJECT" not in age_edge_statement.sql
    assert "MaterialEntity" in str(age_node_statement.params)
    assert "ASSERTION_SUBJECT" in str(age_edge_statement.params)

    ledger = next(
        item
        for item in connection.statements
        if "INSERT INTO public.knowledge_graph_batches" in item.sql
    )
    assert REVIEW_DIGEST in ledger.params
    assert AUDIT_DIGEST in ledger.params
    assert "reviewer" not in str(ledger.params).casefold()
    assert "private/nas" not in str(connection.statements)


def test_same_idempotency_and_digest_returns_already_present_without_graph_updates() -> None:
    value = batch()
    script = Script()
    script.add(
        "FROM public.knowledge_graph_batches",
        [{"batch_id": value.batch_id, "projection_digest": digest(value)}],
    )
    writer, pool, connection, _, gate = configured_writer(script=script)

    result = run_write(writer, value)

    assert result.status == "already_present"
    assert pool.connections == 1
    assert connection.commits == 1
    assert len(gate.calls) == 1
    assert not any("UNWIND $nodes" in item.sql for item in connection.statements)
    assert not any(
        "INSERT INTO public.knowledge_graph_batches" in item.sql for item in connection.statements
    )


def test_same_idempotency_with_different_projection_conflicts_and_rolls_back() -> None:
    baseline = batch()
    changed = batch(predicate="reduces")
    script = Script()
    script.add(
        "FROM public.knowledge_graph_batches",
        [{"batch_id": baseline.batch_id, "projection_digest": digest(baseline)}],
    )
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(KnowledgeGraphConflict) as error:
        run_write(writer, changed)

    assert str(error.value) == "knowledge_graph_conflict"
    assert connection.rollbacks == 1
    assert not any("jsonb_array_elements" in item.sql for item in connection.statements)


def test_existing_node_drift_outside_aliases_conflicts_atomically() -> None:
    value = batch(generation="extractor-generation-2", page=9)
    candidate_evidence = next(
        node for node in projection(value).nodes if node.labels == ("EvidenceFragment",)
    )
    existing_evidence = candidate_evidence.model_copy(
        update={"properties": {**candidate_evidence.properties, "page": 3}},
        deep=True,
    )
    script = Script()
    configure_new_batch(script, node_rows=[node_row(existing_evidence)])
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(KnowledgeGraphConflict):
        run_write(writer, value)

    assert connection.rollbacks == 1
    assert not any(
        "INSERT INTO public.knowledge_graph_nodes" in item.sql for item in connection.statements
    )


def test_entity_aliases_are_safely_enriched_and_written_to_both_mirrors() -> None:
    baseline_projection = projection(batch())
    existing_material = next(
        node
        for node in baseline_projection.nodes
        if node.properties.get("canonical_name") == "Polymer X"
    )
    value = batch(fragment_id=OTHER_FRAGMENT_ID, enriched=True)
    script = Script()
    configure_new_batch(script, node_rows=[node_row(existing_material, strings=True)])
    writer, _, connection, _, _ = configured_writer(script=script)

    result = run_write(writer, value)

    assert result.status == "written"
    mirror = next(
        item
        for item in connection.statements
        if "INSERT INTO public.knowledge_graph_nodes" in item.sql
    )
    rows = json.loads(str(mirror.params[0]))
    updated = next(item for item in rows if item["node_id"] == existing_material.node_id)
    assert updated["properties"]["aliases"] == [
        "Laboratory alias",
        "PI-X",
        "Polymer X",
    ]
    age_statement = next(item for item in connection.statements if "UNWIND $nodes" in item.sql)
    assert "Laboratory alias" in str(age_statement.params)


def test_entity_property_drift_other_than_aliases_conflicts() -> None:
    value = batch(fragment_id=OTHER_FRAGMENT_ID)
    material = next(
        node
        for node in projection(value).nodes
        if node.properties.get("canonical_name") == "Polymer X"
    )
    drifted = material.model_copy(
        update={"properties": {**material.properties, "canonical_name": "Drifted name"}},
        deep=True,
    )
    script = Script()
    configure_new_batch(script, node_rows=[node_row(drifted)])
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(KnowledgeGraphConflict):
        run_write(writer, value)
    assert connection.rollbacks == 1


def test_existing_edge_drift_conflicts_before_any_upsert() -> None:
    value = batch(fragment_id=OTHER_FRAGMENT_ID)
    candidate_edge = projection(value).edges[0]
    drifted = candidate_edge.model_copy(update={"properties": {"role": "changed"}}, deep=True)
    script = Script()
    configure_new_batch(script, edge_rows=[edge_row(drifted)])
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(KnowledgeGraphConflict):
        run_write(writer, value)

    assert connection.rollbacks == 1
    assert not any("jsonb_array_elements" in item.sql for item in connection.statements)


def test_age_failure_rolls_back_relational_mirror_and_redacts_driver_error() -> None:
    script = Script()
    configure_new_batch(script)
    script.fail("UNWIND $nodes AS item", RuntimeError("db password leaked by AGE"))
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(age.KnowledgeGraphPersistenceError) as error:
        run_write(writer, batch())

    assert str(error.value) == "knowledge_graph_persistence_error"
    assert "password" not in str(error.value)
    assert connection.rollbacks == 1
    assert any(
        "INSERT INTO public.knowledge_graph_nodes" in item.sql for item in connection.statements
    )
    assert not any(
        "INSERT INTO public.knowledge_graph_batches" in item.sql for item in connection.statements
    )


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("missing", "knowledge_graph_approval_missing"),
        ("rejected", "knowledge_graph_approval_rejected"),
        ("expired", "knowledge_graph_approval_expired"),
        ("mismatch", "knowledge_graph_approval_mismatch"),
        ("invalid", "knowledge_graph_approval_invalid"),
        ("unavailable", "knowledge_graph_approval_unavailable"),
        ("stable_error", "knowledge_graph_approval_missing"),
    ],
)
def test_approval_gate_fails_closed_before_opening_database_connection(
    mode: str,
    code: str,
) -> None:
    gate = ApprovalGate(mode)
    writer, pool, _, _, _ = configured_writer(gate=gate)

    with pytest.raises(age.KnowledgeGraphApprovalError) as error:
        run_write(writer, batch())

    assert error.value.code == code
    assert str(error.value) == code
    assert "raw-person" not in str(error.value)
    assert pool.connections == 0


def test_naive_writer_clock_is_rejected_before_database_connection() -> None:
    connection = RecordingConnection(Script())
    pool = RecordingPool(connection)
    writer = age.PostgresAGEGlobalKnowledgeGraphWriter(
        pool,
        ApprovalGate(),
        clock=lambda: datetime(2026, 7, 27, 8, 0),
    )
    with pytest.raises(age.KnowledgeGraphApprovalError) as error:
        run_write(writer, batch())
    assert error.value.code == "knowledge_graph_approval_invalid"
    assert pool.connections == 0


@pytest.mark.parametrize(
    "properties",
    [
        {"root_id": "document_data_1"},
        {"relative_path": "private/nas/paper.pdf"},
        {"complete_mineru_output": {"pages": []}},
        {"value": "%PDF-1.7 raw bytes"},
        {"value": "https://example.cn6.quickconnect.cn/share"},
        {"value": "smb://nas/private/paper.pdf"},
        {"token": "not-even-a-real-token"},
        {"value": "Bearer " + "x" * 24},
    ],
)
def test_unsafe_graph_properties_are_rejected_before_approval_or_connection(
    monkeypatch: pytest.MonkeyPatch,
    properties: dict[str, object],
) -> None:
    unsafe = PropertyGraphProjection(
        nodes=(PropertyGraphNode(node_id="unsafe", labels=("Entity",), properties=properties),),
        edges=(),
    )
    monkeypatch.setattr(age, "project_fact_batch", lambda _batch: unsafe)
    writer, pool, _, _, gate = configured_writer()

    with pytest.raises(age.UnsafeKnowledgeGraphPayload) as error:
        run_write(writer, batch())

    assert str(error.value) == "unsafe_knowledge_graph_payload"
    assert pool.connections == 0
    assert gate.calls == []


def test_binary_projection_is_revalidated_and_rejected_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe = {
        "nodes": [{"node_id": "unsafe", "labels": ["Entity"], "properties": {"value": b"binary"}}],
        "edges": [],
    }
    monkeypatch.setattr(age, "project_fact_batch", lambda _batch: unsafe)
    writer, pool, _, _, _ = configured_writer()

    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        run_write(writer, batch())
    assert pool.connections == 0


def test_logical_labels_and_relationship_types_are_allowlisted_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_label = PropertyGraphProjection(
        nodes=(PropertyGraphNode(node_id="node-1", labels=("UserCypher",)),),
        edges=(),
    )
    monkeypatch.setattr(age, "project_fact_batch", lambda _batch: invalid_label)
    writer, pool, _, _, _ = configured_writer()
    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        run_write(writer, batch())
    assert pool.connections == 0

    left = PropertyGraphNode(node_id="node-1", labels=("Entity",))
    right = PropertyGraphNode(node_id="node-2", labels=("Entity",))
    invalid_edge = PropertyGraphProjection(
        nodes=(left, right),
        edges=(
            PropertyGraphEdge(
                edge_id="edge:v1:" + "0" * 64,
                edge_type="USER_DEFINED_CYPHER",
                source_node_id=left.node_id,
                target_node_id=right.node_id,
            ),
        ),
    )
    monkeypatch.setattr(age, "project_fact_batch", lambda _batch: invalid_edge)
    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        run_write(writer, batch())
    assert pool.connections == 0


def test_projection_count_and_size_limits_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(age, "_MAX_GRAPH_NODES", 0)
    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        age._validate_projection(project_fact_batch(batch()))
    monkeypatch.setattr(age, "_MAX_GRAPH_NODES", 10_000)
    monkeypatch.setattr(age, "_MAX_PROJECTION_BYTES", 1)
    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        age._validate_projection(project_fact_batch(batch()))
    monkeypatch.setattr(age, "_MAX_PARAMETER_BYTES", 1)
    with pytest.raises(age.UnsafeKnowledgeGraphPayload):
        age._encoded({"safe": "value"})


@pytest.mark.parametrize(
    "stored",
    [
        {"node_id": "node", "labels": "not-json", "properties": {}},
        {"node_id": "node", "labels": ["Entity"], "properties": {"root_id": "private"}},
        object(),
    ],
)
def test_invalid_stored_node_is_a_safe_persistence_failure(stored: object) -> None:
    script = Script()
    configure_new_batch(script, node_rows=[stored])
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(age.KnowledgeGraphPersistenceError) as error:
        run_write(writer, batch())

    assert str(error.value) == "knowledge_graph_persistence_error"
    assert connection.rollbacks == 1


def test_invalid_or_duplicate_stored_edges_fail_closed() -> None:
    value = batch()
    candidate = projection(value).edges[0]
    invalid = edge_row(candidate)
    invalid["edge_type"] = "UNSAFE_TYPE"
    script = Script()
    configure_new_batch(script, edge_rows=[invalid])
    writer, _, connection, _, _ = configured_writer(script=script)
    with pytest.raises(age.KnowledgeGraphPersistenceError):
        run_write(writer, value)
    assert connection.rollbacks == 1

    script = Script()
    configure_new_batch(script, edge_rows=[edge_row(candidate), edge_row(candidate)])
    writer, _, connection, _, _ = configured_writer(script=script)
    with pytest.raises(age.KnowledgeGraphPersistenceError):
        run_write(writer, value)
    assert connection.rollbacks == 1


def test_missing_ledger_returning_row_rolls_back() -> None:
    script = Script()
    script.add("FROM public.knowledge_graph_batches", [])
    script.add("FROM public.knowledge_graph_nodes", [])
    script.add("FROM public.knowledge_graph_edges", [])
    script.add("INSERT INTO public.knowledge_graph_batches", [])
    writer, _, connection, _, _ = configured_writer(script=script)

    with pytest.raises(age.KnowledgeGraphPersistenceError):
        run_write(writer, batch())
    assert connection.rollbacks == 1


def test_repository_repr_and_runtime_protocol_do_not_expose_pool_or_dsn() -> None:
    writer, _, _, _, _ = configured_writer()

    assert repr(writer) == "PostgresAGEGlobalKnowledgeGraphWriter()"
    assert "postgresql" not in repr(writer)
    assert "secret" not in repr(writer)
    assert isinstance(writer, GlobalKnowledgeGraphWriter)
    assert isinstance(ApprovalGate(), age.GraphWriteApprovalRepository)


def test_default_clock_constructor_and_parameter_templates_remain_static() -> None:
    connection = RecordingConnection(Script())
    pool = RecordingPool(connection)
    writer = age.PostgresAGEGlobalKnowledgeGraphWriter(pool, ApprovalGate())
    assert repr(writer) == "PostgresAGEGlobalKnowledgeGraphWriter()"
    assert age.AGE_GRAPH_NAME == "material_graph"
    assert age.PHYSICAL_NODE_LABEL == "KnowledgeNode"
    assert age.PHYSICAL_EDGE_TYPE == "KNOWLEDGE_EDGE"
    assert "%s" in age._AGE_UPSERT_NODES
    assert "%s" in age._AGE_UPSERT_EDGES
    assert "material_graph" in age._AGE_UPSERT_NODES
