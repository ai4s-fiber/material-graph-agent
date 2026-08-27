"""Atomic Apache AGE writer for approved global material knowledge.

The adapter deliberately uses one fixed physical node label and relationship
type. Logical labels and edge types remain validated data, never executable
Cypher, so untrusted extracted values cannot alter a query template.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .facts import (
    FactBatch,
    FactWriteResult,
    KnowledgeGraphConflict,
    PropertyGraphEdge,
    PropertyGraphNode,
    PropertyGraphProjection,
    _canonical_json,
    _merge_entity_aliases,
    project_fact_batch,
    validate_fact_batch,
)
from .postgres import AsyncConnectionPool, UnsafeDurablePayload, _validated_json_object


AGE_GRAPH_NAME = "material_graph"
AGE_GRAPH_SCHEMA_VERSION = "age_0001"
PHYSICAL_NODE_LABEL = "KnowledgeNode"
PHYSICAL_EDGE_TYPE = "KNOWLEDGE_EDGE"

_MAX_GRAPH_NODES = 10_000
_MAX_GRAPH_EDGES = 25_000
_MAX_PROJECTION_BYTES = 16 * 1024 * 1024
_MAX_PARAMETER_BYTES = 16 * 1024 * 1024

_ALLOWED_LABEL_SETS = frozenset(
    {
        ("Application", "Entity"),
        ("Composition", "Entity", "MaterialEntity"),
        ("Entity",),
        ("Entity", "MaterialEntity"),
        ("Entity", "MaterialEntity", "Sample"),
        ("Entity", "Process"),
        ("Entity", "Source"),
        ("Entity", "TestMethod"),
        ("EvidenceFragment",),
        ("Fact", "PropertyObservation"),
        ("Fact", "RelationAssertion"),
    }
)
_ALLOWED_EDGE_TYPES = frozenset(
    {"ASSERTION_SUBJECT", "ASSERTION_OBJECT", "OBSERVED_ON", "SUPPORTED_BY"}
)
_FORBIDDEN_GRAPH_KEYS = frozenset(
    {
        "absolute_path",
        "complete_mineru_output",
        "complete_parser_output",
        "content",
        "document_text",
        "full_document",
        "full_document_text",
        "local_path",
        "markdown",
        "mineru_json",
        "mineru_markdown",
        "mineru_output",
        "nas_path",
        "original_pdf",
        "parser_output",
        "pdf_bytes",
        "raw_document",
        "raw_pdf",
        "raw_text",
        "relative_path",
        "root_id",
        "source_bytes",
        "source_path",
        "text",
    }
)
_INTERNAL_LOCATION = re.compile(
    r"(?:quickconnect|(?:^|[\s\"'(])(?:[a-z]:[\\/]|\\\\[^\\/\s]+[\\/]"
    r"|/volume\d+/|(?:smb|nfs|file|nas)://))",
    re.IGNORECASE,
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class KnowledgeGraphPersistenceError(RuntimeError):
    """Stable, value-free AGE or mirror persistence failure."""


class UnsafeKnowledgeGraphPayload(ValueError):
    """A projection attempted to cross the durable graph safety boundary."""


ApprovalErrorCode = Literal[
    "knowledge_graph_approval_missing",
    "knowledge_graph_approval_rejected",
    "knowledge_graph_approval_expired",
    "knowledge_graph_approval_mismatch",
    "knowledge_graph_approval_invalid",
    "knowledge_graph_approval_unavailable",
]


class KnowledgeGraphApprovalError(RuntimeError):
    """Fail-closed approval failure carrying only a stable governance code."""

    def __init__(self, code: ApprovalErrorCode) -> None:
        self.code = code
        super().__init__(code)


class GraphWriteApprovalRequest(BaseModel):
    """Content-free identity sent to a provider-neutral approval repository."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_id: str = Field(pattern=r"^fact-batch:v1:[0-9a-f]{64}$")
    idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    projection_digest: str = Field(pattern=_SHA256_PATTERN)


class GraphWriteApproval(BaseModel):
    """Bounded approval receipt; no reviewer identity or free-text comment."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_id: str = Field(pattern=r"^fact-batch:v1:[0-9a-f]{64}$")
    idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    projection_digest: str = Field(pattern=_SHA256_PATTERN)
    approved: bool
    reviewer_generation_digest: str = Field(pattern=_SHA256_PATTERN)
    audit_generation_digest: str = Field(pattern=_SHA256_PATTERN)
    expires_at: datetime
    approval_digest: str | None = Field(
        default=None,
        pattern=r"^graph-approval:v1:[0-9a-f]{64}$",
    )

    @field_validator("expires_at")
    @classmethod
    def require_aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval_expiry_must_be_timezone_aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def bind_approval_digest(self) -> "GraphWriteApproval":
        expected = (
            "graph-approval:v1:"
            + sha256(
                _canonical_json(
                    {
                        "batch_id": self.batch_id,
                        "idempotency_key": self.idempotency_key,
                        "projection_digest": self.projection_digest,
                        "approved": self.approved,
                        "reviewer_generation_digest": self.reviewer_generation_digest,
                        "audit_generation_digest": self.audit_generation_digest,
                        "expires_at": self.expires_at.isoformat(),
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        if self.approval_digest is not None and self.approval_digest != expected:
            raise ValueError("approval_digest_mismatch")
        object.__setattr__(self, "approval_digest", expected)
        return self


@runtime_checkable
class GraphWriteApprovalRepository(Protocol):
    """Provider-neutral source of audited global-graph write decisions."""

    async def get_approval(
        self,
        request: GraphWriteApprovalRequest,
    ) -> GraphWriteApproval | None: ...


_LOAD_AGE = "LOAD 'age'"
_SET_SEARCH_PATH = "SET LOCAL search_path = ag_catalog, public"
_LOCK_IDENTITIES = """
SELECT pg_advisory_xact_lock(hashtextextended(lock_key, 0))
FROM unnest(%s::text[]) AS locks(lock_key)
ORDER BY lock_key
"""
_SELECT_BATCH = """
SELECT batch_id, projection_digest
FROM public.knowledge_graph_batches
WHERE idempotency_key = %s
FOR UPDATE
"""
_SELECT_NODES = """
SELECT node_id, labels, properties
FROM public.knowledge_graph_nodes
WHERE node_id = ANY(%s::text[])
ORDER BY node_id
FOR UPDATE
"""
_SELECT_EDGES = """
SELECT edge_id, edge_type, source_node_id, target_node_id, properties
FROM public.knowledge_graph_edges
WHERE edge_id = ANY(%s::text[])
ORDER BY edge_id
FOR UPDATE
"""
_UPSERT_NODES = """
INSERT INTO public.knowledge_graph_nodes (node_id, labels, properties, node_digest)
SELECT
    item->>'node_id',
    item->'labels',
    item->'properties',
    item->>'node_digest'
FROM jsonb_array_elements((%s)::jsonb) AS item
ON CONFLICT (node_id) DO UPDATE SET
    labels = EXCLUDED.labels,
    properties = EXCLUDED.properties,
    node_digest = EXCLUDED.node_digest,
    updated_at = now()
"""
_UPSERT_EDGES = """
INSERT INTO public.knowledge_graph_edges (
    edge_id, edge_type, source_node_id, target_node_id, properties, edge_digest
)
SELECT
    item->>'edge_id',
    item->>'edge_type',
    item->>'source_node_id',
    item->>'target_node_id',
    item->'properties',
    item->>'edge_digest'
FROM jsonb_array_elements((%s)::jsonb) AS item
ON CONFLICT (edge_id) DO UPDATE SET
    edge_type = EXCLUDED.edge_type,
    source_node_id = EXCLUDED.source_node_id,
    target_node_id = EXCLUDED.target_node_id,
    properties = EXCLUDED.properties,
    edge_digest = EXCLUDED.edge_digest,
    updated_at = now()
"""
_AGE_UPSERT_NODES = """
SELECT * FROM ag_catalog.cypher(
    'material_graph',
    $cypher$
    UNWIND $nodes AS item
    MERGE (node:KnowledgeNode {node_id: item.node_id})
    SET node.logical_labels_json = item.labels_json,
        node.properties_json = item.properties_json,
        node.node_digest = item.node_digest
    RETURN count(node)
    $cypher$,
    (%s)::ag_catalog.agtype
) AS (written ag_catalog.agtype)
"""
_AGE_UPSERT_EDGES = """
SELECT * FROM ag_catalog.cypher(
    'material_graph',
    $cypher$
    UNWIND $edges AS item
    MATCH (source:KnowledgeNode {node_id: item.source_node_id})
    MATCH (target:KnowledgeNode {node_id: item.target_node_id})
    MERGE (source)-[edge:KNOWLEDGE_EDGE {edge_id: item.edge_id}]->(target)
    SET edge.edge_type = item.edge_type,
        edge.properties_json = item.properties_json,
        edge.edge_digest = item.edge_digest
    RETURN count(edge)
    $cypher$,
    (%s)::ag_catalog.agtype
) AS (written ag_catalog.agtype)
"""
_INSERT_BATCH = """
INSERT INTO public.knowledge_graph_batches (
    idempotency_key,
    batch_id,
    projection_digest,
    node_count,
    edge_count,
    approval_status,
    approval_digest,
    reviewer_generation_digest,
    audit_generation_digest,
    approval_expires_at
)
VALUES (%s, %s, %s, %s, %s, 'approved', %s, %s, %s, %s)
RETURNING idempotency_key
"""


def _required_identity(value: str | None) -> str:
    if value is None:  # pragma: no cover - FactBatch validation always fills identities
        raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error")
    return value


def _encoded(value: object) -> str:
    rendered = _canonical_json(value)
    if len(rendered.encode("utf-8")) > _MAX_PARAMETER_BYTES:
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
    return rendered


def _reject_internal_location(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
            normalized = key.strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_GRAPH_KEYS:
                raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
            _reject_internal_location(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _reject_internal_location(nested)
        return
    if isinstance(value, str) and _INTERNAL_LOCATION.search(value):
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")


def _safe_properties(properties: object) -> dict[str, object]:
    try:
        _reject_internal_location(properties)
        return _validated_json_object(properties, field="graph.properties")
    except (UnsafeDurablePayload, TypeError, ValueError):
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload") from None


def _validate_projection(projection: object) -> PropertyGraphProjection:
    try:
        if isinstance(projection, PropertyGraphProjection):
            projection = projection.model_dump(mode="python")
        validated = PropertyGraphProjection.model_validate(projection)
    except (ValidationError, TypeError, ValueError):
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload") from None

    if len(validated.nodes) > _MAX_GRAPH_NODES or len(validated.edges) > _MAX_GRAPH_EDGES:
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")

    safe_nodes: list[PropertyGraphNode] = []
    for node in validated.nodes:
        if node.labels not in _ALLOWED_LABEL_SETS:
            raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
        safe_nodes.append(
            node.model_copy(update={"properties": _safe_properties(node.properties)}, deep=True)
        )

    safe_edges: list[PropertyGraphEdge] = []
    for edge in validated.edges:
        if edge.edge_type not in _ALLOWED_EDGE_TYPES:
            raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
        safe_edges.append(
            edge.model_copy(update={"properties": _safe_properties(edge.properties)}, deep=True)
        )

    safe_projection = PropertyGraphProjection(nodes=tuple(safe_nodes), edges=tuple(safe_edges))
    if len(_canonical_json(safe_projection).encode("utf-8")) > _MAX_PROJECTION_BYTES:
        raise UnsafeKnowledgeGraphPayload("unsafe_knowledge_graph_payload")
    return safe_projection


def _projection_digest(projection: PropertyGraphProjection) -> str:
    return sha256(_canonical_json(projection).encode("utf-8")).hexdigest()


def build_graph_write_approval_request(batch: FactBatch) -> GraphWriteApprovalRequest:
    """Bind a review decision to the exact safe projection AGE will persist."""

    validated = validate_fact_batch(batch)
    projection = _validate_projection(project_fact_batch(validated))
    return GraphWriteApprovalRequest(
        batch_id=_required_identity(validated.batch_id),
        idempotency_key=_required_identity(validated.idempotency_key),
        projection_digest=_projection_digest(projection),
    )


def _item_digest(item: PropertyGraphNode | PropertyGraphEdge) -> str:
    return sha256(_canonical_json(item).encode("utf-8")).hexdigest()


def _row(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error")
    return value


def _json_from_row(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error") from None
    return value


def _stored_node(value: object) -> PropertyGraphNode:
    row = _row(value)
    try:
        return _validate_projection(
            PropertyGraphProjection(
                nodes=(
                    PropertyGraphNode(
                        node_id=row.get("node_id"),
                        labels=_json_from_row(row.get("labels")),
                        properties=_json_from_row(row.get("properties")),
                    ),
                ),
                edges=(),
            )
        ).nodes[0]
    except (ValidationError, UnsafeKnowledgeGraphPayload, TypeError, ValueError):
        raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error") from None


def _stored_edge(value: object) -> PropertyGraphEdge:
    row = _row(value)
    try:
        edge = PropertyGraphEdge(
            edge_id=row.get("edge_id"),
            edge_type=row.get("edge_type"),
            source_node_id=row.get("source_node_id"),
            target_node_id=row.get("target_node_id"),
            properties=_json_from_row(row.get("properties")),
        )
        if edge.edge_type not in _ALLOWED_EDGE_TYPES:
            raise ValueError("unsupported stored edge type")
        return edge.model_copy(update={"properties": _safe_properties(edge.properties)}, deep=True)
    except (ValidationError, UnsafeKnowledgeGraphPayload, TypeError, ValueError):
        raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error") from None


def _unique_by_id(
    values: Sequence[PropertyGraphNode] | Sequence[PropertyGraphEdge],
) -> dict[str, PropertyGraphNode | PropertyGraphEdge]:
    result: dict[str, PropertyGraphNode | PropertyGraphEdge] = {}
    for value in values:
        identity = value.node_id if isinstance(value, PropertyGraphNode) else value.edge_id
        if identity in result:
            raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error")
        result[identity] = value
    return result


def _node_rows(nodes: Sequence[PropertyGraphNode]) -> list[dict[str, object]]:
    return [
        {
            "node_id": node.node_id,
            "labels": list(node.labels),
            "properties": node.properties,
            "node_digest": _item_digest(node),
        }
        for node in nodes
    ]


def _edge_rows(edges: Sequence[PropertyGraphEdge]) -> list[dict[str, object]]:
    return [
        {
            "edge_id": edge.edge_id,
            "edge_type": edge.edge_type,
            "source_node_id": edge.source_node_id,
            "target_node_id": edge.target_node_id,
            "properties": edge.properties,
            "edge_digest": _item_digest(edge),
        }
        for edge in edges
    ]


def _age_node_parameter(nodes: Sequence[PropertyGraphNode]) -> str:
    return _encoded(
        {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "labels_json": _canonical_json(list(node.labels)),
                    "properties_json": _canonical_json(node.properties),
                    "node_digest": _item_digest(node),
                }
                for node in nodes
            ]
        }
    )


def _age_edge_parameter(edges: Sequence[PropertyGraphEdge]) -> str:
    return _encoded(
        {
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type,
                    "source_node_id": edge.source_node_id,
                    "target_node_id": edge.target_node_id,
                    "properties_json": _canonical_json(edge.properties),
                    "edge_digest": _item_digest(edge),
                }
                for edge in edges
            ]
        }
    )


class PostgresAGEGlobalKnowledgeGraphWriter:
    """Approved, idempotent and atomic PostgreSQL/Apache AGE graph writer."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        approval_repository: GraphWriteApprovalRepository,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._approval_repository = approval_repository
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    async def _require_approval(
        self,
        request: GraphWriteApprovalRequest,
    ) -> GraphWriteApproval:
        try:
            raw = await self._approval_repository.get_approval(request)
        except KnowledgeGraphApprovalError:
            raise
        except Exception:
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_unavailable") from None
        if raw is None:
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_missing")
        try:
            if isinstance(raw, GraphWriteApproval):
                raw = raw.model_dump(mode="python")
            approval = GraphWriteApproval.model_validate(raw)
        except (ValidationError, TypeError, ValueError):
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_invalid") from None
        if (
            approval.batch_id != request.batch_id
            or approval.idempotency_key != request.idempotency_key
            or approval.projection_digest != request.projection_digest
        ):
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_mismatch")
        if not approval.approved:
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_rejected")
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_invalid")
        if approval.expires_at <= now.astimezone(timezone.utc):
            raise KnowledgeGraphApprovalError("knowledge_graph_approval_expired")
        return approval

    async def write_batch(self, batch: FactBatch) -> FactWriteResult:
        validated = validate_fact_batch(batch)
        projection = _validate_projection(project_fact_batch(validated))
        batch_id = _required_identity(validated.batch_id)
        idempotency_key = _required_identity(validated.idempotency_key)
        digest = _projection_digest(projection)
        approval = await self._require_approval(
            GraphWriteApprovalRequest(
                batch_id=batch_id,
                idempotency_key=idempotency_key,
                projection_digest=digest,
            )
        )

        locks = sorted(
            {
                f"knowledge-graph-batch:{idempotency_key}",
                *(f"knowledge-graph-node:{node.node_id}" for node in projection.nodes),
                *(f"knowledge-graph-edge:{edge.edge_id}" for edge in projection.edges),
            }
        )
        try:
            async with self._pool.connection() as connection:
                async with connection.transaction():
                    await connection.execute(_LOAD_AGE)
                    await connection.execute(_SET_SEARCH_PATH)
                    await connection.execute(_LOCK_IDENTITIES, (locks,))

                    cursor = await connection.execute(_SELECT_BATCH, (idempotency_key,))
                    existing_batch = await cursor.fetchone()
                    if existing_batch is not None:
                        stored_digest = _row(existing_batch).get("projection_digest")
                        if stored_digest != digest:
                            raise KnowledgeGraphConflict("knowledge_graph_conflict")
                        return FactWriteResult(
                            batch_id=batch_id,
                            idempotency_key=idempotency_key,
                            status="already_present",
                            node_count=len(projection.nodes),
                            edge_count=len(projection.edges),
                        )

                    cursor = await connection.execute(
                        _SELECT_NODES,
                        ([node.node_id for node in projection.nodes],),
                    )
                    stored_nodes = [_stored_node(row) for row in await cursor.fetchall()]
                    existing_nodes = _unique_by_id(stored_nodes)
                    node_updates: list[PropertyGraphNode] = []
                    for node in projection.nodes:
                        existing = existing_nodes.get(node.node_id)
                        if existing is None or existing == node:
                            node_updates.append(node)
                            continue
                        assert isinstance(existing, PropertyGraphNode)
                        merged = _merge_entity_aliases(existing, node)
                        if merged is None:
                            raise KnowledgeGraphConflict("knowledge_graph_conflict")
                        node_updates.append(
                            merged.model_copy(
                                update={"properties": _safe_properties(merged.properties)},
                                deep=True,
                            )
                        )

                    cursor = await connection.execute(
                        _SELECT_EDGES,
                        ([edge.edge_id for edge in projection.edges],),
                    )
                    stored_edges = [_stored_edge(row) for row in await cursor.fetchall()]
                    existing_edges = _unique_by_id(stored_edges)
                    for edge in projection.edges:
                        existing = existing_edges.get(edge.edge_id)
                        if existing is not None and existing != edge:
                            raise KnowledgeGraphConflict("knowledge_graph_conflict")

                    node_payload = _encoded(_node_rows(node_updates))
                    edge_payload = _encoded(_edge_rows(projection.edges))
                    await connection.execute(_UPSERT_NODES, (node_payload,))
                    await connection.execute(_UPSERT_EDGES, (edge_payload,))
                    await connection.execute(
                        _AGE_UPSERT_NODES, (_age_node_parameter(node_updates),)
                    )
                    await connection.execute(
                        _AGE_UPSERT_EDGES,
                        (_age_edge_parameter(projection.edges),),
                    )
                    cursor = await connection.execute(
                        _INSERT_BATCH,
                        (
                            idempotency_key,
                            batch_id,
                            digest,
                            len(projection.nodes),
                            len(projection.edges),
                            _required_identity(approval.approval_digest),
                            approval.reviewer_generation_digest,
                            approval.audit_generation_digest,
                            approval.expires_at,
                        ),
                    )
                    if await cursor.fetchone() is None:
                        raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error")
        except (KnowledgeGraphConflict, KnowledgeGraphPersistenceError):
            raise
        except Exception:
            raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error") from None

        return FactWriteResult(
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            status="written",
            node_count=len(projection.nodes),
            edge_count=len(projection.edges),
        )


__all__ = [
    "AGE_GRAPH_NAME",
    "AGE_GRAPH_SCHEMA_VERSION",
    "GraphWriteApproval",
    "GraphWriteApprovalRepository",
    "GraphWriteApprovalRequest",
    "KnowledgeGraphApprovalError",
    "KnowledgeGraphPersistenceError",
    "PHYSICAL_EDGE_TYPE",
    "PHYSICAL_NODE_LABEL",
    "PostgresAGEGlobalKnowledgeGraphWriter",
    "UnsafeKnowledgeGraphPayload",
    "build_graph_write_approval_request",
]
