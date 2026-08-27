"""Evidence-gated, material-system-neutral contracts for durable graph facts.

The contracts deliberately exclude source bytes, retained fragment text, parser
output, arbitrary metadata, and credentials.  A production graph adapter only
receives the safe property-graph projection defined in this module.
"""

from __future__ import annotations

import asyncio
import json
import math
import unicodedata
from copy import deepcopy
from hashlib import sha256
from typing import Annotated, Any, Literal, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from .models import SourceLocator


EvidenceQuality = Literal["high", "medium", "low", "unknown"]
AssertionStatus = Literal["affirmed", "negated", "uncertain"]
EvidenceRole = Literal["supports", "contradicts", "context"]
SubjectRole = Literal["material", "sample"]
ConditionText = Annotated[str, StringConstraints(min_length=1, max_length=500)]
ConditionScalar = ConditionText | bool | float

# Open vocabulary: these names establish interoperable core roles without
# rejecting domain packs that introduce metals, ceramics, catalysts, devices,
# biological materials, or future entity kinds.
CORE_ENTITY_TYPES = frozenset(
    {
        "material",
        "composition",
        "process",
        "sample",
        "test_method",
        "application",
        "source",
    }
)

_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "credential",
    "password",
    "secret",
    "session",
    "token",
)


class FactContractError(ValueError):
    """Stable, content-free validation failure for untrusted fact payloads."""


class KnowledgeGraphConflict(RuntimeError):
    """Fail-closed conflict that intentionally omits submitted graph content."""


def _clean_text(value: str, *, error_code: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized or any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError(error_code)
    return normalized


def _canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, payload: object) -> str:
    digest = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:v1:{digest}"


def _model_sort_key(value: BaseModel) -> str:
    return _canonical_json(value)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _sorted_unique_models(
    values: tuple[ModelT, ...],
    *,
    identity: str,
) -> tuple[ModelT, ...]:
    ordered = tuple(sorted(values, key=lambda value: str(getattr(value, identity))))
    seen: set[str] = set()
    for value in ordered:
        key = str(getattr(value, identity))
        if key in seen:
            raise ValueError("duplicate_contract_item")
        seen.add(key)
    return ordered


def _validated_identifier_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("invalid_identifiers")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("invalid_identifiers")
        key = _clean_text(raw_key, error_code="invalid_identifier_key").casefold()
        key = key.replace("-", "_").replace(" ", "_")
        if any(marker in key for marker in _SENSITIVE_KEY_MARKERS):
            raise ValueError("sensitive_identifier_forbidden")
        identifier = _clean_text(raw_value, error_code="invalid_identifier_value")
        if len(key) > 100 or len(identifier) > 512:
            raise ValueError("invalid_identifiers")
        existing = normalized.get(key)
        if existing is not None and existing != identifier:
            raise ValueError("ambiguous_identifier")
        normalized[key] = identifier
    return dict(sorted(normalized.items()))


class EntityRef(BaseModel):
    """Content-addressed entity reference without material-family assumptions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entity_id: str | None = Field(
        default=None,
        pattern=r"^entity:v1:[0-9a-f]{64}$",
    )
    entity_type: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Open entity kind; interoperable core roles include material, composition, "
            "process, sample, test_method, application, and source."
        ),
    )
    canonical_name: str = Field(min_length=1, max_length=500)
    aliases: tuple[str, ...] = Field(default=(), max_length=50)
    identifiers: dict[str, str] = Field(default_factory=dict, max_length=50)

    @field_validator("entity_type")
    @classmethod
    def normalize_entity_type(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_entity_type").casefold()

    @field_validator("canonical_name")
    @classmethod
    def normalize_canonical_name(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_canonical_name")

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: dict[str, str] = {}
        for value in values:
            alias = _clean_text(value, error_code="invalid_alias")
            if len(alias) > 500:
                raise ValueError("invalid_alias")
            normalized.setdefault(alias.casefold(), alias)
        return tuple(normalized[key] for key in sorted(normalized))

    @field_validator("identifiers", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> dict[str, str]:
        return _validated_identifier_map(value)

    @model_validator(mode="after")
    def fill_entity_id(self) -> "EntityRef":
        expected = _stable_id(
            "entity",
            {
                "entity_type": self.entity_type,
                "canonical_name": self.canonical_name.casefold(),
                "identifiers": self.identifiers,
            },
        )
        if self.entity_id is not None and self.entity_id != expected:
            raise ValueError("entity_id_mismatch")
        object.__setattr__(self, "entity_id", expected)
        return self


class ExtractionProvenance(BaseModel):
    """Exact extractor generation and optional model generation provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    extractor_name: str = Field(min_length=1, max_length=200)
    extractor_version: str = Field(min_length=1, max_length=100)
    generation_id: str = Field(min_length=1, max_length=300)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    model_version: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator(
        "extractor_name",
        "extractor_version",
        "generation_id",
        "model_name",
        "model_version",
    )
    @classmethod
    def normalize_provenance_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_text(value, error_code="invalid_extraction_provenance")

    @model_validator(mode="after")
    def validate_model_pair(self) -> "ExtractionProvenance":
        if (self.model_name is None) != (self.model_version is None):
            raise ValueError("model_provenance_incomplete")
        return self


class _ConditionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    value: ConditionScalar
    unit: str = Field(min_length=1, max_length=100)

    @field_validator("value")
    @classmethod
    def validate_condition_value(cls, value: ConditionScalar) -> ConditionScalar:
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("condition_value_not_finite")
            return value
        if isinstance(value, str):
            return _clean_text(value, error_code="invalid_condition_value")
        return value

    @field_validator("unit")
    @classmethod
    def normalize_condition_unit(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_condition_unit")


class TestCondition(_ConditionBase):
    """One explicit condition under which a property or relation was tested."""

    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_test_condition_name")


class ProcessCondition(_ConditionBase):
    """One process parameter attached to a fact's material or sample state."""

    process_step: str = Field(min_length=1, max_length=200)
    parameter: str = Field(min_length=1, max_length=200)

    @field_validator("process_step", "parameter")
    @classmethod
    def normalize_process_text(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_process_condition")


class EvidenceLink(BaseModel):
    """Citable link to a retained fragment; source text is intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    evidence_ref_id: str | None = Field(
        default=None,
        pattern=r"^evidence:v1:[0-9a-f]{64}$",
    )
    fragment_id: UUID
    source_id: UUID
    locator: SourceLocator
    role: EvidenceRole = "supports"

    @model_validator(mode="after")
    def validate_location_and_id(self) -> "EvidenceLink":
        if len(self.locator.root_id) > 100 or any(
            len(anchor) > 500
            for anchor in (
                self.locator.section,
                self.locator.table,
                self.locator.figure,
            )
            if anchor is not None
        ):
            raise ValueError("evidence_location_too_large")
        has_anchor = any(
            (
                self.locator.page is not None,
                bool(self.locator.section),
                bool(self.locator.table),
                bool(self.locator.figure),
                self.locator.block_index is not None,
            )
        )
        if not has_anchor:
            raise ValueError("evidence_location_required")
        expected = _stable_id(
            "evidence",
            {
                "fragment_id": str(self.fragment_id),
                "source_id": str(self.source_id),
                "source_uri": self.public_source_uri,
            },
        )
        if self.evidence_ref_id is not None and self.evidence_ref_id != expected:
            raise ValueError("evidence_ref_id_mismatch")
        object.__setattr__(self, "evidence_ref_id", expected)
        return self

    @property
    def public_source_uri(self) -> str:
        return self.locator.to_public_uri(self.source_id)


class RelationAssertion(BaseModel):
    """Evidence-bearing, epistemically qualified relation between two entities."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    relation_id: str | None = Field(
        default=None,
        pattern=r"^relation:v1:[0-9a-f]{64}$",
    )
    subject: EntityRef
    predicate: str = Field(min_length=1, max_length=200)
    object: EntityRef
    test_conditions: tuple[TestCondition, ...] = Field(default=(), max_length=100)
    process_conditions: tuple[ProcessCondition, ...] = Field(default=(), max_length=100)
    evidence: tuple[EvidenceLink, ...] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_quality: EvidenceQuality
    assertion_status: AssertionStatus
    extraction: ExtractionProvenance

    @field_validator("predicate")
    @classmethod
    def normalize_predicate(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_predicate").casefold()

    @field_validator("test_conditions")
    @classmethod
    def order_test_conditions(cls, values: tuple[TestCondition, ...]) -> tuple[TestCondition, ...]:
        return tuple(sorted(values, key=_model_sort_key))

    @field_validator("process_conditions")
    @classmethod
    def order_process_conditions(
        cls, values: tuple[ProcessCondition, ...]
    ) -> tuple[ProcessCondition, ...]:
        return tuple(sorted(values, key=_model_sort_key))

    @field_validator("evidence")
    @classmethod
    def order_evidence(cls, values: tuple[EvidenceLink, ...]) -> tuple[EvidenceLink, ...]:
        return _sorted_unique_models(values, identity="evidence_ref_id")

    @model_validator(mode="after")
    def fill_relation_id(self) -> "RelationAssertion":
        _enforce_unknown_quality_gate(
            unknown_context=_unknown_test_context(
                test_method=None,
                test_conditions=self.test_conditions,
            ),
            evidence_quality=self.evidence_quality,
            confidence=self.confidence,
        )
        expected = _stable_id(
            "relation",
            {
                "subject_id": self.subject.entity_id,
                "predicate": self.predicate,
                "object_id": self.object.entity_id,
                "test_conditions": [item.model_dump(mode="json") for item in self.test_conditions],
                "process_conditions": [
                    item.model_dump(mode="json") for item in self.process_conditions
                ],
                "evidence_ref_ids": [item.evidence_ref_id for item in self.evidence],
                "confidence": self.confidence,
                "evidence_quality": self.evidence_quality,
                "assertion_status": self.assertion_status,
                "extraction": self.extraction.model_dump(mode="json"),
            },
        )
        if self.relation_id is not None and self.relation_id != expected:
            raise ValueError("relation_id_mismatch")
        object.__setattr__(self, "relation_id", expected)
        return self


class PropertyObservation(BaseModel):
    """A numerical material/sample property with mandatory test provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    observation_id: str | None = Field(
        default=None,
        pattern=r"^observation:v1:[0-9a-f]{64}$",
    )
    subject: EntityRef
    subject_role: SubjectRole
    property_name: str = Field(min_length=1, max_length=300)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=100)
    test_method: str = Field(min_length=1, max_length=300)
    test_conditions: tuple[TestCondition, ...] = Field(min_length=1, max_length=100)
    process_conditions: tuple[ProcessCondition, ...] = Field(default=(), max_length=100)
    evidence: tuple[EvidenceLink, ...] = Field(min_length=1, max_length=50)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_quality: EvidenceQuality
    assertion_status: AssertionStatus
    extraction: ExtractionProvenance

    @field_validator("property_name", "unit", "test_method")
    @classmethod
    def normalize_observation_text(cls, value: str) -> str:
        return _clean_text(value, error_code="invalid_property_observation")

    @field_validator("test_conditions")
    @classmethod
    def order_test_conditions(cls, values: tuple[TestCondition, ...]) -> tuple[TestCondition, ...]:
        return tuple(sorted(values, key=_model_sort_key))

    @field_validator("process_conditions")
    @classmethod
    def order_process_conditions(
        cls, values: tuple[ProcessCondition, ...]
    ) -> tuple[ProcessCondition, ...]:
        return tuple(sorted(values, key=_model_sort_key))

    @field_validator("evidence")
    @classmethod
    def order_evidence(cls, values: tuple[EvidenceLink, ...]) -> tuple[EvidenceLink, ...]:
        return _sorted_unique_models(values, identity="evidence_ref_id")

    @model_validator(mode="after")
    def fill_observation_id(self) -> "PropertyObservation":
        allowed_subject_types = {
            "material": frozenset({"material", "composition"}),
            "sample": frozenset({"sample"}),
        }
        if self.subject.entity_type not in allowed_subject_types[self.subject_role]:
            raise ValueError("observation_subject_role_mismatch")
        _enforce_unknown_quality_gate(
            unknown_context=_unknown_test_context(
                test_method=self.test_method,
                test_conditions=self.test_conditions,
            ),
            evidence_quality=self.evidence_quality,
            confidence=self.confidence,
        )
        expected = _stable_id(
            "observation",
            {
                "subject_id": self.subject.entity_id,
                "subject_role": self.subject_role,
                "property_name": self.property_name.casefold(),
                "value": self.value,
                "unit": self.unit,
                "test_method": self.test_method.casefold(),
                "test_conditions": [item.model_dump(mode="json") for item in self.test_conditions],
                "process_conditions": [
                    item.model_dump(mode="json") for item in self.process_conditions
                ],
                "evidence_ref_ids": [item.evidence_ref_id for item in self.evidence],
                "confidence": self.confidence,
                "evidence_quality": self.evidence_quality,
                "assertion_status": self.assertion_status,
                "extraction": self.extraction.model_dump(mode="json"),
            },
        )
        if self.observation_id is not None and self.observation_id != expected:
            raise ValueError("observation_id_mismatch")
        object.__setattr__(self, "observation_id", expected)
        return self


def _unknown_test_context(
    *,
    test_method: str | None,
    test_conditions: tuple[TestCondition, ...],
) -> bool:
    if test_method is not None and test_method.casefold() == "unknown":
        return True
    return any(
        condition.name.casefold() in {"unknown", "unspecified"}
        or (isinstance(condition.value, str) and condition.value.casefold() == "unknown")
        or condition.unit.casefold() == "unknown"
        for condition in test_conditions
    )


def _enforce_unknown_quality_gate(
    *,
    unknown_context: bool,
    evidence_quality: EvidenceQuality,
    confidence: float,
) -> None:
    if unknown_context and (evidence_quality not in {"low", "unknown"} or confidence > 0.5):
        raise ValueError("unknown_test_context_requires_low_quality")


class FactBatch(BaseModel):
    """Atomic facts extracted from one evidence fragment by one generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_id: str | None = Field(
        default=None,
        pattern=r"^fact-batch:v1:[0-9a-f]{64}$",
    )
    idempotency_key: str | None = Field(
        default=None,
        pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$",
    )
    evidence_fragment_id: UUID
    extraction: ExtractionProvenance
    entities: tuple[EntityRef, ...] = Field(min_length=1, max_length=1000)
    relations: tuple[RelationAssertion, ...] = Field(default=(), max_length=5000)
    observations: tuple[PropertyObservation, ...] = Field(default=(), max_length=5000)

    @field_validator("entities")
    @classmethod
    def order_entities(cls, values: tuple[EntityRef, ...]) -> tuple[EntityRef, ...]:
        return _sorted_unique_models(values, identity="entity_id")

    @field_validator("relations")
    @classmethod
    def order_relations(
        cls, values: tuple[RelationAssertion, ...]
    ) -> tuple[RelationAssertion, ...]:
        return _sorted_unique_models(values, identity="relation_id")

    @field_validator("observations")
    @classmethod
    def order_observations(
        cls, values: tuple[PropertyObservation, ...]
    ) -> tuple[PropertyObservation, ...]:
        return _sorted_unique_models(values, identity="observation_id")

    @model_validator(mode="after")
    def validate_boundary_and_fill_ids(self) -> "FactBatch":
        if not self.relations and not self.observations:
            raise ValueError("fact_batch_requires_facts")

        entities = {entity.entity_id: entity for entity in self.entities}
        facts: tuple[RelationAssertion | PropertyObservation, ...] = (
            *self.relations,
            *self.observations,
        )
        for fact in facts:
            if fact.extraction != self.extraction:
                raise ValueError("fact_extractor_generation_mismatch")
            if not any(link.fragment_id == self.evidence_fragment_id for link in fact.evidence):
                raise ValueError("fact_evidence_fragment_mismatch")

            referenced = (
                (fact.subject, fact.object)
                if isinstance(fact, RelationAssertion)
                else (fact.subject,)
            )
            for entity in referenced:
                stored = entities.get(entity.entity_id)
                if stored is None or stored != entity:
                    raise ValueError("fact_entity_not_declared")

        expected_idempotency_key = _stable_id(
            "fact-batch-idempotency",
            {
                "evidence_fragment_id": str(self.evidence_fragment_id),
                "extractor_generation_id": self.extraction.generation_id,
            },
        )
        if self.idempotency_key is not None and self.idempotency_key != expected_idempotency_key:
            raise ValueError("fact_batch_idempotency_key_mismatch")
        object.__setattr__(self, "idempotency_key", expected_idempotency_key)

        expected_batch_id = _stable_id(
            "fact-batch",
            {
                "idempotency_key": expected_idempotency_key,
                "entity_ids": [entity.entity_id for entity in self.entities],
                "relation_ids": [relation.relation_id for relation in self.relations],
                "observation_ids": [
                    observation.observation_id for observation in self.observations
                ],
            },
        )
        if self.batch_id is not None and self.batch_id != expected_batch_id:
            raise ValueError("fact_batch_id_mismatch")
        object.__setattr__(self, "batch_id", expected_batch_id)
        return self


class PropertyGraphNode(BaseModel):
    """Provider-neutral node payload containing only graph-safe JSON values."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    node_id: str = Field(min_length=1, max_length=200)
    labels: tuple[str, ...] = Field(min_length=1)
    properties: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def normalize_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            sorted({_clean_text(value, error_code="invalid_graph_label") for value in values})
        )


class PropertyGraphEdge(BaseModel):
    """Provider-neutral directed edge payload with deterministic identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    edge_id: str = Field(pattern=r"^edge:v1:[0-9a-f]{64}$")
    edge_type: str = Field(min_length=1, max_length=100)
    source_node_id: str = Field(min_length=1, max_length=200)
    target_node_id: str = Field(min_length=1, max_length=200)
    properties: dict[str, JsonValue] = Field(default_factory=dict)


class PropertyGraphProjection(BaseModel):
    """Atomic graph write payload; no source path, text, bytes, or credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    nodes: tuple[PropertyGraphNode, ...]
    edges: tuple[PropertyGraphEdge, ...]

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "PropertyGraphProjection":
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(node_ids) != len(set(node_ids)) or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("duplicate_graph_identifier")
        known_nodes = set(node_ids)
        if any(
            edge.source_node_id not in known_nodes or edge.target_node_id not in known_nodes
            for edge in self.edges
        ):
            raise ValueError("graph_edge_endpoint_missing")
        return self


class FactWriteResult(BaseModel):
    """Secret-free acknowledgement suitable for checkpoints and traces."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch_id: str = Field(pattern=r"^fact-batch:v1:[0-9a-f]{64}$")
    idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    status: Literal["written", "already_present"]
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)


def _required_id(value: str | None) -> str:
    if value is None:  # pragma: no cover - validated models always populate IDs
        raise FactContractError("invalid_fact_batch")
    return value


def _edge(
    edge_type: str,
    source_node_id: str,
    target_node_id: str,
    *,
    properties: dict[str, JsonValue] | None = None,
) -> PropertyGraphEdge:
    safe_properties = properties or {}
    return PropertyGraphEdge(
        edge_id=_stable_id(
            "edge",
            {
                "edge_type": edge_type,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "properties": safe_properties,
            },
        ),
        edge_type=edge_type,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        properties=safe_properties,
    )


def _evidence_node(link: EvidenceLink) -> PropertyGraphNode:
    properties: dict[str, JsonValue] = {
        "fragment_id": str(link.fragment_id),
        "source_id": str(link.source_id),
        "source_uri": link.public_source_uri,
    }
    anchors = link.locator.model_dump(
        mode="json",
        exclude={"root_id", "relative_path"},
        exclude_none=True,
    )
    properties.update(anchors)
    return PropertyGraphNode(
        node_id=_required_id(link.evidence_ref_id),
        labels=("EvidenceFragment",),
        properties=properties,
    )


def _entity_labels(entity_type: str) -> tuple[str, ...]:
    core_labels = {
        "material": ("Entity", "MaterialEntity"),
        "composition": ("Composition", "Entity", "MaterialEntity"),
        "sample": ("Entity", "MaterialEntity", "Sample"),
        "process": ("Entity", "Process"),
        "test_method": ("Entity", "TestMethod"),
        "application": ("Application", "Entity"),
        "source": ("Entity", "Source"),
    }
    return core_labels.get(entity_type, ("Entity",))


def _add_node(
    nodes: dict[str, PropertyGraphNode],
    candidate: PropertyGraphNode,
) -> None:
    existing = nodes.get(candidate.node_id)
    if existing is not None and existing != candidate:
        raise FactContractError("invalid_graph_projection")
    nodes[candidate.node_id] = candidate


def _add_edge(
    edges: dict[str, PropertyGraphEdge],
    candidate: PropertyGraphEdge,
) -> None:
    existing = edges.get(candidate.edge_id)
    if existing is not None and existing != candidate:
        raise FactContractError("invalid_graph_projection")
    edges[candidate.edge_id] = candidate


def project_fact_batch(batch: FactBatch) -> PropertyGraphProjection:
    """Project facts into an auditable graph without internal paths or source text."""

    validated = validate_fact_batch(batch.model_dump(mode="python"))
    nodes: dict[str, PropertyGraphNode] = {}
    edges: dict[str, PropertyGraphEdge] = {}

    for entity in validated.entities:
        entity_id = _required_id(entity.entity_id)
        _add_node(
            nodes,
            PropertyGraphNode(
                node_id=entity_id,
                labels=_entity_labels(entity.entity_type),
                properties={
                    "entity_type": entity.entity_type,
                    "canonical_name": entity.canonical_name,
                    "aliases": list(entity.aliases),
                    "identifiers": entity.identifiers,
                },
            ),
        )

    for relation in validated.relations:
        relation_id = _required_id(relation.relation_id)
        subject_id = _required_id(relation.subject.entity_id)
        object_id = _required_id(relation.object.entity_id)
        _add_node(
            nodes,
            PropertyGraphNode(
                node_id=relation_id,
                labels=("Fact", "RelationAssertion"),
                properties={
                    "predicate": relation.predicate,
                    "confidence": relation.confidence,
                    "evidence_quality": relation.evidence_quality,
                    "assertion_status": relation.assertion_status,
                    "test_conditions": [
                        item.model_dump(mode="json") for item in relation.test_conditions
                    ],
                    "process_conditions": [
                        item.model_dump(mode="json") for item in relation.process_conditions
                    ],
                    "extraction": relation.extraction.model_dump(mode="json"),
                },
            ),
        )
        _add_edge(edges, _edge("ASSERTION_SUBJECT", subject_id, relation_id))
        _add_edge(edges, _edge("ASSERTION_OBJECT", relation_id, object_id))
        for link in relation.evidence:
            evidence = _evidence_node(link)
            _add_node(nodes, evidence)
            _add_edge(
                edges,
                _edge(
                    "SUPPORTED_BY",
                    relation_id,
                    evidence.node_id,
                    properties={"role": link.role},
                ),
            )

    for observation in validated.observations:
        observation_id = _required_id(observation.observation_id)
        subject_id = _required_id(observation.subject.entity_id)
        _add_node(
            nodes,
            PropertyGraphNode(
                node_id=observation_id,
                labels=("Fact", "PropertyObservation"),
                properties={
                    "subject_role": observation.subject_role,
                    "property_name": observation.property_name,
                    "value": observation.value,
                    "unit": observation.unit,
                    "test_method": observation.test_method,
                    "test_conditions": [
                        item.model_dump(mode="json") for item in observation.test_conditions
                    ],
                    "process_conditions": [
                        item.model_dump(mode="json") for item in observation.process_conditions
                    ],
                    "confidence": observation.confidence,
                    "evidence_quality": observation.evidence_quality,
                    "assertion_status": observation.assertion_status,
                    "extraction": observation.extraction.model_dump(mode="json"),
                },
            ),
        )
        _add_edge(edges, _edge("OBSERVED_ON", observation_id, subject_id))
        for link in observation.evidence:
            evidence = _evidence_node(link)
            _add_node(nodes, evidence)
            _add_edge(
                edges,
                _edge(
                    "SUPPORTED_BY",
                    observation_id,
                    evidence.node_id,
                    properties={"role": link.role},
                ),
            )

    return PropertyGraphProjection(
        nodes=tuple(nodes[key] for key in sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
    )


def validate_fact_batch(payload: object) -> FactBatch:
    """Validate untrusted input while exposing only a stable error code."""

    try:
        if isinstance(payload, FactBatch):
            payload = payload.model_dump(mode="python")
        return FactBatch.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        raise FactContractError("invalid_fact_batch") from None


def export_fact_batch_json_schema() -> dict[str, Any]:
    """Return an independent JSON Schema document for extractor boundaries."""

    return deepcopy(FactBatch.model_json_schema())


@runtime_checkable
class GlobalKnowledgeGraphWriter(Protocol):
    """Provider-neutral atomic writer implemented by graph database adapters."""

    async def write_batch(self, batch: FactBatch) -> FactWriteResult: ...


def _merge_entity_aliases(
    existing: PropertyGraphNode,
    candidate: PropertyGraphNode,
) -> PropertyGraphNode | None:
    if existing.labels != candidate.labels or "Entity" not in existing.labels:
        return None
    existing_properties = dict(existing.properties)
    candidate_properties = dict(candidate.properties)
    existing_aliases = existing_properties.pop("aliases", None)
    candidate_aliases = candidate_properties.pop("aliases", None)
    if existing_properties != candidate_properties:
        return None
    if not isinstance(existing_aliases, list) or not isinstance(candidate_aliases, list):
        return None
    if not all(isinstance(alias, str) for alias in (*existing_aliases, *candidate_aliases)):
        return None
    aliases = sorted(set(existing_aliases) | set(candidate_aliases), key=str.casefold)
    return existing.model_copy(
        update={"properties": {**existing_properties, "aliases": aliases}},
        deep=True,
    )


class InMemoryGlobalKnowledgeGraphWriter:
    """Atomic in-memory writer that retains only safe graph projections.

    The input ``FactBatch`` and its internal ``SourceLocator.relative_path`` are
    used transiently for validation and hashing.  They are never placed in the
    durable in-memory graph state.
    """

    def __init__(self) -> None:
        self._batch_digests: dict[str, str] = {}
        self._nodes: dict[str, PropertyGraphNode] = {}
        self._edges: dict[str, PropertyGraphEdge] = {}
        self._lock = asyncio.Lock()

    @property
    def batch_count(self) -> int:
        return len(self._batch_digests)

    async def write_batch(self, batch: FactBatch) -> FactWriteResult:
        validated = validate_fact_batch(batch)
        projection = project_fact_batch(validated)
        batch_id = _required_id(validated.batch_id)
        idempotency_key = _required_id(validated.idempotency_key)
        digest = sha256(
            _canonical_json(projection.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()

        async with self._lock:
            existing_digest = self._batch_digests.get(idempotency_key)
            if existing_digest is not None:
                if existing_digest != digest:
                    raise KnowledgeGraphConflict("knowledge_graph_conflict")
                return FactWriteResult(
                    batch_id=batch_id,
                    idempotency_key=idempotency_key,
                    status="already_present",
                    node_count=len(projection.nodes),
                    edge_count=len(projection.edges),
                )

            node_updates: dict[str, PropertyGraphNode] = {}
            for node in projection.nodes:
                existing = self._nodes.get(node.node_id)
                if existing is not None and existing != node:
                    merged = _merge_entity_aliases(existing, node)
                    if merged is None:
                        raise KnowledgeGraphConflict("knowledge_graph_conflict")
                    node_updates[node.node_id] = merged
                elif existing is None:
                    node_updates[node.node_id] = node
            for edge in projection.edges:
                existing = self._edges.get(edge.edge_id)
                if existing is not None and existing != edge:
                    raise KnowledgeGraphConflict("knowledge_graph_conflict")

            self._nodes.update(
                {node_id: node.model_copy(deep=True) for node_id, node in node_updates.items()}
            )
            self._edges.update(
                {edge.edge_id: edge.model_copy(deep=True) for edge in projection.edges}
            )
            self._batch_digests[idempotency_key] = digest

        return FactWriteResult(
            batch_id=batch_id,
            idempotency_key=idempotency_key,
            status="written",
            node_count=len(projection.nodes),
            edge_count=len(projection.edges),
        )

    async def snapshot(self) -> PropertyGraphProjection:
        """Return a defensive, deterministically ordered view of safe graph state."""

        async with self._lock:
            return PropertyGraphProjection(
                nodes=tuple(self._nodes[key].model_copy(deep=True) for key in sorted(self._nodes)),
                edges=tuple(self._edges[key].model_copy(deep=True) for key in sorted(self._edges)),
            )


__all__ = [
    "AssertionStatus",
    "CORE_ENTITY_TYPES",
    "EntityRef",
    "EvidenceLink",
    "EvidenceQuality",
    "EvidenceRole",
    "ExtractionProvenance",
    "FactBatch",
    "FactContractError",
    "FactWriteResult",
    "GlobalKnowledgeGraphWriter",
    "InMemoryGlobalKnowledgeGraphWriter",
    "KnowledgeGraphConflict",
    "ProcessCondition",
    "PropertyGraphEdge",
    "PropertyGraphNode",
    "PropertyGraphProjection",
    "PropertyObservation",
    "RelationAssertion",
    "SubjectRole",
    "TestCondition",
    "export_fact_batch_json_schema",
    "project_fact_batch",
    "validate_fact_batch",
]
