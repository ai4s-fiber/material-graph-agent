from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from uuid import UUID

import pytest
from pydantic import ValidationError

from material_graph.knowledge.facts import (
    CORE_ENTITY_TYPES,
    EntityRef,
    EvidenceLink,
    ExtractionProvenance,
    FactBatch,
    FactContractError,
    GlobalKnowledgeGraphWriter,
    InMemoryGlobalKnowledgeGraphWriter,
    KnowledgeGraphConflict,
    ProcessCondition,
    PropertyGraphEdge,
    PropertyGraphNode,
    PropertyGraphProjection,
    PropertyObservation,
    RelationAssertion,
    TestCondition as MaterialTestCondition,
    export_fact_batch_json_schema,
    project_fact_batch,
    validate_fact_batch,
)
from material_graph.knowledge.models import SourceLocator


SOURCE_ID = UUID("11111111-1111-4111-8111-111111111111")
FRAGMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
OTHER_FRAGMENT_ID = UUID("33333333-3333-4333-8333-333333333333")


def _extractor(generation_id: str = "extractor-generation-2026-07-27") -> ExtractionProvenance:
    return ExtractionProvenance(
        extractor_name="materials-fact-extractor",
        extractor_version="2.3.1",
        generation_id=generation_id,
        model_name="reasoning-model",
        model_version="2026-07-25",
    )


def _evidence(fragment_id: UUID = FRAGMENT_ID, *, page: int = 8) -> EvidenceLink:
    return EvidenceLink(
        fragment_id=fragment_id,
        source_id=SOURCE_ID,
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path="private/corpus/material-paper.pdf",
            page=page,
            section="Results",
            block_index=4,
        ),
        role="supports",
    )


def _entities() -> tuple[EntityRef, EntityRef]:
    material = EntityRef(
        entity_type="material",
        canonical_name="MX-17 composite",
        aliases=("MX17", "Material X17"),
        identifiers={"internal_material_id": "MX-17", "registry": "R-0042"},
    )
    additive = EntityRef(
        entity_type="component",
        canonical_name="Additive Z",
        identifiers={"registry": "A-009"},
    )
    return material, additive


def _observation(
    material: EntityRef,
    *,
    value: float = 2.74,
    fragment_id: UUID = FRAGMENT_ID,
    generation_id: str = "extractor-generation-2026-07-27",
    subject_role: str = "material",
) -> PropertyObservation:
    return PropertyObservation(
        subject=material,
        subject_role=subject_role,
        property_name="relative permittivity",
        value=value,
        unit="dimensionless",
        test_method="impedance spectroscopy",
        test_conditions=(
            MaterialTestCondition(name="frequency", value=1.0, unit="MHz"),
            MaterialTestCondition(name="temperature", value=23.0, unit="degC"),
        ),
        process_conditions=(
            ProcessCondition(
                process_step="curing",
                parameter="temperature",
                value=180.0,
                unit="degC",
            ),
        ),
        evidence=(_evidence(fragment_id),),
        confidence=0.91,
        evidence_quality="high",
        assertion_status="affirmed",
        extraction=_extractor(generation_id),
    )


def _relation(
    material: EntityRef,
    additive: EntityRef,
    *,
    fragment_id: UUID = FRAGMENT_ID,
    generation_id: str = "extractor-generation-2026-07-27",
) -> RelationAssertion:
    return RelationAssertion(
        subject=additive,
        predicate="reduces",
        object=material,
        test_conditions=(
            MaterialTestCondition(name="comparison_basis", value="same process", unit="n/a"),
        ),
        evidence=(_evidence(fragment_id),),
        confidence=0.84,
        evidence_quality="medium",
        assertion_status="uncertain",
        extraction=_extractor(generation_id),
    )


def _batch(
    *,
    fragment_id: UUID = FRAGMENT_ID,
    generation_id: str = "extractor-generation-2026-07-27",
    value: float = 2.74,
) -> FactBatch:
    material, additive = _entities()
    return FactBatch(
        evidence_fragment_id=fragment_id,
        extraction=_extractor(generation_id),
        entities=(material, additive),
        relations=(
            _relation(
                material,
                additive,
                fragment_id=fragment_id,
                generation_id=generation_id,
            ),
        ),
        observations=(
            _observation(
                material,
                value=value,
                fragment_id=fragment_id,
                generation_id=generation_id,
            ),
        ),
    )


def test_entity_fact_and_batch_ids_are_deterministic_and_order_independent() -> None:
    first = _batch()
    material, additive = _entities()
    reordered_material = EntityRef(
        entity_type=" MATERIAL ",
        canonical_name="  MX-17   composite ",
        aliases=("Material X17", "MX17", "MX17"),
        identifiers={"registry": "R-0042", "internal_material_id": "MX-17"},
    )
    second = FactBatch(
        evidence_fragment_id=FRAGMENT_ID,
        extraction=_extractor(),
        entities=(additive, reordered_material),
        relations=(_relation(reordered_material, additive),),
        observations=(_observation(reordered_material),),
    )

    assert reordered_material.entity_id == material.entity_id
    assert first.relations[0].relation_id == second.relations[0].relation_id
    assert first.observations[0].observation_id == second.observations[0].observation_id
    assert first.idempotency_key == second.idempotency_key
    assert first.batch_id == second.batch_id
    assert first.entities == tuple(sorted(first.entities, key=lambda item: item.entity_id or ""))


def test_entity_identity_is_stable_when_aliases_are_enriched() -> None:
    material, _ = _entities()
    enriched = EntityRef(
        entity_type=material.entity_type,
        canonical_name=material.canonical_name,
        aliases=(*material.aliases, "New laboratory alias"),
        identifiers=material.identifiers,
    )

    assert enriched.entity_id == material.entity_id
    assert enriched.aliases != material.aliases


def test_idempotency_key_binds_fragment_and_extractor_generation() -> None:
    baseline = _batch()
    changed_fragment = _batch(fragment_id=OTHER_FRAGMENT_ID)
    changed_generation = _batch(generation_id="extractor-generation-next")

    assert baseline.idempotency_key != changed_fragment.idempotency_key
    assert baseline.idempotency_key != changed_generation.idempotency_key
    assert changed_fragment.idempotency_key != changed_generation.idempotency_key


@pytest.mark.parametrize(
    ("update", "expected_field"),
    [
        ({"unit": ""}, "unit"),
        ({"test_method": ""}, "test_method"),
        ({"test_conditions": ()}, "test_conditions"),
        ({"evidence": ()}, "evidence"),
        ({"confidence": 1.01}, "confidence"),
        ({"value": float("nan")}, "value"),
    ],
)
def test_numeric_property_rejects_bare_or_invalid_values(
    update: dict[str, object], expected_field: str
) -> None:
    material, _ = _entities()
    payload = _observation(material).model_dump(mode="python")
    payload.update(update)

    with pytest.raises(ValidationError) as error:
        PropertyObservation.model_validate(payload)

    assert expected_field in str(error.value)


def test_unknown_method_and_condition_are_explicitly_supported() -> None:
    material, _ = _entities()
    payload = _observation(material).model_dump(mode="python")
    payload.update(
        {
            "test_method": "unknown",
            "test_conditions": (
                MaterialTestCondition(name="unspecified", value="unknown", unit="unknown"),
            ),
            "evidence_quality": "low",
            "confidence": 0.4,
        }
    )
    payload.pop("observation_id")

    observation = PropertyObservation.model_validate(payload)

    assert observation.test_method == "unknown"
    assert observation.test_conditions[0].value == "unknown"


def test_unknown_test_context_cannot_be_claimed_as_strong_evidence() -> None:
    material, _ = _entities()
    payload = _observation(material).model_dump(mode="python")
    payload["test_method"] = "unknown"

    with pytest.raises(ValidationError, match="unknown_test_context_requires_low_quality"):
        PropertyObservation.model_validate(payload)


def test_core_entity_roles_are_explicit_but_the_vocabulary_remains_open() -> None:
    assert {
        "material",
        "composition",
        "process",
        "sample",
        "test_method",
        "application",
        "source",
    } <= CORE_ENTITY_TYPES

    custom = EntityRef(entity_type="electrochemical_device", canonical_name="Cell-X")
    assert custom.entity_type == "electrochemical_device"


def test_numeric_conditions_require_units_and_finite_values() -> None:
    with pytest.raises(ValidationError):
        MaterialTestCondition(name="temperature", value=25.0, unit="")
    with pytest.raises(ValidationError):
        ProcessCondition(
            process_step="mixing",
            parameter="speed",
            value=float("inf"),
            unit="rpm",
        )


def test_evidence_requires_fragment_source_and_precise_location() -> None:
    with pytest.raises(ValidationError, match="evidence_location_required"):
        EvidenceLink(
            fragment_id=FRAGMENT_ID,
            source_id=SOURCE_ID,
            locator=SourceLocator(
                root_id="document_data_1",
                relative_path="private/corpus/material-paper.pdf",
            ),
            role="supports",
        )


def test_relation_and_observation_preserve_epistemic_and_extractor_fields() -> None:
    batch = _batch()
    relation = batch.relations[0]
    observation = batch.observations[0]

    assert relation.assertion_status == "uncertain"
    assert relation.evidence_quality == "medium"
    assert relation.confidence == pytest.approx(0.84)
    assert relation.extraction.model_version == "2026-07-25"
    assert observation.assertion_status == "affirmed"
    assert observation.evidence_quality == "high"
    assert observation.extraction.extractor_version == "2.3.1"


@pytest.mark.parametrize(
    "mutator",
    [
        "wrong_fragment",
        "wrong_generation",
        "missing_entity",
        "no_facts",
    ],
)
def test_fact_batch_rejects_cross_boundary_or_incomplete_content(mutator: str) -> None:
    material, additive = _entities()
    relation = _relation(material, additive)
    observation = _observation(material)
    payload: dict[str, object] = {
        "evidence_fragment_id": FRAGMENT_ID,
        "extraction": _extractor(),
        "entities": (material, additive),
        "relations": (relation,),
        "observations": (observation,),
    }
    if mutator == "wrong_fragment":
        payload["relations"] = (_relation(material, additive, fragment_id=OTHER_FRAGMENT_ID),)
    elif mutator == "wrong_generation":
        payload["observations"] = (
            _observation(material, generation_id="extractor-generation-next"),
        )
    elif mutator == "missing_entity":
        payload["entities"] = (material,)
    else:
        payload["relations"] = ()
        payload["observations"] = ()

    with pytest.raises(ValidationError):
        FactBatch.model_validate(payload)


def test_provided_ids_must_match_the_canonical_payload() -> None:
    material, additive = _entities()
    with pytest.raises(ValidationError, match="entity_id_mismatch"):
        EntityRef(
            entity_id="entity:v1:" + "0" * 64,
            entity_type=material.entity_type,
            canonical_name=material.canonical_name,
            aliases=material.aliases,
            identifiers=material.identifiers,
        )

    relation_payload = _relation(material, additive).model_dump(mode="python")
    relation_payload["relation_id"] = "relation:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="relation_id_mismatch"):
        RelationAssertion.model_validate(relation_payload)

    observation_payload = _observation(material).model_dump(mode="python")
    observation_payload["observation_id"] = "observation:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="observation_id_mismatch"):
        PropertyObservation.model_validate(observation_payload)


def test_projection_is_provider_neutral_citable_and_path_safe() -> None:
    projection = project_fact_batch(_batch())
    rendered = projection.model_dump_json()
    node_labels = {label for node in projection.nodes for label in node.labels}
    edge_types = {edge.edge_type for edge in projection.edges}

    assert {"MaterialEntity", "EvidenceFragment", "RelationAssertion", "PropertyObservation"} <= (
        node_labels
    )
    assert {"ASSERTION_SUBJECT", "ASSERTION_OBJECT", "OBSERVED_ON", "SUPPORTED_BY"} <= (edge_types)
    assert str(FRAGMENT_ID) in rendered
    assert f"source://document_data_1/{SOURCE_ID}" in rendered
    assert "private/corpus" not in rendered
    assert "material-paper.pdf" not in rendered
    assert "raw_pdf" not in rendered
    assert "complete_parser_output" not in rendered
    assert "credential" not in rendered


def test_projection_labels_core_roles_without_calling_every_entity_a_material() -> None:
    baseline = _batch()
    extra_entities = tuple(
        EntityRef(entity_type=entity_type, canonical_name=f"entity-{entity_type}")
        for entity_type in (
            "composition",
            "sample",
            "process",
            "test_method",
            "application",
            "source",
            "electrochemical_device",
        )
    )
    batch = FactBatch(
        evidence_fragment_id=baseline.evidence_fragment_id,
        extraction=baseline.extraction,
        entities=(*baseline.entities, *extra_entities),
        relations=baseline.relations,
        observations=baseline.observations,
    )

    projection = project_fact_batch(batch)
    labels_by_name = {
        str(node.properties.get("canonical_name")): set(node.labels)
        for node in projection.nodes
        if "canonical_name" in node.properties
    }

    assert "MaterialEntity" in labels_by_name["entity-composition"]
    assert "MaterialEntity" in labels_by_name["entity-sample"]
    assert labels_by_name["entity-process"] == {"Entity", "Process"}
    assert labels_by_name["entity-test_method"] == {"Entity", "TestMethod"}
    assert labels_by_name["entity-application"] == {"Application", "Entity"}
    assert labels_by_name["entity-source"] == {"Entity", "Source"}
    assert labels_by_name["entity-electrochemical_device"] == {"Entity"}


def test_observation_subject_role_must_match_the_entity_semantics() -> None:
    composition = EntityRef(entity_type="composition", canonical_name="Composition-X")
    material_observation = _observation(composition, subject_role="material")
    assert material_observation.subject.entity_type == "composition"

    sample = EntityRef(entity_type="sample", canonical_name="Sample-X")
    sample_observation = _observation(sample, subject_role="sample")
    assert sample_observation.subject.entity_type == "sample"

    application = EntityRef(entity_type="application", canonical_name="Application-X")
    with pytest.raises(ValidationError, match="observation_subject_role_mismatch"):
        _observation(application, subject_role="sample")
    material, _ = _entities()
    with pytest.raises(ValidationError, match="observation_subject_role_mismatch"):
        _observation(material, subject_role="sample")


def test_in_memory_writer_is_idempotent_and_keeps_only_safe_projection() -> None:
    async def scenario() -> None:
        writer = InMemoryGlobalKnowledgeGraphWriter()
        batch = _batch()

        created = await writer.write_batch(batch)
        repeated = await writer.write_batch(batch)
        snapshot = await writer.snapshot()

        assert created.status == "written"
        assert repeated.status == "already_present"
        assert created.node_count == len(snapshot.nodes)
        assert created.edge_count == len(snapshot.edges)
        assert writer.batch_count == 1
        assert "private/corpus" not in snapshot.model_dump_json()

    asyncio.run(scenario())


def test_writer_idempotency_is_independent_of_private_storage_path() -> None:
    async def scenario() -> None:
        writer = InMemoryGlobalKnowledgeGraphWriter()
        baseline = _batch()
        relocated_payload = baseline.model_dump(mode="python")
        for relation in relocated_payload["relations"]:
            relation["evidence"][0]["locator"]["relative_path"] = "relocated/source.pdf"
        for observation in relocated_payload["observations"]:
            observation["evidence"][0]["locator"]["relative_path"] = "relocated/source.pdf"
        relocated = FactBatch.model_validate(relocated_payload)

        created = await writer.write_batch(baseline)
        repeated = await writer.write_batch(relocated)

        assert created.status == "written"
        assert repeated.status == "already_present"
        assert writer.batch_count == 1

    asyncio.run(scenario())


def test_writer_merges_alias_enrichment_from_a_distinct_evidence_batch() -> None:
    async def scenario() -> None:
        writer = InMemoryGlobalKnowledgeGraphWriter()
        baseline = _batch()
        material, additive = _entities()
        enriched_material = EntityRef(
            entity_type=material.entity_type,
            canonical_name=material.canonical_name,
            aliases=(*material.aliases, "New laboratory alias"),
            identifiers=material.identifiers,
        )
        enriched_batch = FactBatch(
            evidence_fragment_id=OTHER_FRAGMENT_ID,
            extraction=_extractor(),
            entities=(enriched_material, additive),
            relations=(
                _relation(
                    enriched_material,
                    additive,
                    fragment_id=OTHER_FRAGMENT_ID,
                ),
            ),
            observations=(_observation(enriched_material, fragment_id=OTHER_FRAGMENT_ID),),
        )

        await writer.write_batch(baseline)
        await writer.write_batch(enriched_batch)
        snapshot = await writer.snapshot()
        material_node = next(node for node in snapshot.nodes if node.node_id == material.entity_id)

        assert material_node.properties["aliases"] == [
            "Material X17",
            "MX17",
            "New laboratory alias",
        ]
        assert writer.batch_count == 2

    asyncio.run(scenario())


def test_writer_conflict_is_atomic_fail_closed_and_does_not_echo_content() -> None:
    async def scenario() -> None:
        writer = InMemoryGlobalKnowledgeGraphWriter()
        baseline = _batch()
        conflicting = _batch(value=9.99)
        await writer.write_batch(baseline)
        before = await writer.snapshot()

        with pytest.raises(KnowledgeGraphConflict) as error:
            await writer.write_batch(conflicting)

        after = await writer.snapshot()
        assert str(error.value) == "knowledge_graph_conflict"
        assert "9.99" not in str(error.value)
        assert after == before
        assert writer.batch_count == 1

    asyncio.run(scenario())


def test_writer_protocol_is_runtime_checkable() -> None:
    writer = InMemoryGlobalKnowledgeGraphWriter()

    assert isinstance(writer, GlobalKnowledgeGraphWriter)


def test_untrusted_validation_rejects_raw_outputs_and_redacts_the_exception() -> None:
    payload = _batch().model_dump(mode="json")
    payload["complete_parser_output"] = "HIGHLY-SENSITIVE-RAW-CONTENT"

    with pytest.raises(FactContractError) as error:
        validate_fact_batch(payload)

    assert str(error.value) == "invalid_fact_batch"
    assert "HIGHLY-SENSITIVE-RAW-CONTENT" not in str(error.value)


def test_direct_schema_errors_also_hide_untrusted_input_values() -> None:
    raw_marker = "HIGHLY-SENSITIVE-PARSER-PAYLOAD"
    payload = _batch().model_dump(mode="json")
    payload["unexpected_raw_field"] = raw_marker

    with pytest.raises(ValidationError) as error:
        FactBatch.model_validate(payload)

    assert raw_marker not in str(error.value)


def test_untrusted_validation_returns_a_defensive_validated_model() -> None:
    payload = deepcopy(_batch().model_dump(mode="json"))

    validated = validate_fact_batch(payload)
    payload["entities"][0]["canonical_name"] = "mutated after validation"

    assert validated == _batch()


def test_fact_batch_json_schema_is_exportable_generic_and_strict() -> None:
    schema = export_fact_batch_json_schema()
    rendered = json.dumps(schema, ensure_ascii=False, sort_keys=True)

    assert schema["title"] == "FactBatch"
    assert {"EntityRef", "EvidenceLink", "PropertyObservation", "RelationAssertion"} <= set(
        schema["$defs"]
    )
    assert schema["additionalProperties"] is False
    assert "polyimide" not in rendered.casefold()
    assert "pet/pa6" not in rendered.casefold()
    assert "raw_pdf" not in rendered


def test_sensitive_or_raw_extension_fields_are_not_accepted() -> None:
    with pytest.raises(ValidationError):
        EntityRef(
            entity_type="material",
            canonical_name="MX-17",
            identifiers={"api_key": "not-a-real-key"},
        )

    evidence_payload = _evidence().model_dump(mode="python")
    evidence_payload["raw_pdf"] = b"not-a-pdf"
    with pytest.raises(ValidationError):
        EvidenceLink.model_validate(evidence_payload)


def test_textual_projection_fields_are_bounded_against_parser_output_smuggling() -> None:
    with pytest.raises(ValidationError):
        MaterialTestCondition(name="reported_context", value="x" * 501, unit="n/a")
    with pytest.raises(ValidationError):
        EntityRef(
            entity_type="material",
            canonical_name="MX-17",
            aliases=("x" * 501,),
        )
    with pytest.raises(ValidationError, match="evidence_location_too_large"):
        EvidenceLink(
            fragment_id=FRAGMENT_ID,
            source_id=SOURCE_ID,
            locator=SourceLocator(
                root_id="document_data_1",
                relative_path="private/corpus/material-paper.pdf",
                section="x" * 501,
            ),
        )


@pytest.mark.parametrize(
    "identifiers",
    [
        [],
        {"registry": 42},
        {"x" * 101: "value"},
        {"registry-id": "one", "registry_id": "two"},
    ],
)
def test_identifier_map_rejects_ambiguous_or_non_text_extensions(identifiers: object) -> None:
    with pytest.raises(ValidationError):
        EntityRef(
            entity_type="material",
            canonical_name="MX-17",
            identifiers=identifiers,
        )


def test_contracts_reject_control_text_and_duplicate_deterministic_ids() -> None:
    with pytest.raises(ValidationError):
        EntityRef(entity_type="material", canonical_name="invalid\x00name")

    material, _ = _entities()
    with pytest.raises(ValidationError, match="duplicate_contract_item"):
        FactBatch(
            evidence_fragment_id=FRAGMENT_ID,
            extraction=_extractor(),
            entities=(material, material),
            observations=(_observation(material),),
        )


def test_provided_evidence_and_batch_ids_must_match() -> None:
    evidence_payload = _evidence().model_dump(mode="python")
    evidence_payload["evidence_ref_id"] = "evidence:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="evidence_ref_id_mismatch"):
        EvidenceLink.model_validate(evidence_payload)

    batch_payload = _batch().model_dump(mode="python")
    batch_payload["idempotency_key"] = "fact-batch-idempotency:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="fact_batch_idempotency_key_mismatch"):
        FactBatch.model_validate(batch_payload)

    batch_payload = _batch().model_dump(mode="python")
    batch_payload["batch_id"] = "fact-batch:v1:" + "0" * 64
    with pytest.raises(ValidationError, match="fact_batch_id_mismatch"):
        FactBatch.model_validate(batch_payload)


def test_property_graph_projection_rejects_duplicate_ids_and_missing_endpoints() -> None:
    node = PropertyGraphNode(node_id="node-1", labels=("Entity",))
    with pytest.raises(ValidationError, match="duplicate_graph_identifier"):
        PropertyGraphProjection(nodes=(node, node), edges=())

    edge = PropertyGraphEdge(
        edge_id="edge:v1:" + "0" * 64,
        edge_type="RELATED_TO",
        source_node_id="node-1",
        target_node_id="missing-node",
    )
    with pytest.raises(ValidationError, match="graph_edge_endpoint_missing"):
        PropertyGraphProjection(nodes=(node,), edges=(edge,))


def test_model_version_pair_is_all_or_nothing() -> None:
    with pytest.raises(ValidationError, match="model_provenance_incomplete"):
        ExtractionProvenance(
            extractor_name="rule-extractor",
            extractor_version="1",
            generation_id="rules-v1",
            model_name="model-without-version",
        )

    rule_based = ExtractionProvenance(
        extractor_name="rule-extractor",
        extractor_version="1",
        generation_id="rules-v1",
        model_name=None,
        model_version=None,
    )
    assert rule_based.model_name is None
    assert rule_based.model_version is None


def test_boolean_condition_is_preserved_without_numeric_coercion() -> None:
    condition = MaterialTestCondition(name="annealed", value=True, unit="boolean")

    assert condition.value is True
