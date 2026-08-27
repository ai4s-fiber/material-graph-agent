from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from material_graph.knowledge.mineru_client import MinerUBlock, MinerUParseResult
from material_graph.knowledge.models import SelectionDecision, SourceLocator
from material_graph.knowledge.retention import (
    BlockEvidenceAssessment,
    EvidenceRetentionPolicy,
    EvidenceSelector,
    keyword_assessments,
)


def _decision(*, selected: bool = True) -> SelectionDecision:
    return SelectionDecision(
        source_id=uuid4(),
        selected=selected,
        reason_code="active_evidence_gap" if selected else "budget_deferred",
        evidence_gap_id=uuid4(),
        rank=1 if selected else None,
        policy_version="retention-policy-v1",
    )


def _parsed() -> MinerUParseResult:
    return MinerUParseResult(
        batch_id="batch",
        task_id="task",
        filename="paper.pdf",
        parser_version="3.4.4",
        model_version="vlm",
        blocks=[
            MinerUBlock(
                block_type="title",
                text="Results",
                page=1,
                block_index=0,
                section="Results",
            ),
            MinerUBlock(
                block_type="text",
                text="The measured glass-transition temperature was 315 °C under nitrogen.",
                page=2,
                block_index=1,
                section="Results > Thermal analysis",
            ),
            MinerUBlock(
                block_type="header",
                text="Journal running header",
                page=2,
                block_index=2,
            ),
            MinerUBlock(
                block_type="table",
                text="Table 2\nTg | 315 °C\nMeasured by DSC",
                page=3,
                block_index=3,
                section="Results > Thermal analysis",
            ),
            MinerUBlock(
                block_type="text",
                text="The measured glass-transition temperature was 315 °C under nitrogen.",
                page=4,
                block_index=4,
                section="Discussion",
            ),
        ],
    )


def test_retain_keeps_only_assessed_citable_blocks_with_exact_locators() -> None:
    decision = _decision()
    selector = EvidenceSelector(
        EvidenceRetentionPolicy(max_fragments_per_source=8, minimum_confidence=0.6)
    )
    assessments = [
        BlockEvidenceAssessment(
            block_index=1,
            accepted=True,
            confidence=0.91,
            retention_reason="supports:glass_transition_temperature",
            supported_entity_ids=["entity:Tg"],
            supported_relation_ids=["relation:sample_has_Tg"],
            evidence_gap_ids=["gap:thermal"],
        ),
        BlockEvidenceAssessment(
            block_index=2,
            accepted=True,
            confidence=0.99,
            retention_reason="header-noise",
        ),
        BlockEvidenceAssessment(
            block_index=3,
            accepted=True,
            confidence=0.88,
            retention_reason="supports:thermal_table",
            supported_relation_ids=["relation:measurement_condition"],
        ),
        BlockEvidenceAssessment(
            block_index=4,
            accepted=True,
            confidence=0.85,
            retention_reason="duplicate-statement",
        ),
    ]

    fragments = selector.retain(
        _parsed(),
        decision=decision,
        source_locator=SourceLocator(
            root_id="document_data_1",
            relative_path="papers/paper.pdf",
        ),
        assessments=assessments,
        embedding_generation_id="fixture-embedding-generation-v1",
    )

    assert len(fragments) == 2
    assert fragments[0].source_id == decision.source_id
    assert fragments[0].text.endswith("under nitrogen.")
    assert fragments[0].locator.page == 2
    assert fragments[0].locator.section == "Results > Thermal analysis"
    assert fragments[0].locator.block_index == 1
    assert fragments[0].supported_entity_ids == ["entity:Tg"]
    assert fragments[0].metadata["evidence_gap_ids"] == ["gap:thermal"]
    assert fragments[1].locator.page == 3
    assert fragments[1].locator.table == "block:3"
    assert fragments[1].parser_name == "mineru"
    assert fragments[1].parser_version == "3.4.4"
    assert all(fragment.locator.relative_path == "papers/paper.pdf" for fragment in fragments)


def test_retain_requires_selected_decision_and_matching_source() -> None:
    selector = EvidenceSelector()
    locator = SourceLocator(root_id="document_data_1", relative_path="paper.pdf")
    assessment = BlockEvidenceAssessment(
        block_index=1,
        accepted=True,
        confidence=1,
        retention_reason="supports:test",
    )

    with pytest.raises(ValueError, match="selected"):
        selector.retain(
            _parsed(),
            decision=_decision(selected=False),
            source_locator=locator,
            assessments=[assessment],
            embedding_generation_id="generation",
        )


def test_invalid_duplicate_or_unknown_assessment_is_rejected() -> None:
    selector = EvidenceSelector()
    decision = _decision()
    locator = SourceLocator(root_id="document_data_1", relative_path="paper.pdf")
    duplicate = BlockEvidenceAssessment(
        block_index=1,
        accepted=True,
        confidence=1,
        retention_reason="supports:test",
    )

    with pytest.raises(ValueError, match="duplicate assessment"):
        selector.retain(
            _parsed(),
            decision=decision,
            source_locator=locator,
            assessments=[duplicate, duplicate.model_copy()],
            embedding_generation_id="generation",
        )

    with pytest.raises(ValueError, match="unknown block"):
        selector.retain(
            _parsed(),
            decision=decision,
            source_locator=locator,
            assessments=[duplicate.model_copy(update={"block_index": 99})],
            embedding_generation_id="generation",
        )


def test_confidence_budget_and_stable_order_are_enforced() -> None:
    parsed = _parsed().model_copy(
        update={
            "blocks": [
                MinerUBlock(
                    block_type="text",
                    text=f"Evidence statement number {index} has enough useful material detail.",
                    page=1,
                    block_index=index,
                )
                for index in range(5)
            ]
        }
    )
    assessments = [
        BlockEvidenceAssessment(
            block_index=index,
            accepted=True,
            confidence=confidence,
            retention_reason=f"supports:{index}",
        )
        for index, confidence in enumerate([0.9, 0.5, 0.8, 0.95, 0.7])
    ]
    selector = EvidenceSelector(
        EvidenceRetentionPolicy(max_fragments_per_source=2, minimum_confidence=0.7)
    )

    fragments = selector.retain(
        parsed,
        decision=_decision(),
        source_locator=SourceLocator(root_id="data_3", relative_path="paper.pdf"),
        assessments=assessments,
        embedding_generation_id="generation",
    )

    assert [fragment.locator.block_index for fragment in fragments] == [3, 0]


def test_keyword_assessments_are_generic_deterministic_and_reject_noise() -> None:
    assessments = keyword_assessments(
        _parsed().blocks,
        terms=["glass-transition", "315 °c", "not-present"],
        evidence_gap_id="gap:thermal",
        minimum_matches=1,
    )

    accepted = [item for item in assessments if item.accepted]
    assert [item.block_index for item in accepted] == [1, 3, 4]
    assert accepted[0].evidence_gap_ids == ["gap:thermal"]
    assert accepted[0].retention_reason == "keyword_support:glass-transition,315 °c"
    assert all(item.block_index != 2 or not item.accepted for item in assessments)


def test_assessment_contract_rejects_empty_reason_for_accepted_block() -> None:
    with pytest.raises(ValidationError):
        BlockEvidenceAssessment(
            block_index=1,
            accepted=True,
            confidence=1,
            retention_reason="",
        )
