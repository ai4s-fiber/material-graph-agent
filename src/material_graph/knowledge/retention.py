"""Derived-only evidence retention from transient parser blocks."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .mineru_client import MinerUBlock, MinerUParseResult
from .models import EvidenceFragment, SelectionDecision, SourceLocator


_DEFAULT_AUXILIARY_TYPES = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "aside_text",
    }
)


class BlockEvidenceAssessment(BaseModel):
    """Auditable validator output for one transient parser block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_index: int = Field(ge=0)
    accepted: bool
    confidence: float = Field(ge=0, le=1)
    retention_reason: str = ""
    supported_entity_ids: list[str] = Field(default_factory=list)
    supported_relation_ids: list[str] = Field(default_factory=list)
    evidence_gap_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def accepted_assessment_requires_reason(self) -> "BlockEvidenceAssessment":
        if self.accepted and not self.retention_reason.strip():
            raise ValueError("accepted block requires a retention reason")
        return self


class EvidenceRetentionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_fragments_per_source: int = Field(default=8, ge=1, le=64)
    minimum_confidence: float = Field(default=0.6, ge=0, le=1)
    minimum_text_characters: int = Field(default=16, ge=1)
    excluded_block_types: frozenset[str] = _DEFAULT_AUXILIARY_TYPES


class EvidenceSelector:
    """Convert only explicitly supported blocks into durable fragments."""

    def __init__(self, policy: EvidenceRetentionPolicy | None = None) -> None:
        self.policy = policy or EvidenceRetentionPolicy()

    def retain(
        self,
        parsed: MinerUParseResult,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        assessments: list[BlockEvidenceAssessment],
        embedding_generation_id: str,
    ) -> list[EvidenceFragment]:
        if not decision.selected:
            raise ValueError("evidence retention requires a selected source decision")

        block_by_index = {block.block_index: block for block in parsed.blocks}
        assessment_by_index: dict[int, BlockEvidenceAssessment] = {}
        for assessment in assessments:
            if assessment.block_index in assessment_by_index:
                raise ValueError(f"duplicate assessment for block {assessment.block_index}")
            if assessment.block_index not in block_by_index:
                raise ValueError(f"assessment references unknown block {assessment.block_index}")
            assessment_by_index[assessment.block_index] = assessment

        ranked = sorted(
            assessment_by_index.values(),
            key=lambda item: (-item.confidence, item.block_index),
        )
        retained: list[EvidenceFragment] = []
        retained_hashes: set[str] = set()
        for assessment in ranked:
            if not assessment.accepted or assessment.confidence < self.policy.minimum_confidence:
                continue
            block = block_by_index[assessment.block_index]
            if block.block_type in self.policy.excluded_block_types:
                continue
            text = block.text.strip()
            if len(text) < self.policy.minimum_text_characters:
                continue
            content_hash = sha256(text.encode("utf-8")).hexdigest()
            if content_hash in retained_hashes:
                continue

            retained.append(
                self._fragment(
                    parsed=parsed,
                    block=block,
                    decision=decision,
                    source_locator=source_locator,
                    assessment=assessment,
                    embedding_generation_id=embedding_generation_id,
                )
            )
            retained_hashes.add(content_hash)
            if len(retained) >= self.policy.max_fragments_per_source:
                break
        return retained

    @staticmethod
    def _fragment(
        *,
        parsed: MinerUParseResult,
        block: MinerUBlock,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        assessment: BlockEvidenceAssessment,
        embedding_generation_id: str,
    ) -> EvidenceFragment:
        table = f"block:{block.block_index}" if block.block_type == "table" else None
        figure = f"block:{block.block_index}" if block.block_type in {"image", "chart"} else None
        locator = source_locator.model_copy(
            update={
                "page": block.page,
                "section": block.section,
                "table": table,
                "figure": figure,
                "block_index": block.block_index,
            }
        )
        metadata = dict(assessment.metadata)
        metadata.update(
            {
                "assessment_confidence": assessment.confidence,
                "evidence_gap_ids": list(assessment.evidence_gap_ids),
                "block_type": block.block_type,
                "selection_reason_code": decision.reason_code,
                "selection_policy_version": decision.policy_version,
                "mineru_batch_id": parsed.batch_id,
                "mineru_task_id": parsed.task_id,
                "mineru_model_version": parsed.model_version,
            }
        )
        return EvidenceFragment(
            source_id=decision.source_id,
            text=block.text.strip(),
            locator=locator,
            retention_reason=assessment.retention_reason,
            supported_entity_ids=list(assessment.supported_entity_ids),
            supported_relation_ids=list(assessment.supported_relation_ids),
            parser_name=parsed.parser_name,
            parser_version=parsed.parser_version,
            embedding_generation_id=embedding_generation_id,
            metadata=metadata,
        )


def keyword_assessments(
    blocks: list[MinerUBlock],
    *,
    terms: list[str],
    evidence_gap_id: str,
    minimum_matches: int = 1,
) -> list[BlockEvidenceAssessment]:
    """Deterministic generic fallback used before a semantic validator is available."""

    normalized_terms = list(
        dict.fromkeys(term.strip().casefold() for term in terms if term.strip())
    )
    if not normalized_terms:
        raise ValueError("at least one evidence term is required")
    if minimum_matches < 1:
        raise ValueError("minimum_matches must be positive")

    assessments: list[BlockEvidenceAssessment] = []
    for block in blocks:
        if block.block_type in _DEFAULT_AUXILIARY_TYPES:
            assessments.append(
                BlockEvidenceAssessment(
                    block_index=block.block_index,
                    accepted=False,
                    confidence=0,
                    retention_reason="auxiliary_block",
                    evidence_gap_ids=[evidence_gap_id],
                )
            )
            continue
        normalized_text = block.text.casefold()
        matched = [term for term in normalized_terms if term in normalized_text]
        accepted = len(matched) >= minimum_matches
        assessments.append(
            BlockEvidenceAssessment(
                block_index=block.block_index,
                accepted=accepted,
                confidence=len(matched) / len(normalized_terms),
                retention_reason=(
                    f"keyword_support:{','.join(matched)}" if accepted else "keyword_no_support"
                ),
                evidence_gap_ids=[evidence_gap_id],
                metadata={"matched_terms": matched},
            )
        )
    return assessments
