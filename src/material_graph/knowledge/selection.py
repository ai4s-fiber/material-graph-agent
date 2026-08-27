"""Deterministic metadata-first source selection for one evidence gap.

The selector consumes catalog metadata only.  It has no source-reader
dependency, so ranking and budget decisions cannot open a PDF or other remote
body before a durable :class:`SelectionDecision` exists.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from .models import SelectionDecision, SourceCatalogRecord


RankOutput: TypeAlias = Sequence[UUID] | Mapping[UUID, float]


class MetadataRanker(Protocol):
    """Injectable semantic or reranking boundary over catalog metadata."""

    def __call__(
        self,
        query: str,
        sources: Sequence[SourceCatalogRecord],
    ) -> RankOutput: ...


class SelectionPolicy(BaseModel):
    """Versioned limits that make selection decisions replayable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="metadata-selection-v1", min_length=1)
    max_sources_per_gap: int = Field(default=20, ge=1, le=200)
    max_pages_per_run: int = Field(default=2_000, ge=1, le=2_000)
    unknown_page_count: int = Field(default=100, ge=1, le=2_000)
    reciprocal_rank_k: int = Field(default=60, ge=1, le=1_000)


class _PageMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_count: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        strict=True,
        validation_alias=AliasChoices("page_count", "pages", "pdf_page_count"),
    )


_ASCII_WORD = re.compile(r"[a-z0-9]+")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")
_PLACEHOLDER_TITLES = {
    "unknown",
    "unknown source",
    "untitled",
    "n/a",
    "none",
    "未知来源",
    "未命名",
}
_TEXT_METADATA_KEYS = (
    "abstract",
    "keywords",
    "keyword",
    "authors",
    "journal",
    "venue",
    "assignee",
    "patentee",
    "ipc",
    "category",
    "material_categories",
    "top_categories",
)
_PROCESSED_OR_ACTIVE_STATUSES = {
    "deduplicated",
    "selected_for_parse",
    "spooling",
    "parsing",
    "evidence_retained",
    "parsed_no_value",
    "indexed",
}


def _safe_text(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_safe_text(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value, key=str):
            result.extend(_safe_text(value[key]))
        return result
    return []


def _search_text(source: SourceCatalogRecord) -> str:
    values: list[str] = [
        source.display_title,
        source.source_kind,
        source.knowledge_domain,
    ]
    for value in (
        source.normalized_doi,
        source.application_number,
        source.publication_number,
        source.grant_number,
        source.material_category,
    ):
        if value:
            values.append(str(value))
    for key in _TEXT_METADATA_KEYS:
        if key in source.metadata:
            values.extend(_safe_text(source.metadata[key]))
    return " ".join(values)


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = _ASCII_WORD.findall(normalized)
    cjk = _CJK_CHAR.findall(normalized)
    tokens.extend(cjk)
    tokens.extend("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
    return tokens


def _lexical_score(query_tokens: Counter[str], source: SourceCatalogRecord) -> float:
    document_tokens = Counter(_tokens(_search_text(source)))
    return float(
        sum(query_count * document_tokens[token] for token, query_count in query_tokens.items())
    )


def _metadata_is_sufficient(source: SourceCatalogRecord) -> bool:
    title = unicodedata.normalize("NFKC", source.display_title).strip().casefold()
    if title and title not in _PLACEHOLDER_TITLES:
        return True
    if any(
        (
            source.normalized_doi,
            source.application_number,
            source.publication_number,
            source.grant_number,
            source.material_category,
        )
    ):
        return True
    return any(_safe_text(source.metadata.get(key)) for key in _TEXT_METADATA_KEYS)


def _is_process_data(source: SourceCatalogRecord) -> bool:
    return source.locator.root_id == "data_2" and "process_data" in {
        part.casefold() for part in source.locator.relative_path.split("/")
    }


def _source_order(source: SourceCatalogRecord) -> tuple[int, str]:
    return (
        0 if source.locator.root_id == "document_data_1" else 1,
        source.source_id.hex,
    )


def _candidate_preference(source: SourceCatalogRecord) -> tuple[int, int, str]:
    searchable_length = len(_tokens(_search_text(source)))
    root_priority, stable_id = _source_order(source)
    return (-searchable_length, root_priority, stable_id)


def _page_count(source: SourceCatalogRecord, policy: SelectionPolicy) -> int:
    try:
        metadata = _PageMetadata.model_validate(source.metadata)
    except ValidationError:
        return policy.unknown_page_count
    return metadata.page_count or policy.unknown_page_count


def _normalize_rank_output(
    output: RankOutput,
    *,
    allowed_ids: set[UUID],
) -> list[UUID]:
    if isinstance(output, Mapping):
        candidates = [
            source_id
            for source_id, _score in sorted(
                output.items(),
                key=lambda item: (-float(item[1]), item[0].hex),
            )
        ]
    else:
        candidates = list(output)

    result: list[UUID] = []
    seen: set[UUID] = set()
    for source_id in candidates:
        if source_id in allowed_ids and source_id not in seen:
            result.append(source_id)
            seen.add(source_id)
    return result


def _reciprocal_rank_fusion(
    lexical_ids: Sequence[UUID],
    semantic_ids: Sequence[UUID],
    *,
    k: int,
) -> list[UUID]:
    scores: defaultdict[UUID, float] = defaultdict(float)
    for ranking in (lexical_ids, semantic_ids):
        for rank, source_id in enumerate(ranking, start=1):
            scores[source_id] += 1.0 / (k + rank)
    lexical_position = {source_id: rank for rank, source_id in enumerate(lexical_ids, start=1)}
    return sorted(
        lexical_ids,
        key=lambda source_id: (
            -scores[source_id],
            lexical_position[source_id],
            source_id.hex,
        ),
    )


def _decision(
    source: SourceCatalogRecord,
    *,
    selected: bool,
    reason_code: str,
    policy: SelectionPolicy,
    task_id: UUID | None,
    evidence_gap_id: UUID | None,
    rank: int | None = None,
) -> SelectionDecision:
    return SelectionDecision(
        source_id=source.source_id,
        selected=selected,
        reason_code=reason_code,
        task_id=task_id,
        evidence_gap_id=evidence_gap_id,
        rank=rank,
        policy_version=policy.policy_version,
    )


class SourceSelector:
    """Rank and budget canonical catalog records without opening source bodies."""

    def __init__(
        self,
        *,
        policy: SelectionPolicy | None = None,
        semantic_ranker: MetadataRanker | None = None,
        reranker: MetadataRanker | None = None,
    ) -> None:
        self.policy = policy or SelectionPolicy()
        self.semantic_ranker = semantic_ranker
        self.reranker = reranker

    def select(
        self,
        sources: Sequence[SourceCatalogRecord],
        evidence_gap: str,
        *,
        task_id: UUID | None = None,
        evidence_gap_id: UUID | None = None,
        page_budget: int | None = None,
    ) -> list[SelectionDecision]:
        query = unicodedata.normalize("NFKC", evidence_gap).strip()
        query_tokens = Counter(_tokens(query))
        if not query or not query_tokens:
            raise ValueError("evidence gap must contain searchable text")
        if page_budget is not None and page_budget < 0:
            raise ValueError("page_budget must be non-negative")
        available_pages = min(
            self.policy.max_pages_per_run,
            self.policy.max_pages_per_run if page_budget is None else page_budget,
        )

        records = self._unique_records(sources)
        rejected: dict[UUID, str] = {}
        preliminary: list[SourceCatalogRecord] = []
        completed_shas: set[str] = set()

        for source in records:
            reason = self._hard_rejection(source)
            if reason is None:
                preliminary.append(source)
                continue
            rejected[source.source_id] = reason
            if source.sha256 and source.status in _PROCESSED_OR_ACTIVE_STATUSES:
                completed_shas.add(source.sha256)

        canonical_candidates: list[SourceCatalogRecord] = []
        sha_groups: defaultdict[str, list[SourceCatalogRecord]] = defaultdict(list)
        for source in preliminary:
            if source.sha256 and source.sha256 in completed_shas:
                rejected[source.source_id] = "duplicate"
            elif source.sha256:
                sha_groups[source.sha256].append(source)
            else:
                canonical_candidates.append(source)

        for group in sha_groups.values():
            ordered = sorted(group, key=_candidate_preference)
            canonical_candidates.append(ordered[0])
            for duplicate in ordered[1:]:
                rejected[duplicate.source_id] = "duplicate"

        eligible: list[SourceCatalogRecord] = []
        for source in canonical_candidates:
            if _metadata_is_sufficient(source):
                eligible.append(source)
            else:
                rejected[source.source_id] = "insufficient_metadata"

        ranked = self._rank(query, query_tokens, eligible)
        ranked_decisions: list[SelectionDecision] = []
        selected_count = 0
        selected_pages = 0
        for rank, source in enumerate(ranked, start=1):
            pages = _page_count(source, self.policy)
            within_count = selected_count < self.policy.max_sources_per_gap
            within_pages = selected_pages + pages <= available_pages
            selected = within_count and within_pages
            if selected:
                selected_count += 1
                selected_pages += pages
            ranked_decisions.append(
                _decision(
                    source,
                    selected=selected,
                    reason_code=("active_evidence_gap" if selected else "budget_deferred"),
                    policy=self.policy,
                    task_id=task_id,
                    evidence_gap_id=evidence_gap_id,
                    rank=rank,
                )
            )

        by_id = {source.source_id: source for source in records}
        rejected_decisions = [
            _decision(
                by_id[source_id],
                selected=False,
                reason_code=reason,
                policy=self.policy,
                task_id=task_id,
                evidence_gap_id=evidence_gap_id,
            )
            for source_id, reason in sorted(rejected.items(), key=lambda item: item[0].hex)
        ]
        return [*ranked_decisions, *rejected_decisions]

    @staticmethod
    def _unique_records(
        sources: Sequence[SourceCatalogRecord],
    ) -> list[SourceCatalogRecord]:
        grouped: defaultdict[UUID, list[SourceCatalogRecord]] = defaultdict(list)
        for source in sources:
            grouped[source.source_id].append(source)
        return [
            sorted(group, key=_candidate_preference)[0]
            for _source_id, group in sorted(grouped.items(), key=lambda item: item[0].hex)
        ]

    @staticmethod
    def _hard_rejection(source: SourceCatalogRecord) -> str | None:
        if _is_process_data(source) or source.status == "excluded_process_data":
            return "process_data_excluded"
        if source.canonical_source_id is not None:
            return "duplicate"
        if source.status in _PROCESSED_OR_ACTIVE_STATUSES:
            return "duplicate"
        if source.status == "failed_permanent":
            return "insufficient_metadata"
        return None

    def _rank(
        self,
        query: str,
        query_tokens: Counter[str],
        sources: Sequence[SourceCatalogRecord],
    ) -> list[SourceCatalogRecord]:
        lexical = sorted(
            sources,
            key=lambda source: (
                -_lexical_score(query_tokens, source),
                *_source_order(source),
            ),
        )
        if not lexical:
            return []

        by_id = {source.source_id: source for source in lexical}
        allowed_ids = set(by_id)
        lexical_ids = [source.source_id for source in lexical]
        semantic_ids: list[UUID] = []
        if self.semantic_ranker is not None:
            semantic_ids = _normalize_rank_output(
                self.semantic_ranker(query, lexical),
                allowed_ids=allowed_ids,
            )
        fused_ids = _reciprocal_rank_fusion(
            lexical_ids,
            semantic_ids,
            k=self.policy.reciprocal_rank_k,
        )

        final_ids = fused_ids
        if self.reranker is not None:
            fused_sources = [by_id[source_id] for source_id in fused_ids]
            reranked_ids = _normalize_rank_output(
                self.reranker(query, fused_sources),
                allowed_ids=allowed_ids,
            )
            reranked_set = set(reranked_ids)
            final_ids = [
                *reranked_ids,
                *(source_id for source_id in fused_ids if source_id not in reranked_set),
            ]
        return [by_id[source_id] for source_id in final_ids]


__all__ = ["MetadataRanker", "SelectionPolicy", "SourceSelector"]
