from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import pytest

from material_graph.knowledge.models import SourceCatalogRecord, SourceLocator
from material_graph.knowledge.selection import SelectionPolicy, SourceSelector


def _source(
    number: int,
    *,
    title: str,
    root_id: str = "document_data_1",
    path: str | None = None,
    status: str = "metadata_indexed",
    digest: str | None = None,
    canonical_source_id: UUID | None = None,
    pages: object = 10,
    metadata: dict[str, object] | None = None,
    source_kind: str = "literature",
    doi: str | None = None,
) -> SourceCatalogRecord:
    merged_metadata = dict(metadata or {})
    if pages is not None:
        merged_metadata["page_count"] = pages
    return SourceCatalogRecord(
        source_id=UUID(int=number),
        locator=SourceLocator(
            root_id=root_id,
            relative_path=path or f"papers/{number}.pdf",
        ),
        source_kind=source_kind,
        display_title=title,
        status=status,
        normalized_doi=doi,
        sha256=digest,
        canonical_source_id=canonical_source_id,
        metadata=merged_metadata,
    )


def _by_id(decisions):
    return {decision.source_id: decision for decision in decisions}


class _BodyReadSpy:
    def __init__(self) -> None:
        self.call_count = 0

    def read(self) -> bytes:
        self.call_count += 1
        raise AssertionError("selection must never read a source body")


class _RankSpy:
    def __init__(self, order: Sequence[UUID]) -> None:
        self.order = list(order)
        self.calls: list[tuple[str, list[UUID]]] = []

    def __call__(
        self,
        query: str,
        sources: Sequence[SourceCatalogRecord],
    ) -> Sequence[UUID]:
        self.calls.append((query, [source.source_id for source in sources]))
        return self.order


def test_only_canonical_unprocessed_nonexcluded_sources_are_selected() -> None:
    body_spy = _BodyReadSpy()
    canonical = _source(
        1,
        title="Polyimide dielectric loss evidence",
        digest="a" * 64,
        metadata={"body_reader": body_spy.read, "abstract": "dielectric frequency"},
    )
    sources = [
        canonical,
        _source(
            2,
            title="Noncanonical copy",
            digest="a" * 64,
            canonical_source_id=canonical.source_id,
        ),
        _source(
            3,
            title="Forbidden process artifact",
            root_id="data_2",
            path="process_data/mineru/intermediate.pdf",
        ),
        _source(4, title="Already retained", status="evidence_retained"),
        _source(5, title="Parsed with no value", status="parsed_no_value"),
        _source(6, title="Unmarked SHA copy", digest="a" * 64),
        _source(7, title="Unknown source", pages=None, source_kind="unknown"),
    ]

    task_id = UUID(int=90)
    gap_id = UUID(int=91)
    decisions = SourceSelector().select(
        sources,
        "polyimide dielectric loss frequency",
        task_id=task_id,
        evidence_gap_id=gap_id,
    )
    mapped = _by_id(decisions)

    assert [item.source_id for item in decisions if item.selected] == [canonical.source_id]
    assert mapped[UUID(int=2)].reason_code == "duplicate"
    assert mapped[UUID(int=3)].reason_code == "process_data_excluded"
    assert mapped[UUID(int=4)].reason_code == "duplicate"
    assert mapped[UUID(int=5)].reason_code == "duplicate"
    assert mapped[UUID(int=6)].reason_code == "duplicate"
    assert mapped[UUID(int=7)].reason_code == "insufficient_metadata"
    assert mapped[canonical.source_id].rank == 1
    assert mapped[canonical.source_id].task_id == task_id
    assert mapped[canonical.source_id].evidence_gap_id == gap_id
    assert mapped[canonical.source_id].policy_version == "metadata-selection-v1"
    assert body_spy.call_count == 0


def test_explicit_excluded_status_never_reaches_rankers() -> None:
    excluded = _source(
        10,
        title="Excluded corpus object",
        status="excluded_process_data",
    )
    semantic = _RankSpy([excluded.source_id])
    reranker = _RankSpy([excluded.source_id])

    decision = SourceSelector(
        semantic_ranker=semantic,
        reranker=reranker,
    ).select([excluded], "materials evidence")[0]

    assert decision.selected is False
    assert decision.reason_code == "process_data_excluded"
    assert semantic.calls == []
    assert reranker.calls == []


def test_lexical_semantic_rrf_then_reranker_define_stable_rank() -> None:
    lexical_first = _source(21, title="dielectric loss dielectric evidence")
    semantic_first = _source(22, title="dielectric processing evidence")
    reranked_first = _source(23, title="thermal stability evidence")
    semantic = _RankSpy(
        [semantic_first.source_id, reranked_first.source_id, lexical_first.source_id]
    )
    reranker = _RankSpy(
        [reranked_first.source_id, semantic_first.source_id, lexical_first.source_id]
    )
    selector = SourceSelector(semantic_ranker=semantic, reranker=reranker)

    first = selector.select(
        [reranked_first, lexical_first, semantic_first],
        "dielectric loss",
    )
    second = selector.select(
        [semantic_first, lexical_first, reranked_first],
        "dielectric loss",
    )

    expected = [
        reranked_first.source_id,
        semantic_first.source_id,
        lexical_first.source_id,
    ]
    assert [decision.source_id for decision in first if decision.selected] == expected
    assert [decision.rank for decision in first if decision.selected] == [1, 2, 3]
    assert [decision.model_dump() for decision in first] == [
        decision.model_dump() for decision in second
    ]
    assert semantic.calls[0][1] == [
        lexical_first.source_id,
        semantic_first.source_id,
        reranked_first.source_id,
    ]
    assert reranker.calls[0][1] == [
        semantic_first.source_id,
        lexical_first.source_id,
        reranked_first.source_id,
    ]


def test_page_and_source_budgets_defer_lower_ranked_sources() -> None:
    policy = SelectionPolicy(
        max_sources_per_gap=2,
        max_pages_per_run=2_000,
        unknown_page_count=100,
    )
    sources = [
        _source(31, title="dielectric primary evidence", pages=1_500),
        _source(32, title="dielectric secondary evidence", pages=None),
        _source(33, title="dielectric tertiary evidence", pages=600),
        _source(34, title="dielectric fourth evidence", pages=50),
    ]

    decisions = SourceSelector(policy=policy).select(sources, "dielectric")
    mapped = _by_id(decisions)

    assert mapped[UUID(int=31)].selected is True
    assert mapped[UUID(int=32)].selected is True
    assert mapped[UUID(int=33)].reason_code == "budget_deferred"
    assert mapped[UUID(int=34)].reason_code == "budget_deferred"
    assert mapped[UUID(int=33)].rank == 3
    assert mapped[UUID(int=34)].rank == 4


def test_invalid_or_unknown_page_count_uses_conservative_default() -> None:
    policy = SelectionPolicy(max_pages_per_run=100, unknown_page_count=100)
    unknown = _source(41, title="dielectric unknown pages", pages="not-a-number")
    decisions = SourceSelector(policy=policy).select([unknown], "dielectric")

    assert decisions[0].selected is True

    no_budget = SourceSelector(policy=policy).select(
        [unknown],
        "dielectric",
        page_budget=99,
    )
    assert no_budget[0].selected is False
    assert no_budget[0].reason_code == "budget_deferred"


def test_default_policy_caps_each_gap_at_twenty_sources() -> None:
    sources = [
        _source(number, title=f"dielectric evidence {number}", pages=1)
        for number in range(100, 121)
    ]

    decisions = SourceSelector().select(sources, "dielectric")

    assert sum(decision.selected for decision in decisions) == 20
    assert sum(decision.reason_code == "budget_deferred" for decision in decisions) == 1


def test_nested_metadata_and_identifier_can_make_placeholder_record_searchable() -> None:
    nested = _source(
        130,
        title="Unknown source",
        pages=2,
        metadata={
            "keywords": [{"term": "dielectric", "weight": 1.5}],
        },
    )
    identified = _source(
        131,
        title="Unknown source",
        pages=2,
        doi="10.1000/material",
    )

    decisions = SourceSelector().select([nested, identified], "dielectric material")

    assert all(decision.selected for decision in decisions)


def test_mapping_rank_output_ignores_unknown_and_duplicate_ids() -> None:
    first = _source(140, title="dielectric first")
    second = _source(141, title="dielectric second")
    unknown_id = UUID(int=9_999)

    def semantic(_query, _sources):
        return {second.source_id: 0.9, unknown_id: 1.0, first.source_id: 0.1}

    decisions = SourceSelector(semantic_ranker=semantic).select(
        [first, second],
        "dielectric",
    )

    assert {decision.source_id for decision in decisions} == {
        first.source_id,
        second.source_id,
    }


def test_completed_sha_blocks_an_unmarked_copy_and_failed_source_is_insufficient() -> None:
    completed = _source(
        150,
        title="retained evidence",
        status="evidence_retained",
        digest="b" * 64,
    )
    copy = _source(151, title="unmarked copy", digest="b" * 64)
    failed = _source(152, title="failed source", status="failed_permanent")

    mapped = _by_id(SourceSelector().select([copy, failed, completed], "evidence"))

    assert mapped[copy.source_id].reason_code == "duplicate"
    assert mapped[completed.source_id].reason_code == "duplicate"
    assert mapped[failed.source_id].reason_code == "insufficient_metadata"


@pytest.mark.parametrize(
    ("query", "page_budget", "message"),
    [
        ("   ", None, "searchable text"),
        ("dielectric", -1, "non-negative"),
    ],
)
def test_selection_rejects_invalid_gap_or_budget(
    query: str,
    page_budget: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SourceSelector().select([], query, page_budget=page_budget)


def test_policy_cannot_raise_the_hard_run_page_cap() -> None:
    with pytest.raises(ValueError):
        SelectionPolicy(max_pages_per_run=2_001)
