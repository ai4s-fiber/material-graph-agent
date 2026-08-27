from __future__ import annotations

import json
from pathlib import Path

from material_graph.knowledge.textbook_chunking import TextbookChunkingPolicy
from material_graph.knowledge.textbook_import import (
    iter_fragment_jsonl,
    prepare_textbook_corpus,
)


def _write_part(root: Path, batch: str, title: str, part: int, body: str) -> None:
    stem = f"{title}__part{part:02d}"
    path = root / "source_hu" / batch / stem / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {stem}\n\n<!-- PAGE 3 -->\n\n### 第一章\n\n{body}",
        encoding="utf-8",
    )


def test_preparation_deduplicates_documents_and_is_deterministic(tmp_path: Path) -> None:
    body = "高分子材料的结构决定其宏观性能。" * 30
    _write_part(tmp_path, "第1批", "教材甲", 1, body)
    _write_part(tmp_path, "第2批", "教材甲", 1, body)
    policy = TextbookChunkingPolicy(
        target_chars=260,
        max_chars=340,
        overlap_chars=20,
        min_content_chars=1,
    )

    first = prepare_textbook_corpus(
        tmp_path,
        embedding_generation_id="embedding-generation-test",
        chunking_policy=policy,
    )
    second = prepare_textbook_corpus(
        tmp_path,
        embedding_generation_id="embedding-generation-test",
        chunking_policy=policy,
    )

    assert first.discovered_document_count == 2
    assert first.unique_document_count == 1
    assert first.duplicate_document_count == 1
    assert first.fragment_count > 1
    assert first.corpus_digest == second.corpus_digest
    assert [fragment.fragment_id for fragment in first.fragments] == [
        fragment.fragment_id for fragment in second.fragments
    ]


def test_fragments_use_existing_contract_and_safe_scalar_metadata(tmp_path: Path) -> None:
    _write_part(
        tmp_path,
        "第1批",
        "教材乙",
        2,
        "纺丝温度和牵伸倍数共同影响纤维取向与结晶。" * 8,
    )

    prepared = prepare_textbook_corpus(
        tmp_path,
        embedding_generation_id="embedding-generation-test",
        chunking_policy=TextbookChunkingPolicy(
            target_chars=300,
            max_chars=400,
            overlap_chars=20,
            min_content_chars=1,
        ),
    )

    fragment = prepared.fragments[0]
    assert fragment.embedding_generation_id == "embedding-generation-test"
    assert fragment.locator.root_id == "cyj_source_hu"
    assert fragment.locator.relative_path.startswith("source_hu/")
    assert fragment.locator.page == 3
    assert fragment.locator.section == "第一章"
    assert fragment.parser_name == "source_hu_markdown"
    assert fragment.parser_version == "local-textbook-v1"
    assert fragment.metadata["logical_title"] == "教材乙"
    assert fragment.metadata["part_number"] == 2
    assert fragment.metadata["page_end"] == 3
    assert all(
        isinstance(value, str | int | float | bool) or value is None
        for value in fragment.metadata.values()
    )
    assert not {
        "mineru_json",
        "mineru_markdown",
        "raw_document",
        "full_document_text",
    }.intersection(fragment.metadata)


def test_jsonl_serialization_is_stable_and_round_trips(tmp_path: Path) -> None:
    _write_part(tmp_path, "第1批", "教材丙", 1, "聚合反应动力学用于描述转化率变化。" * 10)
    prepared = prepare_textbook_corpus(
        tmp_path,
        embedding_generation_id="embedding-generation-test",
        chunking_policy=TextbookChunkingPolicy(
            target_chars=300,
            max_chars=400,
            overlap_chars=20,
            min_content_chars=1,
        ),
    )

    first = list(iter_fragment_jsonl(prepared))
    second = list(iter_fragment_jsonl(prepared))

    assert first == second
    assert len(first) == prepared.fragment_count
    payload = json.loads(first[0])
    assert payload["fragment_id"] == str(prepared.fragments[0].fragment_id)
    assert payload["locator"]["relative_path"].startswith("source_hu/")
    assert str(tmp_path) not in first[0]
