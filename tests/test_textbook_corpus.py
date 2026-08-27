from __future__ import annotations

from pathlib import Path

import pytest

from material_graph.knowledge.textbook_corpus import (
    TextbookCorpusError,
    discover_textbook_corpus,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_discovers_mineru_and_source_hu_without_process_artifacts(tmp_path: Path) -> None:
    _write(
        tmp_path / "教材甲" / "教材甲" / "ocr" / "教材甲.md",
        "# 教材甲\n\n## 第一章\n\n正文甲。",
    )
    _write(
        tmp_path / "_mineru_workdirs" / "job" / "ocr" / "不应出现.md",
        "# workdir",
    )
    _write(tmp_path / "文件结构说明.md", "# 说明")
    (tmp_path / "仅有日志").mkdir()
    _write(
        tmp_path / "source_hu" / "第1批" / "教材乙__part02" / "教材乙__part02.md",
        "# 教材乙__part02\n\n<!-- PAGE 1 -->\n\n正文乙。",
    )

    inventory = discover_textbook_corpus(tmp_path)

    assert inventory.discovered_document_count == 2
    assert inventory.unique_document_count == 2
    by_title = {document.logical_title: document for document in inventory.documents}
    assert set(by_title) == {"教材甲", "教材乙"}
    assert by_title["教材甲"].source_family == "mineru_markdown"
    assert by_title["教材甲"].part_number is None
    assert by_title["教材乙"].source_family == "source_hu_markdown"
    assert by_title["教材乙"].part_number == 2
    assert all("\\" not in document.relative_path for document in inventory.documents)


def test_marks_normalized_exact_content_duplicates_deterministically(tmp_path: Path) -> None:
    content = "# 同一本书\n\n相同正文。\n"
    first = tmp_path / "source_hu" / "第1批" / "同一本书__part01" / "同一本书__part01.md"
    second = tmp_path / "source_hu" / "第2批" / "同一本书__part01" / "同一本书__part01.md"
    _write(first, content)
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    first_inventory = discover_textbook_corpus(tmp_path)
    second_inventory = discover_textbook_corpus(tmp_path)

    assert first_inventory.discovered_document_count == 2
    assert first_inventory.unique_document_count == 1
    assert first_inventory.duplicate_document_count == 1
    assert [item.source_id for item in first_inventory.documents] == [
        item.source_id for item in second_inventory.documents
    ]
    duplicate = next(item for item in first_inventory.documents if item.duplicate_of is not None)
    canonical = next(item for item in first_inventory.documents if item.duplicate_of is None)
    assert duplicate.duplicate_of == canonical.source_id
    assert duplicate.content_sha256 == canonical.content_sha256


def test_title_collision_with_different_content_is_not_deduplicated(tmp_path: Path) -> None:
    _write(
        tmp_path / "source_hu" / "批次甲" / "高分子化学__part01" / "高分子化学__part01.md",
        "# 高分子化学\n\n第一版内容",
    )
    _write(
        tmp_path / "source_hu" / "批次乙" / "高分子化学__part01" / "高分子化学__part01.md",
        "# 高分子化学\n\n第二版内容",
    )

    inventory = discover_textbook_corpus(tmp_path)

    assert inventory.discovered_document_count == 2
    assert inventory.unique_document_count == 2
    assert inventory.duplicate_document_count == 0


@pytest.mark.parametrize("missing_kind", ["missing", "file"])
def test_rejects_non_directory_root(tmp_path: Path, missing_kind: str) -> None:
    root = tmp_path / missing_kind
    if missing_kind == "file":
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(TextbookCorpusError):
        discover_textbook_corpus(root)
