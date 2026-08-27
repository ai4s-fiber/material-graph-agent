from __future__ import annotations

from pathlib import Path

from material_graph.knowledge.textbook_chunking import (
    TextbookChunkingPolicy,
    chunk_textbook_document,
)
from material_graph.knowledge.textbook_corpus import discover_textbook_corpus


def _document(tmp_path: Path, text: str):
    path = tmp_path / "source_hu" / "第1批" / "高分子材料__part02" / "高分子材料__part02.md"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return discover_textbook_corpus(tmp_path).unique_documents[0]


def test_preserves_page_section_and_removes_parser_noise(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        """# 高分子材料__part02

> 来源: `高分子材料__part02_p0201-0400.pdf`
> 切片: 1..200 (共 200 页)
> 模型: MiniMax-M3
> DPI: 200

---
<!-- PAGE 7 -->

## 第 7 页

### 2.1 聚合反应

聚合反应正文，说明单体如何形成高分子。

<sub>tokens(in=3759, out=55) · 11.4s</sub>

![image](images/secret-local-name.jpg)

[BLANK PAGE]
""",
    )

    chunks = chunk_textbook_document(
        document,
        TextbookChunkingPolicy(
            target_chars=220,
            max_chars=320,
            overlap_chars=20,
            min_content_chars=1,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].page == 7
    assert chunks[0].section == "2.1 聚合反应"
    assert "[教材] 高分子材料" in chunks[0].text
    assert "[分片] 2" in chunks[0].text
    assert "聚合反应正文" in chunks[0].text
    assert "tokens(in=" not in chunks[0].text
    assert "secret-local-name" not in chunks[0].text
    assert "BLANK PAGE" not in chunks[0].text
    assert "MiniMax-M3" not in chunks[0].text


def test_long_content_is_bounded_and_deterministic(tmp_path: Path) -> None:
    sentence = "聚合物熔体在剪切场中表现出黏弹性，并受到温度和分子量分布影响。"
    document = _document(
        tmp_path,
        "# 高分子材料\n\n<!-- PAGE 8 -->\n\n### 流变性能\n\n" + sentence * 40,
    )
    policy = TextbookChunkingPolicy(
        target_chars=210,
        max_chars=260,
        overlap_chars=24,
        min_content_chars=1,
    )

    first = chunk_textbook_document(document, policy)
    second = chunk_textbook_document(document, policy)

    assert len(first) > 2
    assert first == second
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))
    assert all(len(chunk.text) <= policy.max_chars for chunk in first)
    assert all(chunk.page == 8 for chunk in first)
    assert all(chunk.section == "流变性能" for chunk in first)
    assert all(len(chunk.content_sha256) == 64 for chunk in first)


def test_short_adjacent_headings_are_retained_without_fragment_explosion(tmp_path: Path) -> None:
    document = _document(
        tmp_path,
        """<!-- PAGE 10 -->
### 第一节
第一节内容足够形成证据。

<!-- PAGE 11 -->
### 第二节
第二节内容也形成独立证据。
""",
    )

    chunks = chunk_textbook_document(
        document,
        TextbookChunkingPolicy(
            target_chars=500,
            max_chars=700,
            overlap_chars=0,
            min_content_chars=1,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].section == "第一节"
    assert chunks[0].page == 10
    assert chunks[0].page_end == 11
    assert "## 第一节" in chunks[0].text
    assert "## 第二节" in chunks[0].text
    assert "[页码] 11" in chunks[0].text
