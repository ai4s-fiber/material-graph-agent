"""Deterministic Markdown chunking for local textbook evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re

from .textbook_corpus import TextbookSourceDocument, normalize_textbook_title


_PAGE_COMMENT = re.compile(r"^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$", re.IGNORECASE)
_PAGE_HEADING = re.compile(r"^第\s*(\d+)\s*页$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")
_SOURCE_HU_METADATA = re.compile(r"^\s*>\s*(?:来源|切片|模型|DPI)\s*[:：]", re.IGNORECASE)
_PARSER_TELEMETRY = re.compile(r"^\s*<sub>\s*tokens\s*\(", re.IGNORECASE)
_IMAGE_ONLY = re.compile(r"^\s*!\[[^\]]*]\([^)]*\)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?；;])")


@dataclass(frozen=True, slots=True)
class TextbookChunkingPolicy:
    """Character budgets chosen to leave room for LightRAG request metadata."""

    target_chars: int = 2_400
    max_chars: int = 4_000
    overlap_chars: int = 200
    min_content_chars: int = 20

    def __post_init__(self) -> None:
        if self.target_chars < 80:
            raise ValueError("target_chars must be at least 80")
        if self.max_chars < self.target_chars:
            raise ValueError("max_chars must be at least target_chars")
        if self.overlap_chars < 0 or self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than target_chars")
        if self.min_content_chars < 1:
            raise ValueError("min_content_chars must be positive")


@dataclass(frozen=True, slots=True)
class TextbookChunk:
    """One bounded chunk with enough provenance to become evidence."""

    chunk_index: int
    text: str
    page: int | None
    page_end: int | None
    section: str | None
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _TextBlock:
    text: str
    page: int | None
    section: str | None


def _parse_blocks(document: TextbookSourceDocument) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []
    paragraph: list[str] = []
    page: int | None = None
    section: str | None = None
    saw_substantive_text = False

    def flush() -> None:
        nonlocal paragraph, saw_substantive_text
        text = "\n".join(line for line in paragraph if line).strip()
        paragraph = []
        if text:
            blocks.append(_TextBlock(text=text, page=page, section=section))
            saw_substantive_text = True

    for raw_line in document.text.splitlines():
        line = raw_line.strip()
        page_match = _PAGE_COMMENT.fullmatch(line)
        if page_match:
            flush()
            page = int(page_match.group(1))
            if blocks:
                blocks.append(_TextBlock(text=f"[页码] {page}", page=page, section=section))
            continue
        heading_match = _HEADING.fullmatch(line)
        if heading_match:
            flush()
            heading = heading_match.group(1).strip()
            page_heading = _PAGE_HEADING.fullmatch(heading)
            if page_heading:
                heading_page = int(page_heading.group(1))
                if heading_page != page:
                    page = heading_page
                    blocks.append(_TextBlock(text=f"[页码] {page}", page=page, section=section))
                continue
            heading_is_wrapper_title = not saw_substantive_text and (
                normalize_textbook_title(heading) == document.normalized_title
                or normalize_textbook_title(heading)
                == normalize_textbook_title(
                    document.logical_title
                    + (
                        f"__part{document.part_number:02d}"
                        if document.part_number is not None
                        else ""
                    )
                )
            )
            if not heading_is_wrapper_title:
                section = heading
                blocks.append(_TextBlock(text=f"## {heading}", page=page, section=section))
            continue
        if (
            not line
            or line == "---"
            or _SOURCE_HU_METADATA.match(line)
            or _PARSER_TELEMETRY.match(line)
            or _IMAGE_ONLY.fullmatch(line)
            or line.casefold() == "[blank page]"
        ):
            flush()
            continue
        paragraph.append(line)
    flush()
    return blocks


def _bounded_label(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _prefix(
    document: TextbookSourceDocument,
    *,
    page: int | None,
    section: str | None,
) -> str:
    lines = [f"[教材] {_bounded_label(document.logical_title)}"]
    if document.part_number is not None:
        lines.append(f"[分片] {document.part_number}")
    if section:
        lines.append(f"[章节] {_bounded_label(section)}")
    if page is not None:
        lines.append(f"[页码] {page}")
    return "\n".join(lines)


def _split_to_units(text: str, budget: int) -> list[str]:
    sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(text) if item.strip()]
    units: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > budget:
            if current:
                units.append(current)
                current = ""
            units.extend(
                sentence[start : start + budget].strip()
                for start in range(0, len(sentence), budget)
                if sentence[start : start + budget].strip()
            )
            continue
        candidate = sentence if not current else current + sentence
        if current and len(candidate) > budget:
            units.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        units.append(current)
    return units


def _render(
    document: TextbookSourceDocument,
    body: str,
    *,
    page: int | None,
    section: str | None,
) -> str:
    return f"{_prefix(document, page=page, section=section)}\n\n{body.strip()}".strip()


def _chunk_blocks(
    document: TextbookSourceDocument,
    blocks: list[_TextBlock],
    policy: TextbookChunkingPolicy,
) -> list[tuple[str, int | None, int | None, str | None]]:
    if not blocks:
        return []
    results: list[tuple[str, int | None, int | None, str | None]] = []
    current_body = ""
    current_page = blocks[0].page
    current_page_end = blocks[0].page
    current_section = blocks[0].section

    def emit() -> None:
        nonlocal current_body
        if len(current_body.strip()) >= policy.min_content_chars:
            results.append(
                (
                    current_body.strip(),
                    current_page,
                    current_page_end,
                    current_section,
                )
            )

    for block in blocks:
        prefix_size = len(_prefix(document, page=block.page, section=block.section)) + 2
        unit_budget = max(32, policy.target_chars - prefix_size)
        for unit in _split_to_units(block.text, unit_budget):
            if not current_body:
                current_page = block.page
                current_page_end = block.page
                current_section = block.section
            candidate = unit if not current_body else f"{current_body}\n\n{unit}"
            rendered = _render(
                document,
                candidate,
                page=current_page,
                section=current_section,
            )
            if current_body and len(rendered) > policy.target_chars:
                previous = current_body
                previous_page_end = current_page_end
                emit()
                overlap = previous[-policy.overlap_chars :].strip() if policy.overlap_chars else ""
                current_page = previous_page_end if overlap else block.page
                current_page_end = block.page
                current_section = block.section
                current_body = f"{overlap}\n\n{unit}".strip() if overlap else unit
                rendered = _render(
                    document,
                    current_body,
                    page=current_page,
                    section=current_section,
                )
                if len(rendered) > policy.max_chars and overlap:
                    current_body = unit
                    current_page = block.page
            else:
                current_body = candidate
                if block.page is not None:
                    current_page_end = block.page
            rendered = _render(
                document,
                current_body,
                page=current_page,
                section=current_section,
            )
            if len(rendered) > policy.max_chars:
                allowed = max(
                    1,
                    policy.max_chars
                    - len(_prefix(document, page=current_page, section=current_section))
                    - 2,
                )
                current_body = current_body[:allowed].rstrip()
    emit()
    return results


def chunk_textbook_document(
    document: TextbookSourceDocument,
    policy: TextbookChunkingPolicy | None = None,
) -> tuple[TextbookChunk, ...]:
    """Create deterministic citation-ready chunks from one Markdown source."""

    resolved_policy = policy or TextbookChunkingPolicy()
    blocks = _parse_blocks(document)
    prepared = _chunk_blocks(document, blocks, resolved_policy)

    chunks: list[TextbookChunk] = []
    for index, (body, page, page_end, section) in enumerate(prepared):
        text = _render(document, body, page=page, section=section)
        if len(text) > resolved_policy.max_chars:  # pragma: no cover - defensive invariant
            raise ValueError("chunk exceeded max_chars")
        chunks.append(
            TextbookChunk(
                chunk_index=index,
                text=text,
                page=page,
                page_end=page_end,
                section=section,
                content_sha256=sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(chunks)
