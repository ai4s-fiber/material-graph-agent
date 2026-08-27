"""Read-only discovery of locally processed textbook Markdown."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import re
import unicodedata
from uuid import NAMESPACE_URL, UUID, uuid5


_SOURCE_HU_PART = re.compile(r"^(?P<title>.+)__part(?P<part>\d+)$", re.IGNORECASE)
_IGNORED_TOP_LEVEL_DIRECTORIES = frozenset({"source_hu", "_mineru_workdirs", "_staging_inputs"})


class TextbookCorpusError(RuntimeError):
    """Raised when the local textbook corpus cannot be safely discovered."""


@dataclass(frozen=True, slots=True)
class TextbookSourceDocument:
    """One decoded Markdown source with deterministic provenance."""

    source_id: UUID
    root_id: str
    relative_path: str
    logical_title: str
    normalized_title: str
    source_family: str
    part_number: int | None
    content_sha256: str
    byte_size: int
    text: str
    duplicate_of: UUID | None = None


@dataclass(frozen=True, slots=True)
class TextbookCorpusInventory:
    """Complete discovery result, including duplicate observations."""

    documents: tuple[TextbookSourceDocument, ...]

    @property
    def discovered_document_count(self) -> int:
        return len(self.documents)

    @property
    def unique_documents(self) -> tuple[TextbookSourceDocument, ...]:
        return tuple(document for document in self.documents if document.duplicate_of is None)

    @property
    def unique_document_count(self) -> int:
        return len(self.unique_documents)

    @property
    def duplicate_document_count(self) -> int:
        return self.discovered_document_count - self.unique_document_count

    @property
    def logical_book_count(self) -> int:
        return len({document.normalized_title for document in self.unique_documents})

    @property
    def total_source_bytes(self) -> int:
        return sum(document.byte_size for document in self.documents)


def normalize_textbook_title(value: str) -> str:
    """Return a conservative comparison key without collapsing editions."""

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def _normalized_markdown(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as error:
        raise TextbookCorpusError("textbook Markdown is not valid UTF-8") from error
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _build_document(
    *,
    root: Path,
    path: Path,
    root_id: str,
    source_family: str,
    logical_title: str,
    part_number: int | None,
) -> TextbookSourceDocument:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise TextbookCorpusError("textbook Markdown is unreadable") from error
    text = _normalized_markdown(raw)
    if not text:
        raise TextbookCorpusError("textbook Markdown is empty")
    relative_path = path.relative_to(root).as_posix()
    identity = f"material-graph:textbook-source:v1:{root_id}:{relative_path}"
    return TextbookSourceDocument(
        source_id=uuid5(NAMESPACE_URL, identity),
        root_id=root_id,
        relative_path=relative_path,
        logical_title=logical_title.strip(),
        normalized_title=normalize_textbook_title(logical_title),
        source_family=source_family,
        part_number=part_number,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
        byte_size=len(raw),
        text=text,
    )


def _mineru_markdown_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if (
            not directory.is_dir()
            or directory.name in _IGNORED_TOP_LEVEL_DIRECTORIES
            or directory.name.startswith("_")
        ):
            continue
        candidates = [
            path for path in directory.rglob("*.md") if path.parent.name.casefold() == "ocr"
        ]
        paths.extend(sorted(candidates, key=lambda item: item.as_posix().casefold()))
    return paths


def _source_hu_markdown_paths(root: Path) -> list[Path]:
    source_hu = root / "source_hu"
    if not source_hu.is_dir():
        return []
    return sorted(source_hu.rglob("*.md"), key=lambda item: item.as_posix().casefold())


def _mark_duplicates(
    documents: list[TextbookSourceDocument],
) -> tuple[TextbookSourceDocument, ...]:
    canonical_by_hash: dict[str, UUID] = {}
    marked: list[TextbookSourceDocument] = []
    for document in sorted(
        documents,
        key=lambda item: (
            0 if item.source_family == "mineru_markdown" else 1,
            item.relative_path.casefold(),
        ),
    ):
        canonical_id = canonical_by_hash.get(document.content_sha256)
        if canonical_id is None:
            canonical_by_hash[document.content_sha256] = document.source_id
            marked.append(document)
        else:
            marked.append(replace(document, duplicate_of=canonical_id))
    return tuple(sorted(marked, key=lambda item: item.relative_path.casefold()))


def discover_textbook_corpus(root: str | Path) -> TextbookCorpusInventory:
    """Discover the frozen local corpus without mutating any source file."""

    corpus_root = Path(root)
    if not corpus_root.is_dir():
        raise TextbookCorpusError("textbook corpus root is not a directory")
    try:
        corpus_root = corpus_root.resolve(strict=True)
        mineru_paths = _mineru_markdown_paths(corpus_root)
        source_hu_paths = _source_hu_markdown_paths(corpus_root)
    except OSError as error:
        raise TextbookCorpusError("textbook corpus cannot be enumerated") from error

    documents: list[TextbookSourceDocument] = []
    for path in mineru_paths:
        documents.append(
            _build_document(
                root=corpus_root,
                path=path,
                root_id="cyj_mineru",
                source_family="mineru_markdown",
                logical_title=path.stem,
                part_number=None,
            )
        )
    for path in source_hu_paths:
        match = _SOURCE_HU_PART.fullmatch(path.stem)
        logical_title = match.group("title") if match else path.stem
        part_number = int(match.group("part")) if match else None
        documents.append(
            _build_document(
                root=corpus_root,
                path=path,
                root_id="cyj_source_hu",
                source_family="source_hu_markdown",
                logical_title=logical_title,
                part_number=part_number,
            )
        )
    return TextbookCorpusInventory(documents=_mark_duplicates(documents))
