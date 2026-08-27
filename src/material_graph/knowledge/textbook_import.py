"""Preparation of local textbook chunks for the existing LightRAG boundary."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator
from uuid import NAMESPACE_URL, uuid5

from .models import EvidenceFragment, SourceLocator
from .textbook_chunking import TextbookChunkingPolicy, chunk_textbook_document
from .textbook_corpus import TextbookCorpusInventory, discover_textbook_corpus


_PARSER_VERSION = "local-textbook-v1"


@dataclass(frozen=True, slots=True)
class PreparedTextbookCorpus:
    """Deduplicated evidence fragments plus reproducibility counters."""

    inventory: TextbookCorpusInventory
    fragments: tuple[EvidenceFragment, ...]
    corpus_digest: str
    embedding_generation_id: str
    chunking_policy: TextbookChunkingPolicy

    @property
    def discovered_document_count(self) -> int:
        return self.inventory.discovered_document_count

    @property
    def unique_document_count(self) -> int:
        return self.inventory.unique_document_count

    @property
    def duplicate_document_count(self) -> int:
        return self.inventory.duplicate_document_count

    @property
    def logical_book_count(self) -> int:
        return self.inventory.logical_book_count

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    @property
    def total_fragment_chars(self) -> int:
        return sum(len(fragment.text) for fragment in self.fragments)


def _corpus_digest(
    inventory: TextbookCorpusInventory,
    *,
    embedding_generation_id: str,
    policy: TextbookChunkingPolicy,
) -> str:
    payload = {
        "chunking": {
            "max_chars": policy.max_chars,
            "min_content_chars": policy.min_content_chars,
            "overlap_chars": policy.overlap_chars,
            "target_chars": policy.target_chars,
        },
        "chunker_schema": "heading-coalescing-v1",
        "documents": [
            {
                "content_sha256": document.content_sha256,
                "relative_path": document.relative_path,
                "root_id": document.root_id,
                "source_id": str(document.source_id),
            }
            for document in inventory.unique_documents
        ],
        "embedding_generation_id": embedding_generation_id,
        "schema": "local-textbook-corpus-v1",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def prepare_textbook_corpus(
    root: str | Path,
    *,
    embedding_generation_id: str,
    chunking_policy: TextbookChunkingPolicy | None = None,
) -> PreparedTextbookCorpus:
    """Discover, deduplicate, and chunk the frozen local textbook corpus."""

    generation = embedding_generation_id.strip()
    if not generation:
        raise ValueError("embedding_generation_id is required")
    policy = chunking_policy or TextbookChunkingPolicy()
    inventory = discover_textbook_corpus(root)
    fragments: list[EvidenceFragment] = []
    for document in inventory.unique_documents:
        chunks = chunk_textbook_document(document, policy)
        for chunk in chunks:
            identity = (
                "material-graph:textbook-fragment:v1:"
                f"{document.source_id}:{chunk.chunk_index}:{chunk.content_sha256}"
            )
            fragments.append(
                EvidenceFragment(
                    fragment_id=uuid5(NAMESPACE_URL, identity),
                    source_id=document.source_id,
                    text=chunk.text,
                    locator=SourceLocator(
                        root_id=document.root_id,
                        relative_path=document.relative_path,
                        page=chunk.page,
                        section=chunk.section,
                        block_index=chunk.chunk_index,
                    ),
                    content_sha256=chunk.content_sha256,
                    retention_reason="textbook_full_corpus",
                    parser_name=document.source_family,
                    parser_version=_PARSER_VERSION,
                    embedding_generation_id=generation,
                    metadata={
                        "chunk_index": chunk.chunk_index,
                        "document_content_sha256": document.content_sha256,
                        "logical_title": document.logical_title,
                        "page_end": chunk.page_end,
                        "part_number": document.part_number,
                        "source_family": document.source_family,
                    },
                )
            )
    return PreparedTextbookCorpus(
        inventory=inventory,
        fragments=tuple(fragments),
        corpus_digest=_corpus_digest(
            inventory,
            embedding_generation_id=generation,
            policy=policy,
        ),
        embedding_generation_id=generation,
        chunking_policy=policy,
    )


def iter_fragment_jsonl(prepared: PreparedTextbookCorpus) -> Iterator[str]:
    """Yield stable, UTF-8-ready JSONL records without absolute source paths."""

    for fragment in prepared.fragments:
        yield json.dumps(
            fragment.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
