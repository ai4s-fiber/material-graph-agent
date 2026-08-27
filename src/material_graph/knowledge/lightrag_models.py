"""Typed LightRAG insertion, tracking, and source-reconciliation contracts."""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import EvidenceFragment, SourceLocator


_BASENAME_PATTERN = r"^mg_[0-9a-f]{32}_[0-9a-f]{32}_[0-9a-f]{16}\.txt$"


def build_lightrag_basename(fragment: EvidenceFragment) -> str:
    """Return the deterministic basename LightRAG may safely canonicalize."""

    digest = fragment.content_sha256 or sha256(fragment.text.encode("utf-8")).hexdigest()
    return f"mg_{fragment.source_id.hex}_{fragment.fragment_id.hex}_{digest[:16]}.txt"


class LightRAGSourceMapping(BaseModel):
    """Durable bridge from a LightRAG basename to exact source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    basename: str = Field(pattern=_BASENAME_PATTERN)
    fragment_id: UUID
    source_id: UUID
    locator: SourceLocator
    logical_source_uri: str = Field(pattern=r"^source://")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_generation_id: str = Field(min_length=1)

    @classmethod
    def from_fragment(cls, fragment: EvidenceFragment) -> "LightRAGSourceMapping":
        digest = fragment.content_sha256 or sha256(fragment.text.encode("utf-8")).hexdigest()
        return cls(
            basename=build_lightrag_basename(fragment),
            fragment_id=fragment.fragment_id,
            source_id=fragment.source_id,
            locator=fragment.locator,
            logical_source_uri=fragment.locator.to_public_uri(fragment.source_id),
            content_sha256=digest,
            embedding_generation_id=fragment.embedding_generation_id,
        )

    @model_validator(mode="after")
    def validate_reconciliation_fields(self) -> "LightRAGSourceMapping":
        expected_basename = (
            f"mg_{self.source_id.hex}_{self.fragment_id.hex}_{self.content_sha256[:16]}.txt"
        )
        if self.basename != expected_basename:
            raise ValueError("basename does not match source, fragment, and content hash")
        if self.logical_source_uri != self.locator.to_public_uri(self.source_id):
            raise ValueError("logical_source_uri does not match the full source locator")
        return self


class LightRAGFixedTokenParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_token_size: int = Field(default=1200, gt=0)
    chunk_overlap_token_size: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "LightRAGFixedTokenParams":
        if self.chunk_overlap_token_size >= self.chunk_token_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class LightRAGChunking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["fixed_token"] = "fixed_token"
    params: LightRAGFixedTokenParams = Field(default_factory=LightRAGFixedTokenParams)


class LightRAGTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    file_source: str = Field(pattern=_BASENAME_PATTERN)
    chunking: LightRAGChunking = Field(default_factory=LightRAGChunking)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("retained evidence text cannot be blank")
        return stripped


class LightRAGTextsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    texts: list[str] = Field(min_length=1)
    file_sources: list[str] = Field(min_length=1)
    chunking: LightRAGChunking = Field(default_factory=LightRAGChunking)

    @field_validator("texts")
    @classmethod
    def strip_texts(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("retained evidence text cannot be blank")
        return stripped

    @field_validator("file_sources")
    @classmethod
    def validate_file_sources(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("file_sources must be unique")
        return values

    @model_validator(mode="after")
    def validate_parallel_lists(self) -> "LightRAGTextsRequest":
        if len(self.texts) != len(self.file_sources):
            raise ValueError("one file_source is required for every retained evidence text")
        return self


class LightRAGQueryRequest(BaseModel):
    """Pinned request contract for LightRAG v1.5.4 structured retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=3)
    mode: Literal["mix"] = "mix"
    only_need_context: Literal[True] = True
    top_k: int = Field(ge=1, le=100)
    chunk_top_k: int = Field(ge=1, le=200)
    enable_rerank: Literal[True] = True
    include_references: Literal[True] = True
    include_chunk_content: Literal[True] = True

    @classmethod
    def for_query(cls, query: str, *, top_k: int) -> "LightRAGQueryRequest":
        return cls(
            query=query,
            top_k=top_k,
            chunk_top_k=max(top_k * 2, 12),
        )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("query must contain at least three non-space characters")
        return stripped

    @model_validator(mode="after")
    def validate_chunk_budget(self) -> "LightRAGQueryRequest":
        if self.chunk_top_k < self.top_k:
            raise ValueError("chunk_top_k cannot be smaller than top_k")
        return self


class LightRAGQueryReference(BaseModel):
    """Reference row returned by LightRAG and reconciled before public use."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    reference_id: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    content: list[str] = Field(default_factory=list)

    @field_validator("reference_id", "file_path")
    @classmethod
    def reject_blank_identity(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reference identity cannot be blank")
        return stripped

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("content")
    @classmethod
    def strip_content(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class LightRAGQueryData(BaseModel):
    """Structured data boundary returned by the official /query/data route."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    references: list[LightRAGQueryReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_references(self) -> "LightRAGQueryData":
        reference_ids = [reference.reference_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("LightRAG reference IDs must be unique")
        return self


class LightRAGQueryEnvelope(BaseModel):
    """Top-level response contract for LightRAG v1.5.4 query data."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["success", "failure"]
    message: str
    data: LightRAGQueryData
    metadata: dict[str, Any] = Field(default_factory=dict)


class LightRAGInsertAcceptance(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["success", "partial_success", "failure"]
    message: str
    track_id: str = Field(min_length=1)


class LightRAGDocumentState(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    PREPROCESSED = "preprocessed"
    PROCESSED = "processed"
    FAILED = "failed"


_TERMINAL_STATES = {
    LightRAGDocumentState.PROCESSED,
    LightRAGDocumentState.FAILED,
}


class LightRAGDocumentStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    status: LightRAGDocumentState
    file_path: str = Field(min_length=1)
    error_msg: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("error_msg", mode="before")
    @classmethod
    def sanitize_error_message(cls, value: object) -> str | None:
        if value is None or value == "":
            return None
        return "processing_failed"


class LightRAGTrackStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    track_id: str = Field(min_length=1)
    documents: list[LightRAGDocumentStatus] = Field(default_factory=list)
    total_count: int = Field(ge=0)
    status_summary: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_document_count(self) -> "LightRAGTrackStatus":
        if self.total_count != len(self.documents):
            raise ValueError("track total_count does not match documents")
        return self

    @property
    def is_terminal(self) -> bool:
        return bool(self.documents) and all(
            document.status in _TERMINAL_STATES for document in self.documents
        )

    @property
    def has_failures(self) -> bool:
        return any(document.status is LightRAGDocumentState.FAILED for document in self.documents)


class LightRAGInsertResult(BaseModel):
    """Secret-free result safe for state/checkpoint serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["processed", "failed", "idempotent_conflict"]
    mappings: list[LightRAGSourceMapping] = Field(min_length=1)
    track_id: str | None = None
    track_status: LightRAGTrackStatus | None = None
    message: str = ""

    @field_validator("message")
    @classmethod
    def validate_safe_message(cls, value: str) -> str:
        if value not in {"", "accepted", "idempotent_conflict"}:
            raise ValueError("message must be a stable safe status")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> "LightRAGInsertResult":
        if self.outcome == "idempotent_conflict":
            if self.track_id is not None or self.track_status is not None:
                raise ValueError("idempotent conflict cannot claim a completed track")
            return self
        if self.track_id is None or self.track_status is None:
            raise ValueError("terminal insertion result requires track status")
        if self.track_id != self.track_status.track_id:
            raise ValueError("track IDs do not match")
        expected = "failed" if self.track_status.has_failures else "processed"
        if self.outcome != expected:
            raise ValueError("outcome does not match terminal document statuses")
        return self
