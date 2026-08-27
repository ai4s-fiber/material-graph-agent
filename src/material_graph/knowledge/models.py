"""Stable contracts for catalogued sources and retained evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SourceStatus = Literal[
    "metadata_discovered",
    "metadata_indexed",
    "deduplicated",
    "excluded_process_data",
    "selected_for_parse",
    "spooling",
    "parsing",
    "evidence_retained",
    "parsed_no_value",
    "indexed",
    "failed_retryable",
    "failed_permanent",
]

SourceKind = Literal[
    "literature",
    "patent",
    "standard",
    "textbook",
    "experiment",
    "industrial_data",
    "unknown",
]


class SourceLocator(BaseModel):
    """Internal logical location for a source or one citable fragment.

    ``relative_path`` is intentionally excluded from :meth:`to_public_uri` so
    reports and API clients cannot learn the NAS mount, QuickConnect route, or
    storage layout.
    """

    model_config = ConfigDict(extra="forbid")

    root_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    relative_path: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    table: str | None = None
    figure: str | None = None
    block_index: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate or "\x00" in candidate:
            raise ValueError("relative path is empty or contains NUL")
        parsed = urlsplit(candidate)
        windows_path = PureWindowsPath(candidate)
        normalized = candidate.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        if parsed.scheme or parsed.netloc:
            raise ValueError("source locator cannot contain a URL authority")
        if (
            normalized.startswith("/")
            or normalized.startswith("//")
            or windows_path.is_absolute()
            or bool(windows_path.drive)
        ):
            raise ValueError("source locator must be relative to a logical root")
        if ".." in posix_path.parts:
            raise ValueError("source locator cannot escape its logical root")
        return posix_path.as_posix()

    def to_public_uri(self, source_id: UUID) -> str:
        anchor = self.model_dump(
            exclude={"root_id", "relative_path"},
            exclude_none=True,
            mode="json",
        )
        suffix = urlencode(anchor)
        return f"source://{self.root_id}/{source_id}" + (f"#{suffix}" if suffix else "")


class SourceCatalogRecord(BaseModel):
    """Canonical metadata record; source bytes remain remote."""

    model_config = ConfigDict(extra="allow")

    source_id: UUID = Field(default_factory=uuid4)
    locator: SourceLocator
    source_kind: SourceKind = "unknown"
    display_title: str = Field(min_length=1)
    status: SourceStatus = "metadata_discovered"
    directory_year: int | None = Field(default=None, ge=1800, le=2200)
    normalized_doi: str | None = None
    application_number: str | None = None
    publication_number: str | None = None
    grant_number: str | None = None
    legal_status: str = "unknown"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    byte_size: int | None = Field(default=None, ge=0)
    material_category: str | None = None
    knowledge_domain: str = "domain_literature"
    canonical_source_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonical_record_cannot_reference_itself(self) -> "SourceCatalogRecord":
        if self.canonical_source_id == self.source_id:
            raise ValueError("canonical_source_id cannot reference the same record")
        return self


class SelectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    selected: bool
    reason_code: Literal[
        "active_evidence_gap",
        "task_semantic_match",
        "domain_pack_required_topic",
        "approved_curation",
        "duplicate",
        "process_data_excluded",
        "budget_deferred",
        "insufficient_metadata",
    ]
    task_id: UUID | None = None
    evidence_gap_id: UUID | None = None
    rank: int | None = Field(default=None, ge=1)
    policy_version: str = Field(min_length=1)


class EvidenceFragment(BaseModel):
    """Only a selected, citable block retained from a transient parse."""

    model_config = ConfigDict(extra="forbid")

    fragment_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    text: str = Field(min_length=1)
    locator: SourceLocator
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    retention_reason: str = Field(min_length=1)
    supported_entity_ids: list[str] = Field(default_factory=list)
    supported_relation_ids: list[str] = Field(default_factory=list)
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    embedding_generation_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_content_hash(self) -> "EvidenceFragment":
        if self.locator.root_id == "" or self.source_id is None:  # pragma: no cover
            raise ValueError("fragment requires a source")
        if not self.content_sha256:
            self.content_sha256 = sha256(self.text.encode("utf-8")).hexdigest()
        return self
