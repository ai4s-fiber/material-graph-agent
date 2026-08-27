"""Idempotent metadata catalog for remote, read-only source corpora.

The catalog never opens source bodies.  It normalizes metadata, records logical
remote versions, and reconciles duplicate/version relationships before any
source is eligible for the transient parsing pipeline.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Iterable, Literal, Protocol
from urllib.parse import unquote
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .models import SourceCatalogRecord, SourceLocator


RelationType = Literal["DUPLICATE_OF", "IS_VERSION_OF"]
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DOI_PREFIX = re.compile(
    r"^(?:doi\s*:\s*|https?://(?:(?:dx\.)?doi\.org)/)",
    re.IGNORECASE,
)
_INTERNAL_METADATA_KEYS = {
    "source_version_key",
    "remote_modified_at",
    "exclusion_reason",
}


class SourceRelation(BaseModel):
    """A non-destructive relationship between separately catalogued sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_type: RelationType
    source_id: UUID
    target_source_id: UUID
    normalized_doi: str | None = None
    reason: str = Field(min_length=1)


class CatalogWriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record: SourceCatalogRecord
    created: bool
    source_version_key: str


class CatalogStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_count: int = Field(default=0, ge=0)
    excluded_count: int = Field(default=0, ge=0)
    body_read_count: int = Field(default=0, ge=0)


class SourceCatalogRepository(Protocol):
    """Persistence boundary implemented by memory and future PostgreSQL stores."""

    def upsert(
        self,
        record: SourceCatalogRecord,
        *,
        remote_modified_at: datetime | str | int | float | None = None,
    ) -> CatalogWriteResult: ...

    def get(self, source_id: UUID) -> SourceCatalogRecord: ...

    def canonical_for(self, source_id: UUID) -> SourceCatalogRecord: ...

    def relations(self, relation_type: RelationType | None = None) -> list[SourceRelation]: ...


def normalize_doi(value: str | None) -> str | None:
    """Return a conservative canonical DOI suitable for exact-match deduplication."""

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", unquote(str(value))).strip()
    if not normalized:
        return None

    normalized = normalized.strip("<>[]{}\"' \t\r\n")
    normalized = _DOI_PREFIX.sub("", normalized, count=1).strip()
    normalized = normalized.casefold().rstrip(".,;:")
    while normalized.endswith((")", "]", "}")):
        closing, opening = normalized[-1], {")": "(", "]": "[", "}": "{"}[normalized[-1]]
        if normalized.count(closing) <= normalized.count(opening):
            break
        normalized = normalized[:-1].rstrip()

    if not _DOI_PATTERN.fullmatch(normalized):
        raise ValueError("value is not a valid DOI")
    return normalized


def _canonical_remote_modified_at(value: datetime | str | int | float | None) -> object:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        return value
    else:
        candidate = unicodedata.normalize("NFKC", value).strip()
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return candidate

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def build_source_version_key(
    *,
    locator: SourceLocator,
    byte_size: int | None,
    remote_modified_at: datetime | str | int | float | None,
) -> str:
    """Hash metadata available without reading a remote source body."""

    payload = {
        "root_id": locator.root_id,
        "relative_path": locator.relative_path,
        "byte_size": byte_size,
        "remote_modified_at": _canonical_remote_modified_at(remote_modified_at),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"source-version-v1:{sha256(encoded.encode('utf-8')).hexdigest()}"


def _has_value(value: object) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    return True


def _metadata_completeness(record: SourceCatalogRecord) -> int:
    typed_values = (
        record.normalized_doi,
        record.application_number,
        record.publication_number,
        record.grant_number,
        record.directory_year,
        record.sha256,
        record.byte_size,
        record.material_category,
    )
    score = sum(_has_value(value) for value in typed_values)
    score += sum(
        _has_value(value)
        for key, value in record.metadata.items()
        if key not in _INTERNAL_METADATA_KEYS and not key.startswith("_")
    )
    return score


def choose_canonical_source(records: Iterable[SourceCatalogRecord]) -> SourceCatalogRecord:
    """Choose the richest source, then the primary corpus, then a stable ID."""

    candidates = list(records)
    if not candidates:
        raise ValueError("at least one source is required")
    return min(
        candidates,
        key=lambda record: (
            -_metadata_completeness(record),
            0 if record.locator.root_id == "document_data_1" else 1,
            record.source_id.hex,
        ),
    )


def _is_excluded_process_data(locator: SourceLocator) -> bool:
    return locator.root_id == "data_2" and any(
        part.casefold() == "process_data" for part in locator.relative_path.split("/")
    )


def _merge_records(
    existing: SourceCatalogRecord,
    incoming: SourceCatalogRecord,
) -> SourceCatalogRecord:
    metadata = dict(existing.metadata)
    metadata.update(incoming.metadata)

    def prefer(incoming_value: object, existing_value: object) -> object:
        return incoming_value if _has_value(incoming_value) else existing_value

    legal_status = existing.legal_status
    if incoming.legal_status != "unknown" or legal_status == "unknown":
        legal_status = incoming.legal_status

    source_kind = existing.source_kind
    if incoming.source_kind != "unknown" or source_kind == "unknown":
        source_kind = incoming.source_kind

    return existing.model_copy(
        update={
            "display_title": prefer(incoming.display_title, existing.display_title),
            "source_kind": source_kind,
            "status": incoming.status,
            "directory_year": prefer(incoming.directory_year, existing.directory_year),
            "normalized_doi": prefer(incoming.normalized_doi, existing.normalized_doi),
            "application_number": prefer(
                incoming.application_number,
                existing.application_number,
            ),
            "publication_number": prefer(
                incoming.publication_number,
                existing.publication_number,
            ),
            "grant_number": prefer(incoming.grant_number, existing.grant_number),
            "legal_status": legal_status,
            "sha256": prefer(incoming.sha256, existing.sha256),
            "byte_size": prefer(incoming.byte_size, existing.byte_size),
            "material_category": prefer(
                incoming.material_category,
                existing.material_category,
            ),
            "knowledge_domain": prefer(
                incoming.knowledge_domain,
                existing.knowledge_domain,
            ),
            "metadata": metadata,
            "canonical_source_id": None,
        }
    )


class InMemorySourceCatalog:
    """Deterministic repository test double with production catalog semantics."""

    def __init__(self) -> None:
        self._records: dict[UUID, SourceCatalogRecord] = {}
        self._paths: dict[tuple[str, str], UUID] = {}
        self._relations: list[SourceRelation] = []
        self.stats = CatalogStats()

    def upsert(
        self,
        record: SourceCatalogRecord,
        *,
        remote_modified_at: datetime | str | int | float | None = None,
    ) -> CatalogWriteResult:
        prepared, version_key = self._prepare(record, remote_modified_at)
        path_key = (prepared.locator.root_id, prepared.locator.relative_path)
        existing_id = self._paths.get(path_key)
        created = existing_id is None

        if existing_id is None:
            stored = prepared.model_copy(deep=True)
            self._records[stored.source_id] = stored
            self._paths[path_key] = stored.source_id
        else:
            stored = _merge_records(self._records[existing_id], prepared)
            self._records[existing_id] = stored

        self._reconcile()
        current = self.get(stored.source_id)
        return CatalogWriteResult(
            record=current,
            created=created,
            source_version_key=version_key,
        )

    def _prepare(
        self,
        record: SourceCatalogRecord,
        remote_modified_at: datetime | str | int | float | None,
    ) -> tuple[SourceCatalogRecord, str]:
        normalized_doi = normalize_doi(record.normalized_doi)
        version_key = build_source_version_key(
            locator=record.locator,
            byte_size=record.byte_size,
            remote_modified_at=remote_modified_at,
        )
        metadata = dict(record.metadata)
        metadata["source_version_key"] = version_key
        canonical_mtime = _canonical_remote_modified_at(remote_modified_at)
        if canonical_mtime is not None:
            metadata["remote_modified_at"] = canonical_mtime

        status = record.status
        if _is_excluded_process_data(record.locator):
            status = "excluded_process_data"
            metadata["exclusion_reason"] = "process_data_never_open"

        return (
            record.model_copy(
                update={
                    "normalized_doi": normalized_doi,
                    "status": status,
                    "metadata": metadata,
                    "canonical_source_id": None,
                },
                deep=True,
            ),
            version_key,
        )

    def _reconcile(self) -> None:
        reset = {
            source_id: record.model_copy(update={"canonical_source_id": None})
            for source_id, record in self._records.items()
        }
        self._records = reset

        sha_groups: dict[str, list[SourceCatalogRecord]] = defaultdict(list)
        for record in self._records.values():
            if record.sha256:
                sha_groups[record.sha256].append(record)

        for records in sha_groups.values():
            if len(records) < 2:
                continue
            canonical = choose_canonical_source(records)
            for record in records:
                if record.source_id != canonical.source_id:
                    self._records[record.source_id] = record.model_copy(
                        update={"canonical_source_id": canonical.source_id}
                    )

        self._relations = []
        doi_groups: dict[str, list[SourceCatalogRecord]] = defaultdict(list)
        for record in self._records.values():
            if record.normalized_doi and record.sha256:
                doi_groups[record.normalized_doi].append(record)

        for doi, records in doi_groups.items():
            representatives: dict[str, SourceCatalogRecord] = {}
            for record in records:
                canonical = self.canonical_for(record.source_id)
                representatives[record.sha256 or ""] = canonical
            versions = list(representatives.values())
            if len(versions) < 2:
                continue
            canonical_version = choose_canonical_source(versions)
            for version in versions:
                if version.source_id == canonical_version.source_id:
                    continue
                self._relations.append(
                    SourceRelation(
                        relation_type="IS_VERSION_OF",
                        source_id=version.source_id,
                        target_source_id=canonical_version.source_id,
                        normalized_doi=doi,
                        reason="same_normalized_doi_with_different_sha256",
                    )
                )

        self._relations.sort(
            key=lambda relation: (
                relation.relation_type,
                relation.source_id.hex,
                relation.target_source_id.hex,
            )
        )
        self.stats = CatalogStats(
            source_count=len(self._records),
            excluded_count=sum(
                record.status == "excluded_process_data" for record in self._records.values()
            ),
            body_read_count=0,
        )

    def get(self, source_id: UUID) -> SourceCatalogRecord:
        try:
            return self._records[source_id].model_copy(deep=True)
        except KeyError as error:
            raise KeyError(f"unknown source: {source_id}") from error

    def canonical_for(self, source_id: UUID) -> SourceCatalogRecord:
        record = self.get(source_id)
        if record.canonical_source_id is None:
            return record
        return self.get(record.canonical_source_id)

    def relations(self, relation_type: RelationType | None = None) -> list[SourceRelation]:
        selected = self._relations
        if relation_type is not None:
            selected = [item for item in selected if item.relation_type == relation_type]
        return [item.model_copy(deep=True) for item in selected]

    def count(self) -> int:
        return len(self._records)

    def list_records(self) -> list[SourceCatalogRecord]:
        return [
            self.get(source_id) for source_id in sorted(self._records, key=lambda item: item.hex)
        ]
