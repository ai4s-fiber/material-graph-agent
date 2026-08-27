"""Bounded, resumable remote manifest ingestion."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import unicodedata
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from .catalog import (
    CatalogWriteResult,
    SourceCatalogRepository,
    build_source_version_key,
    normalize_doi,
)
from .models import SourceCatalogRecord, SourceLocator
from .remote_reader import (
    RemoteLineTooLongError,
    RemoteSourceReader,
    normalize_identifier,
    normalize_relative_path,
)


ManifestFormat = Literal["jsonl", "csv"]
MetadataFailureClass = Literal["provider", "parse", "catalog", "cursor"]

_VERSION_KEY = re.compile(r"^source-version-v1:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CURSOR_FIELDS = frozenset(
    {
        "schema_version",
        "root_id",
        "slice_id",
        "manifest_path",
        "manifest_format",
        "manifest_version_key",
        "next_byte_offset",
        "records_committed",
        "csv_fieldnames",
    }
)
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "device_token",
        "endpoint",
        "host",
        "password",
        "passwd",
        "quickconnect_id",
        "refresh_token",
        "secret",
        "session",
        "sid",
        "synotoken",
        "token",
    }
)
_CONTROLLED_FIELDS = frozenset({"canonical_source_id", "source_id", "source_version_key", "status"})
_ALIASES = {
    "application_no": "application_number",
    "application_number": "application_number",
    "申请号": "application_number",
    "byte_size": "byte_size",
    "file_size": "byte_size",
    "size": "byte_size",
    "文件大小": "byte_size",
    "directory_year": "directory_year",
    "publication_year": "directory_year",
    "year": "directory_year",
    "公开年份": "directory_year",
    "年份": "directory_year",
    "display_title": "display_title",
    "name": "display_title",
    "title": "display_title",
    "专利名称": "display_title",
    "标题": "display_title",
    "论文名称": "display_title",
    "题名": "display_title",
    "doi": "normalized_doi",
    "normalized_doi": "normalized_doi",
    "document_path": "relative_path",
    "file_path": "relative_path",
    "path": "relative_path",
    "relative_path": "relative_path",
    "文件路径": "relative_path",
    "路径": "relative_path",
    "grant_no": "grant_number",
    "grant_number": "grant_number",
    "授权号": "grant_number",
    "kind": "source_kind",
    "document_type": "source_kind",
    "source_kind": "source_kind",
    "文献类型": "source_kind",
    "类型": "source_kind",
    "knowledge_domain": "knowledge_domain",
    "legal_status": "legal_status",
    "material_category": "material_category",
    "材料类别": "material_category",
    "metadata": "metadata",
    "modified_at": "remote_modified_at",
    "mtime": "remote_modified_at",
    "remote_modified_at": "remote_modified_at",
    "修改时间": "remote_modified_at",
    "publication_no": "publication_number",
    "publication_number": "publication_number",
    "公开号": "publication_number",
    "root_id": "root_id",
    "sha": "sha256",
    "sha256": "sha256",
    "slice_id": "slice_id",
}


class MetadataStreamError(RuntimeError):
    """Sanitized failure with a stable machine-readable classification."""

    def __init__(
        self,
        code: str,
        *,
        failure_class: MetadataFailureClass,
        retryable: bool,
    ) -> None:
        self.code = code
        self.failure_class = failure_class
        self.retryable = retryable
        super().__init__(code)


def _failure(
    code: str,
    *,
    failure_class: MetadataFailureClass = "parse",
    retryable: bool = False,
) -> MetadataStreamError:
    return MetadataStreamError(code, failure_class=failure_class, retryable=retryable)


@dataclass(frozen=True, slots=True)
class MetadataStreamLimits:
    """Per-line and per-record bounds for untrusted remote metadata."""

    max_line_bytes: int = 1024 * 1024
    max_fields: int = 128
    max_cell_bytes: int = 256 * 1024
    max_metadata_bytes: int = 2 * 1024 * 1024
    max_depth: int = 8
    stream_chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for field_name in (
            "max_line_bytes",
            "max_fields",
            "max_cell_bytes",
            "max_metadata_bytes",
            "max_depth",
            "stream_chunk_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_line_bytes > self.max_metadata_bytes:
            raise ValueError("max_line_bytes cannot exceed max_metadata_bytes")


@dataclass(frozen=True, slots=True)
class MetadataCursorKey:
    root_id: str
    slice_id: str
    manifest_path: str
    manifest_format: ManifestFormat

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", normalize_identifier(self.root_id, field="root_id"))
        object.__setattr__(
            self,
            "slice_id",
            normalize_identifier(self.slice_id, field="slice_id"),
        )
        object.__setattr__(
            self,
            "manifest_path",
            normalize_relative_path(self.manifest_path),
        )
        if self.manifest_format not in ("jsonl", "csv"):
            raise ValueError("unsupported manifest format")


@dataclass(frozen=True, slots=True)
class MetadataCursor:
    """Durable logical progress; never contains a raw row or authentication state."""

    root_id: str
    slice_id: str
    manifest_path: str
    manifest_format: ManifestFormat
    manifest_version_key: str
    next_byte_offset: int = 0
    records_committed: int = 0
    csv_fieldnames: tuple[str, ...] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        key = MetadataCursorKey(
            root_id=self.root_id,
            slice_id=self.slice_id,
            manifest_path=self.manifest_path,
            manifest_format=self.manifest_format,
        )
        object.__setattr__(self, "root_id", key.root_id)
        object.__setattr__(self, "slice_id", key.slice_id)
        object.__setattr__(self, "manifest_path", key.manifest_path)
        if self.schema_version != 1:
            raise ValueError("unsupported metadata cursor schema version")
        if not _VERSION_KEY.fullmatch(self.manifest_version_key):
            raise ValueError("invalid manifest version key")
        for name in ("next_byte_offset", "records_committed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.manifest_format == "jsonl" and self.csv_fieldnames is not None:
            raise ValueError("JSONL cursor cannot contain CSV field names")
        if self.csv_fieldnames is not None:
            _validate_fieldnames(self.csv_fieldnames)
        if (
            self.manifest_format == "csv"
            and self.next_byte_offset > 0
            and self.csv_fieldnames is None
        ):
            raise ValueError("resumed CSV cursor requires field names")

    @property
    def key(self) -> MetadataCursorKey:
        return MetadataCursorKey(
            root_id=self.root_id,
            slice_id=self.slice_id,
            manifest_path=self.manifest_path,
            manifest_format=self.manifest_format,
        )

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_id": self.root_id,
            "slice_id": self.slice_id,
            "manifest_path": self.manifest_path,
            "manifest_format": self.manifest_format,
            "manifest_version_key": self.manifest_version_key,
            "next_byte_offset": self.next_byte_offset,
            "records_committed": self.records_committed,
            "csv_fieldnames": list(self.csv_fieldnames) if self.csv_fieldnames else None,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "MetadataCursor":
        if frozenset(payload) != _CURSOR_FIELDS:
            raise ValueError("invalid metadata cursor fields")
        raw_fieldnames = payload["csv_fieldnames"]
        if raw_fieldnames is not None and (
            isinstance(raw_fieldnames, (str, bytes)) or not isinstance(raw_fieldnames, Sequence)
        ):
            raise ValueError("invalid CSV field names")
        if raw_fieldnames is not None and any(
            not isinstance(value, str) for value in raw_fieldnames
        ):
            raise ValueError("invalid CSV field names")
        fieldnames = tuple(raw_fieldnames) if raw_fieldnames is not None else None
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            root_id=payload["root_id"],  # type: ignore[arg-type]
            slice_id=payload["slice_id"],  # type: ignore[arg-type]
            manifest_path=payload["manifest_path"],  # type: ignore[arg-type]
            manifest_format=payload["manifest_format"],  # type: ignore[arg-type]
            manifest_version_key=payload["manifest_version_key"],  # type: ignore[arg-type]
            next_byte_offset=payload["next_byte_offset"],  # type: ignore[arg-type]
            records_committed=payload["records_committed"],  # type: ignore[arg-type]
            csv_fieldnames=fieldnames,
        )


class CursorRepository(Protocol):
    def load(self, key: MetadataCursorKey) -> MetadataCursor | None: ...

    def save(self, cursor: MetadataCursor) -> None: ...


class InMemoryCursorRepository:
    """Deterministic cursor repository for tests and single-process runs."""

    def __init__(self) -> None:
        self._cursors: dict[MetadataCursorKey, MetadataCursor] = {}

    def load(self, key: MetadataCursorKey) -> MetadataCursor | None:
        return self._cursors.get(key)

    def save(self, cursor: MetadataCursor) -> None:
        self._cursors[cursor.key] = cursor


@dataclass(frozen=True, slots=True)
class MetadataStreamResult:
    cursor: MetadataCursor
    records_seen: int
    records_created: int
    records_updated: int
    bounded_digest_required: int


class _DuplicateField(ValueError):
    pass


def _normalized_field_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return re.sub(r"[\s\-.]+", "_", normalized)


def _validate_fieldnames(fieldnames: Sequence[str], *, max_fields: int | None = None) -> None:
    if not fieldnames or (max_fields is not None and len(fieldnames) > max_fields):
        raise ValueError("invalid CSV field count")
    seen: set[str] = set()
    for fieldname in fieldnames:
        if not isinstance(fieldname, str) or not fieldname.strip():
            raise ValueError("invalid CSV field name")
        normalized = _normalized_field_name(fieldname.lstrip("\ufeff"))
        if not normalized:
            raise ValueError("invalid CSV field name")
        if normalized in seen:
            raise _DuplicateField("duplicate CSV field")
        if normalized in _SECRET_FIELDS or normalized in _CONTROLLED_FIELDS:
            raise ValueError("forbidden CSV field")
        seen.add(normalized)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    normalized_keys: set[str] = set()
    for key, value in pairs:
        normalized = _normalized_field_name(key)
        if normalized in normalized_keys:
            raise _DuplicateField("duplicate JSON field")
        normalized_keys.add(normalized)
        result[key] = value
    return result


def _decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _failure("metadata.parse.invalid_utf8") from None


def _utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise _failure("metadata.parse.invalid_utf8") from None


def _parse_json(raw: bytes) -> object:
    try:
        return json.loads(_decode_utf8(raw), object_pairs_hook=_strict_object)
    except MetadataStreamError:
        raise
    except _DuplicateField:
        raise _failure("metadata.parse.duplicate_field") from None
    except (json.JSONDecodeError, RecursionError, ValueError):
        raise _failure("metadata.parse.invalid_json") from None


def _validate_tree(value: object, limits: MetadataStreamLimits, *, depth: int = 0) -> None:
    if depth > limits.max_depth:
        raise _failure("metadata.parse.nesting_too_deep")
    if isinstance(value, str):
        if _utf8_size(value) > limits.max_cell_bytes:
            raise _failure("metadata.parse.cell_too_large")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _failure("metadata.parse.non_finite_number")
        return
    if isinstance(value, Mapping):
        if len(value) > limits.max_fields:
            raise _failure("metadata.parse.too_many_fields")
        normalized_keys: set[str] = set()
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise _failure("metadata.parse.invalid_field")
            if _utf8_size(raw_key) > limits.max_cell_bytes:
                raise _failure("metadata.parse.cell_too_large")
            key = _normalized_field_name(raw_key)
            if not key:
                raise _failure("metadata.parse.invalid_field")
            if key in normalized_keys:
                raise _failure("metadata.parse.duplicate_field")
            if key in _SECRET_FIELDS or key in _CONTROLLED_FIELDS:
                raise _failure("metadata.parse.forbidden_field")
            normalized_keys.add(key)
            _validate_tree(child, limits, depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > limits.max_fields:
            raise _failure("metadata.parse.too_many_items")
        for child in value:
            _validate_tree(child, limits, depth=depth + 1)
        return
    raise _failure("metadata.parse.invalid_value")


def _bounded_metadata_size(value: Mapping[str, object], limits: MetadataStreamLimits) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > limits.max_metadata_bytes:
        raise _failure("metadata.parse.object_too_large")


def _coerce_optional_int(value: object, *, code: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise _failure(code)
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except (ValueError, OverflowError):
            raise _failure(code) from None
    else:
        raise _failure(code)
    if parsed < 0:
        raise _failure(code)
    return parsed


def _coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _failure("metadata.parse.invalid_schema")
    stripped = value.strip()
    return stripped or None


def _source_id(locator: SourceLocator) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"material-graph-source-v1:{locator.root_id}:{locator.relative_path}",
    )


@dataclass(frozen=True, slots=True)
class _MappedRecord:
    record: SourceCatalogRecord
    remote_modified_at: str | int | float | None
    bounded_digest_required: bool


def _map_record(
    raw_record: Mapping[str, object],
    *,
    key: MetadataCursorKey,
    limits: MetadataStreamLimits,
) -> _MappedRecord:
    _validate_tree(raw_record, limits)
    canonical: dict[str, object] = {}
    extras: dict[str, object] = {}
    for raw_name, value in raw_record.items():
        normalized_name = _normalized_field_name(raw_name.lstrip("\ufeff"))
        canonical_name = _ALIASES.get(normalized_name)
        target = canonical if canonical_name is not None else extras
        target_name = canonical_name or normalized_name
        if target_name in target:
            raise _failure("metadata.parse.duplicate_field")
        target[target_name] = value

    declared_root = _coerce_optional_text(canonical.get("root_id"))
    declared_slice = _coerce_optional_text(canonical.get("slice_id"))
    if declared_root is not None and declared_root != key.root_id:
        raise _failure("metadata.parse.root_mismatch")
    if declared_slice is not None and declared_slice != key.slice_id:
        raise _failure("metadata.parse.slice_mismatch")

    raw_path = _coerce_optional_text(canonical.get("relative_path"))
    if raw_path is None:
        raise _failure("metadata.parse.missing_path")
    try:
        locator = SourceLocator(root_id=key.root_id, relative_path=raw_path)
    except ValueError:
        raise _failure("metadata.parse.invalid_path") from None

    title = _coerce_optional_text(canonical.get("display_title"))
    if title is None:
        title = PurePosixPath(locator.relative_path).name

    raw_doi = _coerce_optional_text(canonical.get("normalized_doi"))
    try:
        doi = normalize_doi(raw_doi)
    except ValueError:
        raise _failure("metadata.parse.invalid_doi") from None

    raw_sha = _coerce_optional_text(canonical.get("sha256"))
    digest = raw_sha.casefold() if raw_sha is not None else None
    if digest is not None and not _SHA256.fullmatch(digest):
        raise _failure("metadata.parse.invalid_sha256")

    raw_metadata = canonical.get("metadata")
    if raw_metadata in (None, ""):
        metadata: dict[str, object] = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    elif isinstance(raw_metadata, str):
        parsed_metadata = _parse_json(raw_metadata.encode("utf-8"))
        if not isinstance(parsed_metadata, Mapping):
            raise _failure("metadata.parse.invalid_metadata")
        metadata = dict(parsed_metadata)
    else:
        raise _failure("metadata.parse.invalid_metadata")

    metadata_names = {_normalized_field_name(name) for name in metadata}
    for field_name, value in extras.items():
        if value in (None, ""):
            continue
        if field_name in metadata_names:
            raise _failure("metadata.parse.duplicate_field")
        metadata[field_name] = value
        metadata_names.add(field_name)

    bounded_digest_required = doi is None and digest is None
    metadata["identity_basis"] = "doi" if doi else ("sha256" if digest else "logical_path")
    metadata["bounded_digest_required"] = bounded_digest_required

    byte_size = _coerce_optional_int(
        canonical.get("byte_size"),
        code="metadata.parse.invalid_byte_size",
    )
    remote_modified_at = canonical.get("remote_modified_at")
    if remote_modified_at == "":
        remote_modified_at = None
    if remote_modified_at is not None and (
        isinstance(remote_modified_at, bool)
        or not isinstance(remote_modified_at, (str, int, float))
    ):
        raise _failure("metadata.parse.invalid_modified_at")

    build_source_version_key(
        locator=locator,
        byte_size=byte_size,
        remote_modified_at=remote_modified_at,
    )
    _validate_tree(metadata, limits)
    _bounded_metadata_size(metadata, limits)

    source_kind = _coerce_optional_text(canonical.get("source_kind")) or "unknown"
    try:
        record = SourceCatalogRecord(
            source_id=_source_id(locator),
            locator=locator,
            source_kind=source_kind,  # type: ignore[arg-type]
            display_title=title,
            directory_year=_coerce_optional_int(
                canonical.get("directory_year"),
                code="metadata.parse.invalid_year",
            ),
            normalized_doi=doi,
            application_number=_coerce_optional_text(canonical.get("application_number")),
            publication_number=_coerce_optional_text(canonical.get("publication_number")),
            grant_number=_coerce_optional_text(canonical.get("grant_number")),
            legal_status=_coerce_optional_text(canonical.get("legal_status")) or "unknown",
            sha256=digest,
            byte_size=byte_size,
            material_category=_coerce_optional_text(canonical.get("material_category")),
            knowledge_domain=(
                _coerce_optional_text(canonical.get("knowledge_domain")) or "domain_literature"
            ),
            metadata=metadata,
        )
    except (TypeError, ValueError):
        raise _failure("metadata.parse.invalid_schema") from None
    return _MappedRecord(
        record=record,
        remote_modified_at=remote_modified_at,
        bounded_digest_required=bounded_digest_required,
    )


def _parse_csv_row(raw: bytes, limits: MetadataStreamLimits) -> list[str]:
    text = _decode_utf8(raw)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:
        raise _failure("metadata.parse.invalid_csv") from None
    if len(rows) != 1:
        raise _failure("metadata.parse.invalid_csv")
    row = rows[0]
    if len(row) > limits.max_fields:
        raise _failure("metadata.parse.too_many_fields")
    for cell in row:
        if _utf8_size(cell) > limits.max_cell_bytes:
            raise _failure("metadata.parse.cell_too_large")
    return row


def _csv_quote_state(raw: bytes, *, in_quotes: bool) -> bool:
    at_field_start = not in_quotes
    index = 0
    while index < len(raw):
        value = raw[index]
        if in_quotes:
            if value == 0x22:
                if index + 1 < len(raw) and raw[index + 1] == 0x22:
                    index += 2
                    continue
                in_quotes = False
            index += 1
            continue
        if value == 0x2C:
            at_field_start = True
        elif value == 0x22 and at_field_start:
            in_quotes = True
            at_field_start = False
        elif value not in (0x0A, 0x0D):
            at_field_start = False
        index += 1
    return in_quotes


class MetadataManifestIngestor:
    """Stream one remote metadata manifest into an idempotent source catalog."""

    def __init__(
        self,
        *,
        reader: RemoteSourceReader,
        catalog: SourceCatalogRepository,
        cursors: CursorRepository,
        limits: MetadataStreamLimits | None = None,
    ) -> None:
        self._reader = reader
        self._catalog = catalog
        self._cursors = cursors
        self._limits = limits or MetadataStreamLimits()

    async def ingest(
        self,
        *,
        root_id: str,
        slice_id: str,
        manifest_path: str,
        manifest_format: ManifestFormat,
    ) -> MetadataStreamResult:
        try:
            key = MetadataCursorKey(
                root_id=root_id,
                slice_id=slice_id,
                manifest_path=manifest_path,
                manifest_format=manifest_format,
            )
        except ValueError:
            raise _failure("metadata.parse.invalid_manifest_locator") from None

        try:
            stat = await self._reader.stat(key.root_id, key.slice_id, key.manifest_path)
        except Exception:
            raise _failure(
                "metadata.provider.stat_failed",
                failure_class="provider",
                retryable=True,
            ) from None
        if stat.is_dir:
            raise _failure("metadata.parse.manifest_is_directory")

        manifest_version_key = build_source_version_key(
            locator=SourceLocator(root_id=key.root_id, relative_path=key.manifest_path),
            byte_size=stat.byte_size,
            remote_modified_at=stat.modified_at,
        )
        cursor = self._load_cursor(key)
        if cursor is None:
            cursor = MetadataCursor(
                root_id=key.root_id,
                slice_id=key.slice_id,
                manifest_path=key.manifest_path,
                manifest_format=key.manifest_format,
                manifest_version_key=manifest_version_key,
            )
        elif cursor.manifest_version_key != manifest_version_key:
            raise _failure(
                "metadata.cursor.version_mismatch",
                failure_class="cursor",
                retryable=False,
            )
        if stat.byte_size is not None and cursor.next_byte_offset > stat.byte_size:
            raise _failure(
                "metadata.cursor.offset_out_of_range",
                failure_class="cursor",
                retryable=False,
            )

        if key.manifest_format == "jsonl":
            return await self._ingest_jsonl(
                key,
                cursor,
                expected_size=stat.byte_size,
                expected_mtime=stat.modified_at,
            )
        return await self._ingest_csv(
            key,
            cursor,
            expected_size=stat.byte_size,
            expected_mtime=stat.modified_at,
        )

    def _load_cursor(self, key: MetadataCursorKey) -> MetadataCursor | None:
        try:
            cursor = self._cursors.load(key)
        except Exception:
            raise _failure(
                "metadata.cursor.load_failed",
                failure_class="cursor",
                retryable=True,
            ) from None
        if cursor is not None and cursor.key != key:
            raise _failure(
                "metadata.cursor.identity_mismatch",
                failure_class="cursor",
                retryable=False,
            )
        return cursor

    def _save_cursor(self, cursor: MetadataCursor) -> None:
        try:
            self._cursors.save(cursor)
        except Exception:
            raise _failure(
                "metadata.cursor.save_failed",
                failure_class="cursor",
                retryable=True,
            ) from None

    async def _open_lines(
        self,
        key: MetadataCursorKey,
        *,
        offset: int,
        expected_size: int | None,
        expected_mtime: int | None,
    ) -> AsyncIterator[bytes]:
        try:
            async for raw_line in self._reader.iter_lines(
                key.root_id,
                key.slice_id,
                key.manifest_path,
                offset=offset,
                expected_size=expected_size,
                expected_mtime=expected_mtime,
                chunk_size=self._limits.stream_chunk_bytes,
                max_line_bytes=self._limits.max_line_bytes,
            ):
                if not isinstance(raw_line, bytes):
                    raise TypeError("metadata line must be bytes")
                yield raw_line
        except RemoteLineTooLongError:
            raise _failure("metadata.parse.line_too_large") from None
        except MetadataStreamError:
            raise
        except Exception:
            raise _failure(
                "metadata.provider.stream_failed",
                failure_class="provider",
                retryable=True,
            ) from None

    def _check_physical_line(self, raw_line: bytes) -> None:
        if len(raw_line) > self._limits.max_line_bytes:
            raise _failure("metadata.parse.line_too_large")

    def _upsert(self, mapped: _MappedRecord) -> CatalogWriteResult:
        try:
            return self._catalog.upsert(
                mapped.record,
                remote_modified_at=mapped.remote_modified_at,
            )
        except Exception:
            raise _failure(
                "metadata.catalog.upsert_failed",
                failure_class="catalog",
                retryable=True,
            ) from None

    async def _ingest_jsonl(
        self,
        key: MetadataCursorKey,
        cursor: MetadataCursor,
        *,
        expected_size: int | None,
        expected_mtime: int | None,
    ) -> MetadataStreamResult:
        seen = created = updated = digest_required = 0
        offset = cursor.next_byte_offset
        async for raw_line in self._open_lines(
            key,
            offset=offset,
            expected_size=expected_size,
            expected_mtime=expected_mtime,
        ):
            self._check_physical_line(raw_line)
            next_offset = offset + len(raw_line)
            payload = raw_line.rstrip(b"\r\n")
            if not payload:
                cursor = replace(cursor, next_byte_offset=next_offset)
                self._save_cursor(cursor)
                offset = next_offset
                continue
            decoded = _parse_json(payload)
            if not isinstance(decoded, Mapping):
                raise _failure("metadata.parse.invalid_schema")
            mapped = _map_record(decoded, key=key, limits=self._limits)
            write_result = self._upsert(mapped)
            cursor = replace(
                cursor,
                next_byte_offset=next_offset,
                records_committed=cursor.records_committed + 1,
            )
            self._save_cursor(cursor)
            seen += 1
            created += int(write_result.created)
            updated += int(not write_result.created)
            digest_required += int(mapped.bounded_digest_required)
            offset = next_offset
        return MetadataStreamResult(cursor, seen, created, updated, digest_required)

    async def _ingest_csv(
        self,
        key: MetadataCursorKey,
        cursor: MetadataCursor,
        *,
        expected_size: int | None,
        expected_mtime: int | None,
    ) -> MetadataStreamResult:
        seen = created = updated = digest_required = 0
        offset = cursor.next_byte_offset
        fieldnames = cursor.csv_fieldnames
        pending = bytearray()
        in_quotes = False

        async for raw_line in self._open_lines(
            key,
            offset=offset,
            expected_size=expected_size,
            expected_mtime=expected_mtime,
        ):
            self._check_physical_line(raw_line)
            if len(pending) + len(raw_line) > self._limits.max_metadata_bytes:
                raise _failure("metadata.parse.record_too_large")
            pending.extend(raw_line)
            in_quotes = _csv_quote_state(raw_line, in_quotes=in_quotes)
            if in_quotes:
                continue

            next_offset = offset + len(pending)
            row = _parse_csv_row(bytes(pending), self._limits)
            pending.clear()
            if fieldnames is None:
                try:
                    _validate_fieldnames(row, max_fields=self._limits.max_fields)
                except _DuplicateField:
                    raise _failure("metadata.parse.duplicate_field") from None
                except ValueError:
                    raise _failure("metadata.parse.invalid_csv_header") from None
                fieldnames = tuple(name.lstrip("\ufeff").strip() for name in row)
                cursor = replace(
                    cursor,
                    next_byte_offset=next_offset,
                    csv_fieldnames=fieldnames,
                )
                self._save_cursor(cursor)
                offset = next_offset
                continue
            if len(row) != len(fieldnames):
                raise _failure("metadata.parse.csv_field_count_mismatch")
            raw_record = dict(zip(fieldnames, row, strict=True))
            mapped = _map_record(raw_record, key=key, limits=self._limits)
            write_result = self._upsert(mapped)
            cursor = replace(
                cursor,
                next_byte_offset=next_offset,
                records_committed=cursor.records_committed + 1,
            )
            self._save_cursor(cursor)
            seen += 1
            created += int(write_result.created)
            updated += int(not write_result.created)
            digest_required += int(mapped.bounded_digest_required)
            offset = next_offset

        if pending or in_quotes:
            raise _failure("metadata.parse.invalid_csv")
        if fieldnames is None:
            raise _failure("metadata.parse.missing_csv_header")
        return MetadataStreamResult(cursor, seen, created, updated, digest_required)


__all__ = [
    "CursorRepository",
    "InMemoryCursorRepository",
    "ManifestFormat",
    "MetadataCursor",
    "MetadataCursorKey",
    "MetadataFailureClass",
    "MetadataManifestIngestor",
    "MetadataStreamError",
    "MetadataStreamLimits",
    "MetadataStreamResult",
]
