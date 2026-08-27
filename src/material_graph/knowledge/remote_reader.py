"""Read-only contracts for remote corpus discovery and byte streaming.

The contracts deliberately contain only logical root/slice identifiers and
relative paths.  Connection endpoints, credentials, sessions, and device
tokens belong to connector-local transports and can never enter a cursor.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_OPEN_ENDED_BYTE_RANGE = re.compile(r"^bytes=(\d+)-$")
_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_CURSOR_FIELDS = frozenset(
    {"schema_version", "root_id", "slice_id", "relative_directory", "offset"}
)


def normalize_identifier(value: str, *, field: str) -> str:
    """Return a stable logical identifier or fail closed."""

    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def normalize_relative_path(value: str, *, allow_root: bool = False) -> str:
    """Normalize a remote relative path without allowing root escape."""

    if not isinstance(value, str):
        raise ValueError("relative path must be a string")
    candidate = value.strip().replace("\\", "/")
    if not candidate or "\x00" in candidate:
        raise ValueError("relative path is empty or contains NUL")

    parsed = urlsplit(candidate)
    windows = PureWindowsPath(value)
    path = PurePosixPath(candidate)
    if parsed.scheme or parsed.netloc:
        raise ValueError("relative path cannot contain a URL authority")
    if candidate.startswith(("/", "//")) or windows.is_absolute() or windows.drive:
        raise ValueError("relative path cannot be absolute")
    if ".." in path.parts:
        raise ValueError("relative path cannot escape its logical root")

    normalized = path.as_posix()
    if normalized == "." and not allow_root:
        raise ValueError("relative path must identify an entry")
    return normalized


def _validate_non_negative(value: int | None, *, field: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RemoteEntry:
    """One directory entry addressed only through a logical corpus root."""

    root_id: str
    slice_id: str
    relative_path: str
    name: str
    is_dir: bool
    byte_size: int | None = None
    modified_at: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", normalize_identifier(self.root_id, field="root_id"))
        object.__setattr__(
            self,
            "slice_id",
            normalize_identifier(self.slice_id, field="slice_id"),
        )
        object.__setattr__(self, "relative_path", normalize_relative_path(self.relative_path))
        if not isinstance(self.name, str) or not self.name or "\x00" in self.name:
            raise ValueError("entry name is empty or contains NUL")
        if "/" in self.name or "\\" in self.name:
            raise ValueError("entry name cannot contain a path separator")
        if not isinstance(self.is_dir, bool):
            raise ValueError("is_dir must be a boolean")
        _validate_non_negative(self.byte_size, field="byte_size")
        _validate_non_negative(self.modified_at, field="modified_at")


@dataclass(frozen=True, slots=True)
class RemoteStat:
    """Safe metadata returned by a remote ``getinfo`` operation."""

    root_id: str
    slice_id: str
    relative_path: str
    is_dir: bool
    byte_size: int | None = None
    modified_at: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", normalize_identifier(self.root_id, field="root_id"))
        object.__setattr__(
            self,
            "slice_id",
            normalize_identifier(self.slice_id, field="slice_id"),
        )
        object.__setattr__(
            self,
            "relative_path",
            normalize_relative_path(self.relative_path, allow_root=True),
        )
        if not isinstance(self.is_dir, bool):
            raise ValueError("is_dir must be a boolean")
        _validate_non_negative(self.byte_size, field="byte_size")
        _validate_non_negative(self.modified_at, field="modified_at")


@dataclass(frozen=True, slots=True)
class DirectoryCursor:
    """Durable offset cursor containing no connection or authentication state."""

    root_id: str
    slice_id: str
    relative_directory: str = "."
    offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_id", normalize_identifier(self.root_id, field="root_id"))
        object.__setattr__(
            self,
            "slice_id",
            normalize_identifier(self.slice_id, field="slice_id"),
        )
        object.__setattr__(
            self,
            "relative_directory",
            normalize_relative_path(self.relative_directory, allow_root=True),
        )
        _validate_non_negative(self.offset, field="offset")

    def advance(self, *, response_offset: int, item_count: int) -> "DirectoryCursor":
        """Advance from the offset reported by DSM, not the requested limit."""

        _validate_non_negative(response_offset, field="response_offset")
        _validate_non_negative(item_count, field="item_count")
        assert response_offset is not None and item_count is not None
        return replace(self, offset=response_offset + item_count)

    def to_checkpoint(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "root_id": self.root_id,
            "slice_id": self.slice_id,
            "relative_directory": self.relative_directory,
            "offset": self.offset,
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, object]) -> "DirectoryCursor":
        fields = frozenset(payload)
        extra = fields - _CURSOR_FIELDS
        missing = _CURSOR_FIELDS - fields
        if extra:
            raise ValueError(f"unsupported cursor fields: {', '.join(sorted(extra))}")
        if missing:
            raise ValueError(f"missing cursor fields: {', '.join(sorted(missing))}")
        if payload["schema_version"] != 1:
            raise ValueError("unsupported cursor schema version")
        return cls(
            root_id=str(payload["root_id"]),
            slice_id=str(payload["slice_id"]),
            relative_directory=str(payload["relative_directory"]),
            offset=payload["offset"],  # type: ignore[arg-type]
        )


class RemoteLineTooLongError(ValueError):
    """A physical record exceeded the configured bounded line buffer."""


class RemoteRangeContractError(RuntimeError):
    """A range request or response violated the resumable-read contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ValidatedByteRange:
    """Validated inclusive byte range returned for an open-ended request."""

    start: int
    end: int
    total: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def parse_open_ended_byte_range(value: object) -> int:
    """Parse ``Range: bytes=<offset>-`` without accepting ambiguous syntax."""

    match = _OPEN_ENDED_BYTE_RANGE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise RemoteRangeContractError("range_request_invalid")
    return int(match.group(1))


def _single_http_header(headers: Mapping[str, str], name: str) -> str | None:
    if not isinstance(headers, Mapping):
        raise RemoteRangeContractError("range_response_headers_invalid")
    expected = name.casefold()
    matches: list[str] = []
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise RemoteRangeContractError("range_response_headers_invalid")
        if key.casefold() == expected:
            matches.append(value)
    if len(matches) > 1:
        raise RemoteRangeContractError("range_response_headers_invalid")
    return matches[0] if matches else None


def validate_open_ended_range_response(
    *,
    requested_offset: int,
    status_code: int,
    headers: Mapping[str, str],
    expected_total: int | None = None,
) -> ValidatedByteRange:
    """Validate a response before consuming bytes from a resumed read.

    A resume is accepted only when the provider returns ``206`` and an exact
    ``Content-Range`` beginning at the requested offset. An ignored Range
    request (normally ``200``) is rejected rather than silently rescanned.
    """

    if (
        isinstance(requested_offset, bool)
        or not isinstance(requested_offset, int)
        or requested_offset < 0
    ):
        raise RemoteRangeContractError("range_request_invalid")
    if expected_total is not None and (
        isinstance(expected_total, bool)
        or not isinstance(expected_total, int)
        or expected_total < 0
    ):
        raise RemoteRangeContractError("range_expected_total_invalid")
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise RemoteRangeContractError("range_response_status_invalid")
    if status_code == 416:
        raise RemoteRangeContractError("range_not_satisfiable")
    if status_code != 206:
        raise RemoteRangeContractError("range_response_not_partial")

    raw_content_range = _single_http_header(headers, "Content-Range")
    match = (
        _CONTENT_RANGE.fullmatch(raw_content_range.strip())
        if raw_content_range is not None
        else None
    )
    if match is None:
        raise RemoteRangeContractError("content_range_invalid")
    start, end, total = (int(part) for part in match.groups())
    if start != requested_offset:
        raise RemoteRangeContractError("content_range_start_mismatch")
    if end < start or total <= end or end != total - 1:
        raise RemoteRangeContractError("content_range_bounds_invalid")
    if expected_total is not None and total != expected_total:
        raise RemoteRangeContractError("content_range_total_mismatch")

    validated = ValidatedByteRange(start=start, end=end, total=total)
    raw_content_length = _single_http_header(headers, "Content-Length")
    if raw_content_length is not None:
        stripped = raw_content_length.strip()
        if re.fullmatch(r"\d+", stripped) is None:
            raise RemoteRangeContractError("content_length_invalid")
        if int(stripped) != validated.length:
            raise RemoteRangeContractError("content_length_mismatch")
    return validated


class RemoteSourceReader(ABC):
    """Abstract, read-only source interface shared by every remote connector."""

    @abstractmethod
    def iter_entries(
        self,
        root_id: str,
        slice_id: str,
        *,
        cursor: DirectoryCursor | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[RemoteEntry]: ...

    @abstractmethod
    async def stat(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
    ) -> RemoteStat: ...

    @abstractmethod
    def open_stream(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
        *,
        offset: int = 0,
        expected_size: int | None = None,
        expected_mtime: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]: ...

    async def iter_lines(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
        *,
        offset: int = 0,
        expected_size: int | None = None,
        expected_mtime: int | None = None,
        chunk_size: int = 1024 * 1024,
        max_line_bytes: int = 2 * 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        """Yield newline-preserving records with a finite per-line buffer."""

        if (
            isinstance(max_line_bytes, bool)
            or not isinstance(max_line_bytes, int)
            or max_line_bytes <= 0
        ):
            raise ValueError("max_line_bytes must be a positive integer")
        pending = bytearray()
        async for chunk in self.open_stream(
            root_id,
            slice_id,
            relative_path,
            offset=offset,
            expected_size=expected_size,
            expected_mtime=expected_mtime,
            chunk_size=chunk_size,
        ):
            if not isinstance(chunk, bytes):
                raise TypeError("remote stream yielded a non-bytes chunk")
            start = 0
            while (newline := chunk.find(b"\n", start)) >= 0:
                end = newline + 1
                segment = chunk[start:end]
                if len(pending) + len(segment) > max_line_bytes:
                    raise RemoteLineTooLongError("remote line exceeds max_line_bytes")
                pending.extend(segment)
                yield bytes(pending)
                pending.clear()
                start = end
            tail = chunk[start:]
            if len(pending) + len(tail) > max_line_bytes:
                raise RemoteLineTooLongError("remote line exceeds max_line_bytes")
            pending.extend(tail)
        if pending:
            yield bytes(pending)

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "RemoteSourceReader":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


__all__ = [
    "DirectoryCursor",
    "RemoteEntry",
    "RemoteSourceReader",
    "RemoteLineTooLongError",
    "RemoteRangeContractError",
    "RemoteStat",
    "ValidatedByteRange",
    "normalize_identifier",
    "normalize_relative_path",
    "parse_open_ended_byte_range",
    "validate_open_ended_range_response",
]
