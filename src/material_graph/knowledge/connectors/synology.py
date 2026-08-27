"""Strictly read-only Synology File Station connector.

The transport boundary is intentionally compatible with a narrowly adapted
``N4S4/synology-api`` v0.9.1 session.  QuickConnect discovery, authentication,
relay cookies, Noise and SynoToken handling stay transport-local; this module
only exposes approved File Station reads and never persists connection state in
directory cursors.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, Self
from urllib.parse import urlsplit

from ..remote_reader import (
    DirectoryCursor,
    RemoteEntry,
    RemoteRangeContractError,
    RemoteSourceReader,
    RemoteStat,
    normalize_identifier,
    normalize_relative_path,
    validate_open_ended_range_response,
)


ALLOWED_API_METHODS = (
    ("SYNO.FileStation.Info", "get"),
    ("SYNO.FileStation.List", "list_share"),
    ("SYNO.FileStation.List", "list"),
    ("SYNO.FileStation.List", "getinfo"),
    ("SYNO.FileStation.Download", "download"),
)

_REQUIRED_APIS = (
    "SYNO.FileStation.Info",
    "SYNO.FileStation.List",
    "SYNO.FileStation.Download",
)
_LOCAL_MIN_VERSION = 1
_LOCAL_MAX_VERSION = 2


class SynologyProtocolError(RuntimeError):
    """The NAS returned data that violates the read-only connector contract."""


class SynologyRangeError(SynologyProtocolError):
    """A byte-range response cannot safely resume the requested object."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SynologyApiError(RuntimeError):
    """A File Station JSON API call returned a DSM error envelope."""

    def __init__(self, *, api_name: str, method: str, code: object) -> None:
        self.api_name = api_name
        self.method = method
        self.code = code
        super().__init__(f"Synology API {api_name}/{method} failed with code {code!s}")


class SynologyStreamResponse(Protocol):
    """Minimal streaming response supplied by the authenticated transport."""

    status_code: int
    headers: Mapping[str, str]

    def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class N4S4ReadTransportV091(Protocol):
    """Narrow adapter around one authenticated ``synology-api`` v0.9.1 session.

    A production adapter may add device-token persistence, but token, endpoint,
    cookie and session values must remain private to that adapter.
    """

    async def login(self) -> None: ...

    async def query_api_info(
        self,
        api_names: tuple[str, ...],
    ) -> Mapping[str, Mapping[str, object]]: ...

    async def request_json(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    async def open_stream(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
        headers: Mapping[str, str] | None,
    ) -> SynologyStreamResponse: ...

    async def logout(self) -> None: ...

    def clear_local_state(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _ApiSpec:
    path: str
    version: int


def _require_non_negative(value: int | None, *, field: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{field} must be a non-negative integer")


def _require_positive(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _normalize_remote_absolute(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SynologyProtocolError(f"invalid {field}")
    if "\\" in value or not value.startswith("/") or value.startswith("//"):
        raise SynologyProtocolError(f"invalid {field}")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise SynologyProtocolError(f"invalid {field}")

    segments = value.split("/")[1:]
    if any(segment in {"", ".", ".."} for segment in segments[:-1]):
        raise SynologyProtocolError(f"ambiguous {field}")
    if segments and segments[-1] in {".", ".."}:
        raise SynologyProtocolError(f"ambiguous {field}")
    normalized = value.rstrip("/") or "/"
    return normalized


def _configured_slice_path(value: object) -> str:
    path = _normalize_remote_absolute(value, field="configured slice path")
    if path == "/":
        raise ValueError("configured slice path cannot expose the DSM root")
    return path


def _optional_uint(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SynologyProtocolError(f"invalid {field}")
    return value


def _required_uint(value: object, *, field: str) -> int:
    parsed = _optional_uint(value, field=field)
    if parsed is None:
        raise SynologyProtocolError(f"missing {field}")
    return parsed


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SynologyProtocolError("download returned invalid HTTP headers")
        if key.casefold() == expected:
            return value
    return None


def _reject_json_response(response: SynologyStreamResponse) -> None:
    content_type = _header(response.headers, "Content-Type")
    if content_type is None:
        return
    media_type = content_type.partition(";")[0].strip().casefold()
    if media_type == "application/json" or media_type.endswith("+json"):
        raise SynologyProtocolError("download returned a JSON response")


class SynologyFileStationReader(RemoteSourceReader):
    """Read approved corpus slices through one authenticated File Station session."""

    def __init__(
        self,
        *,
        transport: N4S4ReadTransportV091,
        roots: Mapping[str, Mapping[str, str]],
    ) -> None:
        if not isinstance(roots, Mapping) or not roots:
            raise ValueError("roots must contain at least one logical root")

        approved: dict[str, dict[str, str]] = {}
        for raw_root_id, raw_slices in roots.items():
            root_id = normalize_identifier(raw_root_id, field="root_id")
            if not isinstance(raw_slices, Mapping) or not raw_slices:
                raise ValueError(f"root {root_id!r} must contain at least one slice")
            slices: dict[str, str] = {}
            for raw_slice_id, raw_path in raw_slices.items():
                slice_id = normalize_identifier(raw_slice_id, field="slice_id")
                try:
                    slices[slice_id] = _configured_slice_path(raw_path)
                except SynologyProtocolError as exc:
                    raise ValueError(str(exc)) from exc
            approved[root_id] = slices

        self._transport: N4S4ReadTransportV091 | None = transport
        self._roots = approved
        self._api_info: dict[str, _ApiSpec] = {}
        self._connected = False
        self._closed = False
        self.last_cursor: DirectoryCursor | None = None
        self.last_effective_stream_offset: int | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def transport_attached(self) -> bool:
        return self._transport is not None

    async def connect(self) -> Self:
        """Authenticate and negotiate only the File Station APIs we consume."""

        if self._connected:
            return self
        transport = self._transport
        if transport is None or self._closed:
            raise RuntimeError("reader is closed")

        try:
            await transport.login()
            discovered = await transport.query_api_info(_REQUIRED_APIS)
            self._api_info = self._negotiate_api_info(discovered)
        except BaseException:
            await self._cleanup_failed_connect(transport)
            raise

        self._connected = True
        return self

    @staticmethod
    def _negotiate_api_info(
        discovered: Mapping[str, Mapping[str, object]],
    ) -> dict[str, _ApiSpec]:
        if not isinstance(discovered, Mapping):
            raise SynologyProtocolError("missing API information")

        negotiated: dict[str, _ApiSpec] = {}
        for api_name in _REQUIRED_APIS:
            raw = discovered.get(api_name)
            if not isinstance(raw, Mapping):
                raise SynologyProtocolError(f"missing API information for {api_name}")

            raw_path = raw.get("path")
            try:
                api_path = normalize_relative_path(raw_path)  # type: ignore[arg-type]
            except ValueError as exc:
                raise SynologyProtocolError(f"invalid API path for {api_name}") from exc
            parsed_path = urlsplit(api_path)
            if parsed_path.query or parsed_path.fragment:
                raise SynologyProtocolError(f"invalid API path for {api_name}")

            minimum = raw.get("minVersion")
            maximum = raw.get("maxVersion")
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or minimum < 0
                or maximum < minimum
            ):
                raise SynologyProtocolError(f"invalid API version range for {api_name}")

            version = min(maximum, _LOCAL_MAX_VERSION)
            if version < max(minimum, _LOCAL_MIN_VERSION):
                raise SynologyProtocolError(f"unsupported API version for {api_name}")
            negotiated[api_name] = _ApiSpec(path=api_path, version=version)
        return negotiated

    async def _cleanup_failed_connect(self, transport: N4S4ReadTransportV091) -> None:
        try:
            await transport.logout()
        except BaseException:
            pass
        try:
            transport.clear_local_state()
        except BaseException:
            pass
        finally:
            self._transport = None
            self._api_info.clear()
            self._connected = False
            self._closed = True

    def _require_connected(self) -> N4S4ReadTransportV091:
        if not self._connected or self._transport is None or self._closed:
            raise RuntimeError("reader is not connected")
        return self._transport

    def _resolve_slice(self, root_id: str, slice_id: str) -> str:
        normalized_root = normalize_identifier(root_id, field="root_id")
        normalized_slice = normalize_identifier(slice_id, field="slice_id")
        slices = self._roots.get(normalized_root)
        if slices is None:
            raise KeyError(f"unknown root: {normalized_root}")
        try:
            return slices[normalized_slice]
        except KeyError as exc:
            raise KeyError(f"unknown slice: {normalized_slice}") from exc

    @staticmethod
    def _join_remote(base: str, relative_path: str, *, allow_root: bool) -> str:
        relative = normalize_relative_path(relative_path, allow_root=allow_root)
        if relative == ".":
            return base
        return f"{base}/{relative}"

    def _api_spec(self, api_name: str, method: str) -> _ApiSpec:
        if (api_name, method) not in ALLOWED_API_METHODS:
            raise SynologyProtocolError("attempted a non-approved File Station operation")
        try:
            return self._api_info[api_name]
        except KeyError as exc:
            raise SynologyProtocolError(f"API was not negotiated: {api_name}") from exc

    async def _request_data(
        self,
        api_name: str,
        method: str,
        extra_params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        transport = self._require_connected()
        spec = self._api_spec(api_name, method)
        params: dict[str, object] = {"version": spec.version, "method": method}
        if extra_params:
            params.update(extra_params)
        payload = await transport.request_json(
            api_name=api_name,
            api_path=spec.path,
            params=params,
        )
        if not isinstance(payload, Mapping):
            raise SynologyProtocolError("File Station returned a non-object JSON response")
        if payload.get("success") is False:
            error = payload.get("error")
            code = error.get("code", "unknown") if isinstance(error, Mapping) else "unknown"
            if isinstance(code, bool) or not isinstance(code, int):
                code = "unknown"
            raise SynologyApiError(api_name=api_name, method=method, code=code)
        if payload.get("success") is not True:
            raise SynologyProtocolError("File Station returned an invalid JSON envelope")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise SynologyProtocolError("File Station response is missing object data")
        return data

    async def get_info(self) -> Mapping[str, object]:
        return await self._request_data("SYNO.FileStation.Info", "get")

    @staticmethod
    def _relative_from_remote(base: str, remote_path: object) -> str:
        normalized = _normalize_remote_absolute(remote_path, field="File Station path")
        try:
            relative = PurePosixPath(normalized).relative_to(PurePosixPath(base))
        except ValueError as exc:
            raise SynologyProtocolError("File Station path is outside approved slice") from exc
        return relative.as_posix()

    @staticmethod
    def _metadata(raw: Mapping[str, object]) -> tuple[int | None, int | None]:
        additional = raw.get("additional")
        if additional is None:
            return None, None
        if not isinstance(additional, Mapping):
            raise SynologyProtocolError("invalid File Station additional metadata")
        size = _optional_uint(additional.get("size"), field="File Station size")
        raw_time = additional.get("time")
        if raw_time is None:
            modified_at = None
        elif isinstance(raw_time, Mapping):
            modified_at = _optional_uint(
                raw_time.get("mtime"),
                field="File Station mtime",
            )
        else:
            raise SynologyProtocolError("invalid File Station time metadata")
        return size, modified_at

    def _entry_from_payload(
        self,
        *,
        root_id: str,
        slice_id: str,
        base: str,
        raw: object,
    ) -> RemoteEntry:
        if not isinstance(raw, Mapping):
            raise SynologyProtocolError("directory listing contains a non-object entry")
        relative = self._relative_from_remote(base, raw.get("path"))
        if relative == ".":
            raise SynologyProtocolError("directory listing returned the slice root as an entry")

        name = raw.get("name")
        if not isinstance(name, str) or name != PurePosixPath(relative).name:
            raise SynologyProtocolError("directory entry name does not match its path")
        is_dir = raw.get("isdir")
        if not isinstance(is_dir, bool):
            raise SynologyProtocolError("directory entry has an invalid directory flag")
        size, modified_at = self._metadata(raw)
        return RemoteEntry(
            root_id=root_id,
            slice_id=slice_id,
            relative_path=relative,
            name=name,
            is_dir=is_dir,
            byte_size=size,
            modified_at=modified_at,
        )

    def iter_entries(
        self,
        root_id: str,
        slice_id: str,
        *,
        cursor: DirectoryCursor | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[RemoteEntry]:
        async def iterate() -> AsyncIterator[RemoteEntry]:
            self._require_connected()
            _require_positive(page_size, field="page_size")
            base = self._resolve_slice(root_id, slice_id)

            if cursor is None:
                current = DirectoryCursor(root_id=root_id, slice_id=slice_id)
            else:
                if not isinstance(cursor, DirectoryCursor):
                    raise TypeError("cursor must be a DirectoryCursor")
                if cursor.root_id != root_id or cursor.slice_id != slice_id:
                    raise ValueError("cursor does not belong to the requested root and slice")
                current = cursor

            folder_path = self._join_remote(
                base,
                current.relative_directory,
                allow_root=True,
            )
            seen_offsets: set[int] = set()
            while True:
                requested_offset = current.offset
                seen_offsets.add(requested_offset)
                data = await self._request_data(
                    "SYNO.FileStation.List",
                    "list",
                    {
                        "folder_path": folder_path,
                        "offset": requested_offset,
                        "limit": page_size,
                        "additional": '["size","time"]',
                    },
                )
                response_offset = _required_uint(
                    data.get("offset"),
                    field="directory response offset",
                )
                total = _required_uint(data.get("total"), field="directory total")
                raw_files = data.get("files")
                if not isinstance(raw_files, Sequence) or isinstance(
                    raw_files, (str, bytes, bytearray)
                ):
                    raise SynologyProtocolError("directory response contains invalid files")

                page_entries = [
                    self._entry_from_payload(
                        root_id=root_id,
                        slice_id=slice_id,
                        base=base,
                        raw=raw,
                    )
                    for raw in raw_files
                ]
                next_cursor = current.advance(
                    response_offset=response_offset,
                    item_count=len(page_entries),
                )

                for entry in page_entries:
                    yield entry
                self.last_cursor = next_cursor

                if next_cursor.offset >= total:
                    break
                if not page_entries or next_cursor.offset in seen_offsets:
                    raise SynologyProtocolError("directory pagination made no progress")
                current = next_cursor

        return iterate()

    async def stat(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
    ) -> RemoteStat:
        self._require_connected()
        base = self._resolve_slice(root_id, slice_id)
        target = self._join_remote(base, relative_path, allow_root=True)
        data = await self._request_data(
            "SYNO.FileStation.List",
            "getinfo",
            {
                "path": [target],
                "additional": '["size","time"]',
            },
        )
        raw_files = data.get("files")
        if (
            not isinstance(raw_files, Sequence)
            or isinstance(raw_files, (str, bytes, bytearray))
            or len(raw_files) != 1
            or not isinstance(raw_files[0], Mapping)
        ):
            raise SynologyProtocolError("getinfo must return exactly one file object")

        raw = raw_files[0]
        returned = _normalize_remote_absolute(raw.get("path"), field="getinfo path")
        if returned != target:
            self._relative_from_remote(base, returned)
            raise SynologyProtocolError("getinfo returned a different approved path")
        relative = self._relative_from_remote(base, returned)
        is_dir = raw.get("isdir")
        if not isinstance(is_dir, bool):
            raise SynologyProtocolError("getinfo returned an invalid directory flag")
        size, modified_at = self._metadata(raw)
        return RemoteStat(
            root_id=root_id,
            slice_id=slice_id,
            relative_path=relative,
            is_dir=is_dir,
            byte_size=size,
            modified_at=modified_at,
        )

    async def _open_download(
        self,
        *,
        target: str,
        headers: Mapping[str, str] | None,
    ) -> SynologyStreamResponse:
        transport = self._require_connected()
        spec = self._api_spec("SYNO.FileStation.Download", "download")
        return await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path=spec.path,
            params={
                "version": spec.version,
                "method": "download",
                "path": target,
                "mode": "download",
            },
            headers=headers,
        )

    @staticmethod
    def _validate_content_range(
        response: SynologyStreamResponse,
        *,
        requested_offset: int,
        expected_size: int | None,
    ) -> int:
        try:
            validated = validate_open_ended_range_response(
                requested_offset=requested_offset,
                status_code=response.status_code,
                headers=response.headers,
                expected_total=expected_size,
            )
        except RemoteRangeContractError as exc:
            raise SynologyRangeError(exc.code) from None
        return validated.length

    @staticmethod
    def _validate_content_length(
        response: SynologyStreamResponse,
        *,
        expected_size: int | None,
    ) -> None:
        if expected_size is None:
            return
        raw = _header(response.headers, "Content-Length")
        if raw is None:
            return
        try:
            length = int(raw)
        except ValueError as exc:
            raise SynologyProtocolError("download returned an invalid Content-Length") from exc
        if length < 0 or length != expected_size:
            raise SynologyProtocolError("download size does not match expected size")

    @staticmethod
    async def _yield_response(
        response: SynologyStreamResponse,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        async for chunk in response.iter_bytes(chunk_size):
            if not isinstance(chunk, bytes):
                raise SynologyProtocolError("download stream yielded a non-bytes chunk")
            if chunk:
                yield chunk

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
    ) -> AsyncIterator[bytes]:
        async def stream() -> AsyncIterator[bytes]:
            self._require_connected()
            _require_non_negative(offset, field="offset")
            _require_non_negative(expected_size, field="expected_size")
            _require_non_negative(expected_mtime, field="expected_mtime")
            _require_positive(chunk_size, field="chunk_size")
            if expected_size is not None and offset > expected_size:
                raise ValueError("offset cannot exceed expected_size")

            base = self._resolve_slice(root_id, slice_id)
            target = self._join_remote(base, relative_path, allow_root=False)
            self.last_effective_stream_offset = None

            if expected_mtime is not None:
                current = await self.stat(root_id, slice_id, relative_path)
                if current.is_dir:
                    raise SynologyProtocolError("cannot stream a directory")
                if current.modified_at != expected_mtime:
                    raise SynologyProtocolError("remote source changed before streaming")
                if expected_size is not None and current.byte_size != expected_size:
                    raise SynologyProtocolError("remote source changed before streaming")

            if offset > 0:
                response = await self._open_download(
                    target=target,
                    headers={"Range": f"bytes={offset}-"},
                )
                try:
                    _reject_json_response(response)
                    self._validate_content_range(
                        response,
                        requested_offset=offset,
                        expected_size=expected_size,
                    )
                    self.last_effective_stream_offset = offset
                    async for chunk in self._yield_response(
                        response,
                        chunk_size=chunk_size,
                    ):
                        yield chunk
                    return
                finally:
                    await response.close()

            response = await self._open_download(target=target, headers=None)
            try:
                _reject_json_response(response)
                if response.status_code == 416:
                    raise SynologyRangeError("range_not_satisfiable")
                if response.status_code != 200:
                    raise SynologyProtocolError(
                        f"unexpected download status: {response.status_code}"
                    )
                self._validate_content_length(response, expected_size=expected_size)
                self.last_effective_stream_offset = 0
                async for chunk in self._yield_response(response, chunk_size=chunk_size):
                    yield chunk
            finally:
                await response.close()

        return stream()

    async def close(self) -> None:
        transport = self._transport
        if transport is None:
            self._api_info.clear()
            self._connected = False
            self._closed = True
            return

        logout_error: BaseException | None = None
        clear_error: BaseException | None = None
        try:
            if self._connected:
                await transport.logout()
        except BaseException as exc:
            logout_error = exc
        try:
            transport.clear_local_state()
        except BaseException as exc:
            clear_error = exc
        finally:
            self._transport = None
            self._api_info.clear()
            self._connected = False
            self._closed = True

        if logout_error is not None:
            raise logout_error
        if clear_error is not None:
            raise clear_error


__all__ = [
    "ALLOWED_API_METHODS",
    "N4S4ReadTransportV091",
    "SynologyApiError",
    "SynologyFileStationReader",
    "SynologyProtocolError",
    "SynologyRangeError",
    "SynologyStreamResponse",
]
