"""Async read-only bridge for the locked Synology API patch stack.

The upstream v0.9.1 source plus the checked-in audited patch remains responsible
for QuickConnect discovery, DSM authentication, cookies, device tokens, and
HTTP streaming. This module only adapts that synchronous session to the narrow
asynchronous transport consumed by SynologyFileStationReader.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Protocol

from ..remote_reader import (
    RemoteRangeContractError,
    parse_open_ended_byte_range,
    validate_open_ended_range_response,
)
from .synology import ALLOWED_API_METHODS, SynologyStreamResponse


_ALLOWED_APIS = frozenset(api_name for api_name, _ in ALLOWED_API_METHODS)
_JSON_METHODS = frozenset(
    pair for pair in ALLOWED_API_METHODS if pair != ("SYNO.FileStation.Download", "download")
)
_STREAM_METHOD = ("SYNO.FileStation.Download", "download")
_SAFE_RESPONSE_HEADERS = frozenset({"content-length", "content-range", "content-type"})
_END_OF_STREAM = object()


class SynologyTransportError(RuntimeError):
    """Stable, credential-free failure raised by the session bridge."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RequestsResponseV091(Protocol):
    """Subset of requests.Response used without importing requests."""

    status_code: int
    headers: Mapping[str, str]

    def iter_content(self, chunk_size: int) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class SynologyApiSessionV091(Protocol):
    """Subset exposed by the locally locked v0.9.1 streaming patch."""

    full_api_list: Mapping[str, Mapping[str, object]]

    def login(self) -> None: ...

    def get_api_list(self, app: str | None = None) -> None: ...

    def request_data(
        self,
        api_name: str,
        api_path: str,
        req_param: dict[str, object],
        method: str | None = None,
        data: object | None = None,
        response_json: bool = True,
    ) -> object: ...

    def request_stream(
        self,
        api_name: str,
        api_path: str,
        req_param: dict[str, object],
        method: str = "get",
        data: object | None = None,
        headers: dict[str, object] | None = None,
        timeout: float | tuple[float, float] | None = None,
    ) -> RequestsResponseV091: ...

    def logout(self) -> None: ...


def _next_or_end(iterator: Iterator[bytes]) -> bytes | object:
    try:
        return next(iterator)
    except StopIteration:
        return _END_OF_STREAM


def _validated_timeout(value: tuple[float, float]) -> tuple[float, float]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        or any(item <= 0 for item in value)
    ):
        raise ValueError("request_timeout must contain two positive numbers")
    return float(value[0]), float(value[1])


class RequestsStreamAdapter(SynologyStreamResponse):
    """Own one unbuffered requests response and expose async chunks."""

    def __init__(self, response: RequestsResponseV091) -> None:
        status_code = response.status_code
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise SynologyTransportError("invalid_stream_status")
        if not isinstance(response.headers, Mapping):
            raise SynologyTransportError("invalid_stream_headers")
        headers: dict[str, str] = {}
        for key, value in response.headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SynologyTransportError("invalid_stream_headers")
            if key.casefold() in _SAFE_RESPONSE_HEADERS:
                headers[key] = value

        self.status_code = status_code
        self.headers = headers
        self._response: RequestsResponseV091 | None = response
        self._iterating = False
        self._closed = False

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self._closed or self._response is None:
            raise SynologyTransportError("stream_closed")
        if self._iterating:
            raise SynologyTransportError("stream_already_consumed")

        self._iterating = True
        iterator_failed = False
        try:
            iterator = self._response.iter_content(chunk_size=chunk_size)
        except Exception:
            iterator_failed = True
            iterator = iter(())
        if iterator_failed:
            self._iterating = False
            raise SynologyTransportError("stream_read_failed")
        try:
            while True:
                read_failed = False
                try:
                    chunk = await asyncio.to_thread(_next_or_end, iterator)
                except Exception:
                    read_failed = True
                    chunk = _END_OF_STREAM
                if read_failed:
                    raise SynologyTransportError("stream_read_failed")
                if chunk is _END_OF_STREAM:
                    break
                if not isinstance(chunk, (bytes, bytearray)):
                    raise SynologyTransportError("stream_returned_non_bytes")
                if chunk:
                    yield bytes(chunk)
        finally:
            self._iterating = False

    async def close(self) -> None:
        if self._closed:
            return
        response = self._response
        self._response = None
        self._closed = True
        if response is None:
            return
        close_failed = False
        try:
            await asyncio.to_thread(response.close)
        except Exception:
            close_failed = True
        if close_failed:
            raise SynologyTransportError("stream_close_failed")


class SynologyApiTransportV091:
    """Narrow async adapter around one authenticated public-fork session."""

    def __init__(
        self,
        session: SynologyApiSessionV091,
        *,
        request_timeout: tuple[float, float] = (10.0, 120.0),
        require_tls_verification: bool = True,
    ) -> None:
        if getattr(session, "_debug", False) is True:
            raise ValueError("synology session debug output must be disabled")
        if getattr(session, "_secure", True) is not True:
            raise ValueError("synology session must use HTTPS")
        if require_tls_verification and getattr(session, "_verify", True) is not True:
            raise ValueError("synology session must verify TLS certificates")

        self._session: SynologyApiSessionV091 | None = session
        self._request_timeout = _validated_timeout(request_timeout)
        self._api_paths: dict[str, str] = {}
        self._logged_in = False
        self._closed = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(logged_in={self._logged_in!r}, closed={self._closed!r})"

    def _require_session(self) -> SynologyApiSessionV091:
        if self._closed or self._session is None:
            raise SynologyTransportError("transport_closed")
        return self._session

    def _require_logged_in(self) -> SynologyApiSessionV091:
        session = self._require_session()
        if not self._logged_in:
            raise SynologyTransportError("transport_not_logged_in")
        return session

    async def login(self) -> None:
        if self._logged_in:
            return
        session = self._require_session()
        login_failed = False
        try:
            await asyncio.to_thread(session.login)
        except asyncio.CancelledError:
            raise
        except Exception:
            login_failed = True
        if login_failed:
            try:
                await asyncio.to_thread(session.logout)
            except Exception:
                pass
            self.clear_local_state()
            raise SynologyTransportError("login_failed")
        self._logged_in = True

    async def query_api_info(
        self,
        api_names: tuple[str, ...],
    ) -> Mapping[str, Mapping[str, object]]:
        session = self._require_logged_in()
        if (
            not isinstance(api_names, tuple)
            or not api_names
            or any(name not in _ALLOWED_APIS for name in api_names)
        ):
            raise ValueError("api_names contains a non-approved API")
        discovery_failed = False
        try:
            await asyncio.to_thread(session.get_api_list)
            discovered = session.full_api_list
        except Exception:
            discovery_failed = True
            discovered = {}
        if discovery_failed:
            raise SynologyTransportError("api_discovery_failed")
        if not isinstance(discovered, Mapping):
            raise SynologyTransportError("invalid_api_discovery")

        approved: dict[str, dict[str, object]] = {}
        paths: dict[str, str] = {}
        for api_name in api_names:
            raw = discovered.get(api_name)
            if not isinstance(raw, Mapping):
                continue
            record = {
                key: raw.get(key) for key in ("path", "minVersion", "maxVersion") if key in raw
            }
            approved[api_name] = record
            path = record.get("path")
            if isinstance(path, str) and path:
                paths[api_name] = path
        self._api_paths = paths
        return approved

    def _validate_operation(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
        stream: bool,
    ) -> str:
        method = params.get("method")
        expected = _STREAM_METHOD if stream else None
        if stream:
            if (api_name, method) != expected:
                raise ValueError("attempted a non-approved stream operation")
        elif (api_name, method) not in _JSON_METHODS:
            raise ValueError("attempted a non-approved JSON operation")
        if self._api_paths.get(api_name) != api_path:
            raise ValueError("api_path does not match negotiated API information")
        return str(method)

    async def request_json(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
    ) -> Mapping[str, object]:
        session = self._require_logged_in()
        self._validate_operation(
            api_name=api_name,
            api_path=api_path,
            params=params,
            stream=False,
        )
        payload = dict(params)
        request_failed = False
        try:
            response = await asyncio.to_thread(
                session.request_data,
                api_name,
                api_path,
                payload,
                "get",
                None,
                True,
            )
        except Exception:
            request_failed = True
            response = None
        if request_failed:
            raise SynologyTransportError("json_request_failed")
        if not isinstance(response, Mapping):
            raise SynologyTransportError("json_response_invalid")
        return dict(response)

    async def open_stream(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
        headers: Mapping[str, str] | None,
    ) -> RequestsStreamAdapter:
        session = self._require_logged_in()
        self._validate_operation(
            api_name=api_name,
            api_path=api_path,
            params=params,
            stream=True,
        )
        payload = dict(params)
        requested_offset: int | None = None
        request_headers: dict[str, str] | None = None
        if headers is not None:
            if not isinstance(headers, Mapping):
                raise ValueError("stream_headers_invalid")
            request_headers = {}
            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValueError("stream_headers_invalid")
                if key.casefold() != "range":
                    raise ValueError("stream_headers_not_approved")
                if requested_offset is not None:
                    raise ValueError("stream_headers_invalid")
                try:
                    requested_offset = parse_open_ended_byte_range(value)
                except RemoteRangeContractError as exc:
                    raise SynologyTransportError(exc.code) from None
                request_headers[key] = value
        request_failed = False
        try:
            response = await asyncio.to_thread(
                session.request_stream,
                api_name,
                api_path,
                payload,
                "get",
                None,
                request_headers,
                self._request_timeout,
            )
        except Exception:
            request_failed = True
            response = None
        if request_failed or response is None:
            raise SynologyTransportError("stream_open_failed")
        adapter = RequestsStreamAdapter(response)
        if requested_offset is not None:
            try:
                validate_open_ended_range_response(
                    requested_offset=requested_offset,
                    status_code=adapter.status_code,
                    headers=adapter.headers,
                )
            except RemoteRangeContractError as exc:
                try:
                    await adapter.close()
                except SynologyTransportError:
                    pass
                raise SynologyTransportError(exc.code) from None
        return adapter

    async def logout(self) -> None:
        session = self._session
        self.clear_local_state()
        if session is None:
            return
        logout_failed = False
        try:
            await asyncio.to_thread(session.logout)
        except Exception:
            logout_failed = True
        if logout_failed:
            raise SynologyTransportError("logout_failed")

    def clear_local_state(self) -> None:
        self._session = None
        self._api_paths.clear()
        self._logged_in = False
        self._closed = True


__all__ = [
    "RequestsResponseV091",
    "RequestsStreamAdapter",
    "SynologyApiSessionV091",
    "SynologyApiTransportV091",
    "SynologyTransportError",
]
