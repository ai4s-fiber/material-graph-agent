from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from material_graph.knowledge.connectors.synology_transport import (
    RequestsStreamAdapter,
    SynologyApiTransportV091,
    SynologyTransportError,
)
from material_graph.knowledge.connectors.synology import SynologyFileStationReader


API_INFO = {
    "SYNO.FileStation.Info": {
        "path": "entry.cgi",
        "minVersion": 1,
        "maxVersion": 2,
    },
    "SYNO.FileStation.List": {
        "path": "entry.cgi",
        "minVersion": 1,
        "maxVersion": 2,
    },
    "SYNO.FileStation.Download": {
        "path": "entry.cgi",
        "minVersion": 1,
        "maxVersion": 2,
    },
}


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes | bytearray | object] | None = None,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        read_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {
            "Content-Length": "6",
            "Set-Cookie": "session=must-not-cross-transport-boundary",
        }
        self.chunks = list(chunks or [b"abc", b"", b"def"])
        self.read_error = read_error
        self.close_error = close_error
        self.requested_chunk_size: int | None = None
        self.closed = False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        self.requested_chunk_size = chunk_size
        if self.read_error is not None:
            raise self.read_error
        yield from self.chunks  # type: ignore[misc]

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class FakeSession:
    _debug = False
    _secure = True
    _verify = True

    def __init__(self) -> None:
        self.full_api_list: dict[str, dict[str, object]] = dict(API_INFO)
        self.login_calls = 0
        self.discovery_calls = 0
        self.logout_calls = 0
        self.json_calls: list[tuple[Any, ...]] = []
        self.stream_calls: list[tuple[Any, ...]] = []
        self.login_error: Exception | None = None
        self.discovery_error: Exception | None = None
        self.json_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.logout_error: Exception | None = None
        self.json_response: object = {"success": True, "data": {"hostname": "nas"}}
        self.stream_response = FakeResponse()

    def login(self) -> None:
        self.login_calls += 1
        if self.login_error is not None:
            raise self.login_error

    def get_api_list(self, app: str | None = None) -> None:
        assert app is None
        self.discovery_calls += 1
        if self.discovery_error is not None:
            raise self.discovery_error

    def request_data(
        self,
        api_name: str,
        api_path: str,
        req_param: dict[str, object],
        method: str | None = None,
        data: object | None = None,
        response_json: bool = True,
    ) -> object:
        self.json_calls.append((api_name, api_path, dict(req_param), method, data, response_json))
        req_param["_sid"] = "mutated-copy-only"
        if self.json_error is not None:
            raise self.json_error
        return self.json_response

    def request_stream(
        self,
        api_name: str,
        api_path: str,
        req_param: dict[str, object],
        method: str = "get",
        data: object | None = None,
        headers: dict[str, object] | None = None,
        timeout: float | tuple[float, float] | None = None,
    ) -> FakeResponse:
        self.stream_calls.append(
            (
                api_name,
                api_path,
                dict(req_param),
                method,
                data,
                None if headers is None else dict(headers),
                timeout,
            )
        )
        req_param["_sid"] = "mutated-copy-only"
        if self.stream_error is not None:
            raise self.stream_error
        return self.stream_response

    def logout(self) -> None:
        self.logout_calls += 1
        if self.logout_error is not None:
            raise self.logout_error


async def _connected_transport(
    session: FakeSession | None = None,
) -> tuple[SynologyApiTransportV091, FakeSession]:
    resolved = session or FakeSession()
    transport = SynologyApiTransportV091(resolved)
    await transport.login()
    await transport.query_api_info(tuple(API_INFO))
    return transport, resolved


@pytest.mark.asyncio
async def test_public_fork_transport_satisfies_reader_contract_end_to_end() -> None:
    session = FakeSession()
    transport = SynologyApiTransportV091(session)
    reader = SynologyFileStationReader(
        transport=transport,
        roots={"document_data_1": {"literature": "/documents/literature"}},
    )

    await reader.connect()
    info = await reader.get_info()

    assert info == {"hostname": "nas"}
    assert reader.connected is True
    await reader.close()
    assert session.logout_calls == 1
    assert reader.closed is True


@pytest.mark.asyncio
async def test_transport_reuses_session_and_copies_json_parameters() -> None:
    transport, session = await _connected_transport()
    params: dict[str, object] = {"version": 2, "method": "get"}

    response = await transport.request_json(
        api_name="SYNO.FileStation.Info",
        api_path="entry.cgi",
        params=params,
    )

    assert response == {"success": True, "data": {"hostname": "nas"}}
    assert params == {"version": 2, "method": "get"}
    assert session.login_calls == session.discovery_calls == 1
    assert session.json_calls == [
        (
            "SYNO.FileStation.Info",
            "entry.cgi",
            {"version": 2, "method": "get"},
            "get",
            None,
            True,
        )
    ]
    assert "secret" not in repr(transport).casefold()
    await transport.login()
    assert session.login_calls == 1


@pytest.mark.asyncio
async def test_stream_is_unbuffered_async_and_owned_by_caller() -> None:
    transport, session = await _connected_transport()
    params: dict[str, object] = {"version": 2, "method": "download", "path": "/x.pdf"}
    session.stream_response = FakeResponse(
        status_code=206,
        headers={
            "Content-Range": "bytes 3-8/9",
            "Content-Length": "6",
            "Set-Cookie": "session=must-not-cross-transport-boundary",
        },
    )

    response = await transport.open_stream(
        api_name="SYNO.FileStation.Download",
        api_path="entry.cgi",
        params=params,
        headers={"Range": "bytes=3-"},
    )
    chunks = [chunk async for chunk in response.iter_bytes(4)]

    assert chunks == [b"abc", b"def"]
    assert response.headers == {
        "Content-Range": "bytes 3-8/9",
        "Content-Length": "6",
    }
    assert session.stream_response.requested_chunk_size == 4
    assert session.stream_response.closed is False
    assert params == {"version": 2, "method": "download", "path": "/x.pdf"}
    assert session.stream_calls[0][-2:] == ({"Range": "bytes=3-"}, (10.0, 120.0))
    await response.close()
    await response.close()
    assert session.stream_response.closed is True


@pytest.mark.asyncio
async def test_transport_fails_closed_when_provider_ignores_resume_range() -> None:
    transport, session = await _connected_transport()
    session.stream_response = FakeResponse(
        chunks=[b"full-body-must-not-be-read"],
        status_code=200,
        headers={"Content-Length": "26"},
    )

    with pytest.raises(SynologyTransportError) as captured:
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "download", "path": "/x.pdf"},
            headers={"Range": "bytes=3-"},
        )

    assert captured.value.code == "range_response_not_partial"
    assert str(captured.value) == "range_response_not_partial"
    assert session.stream_response.requested_chunk_size is None
    assert session.stream_response.closed is True
    assert len(session.stream_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code"),
    [
        (FakeResponse(status_code=206, headers={}), "content_range_invalid"),
        (
            FakeResponse(status_code=206, headers={"Content-Range": "bytes 4-8/9"}),
            "content_range_start_mismatch",
        ),
        (
            FakeResponse(status_code=206, headers={"Content-Range": "bytes 3-9/9"}),
            "content_range_bounds_invalid",
        ),
        (
            FakeResponse(
                status_code=206,
                headers={"Content-Range": "bytes 3-8/9", "Content-Length": "5"},
            ),
            "content_length_mismatch",
        ),
        (FakeResponse(status_code=416), "range_not_satisfiable"),
    ],
)
async def test_transport_rejects_invalid_resume_responses_before_body_read(
    response: FakeResponse,
    code: str,
) -> None:
    session = FakeSession()
    session.stream_response = response
    transport, _ = await _connected_transport(session)

    with pytest.raises(SynologyTransportError) as captured:
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "download", "path": "/x.pdf"},
            headers={"Range": "bytes=3-"},
        )

    assert captured.value.code == code
    assert response.requested_chunk_size is None
    assert response.closed is True


@pytest.mark.asyncio
async def test_transport_rejects_invalid_range_request_before_provider_call() -> None:
    transport, session = await _connected_transport()

    with pytest.raises(SynologyTransportError) as captured:
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "download", "path": "/x.pdf"},
            headers={"Range": "bytes=3-4"},
        )

    assert captured.value.code == "range_request_invalid"
    assert session.stream_calls == []


@pytest.mark.asyncio
async def test_transport_rejects_non_allowlisted_operations_and_paths() -> None:
    transport, session = await _connected_transport()

    with pytest.raises(ValueError, match="non-approved JSON"):
        await transport.request_json(
            api_name="SYNO.FileStation.Delete",
            api_path="entry.cgi",
            params={"version": 2, "method": "delete"},
        )
    with pytest.raises(ValueError, match="negotiated"):
        await transport.request_json(
            api_name="SYNO.FileStation.Info",
            api_path="admin.cgi",
            params={"version": 2, "method": "get"},
        )
    with pytest.raises(ValueError, match="non-approved stream"):
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "delete"},
            headers=None,
        )
    with pytest.raises(ValueError, match="headers"):
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "download"},
            headers={"Authorization": "must-not-cross"},
        )

    assert session.json_calls == []
    assert session.stream_calls == []


@pytest.mark.asyncio
async def test_transport_errors_are_stable_and_do_not_echo_provider_details() -> None:
    session = FakeSession()
    session.json_error = RuntimeError("password=provider-secret")
    transport, _ = await _connected_transport(session)

    with pytest.raises(SynologyTransportError) as captured:
        await transport.request_json(
            api_name="SYNO.FileStation.Info",
            api_path="entry.cgi",
            params={"version": 2, "method": "get"},
        )

    assert captured.value.code == "json_request_failed"
    assert "provider-secret" not in str(captured.value)
    assert captured.value.__context__ is None

    session.json_error = None
    session.json_response = ["not", "an", "object"]
    with pytest.raises(SynologyTransportError, match="json_response_invalid"):
        await transport.request_json(
            api_name="SYNO.FileStation.Info",
            api_path="entry.cgi",
            params={"version": 2, "method": "get"},
        )


@pytest.mark.asyncio
async def test_login_failure_logs_out_and_destroys_session_reference() -> None:
    session = FakeSession()
    session.login_error = RuntimeError("secret endpoint")
    transport = SynologyApiTransportV091(session)

    with pytest.raises(SynologyTransportError, match="login_failed"):
        await transport.login()

    assert session.logout_calls == 1
    with pytest.raises(SynologyTransportError, match="transport_closed"):
        await transport.login()

    session = FakeSession()
    session.login_error = RuntimeError("secret login")
    session.logout_error = RuntimeError("secret cleanup")
    transport = SynologyApiTransportV091(session)
    with pytest.raises(SynologyTransportError, match="login_failed"):
        await transport.login()


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_debug", True, "debug"),
        ("_secure", False, "HTTPS"),
        ("_verify", False, "verify TLS"),
    ],
)
def test_transport_rejects_unsafe_session_configuration(
    attribute: str,
    value: bool,
    message: str,
) -> None:
    session = FakeSession()
    setattr(session, attribute, value)

    with pytest.raises(ValueError, match=message):
        SynologyApiTransportV091(session)

    session = FakeSession()
    session._verify = False
    SynologyApiTransportV091(session, require_tls_verification=False)


@pytest.mark.asyncio
async def test_discovery_and_response_contract_fail_closed() -> None:
    session = FakeSession()
    transport = SynologyApiTransportV091(session)
    await transport.login()

    with pytest.raises(ValueError, match="non-approved"):
        await transport.query_api_info(("SYNO.FileStation.Delete",))

    session.discovery_error = RuntimeError("secret discovery")
    with pytest.raises(SynologyTransportError, match="api_discovery_failed"):
        await transport.query_api_info(tuple(API_INFO))

    session.discovery_error = None
    session.full_api_list = []  # type: ignore[assignment]
    with pytest.raises(SynologyTransportError, match="invalid_api_discovery"):
        await transport.query_api_info(tuple(API_INFO))

    session.full_api_list = {
        "SYNO.FileStation.Info": dict(API_INFO["SYNO.FileStation.Info"]),
        "SYNO.FileStation.List": {"path": 123},
    }
    approved = await transport.query_api_info(tuple(API_INFO))
    assert set(approved) == {"SYNO.FileStation.Info", "SYNO.FileStation.List"}


@pytest.mark.asyncio
async def test_requests_require_login_and_stream_open_failure_is_sanitized() -> None:
    session = FakeSession()
    transport = SynologyApiTransportV091(session)
    with pytest.raises(SynologyTransportError, match="transport_not_logged_in"):
        await transport.request_json(
            api_name="SYNO.FileStation.Info",
            api_path="entry.cgi",
            params={"version": 2, "method": "get"},
        )

    await transport.login()
    await transport.query_api_info(tuple(API_INFO))
    session.stream_error = RuntimeError("secret stream endpoint")
    with pytest.raises(SynologyTransportError) as captured:
        await transport.open_stream(
            api_name="SYNO.FileStation.Download",
            api_path="entry.cgi",
            params={"version": 2, "method": "download"},
            headers=None,
        )
    assert captured.value.code == "stream_open_failed"
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_stream_contract_and_close_failures_are_stable() -> None:
    invalid = FakeResponse(chunks=[object()])
    stream = RequestsStreamAdapter(invalid)
    with pytest.raises(SynologyTransportError, match="stream_returned_non_bytes"):
        _ = [chunk async for chunk in stream.iter_bytes(1)]

    failed = FakeResponse(read_error=RuntimeError("secret body"))
    stream = RequestsStreamAdapter(failed)
    with pytest.raises(SynologyTransportError, match="stream_read_failed"):
        _ = [chunk async for chunk in stream.iter_bytes(1)]

    close_failed = FakeResponse(close_error=RuntimeError("secret close"))
    stream = RequestsStreamAdapter(close_failed)
    with pytest.raises(SynologyTransportError, match="stream_close_failed"):
        await stream.close()
    await stream.close()

    stream = RequestsStreamAdapter(FakeResponse())
    with pytest.raises(ValueError, match="positive"):
        _ = [chunk async for chunk in stream.iter_bytes(0)]
    stream._iterating = True
    with pytest.raises(SynologyTransportError, match="already_consumed"):
        _ = [chunk async for chunk in stream.iter_bytes(1)]
    stream._iterating = False
    await stream.close()
    with pytest.raises(SynologyTransportError, match="stream_closed"):
        _ = [chunk async for chunk in stream.iter_bytes(1)]


@pytest.mark.asyncio
async def test_logout_clears_state_even_when_provider_logout_fails() -> None:
    session = FakeSession()
    session.logout_error = RuntimeError("secret logout")
    transport, _ = await _connected_transport(session)

    with pytest.raises(SynologyTransportError, match="logout_failed"):
        await transport.logout()

    with pytest.raises(SynologyTransportError, match="transport_closed"):
        await transport.request_json(
            api_name="SYNO.FileStation.Info",
            api_path="entry.cgi",
            params={"version": 2, "method": "get"},
        )

    await transport.logout()


def test_timeout_and_stream_constructor_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        SynologyApiTransportV091(FakeSession(), request_timeout=(0.0, 1.0))

    invalid_status = FakeResponse()
    invalid_status.status_code = True  # type: ignore[assignment]
    with pytest.raises(SynologyTransportError, match="invalid_stream_status"):
        RequestsStreamAdapter(invalid_status)

    invalid_headers = FakeResponse()
    invalid_headers.headers = {"X": object()}  # type: ignore[dict-item]
    with pytest.raises(SynologyTransportError, match="invalid_stream_headers"):
        RequestsStreamAdapter(invalid_headers)

    invalid_headers.headers = []  # type: ignore[assignment]
    with pytest.raises(SynologyTransportError, match="invalid_stream_headers"):
        RequestsStreamAdapter(invalid_headers)
