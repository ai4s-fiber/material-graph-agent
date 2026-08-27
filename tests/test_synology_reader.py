from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest

from material_graph.knowledge.connectors.synology import (
    ALLOWED_API_METHODS,
    SynologyApiError,
    SynologyFileStationReader,
    SynologyProtocolError,
    SynologyRangeError,
)
from material_graph.knowledge.remote_reader import DirectoryCursor


API_INFO = {
    "SYNO.FileStation.Info": {"path": "entry.cgi", "minVersion": 1, "maxVersion": 2},
    "SYNO.FileStation.List": {"path": "entry.cgi", "minVersion": 1, "maxVersion": 2},
    "SYNO.FileStation.Download": {"path": "entry.cgi", "minVersion": 1, "maxVersion": 2},
}

ROOTS = {
    "document_data_1": {
        "literature": "/文档数据1/信智学院文献数据",
        "patents": "/文档数据1/专利数据",
    }
}


class FakeStreamResponse:
    def __init__(
        self,
        status_code: int,
        chunks: list[Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._chunks = chunks or []
        self.closed = False
        self.iterated = False

    def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size > 0

        async def iterator() -> AsyncIterator[bytes]:
            self.iterated = True
            for chunk in self._chunks:
                yield chunk

        return iterator()

    async def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self) -> None:
        self.login_calls = 0
        self.info_queries: list[tuple[str, ...]] = []
        self.json_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.logout_calls = 0
        self.clear_calls = 0
        self.logout_error: Exception | None = None
        self.clear_error: Exception | None = None
        self.info_response: Any = API_INFO
        self.json_responses: list[Any] = []
        self.stream_responses: list[FakeStreamResponse] = []

    async def login(self) -> None:
        self.login_calls += 1

    async def query_api_info(
        self,
        api_names: tuple[str, ...],
    ) -> Mapping[str, Mapping[str, object]]:
        self.info_queries.append(api_names)
        return self.info_response

    async def request_json(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.json_calls.append({"api_name": api_name, "api_path": api_path, "params": dict(params)})
        if not self.json_responses:
            raise AssertionError("unexpected JSON request")
        return self.json_responses.pop(0)

    async def open_stream(
        self,
        *,
        api_name: str,
        api_path: str,
        params: Mapping[str, object],
        headers: Mapping[str, str] | None,
    ) -> FakeStreamResponse:
        self.stream_calls.append(
            {
                "api_name": api_name,
                "api_path": api_path,
                "params": dict(params),
                "headers": dict(headers or {}),
            }
        )
        if not self.stream_responses:
            raise AssertionError("unexpected stream request")
        return self.stream_responses.pop(0)

    async def logout(self) -> None:
        self.logout_calls += 1
        if self.logout_error is not None:
            raise self.logout_error

    def clear_local_state(self) -> None:
        self.clear_calls += 1
        if self.clear_error is not None:
            raise self.clear_error


def make_reader(transport: FakeTransport) -> SynologyFileStationReader:
    return SynologyFileStationReader(transport=transport, roots=ROOTS)


async def connected_reader(transport: FakeTransport) -> SynologyFileStationReader:
    reader = make_reader(transport)
    await reader.connect()
    return reader


def test_business_surface_has_no_generic_or_mutating_operations() -> None:
    reader = make_reader(FakeTransport())

    for name in (
        "request",
        "request_json",
        "upload",
        "delete",
        "rename",
        "copy",
        "move",
        "create_folder",
        "search_start",
    ):
        assert not hasattr(reader, name)

    assert not any("Upload" in api or "Delete" in api for api, _ in ALLOWED_API_METHODS)


@pytest.mark.asyncio
async def test_connect_queries_only_required_filestation_apis() -> None:
    transport = FakeTransport()
    reader = await connected_reader(transport)

    assert transport.login_calls == 1
    assert transport.info_queries == [tuple(API_INFO)]
    assert reader.connected is True


@pytest.mark.asyncio
async def test_get_info_uses_negotiated_read_only_api() -> None:
    transport = FakeTransport()
    transport.json_responses = [{"success": True, "data": {"hostname": "nas"}}]
    reader = await connected_reader(transport)

    assert await reader.get_info() == {"hostname": "nas"}
    assert transport.json_calls == [
        {
            "api_name": "SYNO.FileStation.Info",
            "api_path": "entry.cgi",
            "params": {"version": 2, "method": "get"},
        }
    ]


@pytest.mark.asyncio
async def test_unknown_root_slice_and_escape_fail_before_transport_calls() -> None:
    transport = FakeTransport()
    reader = await connected_reader(transport)

    with pytest.raises(KeyError, match="unknown root"):
        await reader.stat("missing", "literature", "paper.pdf")
    with pytest.raises(KeyError, match="unknown slice"):
        await reader.stat("document_data_1", "missing", "paper.pdf")
    with pytest.raises(ValueError):
        await reader.stat("document_data_1", "literature", "../private.pdf")

    assert transport.json_calls == []


@pytest.mark.asyncio
async def test_directory_pagination_uses_response_offset_plus_item_count() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "offset": 7,
                "total": 10,
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/a.pdf",
                        "name": "a.pdf",
                        "isdir": False,
                        "additional": {"size": 11, "time": {"mtime": 101}},
                    },
                    {
                        "path": "/文档数据1/信智学院文献数据/sub",
                        "name": "sub",
                        "isdir": True,
                        "additional": {"time": {"mtime": 102}},
                    },
                ],
            },
        },
        {
            "success": True,
            "data": {
                "offset": 9,
                "total": 10,
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/z.pdf",
                        "name": "z.pdf",
                        "isdir": False,
                        "additional": {"size": 12, "time": {"mtime": 103}},
                    }
                ],
            },
        },
    ]
    reader = await connected_reader(transport)
    cursor = DirectoryCursor(
        root_id="document_data_1",
        slice_id="literature",
        relative_directory=".",
        offset=100,
    )

    entries = [
        entry
        async for entry in reader.iter_entries(
            "document_data_1", "literature", cursor=cursor, page_size=2
        )
    ]

    assert [entry.relative_path for entry in entries] == ["a.pdf", "sub", "z.pdf"]
    assert [call["params"]["offset"] for call in transport.json_calls] == [100, 9]
    assert reader.last_cursor is not None
    assert reader.last_cursor.offset == 10
    assert reader.last_cursor.to_checkpoint().keys() == {
        "schema_version",
        "root_id",
        "slice_id",
        "relative_directory",
        "offset",
    }


@pytest.mark.asyncio
async def test_pagination_rejects_entries_outside_approved_slice() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "offset": 0,
                "total": 1,
                "files": [
                    {
                        "path": "/文档数据1/其他目录/private.pdf",
                        "name": "private.pdf",
                        "isdir": False,
                    }
                ],
            },
        }
    ]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match="outside approved slice"):
        _ = [entry async for entry in reader.iter_entries("document_data_1", "literature")]


@pytest.mark.asyncio
async def test_stat_maps_only_approved_remote_path() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/polymer/a.pdf",
                        "name": "a.pdf",
                        "isdir": False,
                        "additional": {"size": 99, "time": {"mtime": 1234}},
                    }
                ]
            },
        }
    ]
    reader = await connected_reader(transport)

    stat = await reader.stat("document_data_1", "literature", "polymer/a.pdf")

    assert stat.relative_path == "polymer/a.pdf"
    assert stat.byte_size == 99
    assert stat.modified_at == 1234
    assert transport.json_calls[0]["params"]["path"] == [
        "/文档数据1/信智学院文献数据/polymer/a.pdf"
    ]


@pytest.mark.asyncio
async def test_valid_range_response_streams_only_matching_206() -> None:
    transport = FakeTransport()
    response = FakeStreamResponse(
        206,
        [b"def", b"ghi"],
        {
            "Content-Range": "bytes 3-8/9",
            "Content-Type": "application/octet-stream",
        },
    )
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    chunks = [
        chunk
        async for chunk in reader.open_stream(
            "document_data_1",
            "literature",
            "paper.pdf",
            offset=3,
            expected_size=9,
        )
    ]

    assert chunks == [b"def", b"ghi"]
    assert transport.stream_calls[0]["headers"] == {"Range": "bytes=3-"}
    assert reader.last_effective_stream_offset == 3
    assert response.closed is True


@pytest.mark.asyncio
async def test_expected_mtime_is_rechecked_before_opening_source_body() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/paper.pdf",
                        "name": "paper.pdf",
                        "isdir": False,
                        "additional": {"size": 9, "time": {"mtime": 102}},
                    }
                ]
            },
        }
    ]
    transport.stream_responses = [FakeStreamResponse(200, [b"must-not-open"])]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match="changed before streaming"):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1",
                "literature",
                "paper.pdf",
                expected_size=9,
                expected_mtime=101,
            )
        ]

    assert transport.stream_calls == []


@pytest.mark.asyncio
async def test_matching_remote_version_can_open_after_preflight() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/paper.pdf",
                        "name": "paper.pdf",
                        "isdir": False,
                        "additional": {"size": 1, "time": {"mtime": 101}},
                    }
                ]
            },
        }
    ]
    response = FakeStreamResponse(200, [b"x"], {"Content-Length": "1"})
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    chunks = [
        chunk
        async for chunk in reader.open_stream(
            "document_data_1",
            "literature",
            "paper.pdf",
            expected_size=1,
            expected_mtime=101,
        )
    ]

    assert chunks == [b"x"]
    assert response.closed is True


@pytest.mark.asyncio
async def test_mismatched_content_range_is_never_iterated() -> None:
    transport = FakeTransport()
    response = FakeStreamResponse(
        206,
        [b"must-not-yield"],
        {"Content-Range": "bytes 4-8/9"},
    )
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyRangeError, match="content_range_start_mismatch"):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1", "literature", "paper.pdf", offset=3
            )
        ]

    assert response.iterated is False


@pytest.mark.asyncio
async def test_api_error_envelope_is_not_treated_as_data() -> None:
    transport = FakeTransport()
    transport.json_responses = [{"success": False, "error": {"code": 105}}]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyApiError, match="105"):
        await reader.get_info()


@pytest.mark.asyncio
async def test_logout_failure_still_clears_and_detaches_transport() -> None:
    transport = FakeTransport()
    transport.logout_error = RuntimeError("logout unavailable")
    reader = await connected_reader(transport)

    with pytest.raises(RuntimeError, match="logout unavailable"):
        await reader.close()

    assert transport.logout_calls == 1
    assert transport.clear_calls == 1
    assert reader.connected is False
    assert reader.closed is True
    assert reader.transport_attached is False


@pytest.mark.asyncio
async def test_connect_failure_cleans_local_transport_state() -> None:
    transport = FakeTransport()
    transport.info_response = {}
    reader = make_reader(transport)

    with pytest.raises(SynologyProtocolError, match="missing API information"):
        await reader.connect()

    assert transport.logout_calls == 1
    assert transport.clear_calls == 1
    assert reader.transport_attached is False


@pytest.mark.asyncio
async def test_range_ignored_with_200_fails_closed_without_full_rescan() -> None:
    transport = FakeTransport()
    ignored = FakeStreamResponse(200, [b"ignored-full-body"])
    transport.stream_responses = [ignored]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyRangeError) as captured:
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1",
                "literature",
                "paper.pdf",
                offset=3,
                expected_size=6,
            )
        ]

    assert captured.value.code == "range_response_not_partial"
    assert transport.stream_calls[0]["headers"] == {"Range": "bytes=3-"}
    assert len(transport.stream_calls) == 1
    assert ignored.iterated is False
    assert ignored.closed is True
    assert reader.last_effective_stream_offset is None


@pytest.mark.asyncio
async def test_416_is_strict_even_when_offset_equals_expected_size() -> None:
    transport = FakeTransport()
    response = FakeStreamResponse(416, headers={"Content-Range": "bytes */9"})
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyRangeError, match="range_not_satisfiable"):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1",
                "literature",
                "paper.pdf",
                offset=9,
                expected_size=9,
            )
        ]

    assert response.closed is True


@pytest.mark.asyncio
async def test_json_download_response_is_rejected_before_yielding() -> None:
    transport = FakeTransport()
    response = FakeStreamResponse(
        200,
        [b'{"success":false}'],
        {"Content-Type": "application/json"},
    )
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match="JSON response"):
        _ = [
            chunk
            async for chunk in reader.open_stream("document_data_1", "literature", "paper.pdf")
        ]

    assert response.iterated is False


@pytest.mark.parametrize(
    "roots",
    [
        {},
        {"document_data_1": {}},
        {"document_data_1": {"literature": "/"}},
        {"document_data_1": {"literature": "relative/path"}},
        {"document_data_1": {"literature": "/bad?query"}},
        {"document_data_1": {"literature": "/bad//child"}},
        {"document_data_1": {"literature": "/bad/.."}},
        {"document_data_1": {"literature": None}},
    ],
)
def test_reader_rejects_unsafe_or_empty_root_configuration(roots: Any) -> None:
    with pytest.raises(ValueError):
        SynologyFileStationReader(transport=FakeTransport(), roots=roots)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement, message",
    [
        (
            {"path": "/absolute.cgi", "minVersion": 1, "maxVersion": 2},
            "invalid API path",
        ),
        (
            {"path": "entry.cgi?api=other", "minVersion": 1, "maxVersion": 2},
            "invalid API path",
        ),
        (
            {"path": "entry.cgi", "minVersion": True, "maxVersion": 2},
            "invalid API version range",
        ),
        (
            {"path": "entry.cgi", "minVersion": 3, "maxVersion": 4},
            "unsupported API version",
        ),
    ],
)
async def test_connect_rejects_unsafe_or_unsupported_api_info(
    replacement: Mapping[str, object],
    message: str,
) -> None:
    transport = FakeTransport()
    transport.info_response = {name: dict(value) for name, value in API_INFO.items()}
    transport.info_response["SYNO.FileStation.Info"] = replacement
    reader = make_reader(transport)

    with pytest.raises(SynologyProtocolError, match=message):
        await reader.connect()

    assert reader.closed is True
    assert reader.transport_attached is False


@pytest.mark.asyncio
async def test_connect_rejects_non_mapping_discovery_and_preserves_primary_error() -> None:
    transport = FakeTransport()
    transport.info_response = []
    transport.logout_error = RuntimeError("cleanup logout failed")
    transport.clear_error = RuntimeError("cleanup clear failed")
    reader = make_reader(transport)

    with pytest.raises(SynologyProtocolError, match="missing API information"):
        await reader.connect()

    assert transport.logout_calls == 1
    assert transport.clear_calls == 1
    assert reader.transport_attached is False


@pytest.mark.asyncio
async def test_connect_is_idempotent_and_closed_reader_cannot_reconnect() -> None:
    transport = FakeTransport()
    reader = await connected_reader(transport)

    assert await reader.connect() is reader
    assert transport.login_calls == 1
    await reader.close()
    await reader.close()

    with pytest.raises(RuntimeError, match="closed"):
        await reader.connect()
    assert transport.clear_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "non-object JSON response"),
        ({"success": "yes", "data": {}}, "invalid JSON envelope"),
        ({"success": True, "data": []}, "missing object data"),
    ],
)
async def test_json_protocol_envelopes_fail_closed(payload: Any, message: str) -> None:
    transport = FakeTransport()
    transport.json_responses = [payload]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match=message):
        await reader.get_info()


@pytest.mark.asyncio
async def test_api_error_without_error_object_is_sanitized() -> None:
    transport = FakeTransport()
    transport.json_responses = [{"success": False, "error": "redacted"}]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyApiError, match="unknown"):
        await reader.get_info()


@pytest.mark.asyncio
async def test_reader_operations_require_connection() -> None:
    reader = make_reader(FakeTransport())

    with pytest.raises(RuntimeError, match="not connected"):
        await reader.stat("document_data_1", "literature", "paper.pdf")


@pytest.mark.asyncio
async def test_private_dispatch_guard_rejects_unapproved_or_unnegotiated_api() -> None:
    reader = await connected_reader(FakeTransport())

    with pytest.raises(SynologyProtocolError, match="non-approved"):
        reader._api_spec("SYNO.FileStation.Delete", "delete")
    reader._api_info.pop("SYNO.FileStation.Info")
    with pytest.raises(SynologyProtocolError, match="not negotiated"):
        reader._api_spec("SYNO.FileStation.Info", "get")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw, message",
    [
        ("not-an-object", "non-object entry"),
        (
            {
                "path": "/文档数据1/信智学院文献数据",
                "name": "信智学院文献数据",
                "isdir": True,
            },
            "slice root",
        ),
        (
            {
                "path": "/文档数据1/信智学院文献数据/a.pdf",
                "name": "spoofed.pdf",
                "isdir": False,
            },
            "name does not match",
        ),
        (
            {
                "path": "/文档数据1/信智学院文献数据/a.pdf",
                "name": "a.pdf",
                "isdir": 0,
            },
            "directory flag",
        ),
        (
            {
                "path": "/文档数据1/信智学院文献数据/a.pdf",
                "name": "a.pdf",
                "isdir": False,
                "additional": [],
            },
            "additional metadata",
        ),
        (
            {
                "path": "/文档数据1/信智学院文献数据/a.pdf",
                "name": "a.pdf",
                "isdir": False,
                "additional": {"size": -1},
            },
            "File Station size",
        ),
        (
            {
                "path": "/文档数据1/信智学院文献数据/a.pdf",
                "name": "a.pdf",
                "isdir": False,
                "additional": {"size": 1, "time": []},
            },
            "time metadata",
        ),
    ],
)
async def test_directory_entries_reject_malformed_remote_metadata(
    raw: Any,
    message: str,
) -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {"offset": 0, "total": 1, "files": [raw]},
        }
    ]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match=message):
        _ = [entry async for entry in reader.iter_entries("document_data_1", "literature")]


@pytest.mark.asyncio
async def test_directory_entry_allows_absent_optional_metadata() -> None:
    transport = FakeTransport()
    transport.json_responses = [
        {
            "success": True,
            "data": {
                "offset": 0,
                "total": 1,
                "files": [
                    {
                        "path": "/文档数据1/信智学院文献数据/a.pdf",
                        "name": "a.pdf",
                        "isdir": False,
                    }
                ],
            },
        }
    ]
    reader = await connected_reader(transport)

    entries = [entry async for entry in reader.iter_entries("document_data_1", "literature")]

    assert entries[0].byte_size is None
    assert entries[0].modified_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data, message",
    [
        ({"total": 1, "files": []}, "missing directory response offset"),
        ({"offset": 0, "total": 1, "files": "bad"}, "invalid files"),
        ({"offset": 0, "total": 1, "files": []}, "made no progress"),
    ],
)
async def test_directory_pagination_rejects_invalid_or_stalled_pages(
    data: Mapping[str, object],
    message: str,
) -> None:
    transport = FakeTransport()
    transport.json_responses = [{"success": True, "data": data}]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match=message):
        _ = [entry async for entry in reader.iter_entries("document_data_1", "literature")]


@pytest.mark.asyncio
async def test_directory_cursor_and_page_size_are_validated() -> None:
    reader = await connected_reader(FakeTransport())

    with pytest.raises(TypeError, match="DirectoryCursor"):
        _ = [
            entry
            async for entry in reader.iter_entries(
                "document_data_1",
                "literature",
                cursor="bad",  # type: ignore[arg-type]
            )
        ]
    wrong = DirectoryCursor(root_id="document_data_1", slice_id="patents")
    with pytest.raises(ValueError, match="does not belong"):
        _ = [
            entry
            async for entry in reader.iter_entries("document_data_1", "literature", cursor=wrong)
        ]
    with pytest.raises(ValueError, match="page_size"):
        _ = [
            entry
            async for entry in reader.iter_entries("document_data_1", "literature", page_size=0)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files, message",
    [
        ([], "exactly one"),
        (
            [
                {
                    "path": "/文档数据1/信智学院文献数据/b.pdf",
                    "isdir": False,
                }
            ],
            "different approved path",
        ),
        (
            [
                {
                    "path": "/文档数据1/信智学院文献数据/a.pdf",
                    "isdir": 1,
                }
            ],
            "directory flag",
        ),
    ],
)
async def test_stat_rejects_ambiguous_or_mismatched_results(
    files: list[object],
    message: str,
) -> None:
    transport = FakeTransport()
    transport.json_responses = [{"success": True, "data": {"files": files}}]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match=message):
        await reader.stat("document_data_1", "literature", "a.pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected_size, error_type, message",
    [
        (
            FakeStreamResponse(206, [b"x"]),
            None,
            SynologyRangeError,
            "content_range_invalid",
        ),
        (
            FakeStreamResponse(206, [b"x"], {"Content-Range": "bytes 3-9/9"}),
            None,
            SynologyRangeError,
            "content_range_bounds_invalid",
        ),
        (
            FakeStreamResponse(206, [b"x"], {"Content-Range": "bytes 3-9/10"}),
            9,
            SynologyRangeError,
            "content_range_total_mismatch",
        ),
        (
            FakeStreamResponse(503, [b"x"]),
            None,
            SynologyRangeError,
            "range_response_not_partial",
        ),
    ],
)
async def test_range_download_rejects_invalid_headers_and_statuses(
    response: FakeStreamResponse,
    expected_size: int | None,
    error_type: type[Exception],
    message: str,
) -> None:
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(error_type, match=message):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1",
                "literature",
                "paper.pdf",
                offset=3,
                expected_size=expected_size,
            )
        ]

    assert response.closed is True
    assert response.iterated is False


@pytest.mark.asyncio
async def test_range_download_validates_remaining_content_length() -> None:
    response = FakeStreamResponse(
        206,
        [b"def"],
        {"Content-Range": "bytes 3-8/9", "Content-Length": "5"},
    )
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyRangeError, match="content_length_mismatch"):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1", "literature", "paper.pdf", offset=3
            )
        ]

    assert response.iterated is False
    assert response.closed is True


@pytest.mark.asyncio
async def test_download_rejects_non_string_http_headers() -> None:
    response = FakeStreamResponse(200, [b"x"])
    response.headers = {1: "invalid"}  # type: ignore[dict-item]
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match="invalid HTTP headers"):
        _ = [
            chunk
            async for chunk in reader.open_stream("document_data_1", "literature", "paper.pdf")
        ]

    assert response.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, error_type, message",
    [
        (
            FakeStreamResponse(416),
            SynologyRangeError,
            "range_not_satisfiable",
        ),
        (
            FakeStreamResponse(503),
            SynologyProtocolError,
            "unexpected download status",
        ),
        (
            FakeStreamResponse(200, [b"x"], {"Content-Length": "not-an-int"}),
            SynologyProtocolError,
            "invalid Content-Length",
        ),
        (
            FakeStreamResponse(200, [b"x"], {"Content-Length": "2"}),
            SynologyProtocolError,
            "does not match expected size",
        ),
    ],
)
async def test_full_download_rejects_invalid_status_or_length(
    response: FakeStreamResponse,
    error_type: type[Exception],
    message: str,
) -> None:
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    kwargs = {"expected_size": 1} if response.status_code == 200 else {}
    with pytest.raises(error_type, match=message):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1", "literature", "paper.pdf", **kwargs
            )
        ]

    assert response.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_size", [None, 1])
async def test_full_download_allows_optional_or_missing_content_length(
    expected_size: int | None,
) -> None:
    response = FakeStreamResponse(200, [b"x"])
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    chunks = [
        chunk
        async for chunk in reader.open_stream(
            "document_data_1",
            "literature",
            "paper.pdf",
            expected_size=expected_size,
        )
    ]

    assert chunks == [b"x"]
    assert response.closed is True


@pytest.mark.asyncio
async def test_download_rejects_non_bytes_chunks_and_still_closes() -> None:
    response = FakeStreamResponse(200, [object()])
    transport = FakeTransport()
    transport.stream_responses = [response]
    reader = await connected_reader(transport)

    with pytest.raises(SynologyProtocolError, match="non-bytes"):
        _ = [
            chunk
            async for chunk in reader.open_stream("document_data_1", "literature", "paper.pdf")
        ]

    assert response.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"offset": -1}, "offset"),
        ({"expected_size": -1}, "expected_size"),
        ({"expected_mtime": -1}, "expected_mtime"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"offset": 2, "expected_size": 1}, "cannot exceed"),
    ],
)
async def test_download_arguments_are_validated_before_transport(
    kwargs: Mapping[str, int],
    message: str,
) -> None:
    transport = FakeTransport()
    reader = await connected_reader(transport)

    with pytest.raises(ValueError, match=message):
        _ = [
            chunk
            async for chunk in reader.open_stream(
                "document_data_1", "literature", "paper.pdf", **kwargs
            )
        ]

    assert transport.stream_calls == []


@pytest.mark.asyncio
async def test_close_clear_failure_still_detaches_transport() -> None:
    transport = FakeTransport()
    transport.clear_error = RuntimeError("clear failed")
    reader = make_reader(transport)

    with pytest.raises(RuntimeError, match="clear failed"):
        await reader.close()

    assert reader.closed is True
    assert reader.transport_attached is False
