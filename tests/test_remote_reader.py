from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from material_graph.knowledge.remote_reader import (
    DirectoryCursor,
    RemoteEntry,
    RemoteRangeContractError,
    RemoteSourceReader,
    RemoteStat,
    parse_open_ended_byte_range,
    validate_open_ended_range_response,
)


class ChunkReader(RemoteSourceReader):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def iter_entries(
        self,
        root_id: str,
        slice_id: str,
        *,
        cursor: DirectoryCursor | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[RemoteEntry]:
        del root_id, slice_id, cursor, page_size

        async def empty() -> AsyncIterator[RemoteEntry]:
            if False:  # pragma: no cover - establishes the async-generator type
                yield RemoteEntry(
                    root_id="root",
                    slice_id="slice",
                    relative_path="unused",
                    name="unused",
                    is_dir=False,
                )

        return empty()

    async def stat(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
    ) -> RemoteStat:
        return RemoteStat(
            root_id=root_id,
            slice_id=slice_id,
            relative_path=relative_path,
            is_dir=False,
        )

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
        del root_id, slice_id, relative_path, offset, expected_size, expected_mtime, chunk_size

        async def chunks() -> AsyncIterator[bytes]:
            for chunk in self._chunks:
                yield chunk

        return chunks()

    async def close(self) -> None:
        self.closed = True


def test_remote_entry_and_stat_normalize_safe_relative_paths() -> None:
    entry = RemoteEntry(
        root_id="document_data_1",
        slice_id="literature",
        relative_path=r"polymer\paper.pdf",
        name="paper.pdf",
        is_dir=False,
        byte_size=42,
        modified_at=123,
    )
    stat = RemoteStat(
        root_id="document_data_1",
        slice_id="literature",
        relative_path=r"polymer\paper.pdf",
        is_dir=False,
        byte_size=42,
        modified_at=123,
    )

    assert entry.relative_path == "polymer/paper.pdf"
    assert stat.relative_path == "polymer/paper.pdf"


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.pdf",
        "/volume1/private.pdf",
        r"C:\private.pdf",
        r"\\nas\share\private.pdf",
        "https://example.invalid/private.pdf",
        "bad\x00name.pdf",
    ],
)
def test_remote_contracts_reject_path_escape(relative_path: str) -> None:
    with pytest.raises(ValueError):
        RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path=relative_path,
            name="paper.pdf",
            is_dir=False,
        )


def test_directory_cursor_checkpoint_contains_only_logical_state() -> None:
    cursor = DirectoryCursor(
        root_id="document_data_1",
        slice_id="literature",
        relative_directory="polymer",
        offset=17,
    )
    payload = cursor.to_checkpoint()

    assert payload == {
        "schema_version": 1,
        "root_id": "document_data_1",
        "slice_id": "literature",
        "relative_directory": "polymer",
        "offset": 17,
    }
    assert not {"endpoint", "session", "token", "device_id"} & payload.keys()
    assert DirectoryCursor.from_checkpoint(payload) == cursor


@pytest.mark.parametrize("secret_field", ["endpoint", "session", "token", "device_id"])
def test_directory_cursor_rejects_secret_or_unknown_checkpoint_fields(secret_field: str) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "root_id": "document_data_1",
        "slice_id": "literature",
        "relative_directory": ".",
        "offset": 0,
        secret_field: "must-not-be-here",
    }
    with pytest.raises(ValueError, match="unsupported cursor fields"):
        DirectoryCursor.from_checkpoint(payload)


def test_directory_cursor_uses_server_offset_when_advancing() -> None:
    cursor = DirectoryCursor(
        root_id="document_data_1",
        slice_id="literature",
        relative_directory=".",
        offset=100,
    )

    advanced = cursor.advance(response_offset=7, item_count=3)

    assert advanced.offset == 10
    assert advanced.root_id == cursor.root_id
    assert advanced.slice_id == cursor.slice_id


@pytest.mark.asyncio
async def test_iter_lines_handles_boundaries_and_final_unterminated_line() -> None:
    reader = ChunkReader([b"first\nsec", b"ond\r\nthird"])
    lines = [
        line
        async for line in reader.iter_lines(
            "document_data_1",
            "literature",
            "manifest.jsonl",
        )
    ]

    assert lines == [b"first\n", b"second\r\n", b"third"]


@pytest.mark.asyncio
async def test_reader_context_manager_always_closes() -> None:
    reader = ChunkReader([])
    async with reader as opened:
        assert opened is reader
    assert reader.closed is True


@pytest.mark.asyncio
async def test_iter_lines_rejects_oversized_line_before_pending_growth() -> None:
    reader = ChunkReader([b"123", b"456", b"\n"])

    with pytest.raises(ValueError, match="exceeds max_line_bytes"):
        _ = [
            line
            async for line in reader.iter_lines(
                "document_data_1",
                "literature",
                "manifest.jsonl",
                max_line_bytes=5,
            )
        ]


@pytest.mark.asyncio
async def test_iter_lines_applies_limit_per_line_not_per_chunk() -> None:
    reader = ChunkReader([b"a\nb\n"])

    lines = [
        line
        async for line in reader.iter_lines(
            "document_data_1",
            "literature",
            "manifest.jsonl",
            max_line_bytes=2,
        )
    ]

    assert lines == [b"a\n", b"b\n"]


@pytest.mark.parametrize("max_line_bytes", [0, -1, True])
@pytest.mark.asyncio
async def test_iter_lines_rejects_invalid_line_limit(max_line_bytes: int) -> None:
    reader = ChunkReader([])

    with pytest.raises(ValueError, match="positive integer"):
        _ = [
            line
            async for line in reader.iter_lines(
                "document_data_1",
                "literature",
                "manifest.jsonl",
                max_line_bytes=max_line_bytes,
            )
        ]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DirectoryCursor(root_id="INVALID", slice_id="literature"),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path=1,  # type: ignore[arg-type]
            name="paper.pdf",
            is_dir=False,
        ),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path=".",
            name="paper.pdf",
            is_dir=False,
        ),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="paper.pdf",
            name="",
            is_dir=False,
        ),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="paper.pdf",
            name="dir/paper.pdf",
            is_dir=False,
        ),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="paper.pdf",
            name="paper.pdf",
            is_dir=1,  # type: ignore[arg-type]
        ),
        lambda: RemoteEntry(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="paper.pdf",
            name="paper.pdf",
            is_dir=False,
            byte_size=-1,
        ),
        lambda: RemoteStat(
            root_id="document_data_1",
            slice_id="literature",
            relative_path="paper.pdf",
            is_dir=1,  # type: ignore[arg-type]
        ),
    ],
)
def test_remote_contract_validation_edges(factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        factory()


def test_directory_cursor_rejects_missing_fields_and_schema_version() -> None:
    checkpoint = DirectoryCursor(
        root_id="document_data_1",
        slice_id="literature",
    ).to_checkpoint()
    missing = dict(checkpoint)
    missing.pop("offset")
    with pytest.raises(ValueError, match="missing cursor fields"):
        DirectoryCursor.from_checkpoint(missing)

    unsupported = dict(checkpoint)
    unsupported["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported cursor schema version"):
        DirectoryCursor.from_checkpoint(unsupported)


@pytest.mark.asyncio
async def test_iter_lines_rejects_non_bytes_chunks() -> None:
    reader = ChunkReader(["not-bytes"])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="non-bytes"):
        _ = [
            line
            async for line in reader.iter_lines(
                "document_data_1",
                "literature",
                "manifest.jsonl",
            )
        ]


@pytest.mark.asyncio
async def test_iter_lines_rejects_oversized_newline_segment() -> None:
    reader = ChunkReader([b"123456\n"])

    with pytest.raises(ValueError, match="exceeds max_line_bytes"):
        _ = [
            line
            async for line in reader.iter_lines(
                "document_data_1",
                "literature",
                "manifest.jsonl",
                max_line_bytes=5,
            )
        ]


def test_open_ended_range_contract_accepts_exact_partial_response() -> None:
    assert parse_open_ended_byte_range("bytes=3-") == 3

    validated = validate_open_ended_range_response(
        requested_offset=3,
        status_code=206,
        headers={"Content-Range": "bytes 3-8/9", "Content-Length": "6"},
        expected_total=9,
    )

    assert (validated.start, validated.end, validated.total, validated.length) == (3, 8, 9, 6)


@pytest.mark.parametrize("value", [None, 3, "bytes=3", "bytes=-3", "items=3-"])
def test_open_ended_range_request_rejects_ambiguous_syntax(value: object) -> None:
    with pytest.raises(RemoteRangeContractError, match="range_request_invalid"):
        parse_open_ended_byte_range(value)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"requested_offset": True}, "range_request_invalid"),
        ({"expected_total": -1}, "range_expected_total_invalid"),
        ({"status_code": True}, "range_response_status_invalid"),
        ({"status_code": 416}, "range_not_satisfiable"),
        ({"status_code": 200}, "range_response_not_partial"),
        ({"headers": {}}, "content_range_invalid"),
        (
            {"headers": {"Content-Range": "bytes 4-8/9"}},
            "content_range_start_mismatch",
        ),
        (
            {"headers": {"Content-Range": "bytes 3-9/9"}},
            "content_range_bounds_invalid",
        ),
        (
            {"headers": {"Content-Range": "bytes 3-9/10"}},
            "content_range_total_mismatch",
        ),
        (
            {
                "headers": {
                    "Content-Range": "bytes 3-8/9",
                    "content-range": "bytes 3-8/9",
                }
            },
            "range_response_headers_invalid",
        ),
        (
            {"headers": {"Content-Range": "bytes 3-8/9", "Content-Length": "six"}},
            "content_length_invalid",
        ),
        (
            {"headers": {"Content-Range": "bytes 3-8/9", "Content-Length": "5"}},
            "content_length_mismatch",
        ),
    ],
)
def test_open_ended_range_response_fails_closed_with_stable_codes(
    overrides: dict[str, object],
    code: str,
) -> None:
    arguments: dict[str, object] = {
        "requested_offset": 3,
        "status_code": 206,
        "headers": {"Content-Range": "bytes 3-8/9"},
        "expected_total": 9,
    }
    arguments.update(overrides)

    with pytest.raises(RemoteRangeContractError) as captured:
        validate_open_ended_range_response(**arguments)  # type: ignore[arg-type]

    assert captured.value.code == code
    assert str(captured.value) == code
