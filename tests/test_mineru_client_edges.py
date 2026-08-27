from __future__ import annotations

import asyncio
import json
import zipfile
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from material_graph.knowledge.mineru_client import MinerUClient, MinerUError, MinerUSettings


async def _no_sleep(_: float) -> None:
    return None


def test_parse_validates_source_and_idempotency_before_network(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"network should not be called: {request.url}")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret")
            with pytest.raises(FileNotFoundError):
                await client.parse(
                    tmp_path / "missing.pdf",
                    file_name="sample.pdf",
                    idempotency_key="key",
                    output_dir=tmp_path / "out",
                )
            source = tmp_path / "source.pdf"
            source.write_bytes(b"pdf")
            with pytest.raises(ValueError, match="idempotency"):
                await client.parse(
                    source,
                    file_name="sample.pdf",
                    idempotency_key=" ",
                    output_dir=tmp_path / "out",
                )

    asyncio.run(run())


def test_upload_slot_rejects_invalid_shapes_and_empty_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    async def run() -> None:
        async with httpx.AsyncClient() as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(language="en"),
            )
            client._api_json = AsyncMock(return_value={})  # type: ignore[method-assign]
            with pytest.raises(MinerUError, match="no data object"):
                await client._request_upload_slot(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="key",
                )

            client._api_json = AsyncMock(  # type: ignore[method-assign]
                return_value={"data": {"batch_id": "", "file_urls": []}}
            )
            with pytest.raises(MinerUError, match="missing batch_id"):
                await client._request_upload_slot(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="key",
                )

            mocked = AsyncMock(
                return_value={
                    "data": {"batch_id": "batch", "file_urls": ["https://upload.invalid"]}
                }
            )
            client._api_json = mocked  # type: ignore[method-assign]
            with pytest.raises(MinerUError, match="empty"):
                await client._request_upload_slot(
                    empty,
                    file_name="sample.pdf",
                    idempotency_key="key",
                )
            payload = mocked.await_args.kwargs["json_body"]
            assert payload["language"] == "en"

    asyncio.run(run())


def test_completed_task_requires_task_id_and_archive_url(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("file-urls/batch"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "batch",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )
        if request.url.host == "upload.example":
            await request.aread()
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"extract_result": [{"file_name": "sample.pdf", "state": "done"}]},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret")
            with pytest.raises(MinerUError, match="task ID"):
                await client.parse(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="key",
                    output_dir=tmp_path / "out",
                )

    asyncio.run(run())


def test_upload_transport_retry_source_change_and_rejections(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    delays: list[float] = []
    calls = 0

    async def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await request.aread()
        if calls == 1:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def run_retry() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(retry_handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret", sleep=sleep)
            await client._upload_source(source, "https://upload.example/signed")

    asyncio.run(run_retry())
    assert calls == 2
    assert delays == [1]

    async def changed_handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        source.write_bytes(b"changed-size")
        return httpx.Response(200)

    async def run_changed() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(changed_handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret")
            with pytest.raises(MinerUError) as captured:
                await client._upload_source(source, "https://upload.example/signed")
            assert captured.value.category == "source_changed"

    asyncio.run(run_changed())

    for status, category in [(503, "upload_rate_or_service"), (400, "upload_rejected")]:

        async def reject_handler(
            request: httpx.Request,
            response_status: int = status,
        ) -> httpx.Response:
            await request.aread()
            return httpx.Response(response_status)

        async def run_rejected(expected: str = category) -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(reject_handler)) as http:
                client = MinerUClient(
                    http,
                    token_provider=lambda: "secret",
                    settings=MinerUSettings(retry_max_attempts=1),
                )
                with pytest.raises(MinerUError) as captured:
                    await client._upload_source(source, "https://upload.example/signed")
                assert captured.value.category == expected

        asyncio.run(run_rejected())


def test_upload_transport_exhaustion_is_retryable(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(retry_max_attempts=1),
            )
            with pytest.raises(MinerUError) as captured:
                await client._upload_source(source, "https://upload.example/signed")
            assert captured.value.category == "upload_transport"
            assert captured.value.retryable is True

    asyncio.run(run())


def test_poll_timeout_and_task_matching_helpers() -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(poll_timeout_seconds=0.000001),
            )
            client._api_json = AsyncMock(return_value={"data": {}})  # type: ignore[method-assign]
            with pytest.raises(MinerUError) as captured:
                await client._poll_batch("batch", "sample.pdf")
            assert captured.value.category == "timeout"

    asyncio.run(run())
    assert MinerUClient._match_task(None, "sample.pdf") is None
    assert MinerUClient._match_task([{"file_name": "other.pdf"}], "sample.pdf") == {
        "file_name": "other.pdf"
    }
    with pytest.raises(MinerUError) as captured:
        MinerUClient._raise_task_failure(
            {"err_msg": "parser crashed; Authorization=runtime-secret"}
        )
    assert captured.value.category == "parse_failed"
    assert str(captured.value) == "MinerU document parsing failed"
    assert "runtime-secret" not in str(captured.value)


def test_download_retries_rejects_and_enforces_archive_limit(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b"zip")

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async def run_retry() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret", sleep=sleep)
            await client._download_archive("https://download.example/result", tmp_path / "ok.zip")

    asyncio.run(run_retry())
    assert (tmp_path / "ok.zip").read_bytes() == b"zip"
    assert delays == [1]

    async def reject_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    async def run_reject() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(reject_handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret")
            with pytest.raises(MinerUError) as captured:
                await client._download_archive(
                    "https://download.example/result",
                    tmp_path / "reject.zip",
                )
            assert captured.value.category == "download_failed"
            assert captured.value.retryable is False

    asyncio.run(run_reject())

    async def large_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"too-large")

    async def run_large() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(large_handler)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(max_archive_bytes=2),
            )
            with pytest.raises(MinerUError, match="exceeds"):
                await client._download_archive(
                    "https://download.example/result",
                    tmp_path / "large.zip",
                )

    asyncio.run(run_large())


def test_download_transport_exhaustion_is_retryable(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("offline", request=request)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(retry_max_attempts=1),
            )
            with pytest.raises(MinerUError) as captured:
                await client._download_archive(
                    "https://download.example/result",
                    tmp_path / "transport.zip",
                )
            assert captured.value.category == "download_transport"

    asyncio.run(run())


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)


def test_archive_structural_limits_and_bad_zip_are_rejected(tmp_path: Path) -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as http:
            too_many = tmp_path / "many.zip"
            _write_zip(too_many, {"a.txt": "a", "content_list.json": "[]"})
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(max_zip_entries=1),
            )
            with pytest.raises(MinerUError, match="too many"):
                client._read_normalized_blocks(too_many)

            expanded = tmp_path / "expanded.zip"
            _write_zip(
                expanded, {"content_list.json": json.dumps([{"type": "text", "text": "abc"}])}
            )
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(max_uncompressed_bytes=1),
            )
            with pytest.raises(MinerUError, match="expands"):
                client._read_normalized_blocks(expanded)

            content_large = tmp_path / "content-large.zip"
            _write_zip(content_large, {"content_list.json": "[]"})
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(max_content_list_bytes=1),
            )
            with pytest.raises(MinerUError, match="content_list exceeds"):
                client._read_normalized_blocks(content_large)

            missing = tmp_path / "missing.zip"
            _write_zip(missing, {"readme.txt": "none"})
            client = MinerUClient(http, token_provider=lambda: "secret")
            with pytest.raises(MinerUError, match="no content_list"):
                client._read_normalized_blocks(missing)

            bad = tmp_path / "bad.zip"
            bad.write_bytes(b"not-a-zip")
            with pytest.raises(MinerUError, match="invalid result archive"):
                client._read_normalized_blocks(bad)

    asyncio.run(run())


def test_v2_and_all_readable_block_types_are_normalized() -> None:
    client = MinerUClient(httpx.AsyncClient(), token_provider=lambda: "secret")
    try:
        blocks = client._normalize_content_list(
            [
                [
                    {
                        "type": "title",
                        "content": {
                            "title_content": [{"type": "text", "content": "Introduction"}],
                            "level": 2,
                        },
                        "bbox": "invalid",
                    },
                    {
                        "type": "image",
                        "image_caption": ["Figure 1"],
                        "content": {"value": 1, "img_path": "ignored.png"},
                        "page_idx": "bad-page",
                    },
                    {"type": "code", "code_caption": ["Algorithm"], "code_body": "run()"},
                    {"type": "list", "list_items": ["one", "two"]},
                    {"type": "equation", "latex": "E=mc^2"},
                    {"type": "mystery", "content": {"text": "fallback"}},
                    {"type": "text", "text": ""},
                ],
                [{"type": "text", "text": "body", "bbox": [1, 2, "bad", 4]}],
            ]
        )
    finally:
        asyncio.run(client._http.aclose())

    assert {item.block_type for item in blocks} >= {
        "title",
        "image",
        "code",
        "list",
        "equation",
        "mystery",
        "text",
    }
    assert blocks[0].section
    assert blocks[1].page == 1
    assert blocks[-1].page == 2
    assert MinerUClient._heading_level("text", {"text_level": "bad"}) == 0
    assert MinerUClient._bbox(None) is None
    assert MinerUClient._bbox([1, 2, "bad", 4]) is None
    assert MinerUClient._first_text({}, "missing") == ""
    assert MinerUClient._flatten_text(object()) == ""

    with pytest.raises(MinerUError):
        client._normalize_content_list([{}, []])
    with pytest.raises(MinerUError):
        client._normalize_content_list({"pages": []})
    with pytest.raises(MinerUError, match="no readable"):
        client._normalize_content_list([{"type": "text", "text": ""}])


def test_api_json_error_classification_and_transport() -> None:
    async def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async def empty_token() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(no_network)) as http:
            client = MinerUClient(http, token_provider=lambda: "")
            with pytest.raises(MinerUError) as captured:
                await client._api_json("GET", "/test")
            assert captured.value.category == "authentication"

    asyncio.run(empty_token())

    cases = [
        (httpx.Response(500), "provider_unavailable"),
        (httpx.Response(400), "request_rejected"),
        (httpx.Response(200, content=b"not-json"), "invalid_result"),
        (httpx.Response(200, json=[1, 2]), "invalid_result"),
        (
            httpx.Response(
                200,
                json={"code": 9, "msg": "business; token=runtime-secret"},
            ),
            "provider_business_error",
        ),
    ]
    for response, category in cases:

        async def handler(
            request: httpx.Request,
            current: httpx.Response = response,
        ) -> httpx.Response:
            return current

        async def run(expected: str = category) -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
                client = MinerUClient(
                    http,
                    token_provider=lambda: "secret",
                    settings=MinerUSettings(retry_max_attempts=1),
                )
                with pytest.raises(MinerUError) as captured:
                    await client._api_json("GET", "/test")
                assert captured.value.category == expected
                assert "runtime-secret" not in str(captured.value)

        asyncio.run(run())

    async def transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async def run_transport() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "secret",
                settings=MinerUSettings(retry_max_attempts=1),
            )
            with pytest.raises(MinerUError) as captured:
                await client._api_json("GET", "/test")
            assert captured.value.category == "transport"

    asyncio.run(run_transport())


def test_retry_delay_supports_exponential_seconds_and_http_dates() -> None:
    client = MinerUClient(httpx.AsyncClient(), token_provider=lambda: "secret")
    try:
        assert client._retry_delay(None, 3) == 4
        invalid = httpx.Response(429, headers={"Retry-After": "not-a-date"})
        assert client._retry_delay(invalid, 2) == 2
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5), usegmt=True)
        parsed = client._parse_retry_after(future)
        assert parsed is not None and 0 <= parsed <= 5
        assert client._parse_retry_after("Sun Nov  6 08:49:37 1994") == 0
    finally:
        asyncio.run(client._http.aclose())
