from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from material_graph.knowledge.mineru_client import (
    MinerUClient,
    MinerUError,
    MinerUSettings,
)


def _archive(content_list: object, *, member: str = "sample_content_list.json") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, json.dumps(content_list, ensure_ascii=False))
        archive.writestr("images/ignored.png", b"image-bytes")
    return buffer.getvalue()


def _content_list() -> list[dict[str, object]]:
    return [
        {
            "type": "text",
            "text": "2 Experimental",
            "text_level": 1,
            "bbox": [10, 20, 900, 80],
            "page_idx": 0,
        },
        {
            "type": "text",
            "text": "The measured glass-transition temperature was 315 °C.",
            "bbox": [10, 90, 900, 150],
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_caption": ["Table 1. Thermal properties"],
            "table_body": "<table><tr><td>Tg</td><td>315 °C</td></tr></table>",
            "table_footnote": ["Measured by DSC."],
            "bbox": [10, 160, 900, 500],
            "page_idx": 1,
        },
    ]


def test_precision_flow_streams_upload_polls_and_normalizes_blocks(tmp_path: Path) -> None:
    archive_bytes = _archive(_content_list())
    calls: list[tuple[str, str]] = []
    uploaded = bytearray()
    poll_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v4/file-urls/batch":
            assert request.headers["Authorization"] == "Bearer runtime-secret"
            payload = json.loads((await request.aread()).decode("utf-8"))
            assert payload["model_version"] == "vlm"
            assert payload["files"] == [
                {"name": "sample.pdf", "is_ocr": False, "data_id": "parse-key-1"}
            ]
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )
        if request.url.host == "upload.example":
            assert "authorization" not in request.headers
            assert "content-type" not in request.headers
            uploaded.extend(await request.aread())
            return httpx.Response(200)
        if request.url.path == "/api/v4/extract-results/batch/batch-1":
            poll_count += 1
            state = "running" if poll_count == 1 else "done"
            row: dict[str, object] = {
                "task_id": "task-1",
                "file_name": "sample.pdf",
                "state": state,
            }
            if state == "done":
                row["full_zip_url"] = "https://download.example/result.zip"
            return httpx.Response(
                200,
                json={"code": 0, "data": {"extract_result": [row]}},
            )
        if request.url.host == "download.example":
            assert "authorization" not in request.headers
            return httpx.Response(200, content=archive_bytes)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    source = tmp_path / "source.pdf"
    source.write_bytes(b"streamed-pdf")
    output = tmp_path / "parser-output"
    sleeps: list[float] = []

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "runtime-secret",
                settings=MinerUSettings(poll_initial_seconds=0.01, poll_jitter_ratio=0),
                sleep=lambda delay: _record_sleep(sleeps, delay),
            )
            return await client.parse(
                source,
                file_name="sample.pdf",
                idempotency_key="parse-key-1",
                output_dir=output,
            )

    result = asyncio.run(run())

    assert uploaded == b"streamed-pdf"
    assert result.batch_id == "batch-1"
    assert result.task_id == "task-1"
    assert result.parser_name == "mineru"
    assert result.parser_version == "3.4.4"
    assert result.remote_artifact_deletion_status == "not_supported"
    assert [block.page for block in result.blocks] == [1, 1, 2]
    assert result.blocks[1].section == "2 Experimental"
    assert result.blocks[2].block_type == "table"
    assert "Thermal properties" in result.blocks[2].text
    assert list(output.iterdir()) == []
    assert calls == [
        ("POST", "/api/v4/file-urls/batch"),
        ("PUT", "/signed"),
        ("GET", "/api/v4/extract-results/batch/batch-1"),
        ("GET", "/api/v4/extract-results/batch/batch-1"),
        ("GET", "/result.zip"),
    ]
    assert sleeps == [0.01]


async def _record_sleep(delays: list[float], delay: float) -> None:
    delays.append(delay)


def test_429_honors_retry_after_without_persisting_token(tmp_path: Path) -> None:
    submit_count = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal submit_count
        if request.url.path == "/api/v4/file-urls/batch":
            submit_count += 1
            if submit_count == 1:
                return httpx.Response(429, headers={"Retry-After": "3"})
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-2",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )
        if request.url.host == "upload.example":
            await request.aread()
            return httpx.Response(200)
        if request.url.path.endswith("/batch-2"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "task_id": "task-2",
                                "file_name": "sample.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.example/result.zip",
                            }
                        ]
                    },
                },
            )
        return httpx.Response(200, content=_archive(_content_list()))

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(
                http,
                token_provider=lambda: "do-not-persist",
                settings=MinerUSettings(poll_jitter_ratio=0),
                sleep=lambda delay: _record_sleep(delays, delay),
            )
            assert "do-not-persist" not in repr(client)
            assert "token" not in client.settings.model_dump(mode="json")
            return await client.parse(
                source,
                file_name="sample.pdf",
                idempotency_key="retry-key",
                output_dir=tmp_path / "out",
            )

    result = asyncio.run(run())
    assert result.task_id == "task-2"
    assert delays == [3.0]


def test_auth_failure_is_not_retried_or_leaked(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"code": 401, "msg": "bad token"})

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "secret-value")
            with pytest.raises(MinerUError) as captured:
                await client.parse(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="auth-key",
                    output_dir=tmp_path / "out",
                )
            assert captured.value.category == "authentication"
            assert captured.value.retryable is False
            assert "secret-value" not in str(captured.value)

    asyncio.run(run())
    assert calls == 1


def test_failed_task_stops_without_downloading_result(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v4/file-urls/batch":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "failed-batch",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )
        if request.url.host == "upload.example":
            await request.aread()
            return httpx.Response(200)
        if request.url.path.endswith("/failed-batch"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "task_id": "failed-task",
                                "file_name": "sample.pdf",
                                "state": "failed",
                                "err_code": "-60006",
                                "err_msg": "number of pages exceeds limit",
                            }
                        ]
                    },
                },
            )
        raise AssertionError("result archive must not be downloaded")

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "runtime-secret")
            with pytest.raises(MinerUError, match="size or page count") as captured:
                await client.parse(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="failed-key",
                    output_dir=tmp_path / "out",
                )
            assert captured.value.category == "document_limit"
            assert captured.value.retryable is False
            assert "number of pages exceeds limit" not in str(captured.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "archive_bytes",
    [
        _archive(_content_list(), member="../escape_content_list.json"),
        _archive({"unexpected": "shape"}),
    ],
)
def test_unsafe_or_invalid_archive_is_rejected_and_deleted(
    tmp_path: Path,
    archive_bytes: bytes,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v4/file-urls/batch":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "archive-batch",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )
        if request.url.host == "upload.example":
            await request.aread()
            return httpx.Response(200)
        if request.url.path.endswith("/archive-batch"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "task_id": "archive-task",
                                "file_name": "sample.pdf",
                                "state": "done",
                                "full_zip_url": "https://download.example/result.zip",
                            }
                        ]
                    },
                },
            )
        return httpx.Response(200, content=archive_bytes)

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    output = tmp_path / "out"

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "runtime-secret")
            with pytest.raises(MinerUError) as captured:
                await client.parse(
                    source,
                    file_name="sample.pdf",
                    idempotency_key="archive-key",
                    output_dir=output,
                )
            assert captured.value.category == "invalid_result"

    asyncio.run(run())
    assert list(output.iterdir()) == []


def test_rejects_non_basename_filename_before_network_or_file_read(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network should not be called")

    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MinerUClient(http, token_provider=lambda: "runtime-secret")
            with pytest.raises(ValueError, match="basename"):
                await client.parse(
                    source,
                    file_name="../sample.pdf",
                    idempotency_key="path-key",
                    output_dir=tmp_path / "out",
                )

    asyncio.run(run())
