"""Selective MinerU v4 client adapted from the official SDK protocol.

Protocol behavior follows ``opendatalab/MinerU-Ecosystem``'s Apache-2.0 Go SDK
and the official MinerU 3.4 output contract.  This adapter adds bounded streaming
and returns normalized blocks only; it never retains a complete result archive.
"""

from __future__ import annotations

import asyncio
import json
import random
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class MinerUError(RuntimeError):
    """Safe provider error without request headers, signed URLs, or tokens."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


class MinerUSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "https://mineru.net/api/v4"
    parser_version: str = "3.4.4"
    model_version: str = "vlm"
    language: str | None = None
    enable_formula: bool = True
    enable_table: bool = True
    is_ocr: bool = False
    submit_timeout_seconds: float = Field(default=60, gt=0)
    upload_timeout_seconds: float = Field(default=300, gt=0)
    download_timeout_seconds: float = Field(default=300, gt=0)
    poll_timeout_seconds: float = Field(default=600, gt=0)
    poll_initial_seconds: float = Field(default=2, gt=0)
    poll_max_seconds: float = Field(default=30, gt=0)
    poll_jitter_ratio: float = Field(default=0.2, ge=0, le=1)
    retry_max_attempts: int = Field(default=4, ge=1, le=8)
    retry_base_seconds: float = Field(default=1, gt=0)
    retry_max_seconds: float = Field(default=30, gt=0)
    upload_chunk_bytes: int = Field(default=1024 * 1024, gt=0)
    max_archive_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_uncompressed_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    max_content_list_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    max_zip_entries: int = Field(default=5000, gt=0)


class MinerUBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_type: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page: int = Field(ge=1)
    block_index: int = Field(ge=0)
    section: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MinerUParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    parser_name: Literal["mineru"] = "mineru"
    parser_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    blocks: list[MinerUBlock]
    remote_artifact_deletion_status: Literal[
        "deleted",
        "delete_failed",
        "not_supported",
        "unknown",
    ] = "not_supported"
    warnings: list[str] = Field(default_factory=list)


Sleep = Callable[[float], Awaitable[None]]
TokenProvider = Callable[[], str]


class MinerUClient:
    """One-selected-source-at-a-time MinerU precision API client."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        token_provider: TokenProvider,
        settings: MinerUSettings | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._http = http
        self._token_provider = token_provider
        self.settings = settings or MinerUSettings()
        self._sleep = sleep

    def __repr__(self) -> str:
        return (
            f"MinerUClient(model_version={self.settings.model_version!r}, "
            f"parser_version={self.settings.parser_version!r})"
        )

    async def parse(
        self,
        source_path: str | Path,
        *,
        file_name: str,
        idempotency_key: str,
        output_dir: str | Path,
    ) -> MinerUParseResult:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if (
            not file_name
            or file_name != Path(file_name).name
            or "/" in file_name
            or "\\" in file_name
        ):
            raise ValueError("file_name must be a single basename")

        batch_id, upload_url = await self._request_upload_slot(
            source,
            file_name=file_name,
            idempotency_key=idempotency_key,
        )
        await self._upload_source(source, upload_url)
        task = await self._poll_batch(batch_id, file_name)

        zip_url = str(task.get("full_zip_url") or "").strip()
        task_id = str(task.get("task_id") or "").strip()
        if not zip_url or not task_id:
            raise MinerUError(
                "MinerU completed without a task ID or result archive",
                category="invalid_result",
                retryable=False,
            )

        parser_output = Path(output_dir)
        parser_output.mkdir(parents=True, exist_ok=True)
        archive_path = parser_output / f"mineru-{batch_id}.zip"
        try:
            await self._download_archive(zip_url, archive_path)
            blocks = self._read_normalized_blocks(archive_path)
        finally:
            archive_path.unlink(missing_ok=True)

        return MinerUParseResult(
            batch_id=batch_id,
            task_id=task_id,
            filename=file_name,
            parser_version=self.settings.parser_version,
            model_version=self.settings.model_version,
            blocks=blocks,
            remote_artifact_deletion_status="not_supported",
        )

    async def _request_upload_slot(
        self,
        source: Path,
        *,
        file_name: str,
        idempotency_key: str,
    ) -> tuple[str, str]:
        file_entry: dict[str, object] = {
            "name": file_name,
            "is_ocr": self.settings.is_ocr,
            "data_id": idempotency_key,
        }
        body: dict[str, object] = {
            "files": [file_entry],
            "model_version": self.settings.model_version,
            "enable_formula": self.settings.enable_formula,
            "enable_table": self.settings.enable_table,
        }
        if self.settings.language:
            body["language"] = self.settings.language

        payload = await self._api_json("POST", "/file-urls/batch", json_body=body)
        data = payload.get("data")
        if not isinstance(data, dict):
            self._invalid_result("MinerU upload response has no data object")
        batch_id = str(data.get("batch_id") or "").strip()
        file_urls = data.get("file_urls")
        if (
            not batch_id
            or not isinstance(file_urls, list)
            or len(file_urls) != 1
            or not str(file_urls[0]).strip()
        ):
            self._invalid_result("MinerU upload response is missing batch_id/file_urls")
        if source.stat().st_size <= 0:
            raise MinerUError(
                "selected source is empty",
                category="invalid_source",
                retryable=False,
            )
        return batch_id, str(file_urls[0])

    async def _upload_source(self, source: Path, signed_url: str) -> None:
        before = source.stat()
        for attempt in range(1, self.settings.retry_max_attempts + 1):
            try:
                response = await self._http.put(
                    signed_url,
                    content=self._iter_file(source),
                    headers={"Content-Length": str(before.st_size)},
                    timeout=self.settings.upload_timeout_seconds,
                )
            except httpx.HTTPError:
                if attempt >= self.settings.retry_max_attempts:
                    raise MinerUError(
                        "MinerU signed upload failed",
                        category="upload_transport",
                        retryable=True,
                    ) from None
                await self._sleep(self._retry_delay(None, attempt))
                continue

            if response.status_code < 400:
                after = source.stat()
                if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
                    raise MinerUError(
                        "selected source changed during upload",
                        category="source_changed",
                        retryable=True,
                    )
                return
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.settings.retry_max_attempts:
                    await self._sleep(self._retry_delay(response, attempt))
                    continue
                raise MinerUError(
                    "MinerU signed upload is temporarily unavailable",
                    category="upload_rate_or_service",
                    retryable=True,
                    status_code=response.status_code,
                )
            raise MinerUError(
                f"MinerU signed upload was rejected with HTTP {response.status_code}",
                category="upload_rejected",
                retryable=False,
                status_code=response.status_code,
            )
        raise AssertionError("upload retry loop exhausted")  # pragma: no cover

    async def _iter_file(self, source: Path) -> AsyncIterator[bytes]:
        with source.open("rb") as handle:
            while True:
                chunk = await asyncio.to_thread(handle.read, self.settings.upload_chunk_bytes)
                if not chunk:
                    return
                yield chunk

    async def _poll_batch(self, batch_id: str, file_name: str) -> dict[str, Any]:
        deadline = monotonic() + self.settings.poll_timeout_seconds
        interval = self.settings.poll_initial_seconds
        while True:
            payload = await self._api_json("GET", f"/extract-results/batch/{batch_id}")
            data = payload.get("data")
            rows = data.get("extract_result") if isinstance(data, dict) else None
            task = self._match_task(rows, file_name)
            if task is not None:
                state = str(task.get("state") or "").strip().casefold()
                if state == "done":
                    return task
                if state == "failed":
                    self._raise_task_failure(task)

            if monotonic() >= deadline:
                raise MinerUError(
                    "MinerU parsing timed out",
                    category="timeout",
                    retryable=True,
                )
            jitter = interval * self.settings.poll_jitter_ratio
            delay = (
                interval if jitter == 0 else random.uniform(interval - jitter, interval + jitter)
            )
            await self._sleep(max(0.0, delay))
            interval = min(self.settings.poll_max_seconds, interval * 2)

    @staticmethod
    def _match_task(rows: object, file_name: str) -> dict[str, Any] | None:
        if not isinstance(rows, list):
            return None
        candidates = [row for row in rows if isinstance(row, dict)]
        for row in candidates:
            if str(row.get("file_name") or "") == file_name:
                return row
        return candidates[0] if candidates else None

    @staticmethod
    def _raise_task_failure(task: dict[str, Any]) -> None:
        provider_message = str(task.get("err_msg") or "")
        lowered = provider_message.casefold()
        category = (
            "document_limit"
            if "limit" in lowered and ("page" in lowered or "size" in lowered)
            else "parse_failed"
        )
        message = (
            "MinerU rejected the document size or page count"
            if category == "document_limit"
            else "MinerU document parsing failed"
        )
        raise MinerUError(message, category=category, retryable=False)

    async def _download_archive(self, signed_url: str, target: Path) -> None:
        for attempt in range(1, self.settings.retry_max_attempts + 1):
            try:
                async with self._http.stream(
                    "GET",
                    signed_url,
                    timeout=self.settings.download_timeout_seconds,
                ) as response:
                    if response.status_code >= 400:
                        retryable = response.status_code == 429 or response.status_code >= 500
                        if retryable and attempt < self.settings.retry_max_attempts:
                            delay = self._retry_delay(response, attempt)
                        else:
                            raise MinerUError(
                                "MinerU result archive download failed",
                                category="download_failed",
                                retryable=retryable,
                                status_code=response.status_code,
                            )
                    else:
                        total = 0
                        with target.open("xb") as output:
                            async for chunk in response.aiter_bytes():
                                total += len(chunk)
                                if total > self.settings.max_archive_bytes:
                                    raise MinerUError(
                                        "MinerU result archive exceeds the configured limit",
                                        category="invalid_result",
                                        retryable=False,
                                    )
                                output.write(chunk)
                        return
            except MinerUError:
                raise
            except httpx.HTTPError:
                if attempt >= self.settings.retry_max_attempts:
                    raise MinerUError(
                        "MinerU result archive transport failed",
                        category="download_transport",
                        retryable=True,
                    ) from None
                delay = self._retry_delay(None, attempt)
            target.unlink(missing_ok=True)
            await self._sleep(delay)
        raise AssertionError("download retry loop exhausted")  # pragma: no cover

    def _read_normalized_blocks(self, archive_path: Path) -> list[MinerUBlock]:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) > self.settings.max_zip_entries:
                    self._invalid_result("MinerU result archive has too many entries")
                if sum(item.file_size for item in members) > self.settings.max_uncompressed_bytes:
                    self._invalid_result(
                        "MinerU result archive expands beyond the configured limit"
                    )

                for member in members:
                    normalized = PurePosixPath(member.filename.replace("\\", "/"))
                    if normalized.is_absolute() or ".." in normalized.parts:
                        self._invalid_result("MinerU result archive contains an unsafe path")

                legacy = [
                    item
                    for item in members
                    if PurePosixPath(item.filename).name.endswith("_content_list.json")
                    or PurePosixPath(item.filename).name == "content_list.json"
                ]
                v2 = [
                    item
                    for item in members
                    if PurePosixPath(item.filename).name.endswith("_content_list_v2.json")
                    or PurePosixPath(item.filename).name == "content_list_v2.json"
                ]
                candidates = legacy or v2
                if not candidates:
                    self._invalid_result("MinerU result has no content_list JSON")
                member = sorted(candidates, key=lambda item: item.filename)[0]
                if member.file_size > self.settings.max_content_list_bytes:
                    self._invalid_result("MinerU content_list exceeds the configured limit")
                payload = json.loads(archive.read(member).decode("utf-8"))
        except MinerUError:
            raise
        except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
            raise MinerUError(
                "MinerU returned an invalid result archive",
                category="invalid_result",
                retryable=False,
            ) from None

        return self._normalize_content_list(payload)

    def _normalize_content_list(self, payload: object) -> list[MinerUBlock]:
        rows: list[tuple[int | None, dict[str, Any]]] = []
        if isinstance(payload, list):
            if all(isinstance(item, dict) for item in payload):
                rows = [(None, item) for item in payload]
            elif all(isinstance(page, list) for page in payload):
                for page_index, page in enumerate(payload):
                    rows.extend((page_index, item) for item in page if isinstance(item, dict))
            else:
                self._invalid_result("MinerU content_list has an unsupported shape")
        else:
            self._invalid_result("MinerU content_list has an unsupported shape")

        sections: list[str] = []
        blocks: list[MinerUBlock] = []
        for original_index, (fallback_page, row) in enumerate(rows):
            block_type = str(row.get("type") or "unknown").strip().casefold()
            text = self._block_text(block_type, row).strip()
            if not text:
                continue

            level = self._heading_level(block_type, row)
            if level > 0:
                while len(sections) < level:
                    sections.append("")
                sections[level - 1] = text
                del sections[level:]
            section = " > ".join(item for item in sections if item) or None

            raw_page = row.get("page_idx", fallback_page if fallback_page is not None else 0)
            try:
                page = int(raw_page) + 1
            except (TypeError, ValueError):
                page = 1
            bbox = self._bbox(row.get("bbox"))
            metadata = {
                key: row[key]
                for key in ("sub_type", "text_level")
                if key in row and row[key] is not None
            }
            blocks.append(
                MinerUBlock(
                    block_type=block_type,
                    text=text,
                    page=max(1, page),
                    block_index=original_index,
                    section=section,
                    bbox=bbox,
                    metadata=metadata,
                )
            )

        if not blocks:
            self._invalid_result("MinerU content_list contains no readable blocks")
        return blocks

    @classmethod
    def _block_text(cls, block_type: str, row: dict[str, Any]) -> str:
        if block_type in {
            "text",
            "title",
            "paragraph",
            "header",
            "footer",
            "page_number",
            "aside_text",
            "page_footnote",
        }:
            return cls._first_text(row, "text", "content")
        if block_type in {"table", "chart"}:
            return cls._join_fields(
                row,
                "table_caption",
                "chart_caption",
                "content",
                "table_body",
                "table_footnote",
                "chart_footnote",
            )
        if block_type == "image":
            return cls._join_fields(row, "image_caption", "content", "image_footnote")
        if block_type == "code":
            return cls._join_fields(row, "code_caption", "code_body", "code_footnote")
        if block_type == "list":
            return cls._join_fields(row, "list_items", "text", "content")
        if block_type in {"equation", "equation_interline"}:
            return cls._join_fields(row, "text", "latex", "content")
        return cls._first_text(row, "text", "content")

    @classmethod
    def _first_text(cls, row: dict[str, Any], *keys: str) -> str:
        for key in keys:
            text = cls._flatten_text(row.get(key), key=key)
            if text:
                return text
        return ""

    @classmethod
    def _join_fields(cls, row: dict[str, Any], *keys: str) -> str:
        values = [cls._flatten_text(row.get(key), key=key) for key in keys]
        return "\n".join(value for value in values if value)

    @classmethod
    def _flatten_text(cls, value: object, *, key: str = "") -> str:
        ignored = {"bbox", "img_path", "image_path", "path", "type", "score"}
        if key in ignored or value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\n".join(text for item in value if (text := cls._flatten_text(item)))
        if isinstance(value, dict):
            return "\n".join(
                text
                for child_key, child in value.items()
                if (text := cls._flatten_text(child, key=str(child_key)))
            )
        return ""

    @staticmethod
    def _heading_level(block_type: str, row: dict[str, Any]) -> int:
        raw = row.get("text_level")
        if raw is None and block_type == "title" and isinstance(row.get("content"), dict):
            raw = row["content"].get("level")
        try:
            level = int(raw or 0)
        except (TypeError, ValueError):
            return 0
        return max(0, level)

    @staticmethod
    def _bbox(value: object) -> tuple[float, float, float, float] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None

    async def _api_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        for attempt in range(1, self.settings.retry_max_attempts + 1):
            token = self._token_provider().strip()
            if not token:
                raise MinerUError(
                    "MinerU API token is unavailable",
                    category="authentication",
                    retryable=False,
                )
            try:
                response = await self._http.request(
                    method,
                    url,
                    json=json_body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "X-Source": "ai4s-material-graph-studio",
                    },
                    timeout=self.settings.submit_timeout_seconds,
                )
            except httpx.HTTPError:
                if attempt < self.settings.retry_max_attempts:
                    await self._sleep(self._retry_delay(None, attempt))
                    continue
                raise MinerUError(
                    "MinerU API transport failed",
                    category="transport",
                    retryable=True,
                ) from None

            status = response.status_code
            if status in (401, 403):
                raise MinerUError(
                    "MinerU API rejected the runtime credential",
                    category="authentication",
                    retryable=False,
                    status_code=status,
                )
            if status == 429 or status >= 500:
                if attempt < self.settings.retry_max_attempts:
                    await self._sleep(self._retry_delay(response, attempt))
                    continue
                raise MinerUError(
                    "MinerU API is temporarily unavailable",
                    category="rate_limit" if status == 429 else "provider_unavailable",
                    retryable=True,
                    status_code=status,
                )
            if status >= 400:
                raise MinerUError(
                    f"MinerU API rejected the request with HTTP {status}",
                    category="request_rejected",
                    retryable=False,
                    status_code=status,
                )

            try:
                payload = response.json()
            except ValueError:
                self._invalid_result("MinerU API returned non-JSON data")
            if not isinstance(payload, dict):
                self._invalid_result("MinerU API returned an invalid JSON object")
            code = payload.get("code")
            if code not in (0, None):
                raise MinerUError(
                    "MinerU API returned a business error",
                    category="provider_business_error",
                    retryable=False,
                )
            return payload
        raise AssertionError("API retry loop exhausted")  # pragma: no cover

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                parsed = self._parse_retry_after(retry_after)
                if parsed is not None:
                    return min(self.settings.retry_max_seconds, max(0.0, parsed))
        exponential = self.settings.retry_base_seconds * (2 ** (attempt - 1))
        return min(self.settings.retry_max_seconds, exponential)

    @staticmethod
    def _parse_retry_after(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())

    @staticmethod
    def _invalid_result(message: str) -> None:
        raise MinerUError(message, category="invalid_result", retryable=False)
