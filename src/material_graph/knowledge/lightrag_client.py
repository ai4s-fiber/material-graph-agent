"""Async REST boundary for retained-evidence insertion into LightRAG v1.5.4."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol

import httpx
from pydantic import ValidationError

from .lightrag_models import (
    LightRAGInsertAcceptance,
    LightRAGInsertResult,
    LightRAGSourceMapping,
    LightRAGTextRequest,
    LightRAGTextsRequest,
    LightRAGTrackStatus,
)
from .models import EvidenceFragment


Sleep = Callable[[float], Awaitable[None]]
_ALLOWED_POST_PATHS = frozenset({"/documents/text", "/documents/texts"})
_TRACK_STATUS_PATH = re.compile(r"^/documents/track_status/[A-Za-z0-9._-]+$")
_SAFE_TRACK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_IDEMPOTENCY_KEY = re.compile(r"^lightrag-insert:v1:[0-9a-f]{64}$")
_RAW_METADATA_KEYS = frozenset(
    {
        "complete_mineru_output",
        "complete_parser_output",
        "full_document",
        "full_document_text",
        "mineru_json",
        "mineru_markdown",
        "original_pdf",
        "pdf_bytes",
        "raw_document",
        "raw_pdf",
        "source_bytes",
    }
)


class LightRAGError(RuntimeError):
    """Base error for the local LightRAG REST boundary."""


class LightRAGForbiddenOperation(LightRAGError):
    """Raised before transport when code attempts an unapproved endpoint."""


class LightRAGProtocolError(LightRAGError):
    """Raised when LightRAG returns data outside its pinned API contract."""


class LightRAGPollingTimeout(LightRAGError):
    """Raised when a track does not reach a terminal state within its budget."""


class LightRAGSourceMappingConflict(LightRAGError):
    """Raised when one basename is already bound to different provenance."""


class LightRAGRequestError(LightRAGError):
    """Secret-free HTTP failure suitable for an execution trace."""

    def __init__(self, *, path: str, status_code: int | None, detail: str) -> None:
        self.path = path
        self.status_code = status_code
        self.detail = detail
        status = "transport" if status_code is None else str(status_code)
        super().__init__(f"LightRAG request failed ({status}) at {path}: {detail}")


class LightRAGSourceMappingStore(Protocol):
    """Durable repository contract; production must implement atomic persistence."""

    async def persist_many(self, mappings: Sequence[LightRAGSourceMapping]) -> None: ...

    async def get(self, basename: str) -> LightRAGSourceMapping | None: ...


def build_lightrag_insert_idempotency_key(
    mappings: Sequence[LightRAGSourceMapping],
) -> str:
    """Bind one insertion replay to already-persisted derived provenance."""

    prepared = list(mappings)
    if not prepared:
        raise ValueError("at least one source mapping is required")
    identities = sorted(
        f"{mapping.basename}:{mapping.content_sha256}:{mapping.embedding_generation_id}"
        for mapping in prepared
    )
    digest = sha256("\n".join(identities).encode("utf-8")).hexdigest()
    return f"lightrag-insert:v1:{digest}"


class InMemoryLightRAGSourceMappingStore:
    """Atomic in-memory test double for the production source-mapping repository."""

    def __init__(self) -> None:
        self._mappings: dict[str, LightRAGSourceMapping] = {}
        self._lock = asyncio.Lock()

    async def persist_many(self, mappings: Sequence[LightRAGSourceMapping]) -> None:
        candidates = {mapping.basename: mapping for mapping in mappings}
        if len(candidates) != len(mappings):
            raise LightRAGSourceMappingConflict("duplicate basename in mapping transaction")

        async with self._lock:
            for basename, mapping in candidates.items():
                existing = self._mappings.get(basename)
                if existing is not None and existing != mapping:
                    raise LightRAGSourceMappingConflict(
                        f"basename {basename!r} is already mapped to different provenance"
                    )
            self._mappings.update(
                {
                    basename: mapping.model_copy(deep=True)
                    for basename, mapping in candidates.items()
                }
            )

    async def get(self, basename: str) -> LightRAGSourceMapping | None:
        async with self._lock:
            mapping = self._mappings.get(basename)
            return None if mapping is None else mapping.model_copy(deep=True)

    async def list_all(self) -> list[LightRAGSourceMapping]:
        async with self._lock:
            return [self._mappings[key].model_copy(deep=True) for key in sorted(self._mappings)]


class _APIKeyAuth(httpx.Auth):
    __slots__ = ("__credential",)

    def __init__(self, credential: str) -> None:
        self.__credential = credential

    def auth_flow(self, request: httpx.Request):
        request.headers["X-API-Key"] = self.__credential
        yield request


class LightRAGClient:
    """Insert retained fragments and hold admission until their tracks terminate."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        mapping_store: LightRAGSourceMappingStore,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: httpx.Timeout | float | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        outstanding_track_limit: int = 4,
        poll_interval_seconds: float = 1.0,
        max_poll_attempts: int = 300,
        max_fragment_chars: int = 65_536,
        max_batch_fragments: int = 32,
        max_batch_chars: int = 524_288,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        credential = api_key.strip()
        if not credential:
            raise ValueError("api_key is required at runtime")
        parsed_base_url = httpx.URL(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.host:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        numeric_limits = {
            "outstanding_track_limit": outstanding_track_limit,
            "max_poll_attempts": max_poll_attempts,
            "max_fragment_chars": max_fragment_chars,
            "max_batch_fragments": max_batch_fragments,
            "max_batch_chars": max_batch_chars,
        }
        if any(value <= 0 for value in numeric_limits.values()):
            raise ValueError("client limits must be positive")
        if max_retries < 0 or retry_backoff_seconds < 0 or poll_interval_seconds < 0:
            raise ValueError("retry and polling values cannot be negative")

        self._base_url = str(parsed_base_url).rstrip("/")
        self._mapping_store = mapping_store
        self._auth = _APIKeyAuth(credential)
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_attempts = max_poll_attempts
        self._max_fragment_chars = max_fragment_chars
        self._max_batch_fragments = max_batch_fragments
        self._max_batch_chars = max_batch_chars
        self._sleep = sleep
        self._outstanding_limit = outstanding_track_limit
        self._outstanding_tracks = asyncio.Semaphore(outstanding_track_limit)
        resolved_timeout = timeout or httpx.Timeout(30.0, connect=5.0)
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            auth=self._auth,
            timeout=resolved_timeout,
            transport=transport,
            trust_env=False,
            headers={"Accept": "application/json"},
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"outstanding_track_limit={self._outstanding_limit})"
        )

    async def __aenter__(self) -> "LightRAGClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def insert_retained_fragment(
        self,
        fragment: EvidenceFragment,
    ) -> LightRAGInsertResult:
        fragments, mappings = self._prepare_fragments([fragment])
        await self._mapping_store.persist_many(mappings)
        request = LightRAGTextRequest(
            text=fragments[0].text,
            file_source=mappings[0].basename,
        )
        async with self._outstanding_tracks:
            return await self._submit_and_wait(
                path="/documents/text",
                payload=request.model_dump(mode="json"),
                mappings=mappings,
            )

    async def insert_retained_fragments(
        self,
        fragments: Sequence[EvidenceFragment],
    ) -> LightRAGInsertResult:
        prepared, mappings = self._prepare_fragments(fragments)
        await self._mapping_store.persist_many(mappings)
        request = LightRAGTextsRequest(
            texts=[fragment.text for fragment in prepared],
            file_sources=[mapping.basename for mapping in mappings],
        )
        async with self._outstanding_tracks:
            return await self._submit_and_wait(
                path="/documents/texts",
                payload=request.model_dump(mode="json"),
                mappings=mappings,
            )

    def _prepare_fragments(
        self,
        fragments: Sequence[EvidenceFragment],
    ) -> tuple[list[EvidenceFragment], list[LightRAGSourceMapping]]:
        prepared = list(fragments)
        if not prepared:
            raise ValueError("at least one retained evidence fragment is required")
        if len(prepared) > self._max_batch_fragments:
            raise ValueError("retained evidence batch exceeds configured fragment limit")

        for fragment in prepared:
            self._validate_fragment(fragment)
        generations = {fragment.embedding_generation_id for fragment in prepared}
        if len(generations) != 1:
            raise ValueError("one batch cannot mix embedding generations")
        if sum(len(fragment.text) for fragment in prepared) > self._max_batch_chars:
            raise ValueError("retained evidence batch exceeds configured character limit")

        mappings = [LightRAGSourceMapping.from_fragment(fragment) for fragment in prepared]
        if len({mapping.basename for mapping in mappings}) != len(mappings):
            raise ValueError("retained evidence fragments must have unique identities")
        return prepared, mappings

    def _validate_fragment(self, fragment: EvidenceFragment) -> None:
        if not isinstance(fragment, EvidenceFragment):
            raise TypeError("LightRAG insertion accepts EvidenceFragment instances only")
        text = fragment.text
        if text != text.strip() or not text:
            raise ValueError("retained evidence text must be non-blank and normalized")
        if len(text) > self._max_fragment_chars:
            raise ValueError("retained evidence fragment exceeds configured character limit")
        if text.startswith("%PDF-"):
            raise ValueError("retained evidence cannot contain a raw PDF body")
        if self._contains_raw_parser_payload(fragment.metadata):
            raise ValueError("retained evidence cannot contain complete parser output")

    @classmethod
    def _contains_raw_parser_payload(cls, value: object) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).strip().casefold().replace("-", "_")
                if normalized in _RAW_METADATA_KEYS:
                    return True
                if cls._contains_raw_parser_payload(nested):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(cls._contains_raw_parser_payload(item) for item in value)
        return isinstance(value, (bytes, bytearray, memoryview))

    async def _submit_and_wait(
        self,
        *,
        path: str,
        payload: dict[str, object],
        mappings: list[LightRAGSourceMapping],
    ) -> LightRAGInsertResult:
        status_code, response_payload = await self._request_json(
            "POST",
            path,
            payload=payload,
            idempotency_key=build_lightrag_insert_idempotency_key(mappings),
        )
        if status_code == 409:
            return LightRAGInsertResult(
                outcome="idempotent_conflict",
                mappings=mappings,
                message="idempotent_conflict",
            )

        try:
            acceptance = LightRAGInsertAcceptance.model_validate(response_payload)
        except ValidationError:
            raise LightRAGProtocolError("invalid LightRAG insertion response") from None
        if acceptance.status != "success":
            raise LightRAGProtocolError(
                f"LightRAG did not accept retained evidence: {acceptance.status}"
            )

        track = await self.wait_for_track(acceptance.track_id)
        expected_paths = {mapping.basename for mapping in mappings}
        actual_paths = {document.file_path for document in track.documents}
        if actual_paths != expected_paths:
            raise LightRAGProtocolError("track documents do not match persisted source mappings")
        outcome = "failed" if track.has_failures else "processed"
        return LightRAGInsertResult(
            outcome=outcome,
            mappings=mappings,
            track_id=acceptance.track_id,
            track_status=track,
            message="accepted",
        )

    async def wait_for_track(self, track_id: str) -> LightRAGTrackStatus:
        if not _SAFE_TRACK_ID.fullmatch(track_id):
            raise ValueError("invalid LightRAG track_id")
        path = f"/documents/track_status/{track_id}"
        for attempt in range(self._max_poll_attempts):
            _, payload = await self._request_json("GET", path)
            try:
                status = LightRAGTrackStatus.model_validate(payload)
            except ValidationError:
                raise LightRAGProtocolError("invalid LightRAG track status response") from None
            if status.track_id != track_id:
                raise LightRAGProtocolError("LightRAG returned a different track_id")
            if status.is_terminal:
                return status
            if attempt + 1 < self._max_poll_attempts:
                await self._sleep(self._poll_interval_seconds)
        raise LightRAGPollingTimeout(
            f"LightRAG track {track_id!r} remained non-terminal after "
            f"{self._max_poll_attempts} polls"
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        normalized_method = method.upper()
        self._validate_operation(normalized_method, path)
        if idempotency_key is not None and (
            normalized_method != "POST" or _IDEMPOTENCY_KEY.fullmatch(idempotency_key) is None
        ):
            raise ValueError("invalid LightRAG idempotency key")

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.request(
                    normalized_method,
                    path,
                    json=payload,
                    headers=(
                        None if idempotency_key is None else {"Idempotency-Key": idempotency_key}
                    ),
                )
            except httpx.TransportError as error:
                if attempt >= self._max_retries:
                    raise LightRAGRequestError(
                        path=path,
                        status_code=None,
                        detail=type(error).__name__,
                    ) from None
                await self._sleep(self._retry_delay(attempt))
                continue

            status_code = response.status_code
            response_payload = (
                self._decode_response(response)
                if 200 <= status_code < 300
                else self._decode_error_response(response)
            )
            if 200 <= status_code < 300 or status_code == 409:
                return status_code, response_payload
            if status_code == 429 or 500 <= status_code < 600:
                if attempt >= self._max_retries:
                    raise self._request_error(path, response, response_payload)
                delay = (
                    self._parse_retry_after(response.headers.get("Retry-After"))
                    if status_code == 429
                    else None
                )
                await self._sleep(delay if delay is not None else self._retry_delay(attempt))
                continue
            raise self._request_error(path, response, response_payload)

        raise AssertionError("bounded request loop exhausted without returning")

    @staticmethod
    def _validate_operation(method: str, path: str) -> None:
        if method == "POST" and path in _ALLOWED_POST_PATHS:
            return
        if method == "GET" and _TRACK_STATUS_PATH.fullmatch(path):
            return
        raise LightRAGForbiddenOperation(f"LightRAG operation is not allowlisted: {method} {path}")

    def _decode_response(self, response: httpx.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError:
            raise LightRAGProtocolError(
                f"LightRAG returned non-JSON data with status {response.status_code}"
            ) from None
        if not isinstance(payload, dict):
            raise LightRAGProtocolError("LightRAG response must be a JSON object")
        return payload

    @staticmethod
    def _decode_error_response(response: httpx.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError:
            return {"detail": f"HTTP {response.status_code}"}
        if isinstance(payload, dict):
            return payload
        return {"detail": f"HTTP {response.status_code}"}

    def _request_error(
        self,
        path: str,
        response: httpx.Response,
        payload: dict[str, object],
    ) -> LightRAGRequestError:
        return LightRAGRequestError(
            path=path,
            status_code=response.status_code,
            detail=self._response_detail(payload, response.status_code),
        )

    def _response_detail(self, payload: dict[str, object], status_code: int) -> str:
        del payload
        if status_code == 409:
            return "idempotent_conflict"
        return f"HTTP {status_code}"

    def _retry_delay(self, attempt: int) -> float:
        return self._retry_backoff_seconds * (2**attempt)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
