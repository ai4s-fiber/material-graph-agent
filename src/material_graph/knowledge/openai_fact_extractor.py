"""OpenAI-compatible structured-output adapter for material fact extraction.

Only a bounded retained-evidence text and the pinned narrow JSON Schema cross
the provider boundary. Runtime credentials are fetched per attempt from an
injected provider and are never retained in models, traces, reprs, or errors.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from material_graph.providers.coordination import (
    AsyncProviderCoordinator,
    ProviderCoordinationError,
    ProviderWorkload,
)

from .extraction import (
    FactExtractorProviderError,
    FactExtractorRequest,
    export_extractor_payload_json_schema,
)


OpenAIFactExtractorErrorCode = Literal[
    "openai_fact_extractor.authentication",
    "openai_fact_extractor.credential_unavailable",
    "openai_fact_extractor.incomplete",
    "openai_fact_extractor.invalid_request",
    "openai_fact_extractor.invalid_response",
    "openai_fact_extractor.markdown_forbidden",
    "openai_fact_extractor.rate_limited",
    "openai_fact_extractor.refused",
    "openai_fact_extractor.request_rejected",
    "openai_fact_extractor.response_too_large",
    "openai_fact_extractor.timeout",
    "openai_fact_extractor.tool_calls_forbidden",
    "openai_fact_extractor.trace_unavailable",
    "openai_fact_extractor.transport",
    "openai_fact_extractor.unavailable",
]

Sleep = Callable[[float], Awaitable[None]]
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "session",
        "token",
    }
)
_SAFE_SCHEMA_NAME = "material_fact_candidates_v1"
_TRACE_ID = re.compile(r"^openai-fact-trace:v1:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _clean_text(value: str, *, code: str, max_length: int) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if (
        not normalized
        or len(normalized) > max_length
        or any(unicodedata.category(char) == "Cc" for char in normalized)
    ):
        raise ValueError(code)
    return normalized


class OpenAIFactExtractorConfigurationError(ValueError):
    """Stable error for a non-secret binding that cannot be trusted."""

    def __init__(self) -> None:
        super().__init__("openai_fact_extractor.invalid_configuration")


class OpenAIFactExtractorError(FactExtractorProviderError):
    """Stable provider error that never accepts remote detail or credentials."""

    def __init__(
        self,
        code: OpenAIFactExtractorErrorCode,
        *,
        retryable: bool,
        attempts: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(retryable=retryable)
        self.code = code
        self.attempts = attempts
        self.status_code = status_code
        self.args = (code,)


class OpenAITimeoutBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    connect_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)
    read_seconds: float = Field(gt=0, le=900, allow_inf_nan=False)
    write_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)
    pool_seconds: float = Field(gt=0, le=300, allow_inf_nan=False)

    def to_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect_seconds,
            read=self.read_seconds,
            write=self.write_seconds,
            pool=self.pool_seconds,
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> object:
    raise ValueError("nonfinite_json_number")


def _secret_field_found(value: object) -> bool:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().casefold().replace("-", "_")
            if key in _SECRET_FIELD_NAMES or any(
                key.endswith(f"_{marker}") for marker in _SECRET_FIELD_NAMES
            ):
                return True
            if _secret_field_found(nested):
                return True
        return False
    if isinstance(value, list):
        return any(_secret_field_found(item) for item in value)
    return False


class OpenAIFactExtractorBinding(BaseModel):
    """Pinned, non-secret OpenAI-compatible provider binding."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    provider: Literal["openai_compatible"] = "openai_compatible"
    binding: Literal[
        "responses_json_schema",
        "chat_completions_json_schema",
    ] = "responses_json_schema"
    base_url: str
    model: str = Field(min_length=1, max_length=200)
    model_version: str = Field(min_length=1, max_length=200)
    schema_generation_id: str = Field(min_length=1, max_length=200)
    timeouts: OpenAITimeoutBinding

    @field_validator("model", "model_version", "schema_generation_id")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return _clean_text(value, code="invalid_binding_text", max_length=200)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        candidate = value.strip().rstrip("/")
        if not candidate or len(candidate) > 2048:
            raise ValueError("invalid_base_url")
        parsed = urlsplit(candidate)
        path = PurePosixPath(parsed.path or "/")
        if (
            any(unicodedata.category(char) == "Cc" for char in candidate)
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
            or ".." in path.parts
        ):
            raise ValueError("invalid_base_url")
        return candidate

    @classmethod
    def load(cls, path: str | Path) -> "OpenAIFactExtractorBinding":
        try:
            raw = Path(path).read_bytes()
            if not raw or len(raw) > 65_536:
                raise ValueError("invalid binding size")
            payload = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite,
            )
            if not isinstance(payload, dict) or _secret_field_found(payload):
                raise ValueError("unsafe binding")
            return cls.model_validate(payload)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            RecursionError,
            TypeError,
            ValueError,
            ValidationError,
        ):
            raise OpenAIFactExtractorConfigurationError() from None


def build_openai_fact_idempotency_key(
    request: FactExtractorRequest,
    binding: OpenAIFactExtractorBinding,
) -> str:
    """Return a provider-safe replay identity without source text or locators."""

    validated_request = FactExtractorRequest.model_validate(request.model_dump(mode="python"))
    validated_binding = OpenAIFactExtractorBinding.model_validate(binding.model_dump(mode="python"))
    digest = _digest_text(
        _canonical_json(
            {
                "fragment_id": str(validated_request.fragment_id),
                "source_id": str(validated_request.source_id),
                "content_sha256": validated_request.content_sha256,
                "extraction_generation_id": validated_request.extraction.generation_id,
                "model": validated_binding.model,
                "model_version": validated_binding.model_version,
                "schema_generation_id": validated_binding.schema_generation_id,
                "binding": validated_binding.binding,
            }
        )
    )
    return f"openai-fact-request:v1:{digest}"


@runtime_checkable
class OpenAIAPIKeyProvider(Protocol):
    def __call__(self) -> str: ...


class FileOpenAIAPIKeyProvider:
    """Read a small, single-line API key file for each request attempt."""

    __slots__ = ("__path", "__max_bytes")

    def __init__(self, path: str | Path, *, max_bytes: int = 16_384) -> None:
        if max_bytes <= 0 or max_bytes > 65_536:
            raise ValueError("secret file limit is invalid")
        self.__path = Path(path)
        self.__max_bytes = max_bytes

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __call__(self) -> str:
        size = self.__path.stat().st_size
        if size <= 0 or size > self.__max_bytes:
            raise ValueError("secret file is outside the allowed size")
        raw = self.__path.read_text(encoding="utf-8")
        credential = raw.strip()
        if not credential or "\n" in credential or "\r" in credential or "\x00" in credential:
            raise ValueError("secret file must contain one value")
        return credential


class OpenAIFactTrace(BaseModel):
    """Safe trace summary containing hashes instead of provider identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    trace_id: str | None = Field(default=None, pattern=r"^openai-fact-trace:v1:[0-9a-f]{64}$")
    fragment_id: UUID
    source_id: UUID
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str = Field(min_length=1, max_length=200)
    model_version: str = Field(min_length=1, max_length=200)
    schema_generation_id: str = Field(min_length=1, max_length=200)
    response_id_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    finish_reason: Literal["stop"] = "stop"
    attempts: int = Field(ge=1, le=8)

    @model_validator(mode="after")
    def fill_trace_id(self) -> "OpenAIFactTrace":
        expected = (
            "openai-fact-trace:v1:"
            + sha256(
                _canonical_json(
                    {
                        "fragment_id": str(self.fragment_id),
                        "content_sha256": self.content_sha256,
                        "model": self.model,
                        "model_version": self.model_version,
                        "schema_generation_id": self.schema_generation_id,
                        "response_id_sha256": self.response_id_sha256,
                        "output_sha256": self.output_sha256,
                        "attempts": self.attempts,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        if self.trace_id is not None and self.trace_id != expected:
            raise ValueError("trace_id_mismatch")
        object.__setattr__(self, "trace_id", expected)
        return self


class OpenAIFactTraceConflict(RuntimeError):
    def __init__(self) -> None:
        super().__init__("openai_fact_extractor.trace_conflict")


@runtime_checkable
class OpenAIFactTraceSink(Protocol):
    async def record(self, trace: OpenAIFactTrace) -> None: ...


class InMemoryOpenAIFactTraceSink:
    """Atomic trace sink used by tests and local execution."""

    def __init__(self) -> None:
        self._items: dict[str, OpenAIFactTrace] = {}
        self._lock = asyncio.Lock()

    async def record(self, trace: OpenAIFactTrace) -> None:
        if not isinstance(trace, OpenAIFactTrace):
            raise TypeError("trace must be OpenAIFactTrace")
        candidate = OpenAIFactTrace.model_validate(trace.model_dump(mode="python"))
        if candidate.trace_id is None:  # pragma: no cover - model validator fills it
            raise OpenAIFactTraceConflict()
        async with self._lock:
            existing = self._items.get(candidate.trace_id)
            if existing is not None and existing != candidate:
                raise OpenAIFactTraceConflict()
            self._items[candidate.trace_id] = candidate

    async def list_all(self) -> list[OpenAIFactTrace]:
        async with self._lock:
            return [
                OpenAIFactTrace.model_validate(self._items[key].model_dump(mode="python"))
                for key in sorted(self._items)
            ]


class _ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    role: Literal["assistant"]
    content: str = Field(min_length=1)
    refusal: object | None = None
    tool_calls: object | None = None


class _ChatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    index: Literal[0]
    message: _ChatMessage
    finish_reason: str = Field(min_length=1, max_length=100)


class _ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)

    id: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=500)
    choices: tuple[_ChatChoice, ...] = Field(min_length=1, max_length=1)


class _InvalidResponse(ValueError):
    pass


def _strict_json_object(value: bytes | str) -> dict[str, object]:
    try:
        text = value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload
    except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise _InvalidResponse() from None


class OpenAIFactExtractor:
    """OpenAI-compatible ``FactExtractor`` using strict structured output."""

    SYSTEM_INSTRUCTION = (
        "Extract only material entities, relations, and numerical property observations "
        "explicitly supported by the supplied evidence text. Return exactly one JSON object "
        "matching the response schema. Never return markdown, prose, tool calls, source text, "
        "credentials, or fields outside the schema."
    )

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        binding: OpenAIFactExtractorBinding,
        api_key_provider: OpenAIAPIKeyProvider,
        trace_sink: OpenAIFactTraceSink,
        max_attempts: int = 3,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 8.0,
        max_input_bytes: int = 131_072,
        max_response_bytes: int = 262_144,
        max_completion_tokens: int = 8192,
        coordinator: AsyncProviderCoordinator | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not isinstance(http, httpx.AsyncClient):
            raise TypeError("http must be httpx.AsyncClient")
        if not isinstance(binding, OpenAIFactExtractorBinding):
            raise TypeError("binding must be OpenAIFactExtractorBinding")
        if not callable(api_key_provider):
            raise TypeError("api_key_provider must be callable")
        if max_attempts < 1 or max_attempts > 8:
            raise ValueError("max_attempts must be between 1 and 8")
        if (
            not math.isfinite(retry_base_seconds)
            or not math.isfinite(retry_max_seconds)
            or retry_base_seconds < 0
            or retry_max_seconds < retry_base_seconds
        ):
            raise ValueError("retry delay settings are invalid")
        if max_input_bytes < 1024 or max_input_bytes > 2_097_152:
            raise ValueError("max_input_bytes is invalid")
        if max_response_bytes < 1024 or max_response_bytes > 2_097_152:
            raise ValueError("max_response_bytes is invalid")
        if max_completion_tokens < 1 or max_completion_tokens > 131_072:
            raise ValueError("max_completion_tokens is invalid")

        self._http = http
        self._binding = binding
        self._api_key_provider = api_key_provider
        self._trace_sink = trace_sink
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._max_input_bytes = max_input_bytes
        self._max_response_bytes = max_response_bytes
        self._max_completion_tokens = max_completion_tokens
        self._coordinator = coordinator
        self._sleep = sleep
        self._timeout = binding.timeouts.to_httpx()
        endpoint = "responses" if binding.binding == "responses_json_schema" else "chat/completions"
        self._endpoint = f"{binding.base_url}/{endpoint}"
        self._expected_schema = export_extractor_payload_json_schema()
        self._expected_schema_json = _canonical_json(self._expected_schema)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._binding.base_url!r}, "
            f"model={self._binding.model!r}, "
            f"schema_generation_id={self._binding.schema_generation_id!r})"
        )

    async def extract(self, request: FactExtractorRequest) -> object:
        self._validate_request(request)
        payload = self._request_payload(request)
        idempotency_key = build_openai_fact_idempotency_key(request, self._binding)

        for attempt in range(1, self._max_attempts + 1):
            credential = self._credential()
            try:
                status_code, body, retry_after = await self._send_attempt(
                    payload,
                    credential=credential,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                )
            except ProviderCoordinationError:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt))
                    continue
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.unavailable",
                    retryable=True,
                    attempts=attempt,
                ) from None
            except OpenAIFactExtractorError:
                raise
            except httpx.TimeoutException:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt))
                    continue
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.timeout",
                    retryable=True,
                    attempts=attempt,
                ) from None
            except httpx.HTTPError:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt))
                    continue
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.transport",
                    retryable=True,
                    attempts=attempt,
                ) from None

            if status_code in {401, 403}:
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.authentication",
                    retryable=False,
                    attempts=attempt,
                    status_code=status_code,
                )
            if status_code == 429:
                if attempt < self._max_attempts:
                    await self._sleep(
                        retry_after if retry_after is not None else self._retry_delay(attempt)
                    )
                    continue
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.rate_limited",
                    retryable=True,
                    attempts=attempt,
                    status_code=status_code,
                )
            if status_code >= 500:
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt))
                    continue
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.unavailable",
                    retryable=True,
                    attempts=attempt,
                    status_code=status_code,
                )
            if status_code < 200 or status_code >= 300:
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.request_rejected",
                    retryable=False,
                    attempts=attempt,
                    status_code=status_code,
                )
            if body is None:  # pragma: no cover - successful _send_once always supplies bytes
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.invalid_response",
                    retryable=False,
                    attempts=attempt,
                )

            output, response_id, response_model = self._decode_success(body, attempt=attempt)
            try:
                output_sha256 = sha256(_canonical_json(output).encode("utf-8")).hexdigest()
                trace = OpenAIFactTrace(
                    fragment_id=request.fragment_id,
                    source_id=request.source_id,
                    content_sha256=request.content_sha256,
                    model=self._binding.model,
                    model_version=self._binding.model_version,
                    schema_generation_id=self._binding.schema_generation_id,
                    response_id_sha256=_digest_text(response_id),
                    response_model_sha256=_digest_text(response_model),
                    output_sha256=output_sha256,
                    attempts=attempt,
                )
            except (RecursionError, TypeError, ValueError, ValidationError):
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.invalid_response",
                    retryable=False,
                    attempts=attempt,
                ) from None
            try:
                await self._trace_sink.record(trace)
            except Exception:
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.trace_unavailable",
                    retryable=True,
                    attempts=attempt,
                ) from None
            return output

        raise AssertionError("bounded provider attempts exhausted")  # pragma: no cover

    async def _send_attempt(
        self,
        payload: dict[str, object],
        *,
        credential: str,
        attempt: int,
        idempotency_key: str,
    ) -> tuple[int, bytes | None, float | None]:
        coordinator = self._coordinator
        if coordinator is None:
            return await self._send_once(
                payload,
                credential=credential,
                attempt=attempt,
                idempotency_key=idempotency_key,
            )
        encoded_bytes = len(_canonical_json(payload).encode("utf-8"))
        estimated_tokens = min(
            coordinator.policy.tokens_per_minute,
            max(1, encoded_bytes // 3 + self._max_completion_tokens),
        )
        async with coordinator.slot(
            ProviderWorkload.BULK,
            estimated_tokens=estimated_tokens,
        ) as lease:
            try:
                result = await self._send_once(
                    payload,
                    credential=credential,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                )
            except (httpx.TimeoutException, httpx.HTTPError):
                await coordinator.record_failure(lease)
                raise
            status_code = result[0]
            if status_code == 429:
                await coordinator.record_429(ProviderWorkload.BULK, provider_wide=True)
            elif status_code >= 500:
                await coordinator.record_failure(lease)
            elif 200 <= status_code < 300:
                await coordinator.record_success(lease)
            return result

    def _validate_request(self, request: FactExtractorRequest) -> None:
        if not isinstance(request, FactExtractorRequest):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_request",
                retryable=False,
                attempts=0,
            )
        text = request.text
        try:
            if len(text) > self._max_input_bytes:
                raise ValueError("input too large")
            encoded = text.encode("utf-8")
            if (
                not encoded
                or len(encoded) > self._max_input_bytes
                or text.lstrip().startswith("%PDF-")
                or sha256(encoded).hexdigest() != request.content_sha256
                or _canonical_json(request.output_json_schema) != self._expected_schema_json
            ):
                raise ValueError("invalid extractor request")
        except (TypeError, UnicodeError, RecursionError, ValueError):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_request",
                retryable=False,
                attempts=0,
            ) from None

    def _credential(self) -> str:
        try:
            raw = self._api_key_provider()
            if not isinstance(raw, str):
                raise TypeError("credential must be text")
            credential = raw.strip()
            if (
                not credential
                or len(credential) > 16_384
                or "\n" in credential
                or "\r" in credential
                or "\x00" in credential
            ):
                raise ValueError("credential shape is invalid")
            return credential
        except Exception:
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.credential_unavailable",
                retryable=False,
                attempts=0,
            ) from None

    def _request_payload(self, request: FactExtractorRequest) -> dict[str, object]:
        messages = [
            {"role": "system", "content": self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": request.text},
        ]
        if self._binding.binding == "responses_json_schema":
            return {
                "model": self._binding.model,
                "input": messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": _SAFE_SCHEMA_NAME,
                        "strict": True,
                        "schema": deepcopy(self._expected_schema),
                    },
                },
                "max_output_tokens": self._max_completion_tokens,
                "stream": False,
            }
        return {
            "model": self._binding.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _SAFE_SCHEMA_NAME,
                    "strict": True,
                    "schema": deepcopy(self._expected_schema),
                },
            },
            "max_tokens": self._max_completion_tokens,
            "stream": False,
        }

    @staticmethod
    def _responses_text(
        envelope: dict[str, object],
        *,
        attempt: int,
    ) -> tuple[str, str, str]:
        status = envelope.get("status")
        if status in {"failed", "incomplete"} or envelope.get("incomplete_details") not in (
            None,
            "",
            [],
            {},
        ):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.incomplete",
                retryable=False,
                attempts=attempt,
            )
        if status != "completed":
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )
        if envelope.get("refusal") not in (None, "", [], {}):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.refused",
                retryable=False,
                attempts=attempt,
            )
        if envelope.get("error") not in (None, "", [], {}):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )

        response_id = envelope.get("id")
        response_model = envelope.get("model")
        if (
            not isinstance(response_id, str)
            or not response_id.strip()
            or len(response_id) > 500
            or not isinstance(response_model, str)
            or not response_model.strip()
            or len(response_model) > 500
        ):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )

        top_level_text = envelope.get("output_text")
        if top_level_text is not None and (
            not isinstance(top_level_text, str) or not top_level_text.strip()
        ):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )

        nested_parts: list[str] = []
        message_count = 0
        output = envelope.get("output")
        if output is not None:
            if not isinstance(output, list):
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.invalid_response",
                    retryable=False,
                    attempts=attempt,
                )
            for item in output:
                if not isinstance(item, dict):
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.invalid_response",
                        retryable=False,
                        attempts=attempt,
                    )
                item_type = item.get("type")
                if item_type == "reasoning":
                    continue
                if item_type in {"function_call", "tool_call", "computer_call"}:
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.tool_calls_forbidden",
                        retryable=False,
                        attempts=attempt,
                    )
                if item_type != "message":
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.invalid_response",
                        retryable=False,
                        attempts=attempt,
                    )
                message_count += 1
                if (
                    message_count > 1
                    or item.get("status") != "completed"
                    or item.get("role") != "assistant"
                ):
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.invalid_response",
                        retryable=False,
                        attempts=attempt,
                    )
                content = item.get("content")
                if not isinstance(content, list) or not content:
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.invalid_response",
                        retryable=False,
                        attempts=attempt,
                    )
                for part in content:
                    if not isinstance(part, dict):
                        raise OpenAIFactExtractorError(
                            "openai_fact_extractor.invalid_response",
                            retryable=False,
                            attempts=attempt,
                        )
                    part_type = part.get("type")
                    if part_type == "refusal":
                        raise OpenAIFactExtractorError(
                            "openai_fact_extractor.refused",
                            retryable=False,
                            attempts=attempt,
                        )
                    if part_type != "output_text":
                        raise OpenAIFactExtractorError(
                            "openai_fact_extractor.invalid_response",
                            retryable=False,
                            attempts=attempt,
                        )
                    text = part.get("text")
                    if not isinstance(text, str) or not text:
                        raise OpenAIFactExtractorError(
                            "openai_fact_extractor.invalid_response",
                            retryable=False,
                            attempts=attempt,
                        )
                    nested_parts.append(text)

        nested_text = "".join(nested_parts) if nested_parts else None
        if (
            isinstance(top_level_text, str)
            and nested_text is not None
            and top_level_text != nested_text
        ):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )
        content = nested_text if nested_text is not None else top_level_text
        if not isinstance(content, str):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            )
        return content.strip(), response_id.strip(), response_model.strip()

    async def _send_once(
        self,
        payload: dict[str, object],
        *,
        credential: str,
        attempt: int,
        idempotency_key: str,
    ) -> tuple[int, bytes | None, float | None]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
            "User-Agent": "ai4s-material-graph-studio/fact-extractor-v1",
        }
        async with self._http.stream(
            "POST",
            self._endpoint,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        ) as response:
            status_code = response.status_code
            retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
            if status_code < 200 or status_code >= 300:
                return status_code, None, retry_after

            declared_length = response.headers.get("Content-Length")
            if declared_length is not None:
                try:
                    parsed_length = int(declared_length)
                except ValueError:
                    parsed_length = -1
                if parsed_length < 0:
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.invalid_response",
                        retryable=False,
                        attempts=attempt,
                    )
                if parsed_length > self._max_response_bytes:
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.response_too_large",
                        retryable=False,
                        attempts=attempt,
                    )

            retained = bytearray()
            async for chunk in response.aiter_bytes():
                if len(chunk) > self._max_response_bytes - len(retained):
                    raise OpenAIFactExtractorError(
                        "openai_fact_extractor.response_too_large",
                        retryable=False,
                        attempts=attempt,
                    )
                retained.extend(chunk)
            return status_code, bytes(retained), retry_after

    def _decode_success(
        self,
        body: bytes,
        *,
        attempt: int,
    ) -> tuple[dict[str, object], str, str]:
        try:
            envelope = _strict_json_object(body)
        except _InvalidResponse:
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            ) from None

        if self._binding.binding == "responses_json_schema":
            content, response_id, response_model = self._responses_text(
                envelope,
                attempt=attempt,
            )
            if "```" in content:
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.markdown_forbidden",
                    retryable=False,
                    attempts=attempt,
                )
            try:
                output = _strict_json_object(content)
            except _InvalidResponse:
                raise OpenAIFactExtractorError(
                    "openai_fact_extractor.invalid_response",
                    retryable=False,
                    attempts=attempt,
                ) from None
            return output, response_id, response_model

        try:
            completion = _ChatCompletion.model_validate(envelope)
        except ValidationError:
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            ) from None

        choice = completion.choices[0]
        if choice.finish_reason != "stop":
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.incomplete",
                retryable=False,
                attempts=attempt,
            )
        message = choice.message
        if message.refusal not in (None, "", [], {}):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.refused",
                retryable=False,
                attempts=attempt,
            )
        if message.tool_calls not in (None, [], {}):
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.tool_calls_forbidden",
                retryable=False,
                attempts=attempt,
            )

        content = message.content.strip()
        if "```" in content:
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.markdown_forbidden",
                retryable=False,
                attempts=attempt,
            )
        try:
            output = _strict_json_object(content)
        except _InvalidResponse:
            raise OpenAIFactExtractorError(
                "openai_fact_extractor.invalid_response",
                retryable=False,
                attempts=attempt,
            ) from None
        return output, completion.id, completion.model

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self._retry_max_seconds,
            self._retry_base_seconds * (2 ** (attempt - 1)),
        )

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            parsed_seconds = float(value)
            return parsed_seconds if math.isfinite(parsed_seconds) and parsed_seconds >= 0 else None
        except ValueError:
            try:
                parsed_date = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed_date - datetime.now(timezone.utc)).total_seconds())


__all__ = [
    "FileOpenAIAPIKeyProvider",
    "InMemoryOpenAIFactTraceSink",
    "OpenAIAPIKeyProvider",
    "OpenAIFactExtractor",
    "OpenAIFactExtractorBinding",
    "OpenAIFactExtractorConfigurationError",
    "OpenAIFactExtractorError",
    "OpenAIFactExtractorErrorCode",
    "OpenAIFactTrace",
    "OpenAIFactTraceConflict",
    "OpenAIFactTraceSink",
    "OpenAITimeoutBinding",
    "build_openai_fact_idempotency_key",
]
