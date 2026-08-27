"""Synchronous, bounded LightRAG query boundary for online graph retrieval."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import time

import httpx
from pydantic import ValidationError

from .lightrag_models import LightRAGQueryData, LightRAGQueryEnvelope, LightRAGQueryRequest


class LightRAGQueryError(RuntimeError):
    """Base error whose message is safe for logs, SSE, and audit records."""


class LightRAGQueryProtocolError(LightRAGQueryError):
    """The pinned LightRAG service returned data outside its public contract."""


class LightRAGQueryRequestError(LightRAGQueryError):
    """Bounded HTTP failure without remote response text or credentials."""

    def __init__(self, *, status_code: int | None, code: str) -> None:
        self.status_code = status_code
        self.code = code
        status = "transport" if status_code is None else str(status_code)
        super().__init__(f"LightRAG query failed ({status}): {code}")


APIKeyProvider = Callable[[], str]
Sleep = Callable[[float], None]


class LightRAGQueryClient:
    """Call only the official LightRAG v1.5.4 query-data route."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_provider: APIKeyProvider,
        http: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | float | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        sleep: Sleep = time.sleep,
    ) -> None:
        parsed = httpx.URL(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be a credential-free absolute HTTP(S) URL")
        if not callable(api_key_provider):
            raise TypeError("api_key_provider must be callable")
        if http is not None and transport is not None:
            raise ValueError("transport cannot be supplied with an existing HTTP client")
        if max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("retry settings cannot be negative")

        self._base_url = str(parsed).rstrip("/")
        self._api_key_provider = api_key_provider
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        self._owns_http = http is None
        self._http = http or httpx.Client(
            timeout=timeout or httpx.Timeout(30.0, connect=5.0),
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, max_retries={self._max_retries})"
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def query_data(self, query: str, *, top_k: int = 8) -> LightRAGQueryData:
        request = LightRAGQueryRequest.for_query(query, top_k=top_k)
        path = "/query/data"
        for attempt in range(self._max_retries + 1):
            credential = self._credential()
            try:
                response = self._http.post(
                    f"{self._base_url}{path}",
                    json=request.model_dump(mode="json"),
                    headers={"X-API-Key": credential},
                )
            except httpx.TransportError as error:
                if attempt >= self._max_retries:
                    raise LightRAGQueryRequestError(
                        status_code=None,
                        code=type(error).__name__,
                    ) from None
                self._sleep(self._retry_delay(attempt))
                continue
            finally:
                credential = ""

            if 200 <= response.status_code < 300:
                return self._decode_success(response)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= self._max_retries:
                    raise LightRAGQueryRequestError(
                        status_code=response.status_code,
                        code="retry_exhausted",
                    )
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                self._sleep(retry_after if retry_after is not None else self._retry_delay(attempt))
                continue
            raise LightRAGQueryRequestError(
                status_code=response.status_code,
                code="request_rejected",
            )
        raise AssertionError("bounded LightRAG query loop exhausted")

    def _decode_success(self, response: httpx.Response) -> LightRAGQueryData:
        try:
            payload = response.json()
        except ValueError:
            raise LightRAGQueryProtocolError("query_response_not_json") from None
        if not isinstance(payload, dict):
            raise LightRAGQueryProtocolError("query_response_not_object")
        try:
            envelope = LightRAGQueryEnvelope.model_validate(payload)
        except ValidationError:
            raise LightRAGQueryProtocolError("query_response_invalid") from None
        if envelope.status != "success":
            raise LightRAGQueryProtocolError("query_response_failed")
        metadata = {
            **envelope.data.metadata,
            **envelope.metadata,
        }
        return envelope.data.model_copy(
            update={"metadata": metadata},
        )

    def _credential(self) -> str:
        try:
            value = self._api_key_provider()
        except Exception:
            raise LightRAGQueryRequestError(
                status_code=None,
                code="credential_unavailable",
            ) from None
        if not isinstance(value, str):
            raise LightRAGQueryRequestError(
                status_code=None,
                code="credential_invalid",
            )
        credential = value.strip()
        if not credential or any(marker in credential for marker in ("\n", "\r", "\x00")):
            raise LightRAGQueryRequestError(
                status_code=None,
                code="credential_invalid",
            )
        return credential

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


__all__ = [
    "LightRAGQueryClient",
    "LightRAGQueryError",
    "LightRAGQueryProtocolError",
    "LightRAGQueryRequestError",
]
