"""Safe, resumable extraction of reviewable graph facts from retained evidence.

The provider sees one bounded ``EvidenceFragment`` text plus a strict JSON
schema. It never controls evidence identifiers, source locations, extraction
provenance, batch identifiers, or graph writes. Successful output is a
``FactBatch`` explicitly marked pending review; a separate approved workflow
must write it to the global knowledge graph.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Awaitable, Callable, Literal, Mapping, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from .facts import (
    AssertionStatus,
    EntityRef,
    EvidenceLink,
    EvidenceQuality,
    EvidenceRole,
    ExtractionProvenance,
    FactBatch,
    ProcessCondition,
    PropertyObservation,
    RelationAssertion,
    SubjectRole,
    TestCondition,
    validate_fact_batch,
)
from .models import EvidenceFragment, SourceLocator


FactExtractionErrorCode = Literal[
    "extraction.invalid_fragment",
    "extraction.fragment_too_large",
    "extraction.content_drift",
    "extraction.invalid_provider_output",
    "extraction.unsafe_provider_output",
    "extraction.provider_unavailable",
    "extraction.provider_rejected",
    "extraction.retry_exhausted",
    "extraction.checkpoint_conflict",
]
FactExtractionStatus = Literal["running", "retry_wait", "completed", "failed_permanent"]

_FACT_KEY = re.compile(r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
_SAFE_ERROR_CODE = re.compile(r"^extraction\.[a-z][a-z0-9_]*$")
_UNSAFE_OUTPUT_KEYS = frozenset(
    {
        "complete_parser_output",
        "content_list",
        "document_bytes",
        "document_text",
        "fragment_text",
        "full_markdown",
        "full_text",
        "mineru_output",
        "parser_output",
        "pdf",
        "pdf_bytes",
        "raw_document",
        "raw_markdown",
        "raw_pdf",
        "source_bytes",
        "source_text",
    }
)
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "device_id",
    "password",
    "secret",
    "session",
    "synotoken",
    "token",
)
_PRIVATE_KEY_PREFIX = "-----" + "BEGIN "
_PRIVATE_KEY_SUFFIX = "PRIVATE " + "KEY-----"
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(
        _PRIVATE_KEY_PREFIX + r"(?:RSA |EC |OPENSSH )?" + _PRIVATE_KEY_SUFFIX,
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:password|passwd|api[_ -]?key)\s*[:=]", re.IGNORECASE),
    re.compile(r"%PDF-\d", re.IGNORECASE),
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_fact_extraction_idempotency_key(
    fragment_id: UUID,
    extractor_generation_id: str,
) -> str:
    """Build the exact idempotency key expected by ``FactBatch``."""

    if not isinstance(fragment_id, UUID):
        raise TypeError("fragment_id must be a UUID")
    if not isinstance(extractor_generation_id, str):
        raise ValueError("extractor generation id is required")
    generation_id = " ".join(unicodedata.normalize("NFKC", extractor_generation_id).split())
    if not generation_id or any(unicodedata.category(char) == "Cc" for char in generation_id):
        raise ValueError("extractor generation id is required")
    if len(generation_id) > 300:
        raise ValueError("extractor generation id is too large")
    digest = _digest(
        {
            "evidence_fragment_id": str(fragment_id),
            "extractor_generation_id": generation_id,
        }
    )
    return f"fact-batch-idempotency:v1:{digest}"


class FactExtractionError(RuntimeError):
    """Content-free pipeline failure suitable for checkpoints and API traces."""

    def __init__(
        self,
        code: FactExtractionErrorCode,
        *,
        retryable: bool,
        attempts: int = 0,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.attempts = attempts


class FactExtractorProviderError(RuntimeError):
    """Provider adapter signal without accepting or retaining provider detail."""

    def __init__(self, *, retryable: bool = True) -> None:
        super().__init__("extractor_provider_failure")
        self.retryable = retryable


class FactExtractionCheckpointConflict(RuntimeError):
    """Stable conflict for divergent durable extraction state."""

    def __init__(self) -> None:
        super().__init__("extraction.checkpoint_conflict")


class FactExtractionPolicy(BaseModel):
    """Bounded retry and payload policy for one extraction pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_attempts: int = Field(default=3, ge=1, le=8)
    retry_base_seconds: float = Field(default=0.25, ge=0, le=60, allow_inf_nan=False)
    retry_max_seconds: float = Field(default=4.0, ge=0, le=300, allow_inf_nan=False)
    max_fragment_bytes: int = Field(default=131_072, ge=1, le=2_097_152)
    max_response_bytes: int = Field(default=262_144, ge=1024, le=2_097_152)
    max_json_depth: int = Field(default=24, ge=4, le=64)

    @model_validator(mode="after")
    def validate_retry_window(self) -> "FactExtractionPolicy":
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("retry maximum must not be lower than the base")
        return self


class FactExtractorRequest(BaseModel):
    """Transient, provider-neutral request without NAS paths or parser metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    fragment_id: UUID
    source_id: UUID
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, repr=False)
    source_uri: str = Field(pattern=r"^source://")
    evidence_anchor: dict[str, JsonValue]
    extraction: ExtractionProvenance
    output_json_schema: dict[str, Any]


class _ExtractorRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    subject: EntityRef
    predicate: str = Field(min_length=1, max_length=200)
    object: EntityRef
    test_conditions: tuple[TestCondition, ...] = Field(default=(), max_length=100)
    process_conditions: tuple[ProcessCondition, ...] = Field(default=(), max_length=100)
    evidence_role: EvidenceRole = "supports"
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_quality: EvidenceQuality
    assertion_status: AssertionStatus


class _ExtractorObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    subject: EntityRef
    subject_role: SubjectRole
    property_name: str = Field(min_length=1, max_length=300)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=100)
    test_method: str = Field(min_length=1, max_length=300)
    test_conditions: tuple[TestCondition, ...] = Field(min_length=1, max_length=100)
    process_conditions: tuple[ProcessCondition, ...] = Field(default=(), max_length=100)
    evidence_role: EvidenceRole = "supports"
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    evidence_quality: EvidenceQuality
    assertion_status: AssertionStatus


class ExtractorFactPayload(BaseModel):
    """Strict provider output; evidence and provenance are injected by the pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entities: tuple[EntityRef, ...] = Field(min_length=1, max_length=1000)
    relations: tuple[_ExtractorRelation, ...] = Field(default=(), max_length=5000)
    observations: tuple[_ExtractorObservation, ...] = Field(default=(), max_length=5000)

    @model_validator(mode="after")
    def require_facts(self) -> "ExtractorFactPayload":
        if not self.relations and not self.observations:
            raise ValueError("extractor_payload_requires_facts")
        return self


def export_extractor_payload_json_schema() -> dict[str, Any]:
    """Return an independent strict schema for provider structured output."""

    return deepcopy(ExtractorFactPayload.model_json_schema())


@runtime_checkable
class FactExtractor(Protocol):
    """Provider-neutral structured extractor implemented by model adapters."""

    async def extract(self, request: FactExtractorRequest) -> object: ...


class FactExtractionCheckpoint(BaseModel):
    """Credential- and source-text-free durable extraction task state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    idempotency_key: str = Field(pattern=r"^fact-batch-idempotency:v1:[0-9a-f]{64}$")
    fragment_id: UUID
    source_id: UUID
    fragment_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    extraction: ExtractionProvenance
    status: FactExtractionStatus
    attempts: int = Field(ge=1, le=8)
    batch: FactBatch | None = None
    last_error_code: str | None = Field(default=None, pattern=r"^extraction\.[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def validate_state(self) -> "FactExtractionCheckpoint":
        expected_key = build_fact_extraction_idempotency_key(
            self.fragment_id,
            self.extraction.generation_id,
        )
        if self.idempotency_key != expected_key:
            raise ValueError("checkpoint_idempotency_mismatch")
        if self.status == "completed":
            if self.batch is None or self.last_error_code is not None:
                raise ValueError("completed_checkpoint_requires_batch")
            if (
                self.batch.idempotency_key != self.idempotency_key
                or self.batch.evidence_fragment_id != self.fragment_id
                or self.batch.extraction != self.extraction
            ):
                raise ValueError("checkpoint_batch_mismatch")
            expected_path = f"fragments/{self.fragment_id}"
            facts = (*self.batch.relations, *self.batch.observations)
            for fact in facts:
                if len(fact.evidence) != 1:
                    raise ValueError("checkpoint_evidence_mismatch")
                link = fact.evidence[0]
                if (
                    link.fragment_id != self.fragment_id
                    or link.source_id != self.source_id
                    or link.locator.relative_path != expected_path
                ):
                    raise ValueError("checkpoint_evidence_mismatch")
        elif self.batch is not None:
            raise ValueError("unfinished_checkpoint_cannot_store_batch")

        if self.status in {"retry_wait", "failed_permanent"}:
            if self.last_error_code is None:
                raise ValueError("failed_checkpoint_requires_error_code")
        elif self.last_error_code is not None:
            raise ValueError("running_checkpoint_cannot_store_error")
        return self


class FactExtractionCheckpointRepository(Protocol):
    """Minimal durable checkpoint contract for extraction recovery."""

    async def load(self, idempotency_key: str) -> FactExtractionCheckpoint | None: ...

    async def save(self, checkpoint: FactExtractionCheckpoint) -> FactExtractionCheckpoint: ...


def _copy_checkpoint(checkpoint: FactExtractionCheckpoint) -> FactExtractionCheckpoint:
    return FactExtractionCheckpoint.model_validate(checkpoint.model_dump(mode="python"))


def _checkpoint_identity(checkpoint: FactExtractionCheckpoint) -> tuple[object, ...]:
    return (
        checkpoint.idempotency_key,
        checkpoint.fragment_id,
        checkpoint.source_id,
        checkpoint.fragment_content_sha256,
        checkpoint.request_fingerprint,
        checkpoint.extraction,
    )


def _valid_transition(
    existing: FactExtractionCheckpoint,
    candidate: FactExtractionCheckpoint,
) -> bool:
    if existing == candidate:
        return True
    if existing.status in {"completed", "failed_permanent"}:
        return False
    if candidate.attempts < existing.attempts or candidate.attempts > existing.attempts + 1:
        return False
    if candidate.attempts == existing.attempts + 1:
        return candidate.status == "running" and existing.status in {"running", "retry_wait"}
    allowed_same_attempt = {
        "running": {"retry_wait", "completed", "failed_permanent"},
        "retry_wait": set(),
    }
    return candidate.status in allowed_same_attempt.get(existing.status, set())


class InMemoryFactExtractionCheckpointRepository:
    """Atomic in-memory checkpoint store with production-equivalent conflicts."""

    def __init__(self) -> None:
        self._items: dict[str, FactExtractionCheckpoint] = {}
        self._lock = asyncio.Lock()

    async def load(self, idempotency_key: str) -> FactExtractionCheckpoint | None:
        if not isinstance(idempotency_key, str) or _FACT_KEY.fullmatch(idempotency_key) is None:
            raise ValueError("invalid extraction checkpoint key")
        async with self._lock:
            stored = self._items.get(idempotency_key)
            return None if stored is None else _copy_checkpoint(stored)

    async def save(self, checkpoint: FactExtractionCheckpoint) -> FactExtractionCheckpoint:
        if not isinstance(checkpoint, FactExtractionCheckpoint):
            raise TypeError("checkpoint must be a FactExtractionCheckpoint")
        candidate = _copy_checkpoint(checkpoint)
        async with self._lock:
            existing = self._items.get(candidate.idempotency_key)
            if existing is not None:
                if _checkpoint_identity(existing) != _checkpoint_identity(candidate):
                    raise FactExtractionCheckpointConflict()
                if not _valid_transition(existing, candidate):
                    raise FactExtractionCheckpointConflict()
            self._items[candidate.idempotency_key] = candidate
            return _copy_checkpoint(candidate)


class FactExtractionResult(BaseModel):
    """Review-gated extraction result; it is not a graph-write acknowledgement."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    batch: FactBatch
    checkpoint: FactExtractionCheckpoint
    review_status: Literal["pending_review"] = "pending_review"
    resumed: bool
    provider_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> "FactExtractionResult":
        if self.checkpoint.status != "completed" or self.checkpoint.batch != self.batch:
            raise ValueError("extraction_result_checkpoint_mismatch")
        if self.resumed and self.provider_calls != 0:
            raise ValueError("resumed_result_cannot_claim_provider_calls")
        return self


@dataclass(frozen=True)
class _FragmentSnapshot:
    fragment: EvidenceFragment
    actual_content_sha256: str
    source_uri: str
    safe_locator: SourceLocator
    request_fingerprint: str


class _InvalidProviderOutput(ValueError):
    pass


class _UnsafeProviderOutput(ValueError):
    pass


@dataclass
class _JsonBudget:
    remaining: int

    def consume_text(self, value: str) -> None:
        if len(value) > self.remaining:
            raise _InvalidProviderOutput()
        size = len(value.encode("utf-8"))
        if size > self.remaining:
            raise _InvalidProviderOutput()
        self.remaining -= size

    def consume_scalar(self) -> None:
        if self.remaining < 32:
            raise _InvalidProviderOutput()
        self.remaining -= 32


def _safe_locator(fragment: EvidenceFragment) -> SourceLocator:
    locator = fragment.locator
    return SourceLocator(
        root_id=locator.root_id,
        relative_path=f"fragments/{fragment.fragment_id}",
        page=locator.page,
        section=locator.section,
        table=locator.table,
        figure=locator.figure,
        block_index=locator.block_index,
    )


def _snapshot_fragment(
    fragment: EvidenceFragment,
    policy: FactExtractionPolicy,
) -> _FragmentSnapshot:
    if not isinstance(fragment, EvidenceFragment):
        raise FactExtractionError("extraction.invalid_fragment", retryable=False)
    try:
        validated = EvidenceFragment.model_validate(fragment.model_dump(mode="python"))
        actual_hash = sha256(validated.text.encode("utf-8")).hexdigest()
        if validated.content_sha256 != actual_hash:
            raise ValueError("fragment content hash mismatch")
        if len(validated.text) > policy.max_fragment_bytes:
            raise FactExtractionError("extraction.fragment_too_large", retryable=False)
        if len(validated.text.encode("utf-8")) > policy.max_fragment_bytes:
            raise FactExtractionError("extraction.fragment_too_large", retryable=False)
        safe_locator = _safe_locator(validated)
        EvidenceLink(
            fragment_id=validated.fragment_id,
            source_id=validated.source_id,
            locator=safe_locator,
        )
        source_uri = safe_locator.to_public_uri(validated.source_id)
        request_fingerprint = _digest(
            {
                "fragment_id": str(validated.fragment_id),
                "source_id": str(validated.source_id),
                "content_sha256": actual_hash,
                "source_uri": source_uri,
                "parser_name": validated.parser_name,
                "parser_version": validated.parser_version,
                "embedding_generation_id": validated.embedding_generation_id,
            }
        )
    except FactExtractionError:
        raise
    except (TypeError, ValueError, ValidationError):
        raise FactExtractionError("extraction.invalid_fragment", retryable=False) from None
    return _FragmentSnapshot(
        fragment=validated,
        actual_content_sha256=actual_hash,
        source_uri=source_uri,
        safe_locator=safe_locator,
        request_fingerprint=request_fingerprint,
    )


def _normalized_key(key: str) -> str:
    return key.strip().casefold().replace("-", "_").replace(" ", "_")


def _walk_provider_json(
    value: object,
    *,
    max_depth: int,
    depth: int = 0,
    active: set[int] | None = None,
    budget: _JsonBudget,
) -> None:
    if depth > max_depth:
        raise _InvalidProviderOutput()
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            budget.consume_text(value)
            if any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS):
                raise _UnsafeProviderOutput()
        else:
            budget.consume_scalar()
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidProviderOutput()
        budget.consume_scalar()
        return

    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise _InvalidProviderOutput()
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise _InvalidProviderOutput()
                budget.consume_text(raw_key)
                key = _normalized_key(raw_key)
                if key in _UNSAFE_OUTPUT_KEYS or any(
                    marker in key for marker in _SENSITIVE_KEY_MARKERS
                ):
                    raise _UnsafeProviderOutput()
                _walk_provider_json(
                    item,
                    max_depth=max_depth,
                    depth=depth + 1,
                    active=active,
                    budget=budget,
                )
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk_provider_json(
                    item,
                    max_depth=max_depth,
                    depth=depth + 1,
                    active=active,
                    budget=budget,
                )
            return
        raise _InvalidProviderOutput()
    finally:
        active.remove(identity)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidProviderOutput()
        result[key] = value
    return result


def _reject_nonfinite_json(_value: str) -> object:
    raise _InvalidProviderOutput()


def _decode_provider_output(
    raw: object,
    *,
    fragment_text: str,
    policy: FactExtractionPolicy,
) -> ExtractorFactPayload:
    try:
        if isinstance(raw, bytes):
            if raw.lstrip().startswith(b"%PDF-"):
                raise _UnsafeProviderOutput()
            if len(raw) > policy.max_response_bytes:
                raise _InvalidProviderOutput()
            rendered = raw.decode("utf-8", errors="strict")
            decoded = json.loads(
                rendered,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        elif isinstance(raw, str):
            if len(raw) > policy.max_response_bytes:
                raise _InvalidProviderOutput()
            if len(raw.encode("utf-8")) > policy.max_response_bytes:
                raise _InvalidProviderOutput()
            rendered = raw
            decoded = json.loads(
                rendered,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        elif isinstance(raw, Mapping):
            _walk_provider_json(
                raw,
                max_depth=policy.max_json_depth,
                budget=_JsonBudget(policy.max_response_bytes),
            )
            rendered = json.dumps(
                raw,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(rendered.encode("utf-8")) > policy.max_response_bytes:
                raise _InvalidProviderOutput()
            decoded = json.loads(
                rendered,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        else:
            raise _InvalidProviderOutput()

        _walk_provider_json(
            decoded,
            max_depth=policy.max_json_depth,
            budget=_JsonBudget(policy.max_response_bytes),
        )
        if len(fragment_text) >= 128 and fragment_text in rendered:
            raise _UnsafeProviderOutput()
        if not isinstance(decoded, dict):
            raise _InvalidProviderOutput()
        return ExtractorFactPayload.model_validate(decoded)
    except _UnsafeProviderOutput:
        raise
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        ValidationError,
    ):
        raise _InvalidProviderOutput() from None


def _build_fact_batch(
    payload: ExtractorFactPayload,
    *,
    snapshot: _FragmentSnapshot,
    extraction: ExtractionProvenance,
) -> FactBatch:
    try:
        relations = tuple(
            RelationAssertion(
                **item.model_dump(mode="python", exclude={"evidence_role"}),
                evidence=(
                    EvidenceLink(
                        fragment_id=snapshot.fragment.fragment_id,
                        source_id=snapshot.fragment.source_id,
                        locator=snapshot.safe_locator,
                        role=item.evidence_role,
                    ),
                ),
                extraction=extraction,
            )
            for item in payload.relations
        )
        observations = tuple(
            PropertyObservation(
                **item.model_dump(mode="python", exclude={"evidence_role"}),
                evidence=(
                    EvidenceLink(
                        fragment_id=snapshot.fragment.fragment_id,
                        source_id=snapshot.fragment.source_id,
                        locator=snapshot.safe_locator,
                        role=item.evidence_role,
                    ),
                ),
                extraction=extraction,
            )
            for item in payload.observations
        )
        return validate_fact_batch(
            FactBatch(
                evidence_fragment_id=snapshot.fragment.fragment_id,
                extraction=extraction,
                entities=payload.entities,
                relations=relations,
                observations=observations,
            )
        )
    except (TypeError, ValueError, ValidationError):
        raise _InvalidProviderOutput() from None


def _assert_checkpoint_matches(
    checkpoint: FactExtractionCheckpoint,
    *,
    key: str,
    snapshot: _FragmentSnapshot,
    extraction: ExtractionProvenance,
) -> None:
    if checkpoint.idempotency_key != key:
        raise FactExtractionCheckpointConflict()
    if (
        checkpoint.fragment_id != snapshot.fragment.fragment_id
        or checkpoint.source_id != snapshot.fragment.source_id
        or checkpoint.fragment_content_sha256 != snapshot.actual_content_sha256
        or checkpoint.request_fingerprint != snapshot.request_fingerprint
        or checkpoint.extraction != extraction
    ):
        raise FactExtractionError("extraction.content_drift", retryable=False)
    if checkpoint.batch is None:
        return
    try:
        batch = validate_fact_batch(checkpoint.batch)
    except (TypeError, ValueError):
        raise FactExtractionCheckpointConflict() from None
    expected_uri = snapshot.source_uri
    facts = (*batch.relations, *batch.observations)
    if any(
        len(fact.evidence) != 1
        or fact.evidence[0].fragment_id != snapshot.fragment.fragment_id
        or fact.evidence[0].source_id != snapshot.fragment.source_id
        or fact.evidence[0].public_source_uri != expected_uri
        for fact in facts
    ):
        raise FactExtractionError("extraction.content_drift", retryable=False)


class EvidenceFactExtractionPipeline:
    """Convert one retained fragment into a reviewable, resumable fact batch."""

    def __init__(
        self,
        *,
        extractor: FactExtractor,
        extraction: ExtractionProvenance,
        checkpoints: FactExtractionCheckpointRepository,
        policy: FactExtractionPolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(extraction, ExtractionProvenance):
            raise TypeError("extraction must be ExtractionProvenance")
        self._extractor = extractor
        self._extraction = ExtractionProvenance.model_validate(extraction.model_dump(mode="python"))
        self._checkpoints = checkpoints
        self._policy = policy or FactExtractionPolicy()
        self._sleep = sleep

    async def extract(
        self,
        fragment: EvidenceFragment,
        *,
        resume_from: FactExtractionCheckpoint | None = None,
    ) -> FactExtractionResult:
        snapshot = _snapshot_fragment(fragment, self._policy)
        key = build_fact_extraction_idempotency_key(
            snapshot.fragment.fragment_id,
            self._extraction.generation_id,
        )
        persisted = await self._checkpoints.load(key)
        if resume_from is not None:
            if not isinstance(resume_from, FactExtractionCheckpoint):
                raise FactExtractionCheckpointConflict()
            try:
                supplied = _copy_checkpoint(resume_from)
            except (TypeError, ValueError, ValidationError):
                raise FactExtractionCheckpointConflict() from None
            _assert_checkpoint_matches(
                supplied,
                key=key,
                snapshot=snapshot,
                extraction=self._extraction,
            )
            if persisted is not None and persisted != supplied:
                raise FactExtractionCheckpointConflict()
            if persisted is None:
                persisted = await self._checkpoints.save(supplied)

        state = persisted
        if state is not None:
            _assert_checkpoint_matches(
                state,
                key=key,
                snapshot=snapshot,
                extraction=self._extraction,
            )
            if state.status == "completed":
                if state.batch is None:  # pragma: no cover - model validator enforces this
                    raise FactExtractionCheckpointConflict()
                return FactExtractionResult(
                    batch=state.batch,
                    checkpoint=state,
                    resumed=True,
                    provider_calls=0,
                )
            if state.status == "failed_permanent":
                code = self._error_code(state.last_error_code)
                raise FactExtractionError(code, retryable=False, attempts=state.attempts)

        next_attempt = 1 if state is None else state.attempts + 1
        if next_attempt > self._policy.max_attempts:
            code: FactExtractionErrorCode = (
                self._error_code(state.last_error_code)
                if state is not None and state.last_error_code is not None
                else "extraction.retry_exhausted"
            )
            terminal = self._checkpoint(
                snapshot=snapshot,
                key=key,
                status="failed_permanent",
                attempts=state.attempts if state is not None else self._policy.max_attempts,
                last_error_code=code,
            )
            await self._checkpoints.save(terminal)
            raise FactExtractionError(code, retryable=False, attempts=terminal.attempts)

        provider_calls = 0
        for attempt in range(next_attempt, self._policy.max_attempts + 1):
            self._assert_fragment_unchanged(fragment, snapshot)
            running = self._checkpoint(
                snapshot=snapshot,
                key=key,
                status="running",
                attempts=attempt,
            )
            state = await self._checkpoints.save(running)
            request = self._request(snapshot)
            provider_calls += 1
            try:
                raw = await self._extractor.extract(request)
                payload = _decode_provider_output(
                    raw,
                    fragment_text=snapshot.fragment.text,
                    policy=self._policy,
                )
                batch = _build_fact_batch(
                    payload,
                    snapshot=snapshot,
                    extraction=self._extraction,
                )
                if batch.idempotency_key != key:
                    raise _InvalidProviderOutput()
                self._assert_fragment_unchanged(fragment, snapshot)
            except FactExtractorProviderError as error:
                failure_code: FactExtractionErrorCode = (
                    "extraction.provider_unavailable"
                    if error.retryable
                    else "extraction.provider_rejected"
                )
                retryable = error.retryable
            except _UnsafeProviderOutput:
                failure_code = "extraction.unsafe_provider_output"
                retryable = False
            except _InvalidProviderOutput:
                failure_code = "extraction.invalid_provider_output"
                retryable = True
            except FactExtractionError as error:
                failure_code = error.code
                retryable = error.retryable
            except Exception:
                failure_code = "extraction.provider_unavailable"
                retryable = True
            else:
                completed = self._checkpoint(
                    snapshot=snapshot,
                    key=key,
                    status="completed",
                    attempts=attempt,
                    batch=batch,
                )
                state = await self._checkpoints.save(completed)
                if state.batch is None:  # pragma: no cover - model validator enforces this
                    raise FactExtractionCheckpointConflict()
                return FactExtractionResult(
                    batch=state.batch,
                    checkpoint=state,
                    resumed=False,
                    provider_calls=provider_calls,
                )

            if retryable and attempt < self._policy.max_attempts:
                retry_wait = self._checkpoint(
                    snapshot=snapshot,
                    key=key,
                    status="retry_wait",
                    attempts=attempt,
                    last_error_code=failure_code,
                )
                state = await self._checkpoints.save(retry_wait)
                await self._sleep(self._retry_delay(attempt))
                continue

            terminal = self._checkpoint(
                snapshot=snapshot,
                key=key,
                status="failed_permanent",
                attempts=attempt,
                last_error_code=failure_code,
            )
            await self._checkpoints.save(terminal)
            raise FactExtractionError(failure_code, retryable=False, attempts=attempt)

        raise AssertionError("bounded extraction attempts exhausted")  # pragma: no cover

    def _request(self, snapshot: _FragmentSnapshot) -> FactExtractorRequest:
        anchor = snapshot.safe_locator.model_dump(
            mode="json",
            exclude={"root_id", "relative_path"},
            exclude_none=True,
        )
        return FactExtractorRequest(
            fragment_id=snapshot.fragment.fragment_id,
            source_id=snapshot.fragment.source_id,
            content_sha256=snapshot.actual_content_sha256,
            text=snapshot.fragment.text,
            source_uri=snapshot.source_uri,
            evidence_anchor=anchor,
            extraction=self._extraction,
            output_json_schema=export_extractor_payload_json_schema(),
        )

    def _checkpoint(
        self,
        *,
        snapshot: _FragmentSnapshot,
        key: str,
        status: FactExtractionStatus,
        attempts: int,
        batch: FactBatch | None = None,
        last_error_code: FactExtractionErrorCode | None = None,
    ) -> FactExtractionCheckpoint:
        return FactExtractionCheckpoint(
            idempotency_key=key,
            fragment_id=snapshot.fragment.fragment_id,
            source_id=snapshot.fragment.source_id,
            fragment_content_sha256=snapshot.actual_content_sha256,
            request_fingerprint=snapshot.request_fingerprint,
            extraction=self._extraction,
            status=status,
            attempts=attempts,
            batch=batch,
            last_error_code=last_error_code,
        )

    def _assert_fragment_unchanged(
        self,
        fragment: EvidenceFragment,
        expected: _FragmentSnapshot,
    ) -> None:
        try:
            current = _snapshot_fragment(fragment, self._policy)
        except FactExtractionError:
            raise FactExtractionError("extraction.content_drift", retryable=False) from None
        if (
            current.fragment.fragment_id != expected.fragment.fragment_id
            or current.fragment.source_id != expected.fragment.source_id
            or current.actual_content_sha256 != expected.actual_content_sha256
            or current.request_fingerprint != expected.request_fingerprint
        ):
            raise FactExtractionError("extraction.content_drift", retryable=False)

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self._policy.retry_max_seconds,
            self._policy.retry_base_seconds * (2 ** (attempt - 1)),
        )

    @staticmethod
    def _error_code(value: str | None) -> FactExtractionErrorCode:
        allowed: set[str] = {
            "extraction.invalid_fragment",
            "extraction.fragment_too_large",
            "extraction.content_drift",
            "extraction.invalid_provider_output",
            "extraction.unsafe_provider_output",
            "extraction.provider_unavailable",
            "extraction.provider_rejected",
            "extraction.retry_exhausted",
            "extraction.checkpoint_conflict",
        }
        if value not in allowed or value is None or _SAFE_ERROR_CODE.fullmatch(value) is None:
            raise FactExtractionCheckpointConflict()
        return value  # type: ignore[return-value]


__all__ = [
    "EvidenceFactExtractionPipeline",
    "ExtractorFactPayload",
    "FactExtractionCheckpoint",
    "FactExtractionCheckpointConflict",
    "FactExtractionCheckpointRepository",
    "FactExtractionError",
    "FactExtractionErrorCode",
    "FactExtractionPolicy",
    "FactExtractionResult",
    "FactExtractor",
    "FactExtractorProviderError",
    "FactExtractorRequest",
    "InMemoryFactExtractionCheckpointRepository",
    "build_fact_extraction_idempotency_key",
    "export_extractor_payload_json_schema",
]
