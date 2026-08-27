"""Resumable local LightRAG indexing for prepared textbook fragments."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .bindings import EmbeddingBinding
from .models import EvidenceFragment


MATERIAL_ENTITY_TYPES_GUIDANCE = """Classify every entity using exactly one type below.
Prefer scientifically specific entities that support materials research and retrieval.
Do not turn vague prose, isolated numbers, section titles, or citation boilerplate into entities.

- Material: Polymers, fibers, composites, ceramics, metals, additives, solvents, reagents, and named substances
- Structure: Molecular, crystalline, phase, morphological, interfacial, and hierarchical structures
- Process: Synthesis, polymerization, spinning, molding, heat treatment, surface treatment, and preparation operations
- ProcessCondition: Temperature, pressure, time, concentration, ratio, atmosphere, speed, draw ratio, and process settings
- Property: Mechanical, thermal, rheological, electrical, optical, chemical, barrier, aging, and material properties
- TestMethod: Characterization instruments, standards, test protocols, calculation methods, and experimental procedures
- Mechanism: Reactions, transport, crystallization, degradation, reinforcement, failure, and structure-property mechanisms
- Equipment: Reactors, extruders, spinning lines, furnaces, testing instruments, and other equipment
- Application: Products, use scenarios, engineering functions, and target performance requirements
- Standard: Standards, specifications, grades, units, and formal classification systems
- Organization: Universities, institutes, companies, laboratories, and standards bodies
- Data: Named datasets, tables, curves, equations, models, and quantitative result collections
- Concept: Scientific theories, principles, definitions, and abstract domain concepts
- Other: A meaningful entity that fits none of the types above"""

DEFAULT_EMBEDDING_KEY_ENV = "MATERIAL_GRAPH_EMBEDDING_API_KEY"
DEFAULT_WORKSPACE = "glm-embedding-3-1024-halfvec-v1"
_SAFE_FILE_COMPONENT = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff._-]+")
_TERMINAL_STATUSES = frozenset({"processed", "failed"})


class TextbookLightRAGError(RuntimeError):
    """Fail-closed indexing error whose message never contains a credential."""


class TextbookLLMProvider(BaseModel):
    """One non-secret OpenAI-compatible provider lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]*$")
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    credential_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    max_async: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class TextbookLLMPoolBinding(BaseModel):
    """Versioned provider pool with a stable cache generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    generation_id: str = Field(min_length=1)
    providers: tuple[TextbookLLMProvider, ...] = Field(min_length=1)

    @property
    def total_concurrency(self) -> int:
        return sum(provider.max_async for provider in self.providers)


@dataclass(frozen=True, slots=True)
class LocalTextbookLightRAGSettings:
    """Validated local indexing controls."""

    fragments_path: Path
    working_dir: Path
    embedding_binding_path: Path
    llm_pool_binding_path: Path
    workspace: str = DEFAULT_WORKSPACE
    batch_size: int = 256
    limit: int | None = None
    embedding_concurrency: int = 16
    insert_concurrency: int = 88
    chunk_token_size: int = 6_144
    chunk_overlap_token_size: int = 0
    entity_extract_max_gleaning: int = 0
    entity_extraction_use_json: bool = True

    def __post_init__(self) -> None:
        if not self.fragments_path.is_file():
            raise ValueError("prepared fragment file is unavailable")
        if not self.embedding_binding_path.is_file() or not self.llm_pool_binding_path.is_file():
            raise ValueError("provider binding file is unavailable")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.workspace):
            raise ValueError(
                "workspace must contain only letters, digits, underscores, and hyphens"
            )
        for field_name in (
            "batch_size",
            "embedding_concurrency",
            "insert_concurrency",
            "chunk_token_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.chunk_overlap_token_size < 0:
            raise ValueError("chunk overlap cannot be negative")
        if self.chunk_overlap_token_size >= self.chunk_token_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        if self.entity_extract_max_gleaning < 0:
            raise ValueError("entity extraction gleaning cannot be negative")

    @property
    def runtime_dir(self) -> Path:
        return self.working_dir / self.workspace

    @property
    def state_path(self) -> Path:
        return self.runtime_dir / "textbook-index-state.json"

    @property
    def provider_audit_path(self) -> Path:
        return self.runtime_dir / "provider-calls.jsonl"


@dataclass(frozen=True, slots=True)
class TextbookLightRAGDocument:
    """One prepared fragment represented as a uniquely citable document."""

    document_id: str
    text: str
    file_path: str
    fragment: EvidenceFragment


@dataclass(frozen=True, slots=True)
class LocalTextbookIndexSummary:
    """Safe result counters for one resumable indexing pass."""

    total_fragments_seen: int
    submitted_fragments: int
    existing_fragments: int
    processed_fragments: int
    failed_fragments: int
    pending_fragments: int
    batches_completed: int
    elapsed_seconds: float
    status: str
    status_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class LLMProviderResult:
    """One successful provider response with safe provenance."""

    text: str
    provider_id: str
    model: str
    elapsed_seconds: float


class _DocStatusStorage(Protocol):
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]: ...

    async def get_status_counts(self) -> dict[str, int]: ...


class _LightRAGLike(Protocol):
    doc_status: _DocStatusStorage

    async def initialize_storages(self) -> None: ...

    async def finalize_storages(self) -> None: ...

    async def apipeline_process_enqueue_documents(self) -> None: ...

    async def ainsert(
        self,
        input: str | list[str],
        *,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str: ...


def load_embedding_binding(path: str | Path) -> EmbeddingBinding:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EmbeddingBinding.model_validate(payload)


def load_textbook_llm_pool(path: str | Path) -> TextbookLLMPoolBinding:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    binding = TextbookLLMPoolBinding.model_validate(payload)
    provider_ids = [provider.provider_id for provider in binding.providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise ValueError("provider IDs must be unique")
    return binding


def _safe_component(value: str, *, fallback: str, maximum: int = 64) -> str:
    normalized = _SAFE_FILE_COMPONENT.sub("_", value.strip()).strip("._-")
    return (normalized or fallback)[:maximum]


def fragment_document_id(fragment: EvidenceFragment) -> str:
    """Return a deterministic ID that LightRAG can safely resume."""

    return f"doc-textbook-{fragment.fragment_id.hex}"


def fragment_file_path(fragment: EvidenceFragment) -> str:
    """Build a unique relative citation path without exposing the local drive."""

    original = PurePosixPath(fragment.locator.relative_path)
    parent = PurePosixPath(*original.parts[:-1]) if len(original.parts) > 1 else PurePosixPath()
    title = _safe_component(
        str(fragment.metadata.get("logical_title") or original.stem),
        fallback="textbook",
    )
    chunk_index = int(fragment.metadata.get("chunk_index") or 0)
    page = fragment.locator.page or 0
    unique_name = f"{title}__p{page:04d}__c{chunk_index:05d}__{fragment.fragment_id.hex[:12]}.md"
    return (
        PurePosixPath("textbook_fragments")
        / _safe_component(fragment.locator.root_id, fallback="textbook")
        / parent
        / unique_name
    ).as_posix()


def parse_fragment_record(line: str) -> TextbookLightRAGDocument:
    fragment = EvidenceFragment.model_validate_json(line)
    return TextbookLightRAGDocument(
        document_id=fragment_document_id(fragment),
        text=fragment.text,
        file_path=fragment_file_path(fragment),
        fragment=fragment,
    )


def iter_textbook_document_batches(
    path: str | Path,
    *,
    batch_size: int,
    limit: int | None = None,
) -> Iterator[list[TextbookLightRAGDocument]]:
    """Stream validated fragments in bounded batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    batch: list[TextbookLightRAGDocument] = []
    seen = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if limit is not None and seen >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                batch.append(parse_fragment_record(line))
            except Exception as error:
                raise TextbookLightRAGError(
                    f"prepared fragment is invalid at line {line_number}"
                ) from error
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


class AsyncLLMProviderPool:
    """Capacity-weighted provider pool; the first free lane takes the next call."""

    def __init__(
        self,
        binding: TextbookLLMPoolBinding,
        *,
        environment: dict[str, str],
        audit_path: Path,
        completion_func: Callable[..., Any],
    ) -> None:
        self.binding = binding
        self._environment = environment
        self._audit_path = audit_path
        self._completion_func = completion_func
        self._available: asyncio.Queue[TextbookLLMProvider] = asyncio.Queue()
        self._audit_lock = asyncio.Lock()
        self._seed_available_lanes()

    def _seed_available_lanes(self) -> None:
        remaining = {
            provider.provider_id: provider.max_async for provider in self.binding.providers
        }
        while any(value > 0 for value in remaining.values()):
            for provider in self.binding.providers:
                if remaining[provider.provider_id] <= 0:
                    continue
                self._available.put_nowait(provider)
                remaining[provider.provider_id] -= 1

    def _credential(self, provider: TextbookLLMProvider) -> str:
        value = self._environment.get(provider.credential_env, "").strip()
        if not value:
            raise TextbookLightRAGError(
                f"required provider credential is missing: {provider.credential_env}"
            )
        return value

    async def _audit(
        self,
        *,
        provider: TextbookLLMProvider,
        prompt: str,
        elapsed_seconds: float,
        status: str,
        response_chars: int = 0,
        error_type: str | None = None,
    ) -> None:
        payload = {
            "elapsed_ms": round(elapsed_seconds * 1000),
            "error_type": error_type,
            "model": provider.model,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "provider_id": provider.provider_id,
            "response_chars": response_chars,
            "status": status,
            "timestamp_unix": time.time(),
        }
        async with self._audit_lock:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                stream.write("\n")

    async def complete_with_provenance(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMProviderResult:
        last_error: Exception | None = None
        attempts = min(3, len(self.binding.providers))
        for _ in range(attempts):
            provider = await self._available.get()
            started_at = time.perf_counter()
            try:
                call_kwargs = dict(kwargs)
                supplied_extra_body = call_kwargs.pop("extra_body", {})
                extra_body = dict(provider.extra_body)
                if isinstance(supplied_extra_body, dict):
                    extra_body.update(supplied_extra_body)
                if extra_body:
                    call_kwargs["extra_body"] = extra_body
                result = await self._completion_func(
                    provider.model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=provider.base_url,
                    api_key=self._credential(provider),
                    timeout=provider.timeout_seconds,
                    **call_kwargs,
                )
                text = str(result)
                await self._audit(
                    provider=provider,
                    prompt=prompt,
                    elapsed_seconds=time.perf_counter() - started_at,
                    status="succeeded",
                    response_chars=len(text),
                )
                return LLMProviderResult(
                    text=text,
                    provider_id=provider.provider_id,
                    model=provider.model,
                    elapsed_seconds=time.perf_counter() - started_at,
                )
            except Exception as error:
                last_error = error
                await self._audit(
                    provider=provider,
                    prompt=prompt,
                    elapsed_seconds=time.perf_counter() - started_at,
                    status="failed",
                    error_type=type(error).__name__,
                )
            finally:
                self._available.put_nowait(provider)
        raise TextbookLightRAGError("all selected LLM provider attempts failed") from last_error

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        result = await self.complete_with_provenance(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return result.text


def build_async_llm_provider_pool(
    binding_path: str | Path,
    *,
    audit_path: str | Path,
    environment: dict[str, str] | None = None,
) -> AsyncLLMProviderPool:
    """Build a provider pool from non-secret config and process credentials."""

    resolved_environment = dict(os.environ if environment is None else environment)
    binding = load_textbook_llm_pool(binding_path)
    for provider in binding.providers:
        if not resolved_environment.get(provider.credential_env, "").strip():
            raise TextbookLightRAGError(
                f"required provider credential is missing: {provider.credential_env}"
            )
    try:
        from lightrag.llm.openai import openai_complete_if_cache
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookLightRAGError("LightRAG runtime dependency is unavailable") from error
    return AsyncLLMProviderPool(
        binding,
        environment=resolved_environment,
        audit_path=Path(audit_path),
        completion_func=openai_complete_if_cache,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_state(
    settings: LocalTextbookLightRAGSettings,
    summary: LocalTextbookIndexSummary,
    *,
    input_digest: str,
    started_at_unix: float,
) -> None:
    state_path = settings.state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(f"{state_path.suffix}.tmp")
    embedding = load_embedding_binding(settings.embedding_binding_path)
    llm_pool = load_textbook_llm_pool(settings.llm_pool_binding_path)
    payload = {
        "schema_version": 1,
        "input": {
            "bytes": settings.fragments_path.stat().st_size,
            "sha256": input_digest,
        },
        "models": {
            "embedding_generation_id": embedding.generation_id,
            "llm_pool_generation_id": llm_pool.generation_id,
            "llm_models": [provider.model for provider in llm_pool.providers],
        },
        "runtime": {
            "batch_size": settings.batch_size,
            "embedding_concurrency": settings.embedding_concurrency,
            "entity_extract_max_gleaning": settings.entity_extract_max_gleaning,
            "entity_extraction_use_json": settings.entity_extraction_use_json,
            "insert_concurrency": settings.insert_concurrency,
            "llm_concurrency": llm_pool.total_concurrency,
            "workspace": settings.workspace,
        },
        "started_at_unix": started_at_unix,
        "updated_at_unix": time.time(),
        "summary": asdict(summary),
    }
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(state_path)


def build_local_lightrag(
    settings: LocalTextbookLightRAGSettings,
    *,
    environment: dict[str, str] | None = None,
) -> _LightRAGLike:
    """Construct pinned local LightRAG without persisting credentials."""

    resolved_environment = dict(os.environ if environment is None else environment)
    embedding_api_key = resolved_environment.get(DEFAULT_EMBEDDING_KEY_ENV, "").strip()
    if not embedding_api_key:
        raise TextbookLightRAGError(
            f"required provider credential is missing: {DEFAULT_EMBEDDING_KEY_ENV}"
        )
    embedding_binding = load_embedding_binding(settings.embedding_binding_path)
    llm_binding = load_textbook_llm_pool(settings.llm_pool_binding_path)

    os.environ.setdefault("EMBEDDING_USE_BASE64", "true")
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import EmbeddingFunc
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookLightRAGError("LightRAG runtime dependency is unavailable") from error

    document_prefix = (
        None
        if embedding_binding.document_prefix == "NO_PREFIX"
        else embedding_binding.document_prefix
    )
    embedding_callable = partial(
        openai_embed.func,
        model=embedding_binding.model,
        base_url=embedding_binding.base_url,
        api_key=embedding_api_key,
        query_prefix=embedding_binding.query_prefix,
        document_prefix=document_prefix,
        client_configs={"timeout": embedding_binding.timeout_seconds},
    )
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_binding.dimensions,
        func=embedding_callable,
        max_token_size=embedding_binding.max_input_tokens,
        send_dimensions=embedding_binding.send_dimensions,
        model_name=embedding_binding.model,
        supports_asymmetric=embedding_binding.asymmetric,
    )
    llm_pool = build_async_llm_provider_pool(
        settings.llm_pool_binding_path,
        audit_path=settings.provider_audit_path,
        environment=resolved_environment,
    )

    settings.working_dir.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(settings.working_dir),
        workspace=settings.workspace,
        embedding_func=embedding_func,
        embedding_batch_num=embedding_binding.batch_size,
        embedding_func_max_async=settings.embedding_concurrency,
        default_embedding_timeout=int(embedding_binding.timeout_seconds),
        llm_model_func=llm_pool.complete,
        llm_model_name=llm_binding.generation_id,
        llm_model_max_async=llm_binding.total_concurrency,
        default_llm_timeout=max(provider.timeout_seconds for provider in llm_binding.providers),
        entity_extraction_use_json=settings.entity_extraction_use_json,
        entity_extract_max_gleaning=settings.entity_extract_max_gleaning,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        chunk_token_size=settings.chunk_token_size,
        chunk_overlap_token_size=settings.chunk_overlap_token_size,
        max_parallel_insert=settings.insert_concurrency,
        max_parallel_parse_native=settings.insert_concurrency,
        max_parallel_analyze=settings.insert_concurrency,
        queue_size_insert=max(128, settings.insert_concurrency * 4),
        force_llm_summary_on_merge=32,
        addon_params={
            "language": "Chinese",
            "entity_types_guidance": MATERIAL_ENTITY_TYPES_GUIDANCE,
        },
        auto_manage_storages_states=False,
    )


async def _new_documents(
    rag: _LightRAGLike,
    batch: Sequence[TextbookLightRAGDocument],
) -> tuple[list[TextbookLightRAGDocument], int]:
    records = await rag.doc_status.get_by_ids([item.document_id for item in batch])
    selected = [item for item, record in zip(batch, records, strict=True) if record is None]
    return selected, len(batch) - len(selected)


def _summary_from_counts(
    *,
    total_seen: int,
    submitted: int,
    existing: int,
    batches_completed: int,
    started_at: float,
    status_counts: dict[str, int],
    status: str,
) -> LocalTextbookIndexSummary:
    pending = sum(count for name, count in status_counts.items() if name not in _TERMINAL_STATUSES)
    return LocalTextbookIndexSummary(
        total_fragments_seen=total_seen,
        submitted_fragments=submitted,
        existing_fragments=existing,
        processed_fragments=status_counts.get("processed", 0),
        failed_fragments=status_counts.get("failed", 0),
        pending_fragments=pending,
        batches_completed=batches_completed,
        elapsed_seconds=round(time.time() - started_at, 3),
        status=status,
        status_counts=status_counts,
    )


async def index_local_textbook_fragments(
    settings: LocalTextbookLightRAGSettings,
    *,
    rag_factory: Callable[[LocalTextbookLightRAGSettings], _LightRAGLike] = build_local_lightrag,
    progress_callback: Callable[[LocalTextbookIndexSummary], Any] | None = None,
) -> LocalTextbookIndexSummary:
    """Index all prepared fragments locally, resuming from doc status."""

    started_at = time.time()
    input_digest = _file_sha256(settings.fragments_path)
    total_seen = 0
    submitted = 0
    existing = 0
    batches_completed = 0
    rag = rag_factory(settings)
    await rag.initialize_storages()
    try:
        await rag.apipeline_process_enqueue_documents()
        for batch in iter_textbook_document_batches(
            settings.fragments_path,
            batch_size=settings.batch_size,
            limit=settings.limit,
        ):
            total_seen += len(batch)
            new_documents, already_present = await _new_documents(rag, batch)
            existing += already_present
            if new_documents:
                await rag.ainsert(
                    [item.text for item in new_documents],
                    ids=[item.document_id for item in new_documents],
                    file_paths=[item.file_path for item in new_documents],
                    track_id=f"textbook-{batches_completed + 1:06d}",
                )
                submitted += len(new_documents)
            batches_completed += 1

            running = _summary_from_counts(
                total_seen=total_seen,
                submitted=submitted,
                existing=existing,
                batches_completed=batches_completed,
                started_at=started_at,
                status_counts=await rag.doc_status.get_status_counts(),
                status="running",
            )
            _write_state(
                settings,
                running,
                input_digest=input_digest,
                started_at_unix=started_at,
            )
            if progress_callback is not None:
                callback_result = progress_callback(running)
                if asyncio.iscoroutine(callback_result):
                    await callback_result

        counts = await rag.doc_status.get_status_counts()
        pending = sum(count for name, count in counts.items() if name not in _TERMINAL_STATUSES)
        status = (
            "completed"
            if counts.get("failed", 0) == 0 and pending == 0
            else "completed_with_failures"
        )
        final = _summary_from_counts(
            total_seen=total_seen,
            submitted=submitted,
            existing=existing,
            batches_completed=batches_completed,
            started_at=started_at,
            status_counts=counts,
            status=status,
        )
        _write_state(
            settings,
            final,
            input_digest=input_digest,
            started_at_unix=started_at,
        )
        if progress_callback is not None:
            callback_result = progress_callback(final)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        return final
    finally:
        await rag.finalize_storages()
