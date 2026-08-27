"""Index a portable textbook custom-KG with strict embedding-generation failover."""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from .bindings import EmbeddingBinding
from .catalog import build_source_version_key
from .ingestion import build_ingestion_idempotency_key
from .lightrag_runtime import workspace_for_generation
from .lightrag_models import LightRAGSourceMapping, build_lightrag_basename
from .models import EvidenceFragment, SourceCatalogRecord, SourceLocator
from .textbook_deployment_import import build_derived_provenance_contract
from .textbook_lightrag import (
    DEFAULT_EMBEDDING_KEY_ENV,
    TextbookLightRAGError,
    iter_textbook_document_batches,
    load_embedding_binding,
)


_QUOTA_MARKERS = (
    "balance not enough",
    "insufficient balance",
    "insufficient_balance",
    "quota exhausted",
    "quota_exhausted",
    "余额不足",
    "额度耗尽",
    "30002",
)


class _FallbackProvider(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    base_url: str
    model_candidates: tuple[str, ...] = Field(min_length=1)
    credential_env: str
    dimensions: int
    generation_id: str
    verify_on_activation: bool = True


class _EmbeddingActivation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_reasons: tuple[str, ...]
    switch_at_checkpoint_boundary: bool
    require_full_reembedding: bool
    allow_mixed_generations: bool
    preflight_before_primary_exhaustion: bool


class _EmbeddingFailoverPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    strategy: str
    primary: dict[str, str]
    fallback: _FallbackProvider | None = None
    activation: _EmbeddingActivation


class _CustomKGRAG(Protocol):
    async def initialize_storages(self) -> None: ...

    async def finalize_storages(self) -> None: ...

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str | None = None,
    ) -> None: ...


GenerationRunner = Callable[
    [EmbeddingBinding, str, str, dict[str, Any], Path],
    Awaitable[None],
]
FallbackProbe = Callable[[EmbeddingBinding, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class TextbookCustomKGIndexSettings:
    custom_kg_path: Path
    fragments_path: Path
    working_dir: Path
    deployment_dir: Path
    primary_embedding_binding_path: Path
    failover_policy_path: Path
    state_path: Path

    def __post_init__(self) -> None:
        for path in (
            self.custom_kg_path,
            self.fragments_path,
            self.primary_embedding_binding_path,
            self.failover_policy_path,
        ):
            if not path.is_file():
                raise ValueError("custom-KG input or binding is unavailable")


@dataclass(frozen=True, slots=True)
class TextbookCustomKGIndexSummary:
    generation_id: str
    provider: str
    model: str
    deployment_bundle: str
    failover_activated: bool
    elapsed_seconds: float
    status: str


def _load_policy(path: Path) -> _EmbeddingFailoverPolicy:
    policy = _EmbeddingFailoverPolicy.model_validate_json(path.read_text(encoding="utf-8"))
    if policy.strategy not in {"quota_exhaustion_only", "disabled"}:
        raise ValueError("unsafe embedding failover policy")
    if (
        not policy.activation.switch_at_checkpoint_boundary
        or not policy.activation.require_full_reembedding
        or policy.activation.allow_mixed_generations
        or policy.activation.preflight_before_primary_exhaustion
    ):
        raise ValueError("unsafe embedding failover policy")
    if policy.strategy == "quota_exhaustion_only" and policy.fallback is None:
        raise ValueError("quota failover policy requires a fallback")
    if policy.strategy == "disabled" and policy.activation.allowed_reasons:
        raise ValueError("disabled embedding failover cannot allow activation reasons")
    return policy


def _safe_workspace(generation_id: str) -> str:
    workspace = re.sub(r"[^A-Za-z0-9_]+", "_", generation_id).strip("_")
    if not workspace:
        raise ValueError("embedding generation cannot form a workspace")
    return workspace


def _quota_exhausted(error: BaseException) -> bool:
    """Classify only terminal balance/quota errors, never transient rate limits."""

    parts: list[str] = []
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending and len(seen) < 32:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        parts.append(type(current).__name__)
        try:
            parts.append(str(current))
        except Exception:  # pragma: no cover - hostile provider exception
            pass
        for nested in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
        # tenacity's RetryError stores the provider exception on the last
        # attempt instead of exposing it through __cause__/__context__.
        last_attempt = getattr(current, "last_attempt", None)
        exception_getter = getattr(last_attempt, "exception", None)
        if callable(exception_getter):
            try:
                nested = exception_getter()
            except Exception:  # pragma: no cover - defensive provider boundary
                nested = None
            if isinstance(nested, BaseException):
                pending.append(nested)
    normalized = " ".join(parts).casefold()
    return any(marker.casefold() in normalized for marker in _QUOTA_MARKERS)


def _fallback_binding(
    primary: EmbeddingBinding,
    policy: _EmbeddingFailoverPolicy,
    model: str,
) -> EmbeddingBinding:
    return primary.model_copy(
        update={
            "provider": policy.fallback.provider,
            "base_url": policy.fallback.base_url,
            "model": model,
            "max_input_tokens": min(primary.max_input_tokens, 8_192),
            "batch_size": min(primary.batch_size, 10),
            "max_async": min(primary.max_async, 16),
            "generation_id": policy.fallback.generation_id,
        }
    )


def _source_locator(fragment: EvidenceFragment) -> SourceLocator:
    return SourceLocator(
        root_id=fragment.locator.root_id,
        relative_path=fragment.locator.relative_path,
    )


def _generation_fragment(
    fragment: EvidenceFragment,
    *,
    generation_id: str,
) -> tuple[EvidenceFragment, str, str]:
    source_locator = _source_locator(fragment)
    source_version_key = build_source_version_key(
        locator=source_locator,
        byte_size=None,
        remote_modified_at=None,
    )
    idempotency_key = build_ingestion_idempotency_key(
        fragment.source_id,
        source_version_key=source_version_key,
        embedding_generation_id=generation_id,
    )
    locator = fragment.locator
    identity = "|".join(
        (
            idempotency_key,
            fragment.content_sha256 or "",
            str(locator.page or 0),
            str(locator.block_index or 0),
            locator.section or "",
        )
    )
    return (
        fragment.model_copy(
            update={
                "fragment_id": uuid5(NAMESPACE_URL, identity),
                "embedding_generation_id": generation_id,
            },
            deep=True,
        ),
        source_version_key,
        idempotency_key,
    )


def _load_fragments(path: Path) -> dict[str, EvidenceFragment]:
    fragments: dict[str, EvidenceFragment] = {}
    for batch in iter_textbook_document_batches(path, batch_size=512):
        for document in batch:
            raw_id = str(document.fragment.fragment_id)
            if raw_id in fragments:
                raise TextbookLightRAGError("duplicate prepared fragment")
            fragments[raw_id] = document.fragment
    return fragments


def _bind_generation(
    custom_kg: dict[str, Any],
    source_fragments: dict[str, EvidenceFragment],
    binding: EmbeddingBinding,
) -> tuple[
    dict[str, Any],
    list[EvidenceFragment],
    list[LightRAGSourceMapping],
    list[SourceCatalogRecord],
    list[dict[str, Any]],
]:
    stable_by_raw_id: dict[str, EvidenceFragment] = {}
    source_versions: dict[UUID, tuple[str, str]] = {}
    for raw_id, fragment in source_fragments.items():
        stable, version_key, idempotency_key = _generation_fragment(
            fragment,
            generation_id=binding.generation_id,
        )
        stable_by_raw_id[raw_id] = stable
        existing = source_versions.get(fragment.source_id)
        candidate = (version_key, idempotency_key)
        if existing is not None and existing != candidate:
            raise TextbookLightRAGError("one source resolved to multiple versions")
        source_versions[fragment.source_id] = candidate

    def rebound(item: object) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise TextbookLightRAGError("custom-KG member must be an object")
        raw_id = str(item.get("source_id") or "")
        fragment = stable_by_raw_id.get(raw_id)
        if fragment is None:
            raise TextbookLightRAGError("custom-KG references an unknown fragment")
        return {
            **item,
            "source_id": str(fragment.fragment_id),
            "file_path": build_lightrag_basename(fragment),
        }

    rebound_kg = {
        "chunks": [rebound(item) for item in custom_kg.get("chunks", [])],
        "entities": [rebound(item) for item in custom_kg.get("entities", [])],
        "relationships": [rebound(item) for item in custom_kg.get("relationships", [])],
    }
    stable_fragments = sorted(
        stable_by_raw_id.values(),
        key=lambda item: item.fragment_id.hex,
    )
    mappings = [LightRAGSourceMapping.from_fragment(item) for item in stable_fragments]

    sources_by_id: dict[UUID, SourceCatalogRecord] = {}
    for fragment in stable_fragments:
        if fragment.source_id in sources_by_id:
            continue
        source_version_key, _ = source_versions[fragment.source_id]
        sources_by_id[fragment.source_id] = SourceCatalogRecord(
            source_id=fragment.source_id,
            locator=_source_locator(fragment),
            source_kind="textbook",
            display_title=str(fragment.metadata.get("logical_title") or "教材"),
            status="indexed",
            sha256=str(fragment.metadata.get("document_content_sha256") or "") or None,
            byte_size=None,
            knowledge_domain="materials_textbook",
            metadata={
                "source_family": fragment.metadata.get("source_family"),
                "part_number": fragment.metadata.get("part_number"),
                "source_version_key": source_version_key,
            },
        )
    checkpoints: list[dict[str, Any]] = []
    fragment_counts = Counter(item.source_id for item in stable_fragments)
    for source_id in sorted(sources_by_id, key=lambda item: item.hex):
        source_version_key, idempotency_key = source_versions[source_id]
        fingerprint = source_version_key.rsplit(":", 1)[-1]
        checkpoints.append(
            {
                "idempotency_key": idempotency_key,
                "source_id": str(source_id),
                "lifecycle_status": "evidence_retained",
                "stage": "index",
                "job_status": "succeeded",
                "attempt": 1,
                "selection": {
                    "source_id": str(source_id),
                    "selected": True,
                    "reason_code": "approved_curation",
                    "task_id": None,
                    "evidence_gap_id": None,
                    "rank": 1,
                    "policy_version": "textbook-full-corpus-v1",
                },
                "cursor": {},
                "metadata": {
                    "embedding_generation_id": binding.generation_id,
                    "fragment_count": fragment_counts[source_id],
                    "index_completed": True,
                    "index_outcome": "processed",
                    "source_version_fingerprint": fingerprint,
                },
                "last_error_category": None,
            }
        )
    return (
        rebound_kg,
        stable_fragments,
        mappings,
        [sources_by_id[key] for key in sorted(sources_by_id, key=lambda item: item.hex)],
        checkpoints,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, values: list[object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_deployment_bundle(
    settings: TextbookCustomKGIndexSettings,
    binding: EmbeddingBinding,
    custom_kg: dict[str, Any],
    fragments: list[EvidenceFragment],
    mappings: list[LightRAGSourceMapping],
    sources: list[SourceCatalogRecord],
    checkpoints: list[dict[str, Any]],
) -> Path:
    bundle_dir = settings.deployment_dir / _safe_workspace(binding.generation_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "sources": bundle_dir / "sources.jsonl",
        "checkpoints": bundle_dir / "checkpoints.jsonl",
        "source_mappings": bundle_dir / "source-mappings.jsonl",
        "embedding_binding": bundle_dir / "embedding-binding.json",
        "custom_kg_chunks": bundle_dir / "custom-kg-chunks.jsonl",
        "custom_kg_entities": bundle_dir / "custom-kg-entities.jsonl",
        "custom_kg_relationships": bundle_dir / "custom-kg-relationships.jsonl",
    }
    _write_jsonl(artifacts["sources"], list(sources))
    _write_jsonl(artifacts["checkpoints"], list(checkpoints))
    _write_jsonl(artifacts["source_mappings"], list(mappings))
    artifacts["embedding_binding"].write_text(
        binding.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(artifacts["custom_kg_chunks"], list(custom_kg["chunks"]))
    _write_jsonl(
        artifacts["custom_kg_entities"],
        sorted(
            custom_kg["entities"],
            key=lambda row: (str(row["source_id"]), str(row["entity_name"])),
        ),
    )
    _write_jsonl(
        artifacts["custom_kg_relationships"],
        sorted(
            custom_kg["relationships"],
            key=lambda row: (
                str(row["source_id"]),
                str(row["src_id"]),
                str(row["tgt_id"]),
            ),
        ),
    )
    counts = {
        "sources": len(sources),
        "checkpoints": len(checkpoints),
        "evidence_fragments": len(fragments),
        "source_mappings": len(mappings),
        "chunks": len(custom_kg["chunks"]),
        "entities": len(custom_kg["entities"]),
        "relationships": len(custom_kg["relationships"]),
        "custom_kg_chunks": len(custom_kg["chunks"]),
        "custom_kg_entities": len(custom_kg["entities"]),
        "custom_kg_relationships": len(custom_kg["relationships"]),
    }
    artifact_records = {
        name: {
            "path": path.name,
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in artifacts.items()
    }
    manifest = {
        "schema_version": 2,
        "embedding": {
            "generation_id": binding.generation_id,
            "provider": binding.provider,
            "model": binding.model,
            "dimensions": binding.dimensions,
        },
        "provenance_import": build_derived_provenance_contract(
            artifacts=artifact_records,
            counts=counts,
            generation_id=binding.generation_id,
        ),
        "counts": counts,
        "artifacts": artifact_records,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle_dir


async def _default_probe(binding: EmbeddingBinding, api_key: str) -> None:
    try:
        from lightrag.llm.openai import openai_embed
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookLightRAGError("LightRAG runtime dependency is unavailable") from error
    vectors = await openai_embed.func(
        texts=["材料科学知识图谱 embedding fallback activation"],
        model=binding.model,
        base_url=binding.base_url,
        api_key=api_key,
        embedding_dim=binding.dimensions,
        max_token_size=binding.max_input_tokens,
        client_configs={"timeout": binding.timeout_seconds},
    )
    if getattr(vectors, "shape", None) != (1, binding.dimensions):
        raise TextbookLightRAGError("fallback embedding dimension mismatch")


async def _default_generation_runner(
    binding: EmbeddingBinding,
    api_key: str,
    workspace: str,
    custom_kg: dict[str, Any],
    working_dir: Path,
) -> None:
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import EmbeddingFunc
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookLightRAGError("LightRAG runtime dependency is unavailable") from error

    document_prefix = None if binding.document_prefix == "NO_PREFIX" else binding.document_prefix

    async def unused_llm(*args: Any, **kwargs: Any) -> str:
        raise TextbookLightRAGError("custom-KG indexing must not invoke an LLM")

    embedding_func = EmbeddingFunc(
        embedding_dim=binding.dimensions,
        func=lambda texts, **kwargs: openai_embed.func(
            texts=texts,
            model=binding.model,
            base_url=binding.base_url,
            api_key=api_key,
            query_prefix=binding.query_prefix,
            document_prefix=document_prefix,
            client_configs={"timeout": binding.timeout_seconds},
            **kwargs,
        ),
        max_token_size=binding.max_input_tokens,
        send_dimensions=binding.send_dimensions,
        model_name=binding.model,
        supports_asymmetric=binding.asymmetric,
    )
    working_dir.mkdir(parents=True, exist_ok=True)
    rag: _CustomKGRAG = LightRAG(
        working_dir=str(working_dir),
        kv_storage=os.environ.get("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        vector_storage=os.environ.get(
            "LIGHTRAG_VECTOR_STORAGE",
            "NanoVectorDBStorage",
        ),
        graph_storage=os.environ.get("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        doc_status_storage=os.environ.get(
            "LIGHTRAG_DOC_STATUS_STORAGE",
            "JsonDocStatusStorage",
        ),
        workspace=workspace,
        embedding_func=embedding_func,
        embedding_batch_num=binding.batch_size,
        embedding_func_max_async=binding.max_async,
        default_embedding_timeout=int(binding.timeout_seconds),
        llm_model_func=unused_llm,
        llm_model_name="custom_kg_no_llm",
        llm_model_max_async=1,
        default_llm_timeout=1,
        auto_manage_storages_states=False,
    )
    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(custom_kg)
    finally:
        await rag.finalize_storages()


def _write_state(
    settings: TextbookCustomKGIndexSettings,
    summary: TextbookCustomKGIndexSummary,
) -> None:
    settings.state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = settings.state_path.with_suffix(f"{settings.state_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at_unix": time.time(),
                "summary": asdict(summary),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(settings.state_path)


async def index_textbook_custom_kg(
    settings: TextbookCustomKGIndexSettings,
    *,
    environment: dict[str, str] | None = None,
    generation_runner: GenerationRunner = _default_generation_runner,
    fallback_probe: FallbackProbe = _default_probe,
) -> TextbookCustomKGIndexSummary:
    """Index primary generation, switching only after a terminal quota error."""

    started_at = time.time()
    resolved_environment = dict(os.environ if environment is None else environment)
    primary = load_embedding_binding(settings.primary_embedding_binding_path)
    policy = _load_policy(settings.failover_policy_path)
    primary_key = resolved_environment.get(DEFAULT_EMBEDDING_KEY_ENV, "").strip()
    if not primary_key:
        raise TextbookLightRAGError(
            f"required provider credential is missing: {DEFAULT_EMBEDDING_KEY_ENV}"
        )
    with settings.custom_kg_path.open("r", encoding="utf-8") as stream:
        custom_kg = json.load(stream)
    if not isinstance(custom_kg, dict):
        raise TextbookLightRAGError("custom-KG payload must be an object")
    source_fragments = _load_fragments(settings.fragments_path)
    (
        primary_kg,
        primary_fragments,
        primary_mappings,
        primary_sources,
        primary_checkpoints,
    ) = _bind_generation(custom_kg, source_fragments, primary)
    primary_bundle = _write_deployment_bundle(
        settings,
        primary,
        primary_kg,
        primary_fragments,
        primary_mappings,
        primary_sources,
        primary_checkpoints,
    )

    try:
        await generation_runner(
            primary,
            primary_key,
            workspace_for_generation(primary.generation_id),
            primary_kg,
            settings.working_dir,
        )
    except Exception as primary_error:
        if policy.strategy == "disabled" or policy.fallback is None:
            raise TextbookLightRAGError("primary embedding indexing failed") from primary_error
        if not _quota_exhausted(primary_error):
            raise TextbookLightRAGError("primary embedding indexing failed") from primary_error
        fallback_key = resolved_environment.get(
            policy.fallback.credential_env,
            "",
        ).strip()
        if not fallback_key:
            raise TextbookLightRAGError(
                f"required provider credential is missing: {policy.fallback.credential_env}"
            ) from primary_error

        selected: EmbeddingBinding | None = None
        for model in policy.fallback.model_candidates:
            candidate = _fallback_binding(primary, policy, model)
            try:
                await fallback_probe(candidate, fallback_key)
            except Exception:
                continue
            selected = candidate
            break
        if selected is None:
            raise TextbookLightRAGError(
                "no fallback embedding model passed activation"
            ) from primary_error
        (
            fallback_kg,
            fallback_fragments,
            fallback_mappings,
            fallback_sources,
            fallback_checkpoints,
        ) = _bind_generation(custom_kg, source_fragments, selected)
        fallback_bundle = _write_deployment_bundle(
            settings,
            selected,
            fallback_kg,
            fallback_fragments,
            fallback_mappings,
            fallback_sources,
            fallback_checkpoints,
        )
        await generation_runner(
            selected,
            fallback_key,
            workspace_for_generation(selected.generation_id),
            fallback_kg,
            settings.working_dir,
        )
        summary = TextbookCustomKGIndexSummary(
            generation_id=selected.generation_id,
            provider=selected.provider,
            model=selected.model,
            deployment_bundle=fallback_bundle.as_posix(),
            failover_activated=True,
            elapsed_seconds=round(time.time() - started_at, 3),
            status="completed",
        )
        _write_state(settings, summary)
        return summary

    summary = TextbookCustomKGIndexSummary(
        generation_id=primary.generation_id,
        provider=primary.provider,
        model=primary.model,
        deployment_bundle=primary_bundle.as_posix(),
        failover_activated=False,
        elapsed_seconds=round(time.time() - started_at, 3),
        status="completed",
    )
    _write_state(settings, summary)
    return summary
