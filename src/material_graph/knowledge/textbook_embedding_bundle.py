"""Build a resumable, generation-bound textbook embedding archive."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

from .bindings import EmbeddingBinding
from .textbook_custom_kg import (
    FallbackProbe,
    TextbookCustomKGIndexSettings,
    index_textbook_custom_kg,
)
from .textbook_lightrag import TextbookLightRAGError


_COMPONENTS = (
    ("chunk", "custom_kg_chunks", "custom-kg-chunks.jsonl"),
    ("entity", "custom_kg_entities", "custom-kg-entities.jsonl"),
    (
        "relationship",
        "custom_kg_relationships",
        "custom-kg-relationships.jsonl",
    ),
)


class TextbookEmbeddingArchiveError(RuntimeError):
    """Stable embedding archive error that never exposes source text or credentials."""


_TERMINAL_QUOTA_MARKERS = (
    "insufficient_balance",
    "insufficient balance",
    "insufficient_quota",
    "insufficient quota",
    "quota_exhausted",
    "quota exhausted",
    "balance exhausted",
    "余额不足",
    "配额已耗尽",
)
_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_TRANSIENT_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimiterror",
    "too many requests",
    "tpm limit",
    "temporarily unavailable",
    "service unavailable",
    "timed out",
    "timeout",
    "timeouterror",
    "connecttimeout",
    "readtimeout",
    "write_timeout",
    " 408",
    " 429",
    " 500",
    " 502",
    " 503",
    " 504",
    "status_code=408",
    "status_code=429",
    "status_code=500",
    "status_code=502",
    "status_code=503",
    "status_code=504",
)


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Return an exception and provider-wrapped causes without logging details."""

    chain: list[BaseException] = []
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        chain.append(current)
        for nested in (
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
        # tenacity's RetryError keeps the provider exception on its last
        # attempt instead of exposing it through __cause__.
        last_attempt = getattr(current, "last_attempt", None)
        exception_getter = getattr(last_attempt, "exception", None)
        if callable(exception_getter):
            try:
                nested = exception_getter()
            except Exception:  # pragma: no cover - defensive provider boundary
                nested = None
            if isinstance(nested, BaseException):
                pending.append(nested)
    return chain


def _exception_text(error: BaseException) -> str:
    parts: list[str] = []
    for current in _exception_chain(error):
        parts.append(type(current).__name__)
        try:
            parts.append(str(current))
        except Exception:  # pragma: no cover - hostile provider exception
            continue
    return " ".join(parts).casefold()


def _is_terminal_quota_error(error: BaseException) -> bool:
    text = _exception_text(error)
    return any(marker in text for marker in _TERMINAL_QUOTA_MARKERS)


def _is_transient_embedding_error(error: BaseException) -> bool:
    if _is_terminal_quota_error(error):
        return False
    for current in _exception_chain(error):
        status_code = getattr(current, "status_code", None)
        if status_code in _TRANSIENT_STATUS_CODES:
            return True
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        if response_status in _TRANSIENT_STATUS_CODES:
            return True
    return any(marker in _exception_text(error) for marker in _TRANSIENT_ERROR_MARKERS)


@dataclass(frozen=True, slots=True)
class TextbookEmbeddingArchiveSettings:
    bundle_dir: Path
    output_dir: Path
    flush_items: int = 512
    max_async: int | None = None
    retry_max_attempts: int = 4
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.bundle_dir.is_dir() or not (self.bundle_dir / "manifest.json").is_file():
            raise ValueError("generation-bound deployment bundle is unavailable")
        if self.flush_items <= 0:
            raise ValueError("flush_items must be positive")
        if self.max_async is not None and self.max_async <= 0:
            raise ValueError("max_async must be positive")
        if self.retry_max_attempts <= 0:
            raise ValueError("retry_max_attempts must be positive")
        if self.retry_backoff_base_seconds < 0:
            raise ValueError("retry_backoff_base_seconds must be non-negative")
        if self.retry_backoff_max_seconds < self.retry_backoff_base_seconds:
            raise ValueError("retry_backoff_max_seconds must be at least the base delay")


@dataclass(frozen=True, slots=True)
class TextbookEmbeddingArchiveSummary:
    generation_id: str
    provider: str
    model: str
    vector_count: int
    item_count: int
    reused_item_count: int
    archive_dir: str
    elapsed_seconds: float
    status: str


@dataclass(frozen=True, slots=True)
class TextbookEmbeddingBundleSummary:
    generation_id: str
    provider: str
    model: str
    deployment_bundle: str
    embedding_archive: str
    vector_count: int
    item_count: int
    failover_activated: bool
    elapsed_seconds: float
    status: str


EmbeddingBatchFunction = Callable[
    [EmbeddingBinding, str, list[str]],
    Awaitable[np.ndarray],
]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact(bundle_dir: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise TextbookEmbeddingArchiveError("deployment artifact is invalid")
    try:
        relative_path = str(record["path"])
        expected_digest = str(record["sha256"])
        expected_bytes = int(record["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise TextbookEmbeddingArchiveError("deployment artifact is invalid") from error
    path = bundle_dir / relative_path
    if (
        not path.is_file()
        or path.resolve().parent != bundle_dir.resolve()
        or path.stat().st_size != expected_bytes
        or _file_sha256(path) != expected_digest
    ):
        raise TextbookEmbeddingArchiveError("deployment artifact digest mismatch")
    return path


def _validated_components(
    settings: TextbookEmbeddingArchiveSettings,
    binding: EmbeddingBinding,
) -> tuple[dict[str, Any], list[tuple[str, Path, int]], str]:
    manifest_path = settings.bundle_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embedding = manifest["embedding"]
        artifacts = manifest["artifacts"]
        counts = manifest["counts"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TextbookEmbeddingArchiveError("deployment manifest is invalid") from error
    expected_embedding = {
        "generation_id": binding.generation_id,
        "provider": binding.provider,
        "model": binding.model,
        "dimensions": binding.dimensions,
    }
    if any(embedding.get(key) != value for key, value in expected_embedding.items()):
        raise TextbookEmbeddingArchiveError("deployment embedding generation mismatch")

    components: list[tuple[str, Path, int]] = []
    for kind, artifact_name, _ in _COMPONENTS:
        try:
            expected_count = int(counts[artifact_name])
            artifact_record = artifacts[artifact_name]
        except (KeyError, TypeError, ValueError) as error:
            raise TextbookEmbeddingArchiveError(
                "deployment component manifest is invalid"
            ) from error
        if expected_count < 0:
            raise TextbookEmbeddingArchiveError("deployment component manifest is invalid")
        components.append(
            (kind, _safe_artifact(settings.bundle_dir, artifact_record), expected_count)
        )
    return manifest, components, _file_sha256(manifest_path)


def embedding_text(kind: str, row: dict[str, Any]) -> str:
    """Return the exact content string LightRAG embeds for a custom-KG item."""

    try:
        if kind == "chunk":
            return str(row["content"])
        if kind == "entity":
            return f"{row['entity_name']}\n{row['description']}"
        if kind == "relationship":
            return f"{row['keywords']}\t{row['src_id']}\n{row['tgt_id']}\n{row['description']}"
    except (KeyError, TypeError) as error:
        raise TextbookEmbeddingArchiveError("custom-KG embedding item is invalid") from error
    raise ValueError("unsupported embedding item kind")


def _item_id(kind: str, row: dict[str, Any], text: str) -> str:
    del text
    prefixes = {"chunk": "chunk-", "entity": "ent-", "relationship": "rel-"}
    prefix = prefixes.get(kind)
    if prefix is None:
        raise ValueError("unsupported embedding item kind")
    try:
        identity = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TextbookEmbeddingArchiveError("custom-KG embedding item is invalid") from error
    digest = md5(identity.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{prefix}{digest}"


def _initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS vectors (
            content_sha256 TEXT PRIMARY KEY,
            vector_index INTEGER UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS items (
            kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            PRIMARY KEY (kind, item_id),
            FOREIGN KEY (content_sha256) REFERENCES vectors(content_sha256)
        );
        """
    )
    connection.commit()
    return connection


def _database_counts(connection: sqlite3.Connection) -> tuple[int, int]:
    vector_count, max_index = connection.execute(
        "SELECT COUNT(*), MAX(vector_index) FROM vectors"
    ).fetchone()
    item_count = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    if vector_count and max_index != vector_count - 1:
        raise TextbookEmbeddingArchiveError("embedding vector indexes are not contiguous")
    return int(vector_count), int(item_count)


def _repair_binary_tail(path: Path, *, vector_count: int, dimensions: int) -> None:
    expected_bytes = vector_count * dimensions * np.dtype("<f2").itemsize
    path.touch(exist_ok=True)
    actual_bytes = path.stat().st_size
    if actual_bytes < expected_bytes:
        raise TextbookEmbeddingArchiveError("embedding vector binary is truncated")
    if actual_bytes > expected_bytes:
        with path.open("r+b") as stream:
            stream.truncate(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_or_initialize_state(
    path: Path,
    *,
    bundle_manifest_sha256: str,
    binding: EmbeddingBinding,
    vector_count: int,
    item_count: int,
) -> None:
    identity = {
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "generation_id": binding.generation_id,
        "provider": binding.provider,
        "model": binding.model,
        "dimensions": binding.dimensions,
    }
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise TextbookEmbeddingArchiveError("embedding state is invalid") from error
        if any(existing.get(key) != value for key, value in identity.items()):
            raise TextbookEmbeddingArchiveError("embedding state generation mismatch")
    _write_json(
        path,
        {
            "schema_version": 1,
            **identity,
            "vector_count": vector_count,
            "item_count": item_count,
            "status": "running",
        },
    )


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _completed_items(
    connection: sqlite3.Connection,
    kind: str,
    item_ids: list[str],
) -> dict[str, str]:
    if not item_ids:
        return {}
    rows = connection.execute(
        f"SELECT item_id, content_sha256 FROM items "
        f"WHERE kind = ? AND item_id IN ({_placeholders(len(item_ids))})",
        [kind, *item_ids],
    ).fetchall()
    return {str(item_id): str(content_hash) for item_id, content_hash in rows}


def _existing_vectors(
    connection: sqlite3.Connection,
    content_hashes: list[str],
) -> dict[str, int]:
    if not content_hashes:
        return {}
    rows = connection.execute(
        f"SELECT content_sha256, vector_index FROM vectors "
        f"WHERE content_sha256 IN ({_placeholders(len(content_hashes))})",
        content_hashes,
    ).fetchall()
    return {str(content_hash): int(index) for content_hash, index in rows}


async def _default_embedder(
    binding: EmbeddingBinding,
    api_key: str,
    texts: list[str],
) -> np.ndarray:
    try:
        from lightrag.llm.openai import openai_embed
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookEmbeddingArchiveError(
            "LightRAG embedding dependency is unavailable"
        ) from error
    document_prefix = None if binding.document_prefix == "NO_PREFIX" else binding.document_prefix
    return await openai_embed.func(
        texts=texts,
        model=binding.model,
        base_url=binding.base_url,
        api_key=api_key,
        embedding_dim=binding.dimensions,
        max_token_size=binding.max_input_tokens,
        client_configs={"timeout": binding.timeout_seconds},
        context="document",
        query_prefix=binding.query_prefix,
        document_prefix=document_prefix,
    )


async def _embed_missing(
    binding: EmbeddingBinding,
    api_key: str,
    texts: list[str],
    embedder: EmbeddingBatchFunction,
    *,
    max_async: int | None = None,
    retry_max_attempts: int = 4,
    retry_backoff_base_seconds: float = 2.0,
    retry_backoff_max_seconds: float = 60.0,
) -> np.ndarray:
    if not texts:
        return np.empty((0, binding.dimensions), dtype=np.float32)
    if retry_max_attempts <= 0:
        raise ValueError("retry_max_attempts must be positive")
    if retry_backoff_base_seconds < 0:
        raise ValueError("retry_backoff_base_seconds must be non-negative")
    if retry_backoff_max_seconds < retry_backoff_base_seconds:
        raise ValueError("retry_backoff_max_seconds must be at least the base delay")
    batches = [
        texts[start : start + binding.batch_size]
        for start in range(0, len(texts), binding.batch_size)
    ]
    semaphore = asyncio.Semaphore(max_async or binding.max_async)

    async def run(batch: list[str]) -> np.ndarray:
        for attempt in range(retry_max_attempts):
            try:
                async with semaphore:
                    values = await embedder(binding, api_key, batch)
                return np.asarray(values, dtype=np.float32)
            except Exception as error:
                if not _is_transient_embedding_error(error) or attempt >= retry_max_attempts - 1:
                    # Terminal quota/balance errors deliberately retain the
                    # provider exception so the quota-only failover policy can
                    # distinguish them from transient rate limiting.
                    raise
                delay = min(
                    retry_backoff_max_seconds,
                    retry_backoff_base_seconds * (2 ** min(attempt, 30)),
                )
                if delay > 0:
                    await asyncio.sleep(delay)

    arrays = await asyncio.gather(*(run(batch) for batch in batches))
    try:
        values = np.concatenate(arrays, axis=0)
    except ValueError as error:
        raise TextbookEmbeddingArchiveError("embedding result shape is invalid") from error
    if values.shape != (len(texts), binding.dimensions):
        raise TextbookEmbeddingArchiveError("embedding result dimension is invalid")
    if not np.isfinite(values).all():
        raise TextbookEmbeddingArchiveError("embedding result contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise TextbookEmbeddingArchiveError("embedding result has a zero norm")
    return values / norms


async def _flush_batch(
    *,
    kind: str,
    rows: list[dict[str, Any]],
    connection: sqlite3.Connection,
    vector_stream: Any,
    binding: EmbeddingBinding,
    api_key: str,
    embedder: EmbeddingBatchFunction,
    max_async: int | None,
    retry_max_attempts: int,
    retry_backoff_base_seconds: float,
    retry_backoff_max_seconds: float,
) -> tuple[int, int]:
    prepared: list[tuple[str, str, str]] = []
    for row in rows:
        text = embedding_text(kind, row)
        prepared.append(
            (
                _item_id(kind, row, text),
                sha256(text.encode("utf-8")).hexdigest(),
                text,
            )
        )
    completed = _completed_items(
        connection,
        kind,
        [item_id for item_id, _, _ in prepared],
    )
    pending: list[tuple[str, str, str]] = []
    for item_id, content_hash, text in prepared:
        existing_hash = completed.get(item_id)
        if existing_hash is not None:
            if existing_hash != content_hash:
                raise TextbookEmbeddingArchiveError(
                    "embedding item identity changed inside one generation"
                )
            continue
        pending.append((item_id, content_hash, text))
    if not pending:
        return 0, len(prepared)

    unique_text_by_hash: dict[str, str] = {}
    for _, content_hash, text in pending:
        unique_text_by_hash.setdefault(content_hash, text)
    existing_vectors = _existing_vectors(connection, list(unique_text_by_hash))
    missing_hashes = [
        content_hash for content_hash in unique_text_by_hash if content_hash not in existing_vectors
    ]
    new_vectors = await _embed_missing(
        binding,
        api_key,
        [unique_text_by_hash[content_hash] for content_hash in missing_hashes],
        embedder,
        max_async=max_async,
        retry_max_attempts=retry_max_attempts,
        retry_backoff_base_seconds=retry_backoff_base_seconds,
        retry_backoff_max_seconds=retry_backoff_max_seconds,
    )
    starting_index, _ = _database_counts(connection)
    new_vector_rows = [
        (content_hash, starting_index + offset)
        for offset, content_hash in enumerate(missing_hashes)
    ]

    if len(new_vectors):
        vector_stream.seek(0, os.SEEK_END)
        vector_stream.write(np.asarray(new_vectors, dtype="<f2").tobytes(order="C"))
        vector_stream.flush()
        os.fsync(vector_stream.fileno())
    with connection:
        connection.executemany(
            "INSERT INTO vectors(content_sha256, vector_index) VALUES (?, ?)",
            new_vector_rows,
        )
        connection.executemany(
            "INSERT INTO items(kind, item_id, content_sha256) VALUES (?, ?, ?)",
            [(kind, item_id, content_hash) for item_id, content_hash, _ in pending],
        )
    return len(missing_hashes), len(completed)


async def build_textbook_embedding_archive(
    settings: TextbookEmbeddingArchiveSettings,
    binding: EmbeddingBinding,
    api_key: str,
    *,
    embedder: EmbeddingBatchFunction = _default_embedder,
) -> TextbookEmbeddingArchiveSummary:
    """Create or resume a verified archive without mixing embedding generations."""

    started_at = time.time()
    if not api_key.strip():
        raise TextbookEmbeddingArchiveError("embedding credential is unavailable")
    _, components, bundle_manifest_sha256 = _validated_components(settings, binding)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = settings.output_dir / "vectors.sqlite3"
    vector_path = settings.output_dir / "vectors.f16.bin"
    state_path = settings.output_dir / "embedding-state.json"
    archive_manifest_path = settings.output_dir / "archive-manifest.json"
    connection = _initialize_database(database_path)
    reused_items = 0
    line_counts: dict[str, int] = {}
    try:
        vector_count, item_count = _database_counts(connection)
        _repair_binary_tail(
            vector_path,
            vector_count=vector_count,
            dimensions=binding.dimensions,
        )
        _validate_or_initialize_state(
            state_path,
            bundle_manifest_sha256=bundle_manifest_sha256,
            binding=binding,
            vector_count=vector_count,
            item_count=item_count,
        )
        with vector_path.open("r+b") as vector_stream:
            for kind, component_path, expected_count in components:
                batch: list[dict[str, Any]] = []
                line_count = 0
                with component_path.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise TextbookEmbeddingArchiveError(
                                f"custom-KG component is invalid at line {line_number}"
                            ) from error
                        if not isinstance(row, dict):
                            raise TextbookEmbeddingArchiveError(
                                f"custom-KG component is invalid at line {line_number}"
                            )
                        batch.append(row)
                        line_count += 1
                        if len(batch) < settings.flush_items:
                            continue
                        _, reused = await _flush_batch(
                            kind=kind,
                            rows=batch,
                            connection=connection,
                            vector_stream=vector_stream,
                            binding=binding,
                            api_key=api_key,
                            embedder=embedder,
                            max_async=settings.max_async,
                            retry_max_attempts=settings.retry_max_attempts,
                            retry_backoff_base_seconds=settings.retry_backoff_base_seconds,
                            retry_backoff_max_seconds=settings.retry_backoff_max_seconds,
                        )
                        reused_items += reused
                        batch.clear()
                        vector_count, item_count = _database_counts(connection)
                        _validate_or_initialize_state(
                            state_path,
                            bundle_manifest_sha256=bundle_manifest_sha256,
                            binding=binding,
                            vector_count=vector_count,
                            item_count=item_count,
                        )
                if batch:
                    _, reused = await _flush_batch(
                        kind=kind,
                        rows=batch,
                        connection=connection,
                        vector_stream=vector_stream,
                        binding=binding,
                        api_key=api_key,
                        embedder=embedder,
                        max_async=settings.max_async,
                        retry_max_attempts=settings.retry_max_attempts,
                        retry_backoff_base_seconds=settings.retry_backoff_base_seconds,
                        retry_backoff_max_seconds=settings.retry_backoff_max_seconds,
                    )
                    reused_items += reused
                if line_count != expected_count:
                    raise TextbookEmbeddingArchiveError("custom-KG component count mismatch")
                line_counts[kind] = line_count
                vector_count, item_count = _database_counts(connection)
                _validate_or_initialize_state(
                    state_path,
                    bundle_manifest_sha256=bundle_manifest_sha256,
                    binding=binding,
                    vector_count=vector_count,
                    item_count=item_count,
                )
        expected_items = sum(line_counts.values())
        vector_count, item_count = _database_counts(connection)
        if item_count != expected_items:
            raise TextbookEmbeddingArchiveError("embedding item count mismatch")
    except Exception as error:
        try:
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    state = {}
                state.update({"status": "failed", "error_type": type(error).__name__})
                _write_json(state_path, state)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        raise
    finally:
        connection.close()

    manifest = {
        "schema_version": 1,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "embedding": {
            "generation_id": binding.generation_id,
            "provider": binding.provider,
            "model": binding.model,
            "dimensions": binding.dimensions,
            "dtype": "float16-little-endian",
            "normalized": True,
            "distance": binding.distance,
        },
        "counts": {
            "vectors": vector_count,
            "items": item_count,
            **{f"{kind}_items": count for kind, count in line_counts.items()},
        },
        "artifacts": {
            "vectors": {
                "path": vector_path.name,
                "sha256": _file_sha256(vector_path),
                "bytes": vector_path.stat().st_size,
            },
            "index": {
                "path": database_path.name,
                "sha256": _file_sha256(database_path),
                "bytes": database_path.stat().st_size,
            },
        },
    }
    _write_json(archive_manifest_path, manifest)
    _write_json(
        state_path,
        {
            "schema_version": 1,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "generation_id": binding.generation_id,
            "provider": binding.provider,
            "model": binding.model,
            "dimensions": binding.dimensions,
            "vector_count": vector_count,
            "item_count": item_count,
            "status": "completed",
        },
    )
    return TextbookEmbeddingArchiveSummary(
        generation_id=binding.generation_id,
        provider=binding.provider,
        model=binding.model,
        vector_count=vector_count,
        item_count=item_count,
        reused_item_count=reused_items,
        archive_dir=settings.output_dir.as_posix(),
        elapsed_seconds=round(max(0.001, time.time() - started_at), 3),
        status="completed",
    )


async def build_textbook_embedding_bundle(
    settings: TextbookCustomKGIndexSettings,
    *,
    archive_root: Path,
    environment: dict[str, str] | None = None,
    embedder: EmbeddingBatchFunction = _default_embedder,
    fallback_probe: FallbackProbe | None = None,
    flush_items: int = 512,
    max_async: int | None = None,
) -> TextbookEmbeddingBundleSummary:
    """Coordinate strict quota-only failover around portable archive generation."""

    started_at = time.time()
    archive_summaries: dict[str, TextbookEmbeddingArchiveSummary] = {}

    async def generation_runner(
        binding: EmbeddingBinding,
        api_key: str,
        workspace: str,
        custom_kg: dict[str, Any],
        working_dir: Path,
    ) -> None:
        del custom_kg, working_dir
        bundle_dir = settings.deployment_dir / workspace
        summary = await build_textbook_embedding_archive(
            TextbookEmbeddingArchiveSettings(
                bundle_dir=bundle_dir,
                output_dir=archive_root / workspace,
                flush_items=flush_items,
                max_async=max_async,
            ),
            binding,
            api_key,
            embedder=embedder,
        )
        archive_summaries[workspace] = summary

    keyword_arguments: dict[str, Any] = {
        "environment": environment,
        "generation_runner": generation_runner,
    }
    if fallback_probe is not None:
        keyword_arguments["fallback_probe"] = fallback_probe
    try:
        index_summary = await index_textbook_custom_kg(
            settings,
            **keyword_arguments,
        )
    except TextbookLightRAGError:
        raise
    except Exception as error:
        raise TextbookLightRAGError("textbook embedding bundle failed") from error

    workspace = Path(index_summary.deployment_bundle).name
    archive_summary = archive_summaries.get(workspace)
    if archive_summary is None:
        raise TextbookLightRAGError("embedding archive summary is unavailable")
    return TextbookEmbeddingBundleSummary(
        generation_id=index_summary.generation_id,
        provider=index_summary.provider,
        model=index_summary.model,
        deployment_bundle=index_summary.deployment_bundle,
        embedding_archive=archive_summary.archive_dir,
        vector_count=archive_summary.vector_count,
        item_count=archive_summary.item_count,
        failover_activated=index_summary.failover_activated,
        elapsed_seconds=round(max(0.001, time.time() - started_at), 3),
        status="completed",
    )
