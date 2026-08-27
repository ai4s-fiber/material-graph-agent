"""Import a portable textbook embedding archive through bounded LightRAG writes."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Protocol
import zlib

import numpy as np

from .bindings import EmbeddingBinding
from .lightrag_runtime import workspace_for_generation


_PHASES = ("chunks", "entities", "relationships")
_ARTIFACTS = {
    "chunks": "custom_kg_chunks",
    "entities": "custom_kg_entities",
    "relationships": "custom_kg_relationships",
}
_RELATIONSHIP_REPAIR_VERSION = "collision-safe-sha256.v2"
_NANOVDB_HEADER_RE = re.compile(r'^\s*\{\s*"embedding_dim"\s*:\s*(\d+)\s*,\s*"data"\s*:\s*\[')
_STREAM_CHUNK_SIZE = 1024 * 1024
_MAX_STREAM_RECORD_SIZE = 16 * 1024 * 1024


class TextbookPrecomputedImportError(RuntimeError):
    """Stable import error that never exposes source text or credentials."""


class _CustomKGRAG(Protocol):
    async def initialize_storages(self) -> None: ...

    async def finalize_storages(self) -> None: ...

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TextbookPrecomputedImportSettings:
    bundle_dir: Path
    archive_dir: Path
    working_dir: Path
    state_path: Path
    batch_size: int = 512

    def __post_init__(self) -> None:
        if not self.bundle_dir.is_dir() or not (self.bundle_dir / "manifest.json").is_file():
            raise ValueError("generation-bound deployment bundle is unavailable")
        if (
            not self.archive_dir.is_dir()
            or not (self.archive_dir / "archive-manifest.json").is_file()
        ):
            raise ValueError("precomputed embedding archive is unavailable")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class TextbookPrecomputedImportSummary:
    generation_id: str
    model: str
    chunks: int
    entities: int
    relationships: int
    status: str


@dataclass(frozen=True, slots=True)
class _RelationshipCollisionPlan:
    concat_collision_ids: frozenset[str]
    reverse_delete_collision_ids: frozenset[str]
    dangerous_canonical_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _RelationshipVDBRewriteSummary:
    rows_before: int
    rows_after: int
    unique_ids: int
    matrix_rows: int
    deleted_dangerous_count: int
    inserted_safe_count: int


@dataclass(frozen=True, slots=True)
class _RelationshipRepairMaterial:
    plan: _RelationshipCollisionPlan
    affected_count: int
    replacements: tuple[dict[str, Any], ...]


RAGFactory = Callable[
    [EmbeddingBinding, str, "PrecomputedEmbeddingLookup", Path],
    _CustomKGRAG,
]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise TextbookPrecomputedImportError(f"{label} manifest is invalid") from error
    if not isinstance(payload, dict):
        raise TextbookPrecomputedImportError(f"{label} manifest is invalid")
    return payload, _file_sha256(path)


def _safe_artifact(root: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise TextbookPrecomputedImportError("precomputed artifact is invalid")
    try:
        relative = str(record["path"])
        expected_digest = str(record["sha256"])
        expected_bytes = int(record["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise TextbookPrecomputedImportError("precomputed artifact is invalid") from error
    path = root / relative
    if (
        not path.is_file()
        or path.resolve().parent != root.resolve()
        or path.stat().st_size != expected_bytes
        or _file_sha256(path) != expected_digest
    ):
        raise TextbookPrecomputedImportError("precomputed artifact digest mismatch")
    return path


class PrecomputedEmbeddingLookup:
    """Read-only text lookup backed by SQLite hashes and a float16 mmap."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        vectors: np.ndarray,
        dimensions: int,
        content_hash_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._connection = connection
        self._vectors = vectors
        self.dimensions = dimensions
        self._content_hash_aliases = dict(content_hash_aliases or {})

    @classmethod
    def open(
        cls,
        archive_dir: Path,
        *,
        content_hash_aliases: Mapping[str, str] | None = None,
    ) -> "PrecomputedEmbeddingLookup":
        manifest, _ = _manifest(
            archive_dir / "archive-manifest.json",
            label="embedding archive",
        )
        try:
            dimensions = int(manifest["embedding"]["dimensions"])
            vector_count = int(manifest["counts"]["vectors"])
            artifacts = manifest["artifacts"]
        except (KeyError, TypeError, ValueError) as error:
            raise TextbookPrecomputedImportError("embedding archive manifest is invalid") from error
        if dimensions <= 0 or vector_count < 0:
            raise TextbookPrecomputedImportError("embedding archive manifest is invalid")
        vector_path = _safe_artifact(archive_dir, artifacts.get("vectors"))
        index_path = _safe_artifact(archive_dir, artifacts.get("index"))
        uri = f"{index_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        try:
            database_count = int(connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
            if database_count != vector_count:
                raise TextbookPrecomputedImportError("embedding archive vector count mismatch")
            expected_bytes = vector_count * dimensions * np.dtype("<f2").itemsize
            if vector_path.stat().st_size != expected_bytes:
                raise TextbookPrecomputedImportError(
                    "embedding archive vector binary size mismatch"
                )
            vectors: np.ndarray
            if vector_count:
                vectors = np.memmap(
                    vector_path,
                    dtype="<f2",
                    mode="r",
                    shape=(vector_count, dimensions),
                )
            else:
                vectors = np.empty((0, dimensions), dtype=np.float16)
        except Exception:
            connection.close()
            raise
        return cls(
            connection=connection,
            vectors=vectors,
            dimensions=dimensions,
            content_hash_aliases=content_hash_aliases,
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "PrecomputedEmbeddingLookup":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __call__(
        self,
        texts: list[str],
        *,
        context: str = "document",
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        if context != "document":
            raise TextbookPrecomputedImportError(
                "precomputed import lookup cannot serve query embeddings"
            )
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        content_hashes = [sha256(text.encode("utf-8")).hexdigest() for text in texts]
        lookup_hashes = list(
            dict.fromkeys(
                [
                    *content_hashes,
                    *(
                        self._content_hash_aliases[content_hash]
                        for content_hash in content_hashes
                        if content_hash in self._content_hash_aliases
                    ),
                ]
            )
        )
        index_by_hash: dict[str, int] = {}
        for start in range(0, len(lookup_hashes), 900):
            batch = lookup_hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self._connection.execute(
                f"SELECT content_sha256, vector_index FROM vectors "
                f"WHERE content_sha256 IN ({placeholders})",
                batch,
            ).fetchall()
            index_by_hash.update(
                (str(content_hash), int(vector_index)) for content_hash, vector_index in rows
            )
        indexes: list[int] = []
        for content_hash in content_hashes:
            resolved_hash = (
                content_hash
                if content_hash in index_by_hash
                else self._content_hash_aliases.get(content_hash)
            )
            if resolved_hash is None or resolved_hash not in index_by_hash:
                raise TextbookPrecomputedImportError(
                    "precomputed embedding is missing for import content"
                )
            indexes.append(index_by_hash[resolved_hash])
        return np.asarray(self._vectors[indexes], dtype=np.float32)


class _ChunkStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._offsets: dict[str, tuple[int, int]] = {}
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                    source_id = str(row["source_id"])
                except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
                    raise TextbookPrecomputedImportError(
                        "custom-KG chunk component is invalid"
                    ) from error
                if source_id in self._offsets:
                    raise TextbookPrecomputedImportError("duplicate custom-KG chunk source")
                self._offsets[source_id] = (offset, len(line))
        self._stream = path.open("rb")

    def close(self) -> None:
        self._stream.close()

    def get(self, source_id: str) -> dict[str, Any]:
        location = self._offsets.get(source_id)
        if location is None:
            raise TextbookPrecomputedImportError("custom-KG item references an unknown chunk")
        offset, length = location
        self._stream.seek(offset)
        try:
            row = json.loads(self._stream.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise TextbookPrecomputedImportError("custom-KG chunk component is invalid") from error
        if not isinstance(row, dict):
            raise TextbookPrecomputedImportError("custom-KG chunk component is invalid")
        return row


def _validated_contract(
    settings: TextbookPrecomputedImportSettings,
) -> tuple[
    EmbeddingBinding,
    dict[str, Path],
    dict[str, int],
    str,
    str,
]:
    bundle_manifest_path = settings.bundle_dir / "manifest.json"
    archive_manifest_path = settings.archive_dir / "archive-manifest.json"
    bundle, bundle_digest = _manifest(bundle_manifest_path, label="deployment")
    archive, archive_digest = _manifest(
        archive_manifest_path,
        label="embedding archive",
    )
    try:
        bundle_artifacts = bundle["artifacts"]
        bundle_counts = bundle["counts"]
        binding_path = _safe_artifact(
            settings.bundle_dir,
            bundle_artifacts["embedding_binding"],
        )
        binding = EmbeddingBinding.model_validate_json(binding_path.read_text(encoding="utf-8"))
        archive_embedding = archive["embedding"]
    except Exception as error:
        if isinstance(error, TextbookPrecomputedImportError):
            raise
        raise TextbookPrecomputedImportError("precomputed import contract is invalid") from error
    expected_embedding = {
        "generation_id": binding.generation_id,
        "provider": binding.provider,
        "model": binding.model,
        "dimensions": binding.dimensions,
    }
    if archive.get("bundle_manifest_sha256") != bundle_digest or any(
        archive_embedding.get(key) != value for key, value in expected_embedding.items()
    ):
        raise TextbookPrecomputedImportError("precomputed import generation mismatch")
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for phase in _PHASES:
        artifact_name = _ARTIFACTS[phase]
        try:
            counts[phase] = int(bundle_counts[artifact_name])
            paths[phase] = _safe_artifact(
                settings.bundle_dir,
                bundle_artifacts[artifact_name],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TextbookPrecomputedImportError(
                "precomputed import component is invalid"
            ) from error
        if counts[phase] < 0:
            raise TextbookPrecomputedImportError("precomputed import component is invalid")
    return binding, paths, counts, bundle_digest, archive_digest


def _state_identity(
    binding: EmbeddingBinding,
    *,
    bundle_digest: str,
    archive_digest: str,
) -> dict[str, Any]:
    return {
        "bundle_manifest_sha256": bundle_digest,
        "archive_manifest_sha256": archive_digest,
        "generation_id": binding.generation_id,
        "model": binding.model,
    }


def _legacy_mdhash_id(content: str, *, prefix: str) -> str:
    return prefix + md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()


def _normalized_relationship_pair(src_id: object, tgt_id: object) -> tuple[str, str]:
    src = str(src_id)
    tgt = str(tgt_id)
    return tuple(sorted((src, tgt)))


def _canonical_relationship_id(src_id: object, tgt_id: object) -> str:
    src, tgt = _normalized_relationship_pair(src_id, tgt_id)
    return _legacy_mdhash_id(src + tgt, prefix="rel-")


def _legacy_reverse_relationship_id(src_id: object, tgt_id: object) -> str | None:
    src, tgt = _normalized_relationship_pair(src_id, tgt_id)
    canonical = _legacy_mdhash_id(src + tgt, prefix="rel-")
    reverse = _legacy_mdhash_id(tgt + src, prefix="rel-")
    return reverse if reverse != canonical else None


def _safe_relationship_id(src_id: object, tgt_id: object) -> str:
    src, tgt = _normalized_relationship_pair(src_id, tgt_id)
    payload = json.dumps(
        [src, tgt],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "rel-sha256-" + sha256(payload.encode("utf-8")).hexdigest()


def _relationship_collision_plan(
    rows: Iterable[Mapping[str, Any]],
) -> _RelationshipCollisionPlan:
    owner_by_canonical_id: dict[str, tuple[str, str]] = {}
    reverse_ids: set[str] = set()
    concat_collision_ids: set[str] = set()
    for row in rows:
        try:
            pair = _normalized_relationship_pair(row["src_id"], row["tgt_id"])
        except (KeyError, TypeError) as error:
            raise TextbookPrecomputedImportError(
                "custom-KG relationship component is invalid"
            ) from error
        canonical_id = _canonical_relationship_id(*pair)
        existing = owner_by_canonical_id.get(canonical_id)
        if existing is not None and existing != pair:
            concat_collision_ids.add(canonical_id)
        else:
            owner_by_canonical_id.setdefault(canonical_id, pair)
        reverse_id = _legacy_reverse_relationship_id(*pair)
        if reverse_id is not None:
            reverse_ids.add(reverse_id)
    reverse_delete_collision_ids = set(owner_by_canonical_id).intersection(reverse_ids)
    dangerous = concat_collision_ids.union(reverse_delete_collision_ids)
    return _RelationshipCollisionPlan(
        concat_collision_ids=frozenset(concat_collision_ids),
        reverse_delete_collision_ids=frozenset(reverse_delete_collision_ids),
        dangerous_canonical_ids=frozenset(dangerous),
    )


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_state(
    settings: TextbookPrecomputedImportSettings,
    identity: dict[str, Any],
) -> dict[str, Any]:
    if not settings.state_path.is_file():
        return {
            "schema_version": 1,
            **identity,
            "phase": "chunks",
            "offset": 0,
            "status": "running",
        }
    try:
        state = json.loads(settings.state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise TextbookPrecomputedImportError("precomputed import state is invalid") from error
    if not isinstance(state, dict) or any(
        state.get(key) != value for key, value in identity.items()
    ):
        raise TextbookPrecomputedImportError("precomputed import state generation mismatch")
    phase = state.get("phase")
    if phase not in (*_PHASES, "completed"):
        raise TextbookPrecomputedImportError("precomputed import state is invalid")
    try:
        offset = int(state.get("offset", 0))
    except (TypeError, ValueError) as error:
        raise TextbookPrecomputedImportError("precomputed import state is invalid") from error
    if offset < 0:
        raise TextbookPrecomputedImportError("precomputed import state is invalid")
    return {**state, "offset": offset}


def _iter_rows(path: Path, *, skip: int) -> Iterator[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TextbookPrecomputedImportError(
                    f"custom-KG component is invalid at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise TextbookPrecomputedImportError(
                    f"custom-KG component is invalid at line {line_number}"
                )
            seen += 1
            yield row


def _iter_nanovdb_records(
    path: Path,
    *,
    dimensions: int,
    require_matrix: bool = True,
) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = stream.read(_STREAM_CHUNK_SIZE)
        match = _NANOVDB_HEADER_RE.match(buffer)
        while match is None and len(buffer) <= _MAX_STREAM_RECORD_SIZE:
            chunk = stream.read(_STREAM_CHUNK_SIZE)
            if not chunk:
                break
            buffer += chunk
            match = _NANOVDB_HEADER_RE.match(buffer)
        if match is None or int(match.group(1)) != dimensions:
            raise TextbookPrecomputedImportError("relationship vector database header is invalid")
        buffer = buffer[match.end() :]
        first = True
        record_count = 0
        while True:
            while not buffer.strip():
                chunk = stream.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    raise TextbookPrecomputedImportError(
                        "relationship vector database is truncated"
                    )
                buffer += chunk
            buffer = buffer.lstrip()
            if first and buffer.startswith("]"):
                buffer = buffer[1:]
                break
            if not first:
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    break
                if not buffer.startswith(","):
                    raise TextbookPrecomputedImportError("relationship vector database is invalid")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if len(buffer) > _MAX_STREAM_RECORD_SIZE:
                        raise TextbookPrecomputedImportError(
                            "relationship vector database record is invalid"
                        ) from error
                    chunk = stream.read(_STREAM_CHUNK_SIZE)
                    if not chunk:
                        raise TextbookPrecomputedImportError(
                            "relationship vector database is truncated"
                        ) from error
                    buffer += chunk
            if not isinstance(row, dict):
                raise TextbookPrecomputedImportError(
                    "relationship vector database record is invalid"
                )
            buffer = buffer[end:]
            first = False
            record_count += 1
            yield row
        _consume_nanovdb_trailer(
            buffer,
            stream,
            dimensions=dimensions,
            record_count=record_count,
            require_matrix=require_matrix,
        )


def _consume_nanovdb_trailer(
    buffer: str,
    stream: Any,
    *,
    dimensions: int,
    record_count: int,
    require_matrix: bool,
) -> None:
    trailing = buffer.lstrip()
    while not trailing:
        chunk = stream.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            raise TextbookPrecomputedImportError("relationship vector database trailer is invalid")
        trailing += chunk
    if trailing.startswith("}"):
        if require_matrix:
            raise TextbookPrecomputedImportError(
                "relationship vector database matrix is unavailable"
            )
        if (trailing[1:] + stream.read()).strip():
            raise TextbookPrecomputedImportError("relationship vector database trailer is invalid")
        return

    matrix_prefix = re.compile(r'^,\s*"matrix"\s*:\s*"')
    while matrix_prefix.match(trailing) is None and len(trailing) < 256:
        chunk = stream.read(256)
        if not chunk:
            break
        trailing += chunk
    prefix_match = matrix_prefix.match(trailing)
    if prefix_match is None:
        raise TextbookPrecomputedImportError("relationship vector database trailer is invalid")
    trailing = trailing[prefix_match.end() :]
    encoded_length = 0
    while True:
        quote_index = trailing.find('"')
        if quote_index >= 0:
            encoded_length += quote_index
            trailing = trailing[quote_index + 1 :]
            break
        encoded_length += len(trailing)
        trailing = stream.read(_STREAM_CHUNK_SIZE)
        if not trailing:
            raise TextbookPrecomputedImportError("relationship vector database matrix is truncated")
    expected_bytes = record_count * dimensions * np.dtype(np.float32).itemsize
    expected_encoded_length = 4 * ((expected_bytes + 2) // 3)
    if encoded_length != expected_encoded_length:
        raise TextbookPrecomputedImportError("relationship vector database matrix is incomplete")
    if (trailing + stream.read()).strip() != "}":
        raise TextbookPrecomputedImportError("relationship vector database trailer is invalid")


def _decode_relationship_vector(vector: Any, *, dimensions: int) -> np.ndarray:
    if not isinstance(vector, str) or not vector:
        raise TextbookPrecomputedImportError("relationship vector database matrix is incomplete")
    try:
        compressed = base64.b64decode(vector, validate=True)
        raw = zlib.decompress(compressed)
    except (ValueError, zlib.error) as error:
        raise TextbookPrecomputedImportError(
            "relationship vector database matrix is invalid"
        ) from error
    resolved = np.frombuffer(raw, dtype="<f2")
    if resolved.size != dimensions or not np.isfinite(resolved).all():
        raise TextbookPrecomputedImportError("relationship vector database matrix is invalid")
    return resolved.astype(np.float32)


def _relationship_matrix_row(vector: Any, *, dimensions: int) -> bytes:
    matrix_row = _decode_relationship_vector(vector, dimensions=dimensions)
    norm = np.linalg.norm(matrix_row)
    if not np.isfinite(norm) or norm <= 0:
        raise TextbookPrecomputedImportError("relationship vector database matrix is invalid")
    matrix_row = matrix_row / norm
    if not np.isfinite(matrix_row).all():
        raise TextbookPrecomputedImportError("relationship vector database matrix is invalid")
    return np.asarray(matrix_row, dtype="<f4").tobytes()


def _write_base64_stream(source: Any, output: Any) -> None:
    carry = b""
    while block := source.read(_STREAM_CHUNK_SIZE):
        combined = carry + block
        complete_length = len(combined) - (len(combined) % 3)
        if complete_length:
            output.write(base64.b64encode(combined[:complete_length]).decode("ascii"))
        carry = combined[complete_length:]
    if carry:
        output.write(base64.b64encode(carry).decode("ascii"))


def _repair_record_matches(
    existing: Mapping[str, Any],
    replacement: Mapping[str, Any],
) -> bool:
    fields = ("__id__", "src_id", "tgt_id", "source_id", "content", "file_path", "vector")
    return all(existing.get(field) == replacement.get(field) for field in fields)


def _rewrite_relationship_vdb(
    path: Path,
    *,
    dimensions: int,
    expected_count: int,
    dangerous_ids: frozenset[str],
    replacements: list[dict[str, Any]],
) -> _RelationshipVDBRewriteSummary:
    replacement_by_id: dict[str, dict[str, Any]] = {}
    for row in replacements:
        row_id = str(row.get("__id__") or "")
        if not row_id.startswith("rel-sha256-") or row_id in replacement_by_id:
            raise TextbookPrecomputedImportError(
                "relationship vector repair replacement is invalid"
            )
        replacement_by_id[row_id] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    matrix_temporary_path: Path | None = None
    rows_before = 0
    rows_after = 0
    matrix_rows = 0
    deleted_dangerous_count = 0
    inserted_safe_count = 0
    seen_ids: set[str] = set()
    present_safe_ids: set[str] = set()
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=path.parent,
                prefix=f".{path.name}.repair-",
                suffix=".tmp",
                delete=False,
            ) as output,
            tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{path.name}.matrix-",
                suffix=".tmp",
                delete=False,
            ) as matrix_output,
        ):
            temporary_path = Path(output.name)
            matrix_temporary_path = Path(matrix_output.name)
            output.write(
                json.dumps(
                    {"embedding_dim": dimensions},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )[:-1]
            )
            output.write(',"data":[')
            first_output = True

            def write_row(row: Mapping[str, Any]) -> None:
                nonlocal first_output, rows_after, matrix_rows
                row_id = str(row.get("__id__") or "")
                vector = row.get("vector")
                if not row_id or row_id in seen_ids:
                    raise TextbookPrecomputedImportError(
                        "relationship vector database contains duplicate IDs"
                    )
                matrix_row = _relationship_matrix_row(
                    vector,
                    dimensions=dimensions,
                )
                if not first_output:
                    output.write(",")
                json.dump(
                    dict(row),
                    output,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                first_output = False
                seen_ids.add(row_id)
                rows_after += 1
                matrix_rows += 1
                matrix_output.write(matrix_row)

            for row in _iter_nanovdb_records(
                path,
                dimensions=dimensions,
                require_matrix=False,
            ):
                rows_before += 1
                row_id = str(row.get("__id__") or "")
                if row_id in dangerous_ids:
                    deleted_dangerous_count += 1
                    continue
                replacement = replacement_by_id.get(row_id)
                if replacement is not None:
                    if not _repair_record_matches(row, replacement):
                        raise TextbookPrecomputedImportError(
                            "relationship vector repair record mismatch"
                        )
                    present_safe_ids.add(row_id)
                write_row(row)

            for row_id in sorted(set(replacement_by_id).difference(present_safe_ids)):
                write_row(replacement_by_id[row_id])
                inserted_safe_count += 1
            matrix_output.flush()
            os.fsync(matrix_output.fileno())
            expected_matrix_bytes = rows_after * dimensions * np.dtype(np.float32).itemsize
            if matrix_output.tell() != expected_matrix_bytes:
                raise TextbookPrecomputedImportError(
                    "relationship vector repair matrix count mismatch"
                )
            matrix_output.seek(0)
            output.write('],"matrix":"')
            _write_base64_stream(matrix_output, output)
            output.write('"}')
            output.flush()
            os.fsync(output.fileno())

        if rows_after != expected_count or matrix_rows != expected_count:
            raise TextbookPrecomputedImportError("relationship vector repair count mismatch")
        if dangerous_ids.intersection(seen_ids):
            raise TextbookPrecomputedImportError("relationship vector repair left a dangerous ID")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if matrix_temporary_path is not None:
            matrix_temporary_path.unlink(missing_ok=True)

    return _RelationshipVDBRewriteSummary(
        rows_before=rows_before,
        rows_after=rows_after,
        unique_ids=len(seen_ids),
        matrix_rows=matrix_rows,
        deleted_dangerous_count=deleted_dangerous_count,
        inserted_safe_count=inserted_safe_count,
    )


def _lightrag_text_sanitizers() -> tuple[Callable[[str], str], Callable[[str], str]]:
    try:
        from lightrag.utils import sanitize_text_for_encoding, strip_control_characters
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookPrecomputedImportError(
            "LightRAG runtime dependency is unavailable"
        ) from error

    def sanitize_graph_text(value: str) -> str:
        return strip_control_characters(value, replacement_char="\ufffd")

    return sanitize_text_for_encoding, sanitize_graph_text


def _embedding_text(
    phase: str,
    row: Mapping[str, Any],
    *,
    normalize_relationship_endpoints: bool = False,
) -> str:
    try:
        if phase == "chunks":
            return str(row["content"])
        if phase == "entities":
            return f"{row['entity_name']}\n{row['description']}"
        if phase == "relationships":
            src_id = str(row["src_id"])
            tgt_id = str(row["tgt_id"])
            if normalize_relationship_endpoints:
                src_id, tgt_id = sorted((src_id, tgt_id))
            return f"{row['keywords']}\t{src_id}\n{tgt_id}\n{row['description']}"
    except (KeyError, TypeError) as error:
        raise TextbookPrecomputedImportError("custom-KG embedding component is invalid") from error
    raise TextbookPrecomputedImportError("custom-KG embedding phase is invalid")


def _sanitize_custom_kg_row(
    phase: str,
    row: Mapping[str, Any],
    *,
    sanitize_chunk_text: Callable[[str], str],
    sanitize_graph_text: Callable[[str], str],
) -> dict[str, Any]:
    sanitized = dict(row)
    fields = {
        "chunks": ("content", "file_path"),
        "entities": ("entity_name", "entity_type", "description", "file_path"),
        "relationships": (
            "src_id",
            "tgt_id",
            "description",
            "keywords",
            "file_path",
        ),
    }.get(phase)
    if fields is None:
        raise TextbookPrecomputedImportError("custom-KG embedding phase is invalid")
    for field in fields:
        if field not in sanitized:
            continue
        value = str(sanitized[field])
        sanitized[field] = (
            sanitize_chunk_text(value)
            if phase == "chunks" and field == "content"
            else sanitize_graph_text(value)
        )
    return sanitized


def _lightrag_content_hash_aliases(
    component_paths: Mapping[str, Path],
    *,
    sanitize_chunk_text: Callable[[str], str],
    sanitize_graph_text: Callable[[str], str],
) -> dict[str, str]:
    """Map LightRAG-safe content hashes to the frozen archive hashes."""

    owner_by_sanitized_hash: dict[str, str] = {}
    owner_by_graph_identifier: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for phase in _PHASES:
        for row in _iter_rows(component_paths[phase], skip=0):
            sanitized_row = _sanitize_custom_kg_row(
                phase,
                row,
                sanitize_chunk_text=sanitize_chunk_text,
                sanitize_graph_text=sanitize_graph_text,
            )
            if phase == "entities":
                identifiers = ((row["entity_name"], sanitized_row["entity_name"]),)
            elif phase == "relationships":
                identifiers = tuple(
                    (row[field], sanitized_row[field]) for field in ("src_id", "tgt_id")
                )
            else:
                identifiers = ()
            for raw_identifier, sanitized_identifier in identifiers:
                raw_identifier = str(raw_identifier)
                sanitized_identifier = str(sanitized_identifier)
                existing_identifier = owner_by_graph_identifier.get(sanitized_identifier)
                if existing_identifier is not None and existing_identifier != raw_identifier:
                    raise TextbookPrecomputedImportError(
                        "precomputed import graph identifier collision"
                    )
                owner_by_graph_identifier[sanitized_identifier] = raw_identifier

            raw_text = _embedding_text(phase, row)
            sanitized_text = _embedding_text(
                phase,
                sanitized_row,
                normalize_relationship_endpoints=phase == "relationships",
            )
            raw_hash = sha256(raw_text.encode("utf-8")).hexdigest()
            sanitized_hash = sha256(sanitized_text.encode("utf-8")).hexdigest()
            existing_owner = owner_by_sanitized_hash.get(sanitized_hash)
            if existing_owner is not None and existing_owner != raw_hash:
                raise TextbookPrecomputedImportError("precomputed import content alias collision")
            owner_by_sanitized_hash[sanitized_hash] = raw_hash
            if sanitized_hash != raw_hash:
                aliases[sanitized_hash] = raw_hash
    return aliases


def _source_chunks(
    rows: list[dict[str, Any]],
    chunk_store: _ChunkStore,
    *,
    sanitize_chunk_text: Callable[[str], str],
    sanitize_graph_text: Callable[[str], str],
) -> list[dict[str, Any]]:
    source_ids: list[str] = []
    for row in rows:
        try:
            source_id = str(row["source_id"])
        except (KeyError, TypeError) as error:
            raise TextbookPrecomputedImportError("custom-KG item source is invalid") from error
        if source_id not in source_ids:
            source_ids.append(source_id)
    return [
        _sanitize_custom_kg_row(
            "chunks",
            chunk_store.get(source_id),
            sanitize_chunk_text=sanitize_chunk_text,
            sanitize_graph_text=sanitize_graph_text,
        )
        for source_id in source_ids
    ]


def _iter_sanitized_relationship_rows(
    path: Path,
    *,
    sanitize_chunk_text: Callable[[str], str],
    sanitize_graph_text: Callable[[str], str],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    del sanitize_chunk_text
    for raw_row in _iter_rows(path, skip=0):
        sanitized = _sanitize_custom_kg_row(
            "relationships",
            raw_row,
            sanitize_chunk_text=lambda value: value,
            sanitize_graph_text=sanitize_graph_text,
        )
        yield raw_row, sanitized


def _relationship_content(row: Mapping[str, Any]) -> str:
    try:
        src_id, tgt_id = _normalized_relationship_pair(
            row["src_id"],
            row["tgt_id"],
        )
        return f"{row['keywords']}\t{src_id}\n{tgt_id}\n{row['description']}"
    except (KeyError, TypeError) as error:
        raise TextbookPrecomputedImportError(
            "custom-KG relationship component is invalid"
        ) from error


def _compressed_vector(vector: np.ndarray, *, dimensions: int) -> str:
    resolved = np.asarray(vector, dtype=np.float32).reshape(-1)
    if resolved.size != dimensions or not np.isfinite(resolved).all():
        raise TextbookPrecomputedImportError("relationship vector repair embedding is invalid")
    vector_f16 = resolved.astype(np.float16)
    return base64.b64encode(zlib.compress(vector_f16.tobytes())).decode("utf-8")


async def _relationship_replacements(
    affected_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    archive_dir: Path,
    chunk_store: _ChunkStore,
    dimensions: int,
    sanitize_chunk_text: Callable[[str], str],
    sanitize_graph_text: Callable[[str], str],
) -> list[dict[str, Any]]:
    aliases: dict[str, str] = {}
    contents: list[str] = []
    records: list[dict[str, Any]] = []
    seen_safe_ids: set[str] = set()
    for raw_row, sanitized_row in affected_rows:
        try:
            source_key = str(raw_row["source_id"])
            source_chunk = _sanitize_custom_kg_row(
                "chunks",
                chunk_store.get(source_key),
                sanitize_chunk_text=sanitize_chunk_text,
                sanitize_graph_text=sanitize_graph_text,
            )
            source_id = _legacy_mdhash_id(
                str(source_chunk["content"]),
                prefix="chunk-",
            )
            src_id, tgt_id = _normalized_relationship_pair(
                sanitized_row["src_id"],
                sanitized_row["tgt_id"],
            )
            safe_id = _safe_relationship_id(src_id, tgt_id)
            content = _relationship_content(sanitized_row)
            raw_content = _embedding_text("relationships", raw_row)
            file_path = str(sanitized_row.get("file_path", "custom_kg"))
        except (KeyError, TypeError) as error:
            raise TextbookPrecomputedImportError(
                "relationship vector repair source is invalid"
            ) from error
        if safe_id in seen_safe_ids:
            raise TextbookPrecomputedImportError("relationship vector repair safe ID collision")
        seen_safe_ids.add(safe_id)
        raw_hash = sha256(raw_content.encode("utf-8")).hexdigest()
        repaired_hash = sha256(content.encode("utf-8")).hexdigest()
        existing_alias = aliases.get(repaired_hash)
        if existing_alias is not None and existing_alias != raw_hash:
            raise TextbookPrecomputedImportError(
                "relationship vector repair content alias collision"
            )
        if repaired_hash != raw_hash:
            aliases[repaired_hash] = raw_hash
        contents.append(content)
        records.append(
            {
                "__id__": safe_id,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": source_id,
                "content": content,
                "file_path": file_path,
            }
        )

    if not records:
        return []
    with PrecomputedEmbeddingLookup.open(
        archive_dir,
        content_hash_aliases=aliases,
    ) as lookup:
        vectors = await lookup(contents, context="document")
    if len(vectors) != len(records):
        raise TextbookPrecomputedImportError("relationship vector repair embedding count mismatch")
    return [
        {
            **record,
            "vector": _compressed_vector(vector, dimensions=dimensions),
        }
        for record, vector in zip(records, vectors)
    ]


async def _relationship_repair_material(
    settings: TextbookPrecomputedImportSettings,
    *,
    binding: EmbeddingBinding,
    component_paths: Mapping[str, Path],
) -> _RelationshipRepairMaterial:
    sanitize_chunk_text, sanitize_graph_text = _lightrag_text_sanitizers()

    def sanitized_rows() -> Iterator[dict[str, Any]]:
        for _raw, sanitized in _iter_sanitized_relationship_rows(
            component_paths["relationships"],
            sanitize_chunk_text=sanitize_chunk_text,
            sanitize_graph_text=sanitize_graph_text,
        ):
            yield sanitized

    plan = _relationship_collision_plan(sanitized_rows())
    affected_rows = [
        (raw, sanitized)
        for raw, sanitized in _iter_sanitized_relationship_rows(
            component_paths["relationships"],
            sanitize_chunk_text=sanitize_chunk_text,
            sanitize_graph_text=sanitize_graph_text,
        )
        if _canonical_relationship_id(
            sanitized["src_id"],
            sanitized["tgt_id"],
        )
        in plan.dangerous_canonical_ids
    ]
    chunk_store = _ChunkStore(component_paths["chunks"])
    try:
        replacements = await _relationship_replacements(
            affected_rows,
            archive_dir=settings.archive_dir,
            chunk_store=chunk_store,
            dimensions=binding.dimensions,
            sanitize_chunk_text=sanitize_chunk_text,
            sanitize_graph_text=sanitize_graph_text,
        )
    finally:
        chunk_store.close()
    expected_safe_ids = {str(row["__id__"]) for row in replacements}
    if len(expected_safe_ids) != len(affected_rows):
        raise TextbookPrecomputedImportError("relationship vector repair affected count mismatch")
    return _RelationshipRepairMaterial(
        plan=plan,
        affected_count=len(affected_rows),
        replacements=tuple(replacements),
    )


def _uses_postgres_vector_storage() -> bool:
    return (
        os.environ.get(
            "LIGHTRAG_VECTOR_STORAGE",
            "NanoVectorDBStorage",
        )
        == "PGVectorStorage"
    )


def _postgres_relationship_values(
    replacements: Iterable[Mapping[str, Any]],
    *,
    dimensions: int,
) -> tuple[list[tuple[Any, ...]], dict[str, tuple[Any, ...]]]:
    values: list[tuple[Any, ...]] = []
    expected: dict[str, tuple[Any, ...]] = {}
    for replacement in replacements:
        try:
            row_id = str(replacement["__id__"])
            source_id = str(replacement["source_id"])
            src_id = str(replacement["src_id"])
            tgt_id = str(replacement["tgt_id"])
            content = str(replacement["content"])
            file_path = str(replacement["file_path"])
        except (KeyError, TypeError) as error:
            raise TextbookPrecomputedImportError(
                "PostgreSQL relationship vector repair row is invalid"
            ) from error
        if not row_id.startswith("rel-sha256-"):
            raise TextbookPrecomputedImportError(
                "PostgreSQL relationship vector repair row is invalid"
            )
        vector = _decode_relationship_vector(
            replacement.get("vector"),
            dimensions=dimensions,
        ).copy()
        chunk_ids = tuple(source_id.split("<SEP>"))
        values.append(
            (
                row_id,
                src_id,
                tgt_id,
                content,
                vector,
                list(chunk_ids),
                file_path,
            )
        )
        expected[row_id] = (
            src_id,
            tgt_id,
            content,
            chunk_ids,
            file_path,
        )
    if len(values) != len(expected):
        raise TextbookPrecomputedImportError(
            "PostgreSQL relationship vector repair contains duplicate IDs"
        )
    return values, expected


async def _repair_postgres_relationship_vectors(
    storage: Any,
    *,
    generation_id: str,
    model: str,
    dimensions: int,
    expected_count: int,
    material: _RelationshipRepairMaterial,
) -> dict[str, Any]:
    table_name = str(getattr(storage, "table_name", ""))
    workspace = str(getattr(storage, "workspace", ""))
    database = getattr(storage, "db", None)
    safe_model = re.sub(r"[^a-zA-Z0-9_]", "_", model.strip().lower())
    expected_table = f"LIGHTRAG_VDB_RELATION_{safe_model}_{dimensions}d"
    expected_workspace = workspace_for_generation(generation_id)
    if (
        not safe_model
        or workspace != expected_workspace
        or table_name != expected_table
        or len(table_name) > 63
        or database is None
        or not callable(getattr(database, "_run_with_retry", None))
    ):
        raise TextbookPrecomputedImportError(
            "PostgreSQL relationship vector storage is unavailable"
        )
    if expected_count < 0:
        raise TextbookPrecomputedImportError(
            "PostgreSQL relationship vector expected count is invalid"
        )

    dangerous_ids = sorted(material.plan.dangerous_canonical_ids)
    values, expected_rows = _postgres_relationship_values(
        material.replacements,
        dimensions=dimensions,
    )
    safe_ids = sorted(expected_rows)
    upsert_sql = f"""
        INSERT INTO {table_name} (
            workspace, id, source_id, target_id, content, content_vector,
            chunk_ids, file_path, create_time, update_time
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (workspace, id) DO UPDATE
        SET source_id = EXCLUDED.source_id,
            target_id = EXCLUDED.target_id,
            content = EXCLUDED.content,
            content_vector = EXCLUDED.content_vector,
            chunk_ids = EXCLUDED.chunk_ids,
            file_path = EXCLUDED.file_path,
            update_time = CURRENT_TIMESTAMP
    """
    rows_before = 0
    rows_after = 0
    distinct_ids = 0
    distinct_endpoints = 0
    null_vectors = 0
    wrong_dimensions = 0
    deleted_dangerous_count = 0
    safe_rows_verified = 0

    async def repair(connection: Any) -> None:
        nonlocal rows_before
        nonlocal rows_after
        nonlocal distinct_ids
        nonlocal distinct_endpoints
        nonlocal null_vectors
        nonlocal wrong_dimensions
        nonlocal deleted_dangerous_count
        nonlocal safe_rows_verified
        async with connection.transaction(isolation="serializable"):
            await connection.fetchval(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"textbook-precomputed-relationship-repair:{workspace}:{table_name}",
            )
            await connection.execute(f"LOCK TABLE {table_name} IN SHARE ROW EXCLUSIVE MODE")
            rows_before = int(
                await connection.fetchval(
                    f"SELECT COUNT(*) FROM {table_name} WHERE workspace = $1",
                    workspace,
                )
            )
            if dangerous_ids:
                deleted_status = await connection.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE workspace = $1 AND id = ANY($2::varchar[])
                    """,
                    workspace,
                    dangerous_ids,
                )
                try:
                    deleted_dangerous_count = int(str(deleted_status).split()[-1])
                except (ValueError, IndexError) as error:
                    raise TextbookPrecomputedImportError(
                        "PostgreSQL relationship vector delete result is invalid"
                    ) from error
            if values:
                await connection.executemany(
                    upsert_sql,
                    [
                        (
                            workspace,
                            row_id,
                            src_id,
                            tgt_id,
                            content,
                            vector,
                            chunk_ids,
                            file_path,
                        )
                        for (
                            row_id,
                            src_id,
                            tgt_id,
                            content,
                            vector,
                            chunk_ids,
                            file_path,
                        ) in values
                    ],
                )
            aggregate = await connection.fetchrow(
                f"""
                SELECT COUNT(*) AS total_count,
                       COUNT(DISTINCT id) AS distinct_id_count,
                       COUNT(DISTINCT (source_id, target_id))
                           AS distinct_endpoint_count,
                       COUNT(*) FILTER (WHERE content_vector IS NULL)
                           AS null_vector_count,
                       COUNT(*) FILTER (
                           WHERE content_vector IS NOT NULL
                             AND vector_dims(content_vector) <> $2
                       ) AS wrong_dimension_count
                FROM {table_name}
                WHERE workspace = $1
                """,
                workspace,
                dimensions,
            )
            if aggregate is None:
                raise TextbookPrecomputedImportError(
                    "PostgreSQL relationship vector aggregate is unavailable"
                )
            rows_after = int(aggregate["total_count"])
            distinct_ids = int(aggregate["distinct_id_count"])
            distinct_endpoints = int(aggregate["distinct_endpoint_count"])
            null_vectors = int(aggregate["null_vector_count"])
            wrong_dimensions = int(aggregate["wrong_dimension_count"])
            if (
                rows_after != expected_count
                or distinct_ids != expected_count
                or distinct_endpoints != expected_count
                or null_vectors
                or wrong_dimensions
            ):
                raise TextbookPrecomputedImportError(
                    "PostgreSQL relationship vector count mismatch"
                )
            if dangerous_ids:
                remaining_dangerous = int(
                    await connection.fetchval(
                        f"""
                        SELECT COUNT(*) FROM {table_name}
                        WHERE workspace = $1 AND id = ANY($2::varchar[])
                        """,
                        workspace,
                        dangerous_ids,
                    )
                )
                if remaining_dangerous:
                    raise TextbookPrecomputedImportError(
                        "PostgreSQL relationship vector repair left a dangerous ID"
                    )
            if safe_ids:
                safe_rows = await connection.fetch(
                    f"""
                    SELECT id, source_id, target_id, content, chunk_ids, file_path,
                           content_vector IS NOT NULL AS has_vector,
                           vector_dims(content_vector) AS vector_dimensions
                    FROM {table_name}
                    WHERE workspace = $1 AND id = ANY($2::varchar[])
                    """,
                    workspace,
                    safe_ids,
                )
                actual_ids: set[str] = set()
                for row in safe_rows:
                    row_id = str(row["id"])
                    expected = expected_rows.get(row_id)
                    actual = (
                        str(row["source_id"]),
                        str(row["target_id"]),
                        str(row["content"]),
                        tuple(str(item) for item in (row["chunk_ids"] or [])),
                        str(row["file_path"]),
                    )
                    if (
                        expected is None
                        or row_id in actual_ids
                        or actual != expected
                        or not bool(row["has_vector"])
                        or int(row["vector_dimensions"]) != dimensions
                    ):
                        raise TextbookPrecomputedImportError(
                            "PostgreSQL relationship vector provenance mismatch"
                        )
                    actual_ids.add(row_id)
                if actual_ids != set(safe_ids):
                    raise TextbookPrecomputedImportError(
                        "PostgreSQL relationship vector safe ID mismatch"
                    )
                safe_rows_verified = len(actual_ids)

    try:
        await database._run_with_retry(
            repair,
            timing_label=f"{workspace} collision-safe relationship repair",
        )
    except TextbookPrecomputedImportError:
        raise
    except Exception as error:
        raise TextbookPrecomputedImportError(
            "PostgreSQL relationship vector repair failed"
        ) from error

    return {
        "version": _RELATIONSHIP_REPAIR_VERSION,
        "backend": "PGVectorStorage",
        "generation_id": generation_id,
        "model": model,
        "dimensions": dimensions,
        "workspace": workspace,
        "table_name": table_name,
        "affected_relationship_count": material.affected_count,
        "concat_collision_id_count": len(material.plan.concat_collision_ids),
        "reverse_delete_collision_id_count": len(material.plan.reverse_delete_collision_ids),
        "dangerous_canonical_id_count": len(material.plan.dangerous_canonical_ids),
        "deleted_dangerous_count": deleted_dangerous_count,
        "upserted_safe_count": len(values),
        "safe_rows_verified": safe_rows_verified,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "unique_ids": distinct_ids,
        "distinct_endpoint_count": distinct_endpoints,
        "matrix_rows": rows_after,
        "null_vector_count": null_vectors,
        "wrong_dimension_count": wrong_dimensions,
        "graph_mutated": False,
    }


async def _ensure_postgres_relationship_vdb_repair(
    rag: Any,
    settings: TextbookPrecomputedImportSettings,
    *,
    binding: EmbeddingBinding,
    component_paths: Mapping[str, Path],
    expected_count: int,
) -> dict[str, Any]:
    storage = getattr(rag, "relationships_vdb", None)
    if storage is None:
        raise TextbookPrecomputedImportError(
            "PostgreSQL relationship vector storage is unavailable"
        )
    flush = getattr(storage, "index_done_callback", None)
    if not callable(flush):
        raise TextbookPrecomputedImportError("PostgreSQL relationship vector flush is unavailable")
    await flush()
    material = await _relationship_repair_material(
        settings,
        binding=binding,
        component_paths=component_paths,
    )
    return await _repair_postgres_relationship_vectors(
        storage,
        generation_id=binding.generation_id,
        model=binding.model,
        dimensions=binding.dimensions,
        expected_count=expected_count,
        material=material,
    )


def _inspect_relationship_vdb(
    path: Path,
    *,
    dimensions: int,
    expected_count: int,
    dangerous_ids: frozenset[str],
    expected_safe_ids: set[str],
) -> _RelationshipVDBRewriteSummary:
    seen_ids: set[str] = set()
    matrix_rows = 0
    for row in _iter_nanovdb_records(
        path,
        dimensions=dimensions,
        require_matrix=True,
    ):
        row_id = str(row.get("__id__") or "")
        vector = row.get("vector")
        if not row_id or row_id in seen_ids:
            raise TextbookPrecomputedImportError(
                "relationship vector database contains duplicate IDs"
            )
        if not isinstance(vector, str) or not vector:
            raise TextbookPrecomputedImportError(
                "relationship vector database matrix is incomplete"
            )
        seen_ids.add(row_id)
        matrix_rows += 1
    if (
        len(seen_ids) != expected_count
        or matrix_rows != expected_count
        or dangerous_ids.intersection(seen_ids)
        or not expected_safe_ids.issubset(seen_ids)
    ):
        raise TextbookPrecomputedImportError(
            "relationship vector database repair validation failed"
        )
    return _RelationshipVDBRewriteSummary(
        rows_before=len(seen_ids),
        rows_after=len(seen_ids),
        unique_ids=len(seen_ids),
        matrix_rows=matrix_rows,
        deleted_dangerous_count=0,
        inserted_safe_count=0,
    )


def _digest_and_token_count(path: Path, token: bytes) -> tuple[str, int]:
    digest = sha256()
    count = 0
    carry = b""
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            combined = carry + block
            count += combined.count(token)
            carry = combined[-(len(token) - 1) :] if len(token) > 1 else b""
    return digest.hexdigest(), count


def _repair_state_is_current(
    state: Mapping[str, Any],
    *,
    vdb_path: Path,
    graph_path: Path,
    generation_id: str,
    expected_count: int,
) -> bool:
    repair = state.get("relationship_vdb_repair")
    if not isinstance(repair, Mapping):
        return False
    try:
        vdb_stat = vdb_path.stat()
        graph_stat = graph_path.stat()
        return (
            repair.get("version") == _RELATIONSHIP_REPAIR_VERSION
            and repair.get("generation_id") == generation_id
            and int(repair.get("rows_after", -1)) == expected_count
            and int(repair.get("matrix_rows", -1)) == expected_count
            and int(repair.get("unique_ids", -1)) == expected_count
            and int(repair.get("vdb_bytes", -1)) == vdb_stat.st_size
            and int(repair.get("vdb_mtime_ns", -1)) == vdb_stat.st_mtime_ns
            and int(repair.get("graphml_bytes", -1)) == graph_stat.st_size
            and int(repair.get("graphml_mtime_ns", -1)) == graph_stat.st_mtime_ns
        )
    except (OSError, TypeError, ValueError):
        return False


async def _ensure_relationship_vdb_repair(
    settings: TextbookPrecomputedImportSettings,
    *,
    binding: EmbeddingBinding,
    component_paths: Mapping[str, Path],
    expected_count: int,
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    workspace = workspace_for_generation(binding.generation_id)
    workspace_dir = settings.working_dir / workspace
    vdb_path = workspace_dir / "vdb_relationships.json"
    graph_path = workspace_dir / "graph_chunk_entity_relation.graphml"
    if not vdb_path.is_file():
        return None
    if not graph_path.is_file():
        raise TextbookPrecomputedImportError("relationship vector repair graph is unavailable")
    if _repair_state_is_current(
        state,
        vdb_path=vdb_path,
        graph_path=graph_path,
        generation_id=binding.generation_id,
        expected_count=expected_count,
    ):
        repair = state["relationship_vdb_repair"]
        return dict(repair) if isinstance(repair, Mapping) else None

    material = await _relationship_repair_material(
        settings,
        binding=binding,
        component_paths=component_paths,
    )
    plan = material.plan
    replacements = list(material.replacements)
    expected_safe_ids = {str(row["__id__"]) for row in replacements}

    if plan.dangerous_canonical_ids:
        rewrite = _rewrite_relationship_vdb(
            vdb_path,
            dimensions=binding.dimensions,
            expected_count=expected_count,
            dangerous_ids=plan.dangerous_canonical_ids,
            replacements=replacements,
        )
    else:
        rewrite = _inspect_relationship_vdb(
            vdb_path,
            dimensions=binding.dimensions,
            expected_count=expected_count,
            dangerous_ids=frozenset(),
            expected_safe_ids=set(),
        )
    verified = _inspect_relationship_vdb(
        vdb_path,
        dimensions=binding.dimensions,
        expected_count=expected_count,
        dangerous_ids=plan.dangerous_canonical_ids,
        expected_safe_ids=expected_safe_ids,
    )
    graph_digest, graph_edge_count = _digest_and_token_count(graph_path, b"<edge ")
    if graph_edge_count != expected_count:
        raise TextbookPrecomputedImportError("relationship vector repair graph count mismatch")
    vdb_digest = _file_sha256(vdb_path)
    vdb_stat = vdb_path.stat()
    graph_stat = graph_path.stat()
    return {
        "version": _RELATIONSHIP_REPAIR_VERSION,
        "generation_id": binding.generation_id,
        "backend": "NanoVectorDBStorage",
        "affected_relationship_count": material.affected_count,
        "concat_collision_id_count": len(plan.concat_collision_ids),
        "reverse_delete_collision_id_count": len(plan.reverse_delete_collision_ids),
        "dangerous_canonical_id_count": len(plan.dangerous_canonical_ids),
        "deleted_dangerous_count": rewrite.deleted_dangerous_count,
        "inserted_safe_count": rewrite.inserted_safe_count,
        "rows_before": rewrite.rows_before,
        "rows_after": verified.rows_after,
        "matrix_rows": verified.matrix_rows,
        "unique_ids": verified.unique_ids,
        "vdb_sha256": vdb_digest,
        "vdb_bytes": vdb_stat.st_size,
        "vdb_mtime_ns": vdb_stat.st_mtime_ns,
        "graphml_edge_count": graph_edge_count,
        "graphml_sha256": graph_digest,
        "graphml_bytes": graph_stat.st_size,
        "graphml_mtime_ns": graph_stat.st_mtime_ns,
    }


def _default_rag_factory(
    binding: EmbeddingBinding,
    workspace: str,
    lookup: PrecomputedEmbeddingLookup,
    working_dir: Path,
) -> _CustomKGRAG:
    try:
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
    except ImportError as error:  # pragma: no cover - deployment dependency gate
        raise TextbookPrecomputedImportError(
            "LightRAG runtime dependency is unavailable"
        ) from error

    async def unused_llm(*args: Any, **kwargs: Any) -> str:
        raise TextbookPrecomputedImportError("precomputed LightRAG import must not invoke an LLM")

    async def precomputed_embed(
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        return await lookup(texts, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=binding.dimensions,
        func=precomputed_embed,
        max_token_size=binding.max_input_tokens,
        send_dimensions=binding.send_dimensions,
        model_name=binding.model,
        supports_asymmetric=binding.asymmetric,
    )
    working_dir.mkdir(parents=True, exist_ok=True)
    return LightRAG(
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
        llm_model_func=unused_llm,
        llm_model_name="precomputed_custom_kg_no_llm",
        llm_model_max_async=1,
        default_llm_timeout=1,
        auto_manage_storages_states=False,
    )


async def _repair_completed_postgres_import(
    settings: TextbookPrecomputedImportSettings,
    *,
    binding: EmbeddingBinding,
    component_paths: Mapping[str, Path],
    expected_count: int,
    rag_factory: RAGFactory,
) -> dict[str, Any]:
    with PrecomputedEmbeddingLookup.open(settings.archive_dir) as lookup:
        rag = rag_factory(
            binding,
            workspace_for_generation(binding.generation_id),
            lookup,
            settings.working_dir,
        )
        await rag.initialize_storages()
        try:
            return await _ensure_postgres_relationship_vdb_repair(
                rag,
                settings,
                binding=binding,
                component_paths=component_paths,
                expected_count=expected_count,
            )
        finally:
            await rag.finalize_storages()


async def import_textbook_precomputed_lightrag(
    settings: TextbookPrecomputedImportSettings,
    *,
    rag_factory: RAGFactory = _default_rag_factory,
) -> TextbookPrecomputedImportSummary:
    """Replay a verified vector archive through bounded, resumable LightRAG batches."""

    (
        binding,
        component_paths,
        expected_counts,
        bundle_digest,
        archive_digest,
    ) = _validated_contract(settings)
    identity = _state_identity(
        binding,
        bundle_digest=bundle_digest,
        archive_digest=archive_digest,
    )
    state = _load_state(settings, identity)
    summary = TextbookPrecomputedImportSummary(
        generation_id=binding.generation_id,
        model=binding.model,
        chunks=expected_counts["chunks"],
        entities=expected_counts["entities"],
        relationships=expected_counts["relationships"],
        status="completed",
    )
    if state["phase"] == "completed":
        if _uses_postgres_vector_storage():
            repair = await _repair_completed_postgres_import(
                settings,
                binding=binding,
                component_paths=component_paths,
                expected_count=expected_counts["relationships"],
                rag_factory=rag_factory,
            )
        else:
            repair = await _ensure_relationship_vdb_repair(
                settings,
                binding=binding,
                component_paths=component_paths,
                expected_count=expected_counts["relationships"],
                state=state,
            )
        if repair is not None and state.get("relationship_vdb_repair") != repair:
            _write_state(
                settings.state_path,
                {
                    **state,
                    "relationship_vdb_repair": repair,
                },
            )
        return summary

    phase_index = _PHASES.index(str(state["phase"]))
    chunk_store = _ChunkStore(component_paths["chunks"])
    sanitize_chunk_text, sanitize_graph_text = _lightrag_text_sanitizers()
    content_hash_aliases = _lightrag_content_hash_aliases(
        component_paths,
        sanitize_chunk_text=sanitize_chunk_text,
        sanitize_graph_text=sanitize_graph_text,
    )
    postgres_repair: dict[str, Any] | None = None
    try:
        with PrecomputedEmbeddingLookup.open(
            settings.archive_dir,
            content_hash_aliases=content_hash_aliases,
        ) as lookup:
            rag = rag_factory(
                binding,
                workspace_for_generation(binding.generation_id),
                lookup,
                settings.working_dir,
            )
            await rag.initialize_storages()
            try:
                for current_index in range(phase_index, len(_PHASES)):
                    phase = _PHASES[current_index]
                    offset = int(state["offset"]) if current_index == phase_index else 0
                    batch: list[dict[str, Any]] = []
                    completed = offset
                    for row in _iter_rows(component_paths[phase], skip=offset):
                        batch.append(
                            _sanitize_custom_kg_row(
                                phase,
                                row,
                                sanitize_chunk_text=sanitize_chunk_text,
                                sanitize_graph_text=sanitize_graph_text,
                            )
                        )
                        if len(batch) < settings.batch_size:
                            continue
                        chunks = (
                            list(batch)
                            if phase == "chunks"
                            else _source_chunks(
                                batch,
                                chunk_store,
                                sanitize_chunk_text=sanitize_chunk_text,
                                sanitize_graph_text=sanitize_graph_text,
                            )
                        )
                        await rag.ainsert_custom_kg(
                            {
                                "chunks": chunks,
                                "entities": list(batch) if phase == "entities" else [],
                                "relationships": (list(batch) if phase == "relationships" else []),
                            }
                        )
                        completed += len(batch)
                        batch.clear()
                        state = {
                            "schema_version": 1,
                            **identity,
                            "phase": phase,
                            "offset": completed,
                            "status": "running",
                        }
                        _write_state(settings.state_path, state)
                    if batch:
                        chunks = (
                            list(batch)
                            if phase == "chunks"
                            else _source_chunks(
                                batch,
                                chunk_store,
                                sanitize_chunk_text=sanitize_chunk_text,
                                sanitize_graph_text=sanitize_graph_text,
                            )
                        )
                        await rag.ainsert_custom_kg(
                            {
                                "chunks": chunks,
                                "entities": list(batch) if phase == "entities" else [],
                                "relationships": (list(batch) if phase == "relationships" else []),
                            }
                        )
                        completed += len(batch)
                    if completed != expected_counts[phase]:
                        raise TextbookPrecomputedImportError("custom-KG component count mismatch")
                    next_phase = (
                        _PHASES[current_index + 1]
                        if current_index + 1 < len(_PHASES)
                        else "completed"
                    )
                    state_phase = phase if next_phase == "completed" else next_phase
                    state_offset = completed if next_phase == "completed" else 0
                    state = {
                        "schema_version": 1,
                        **identity,
                        "phase": state_phase,
                        "offset": state_offset,
                        "status": "running",
                    }
                    _write_state(settings.state_path, state)
                if _uses_postgres_vector_storage():
                    postgres_repair = await _ensure_postgres_relationship_vdb_repair(
                        rag,
                        settings,
                        binding=binding,
                        component_paths=component_paths,
                        expected_count=expected_counts["relationships"],
                    )
            finally:
                await rag.finalize_storages()
    finally:
        chunk_store.close()
    completed_state: dict[str, Any] = {
        "schema_version": 1,
        **identity,
        "phase": "completed",
        "offset": 0,
        "status": "completed",
    }
    repair = postgres_repair
    if repair is None:
        repair = await _ensure_relationship_vdb_repair(
            settings,
            binding=binding,
            component_paths=component_paths,
            expected_count=expected_counts["relationships"],
            state=completed_state,
        )
    if repair is not None:
        completed_state["relationship_vdb_repair"] = repair
    _write_state(settings.state_path, completed_state)
    return summary
