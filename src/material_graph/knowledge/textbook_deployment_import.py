"""Verified PostgreSQL provenance import for a textbook deployment bundle."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from .catalog import CatalogWriteResult
from .ingestion import build_ingestion_idempotency_key
from .lightrag_models import LightRAGSourceMapping
from .models import EvidenceFragment, SourceCatalogRecord
from .processing import ProcessingCheckpoint


FragmentSource = Literal["derived_chunks", "raw_fragments"]
DERIVED_PROVENANCE_SCHEMA = "material_graph.provenance_import.v1"
DERIVED_PROVENANCE_MODE = "derived_evidence_fragments_v1"
DERIVED_PROVENANCE_ARTIFACTS = (
    "sources",
    "checkpoints",
    "source_mappings",
    "custom_kg_chunks",
)
DERIVED_PROVENANCE_COUNTS = (
    "sources",
    "checkpoints",
    "evidence_fragments",
    "source_mappings",
    "custom_kg_chunks",
)


class TextbookDeploymentImportError(RuntimeError):
    """Stable deployment import failure."""


class _Catalog(Protocol):
    def upsert(
        self,
        record: SourceCatalogRecord,
        *,
        remote_modified_at: object | None = None,
    ) -> CatalogWriteResult: ...


class _Checkpoints(Protocol):
    async def save(self, checkpoint: ProcessingCheckpoint) -> None: ...


class _Evidence(Protocol):
    async def persist_many(
        self,
        source_id: UUID,
        fragments: Sequence[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> None: ...


class _Mappings(Protocol):
    async def persist_many(
        self,
        mappings: Sequence[LightRAGSourceMapping],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class TextbookDeploymentImportSettings:
    bundle_dir: Path
    batch_size: int = 512
    fragment_source: FragmentSource = "derived_chunks"

    def __post_init__(self) -> None:
        if not self.bundle_dir.is_dir() or not self.manifest_path.is_file():
            raise ValueError("deployment bundle is unavailable")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.fragment_source not in ("derived_chunks", "raw_fragments"):
            raise ValueError("fragment_source is invalid")

    @property
    def manifest_path(self) -> Path:
        return self.bundle_dir / "manifest.json"


@dataclass(frozen=True, slots=True)
class TextbookDeploymentImportSummary:
    sources: int
    checkpoints: int
    fragments: int
    source_mappings: int
    generation_id: str
    fragment_source: FragmentSource
    status: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def build_derived_provenance_contract(
    *,
    artifacts: dict[str, dict[str, object]],
    counts: dict[str, int],
    generation_id: str,
    parser_version: str = "local-textbook-v1",
    retention_reason: str = "textbook_full_corpus",
) -> dict[str, Any]:
    """Build the manifest-bound, replayable no-raw provenance contract."""

    return {
        "schema": DERIVED_PROVENANCE_SCHEMA,
        "mode": DERIVED_PROVENANCE_MODE,
        "generation_id": generation_id,
        "content_artifact": "custom_kg_chunks",
        "mapping_artifact": "source_mappings",
        "source_artifact": "sources",
        "checkpoint_artifact": "checkpoints",
        "raw_fragments_required": False,
        "parser_version": parser_version,
        "retention_reason": retention_reason,
        "counts": {name: counts[name] for name in DERIVED_PROVENANCE_COUNTS},
        "artifacts": {name: dict(artifacts[name]) for name in DERIVED_PROVENANCE_ARTIFACTS},
    }


def validate_derived_provenance_contract(
    payload: dict[str, Any],
    *,
    generation_id: str,
) -> dict[str, Any]:
    """Validate that derived provenance is fully replayable from signed artifacts."""

    artifacts = payload.get("artifacts")
    counts = payload.get("counts")
    contract = payload.get("provenance_import")
    if not isinstance(artifacts, dict) or not isinstance(counts, dict):
        raise TextbookDeploymentImportError("deployment manifest is invalid")
    if not isinstance(contract, dict):
        raise TextbookDeploymentImportError("deployment provenance contract is missing")
    expected_scalars = {
        "schema": DERIVED_PROVENANCE_SCHEMA,
        "mode": DERIVED_PROVENANCE_MODE,
        "generation_id": generation_id,
        "content_artifact": "custom_kg_chunks",
        "mapping_artifact": "source_mappings",
        "source_artifact": "sources",
        "checkpoint_artifact": "checkpoints",
        "raw_fragments_required": False,
    }
    if any(contract.get(key) != value for key, value in expected_scalars.items()):
        raise TextbookDeploymentImportError("deployment provenance contract is invalid")
    if not all(
        isinstance(contract.get(name), str) and bool(contract[name])
        for name in ("parser_version", "retention_reason")
    ):
        raise TextbookDeploymentImportError("deployment provenance contract is invalid")

    contract_counts = contract.get("counts")
    if not isinstance(contract_counts, dict) or set(contract_counts) != set(
        DERIVED_PROVENANCE_COUNTS
    ):
        raise TextbookDeploymentImportError("deployment provenance contract count is invalid")
    for name in DERIVED_PROVENANCE_COUNTS:
        value = contract_counts.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or counts.get(name) != value
        ):
            raise TextbookDeploymentImportError("deployment provenance contract count mismatch")
    if (
        contract_counts["sources"] != contract_counts["checkpoints"]
        or contract_counts["evidence_fragments"] != contract_counts["source_mappings"]
        or contract_counts["evidence_fragments"] != contract_counts["custom_kg_chunks"]
    ):
        raise TextbookDeploymentImportError("deployment provenance contract coverage mismatch")

    contract_artifacts = contract.get("artifacts")
    if not isinstance(contract_artifacts, dict) or set(contract_artifacts) != set(
        DERIVED_PROVENANCE_ARTIFACTS
    ):
        raise TextbookDeploymentImportError("deployment provenance artifact is invalid")
    for name in DERIVED_PROVENANCE_ARTIFACTS:
        record = contract_artifacts.get(name)
        if not isinstance(record, dict) or record != artifacts.get(name):
            raise TextbookDeploymentImportError("deployment provenance artifact mismatch")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise TextbookDeploymentImportError("deployment provenance artifact is invalid")
    return contract


def _manifest(settings: TextbookDeploymentImportSettings) -> dict[str, Any]:
    try:
        payload = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
        artifacts = payload["artifacts"]
        generation = payload["embedding"]["generation_id"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise TextbookDeploymentImportError("deployment manifest is invalid") from error
    if not isinstance(artifacts, dict) or not isinstance(generation, str) or not generation:
        raise TextbookDeploymentImportError("deployment manifest is invalid")
    required_artifacts = ["sources", "checkpoints", "source_mappings"]
    required_artifacts.append(
        "custom_kg_chunks" if settings.fragment_source == "derived_chunks" else "fragments"
    )
    for name in required_artifacts:
        try:
            record = artifacts[name]
            relative = str(record["path"])
            expected = str(record["sha256"])
            expected_bytes = int(record["bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise TextbookDeploymentImportError("deployment artifact is invalid") from error
        path = settings.bundle_dir / relative
        if (
            not path.is_file()
            or path.parent.resolve() != settings.bundle_dir.resolve()
            or path.stat().st_size != expected_bytes
            or _file_sha256(path) != expected
        ):
            raise TextbookDeploymentImportError("deployment artifact digest mismatch")
    if settings.fragment_source == "derived_chunks":
        validate_derived_provenance_contract(payload, generation_id=generation)
    return payload


def _jsonl_models(path: Path, model: type[Any]) -> list[Any]:
    values: list[Any] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                values.append(model.model_validate_json(line))
            except Exception as error:
                raise TextbookDeploymentImportError(
                    f"deployment record is invalid at line {line_number}"
                ) from error
    return values


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise TextbookDeploymentImportError(
                    f"deployment record is invalid at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise TextbookDeploymentImportError(
                    f"deployment record is invalid at line {line_number}"
                )
            values.append(payload)
    return values


def _unique_by(
    values: Sequence[Any],
    *,
    field: str,
    label: str,
) -> dict[Any, Any]:
    indexed: dict[Any, Any] = {}
    for value in values:
        identity = getattr(value, field)
        if identity in indexed:
            raise TextbookDeploymentImportError(f"duplicate deployment {label}")
        indexed[identity] = value
    return indexed


def _expected_fragment_id(
    fragment: EvidenceFragment,
    checkpoint: ProcessingCheckpoint,
) -> UUID:
    locator = fragment.locator
    identity = "|".join(
        (
            checkpoint.idempotency_key,
            fragment.content_sha256 or "",
            str(locator.page or 0),
            str(locator.block_index or 0),
            locator.section or "",
        )
    )
    return uuid5(NAMESPACE_URL, identity)


def _materialize_derived_fragments(
    path: Path,
    *,
    sources: Sequence[SourceCatalogRecord],
    checkpoints: Sequence[ProcessingCheckpoint],
    mappings: Sequence[LightRAGSourceMapping],
    generation_id: str,
    manifest: dict[str, Any],
) -> list[EvidenceFragment]:
    source_by_id = _unique_by(sources, field="source_id", label="source")
    checkpoint_by_source = _unique_by(
        checkpoints,
        field="source_id",
        label="source checkpoint",
    )
    mapping_by_fragment = _unique_by(
        mappings,
        field="fragment_id",
        label="source mapping",
    )
    if len({item.basename for item in mappings}) != len(mappings):
        raise TextbookDeploymentImportError("duplicate deployment mapping basename")

    provenance = manifest.get("provenance_import") or {}
    parser_version = str(provenance.get("parser_version") or "local-textbook-v1")
    retention_reason = str(provenance.get("retention_reason") or "textbook_full_corpus")
    fragments: list[EvidenceFragment] = []
    seen_fragment_ids: set[UUID] = set()
    for row in _jsonl_objects(path):
        try:
            fragment_id = UUID(str(row["source_id"]))
            text = row["content"]
            basename = row["file_path"]
        except (KeyError, TypeError, ValueError) as error:
            raise TextbookDeploymentImportError(
                "derived deployment chunk identity is invalid"
            ) from error
        if fragment_id in seen_fragment_ids:
            raise TextbookDeploymentImportError("duplicate derived deployment chunk")
        seen_fragment_ids.add(fragment_id)
        mapping = mapping_by_fragment.get(fragment_id)
        if mapping is None:
            raise TextbookDeploymentImportError("derived deployment chunk has no source mapping")
        if not isinstance(text, str) or not text or basename != mapping.basename:
            raise TextbookDeploymentImportError(
                "derived deployment chunk does not match its source mapping"
            )
        if sha256(text.encode("utf-8")).hexdigest() != mapping.content_sha256:
            raise TextbookDeploymentImportError("derived deployment chunk content hash mismatch")
        source = source_by_id.get(mapping.source_id)
        checkpoint = checkpoint_by_source.get(mapping.source_id)
        if source is None or checkpoint is None:
            raise TextbookDeploymentImportError("derived deployment provenance chain is incomplete")
        if (
            mapping.locator.root_id != source.locator.root_id
            or mapping.locator.relative_path != source.locator.relative_path
        ):
            raise TextbookDeploymentImportError(
                "derived deployment locator does not match its source"
            )
        parser_name = source.metadata.get("source_family")
        if not isinstance(parser_name, str) or not parser_name:
            raise TextbookDeploymentImportError("derived deployment parser identity is unavailable")
        metadata: dict[str, Any] = {
            "materialization": "derived_custom_kg_chunk_v1",
            "logical_title": source.display_title,
            "source_family": parser_name,
        }
        if source.sha256 is not None:
            metadata["document_content_sha256"] = source.sha256
        if source.metadata.get("part_number") is not None:
            metadata["part_number"] = source.metadata["part_number"]
        fragment = EvidenceFragment(
            fragment_id=mapping.fragment_id,
            source_id=mapping.source_id,
            text=text,
            locator=mapping.locator,
            content_sha256=mapping.content_sha256,
            retention_reason=retention_reason,
            parser_name=parser_name,
            parser_version=parser_version,
            embedding_generation_id=generation_id,
            metadata=metadata,
        )
        if fragment.fragment_id != _expected_fragment_id(fragment, checkpoint):
            raise TextbookDeploymentImportError("derived deployment fragment identity mismatch")
        fragments.append(fragment)
    if seen_fragment_ids != set(mapping_by_fragment):
        raise TextbookDeploymentImportError(
            "derived deployment chunk and mapping coverage mismatch"
        )
    return sorted(fragments, key=lambda item: item.fragment_id.hex)


def _validate_provenance_chain(
    *,
    sources: Sequence[SourceCatalogRecord],
    checkpoints: Sequence[ProcessingCheckpoint],
    fragments: Sequence[EvidenceFragment],
    mappings: Sequence[LightRAGSourceMapping],
    generation_id: str,
) -> tuple[
    dict[UUID, ProcessingCheckpoint],
    dict[UUID, list[EvidenceFragment]],
]:
    source_by_id = _unique_by(sources, field="source_id", label="source")
    checkpoint_by_source = _unique_by(
        checkpoints,
        field="source_id",
        label="source checkpoint",
    )
    fragment_by_id = _unique_by(fragments, field="fragment_id", label="fragment")
    mapping_by_fragment = _unique_by(
        mappings,
        field="fragment_id",
        label="source mapping",
    )
    source_ids = set(source_by_id)
    fragments_by_source: dict[UUID, list[EvidenceFragment]] = defaultdict(list)
    for fragment in fragments:
        fragments_by_source[fragment.source_id].append(fragment)
    if (
        set(checkpoint_by_source) != source_ids
        or set(fragments_by_source) != source_ids
        or {item.source_id for item in mappings} != source_ids
        or set(fragment_by_id) != set(mapping_by_fragment)
    ):
        raise TextbookDeploymentImportError("deployment provenance chain is incomplete")
    if len({item.basename for item in mappings}) != len(mappings):
        raise TextbookDeploymentImportError("duplicate deployment mapping basename")

    for source_id, source in source_by_id.items():
        checkpoint = checkpoint_by_source[source_id]
        source_version_key = source.metadata.get("source_version_key")
        if not isinstance(source_version_key, str) or not source_version_key:
            raise TextbookDeploymentImportError("deployment source version identity is unavailable")
        expected_key = build_ingestion_idempotency_key(
            source_id,
            source_version_key=source_version_key,
            embedding_generation_id=generation_id,
        )
        if checkpoint.idempotency_key != expected_key:
            raise TextbookDeploymentImportError("deployment checkpoint identity mismatch")
        if checkpoint.metadata.get(
            "embedding_generation_id"
        ) != generation_id or checkpoint.metadata.get("fragment_count") != len(
            fragments_by_source[source_id]
        ):
            raise TextbookDeploymentImportError("deployment checkpoint provenance mismatch")

    for fragment_id, fragment in fragment_by_id.items():
        mapping = mapping_by_fragment[fragment_id]
        checkpoint = checkpoint_by_source[fragment.source_id]
        if (
            fragment.embedding_generation_id != generation_id
            or mapping.embedding_generation_id != generation_id
            or mapping.source_id != fragment.source_id
            or mapping.locator != fragment.locator
            or mapping.content_sha256 != fragment.content_sha256
            or mapping.logical_source_uri != fragment.locator.to_public_uri(fragment.source_id)
            or fragment.fragment_id != _expected_fragment_id(fragment, checkpoint)
        ):
            raise TextbookDeploymentImportError("deployment fragment and mapping identity mismatch")
    return checkpoint_by_source, fragments_by_source


async def import_textbook_deployment_bundle(
    settings: TextbookDeploymentImportSettings,
    *,
    catalog: _Catalog,
    checkpoints: _Checkpoints,
    evidence: _Evidence,
    mappings: _Mappings,
) -> TextbookDeploymentImportSummary:
    """Validate every artifact, then import the production foreign-key chain."""

    manifest = _manifest(settings)
    artifacts = manifest["artifacts"]

    def artifact(name: str) -> Path:
        return settings.bundle_dir / str(artifacts[name]["path"])

    sources: list[SourceCatalogRecord] = _jsonl_models(
        artifact("sources"),
        SourceCatalogRecord,
    )
    checkpoint_values: list[ProcessingCheckpoint] = _jsonl_models(
        artifact("checkpoints"),
        ProcessingCheckpoint,
    )
    mapping_values: list[LightRAGSourceMapping] = _jsonl_models(
        artifact("source_mappings"),
        LightRAGSourceMapping,
    )
    generation_id = str(manifest["embedding"]["generation_id"])
    if settings.fragment_source == "derived_chunks":
        fragments = _materialize_derived_fragments(
            artifact("custom_kg_chunks"),
            sources=sources,
            checkpoints=checkpoint_values,
            mappings=mapping_values,
            generation_id=generation_id,
            manifest=manifest,
        )
    else:
        fragments = _jsonl_models(artifact("fragments"), EvidenceFragment)

    expected_counts = manifest.get("counts")
    if not isinstance(expected_counts, dict):
        raise TextbookDeploymentImportError("deployment artifact count is invalid")
    expected_fragment_count = expected_counts.get(
        "evidence_fragments",
        expected_counts.get("fragments"),
    )
    expected = {
        "sources": len(sources),
        "checkpoints": len(checkpoint_values),
        "source_mappings": len(mapping_values),
    }
    if any(expected_counts.get(name) != count for name, count in expected.items()):
        raise TextbookDeploymentImportError("deployment artifact count mismatch")
    if expected_fragment_count != len(fragments):
        raise TextbookDeploymentImportError("deployment artifact count mismatch")
    if settings.fragment_source == "derived_chunks" and expected_counts.get(
        "custom_kg_chunks"
    ) != len(fragments):
        raise TextbookDeploymentImportError("deployment artifact count mismatch")

    checkpoints_by_source, fragments_by_source = _validate_provenance_chain(
        sources=sources,
        checkpoints=checkpoint_values,
        fragments=fragments,
        mappings=mapping_values,
        generation_id=generation_id,
    )

    for source in sources:
        catalog.upsert(source, remote_modified_at=None)
    for checkpoint in checkpoint_values:
        await checkpoints.save(checkpoint)
    for source_id in sorted(fragments_by_source, key=lambda item: item.hex):
        source_fragments = fragments_by_source[source_id]
        checkpoint = checkpoints_by_source[source_id]
        for start in range(0, len(source_fragments), settings.batch_size):
            await evidence.persist_many(
                source_id,
                source_fragments[start : start + settings.batch_size],
                idempotency_key=checkpoint.idempotency_key,
            )
    for start in range(0, len(mapping_values), settings.batch_size):
        await mappings.persist_many(mapping_values[start : start + settings.batch_size])

    return TextbookDeploymentImportSummary(
        sources=len(sources),
        checkpoints=len(checkpoint_values),
        fragments=len(fragments),
        source_mappings=len(mapping_values),
        generation_id=generation_id,
        fragment_source=settings.fragment_source,
        status="completed",
    )
