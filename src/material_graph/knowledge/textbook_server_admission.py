"""Build a raw-free, digest-rebound textbook bundle for server admission."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Literal
from uuid import uuid4

from .bindings import EmbeddingBinding
from .textbook_deployment_import import (
    TextbookDeploymentImportError,
    build_derived_provenance_contract,
    validate_derived_provenance_contract,
)


VectorMaterialization = Literal["copy"]
ProvenanceContractValidation = Literal["strict", "legacy_migration"]

_EXPECTED_EMBEDDING = {
    "provider": "glm_openai_compatible",
    "model": "embedding-3",
    "dimensions": 1024,
    "generation_id": "glm-embedding-3-1024-halfvec-v1",
}
_DEPLOYMENT_ARTIFACTS = {
    "sources": ("sources.jsonl", "sources"),
    "checkpoints": ("checkpoints.jsonl", "checkpoints"),
    "source_mappings": ("source-mappings.jsonl", "source_mappings"),
    "custom_kg_chunks": ("custom-kg-chunks.jsonl", "custom_kg_chunks"),
    "custom_kg_entities": ("custom-kg-entities.jsonl", "custom_kg_entities"),
    "custom_kg_relationships": (
        "custom-kg-relationships.jsonl",
        "custom_kg_relationships",
    ),
    "embedding_binding": ("embedding-binding.json", None),
}
_ARCHIVE_ARTIFACTS = {
    "index": "vectors.sqlite3",
    "vectors": "vectors.f16.bin",
}
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


class TextbookServerAdmissionError(RuntimeError):
    """Stable fail-closed error for server-admission bundle construction."""


@dataclass(frozen=True, slots=True)
class TextbookServerAdmissionSettings:
    deployment_bundle: Path
    portable_archive: Path
    output_dir: Path
    vector_mode: VectorMaterialization = "copy"

    def __post_init__(self) -> None:
        _require_ordinary_directory(self.deployment_bundle, label="deployment bundle")
        _require_ordinary_directory(self.portable_archive, label="portable archive")
        _require_ordinary_directory(self.output_dir.parent, label="output parent")
        if os.path.lexists(self.output_dir):
            raise ValueError("server-admission output already exists")
        if self.vector_mode != "copy":
            raise ValueError("server-admission bundles must materialize vectors by copy")
        candidate = self.output_dir.resolve(strict=False)
        for source in (self.deployment_bundle, self.portable_archive):
            resolved = source.resolve()
            if (
                candidate == resolved
                or candidate.is_relative_to(resolved)
                or resolved.is_relative_to(candidate)
            ):
                raise ValueError("server-admission output overlaps an input")


@dataclass(frozen=True, slots=True)
class TextbookServerAdmissionSummary:
    output_dir: str
    canonical_bundle_manifest_sha256: str
    deployment_manifest_sha256: str
    canonical_archive_manifest_sha256: str
    archive_manifest_sha256: str
    generation_id: str
    item_count: int
    vector_count: int
    vector_materialization: VectorMaterialization
    status: str


@dataclass(frozen=True, slots=True)
class TextbookProvenanceContractMigrationSettings:
    deployment_bundle: Path
    portable_archive: Path
    output_dir: Path
    enable_legacy_contract_migration: bool = False

    def __post_init__(self) -> None:
        if not self.enable_legacy_contract_migration:
            raise ValueError("legacy provenance-contract migration is disabled")
        TextbookServerAdmissionSettings(
            deployment_bundle=self.deployment_bundle,
            portable_archive=self.portable_archive,
            output_dir=self.output_dir,
        )


@dataclass(frozen=True, slots=True)
class TextbookProvenanceContractMigrationSummary:
    output_dir: str
    source_deployment_manifest_sha256: str
    source_archive_manifest_sha256: str
    migrated_deployment_manifest_sha256: str
    migrated_archive_manifest_sha256: str
    generation_id: str
    fragment_count: int
    status: str


def _require_ordinary_directory(path: Path, *, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be an ordinary directory")


def _require_ordinary_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise TextbookServerAdmissionError(f"{label} must be an ordinary file")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    _require_ordinary_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise TextbookServerAdmissionError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise TextbookServerAdmissionError(f"{label} is invalid")
    return payload, _file_sha256(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validated_record(
    root: Path,
    record: object,
    *,
    expected_name: str,
    label: str,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, dict):
        raise TextbookServerAdmissionError(f"{label} artifact is invalid")
    try:
        relative = str(record["path"])
        expected_digest = str(record["sha256"])
        expected_bytes = int(record["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise TextbookServerAdmissionError(f"{label} artifact is invalid") from error
    if relative != expected_name or Path(relative).name != relative:
        raise TextbookServerAdmissionError(f"{label} artifact path is invalid")
    if (
        len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or expected_bytes < 0
    ):
        raise TextbookServerAdmissionError(f"{label} artifact record is invalid")
    path = root / relative
    _require_ordinary_file(path, label=f"{label} artifact")
    if (
        path.parent.resolve() != root.resolve()
        or path.stat().st_size != expected_bytes
        or _file_sha256(path) != expected_digest
    ):
        raise TextbookServerAdmissionError(f"{label} artifact digest mismatch")
    return path, {
        "path": expected_name,
        "sha256": expected_digest,
        "bytes": expected_bytes,
    }


def _jsonl_count(path: Path, *, label: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise TextbookServerAdmissionError(
                    f"{label} contains an empty record at line {line_number}"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise TextbookServerAdmissionError(
                    f"{label} contains an invalid record at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise TextbookServerAdmissionError(
                    f"{label} contains an invalid record at line {line_number}"
                )
            count += 1
    return count


def _require_glm_identity(payload: object, *, label: str) -> None:
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in _EXPECTED_EMBEDDING.items()
    ):
        raise TextbookServerAdmissionError(f"{label} GLM embedding identity mismatch")


def _reject_secret_keys(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _SECRET_KEYS:
                raise TextbookServerAdmissionError("embedding binding contains a credential field")
            _reject_secret_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            _reject_secret_keys(value)


def _binding_identity(path: Path) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _reject_secret_keys(raw)
        binding = EmbeddingBinding.model_validate(raw)
    except TextbookServerAdmissionError:
        raise
    except Exception as error:
        raise TextbookServerAdmissionError("embedding binding is invalid") from error
    _require_glm_identity(binding.model_dump(mode="json"), label="binding")


def _integer_count(counts: object, key: str, *, label: str) -> int:
    if not isinstance(counts, dict):
        raise TextbookServerAdmissionError(f"{label} counts are invalid")
    value = counts.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TextbookServerAdmissionError(f"{label} count is invalid: {key}")
    return value


def _validate_vector_index(
    path: Path,
    *,
    expected_items: int,
    expected_vectors: int,
    expected_kinds: dict[str, int],
) -> None:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise TextbookServerAdmissionError("portable vector index integrity failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise TextbookServerAdmissionError("portable vector foreign-key check failed")
        vector_row = connection.execute(
            "SELECT COUNT(*), MIN(vector_index), MAX(vector_index) FROM vectors"
        ).fetchone()
        item_row = connection.execute("SELECT COUNT(*) FROM items").fetchone()
        kind_rows = connection.execute("SELECT kind, COUNT(*) FROM items GROUP BY kind").fetchall()
    except sqlite3.Error as error:
        raise TextbookServerAdmissionError("portable vector index is invalid") from error
    finally:
        connection.close()
    if vector_row is None or item_row is None:
        raise TextbookServerAdmissionError("portable vector index is invalid")
    vector_count, minimum, maximum = (int(value or 0) for value in vector_row)
    item_count = int(item_row[0])
    if (
        vector_count != expected_vectors
        or item_count != expected_items
        or (vector_count and (minimum != 0 or maximum != vector_count - 1))
        or (not vector_count and (minimum is not None or maximum is not None))
    ):
        raise TextbookServerAdmissionError("portable vector count mismatch")
    actual_kinds = {str(kind): int(count) for kind, count in kind_rows}
    if actual_kinds != expected_kinds:
        raise TextbookServerAdmissionError("portable item-kind count mismatch")


def _copy_file(source: Path, destination: Path, *, mode: VectorMaterialization) -> None:
    if mode != "copy":
        raise TextbookServerAdmissionError(
            "server-admission bundles must materialize vectors by copy"
        )
    shutil.copyfile(source, destination, follow_symlinks=False)


def _verify_copy(path: Path, record: dict[str, object], *, label: str) -> None:
    _require_ordinary_file(path, label=label)
    if path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
        raise TextbookServerAdmissionError(f"{label} copy digest mismatch")


def _validate_legacy_provenance_hint(payload: dict[str, Any]) -> None:
    contract = payload.get("provenance_import")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise TextbookServerAdmissionError(
            "legacy deployment provenance materialization is invalid"
        )
    if contract.get("schema") is not None:
        try:
            validate_derived_provenance_contract(
                payload,
                generation_id=_EXPECTED_EMBEDDING["generation_id"],
            )
        except TextbookDeploymentImportError as error:
            raise TextbookServerAdmissionError(
                "legacy deployment provenance materialization is invalid"
            ) from error
        return
    expected = {
        "mode": "derived_evidence_fragments_v1",
        "content_artifact": "custom_kg_chunks",
        "mapping_artifact": "source_mappings",
        "raw_fragments_required": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise TextbookServerAdmissionError(
            "legacy deployment provenance materialization is invalid"
        )
    generation = contract.get("generation_id")
    if generation is not None and generation != _EXPECTED_EMBEDDING["generation_id"]:
        raise TextbookServerAdmissionError(
            "legacy deployment provenance materialization is invalid"
        )


def _validated_inputs(
    settings: TextbookServerAdmissionSettings,
    *,
    provenance_contract_validation: ProvenanceContractValidation = "strict",
) -> dict[str, Any]:
    if provenance_contract_validation not in ("strict", "legacy_migration"):
        raise ValueError("provenance contract validation mode is invalid")
    deployment_manifest, deployment_digest = _read_manifest(
        settings.deployment_bundle / "manifest.json",
        label="canonical deployment manifest",
    )
    archive_manifest, archive_digest = _read_manifest(
        settings.portable_archive / "archive-manifest.json",
        label="canonical archive manifest",
    )
    if archive_manifest.get("bundle_manifest_sha256") != deployment_digest:
        raise TextbookServerAdmissionError(
            "canonical archive and deployment manifest digest mismatch"
        )
    try:
        deployment_artifacts = deployment_manifest["artifacts"]
        deployment_counts = deployment_manifest["counts"]
        archive_artifacts = archive_manifest["artifacts"]
        archive_counts = archive_manifest["counts"]
    except (KeyError, TypeError) as error:
        raise TextbookServerAdmissionError("canonical manifest is incomplete") from error
    if not isinstance(deployment_artifacts, dict) or not isinstance(archive_artifacts, dict):
        raise TextbookServerAdmissionError("canonical artifact manifest is invalid")
    _require_glm_identity(deployment_manifest.get("embedding"), label="deployment")
    _require_glm_identity(archive_manifest.get("embedding"), label="archive")

    deployment_paths: dict[str, Path] = {}
    deployment_records: dict[str, dict[str, object]] = {}
    deployment_actual_counts: dict[str, int] = {}
    for artifact_name, (filename, count_key) in _DEPLOYMENT_ARTIFACTS.items():
        path, record = _validated_record(
            settings.deployment_bundle,
            deployment_artifacts.get(artifact_name),
            expected_name=filename,
            label=f"deployment {artifact_name}",
        )
        deployment_paths[artifact_name] = path
        deployment_records[artifact_name] = record
        if count_key is not None:
            actual = _jsonl_count(path, label=f"deployment {artifact_name}")
            expected = _integer_count(
                deployment_counts,
                count_key,
                label="deployment",
            )
            if actual != expected:
                raise TextbookServerAdmissionError(
                    f"deployment artifact count mismatch: {artifact_name}"
                )
            deployment_actual_counts[count_key] = actual
    _binding_identity(deployment_paths["embedding_binding"])

    fragment_count = _integer_count(
        deployment_counts,
        "evidence_fragments"
        if isinstance(deployment_counts, dict) and "evidence_fragments" in deployment_counts
        else "fragments",
        label="deployment",
    )
    chunks = deployment_actual_counts["custom_kg_chunks"]
    entities = deployment_actual_counts["custom_kg_entities"]
    relationships = deployment_actual_counts["custom_kg_relationships"]
    if (
        fragment_count != chunks
        or deployment_actual_counts["source_mappings"] != chunks
        or _integer_count(deployment_counts, "chunks", label="deployment") != chunks
        or _integer_count(deployment_counts, "entities", label="deployment") != entities
        or _integer_count(deployment_counts, "relationships", label="deployment") != relationships
    ):
        raise TextbookServerAdmissionError("deployment component count mismatch")
    if provenance_contract_validation == "strict":
        try:
            validate_derived_provenance_contract(
                deployment_manifest,
                generation_id=_EXPECTED_EMBEDDING["generation_id"],
            )
        except TextbookDeploymentImportError as error:
            raise TextbookServerAdmissionError(
                "canonical deployment provenance contract is invalid"
            ) from error
    else:
        _validate_legacy_provenance_hint(deployment_manifest)

    archive_paths: dict[str, Path] = {}
    archive_records: dict[str, dict[str, object]] = {}
    for artifact_name, filename in _ARCHIVE_ARTIFACTS.items():
        path, record = _validated_record(
            settings.portable_archive,
            archive_artifacts.get(artifact_name),
            expected_name=filename,
            label=f"archive {artifact_name}",
        )
        archive_paths[artifact_name] = path
        archive_records[artifact_name] = record
    chunk_items = _integer_count(archive_counts, "chunk_items", label="archive")
    entity_items = _integer_count(archive_counts, "entity_items", label="archive")
    relationship_items = _integer_count(
        archive_counts,
        "relationship_items",
        label="archive",
    )
    item_count = _integer_count(archive_counts, "items", label="archive")
    vector_count = _integer_count(archive_counts, "vectors", label="archive")
    if (
        chunk_items != chunks
        or entity_items != entities
        or relationship_items != relationships
        or item_count != chunk_items + entity_items + relationship_items
        or vector_count > item_count
    ):
        raise TextbookServerAdmissionError("archive component count mismatch")
    if archive_paths["vectors"].stat().st_size != vector_count * 1024 * 2:
        raise TextbookServerAdmissionError("portable vector binary size mismatch")
    _validate_vector_index(
        archive_paths["index"],
        expected_items=item_count,
        expected_vectors=vector_count,
        expected_kinds={
            "chunk": chunk_items,
            "entity": entity_items,
            "relationship": relationship_items,
        },
    )
    return {
        "deployment_manifest": deployment_manifest,
        "deployment_digest": deployment_digest,
        "archive_manifest": archive_manifest,
        "archive_digest": archive_digest,
        "deployment_paths": deployment_paths,
        "deployment_records": deployment_records,
        "archive_paths": archive_paths,
        "archive_records": archive_records,
        "deployment_counts": {
            "sources": deployment_actual_counts["sources"],
            "checkpoints": deployment_actual_counts["checkpoints"],
            "evidence_fragments": fragment_count,
            "source_mappings": deployment_actual_counts["source_mappings"],
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "custom_kg_chunks": chunks,
            "custom_kg_entities": entities,
            "custom_kg_relationships": relationships,
        },
        "archive_counts": {
            "chunk_items": chunk_items,
            "entity_items": entity_items,
            "relationship_items": relationship_items,
            "items": item_count,
            "vectors": vector_count,
        },
    }


def migrate_textbook_provenance_contract(
    settings: TextbookProvenanceContractMigrationSettings,
) -> TextbookProvenanceContractMigrationSummary:
    """Explicitly migrate one immutable legacy pair to the strict raw-free contract."""

    validation_settings = TextbookServerAdmissionSettings(
        deployment_bundle=settings.deployment_bundle,
        portable_archive=settings.portable_archive,
        output_dir=settings.output_dir,
    )
    validated = _validated_inputs(
        validation_settings,
        provenance_contract_validation="legacy_migration",
    )
    staging = settings.output_dir.parent / (
        f".{settings.output_dir.name}.provenance-migration-{uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    try:
        deployment_output = staging / "deployment"
        portable_output = staging / "portable"
        deployment_output.mkdir()
        portable_output.mkdir()
        for artifact_name, (filename, _) in _DEPLOYMENT_ARTIFACTS.items():
            destination = deployment_output / filename
            shutil.copyfile(
                validated["deployment_paths"][artifact_name],
                destination,
                follow_symlinks=False,
            )
            _verify_copy(
                destination,
                validated["deployment_records"][artifact_name],
                label=f"migrated deployment {artifact_name}",
            )
        for artifact_name, filename in _ARCHIVE_ARTIFACTS.items():
            destination = portable_output / filename
            _copy_file(
                validated["archive_paths"][artifact_name],
                destination,
                mode="copy",
            )
            _verify_copy(
                destination,
                validated["archive_records"][artifact_name],
                label=f"migrated archive {artifact_name}",
            )

        lineage = {
            "migration": "raw_free_provenance_contract_v1",
            "source_deployment_manifest_sha256": validated["deployment_digest"],
            "source_archive_manifest_sha256": validated["archive_digest"],
        }
        deployment_manifest = {
            "schema_version": 2,
            "embedding": dict(_EXPECTED_EMBEDDING),
            "provenance_import": build_derived_provenance_contract(
                artifacts=validated["deployment_records"],
                counts=validated["deployment_counts"],
                generation_id=_EXPECTED_EMBEDDING["generation_id"],
            ),
            "lineage": lineage,
            "counts": validated["deployment_counts"],
            "artifacts": validated["deployment_records"],
        }
        deployment_manifest_path = deployment_output / "manifest.json"
        _write_json(deployment_manifest_path, deployment_manifest)
        deployment_digest = _file_sha256(deployment_manifest_path)

        archive_manifest = {
            **validated["archive_manifest"],
            "schema_version": 2,
            "bundle_manifest_sha256": deployment_digest,
            "rebound_from_bundle_manifest_sha256": validated["deployment_digest"],
            "lineage": lineage,
            "embedding": validated["archive_manifest"]["embedding"],
            "counts": validated["archive_counts"],
            "artifacts": validated["archive_records"],
            "vector_materialization": "copy",
        }
        archive_manifest_path = portable_output / "archive-manifest.json"
        _write_json(archive_manifest_path, archive_manifest)
        archive_digest = _file_sha256(archive_manifest_path)

        summary_payload = {
            "schema_version": 1,
            "status": "completed",
            "generation_id": _EXPECTED_EMBEDDING["generation_id"],
            "lineage": lineage,
            "manifests": {
                "deployment": {
                    "path": "deployment/manifest.json",
                    "sha256": deployment_digest,
                },
                "portable": {
                    "path": "portable/archive-manifest.json",
                    "sha256": archive_digest,
                },
            },
            "raw_fragments_copied": False,
            "embedding_api_calls": 0,
        }
        _write_json(staging / "migration-summary.json", summary_payload)

        strict_settings = TextbookServerAdmissionSettings(
            deployment_bundle=deployment_output,
            portable_archive=portable_output,
            output_dir=staging / "strict-validation-output",
        )
        _validated_inputs(strict_settings)
        _secure_output_permissions(staging)
        if os.path.lexists(settings.output_dir):
            raise ValueError("provenance migration output already exists")
        os.replace(staging, settings.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return TextbookProvenanceContractMigrationSummary(
        output_dir=settings.output_dir.resolve().as_posix(),
        source_deployment_manifest_sha256=validated["deployment_digest"],
        source_archive_manifest_sha256=validated["archive_digest"],
        migrated_deployment_manifest_sha256=deployment_digest,
        migrated_archive_manifest_sha256=archive_digest,
        generation_id=_EXPECTED_EMBEDDING["generation_id"],
        fragment_count=validated["deployment_counts"]["evidence_fragments"],
        status="completed",
    )


def _write_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != checksum_path)
    expected = {
        "deployment/checkpoints.jsonl",
        "deployment/custom-kg-chunks.jsonl",
        "deployment/custom-kg-entities.jsonl",
        "deployment/custom-kg-relationships.jsonl",
        "deployment/embedding-binding.json",
        "deployment/manifest.json",
        "deployment/source-mappings.jsonl",
        "deployment/sources.jsonl",
        "portable/archive-manifest.json",
        "portable/vectors.f16.bin",
        "portable/vectors.sqlite3",
        "summary.json",
    }
    relative_files = {path.relative_to(root).as_posix() for path in files}
    if relative_files != expected or any(path.is_symlink() for path in files):
        raise TextbookServerAdmissionError("server-admission output whitelist mismatch")
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        for path in files:
            relative = path.relative_to(root).as_posix()
            stream.write(f"{_file_sha256(path)}  {relative}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _secure_output_permissions(root: Path) -> None:
    if os.name != "posix":
        return
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise TextbookServerAdmissionError("server-admission output contains a symbolic link")
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def build_textbook_server_admission(
    settings: TextbookServerAdmissionSettings,
) -> TextbookServerAdmissionSummary:
    """Validate immutable canonical inputs and atomically publish a derived bundle."""

    validated = _validated_inputs(settings)
    staging = settings.output_dir.parent / (f".{settings.output_dir.name}.staging-{uuid4().hex}")
    staging.mkdir(mode=0o700)
    try:
        deployment_output = staging / "deployment"
        portable_output = staging / "portable"
        deployment_output.mkdir()
        portable_output.mkdir()

        for artifact_name, (filename, _) in _DEPLOYMENT_ARTIFACTS.items():
            source = validated["deployment_paths"][artifact_name]
            destination = deployment_output / filename
            shutil.copyfile(source, destination, follow_symlinks=False)
            _verify_copy(
                destination,
                validated["deployment_records"][artifact_name],
                label=f"deployment {artifact_name}",
            )

        for artifact_name, filename in _ARCHIVE_ARTIFACTS.items():
            source = validated["archive_paths"][artifact_name]
            destination = portable_output / filename
            _copy_file(source, destination, mode=settings.vector_mode)
            _verify_copy(
                destination,
                validated["archive_records"][artifact_name],
                label=f"archive {artifact_name}",
            )

        provenance_contract = build_derived_provenance_contract(
            artifacts=validated["deployment_records"],
            counts=validated["deployment_counts"],
            generation_id=_EXPECTED_EMBEDDING["generation_id"],
        )
        deployment_manifest = {
            "schema_version": 2,
            "embedding": dict(_EXPECTED_EMBEDDING),
            "provenance_import": provenance_contract,
            "lineage": {
                "canonical_manifest_sha256": validated["deployment_digest"],
                "canonical_archive_manifest_sha256": validated["archive_digest"],
            },
            "counts": validated["deployment_counts"],
            "artifacts": validated["deployment_records"],
        }
        deployment_manifest_path = deployment_output / "manifest.json"
        _write_json(deployment_manifest_path, deployment_manifest)
        deployment_digest = _file_sha256(deployment_manifest_path)

        canonical_archive = validated["archive_manifest"]
        archive_manifest = {
            "schema_version": 2,
            "bundle_manifest_sha256": deployment_digest,
            "rebound_from_bundle_manifest_sha256": validated["deployment_digest"],
            "lineage": {
                "canonical_bundle_manifest_sha256": validated["deployment_digest"],
                "canonical_archive_manifest_sha256": validated["archive_digest"],
            },
            "embedding": canonical_archive["embedding"],
            "counts": validated["archive_counts"],
            "artifacts": validated["archive_records"],
            "vector_materialization": settings.vector_mode,
        }
        archive_manifest_path = portable_output / "archive-manifest.json"
        _write_json(archive_manifest_path, archive_manifest)
        archive_digest = _file_sha256(archive_manifest_path)

        summary_payload = {
            "schema_version": 1,
            "status": "completed",
            "embedding": dict(_EXPECTED_EMBEDDING),
            "lineage": {
                "canonical_bundle_manifest_sha256": validated["deployment_digest"],
                "canonical_archive_manifest_sha256": validated["archive_digest"],
            },
            "manifests": {
                "deployment": {
                    "path": "deployment/manifest.json",
                    "sha256": deployment_digest,
                },
                "portable": {
                    "path": "portable/archive-manifest.json",
                    "sha256": archive_digest,
                },
            },
            "counts": {
                **validated["deployment_counts"],
                **validated["archive_counts"],
            },
            "vector_materialization": settings.vector_mode,
        }
        _write_json(staging / "summary.json", summary_payload)
        _write_checksums(staging)
        _secure_output_permissions(staging)
        if os.path.lexists(settings.output_dir):
            raise ValueError("server-admission output already exists")
        os.replace(staging, settings.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return TextbookServerAdmissionSummary(
        output_dir=settings.output_dir.resolve().as_posix(),
        canonical_bundle_manifest_sha256=validated["deployment_digest"],
        deployment_manifest_sha256=deployment_digest,
        canonical_archive_manifest_sha256=validated["archive_digest"],
        archive_manifest_sha256=archive_digest,
        generation_id=_EXPECTED_EMBEDDING["generation_id"],
        item_count=validated["archive_counts"]["items"],
        vector_count=validated["archive_counts"]["vectors"],
        vector_materialization=settings.vector_mode,
        status="completed",
    )
