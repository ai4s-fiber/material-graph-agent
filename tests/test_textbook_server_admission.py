from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from typing import Any

import pytest

from material_graph.knowledge import textbook_server_admission as admission
from material_graph.knowledge.textbook_deployment_import import (
    build_derived_provenance_contract,
)
from material_graph.knowledge.textbook_server_admission import (
    TextbookProvenanceContractMigrationSettings,
    TextbookServerAdmissionError,
    TextbookServerAdmissionSettings,
    build_textbook_server_admission,
    migrate_textbook_provenance_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BINDING = ROOT / "config/knowledge/embedding-binding.v1.json"
EMBEDDING = {
    "provider": "glm_openai_compatible",
    "model": "embedding-3",
    "dimensions": 1024,
    "generation_id": "glm-embedding-3-1024-halfvec-v1",
}
DEPLOYMENT_FILES = {
    "sources": "sources.jsonl",
    "checkpoints": "checkpoints.jsonl",
    "source_mappings": "source-mappings.jsonl",
    "custom_kg_chunks": "custom-kg-chunks.jsonl",
    "custom_kg_entities": "custom-kg-entities.jsonl",
    "custom_kg_relationships": "custom-kg-relationships.jsonl",
    "embedding_binding": "embedding-binding.json",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    deployment = tmp_path / "canonical-deployment"
    portable = tmp_path / "canonical-portable"
    deployment.mkdir()
    portable.mkdir()

    payloads = {
        "sources": {"source_id": "source-1"},
        "checkpoints": {"checkpoint_id": "checkpoint-1"},
        "source_mappings": {"fragment_id": "fragment-1"},
        "custom_kg_chunks": {"content": "derived chunk"},
        "custom_kg_entities": {"entity_name": "PET"},
        "custom_kg_relationships": {"src_id": "PET", "tgt_id": "fiber"},
    }
    for name, payload in payloads.items():
        _write_jsonl(deployment / DEPLOYMENT_FILES[name], payload)
    (deployment / DEPLOYMENT_FILES["embedding_binding"]).write_bytes(BINDING.read_bytes())
    forbidden_fragments = deployment / "fragments.jsonl"
    forbidden_graph = deployment / "lightrag-custom-kg.json"
    forbidden_fragments.write_text("forbidden raw fixture\n", encoding="utf-8")
    forbidden_graph.write_text('{"forbidden":"redundant"}\n', encoding="utf-8")

    artifacts = {
        name: _record(deployment / filename) for name, filename in DEPLOYMENT_FILES.items()
    }
    artifacts["fragments"] = _record(forbidden_fragments)
    artifacts["lightrag_custom_kg"] = _record(forbidden_graph)
    counts = {
        "sources": 1,
        "checkpoints": 1,
        "evidence_fragments": 1,
        "source_mappings": 1,
        "chunks": 1,
        "entities": 1,
        "relationships": 1,
        "custom_kg_chunks": 1,
        "custom_kg_entities": 1,
        "custom_kg_relationships": 1,
    }
    deployment_manifest = {
        "schema_version": 2,
        "embedding": EMBEDDING,
        "provenance_import": build_derived_provenance_contract(
            artifacts=artifacts,
            counts=counts,
            generation_id=EMBEDDING["generation_id"],
        ),
        "counts": counts,
        "artifacts": artifacts,
    }
    deployment_manifest_path = deployment / "manifest.json"
    _write_json(deployment_manifest_path, deployment_manifest)

    index = portable / "vectors.sqlite3"
    connection = sqlite3.connect(index)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE vectors (
            content_sha256 TEXT PRIMARY KEY,
            vector_index INTEGER UNIQUE NOT NULL
        );
        CREATE TABLE items (
            kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            PRIMARY KEY (kind, item_id),
            FOREIGN KEY (content_sha256) REFERENCES vectors(content_sha256)
        );
        """
    )
    content_hash = "a" * 64
    connection.execute("INSERT INTO vectors VALUES (?, ?)", (content_hash, 0))
    connection.executemany(
        "INSERT INTO items VALUES (?, ?, ?)",
        [
            ("chunk", "chunk-1", content_hash),
            ("entity", "entity-1", content_hash),
            ("relationship", "relationship-1", content_hash),
        ],
    )
    connection.commit()
    connection.close()
    vectors = portable / "vectors.f16.bin"
    vectors.write_bytes(bytes(1024 * 2))
    archive_manifest = {
        "schema_version": 1,
        "bundle_manifest_sha256": _digest(deployment_manifest_path),
        "embedding": {
            **EMBEDDING,
            "distance": "cosine",
            "dtype": "float16-little-endian",
            "normalized": True,
        },
        "counts": {
            "chunk_items": 1,
            "entity_items": 1,
            "relationship_items": 1,
            "items": 3,
            "vectors": 1,
        },
        "artifacts": {
            "index": _record(index),
            "vectors": _record(vectors),
        },
    }
    _write_json(portable / "archive-manifest.json", archive_manifest)
    return deployment, portable


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_deployment_manifest(
    deployment: Path,
    portable: Path,
    mutate: Any,
) -> None:
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    _write_json(manifest_path, manifest)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["bundle_manifest_sha256"] = _digest(manifest_path)
    _write_json(archive_path, archive)


def _make_legacy_pair(
    deployment: Path,
    portable: Path,
    *,
    keep_legacy_hint: bool = True,
) -> tuple[str, str]:
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if keep_legacy_hint:
        manifest["provenance_import"] = {
            "mode": "derived_evidence_fragments_v1",
            "content_artifact": "custom_kg_chunks",
            "mapping_artifact": "source_mappings",
            "raw_fragments_required": False,
            "parser_version": "local-textbook-v1",
            "retention_reason": "textbook_full_corpus",
        }
    else:
        manifest.pop("provenance_import")
    _write_json(manifest_path, manifest)
    deployment_digest = _digest(manifest_path)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["bundle_manifest_sha256"] = deployment_digest
    _write_json(archive_path, archive)
    return deployment_digest, _digest(archive_path)


def test_builds_deterministic_raw_free_rebound_bundle(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    source_deployment = _tree_snapshot(deployment)
    source_portable = _tree_snapshot(portable)
    old_bundle_digest = _digest(deployment / "manifest.json")

    first = build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=tmp_path / "admission-one",
            vector_mode="copy",
        )
    )
    second = build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=tmp_path / "admission-two",
            vector_mode="copy",
        )
    )

    assert _tree_snapshot(deployment) == source_deployment
    assert _tree_snapshot(portable) == source_portable
    first_root = Path(first.output_dir)
    second_root = Path(second.output_dir)
    for relative in (
        "deployment/manifest.json",
        "portable/archive-manifest.json",
        "summary.json",
        "SHA256SUMS",
    ):
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()

    new_bundle = json.loads((first_root / "deployment/manifest.json").read_text(encoding="utf-8"))
    new_archive = json.loads(
        (first_root / "portable/archive-manifest.json").read_text(encoding="utf-8")
    )
    new_bundle_digest = _digest(first_root / "deployment/manifest.json")
    assert new_bundle["schema_version"] == 2
    assert new_bundle["provenance_import"]["mode"] == ("derived_evidence_fragments_v1")
    assert new_bundle["provenance_import"]["schema"] == ("material_graph.provenance_import.v1")
    assert new_bundle["provenance_import"]["generation_id"] == EMBEDDING["generation_id"]
    assert new_bundle["provenance_import"]["counts"] == {
        "sources": 1,
        "checkpoints": 1,
        "evidence_fragments": 1,
        "source_mappings": 1,
        "custom_kg_chunks": 1,
    }
    assert new_bundle["provenance_import"]["artifacts"] == {
        name: new_bundle["artifacts"][name]
        for name in ("sources", "checkpoints", "source_mappings", "custom_kg_chunks")
    }
    assert new_bundle["lineage"]["canonical_manifest_sha256"] == old_bundle_digest
    assert new_bundle["lineage"]["canonical_archive_manifest_sha256"] == _digest(
        portable / "archive-manifest.json"
    )
    assert new_archive["bundle_manifest_sha256"] == new_bundle_digest
    assert new_archive["rebound_from_bundle_manifest_sha256"] == old_bundle_digest
    assert new_archive["lineage"] == {
        "canonical_bundle_manifest_sha256": old_bundle_digest,
        "canonical_archive_manifest_sha256": _digest(portable / "archive-manifest.json"),
    }
    assert new_archive["vector_materialization"] == "copy"
    assert new_bundle_digest != old_bundle_digest
    assert "fragments" not in new_bundle["artifacts"]
    assert "lightrag_custom_kg" not in new_bundle["artifacts"]
    assert not list(first_root.rglob("fragments.jsonl"))
    assert not list(first_root.rglob("lightrag-custom-kg.json"))
    assert "qwen" not in (first_root / "summary.json").read_text(encoding="utf-8").casefold()


def test_explicit_legacy_migration_builds_strict_raw_free_pair(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    source_deployment_digest, source_archive_digest = _make_legacy_pair(
        deployment,
        portable,
    )
    source_deployment = _tree_snapshot(deployment)
    source_portable = _tree_snapshot(portable)
    output = tmp_path / "migrated"

    summary = migrate_textbook_provenance_contract(
        TextbookProvenanceContractMigrationSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
            enable_legacy_contract_migration=True,
        )
    )

    assert _tree_snapshot(deployment) == source_deployment
    assert _tree_snapshot(portable) == source_portable
    assert summary.source_deployment_manifest_sha256 == source_deployment_digest
    assert summary.source_archive_manifest_sha256 == source_archive_digest
    migrated_deployment = json.loads(
        (output / "deployment/manifest.json").read_text(encoding="utf-8")
    )
    migrated_archive = json.loads(
        (output / "portable/archive-manifest.json").read_text(encoding="utf-8")
    )
    assert migrated_deployment["provenance_import"]["schema"] == (
        "material_graph.provenance_import.v1"
    )
    assert migrated_deployment["lineage"] == {
        "migration": "raw_free_provenance_contract_v1",
        "source_deployment_manifest_sha256": source_deployment_digest,
        "source_archive_manifest_sha256": source_archive_digest,
    }
    assert migrated_archive["bundle_manifest_sha256"] == (
        summary.migrated_deployment_manifest_sha256
    )
    assert migrated_archive["lineage"] == migrated_deployment["lineage"]
    assert not list(output.rglob("fragments.jsonl"))
    assert not list(output.rglob("lightrag-custom-kg.json"))

    admitted = build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=output / "deployment",
            portable_archive=output / "portable",
            output_dir=tmp_path / "admission",
        )
    )
    assert admitted.status == "completed"


def test_legacy_migration_is_disabled_by_default(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    _make_legacy_pair(deployment, portable, keep_legacy_hint=False)

    with pytest.raises(ValueError, match="migration is disabled"):
        TextbookProvenanceContractMigrationSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=tmp_path / "migrated",
        )


def test_legacy_migration_rejects_tamper_before_output(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    _make_legacy_pair(deployment, portable, keep_legacy_hint=False)
    with (deployment / "custom-kg-chunks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"tampered":true}\n')
    output = tmp_path / "migrated"

    with pytest.raises(TextbookServerAdmissionError, match="digest"):
        migrate_textbook_provenance_contract(
            TextbookProvenanceContractMigrationSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
                enable_legacy_contract_migration=True,
            )
        )
    assert not output.exists()


def test_legacy_migration_rejects_contradictory_hint(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    _make_legacy_pair(deployment, portable)
    _rewrite_deployment_manifest(
        deployment,
        portable,
        lambda manifest: manifest["provenance_import"].__setitem__("raw_fragments_required", True),
    )

    with pytest.raises(TextbookServerAdmissionError, match="materialization"):
        migrate_textbook_provenance_contract(
            TextbookProvenanceContractMigrationSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "migrated",
                enable_legacy_contract_migration=True,
            )
        )


def test_legacy_migration_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment, portable = _fixture(tmp_path)
    _make_legacy_pair(deployment, portable)
    output = tmp_path / "migrated"
    real_write_json = admission._write_json
    calls = 0

    def _fail_second_write(path: Path, payload: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced migration failure")
        real_write_json(path, payload)

    monkeypatch.setattr(admission, "_write_json", _fail_second_write)
    with pytest.raises(OSError, match="forced migration failure"):
        migrate_textbook_provenance_contract(
            TextbookProvenanceContractMigrationSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
                enable_legacy_contract_migration=True,
            )
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".migrated.provenance-migration-*"))


def test_legacy_migration_cli_requires_explicit_flag_and_succeeds(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    _make_legacy_pair(deployment, portable)
    script = ROOT / "scripts/migrate_textbook_provenance_contract.py"
    output = tmp_path / "migrated"
    base = [
        sys.executable,
        str(script),
        "--deployment-bundle",
        str(deployment),
        "--portable-archive",
        str(portable),
        "--output-dir",
        str(output),
    ]

    disabled = subprocess.run(
        base,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert disabled.returncode == 2
    assert not output.exists()

    enabled = subprocess.run(
        [*base, "--enable-legacy-contract-migration"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert enabled.returncode == 0, enabled.stderr
    assert json.loads(enabled.stdout)["status"] == "completed"


def test_selected_artifact_digest_mismatch_fails_without_output(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    with (deployment / "custom-kg-chunks.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"tampered":true}\n')
    output = tmp_path / "admission"

    with pytest.raises(TextbookServerAdmissionError, match="digest"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
                vector_mode="copy",
            )
        )
    assert not output.exists()


def test_generation_mismatch_fails_without_output(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["embedding"]["generation_id"] = "qwen-forbidden-generation"
    _write_json(archive_path, archive)
    output = tmp_path / "admission"

    with pytest.raises(TextbookServerAdmissionError, match="GLM embedding identity"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
                vector_mode="copy",
            )
        )
    assert not output.exists()


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    manifest_path = deployment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["sources"]["path"] = "../sources.jsonl"
    _write_json(manifest_path, manifest)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["bundle_manifest_sha256"] = _digest(manifest_path)
    _write_json(archive_path, archive)

    with pytest.raises(TextbookServerAdmissionError, match="artifact path"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "admission",
                vector_mode="copy",
            )
        )


def test_existing_target_is_rejected(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
        )


def test_hardlink_mode_is_rejected_for_production_admission(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"
    with pytest.raises(ValueError, match="materialize vectors by copy"):
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
            vector_mode="hardlink",  # type: ignore[arg-type]
        )
    assert not output.exists()


def test_copy_mode_materializes_independent_vector_files(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"
    summary = build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
            vector_mode="copy",
        )
    )

    assert not os.path.samefile(
        portable / "vectors.f16.bin",
        output / "portable/vectors.f16.bin",
    )
    assert not os.path.samefile(
        portable / "vectors.sqlite3",
        output / "portable/vectors.sqlite3",
    )
    assert summary.vector_materialization == "copy"
    on_disk = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["vector_materialization"] == "copy"


def test_default_vector_mode_is_copy(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"
    summary = build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
        )
    )

    assert summary.vector_materialization == "copy"
    assert not os.path.samefile(
        portable / "vectors.sqlite3",
        output / "portable/vectors.sqlite3",
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits required")
def test_output_tree_is_not_group_or_other_writable(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"
    build_textbook_server_admission(
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=output,
        )
    )

    assert all(
        not path.stat(follow_symlinks=False).st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        for path in (output, *output.rglob("*"))
    )


def test_symlinked_selected_artifact_is_rejected(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    sources = deployment / "sources.jsonl"
    target = tmp_path / "source-target.jsonl"
    target.write_bytes(sources.read_bytes())
    sources.unlink()
    try:
        sources.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable: {type(error).__name__}")

    with pytest.raises(TextbookServerAdmissionError, match="ordinary file"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "admission",
                vector_mode="copy",
            )
        )


def test_settings_reject_missing_and_overlapping_input_trees(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)

    with pytest.raises(ValueError, match="deployment bundle"):
        TextbookServerAdmissionSettings(
            deployment_bundle=tmp_path / "missing",
            portable_archive=portable,
            output_dir=tmp_path / "output",
        )

    with pytest.raises(ValueError, match="overlaps an input"):
        TextbookServerAdmissionSettings(
            deployment_bundle=deployment,
            portable_archive=portable,
            output_dir=deployment / "nested-output",
        )


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_manifest_reader_rejects_malformed_or_non_object_json(
    tmp_path: Path,
    contents: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(contents, encoding="utf-8")

    with pytest.raises(TextbookServerAdmissionError, match="manifest is invalid"):
        admission._read_manifest(manifest, label="manifest")


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (None, "artifact is invalid"),
        ({}, "artifact is invalid"),
        (
            {"path": "payload.jsonl", "sha256": "short", "bytes": 1},
            "artifact record is invalid",
        ),
    ],
)
def test_artifact_record_validation_rejects_incomplete_metadata(
    tmp_path: Path,
    record: object,
    message: str,
) -> None:
    with pytest.raises(TextbookServerAdmissionError, match=message):
        admission._validated_record(
            tmp_path,
            record,
            expected_name="payload.jsonl",
            label="fixture",
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("\n", "empty record"),
        ("{\n", "invalid record"),
        ("[]\n", "invalid record"),
    ],
)
def test_jsonl_validation_rejects_empty_malformed_and_non_object_rows(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(TextbookServerAdmissionError, match=message):
        admission._jsonl_count(path, label="fixture")


def test_secret_scan_rejects_credentials_nested_inside_lists() -> None:
    with pytest.raises(TextbookServerAdmissionError, match="credential field"):
        admission._reject_secret_keys({"providers": [{"api-key": "do-not-ship"}]})


@pytest.mark.parametrize("counts", [None, {"items": True}, {"items": -1}])
def test_integer_count_rejects_missing_boolean_and_negative_values(counts: object) -> None:
    with pytest.raises(TextbookServerAdmissionError, match="counts are invalid|count is invalid"):
        admission._integer_count(counts, "items", label="archive")


def test_vector_index_rejects_missing_schema_and_inconsistent_kind_counts(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.sqlite3"
    sqlite3.connect(malformed).close()
    with pytest.raises(TextbookServerAdmissionError, match="index is invalid"):
        admission._validate_vector_index(
            malformed,
            expected_items=0,
            expected_vectors=0,
            expected_kinds={},
        )

    valid = tmp_path / "valid.sqlite3"
    connection = sqlite3.connect(valid)
    connection.executescript(
        """
        CREATE TABLE vectors (
            content_sha256 TEXT PRIMARY KEY,
            vector_index INTEGER UNIQUE NOT NULL
        );
        CREATE TABLE items (
            kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            PRIMARY KEY (kind, item_id)
        );
        INSERT INTO vectors VALUES ('aaaaaaaa', 0);
        INSERT INTO items VALUES ('chunk', 'chunk-1', 'aaaaaaaa');
        """
    )
    connection.close()

    with pytest.raises(TextbookServerAdmissionError, match="item-kind count mismatch"):
        admission._validate_vector_index(
            valid,
            expected_items=1,
            expected_vectors=1,
            expected_kinds={"entity": 1},
        )


def test_copy_and_copy_verification_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"source")

    with pytest.raises(TextbookServerAdmissionError, match="materialize vectors by copy"):
        admission._copy_file(source, destination, mode="hardlink")  # type: ignore[arg-type]

    destination.write_bytes(b"tampered")
    with pytest.raises(TextbookServerAdmissionError, match="copy digest mismatch"):
        admission._verify_copy(
            destination,
            {"bytes": len(b"source"), "sha256": _digest(source)},
            label="fixture",
        )


def test_checksum_writer_rejects_non_whitelisted_output(tmp_path: Path) -> None:
    (tmp_path / "unexpected.txt").write_text("not admitted", encoding="utf-8")

    with pytest.raises(TextbookServerAdmissionError, match="whitelist mismatch"):
        admission._write_checksums(tmp_path)


def test_binding_validation_preserves_security_errors_and_normalizes_parse_errors(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "secret-binding.json"
    _write_json(secret, {"api_key": "do-not-ship"})
    with pytest.raises(TextbookServerAdmissionError, match="credential field"):
        admission._binding_identity(secret)

    malformed = tmp_path / "malformed-binding.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(TextbookServerAdmissionError, match="binding is invalid"):
        admission._binding_identity(malformed)


def test_vector_index_rejects_dangling_foreign_keys_and_count_mismatch(
    tmp_path: Path,
) -> None:
    index = tmp_path / "vectors.sqlite3"
    connection = sqlite3.connect(index)
    connection.executescript(
        """
        CREATE TABLE vectors (
            content_sha256 TEXT PRIMARY KEY,
            vector_index INTEGER UNIQUE NOT NULL
        );
        CREATE TABLE items (
            kind TEXT NOT NULL,
            item_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            PRIMARY KEY (kind, item_id),
            FOREIGN KEY (content_sha256) REFERENCES vectors(content_sha256)
        );
        INSERT INTO items VALUES ('chunk', 'chunk-1', 'missing-vector');
        """
    )
    connection.close()

    with pytest.raises(TextbookServerAdmissionError, match="foreign-key check failed"):
        admission._validate_vector_index(
            index,
            expected_items=1,
            expected_vectors=0,
            expected_kinds={"chunk": 1},
        )

    connection = sqlite3.connect(index)
    connection.execute("DELETE FROM items")
    connection.execute("INSERT INTO vectors VALUES ('aaaaaaaa', 0)")
    connection.commit()
    connection.close()
    with pytest.raises(TextbookServerAdmissionError, match="vector count mismatch"):
        admission._validate_vector_index(
            index,
            expected_items=0,
            expected_vectors=0,
            expected_kinds={},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.pop("counts"), "manifest is incomplete"),
        (lambda manifest: manifest.__setitem__("artifacts", []), "artifact manifest is invalid"),
        (
            lambda manifest: manifest["counts"].__setitem__("sources", 2),
            "artifact count mismatch",
        ),
        (
            lambda manifest: manifest["counts"].__setitem__("evidence_fragments", 2),
            "component count mismatch",
        ),
    ],
)
def test_deployment_manifest_validation_fails_closed(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    deployment, portable = _fixture(tmp_path)
    _rewrite_deployment_manifest(deployment, portable, mutate)

    with pytest.raises(TextbookServerAdmissionError, match=message):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "admission",
            )
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.pop("provenance_import"),
        lambda manifest: manifest["provenance_import"].__setitem__(
            "generation_id", "wrong-generation"
        ),
        lambda manifest: manifest["provenance_import"]["counts"].__setitem__(
            "evidence_fragments", 2
        ),
        lambda manifest: manifest["provenance_import"]["artifacts"]["custom_kg_chunks"].__setitem__(
            "bytes", 0
        ),
    ],
)
def test_server_admission_requires_replayable_provenance_contract(
    tmp_path: Path,
    mutate: Any,
) -> None:
    deployment, portable = _fixture(tmp_path)
    _rewrite_deployment_manifest(deployment, portable, mutate)

    with pytest.raises(
        TextbookServerAdmissionError,
        match="provenance contract",
    ):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "admission",
            )
        )


def test_archive_manifest_requires_exact_binding_and_component_counts(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["bundle_manifest_sha256"] = "0" * 64
    _write_json(archive_path, archive)
    with pytest.raises(TextbookServerAdmissionError, match="manifest digest mismatch"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "digest-mismatch",
            )
        )

    component_root = tmp_path / "component"
    component_root.mkdir()
    deployment, portable = _fixture(component_root)
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["counts"]["chunk_items"] = 2
    _write_json(archive_path, archive)
    with pytest.raises(TextbookServerAdmissionError, match="component count mismatch"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "component-mismatch",
            )
        )


def test_archive_vector_binary_size_must_match_manifest(tmp_path: Path) -> None:
    deployment, portable = _fixture(tmp_path)
    vectors = portable / "vectors.f16.bin"
    vectors.write_bytes(b"truncated")
    archive_path = portable / "archive-manifest.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive["artifacts"]["vectors"] = _record(vectors)
    _write_json(archive_path, archive)

    with pytest.raises(TextbookServerAdmissionError, match="binary size mismatch"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=tmp_path / "admission",
            )
        )


def test_failed_publish_removes_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"

    def _fail_checksums(_root: Path) -> None:
        raise TextbookServerAdmissionError("forced checksum failure")

    monkeypatch.setattr(admission, "_write_checksums", _fail_checksums)
    with pytest.raises(TextbookServerAdmissionError, match="forced checksum failure"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
            )
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".admission.staging-*"))


def test_publish_detects_output_created_during_atomic_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment, portable = _fixture(tmp_path)
    output = tmp_path / "admission"

    def _simulate_concurrent_publisher(_staging: Path) -> None:
        output.mkdir()

    monkeypatch.setattr(admission, "_secure_output_permissions", _simulate_concurrent_publisher)
    with pytest.raises(ValueError, match="output already exists"):
        build_textbook_server_admission(
            TextbookServerAdmissionSettings(
                deployment_bundle=deployment,
                portable_archive=portable,
                output_dir=output,
            )
        )

    assert output.is_dir()
    assert not list(tmp_path.glob(".admission.staging-*"))
