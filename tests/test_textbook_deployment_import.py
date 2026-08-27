from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from material_graph.knowledge.catalog import CatalogWriteResult
from material_graph.knowledge.lightrag_models import LightRAGSourceMapping
from material_graph.knowledge.models import (
    EvidenceFragment,
    SourceCatalogRecord,
    SourceLocator,
)
from material_graph.knowledge.processing import ProcessingCheckpoint
from material_graph.knowledge.textbook_custom_kg import (
    TextbookCustomKGIndexSettings,
    index_textbook_custom_kg,
)
from material_graph.knowledge.textbook_deployment_import import (
    TextbookDeploymentImportError,
    TextbookDeploymentImportSettings,
    import_textbook_deployment_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "config/knowledge/embedding-binding.v1.json"
FAILOVER = ROOT / "config/knowledge/embedding-failover.v1.json"


async def _bundle(tmp_path: Path, *, include_legacy_fragments: bool = False) -> Path:
    fragment = EvidenceFragment(
        fragment_id=UUID(int=1),
        source_id=UUID(int=2),
        text="PET 经牵伸形成取向结构。",
        locator=SourceLocator(
            root_id="cyj_source_hu",
            relative_path="source_hu/book.md",
            page=1,
            section="纺丝",
            block_index=0,
        ),
        retention_reason="textbook_full_corpus",
        parser_name="source_hu_markdown",
        parser_version="local-textbook-v1",
        embedding_generation_id="prepared",
        metadata={
            "document_content_sha256": "a" * 64,
            "logical_title": "化学纤维工艺学",
            "source_family": "source_hu_markdown",
        },
    )
    fragments = tmp_path / "fragments.jsonl"
    fragments.write_text(fragment.model_dump_json() + "\n", encoding="utf-8")
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "content": fragment.text,
                        "source_id": str(fragment.fragment_id),
                        "file_path": "raw.md",
                    }
                ],
                "entities": [],
                "relationships": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    async def runner(*args: Any, **kwargs: Any) -> None:
        return None

    summary = await index_textbook_custom_kg(
        TextbookCustomKGIndexSettings(
            custom_kg_path=custom,
            fragments_path=fragments,
            working_dir=tmp_path / "rag",
            deployment_dir=tmp_path / "deployment",
            primary_embedding_binding_path=PRIMARY,
            failover_policy_path=FAILOVER,
            state_path=tmp_path / "state.json",
        ),
        environment={"MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret"},
        generation_runner=runner,
    )
    bundle = Path(summary.deployment_bundle)
    if include_legacy_fragments:
        mapping = LightRAGSourceMapping.model_validate_json(
            (bundle / "source-mappings.jsonl").read_text(encoding="utf-8")
        )
        rebound = fragment.model_copy(
            update={
                "fragment_id": mapping.fragment_id,
                "embedding_generation_id": mapping.embedding_generation_id,
            }
        )
        raw_path = bundle / "fragments.jsonl"
        raw_path.write_text(rebound.model_dump_json() + "\n", encoding="utf-8")
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"]["fragments"] = {
            "path": raw_path.name,
            "sha256": sha256(raw_path.read_bytes()).hexdigest(),
            "bytes": raw_path.stat().st_size,
        }
        manifest["counts"]["fragments"] = 1
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return bundle


class _Catalog:
    def __init__(self) -> None:
        self.values: list[SourceCatalogRecord] = []

    def upsert(
        self,
        record: SourceCatalogRecord,
        *,
        remote_modified_at: object | None = None,
    ) -> CatalogWriteResult:
        self.values.append(record)
        return CatalogWriteResult(
            record=record,
            created=True,
            source_version_key=str(record.metadata["source_version_key"]),
        )


class _Checkpoints:
    def __init__(self) -> None:
        self.values: list[ProcessingCheckpoint] = []

    async def save(self, checkpoint: ProcessingCheckpoint) -> None:
        self.values.append(checkpoint)


class _Evidence:
    def __init__(self) -> None:
        self.values: list[EvidenceFragment] = []

    async def persist_many(
        self,
        source_id: UUID,
        fragments: list[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> None:
        assert all(item.source_id == source_id for item in fragments)
        assert idempotency_key.startswith("knowledge-ingestion:v2:")
        self.values.extend(fragments)


class _Mappings:
    def __init__(self) -> None:
        self.values: list[LightRAGSourceMapping] = []

    async def persist_many(self, mappings: list[LightRAGSourceMapping]) -> None:
        self.values.extend(mappings)


@pytest.mark.asyncio
async def test_verified_bundle_imports_derived_fragments_in_foreign_key_order(
    tmp_path: Path,
) -> None:
    bundle = await _bundle(tmp_path)
    catalog = _Catalog()
    checkpoints = _Checkpoints()
    evidence = _Evidence()
    mappings = _Mappings()

    summary = await import_textbook_deployment_bundle(
        TextbookDeploymentImportSettings(bundle_dir=bundle, batch_size=1),
        catalog=catalog,
        checkpoints=checkpoints,
        evidence=evidence,
        mappings=mappings,
    )

    assert summary.status == "completed"
    assert summary.sources == summary.checkpoints == summary.fragments == 1
    assert len(catalog.values) == 1
    assert len(checkpoints.values) == 1
    assert len(evidence.values) == 1
    assert len(mappings.values) == 1
    assert mappings.values[0].fragment_id == evidence.values[0].fragment_id


@pytest.mark.asyncio
async def test_default_import_does_not_require_raw_fragments(tmp_path: Path) -> None:
    bundle = await _bundle(tmp_path)
    assert not (bundle / "fragments.jsonl").exists()
    evidence = _Evidence()

    summary = await import_textbook_deployment_bundle(
        TextbookDeploymentImportSettings(bundle_dir=bundle),
        catalog=_Catalog(),
        checkpoints=_Checkpoints(),
        evidence=evidence,
        mappings=_Mappings(),
    )

    assert summary.fragment_source == "derived_chunks"
    assert summary.fragments == 1
    assert evidence.values[0].text == "PET 经牵伸形成取向结构。"
    assert (
        evidence.values[0].content_sha256
        == sha256(evidence.values[0].text.encode("utf-8")).hexdigest()
    )
    assert evidence.values[0].parser_name == "source_hu_markdown"
    assert evidence.values[0].parser_version == "local-textbook-v1"
    assert evidence.values[0].metadata["materialization"] == ("derived_custom_kg_chunk_v1")


@pytest.mark.asyncio
async def test_explicit_legacy_mode_still_reads_fragments(tmp_path: Path) -> None:
    bundle = await _bundle(tmp_path, include_legacy_fragments=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("provenance_import")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = await import_textbook_deployment_bundle(
        TextbookDeploymentImportSettings(
            bundle_dir=bundle,
            fragment_source="raw_fragments",
        ),
        catalog=_Catalog(),
        checkpoints=_Checkpoints(),
        evidence=_Evidence(),
        mappings=_Mappings(),
    )
    assert summary.fragment_source == "raw_fragments"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.pop("provenance_import"),
        lambda manifest: manifest["provenance_import"].__setitem__(
            "generation_id", "wrong-generation"
        ),
        lambda manifest: manifest["provenance_import"]["counts"].__setitem__("source_mappings", 2),
        lambda manifest: manifest["provenance_import"]["artifacts"]["source_mappings"].__setitem__(
            "sha256", "0" * 64
        ),
    ],
)
async def test_derived_import_requires_manifest_bound_provenance_contract(
    tmp_path: Path,
    mutate: Any,
) -> None:
    bundle = await _bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    catalog = _Catalog()

    with pytest.raises(TextbookDeploymentImportError, match="provenance"):
        await import_textbook_deployment_bundle(
            TextbookDeploymentImportSettings(bundle_dir=bundle),
            catalog=catalog,
            checkpoints=_Checkpoints(),
            evidence=_Evidence(),
            mappings=_Mappings(),
        )
    assert catalog.values == []


@pytest.mark.asyncio
async def test_tampered_bundle_fails_before_any_import(tmp_path: Path) -> None:
    bundle = await _bundle(tmp_path)
    with (bundle / "sources.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    catalog = _Catalog()

    with pytest.raises(TextbookDeploymentImportError, match="digest"):
        await import_textbook_deployment_bundle(
            TextbookDeploymentImportSettings(bundle_dir=bundle),
            catalog=catalog,
            checkpoints=_Checkpoints(),
            evidence=_Evidence(),
            mappings=_Mappings(),
        )
    assert catalog.values == []
