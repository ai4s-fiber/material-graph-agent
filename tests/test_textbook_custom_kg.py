from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from material_graph.knowledge.bindings import EmbeddingBinding
from material_graph.knowledge.models import EvidenceFragment, SourceLocator
from material_graph.knowledge.processing import ProcessingCheckpoint
from material_graph.knowledge.textbook_custom_kg import (
    TextbookCustomKGIndexSettings,
    index_textbook_custom_kg,
)
from material_graph.knowledge.textbook_lightrag import TextbookLightRAGError


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "config/knowledge/embedding-binding.v1.json"
FAILOVER = ROOT / "config/knowledge/embedding-failover.v1.json"


def _settings(
    tmp_path: Path,
    *,
    failover_policy: Path = FAILOVER,
) -> TextbookCustomKGIndexSettings:
    custom = tmp_path / "custom.json"
    custom.write_text(
        json.dumps({"chunks": [], "entities": [], "relationships": []}),
        encoding="utf-8",
    )
    fragments = tmp_path / "fragments.jsonl"
    fragments.write_text("", encoding="utf-8")
    return TextbookCustomKGIndexSettings(
        custom_kg_path=custom,
        fragments_path=fragments,
        working_dir=tmp_path / "rag",
        deployment_dir=tmp_path / "deployment",
        primary_embedding_binding_path=PRIMARY,
        failover_policy_path=failover_policy,
        state_path=tmp_path / "state.json",
    )


def _quota_failover_policy(tmp_path: Path) -> Path:
    path = tmp_path / "test-quota-failover.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "strategy": "quota_exhaustion_only",
                "primary": {
                    "binding_path": "config/knowledge/embedding-binding.v1.json",
                    "provider": "glm_openai_compatible",
                    "generation_id": "glm-embedding-3-1024-halfvec-v1",
                },
                "fallback": {
                    "provider": "test_fallback_openai_compatible",
                    "base_url": "https://fallback.invalid/v1",
                    "model_candidates": ["test-fallback-embedding"],
                    "credential_env": "TEST_FALLBACK_API_KEY",
                    "dimensions": 1024,
                    "generation_id": "test-fallback-embedding-1024-v1",
                    "verify_on_activation": True,
                },
                "activation": {
                    "allowed_reasons": ["insufficient_balance", "quota_exhausted"],
                    "switch_at_checkpoint_boundary": True,
                    "require_full_reembedding": True,
                    "allow_mixed_generations": False,
                    "preflight_before_primary_exhaustion": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_generation_bundle_rebinds_fragment_identity_and_basename(
    tmp_path: Path,
) -> None:
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
        embedding_generation_id="prepared-generation",
        metadata={
            "document_content_sha256": "a" * 64,
            "logical_title": "化学纤维工艺学",
            "part_number": 1,
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
                "entities": [
                    {
                        "entity_name": "PET",
                        "entity_type": "Material",
                        "description": "聚酯",
                        "source_id": str(fragment.fragment_id),
                        "file_path": "raw.md",
                    }
                ],
                "relationships": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = TextbookCustomKGIndexSettings(
        custom_kg_path=custom,
        fragments_path=fragments,
        working_dir=tmp_path / "rag",
        deployment_dir=tmp_path / "deployment",
        primary_embedding_binding_path=PRIMARY,
        failover_policy_path=FAILOVER,
        state_path=tmp_path / "state.json",
    )

    async def runner(
        binding: EmbeddingBinding,
        key: str,
        workspace: str,
        payload: dict[str, Any],
        working_dir: Path,
    ) -> None:
        chunk = payload["chunks"][0]
        assert chunk["source_id"] != str(fragment.fragment_id)
        assert chunk["file_path"].startswith(f"mg_{fragment.source_id.hex}_")
        assert chunk["file_path"].endswith(".txt")
        assert payload["entities"][0]["file_path"] == chunk["file_path"]

    summary = await index_textbook_custom_kg(
        settings,
        environment={"MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret"},
        generation_runner=runner,
    )

    bundle = Path(summary.deployment_bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["sources"] == 1
    assert manifest["schema_version"] == 2
    assert manifest["counts"]["evidence_fragments"] == 1
    provenance = manifest["provenance_import"]
    assert provenance["schema"] == "material_graph.provenance_import.v1"
    assert provenance["mode"] == "derived_evidence_fragments_v1"
    assert provenance["generation_id"] == summary.generation_id
    assert provenance["raw_fragments_required"] is False
    assert provenance["counts"] == {
        "sources": 1,
        "checkpoints": 1,
        "evidence_fragments": 1,
        "source_mappings": 1,
        "custom_kg_chunks": 1,
    }
    assert provenance["artifacts"] == {
        name: manifest["artifacts"][name]
        for name in ("sources", "checkpoints", "source_mappings", "custom_kg_chunks")
    }
    assert manifest["counts"]["custom_kg_chunks"] == 1
    assert manifest["counts"]["custom_kg_entities"] == 1
    assert manifest["counts"]["custom_kg_relationships"] == 0
    assert manifest["artifacts"]["custom_kg_chunks"]["path"] == ("custom-kg-chunks.jsonl")
    assert manifest["artifacts"]["custom_kg_entities"]["path"] == ("custom-kg-entities.jsonl")
    assert manifest["artifacts"]["custom_kg_relationships"]["path"] == (
        "custom-kg-relationships.jsonl"
    )
    selected_binding = EmbeddingBinding.model_validate_json(
        (bundle / manifest["artifacts"]["embedding_binding"]["path"]).read_text(encoding="utf-8")
    )
    assert selected_binding.generation_id == summary.generation_id
    assert "fragments" not in manifest["artifacts"]
    assert not (bundle / "fragments.jsonl").exists()
    assert "lightrag_custom_kg" not in manifest["artifacts"]
    assert not (bundle / "lightrag-custom-kg.json").exists()
    chunk = json.loads((bundle / "custom-kg-chunks.jsonl").read_text(encoding="utf-8"))
    mapping = json.loads((bundle / "source-mappings.jsonl").read_text(encoding="utf-8"))
    checkpoint = ProcessingCheckpoint.model_validate_json(
        (bundle / "checkpoints.jsonl").read_text(encoding="utf-8")
    )
    assert chunk["source_id"] == mapping["fragment_id"]
    assert mapping["basename"].startswith("mg_")
    assert checkpoint.selection is not None
    assert checkpoint.selection.selected


@pytest.mark.asyncio
async def test_primary_embedding_is_used_without_preflight_or_fallback(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    probes = 0

    async def runner(
        binding: EmbeddingBinding,
        key: str,
        workspace: str,
        payload: dict[str, Any],
        working_dir: Path,
    ) -> None:
        calls.append((binding.provider, workspace))
        assert key == "primary-secret"
        assert payload["chunks"] == []

    async def probe(binding: EmbeddingBinding, key: str) -> None:
        nonlocal probes
        probes += 1

    summary = await index_textbook_custom_kg(
        _settings(tmp_path),
        environment={"MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret"},
        generation_runner=runner,
        fallback_probe=probe,
    )

    assert not summary.failover_activated
    assert Path(summary.deployment_bundle).is_dir()
    assert calls == [("glm_openai_compatible", "glm-embedding-3-1024-halfvec-v1")]
    assert probes == 0


@pytest.mark.asyncio
async def test_canonical_policy_disables_quota_failover(
    tmp_path: Path,
) -> None:
    probes = 0

    async def runner(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("code 30002: insufficient balance")

    async def probe(binding: EmbeddingBinding, key: str) -> None:
        nonlocal probes
        probes += 1

    with pytest.raises(TextbookLightRAGError, match="primary embedding"):
        await index_textbook_custom_kg(
            _settings(tmp_path),
            environment={"MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret"},
            generation_runner=runner,
            fallback_probe=probe,
        )
    assert probes == 0


@pytest.mark.asyncio
async def test_wrapped_quota_exhaustion_activates_fallback(tmp_path: Path) -> None:
    class Attempt:
        def exception(self) -> BaseException:
            return RuntimeError("code 30002: insufficient balance")

    class RetryError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("embedding retry exhausted")
            self.last_attempt = Attempt()

    calls: list[str] = []

    async def runner(
        binding: EmbeddingBinding,
        key: str,
        workspace: str,
        payload: dict[str, Any],
        working_dir: Path,
    ) -> None:
        calls.append(binding.provider)
        if binding.provider == "glm_openai_compatible":
            raise RetryError()

    async def probe(binding: EmbeddingBinding, key: str) -> None:
        assert binding.model == "test-fallback-embedding"
        assert key == "fallback-secret"

    summary = await index_textbook_custom_kg(
        _settings(tmp_path, failover_policy=_quota_failover_policy(tmp_path)),
        environment={
            "MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret",
            "TEST_FALLBACK_API_KEY": "fallback-secret",
        },
        generation_runner=runner,
        fallback_probe=probe,
    )

    assert summary.failover_activated
    assert calls == [
        "glm_openai_compatible",
        "test_fallback_openai_compatible",
    ]


@pytest.mark.asyncio
async def test_transient_rate_limit_does_not_activate_fallback(tmp_path: Path) -> None:
    probes = 0

    async def runner(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("rate limit exceeded")

    async def probe(binding: EmbeddingBinding, key: str) -> None:
        nonlocal probes
        probes += 1

    with pytest.raises(TextbookLightRAGError, match="primary embedding"):
        await index_textbook_custom_kg(
            _settings(tmp_path),
            environment={"MATERIAL_GRAPH_EMBEDDING_API_KEY": "primary-secret"},
            generation_runner=runner,
            fallback_probe=probe,
        )
    assert probes == 0
