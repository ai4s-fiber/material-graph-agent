import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from material_graph.knowledge.bindings import ProviderBindings
from material_graph.knowledge.models import EvidenceFragment, SourceLocator


def test_source_locator_public_uri_hides_remote_path():
    source_id = uuid4()
    locator = SourceLocator(
        root_id="document_data_1",
        relative_path="信智学院文献数据/pdf_by_material_type/01_polymer/a.pdf",
        page=7,
        section="Results",
    )

    public_uri = locator.to_public_uri(source_id)

    assert public_uri.startswith(f"source://document_data_1/{source_id}#")
    assert "page=7" in public_uri
    assert "relative_path" not in public_uri
    assert "信智学院文献数据" not in public_uri


@pytest.mark.parametrize(
    "relative_path",
    [
        "/volume1/private/a.pdf",
        r"C:\private\a.pdf",
        r"\\nas\share\a.pdf",
        "../outside/a.pdf",
        "https://example.invalid/a.pdf",
    ],
)
def test_source_locator_rejects_non_relative_or_remote_paths(relative_path):
    with pytest.raises(ValidationError):
        SourceLocator(root_id="document_data_1", relative_path=relative_path)


def test_evidence_fragment_hash_is_reproducible():
    source_id = uuid4()
    fragment = EvidenceFragment(
        source_id=source_id,
        text="含氟结构在给定测试条件下降低介电常数。",
        locator=SourceLocator(
            root_id="document_data_1",
            relative_path="paper.pdf",
            page=3,
        ),
        retention_reason="supports:dielectric_constant",
        supported_entity_ids=["entity:fluorinated_group"],
        supported_relation_ids=["relation:decreases_dielectric_constant"],
        parser_name="mineru",
        parser_version="3.4.4",
        embedding_generation_id="glm-embedding-3-1024-halfvec-v1",
    )

    same = fragment.model_copy(update={"fragment_id": uuid4()})
    assert fragment.content_sha256 == same.content_sha256
    assert len(fragment.content_sha256) == 64


def test_provider_bindings_freeze_glm_embedding_and_qwen_reranker():
    bindings = ProviderBindings.load(
        embedding_path=Path("config/knowledge/embedding-binding.v1.json"),
        reranker_path=Path("config/knowledge/reranker-binding.v1.json"),
    )

    assert bindings.embedding.provider == "glm_openai_compatible"
    assert bindings.embedding.model == "embedding-3"
    assert bindings.embedding.generation_id == "glm-embedding-3-1024-halfvec-v1"
    assert bindings.embedding.dimensions == 1024
    assert bindings.embedding.postgres_vector_index_type == "HNSW_HALFVEC"
    assert bindings.embedding.send_dimensions is True
    assert bindings.reranker.model == "Qwen/Qwen3-Reranker-8B"
    assert bindings.reranker.endpoint.endswith("/v1/rerank")
    assert bindings.reranker.max_async >= 8


def test_canonical_glm_embedding_matches_verified_binding_and_disables_failover():
    canonical = ProviderBindings.load(
        embedding_path=Path("config/knowledge/embedding-binding.v1.json"),
        reranker_path=Path("config/knowledge/reranker-binding.v1.json"),
    ).embedding
    explicit = ProviderBindings.load(
        embedding_path=Path("config/knowledge/embedding-binding.glm.v1.json"),
        reranker_path=Path("config/knowledge/reranker-binding.v1.json"),
    ).embedding
    failover = json.loads(
        Path("config/knowledge/embedding-failover.v1.json").read_text(encoding="utf-8")
    )

    assert canonical == explicit
    assert failover["strategy"] == "disabled"
    assert failover["primary"] == {
        "binding_path": "config/knowledge/embedding-binding.v1.json",
        "provider": "glm_openai_compatible",
        "generation_id": "glm-embedding-3-1024-halfvec-v1",
    }
    assert failover["fallback"] is None
    assert failover["activation"]["allowed_reasons"] == []
    assert failover["activation"]["allow_mixed_generations"] is False


def test_knowledge_package_exports_stable_provider_neutral_contracts():
    import material_graph.knowledge as knowledge

    expected = {
        "CheckpointRepository",
        "CorpusPolicy",
        "EvidenceFactExtractionPipeline",
        "EntityRef",
        "EvidenceFragment",
        "FactBatch",
        "GlobalKnowledgeGraphWriter",
        "GraphWriteApproval",
        "EvidenceRepository",
        "MetadataCursor",
        "MetadataManifestIngestor",
        "KnowledgeIngestionPipeline",
        "KnowledgeCanaryService",
        "PostgresCheckpointRepository",
        "PostgresAGEGlobalKnowledgeGraphWriter",
        "PostgresEvidenceRepository",
        "PostgresSourceCatalogRepository",
        "ProcessingCheckpoint",
        "PropertyObservation",
        "RemoteRangeContractError",
        "RemoteSourceReader",
        "SelectionPolicy",
        "SourceCatalogRepository",
        "build_ingestion_idempotency_key",
        "build_source_version_key",
    }

    assert expected <= set(knowledge.__all__)
    assert all(hasattr(knowledge, name) for name in expected)
