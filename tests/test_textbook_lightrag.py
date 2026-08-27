from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from material_graph.knowledge.models import EvidenceFragment, SourceLocator
from material_graph.knowledge.textbook_lightrag import (
    AsyncLLMProviderPool,
    LocalTextbookLightRAGSettings,
    TextbookLLMPoolBinding,
    fragment_document_id,
    fragment_file_path,
    index_local_textbook_fragments,
    iter_textbook_document_batches,
    load_textbook_llm_pool,
)


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING = ROOT / "config/knowledge/embedding-binding.v1.json"
LLM_POOL = ROOT / "config/knowledge/textbook-llm-pool.v1.json"


def _fragment(index: int) -> EvidenceFragment:
    return EvidenceFragment(
        fragment_id=UUID(int=index + 1),
        source_id=UUID(int=100 + index),
        text=f"聚合物材料片段 {index}",
        locator=SourceLocator(
            root_id="cyj_source_hu",
            relative_path="source_hu/第1批/聚合物加工/聚合物加工.md",
            page=index + 1,
            section="加工原理",
            block_index=index,
        ),
        retention_reason="textbook_full_corpus",
        parser_name="source_hu_markdown",
        parser_version="local-textbook-v1",
        embedding_generation_id="embedding-test",
        metadata={
            "chunk_index": index,
            "logical_title": "聚合物/加工",
            "page_end": index + 1,
            "part_number": 1,
            "source_family": "source_hu_markdown",
        },
    )


def _write_fragments(path: Path, count: int) -> list[EvidenceFragment]:
    fragments = [_fragment(index) for index in range(count)]
    path.write_text(
        "\n".join(item.model_dump_json() for item in fragments) + "\n",
        encoding="utf-8",
    )
    return fragments


def test_fragment_identity_and_citation_path_are_unique_and_drive_free() -> None:
    first = _fragment(0)
    second = _fragment(1)

    assert fragment_document_id(first) == f"doc-textbook-{first.fragment_id.hex}"
    assert fragment_file_path(first) != fragment_file_path(second)
    assert fragment_file_path(first).startswith("textbook_fragments/cyj_source_hu/")
    assert "聚合物_加工" in fragment_file_path(first)
    assert "E:" not in fragment_file_path(first)
    assert "\\" not in fragment_file_path(first)


def test_batches_stream_with_limit(tmp_path: Path) -> None:
    source = tmp_path / "fragments.jsonl"
    fragments = _write_fragments(source, 5)

    batches = list(iter_textbook_document_batches(source, batch_size=2, limit=3))

    assert [len(batch) for batch in batches] == [2, 1]
    assert [item.fragment.fragment_id for batch in batches for item in batch] == [
        fragment.fragment_id for fragment in fragments[:3]
    ]


def test_llm_pool_binding_freezes_maximum_parallel_capacity() -> None:
    binding = load_textbook_llm_pool(LLM_POOL)

    assert binding.generation_id == "textbook-material-entity-pool-v10"
    assert binding.total_concurrency == 16
    assert [provider.model for provider in binding.providers] == ["deepseek-v4-flash"]
    assert all("api_key" not in provider.model_dump() for provider in binding.providers)


@pytest.mark.asyncio
async def test_provider_pool_uses_capacity_lanes_and_writes_secret_free_audit(
    tmp_path: Path,
) -> None:
    binding = TextbookLLMPoolBinding.model_validate(
        {
            "schema_version": 1,
            "generation_id": "test-pool",
            "providers": [
                {
                    "provider_id": "fast",
                    "base_url": "https://provider.invalid/v1",
                    "model": "fast-model",
                    "credential_env": "FAST_KEY",
                    "max_async": 2,
                    "timeout_seconds": 10,
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
                {
                    "provider_id": "backup",
                    "base_url": "https://backup.invalid/v1",
                    "model": "backup-model",
                    "credential_env": "BACKUP_KEY",
                    "max_async": 1,
                    "timeout_seconds": 10,
                    "extra_body": {},
                },
            ],
        }
    )
    active = 0
    maximum_active = 0

    async def complete(model: str, prompt: str, **kwargs: Any) -> str:
        nonlocal active, maximum_active
        assert kwargs["api_key"] in {"top-secret-fast", "top-secret-backup"}
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return json.dumps({"model": model, "prompt": prompt})

    audit = tmp_path / "provider-calls.jsonl"
    pool = AsyncLLMProviderPool(
        binding,
        environment={
            "FAST_KEY": "top-secret-fast",
            "BACKUP_KEY": "top-secret-backup",
        },
        audit_path=audit,
        completion_func=complete,
    )

    results = await asyncio.gather(*(pool.complete(f"prompt-{index}") for index in range(3)))

    assert len(results) == 3
    assert maximum_active == 3
    audit_text = audit.read_text(encoding="utf-8")
    assert len(audit_text.splitlines()) == 3
    assert "top-secret" not in audit_text
    assert "prompt-0" not in audit_text


class _FakeDocStatus:
    def __init__(self, existing_id: str) -> None:
        self.records: dict[str, dict[str, Any]] = {existing_id: {"status": "processed"}}

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        return [self.records.get(item) for item in ids]

    async def get_status_counts(self) -> dict[str, int]:
        counts = {
            "pending": 0,
            "parsing": 0,
            "analyzing": 0,
            "processing": 0,
            "preprocessed": 0,
            "processed": 0,
            "failed": 0,
        }
        for record in self.records.values():
            counts[str(record["status"])] += 1
        return counts


class _FakeLightRAG:
    def __init__(self, existing_id: str) -> None:
        self.doc_status = _FakeDocStatus(existing_id)
        self.initialized = False
        self.finalized = False
        self.resume_called = False
        self.inserted: list[str] = []

    async def initialize_storages(self) -> None:
        self.initialized = True

    async def finalize_storages(self) -> None:
        self.finalized = True

    async def apipeline_process_enqueue_documents(self) -> None:
        self.resume_called = True

    async def ainsert(
        self,
        input: str | list[str],
        *,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        assert isinstance(input, list)
        assert isinstance(ids, list)
        assert isinstance(file_paths, list)
        assert track_id
        assert len(input) == len(ids) == len(file_paths)
        self.inserted.extend(ids)
        self.doc_status.records.update(
            {document_id: {"status": "processed"} for document_id in ids}
        )
        return track_id


@pytest.mark.asyncio
async def test_local_index_resumes_existing_ids_and_persists_progress(tmp_path: Path) -> None:
    source = tmp_path / "fragments.jsonl"
    fragments = _write_fragments(source, 3)
    fake = _FakeLightRAG(fragment_document_id(fragments[0]))
    settings = LocalTextbookLightRAGSettings(
        fragments_path=source,
        working_dir=tmp_path / "rag",
        embedding_binding_path=EMBEDDING,
        llm_pool_binding_path=LLM_POOL,
        batch_size=2,
    )

    summary = await index_local_textbook_fragments(
        settings,
        rag_factory=lambda _: fake,
    )

    assert fake.initialized
    assert fake.resume_called
    assert fake.finalized
    assert fake.inserted == [
        fragment_document_id(fragments[1]),
        fragment_document_id(fragments[2]),
    ]
    assert summary.status == "completed"
    assert summary.total_fragments_seen == 3
    assert summary.submitted_fragments == 2
    assert summary.existing_fragments == 1
    assert summary.processed_fragments == 3
    assert settings.state_path.is_file()


def test_settings_reject_unsafe_workspace(tmp_path: Path) -> None:
    source = tmp_path / "fragments.jsonl"
    _write_fragments(source, 1)

    with pytest.raises(ValueError, match="workspace"):
        LocalTextbookLightRAGSettings(
            fragments_path=source,
            working_dir=tmp_path / "rag",
            embedding_binding_path=EMBEDDING,
            llm_pool_binding_path=LLM_POOL,
            workspace="../unsafe",
        )
