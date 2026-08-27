from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from material_graph.knowledge.models import EvidenceFragment, SourceLocator
from material_graph.knowledge.textbook_lightrag import (
    LLMProviderResult,
    TextbookLightRAGDocument,
    fragment_document_id,
    fragment_file_path,
)
from material_graph.knowledge.textbook_raw_graph import (
    RAW_EXTRACTION_PROMPT_VERSION,
    RawGraphExtractionSettings,
    extract_raw_textbook_graph,
    parse_raw_graph_response,
)


ROOT = Path(__file__).resolve().parents[1]
LLM_POOL = ROOT / "config/knowledge/textbook-llm-pool.v1.json"


def _fragment(index: int) -> EvidenceFragment:
    return EvidenceFragment(
        fragment_id=UUID(int=index + 1),
        source_id=UUID(int=100 + index),
        text="PET 熔体经喷丝板挤出后进行牵伸，牵伸倍数提高取向度。",
        locator=SourceLocator(
            root_id="cyj_source_hu",
            relative_path=f"source_hu/教材/part-{index}.md",
            page=index + 1,
            section="熔体纺丝",
            block_index=index,
        ),
        retention_reason="textbook_full_corpus",
        parser_name="source_hu_markdown",
        parser_version="local-textbook-v1",
        embedding_generation_id="embedding-test",
        metadata={
            "chunk_index": index,
            "logical_title": "化学纤维工艺学",
            "page_end": index + 1,
            "part_number": 1,
            "source_family": "source_hu_markdown",
        },
    )


def _document(index: int) -> TextbookLightRAGDocument:
    fragment = _fragment(index)
    return TextbookLightRAGDocument(
        document_id=fragment_document_id(fragment),
        text=fragment.text,
        file_path=fragment_file_path(fragment),
        fragment=fragment,
    )


def _response() -> LLMProviderResult:
    return LLMProviderResult(
        text=json.dumps(
            {
                "entities": [
                    {"name": "PET", "type": "材料", "description": "聚酯熔体材料"},
                    {"name": "牵伸", "type": "Process", "description": "纺丝后处理工艺"},
                    {"name": "取向度", "type": "性能", "description": "分子链取向指标"},
                    {"name": "PET", "type": "Other", "description": "重复短描述"},
                ],
                "relationships": [
                    {
                        "source": "牵伸",
                        "target": "取向度",
                        "description": "牵伸倍数提高会提高取向度",
                    },
                    {
                        "source": "不存在实体",
                        "target": "PET",
                        "description": "应被丢弃",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        provider_id="deepseek_v4_flash",
        model="deepseek-v4-flash",
        elapsed_seconds=1.25,
    )


def test_raw_response_normalizes_types_deduplicates_and_drops_dangling_edges() -> None:
    extraction = parse_raw_graph_response(_document(0), _response())

    assert extraction.prompt_version == RAW_EXTRACTION_PROMPT_VERSION
    assert [entity.name for entity in extraction.entities] == ["PET", "牵伸", "取向度"]
    assert [entity.entity_type for entity in extraction.entities] == [
        "Material",
        "Process",
        "Property",
    ]
    assert len(extraction.relationships) == 1
    assert extraction.relationships[0].source == "牵伸"
    assert extraction.dropped_relationships == 1
    assert "E:" not in extraction.to_json()


def test_raw_response_extracts_complete_graph_json_from_provider_wrapping() -> None:
    wrapped = _response()
    wrapped = LLMProviderResult(
        text=f"以下是结构化结果：\n{wrapped.text}\n以上内容仅为 JSON 图谱。",
        provider_id=wrapped.provider_id,
        model=wrapped.model,
        elapsed_seconds=wrapped.elapsed_seconds,
    )

    extraction = parse_raw_graph_response(_document(0), wrapped)

    assert [entity.name for entity in extraction.entities] == ["PET", "牵伸", "取向度"]
    assert len(extraction.relationships) == 1


class _FakePool:
    async def complete_with_provenance(self, *args, **kwargs) -> LLMProviderResult:
        return _response()


@pytest.mark.asyncio
async def test_raw_extraction_is_resumable_and_writes_progress(tmp_path: Path) -> None:
    fragments = tmp_path / "fragments.jsonl"
    fragments.write_text(
        "\n".join(_fragment(index).model_dump_json() for index in range(3)) + "\n",
        encoding="utf-8",
    )
    settings = RawGraphExtractionSettings(
        fragments_path=fragments,
        output_path=tmp_path / "out/extractions.jsonl",
        state_path=tmp_path / "out/state.json",
        failure_path=tmp_path / "out/failures.jsonl",
        provider_audit_path=tmp_path / "out/provider.jsonl",
        llm_pool_binding_path=LLM_POOL,
        sync_every=1,
    )

    first = await extract_raw_textbook_graph(settings, pool=_FakePool())
    second = await extract_raw_textbook_graph(settings, pool=_FakePool())

    assert first.status == "completed"
    assert first.newly_extracted_fragments == 3
    assert second.status == "completed"
    assert second.existing_fragments == 3
    assert second.newly_extracted_fragments == 0
    assert len(settings.output_path.read_text(encoding="utf-8").splitlines()) == 3
    state = json.loads(settings.state_path.read_text(encoding="utf-8"))
    assert state["prompt_version"] == RAW_EXTRACTION_PROMPT_VERSION
    assert state["summary"]["completed_fragments"] == 3
    assert settings.failure_path.read_text(encoding="utf-8") == ""
