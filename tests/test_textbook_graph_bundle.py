from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from material_graph.knowledge.models import EvidenceFragment, SourceLocator
from material_graph.knowledge.textbook_graph_bundle import (
    TextbookGraphBundleError,
    TextbookGraphBundleSettings,
    build_textbook_graph_bundle,
)


def _fragment(index: int) -> EvidenceFragment:
    return EvidenceFragment(
        fragment_id=UUID(int=index + 1),
        source_id=UUID(int=100 + index),
        text=f"PET 牵伸片段 {index}",
        locator=SourceLocator(
            root_id="cyj_source_hu",
            relative_path=f"source_hu/book/part-{index}.md",
            page=index + 1,
            section="纺丝",
            block_index=index,
        ),
        retention_reason="textbook_full_corpus",
        parser_name="source_hu_markdown",
        parser_version="local-textbook-v1",
        embedding_generation_id="embedding-test",
        metadata={"chunk_index": index, "logical_title": "化纤工艺"},
    )


def _write_inputs(tmp_path: Path, *, extraction_count: int = 2) -> tuple[Path, Path]:
    fragments = tmp_path / "fragments.jsonl"
    fragments.write_text(
        "\n".join(_fragment(i).model_dump_json() for i in range(2)) + "\n",
        encoding="utf-8",
    )
    extractions = tmp_path / "extractions.jsonl"
    rows = []
    for index in range(extraction_count):
        fragment = _fragment(index)
        rows.append(
            {
                "fragment_id": str(fragment.fragment_id),
                "citation_path": f"textbook_fragments/book/{index}.md",
                "prompt_version": "test-v1",
                "entities": [
                    {
                        "name": "PET" if index == 0 else "pet",
                        "entity_type": "Material",
                        "description": "聚酯材料",
                    },
                    {
                        "name": "牵伸",
                        "entity_type": "Process",
                        "description": "提高分子取向",
                    },
                    {
                        "name": "原文不存在的实体",
                        "entity_type": "Other",
                        "description": "应在打包时移除",
                    },
                ],
                "relationships": [
                    {
                        "source": "PET" if index == 0 else "pet",
                        "target": "牵伸",
                        "description": "PET 经牵伸提高取向",
                    },
                    {
                        "source": "PET" if index == 0 else "pet",
                        "target": "原文不存在的实体",
                        "description": "悬空关系也应移除",
                    },
                ],
            }
        )
    extractions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return fragments, extractions


def test_bundle_canonicalizes_mentions_and_emits_custom_kg(tmp_path: Path) -> None:
    fragments, extractions = _write_inputs(tmp_path)
    settings = TextbookGraphBundleSettings(
        fragments_path=fragments,
        extractions_path=extractions,
        output_dir=tmp_path / "bundle",
    )

    summary = build_textbook_graph_bundle(settings)

    assert summary.fragment_count == 2
    assert summary.entity_count == 2
    assert summary.relationship_count == 1
    assert summary.filtered_entity_mentions == 2
    assert summary.filtered_relationship_mentions == 2
    entities = [
        json.loads(line) for line in settings.entities_path.read_text(encoding="utf-8").splitlines()
    ]
    pet = next(item for item in entities if item["canonical_key"] == "pet")
    assert pet["mentions"] == 2
    assert pet["name"] == "PET"
    custom = json.loads(settings.custom_kg_path.read_text(encoding="utf-8"))
    assert len(custom["chunks"]) == 2
    assert len(custom["entities"]) == 2
    assert custom["chunks"][0]["file_path"].startswith("mg_")
    assert custom["chunks"][0]["file_path"].endswith(".txt")
    assert custom["entities"][0]["file_path"].startswith("mg_")
    assert custom["relationships"][0]["keywords"].startswith("材料科学")
    assert settings.manifest_path.is_file()


def test_bundle_fails_closed_when_extraction_is_incomplete(tmp_path: Path) -> None:
    fragments, extractions = _write_inputs(tmp_path, extraction_count=1)
    settings = TextbookGraphBundleSettings(
        fragments_path=fragments,
        extractions_path=extractions,
        output_dir=tmp_path / "bundle",
    )

    with pytest.raises(TextbookGraphBundleError, match="incomplete"):
        build_textbook_graph_bundle(settings)
