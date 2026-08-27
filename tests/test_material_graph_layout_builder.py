"""Focused tests for semantic/layout separation and compact asset generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import struct

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_material_graph_layout.py"
SPEC = importlib.util.spec_from_file_location("build_material_graph_layout", SCRIPT_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _config(path: Path) -> Path:
    payload = {
        "schema": "material-graph.layout-config.v1",
        "seed": 7,
        "satellite_grouping": {
            "algorithm": "dominant-entity-type-by-component-node-mass.v1",
            "group_count": 4,
            "top_type_group_count": 3,
            "residual_group_label": "外围分量 · 其他实体类型",
            "center_repulsion_iterations": 8,
            "center_minimum_separation": 0.19,
        },
        "macro_leiden": {
            "algorithm": "leiden/RBConfigurationVertexPartition",
            "resolution": 0.5,
            "main_community_target": 1,
            "accepted_macro_community_range": [1, 6],
            "minimum_main_community_size": 1,
        },
        "micro_leiden": {
            "algorithm": "leiden/RBConfigurationVertexPartition",
            "resolution": 1.0,
        },
        "layout_graph": {
            "self_loops": "drop",
            "duplicate_undirected_pairs": "coalesce_sum",
            "directed_semantic_graph": "preserved_read_only",
            "relation_base_weight": 1.0,
            "default_entity_type_pair_weight": 1.0,
            "entity_type_pair_weights": {"Material|Material": 0.8},
            "hub_damping": {
                "enabled": True,
                "degree_threshold": 2,
                "exponent": 0.5,
                "overlay_degree_threshold": 3,
            },
        },
        "multilevel_layout": {
            "backend": "igraph.drl",
            "macro_layout_options": "default",
            "community_layout_options": "default",
            "minimum_force_layout_nodes": 32,
            "main_center_extent": 0.52,
            "local_radius_min": 0.05,
            "local_radius_max": 0.27,
            "satellite_radius_min": 0.72,
            "satellite_radius_max": 1.14,
            "satellite_group_radius_min": 0.06,
            "satellite_group_radius_max": 0.155,
            "isolate_radius_min": 1.02,
            "isolate_radius_max": 1.24,
        },
        "visual": {
            "coordinate_extent": 1.25,
            "normal_node_size_min": 0.7,
            "normal_node_size_max": 8.5,
            "rank_lower_percentile": 5.0,
            "rank_upper_percentile": 99.5,
            "node_opacity": 0.8,
            "max_label_count": 256,
            "hub_label_count": 32,
            "community_label_count": 3,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.skipif(
    importlib.util.find_spec("igraph") is None or importlib.util.find_spec("leidenalg") is None,
    reason="offline graph layout dependencies are not installed",
)
def test_builder_preserves_semantic_rows_and_writes_separate_layout_assets(
    tmp_path: Path,
) -> None:
    entities = tmp_path / "entities.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    _write_jsonl(
        entities,
        [
            {"entity_id": "a", "name": "Alpha", "entity_type": "Material"},
            {"entity_id": "b", "name": "Beta", "entity_type": "Material"},
            {"entity_id": "c", "name": "Gamma", "entity_type": "Property"},
            {"entity_id": "d", "name": "Delta", "entity_type": "Process"},
            {"entity_id": "e", "name": "Epsilon", "entity_type": "Process"},
            {"entity_id": "f", "name": "Zeta", "entity_type": "Material"},
        ],
    )
    _write_jsonl(
        relationships,
        [
            {"source_entity_id": "a", "target_entity_id": "b", "mentions": 1},
            {"source_entity_id": "b", "target_entity_id": "a", "mentions": 2},
            {"source_entity_id": "b", "target_entity_id": "c", "mentions": 1},
            {"source_entity_id": "d", "target_entity_id": "e", "mentions": 1},
        ],
    )
    source_before = relationships.read_bytes()
    output = tmp_path / "layout"
    manifest = builder.build_layout_assets(
        entities, relationships, output, _config(tmp_path / "config.json")
    )

    assert relationships.read_bytes() == source_before
    assert manifest["semantic_graph"]["directed_edges"] == 4
    assert manifest["layout_graph"]["undirected_weighted_edges"] == 3
    assert manifest["counts"]["nodes"] == 6
    assert manifest["counts"]["semantic_edges"] == 4
    assert manifest["counts"]["layout_edges"] == 3
    assert (output / "nodes.f32").stat().st_size == 6 * 8 * 4
    assert (output / "edges.u32").stat().st_size == 4 * 2 * 4
    assert (output / "layout-edges.u32").stat().st_size == 3 * 2 * 4
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    assert labels["schema"] == "material-graph.layout-labels.v1"
    assert set(labels) == {"schema", "communities", "nodes"}
    with sqlite3.connect(output / "node-index.sqlite3") as connection:
        assert connection.execute(
            "SELECT node_index FROM node_index WHERE entity_id = 'a'"
        ).fetchone() == (0,)

    with pytest.raises(builder.LayoutBuildError, match="will not be overwritten"):
        builder.build_layout_assets(
            entities, relationships, output, _config(tmp_path / "config-two.json")
        )


@pytest.mark.skipif(
    importlib.util.find_spec("igraph") is None or importlib.util.find_spec("leidenalg") is None,
    reason="offline graph layout dependencies are not installed",
)
def test_builder_persists_distinct_hierarchical_micro_communities_inside_a_macro(
    tmp_path: Path,
) -> None:
    entities = tmp_path / "entities.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    labels = tuple("abcdefgh")
    _write_jsonl(
        entities,
        [
            {"entity_id": entity_id, "name": entity_id.upper(), "entity_type": "Material"}
            for entity_id in labels
        ],
    )
    _write_jsonl(
        relationships,
        [
            {"source_entity_id": source, "target_entity_id": target, "mentions": 1}
            for source, target in (
                ("a", "b"),
                ("a", "c"),
                ("a", "d"),
                ("b", "c"),
                ("b", "d"),
                ("c", "d"),
                ("e", "f"),
                ("e", "g"),
                ("e", "h"),
                ("f", "g"),
                ("f", "h"),
                ("g", "h"),
                ("d", "e"),
            )
        ],
    )
    output = tmp_path / "hierarchical-layout"
    manifest = builder.build_layout_assets(
        entities, relationships, output, _config(tmp_path / "hierarchical-config.json")
    )

    node_records = list(struct.iter_unpack("<8f", (output / "nodes.f32").read_bytes()))
    metric_records = list(struct.iter_unpack("<6f", (output / "node-metrics.f32").read_bytes()))
    core_indices = [index for index, node in enumerate(node_records) if int(node[6]) == 0]
    core_micro_ids = {int(metric_records[index][4]) for index in core_indices}
    persisted_micro_ids = {int(metric[4]) for metric in metric_records}

    # The one configured core macro contains the full connected component, while
    # its two dense lobes stay distinct at the independently computed micro level.
    assert core_indices == list(range(8))
    assert len(core_micro_ids) >= 2
    assert manifest["counts"]["micro_communities"] == len(persisted_micro_ids)
    by_macro = manifest["community_layout"]["micro_communities_by_macro"]
    assert by_macro[0]["microCommunities"] == len(core_micro_ids)
    assert sum(item["microCommunities"] for item in by_macro) == len(persisted_micro_ids)


@pytest.mark.skipif(
    importlib.util.find_spec("igraph") is None or importlib.util.find_spec("leidenalg") is None,
    reason="offline graph layout dependencies are not installed",
)
def test_builder_groups_disconnected_components_by_dominant_type_with_seeded_irregular_anchors(
    tmp_path: Path,
) -> None:
    entities = tmp_path / "entities.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    core_ids = tuple(f"core-{index}" for index in range(8))
    satellite_components = (
        ("Material", "material"),
        ("Property", "property"),
        ("Process", "process"),
        ("Structure", "structure"),
        ("Application", "application"),
    )
    entity_rows: list[dict[str, object]] = [
        {"entity_id": entity_id, "name": entity_id, "entity_type": "Material"}
        for entity_id in core_ids
    ]
    for entity_type, prefix in satellite_components:
        entity_rows.extend(
            (
                {
                    "entity_id": f"{prefix}-a",
                    "name": f"{entity_type} A",
                    "entity_type": entity_type,
                },
                {
                    "entity_id": f"{prefix}-b",
                    "name": f"{entity_type} B",
                    "entity_type": entity_type,
                },
            )
        )
    entity_rows.append({"entity_id": "isolate", "name": "Isolate", "entity_type": "Other"})
    _write_jsonl(entities, entity_rows)
    relationship_rows: list[dict[str, object]] = [
        {
            "source_entity_id": core_ids[index],
            "target_entity_id": core_ids[index + 1],
            "mentions": 1,
        }
        for index in range(len(core_ids) - 1)
    ]
    relationship_rows.extend(
        {
            "source_entity_id": f"{prefix}-a",
            "target_entity_id": f"{prefix}-b",
            "mentions": 1,
        }
        for _, prefix in satellite_components
    )
    _write_jsonl(relationships, relationship_rows)
    config = _config(tmp_path / "satellite-config.json")
    output = tmp_path / "satellite-layout"
    repeated_output = tmp_path / "satellite-layout-repeat"
    manifest = builder.build_layout_assets(entities, relationships, output, config)
    repeated_manifest = builder.build_layout_assets(
        entities, relationships, repeated_output, config
    )

    grouping = manifest["community_layout"]["satellite_grouping"]
    groups = grouping["groups"]
    assert grouping["configuredGroupCount"] == 4
    assert grouping["activeGroupCount"] == 4
    assert len(groups) == 4
    assert sum(group["componentCount"] for group in groups) == len(satellite_components)
    assert sum(group["nodeCount"] for group in groups) == len(satellite_components) * 2
    assert any(group["kind"] == "residual" for group in groups)
    assert manifest["counts"]["macro_communities"] == 6
    assert manifest["counts"]["satellite_macro_communities"] == 4
    assert manifest["counts"]["satellite_components"] == len(satellite_components)

    anchors = [tuple(group["placementCenter"]) for group in groups]
    anchor_radii = {round((x * x + y * y) ** 0.5, 6) for x, y in anchors}
    assert len(anchor_radii) > 1
    for left, (x1, y1) in enumerate(anchors):
        for x2, y2 in anchors[left + 1 :]:
            assert ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 >= 0.19 - 1e-6

    label_rows = json.loads((output / "labels.json").read_text(encoding="utf-8"))["communities"]
    labels_by_macro = {row["macroId"]: row["label"] for row in label_rows}
    assert {group["label"] for group in groups} == {
        labels_by_macro[group["macroId"]] for group in groups
    }
    assert (output / "nodes.f32").read_bytes() == (repeated_output / "nodes.f32").read_bytes()
    assert manifest["semantic_graph"] == repeated_manifest["semantic_graph"]


@pytest.mark.skipif(
    importlib.util.find_spec("igraph") is None or importlib.util.find_spec("leidenalg") is None,
    reason="offline graph layout dependencies are not installed",
)
def test_builder_resets_igraph_rng_for_seeded_force_layout(tmp_path: Path) -> None:
    entities = tmp_path / "entities.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    entity_ids = tuple(f"node-{index:02d}" for index in range(40))
    _write_jsonl(
        entities,
        [
            {"entity_id": entity_id, "name": entity_id, "entity_type": "Material"}
            for entity_id in entity_ids
        ],
    )
    _write_jsonl(
        relationships,
        [
            {
                "source_entity_id": entity_ids[index],
                "target_entity_id": entity_ids[(index + 1) % len(entity_ids)],
                "mentions": 1,
            }
            for index in range(len(entity_ids))
        ],
    )
    config = _config(tmp_path / "drl-config.json")
    first_output = tmp_path / "drl-layout-one"
    second_output = tmp_path / "drl-layout-two"
    builder.build_layout_assets(entities, relationships, first_output, config)
    builder.build_layout_assets(entities, relationships, second_output, config)

    assert (first_output / "nodes.f32").read_bytes() == (second_output / "nodes.f32").read_bytes()
    assert (first_output / "node-metrics.f32").read_bytes() == (
        second_output / "node-metrics.f32"
    ).read_bytes()
