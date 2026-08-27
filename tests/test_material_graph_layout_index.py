"""Core tests for semantic entity IDs and derived WebGL layout indexes."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import struct

import pytest

from material_graph.knowledge.material_graph_layout_index import (
    LayoutIndexUnavailableError,
    get_layout_neighborhood,
    resolve_layout_nodes,
    search_layout_nodes,
)


def _layout_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "layout"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "node-index.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE node_index (entity_id TEXT PRIMARY KEY, "
            "node_index INTEGER NOT NULL, name TEXT NOT NULL, "
            "entity_type TEXT NOT NULL, macro_id INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO node_index(entity_id, node_index, name, entity_type, macro_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("entity:pi", 0, "polyimide", "Material", 2),
                ("entity:pet", 1, "PET", "Material", 4),
                ("entity:tg", 2, "glass transition", "Property", 2),
                ("entity:cure", 3, "imide curing", "Process", 2),
                ("entity:window", 4, "thermal window", "ProcessCondition", 2),
                ("entity:duplicate-a", 5, "shared-label", "Material", 6),
                ("entity:duplicate-b", 6, "shared-label", "Property", 8),
            ],
        )
    with (directory / "edges.u32").open("wb") as handle:
        for source, target in ((0, 2), (2, 3), (3, 4), (1, 0)):
            handle.write(struct.pack("<2I", source, target))
    return directory


def test_resolve_layout_nodes_returns_only_derived_ids(tmp_path: Path) -> None:
    result = resolve_layout_nodes(
        entity_ids=["entity:pi", "missing", "entity:pi"],
        labels=["PET", "shared-label"],
        asset_dir=_layout_dir(tmp_path),
    )

    assert result["matches"] == [
        {"entityId": "entity:pi", "layoutIndex": 0, "macroId": 2},
        {"entityId": "entity:pet", "layoutIndex": 1, "macroId": 4},
    ]
    assert result["unresolvedEntityIds"] == ["missing"]
    assert result["ambiguousLabels"] == {"shared-label": 2}


def test_search_layout_nodes_is_filtered_and_bounded(tmp_path: Path) -> None:
    directory = _layout_dir(tmp_path)
    result = search_layout_nodes(
        query="imide",
        match_mode="substring",
        entity_type="Material",
        macro_id=2,
        limit=5,
        asset_dir=directory,
    )

    assert result == {
        "results": [
            {
                "entityId": "entity:pi",
                "layoutIndex": 0,
                "name": "polyimide",
                "entityType": "Material",
                "macroId": 2,
            }
        ],
        "truncated": False,
    }
    with pytest.raises(ValueError, match="at least one"):
        search_layout_nodes(asset_dir=directory)
    with pytest.raises(ValueError, match="limit"):
        search_layout_nodes(query="polyimide", limit=51, asset_dir=directory)


def test_neighborhood_returns_capped_semantic_endpoint_pairs(tmp_path: Path) -> None:
    neighborhood = get_layout_neighborhood(
        entity_id="entity:pi",
        hops=2,
        max_nodes=4,
        max_edges=8,
        asset_dir=_layout_dir(tmp_path),
    )

    assert neighborhood["center"]["entityId"] == "entity:pi"
    assert [node["layoutIndex"] for node in neighborhood["nodes"]] == [0, 1, 2, 3]
    assert {
        (edge["sourceLayoutIndex"], edge["targetLayoutIndex"])
        for edge in neighborhood["edges"]
    } == {(0, 2), (1, 0), (2, 3)}
    with pytest.raises(ValueError, match="exactly one"):
        get_layout_neighborhood(
            entity_id="entity:pi",
            layout_index=0,
            asset_dir=_layout_dir(tmp_path / "second"),
        )


def test_layout_index_fails_closed_when_assets_are_missing(tmp_path: Path) -> None:
    with pytest.raises(LayoutIndexUnavailableError, match="unavailable"):
        resolve_layout_nodes(entity_ids=["entity:pi"], asset_dir=tmp_path)
    with pytest.raises(ValueError, match="invalid exact-match"):
        resolve_layout_nodes(entity_ids=[" entity:pi"], asset_dir=_layout_dir(tmp_path / "valid"))
