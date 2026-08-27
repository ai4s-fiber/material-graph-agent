from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

import scripts.publish_material_graph_public_layout as publisher
from scripts.verify_material_graph_layout_assets import LayoutAssetVerificationError
from scripts.publish_material_graph_public_layout import (
    CANONICAL_LAYOUT_VERSION,
    COMMUNITY_EDGE_FIELDS,
    COMMUNITY_EDGE_STRIDE,
    LABEL_SCHEMA,
    NODE_FIELDS,
    NODE_STRIDE,
    PRIVATE_LAYOUT_SCHEMA,
    PUBLIC_ASSETS,
    PUBLIC_FILENAMES,
    PUBLIC_LAYOUT_SCHEMA,
    PublicLayoutPublishError,
    publish_public_layout,
    verify_public_layout,
)


@pytest.fixture(autouse=True)
def _allow_compact_projection_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep publisher tests compact; source verification has its own test suite."""
    monkeypatch.setattr(
        publisher,
        "verify_layout_assets",
        lambda _: {"schema": PRIVATE_LAYOUT_SCHEMA, "status": "verified"},
    )


def _descriptor(path: Path, records: int) -> dict[str, object]:
    return {
        "path": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _canonical_fixture(directory: Path) -> Path:
    directory.mkdir()
    nodes = directory / "nodes.f32"
    nodes.write_bytes(
        b"".join(
            struct.pack(
                "<8f",
                float(index),
                float(-index),
                1.0,
                0.1,
                0.2,
                0.3,
                float(index),
                0.0,
            )
            for index in range(2)
        )
    )
    community_edges = directory / "community-edges.f32"
    community_edges.write_bytes(struct.pack("<5f", 0.0, 1.0, 0.75, 1.0, -1.0))
    labels = directory / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "schema": LABEL_SCHEMA,
                "communities": [
                    {"macroId": 0, "label": "alpha", "x": 0.0, "y": 0.0, "nodeCount": 1},
                    {"macroId": 1, "label": "beta", "x": 1.0, "y": -1.0, "nodeCount": 1},
                ],
                "nodes": [{"index": 0, "label": "hub", "labelRank": 1.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # These represent canonical server-only assets.  They deliberately coexist
    # with the allowlisted source files and must never reach the public target.
    for filename in (
        "edges.u32",
        "layout-edges.u32",
        "layout-weights.f32",
        "node-index.sqlite3",
        "node-metrics.f32",
        "entities.jsonl",
        "relationships.jsonl",
    ):
        (directory / filename).write_bytes(b"private source data")

    manifest = {
        "schema": PRIVATE_LAYOUT_SCHEMA,
        "layout_version": CANONICAL_LAYOUT_VERSION,
        "generated_at_epoch_ms": 1722470400000,
        "assets": {
            "nodes": _descriptor(nodes, 2),
            "community_edges": _descriptor(community_edges, 1),
            "labels": _descriptor(labels, 3),
            "edges": _descriptor(directory / "edges.u32", 1),
            "node_index": _descriptor(directory / "node-index.sqlite3", 2),
        },
        "counts": {
            "nodes": 2,
            "semantic_edges": 1,
            "macro_communities": 2,
            "core_macro_communities": 2,
            "micro_communities": 2,
            "components": 1,
            "isolates": 0,
            "satellite_components": 0,
            "satellite_macro_communities": 0,
            "community_edges": 1,
            "labels": 3,
        },
        "parameters": {
            "layout_algorithm": "fixture-layout",
            "macro_resolution": 0.5,
            "micro_resolution": 1.0,
            "random_seed": 7,
        },
        "provenance": {
            "semantic_graph_preserved": True,
            "raw_corpus_text_included": False,
            "relationship_descriptions_included": False,
        },
        "lineage": {
            "parent_layout_version": "textbook-graph-layout-v1",
            "revision": "satellite-groups-v1",
        },
        "inputs": {"raw_corpus_path": "must-not-leak.jsonl"},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def test_publishes_only_the_verified_public_projection(tmp_path: Path) -> None:
    source = _canonical_fixture(tmp_path / "canonical")
    target = tmp_path / "constellation"

    result = publish_public_layout(source, target)

    assert result == {
        "schema": PUBLIC_LAYOUT_SCHEMA,
        "version": CANONICAL_LAYOUT_VERSION,
        "nodes": 2,
        "community_edges": 1,
        "labels": 3,
        "target": str(target.resolve()),
    }
    assert {path.name for path in target.iterdir()} == PUBLIC_FILENAMES
    assert not any(
        (target / filename).exists()
        for filename in (
            "edges.u32",
            "layout-edges.u32",
            "layout-weights.f32",
            "node-index.sqlite3",
            "node-metrics.f32",
            "entities.jsonl",
            "relationships.jsonl",
        )
    )
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == PUBLIC_LAYOUT_SCHEMA
    assert manifest["version"] == CANONICAL_LAYOUT_VERSION
    assert manifest["nodeCount"] == 2
    assert manifest["nodeStride"] == NODE_STRIDE
    assert manifest["macroCommunityCount"] == 2
    assert manifest["communityEdgeStride"] == COMMUNITY_EDGE_STRIDE
    assert manifest["assets"]["nodes"]["fields"] == NODE_FIELDS
    assert manifest["assets"]["community_edges"]["fields"] == COMMUNITY_EDGE_FIELDS
    assert set(manifest["assets"]) == set(PUBLIC_ASSETS)
    assert manifest["counts"] == {
        "nodes": 2,
        "semantic_edges": 1,
        "macro_communities": 2,
        "core_macro_communities": 2,
        "micro_communities": 2,
        "components": 1,
        "isolates": 0,
        "satellite_components": 0,
        "satellite_macro_communities": 0,
        "community_edges": 1,
        "labels": 3,
    }
    assert manifest["layout"] == {
        "algorithm": "fixture-layout",
        "macro_resolution": 0.5,
        "micro_resolution": 1.0,
        "random_seed": 7,
    }
    assert manifest["provenance"] == {
        "source_layout_version": CANONICAL_LAYOUT_VERSION,
        "lineage": {
            "parent_layout_version": "textbook-graph-layout-v1",
            "revision": "satellite-groups-v1",
        },
        "semantic_graph_preserved": True,
        "raw_corpus_text_included": False,
        "relationship_descriptions_included": False,
        "public_assets": ["nodes.f32", "community-edges.f32", "labels.json"],
    }
    assert "must-not-leak" not in (target / "manifest.json").read_text(encoding="utf-8")
    assert verify_public_layout(target) == result

    first_manifest = (target / "manifest.json").read_bytes()
    assert publish_public_layout(source, target) == result
    assert (target / "manifest.json").read_bytes() == first_manifest


def test_refuses_a_contaminated_target_without_overwriting_it(tmp_path: Path) -> None:
    source = _canonical_fixture(tmp_path / "canonical")
    target = tmp_path / "constellation"
    publish_public_layout(source, target)
    before = (target / "manifest.json").read_bytes()
    forbidden_file = target / "edges.u32"
    forbidden_file.write_bytes(b"private data must remain untouched")

    with pytest.raises(PublicLayoutPublishError, match="forbidden files"):
        publish_public_layout(source, target)

    assert forbidden_file.read_bytes() == b"private data must remain untouched"
    assert (target / "manifest.json").read_bytes() == before


def test_refuses_a_source_with_a_corrupt_allowed_asset_digest(tmp_path: Path) -> None:
    source = _canonical_fixture(tmp_path / "canonical")
    source_manifest = source / "manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["assets"]["nodes"]["sha256"] = "0" * 64
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    target = tmp_path / "constellation"

    with pytest.raises(PublicLayoutPublishError, match=r"assets\.nodes\.sha256"):
        publish_public_layout(source, target)

    assert not target.exists()


def test_wraps_a_failed_canonical_source_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _canonical_fixture(tmp_path / "canonical")
    target = tmp_path / "constellation"

    def reject_source(_: Path) -> None:
        raise LayoutAssetVerificationError("fixture source is invalid")

    monkeypatch.setattr(publisher, "verify_layout_assets", reject_source)
    with pytest.raises(PublicLayoutPublishError, match="canonical layout verification failed"):
        publish_public_layout(source, target)
