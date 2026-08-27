from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from scripts.verify_material_graph_layout_assets import (
    ASSET_SCHEMA,
    LABEL_SCHEMA,
    LayoutAssetVerificationError,
    verify_layout_assets,
)


def _descriptor(path: Path, records: int) -> dict[str, object]:
    return {
        "path": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(directory: Path, layout: bool = True) -> Path:
    directory.mkdir()
    nodes = directory / "nodes.f32"
    nodes.write_bytes(
        b"".join(
            struct.pack(
                "<8f", -1 + index / 10, index % 3 / 10, 0.8, 0.1, 0.2, 0.3, float(index), 0.0
            )
            for index in range(12)
        )
    )
    edges = directory / "edges.u32"
    edges.write_bytes(b"".join(struct.pack("<2I", index, (index + 1) % 12) for index in range(12)))
    community = directory / "community-edges.f32"
    community.write_bytes(
        b"".join(
            struct.pack(
                "<5f",
                float(index),
                float(index + 1),
                1.0,
                -1 + (index + 1) / 10,
                (index + 1) % 3 / 10,
            )
            for index in range(11)
        )
    )
    labels = directory / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "schema": LABEL_SCHEMA,
                "communities": [
                    {
                        "macroId": index,
                        "label": f"community-{index}",
                        "x": -1 + index / 10,
                        "y": index % 3 / 10,
                        "nodeCount": 1,
                    }
                    for index in range(12)
                ],
                "nodes": [{"index": 0, "label": "hub", "labelRank": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    assets: dict[str, dict[str, object]] = {
        "nodes": _descriptor(nodes, 12),
        "edges": _descriptor(edges, 12),
        "community_edges": _descriptor(community, 11),
        "labels": _descriptor(labels, 13),
    }
    counts: dict[str, int] = {
        "nodes": 12,
        "semantic_edges": 12,
        "community_edges": 11,
        "macro_communities": 12,
        "labels": 13,
    }
    if layout:
        layout_edges = directory / "layout-edges.u32"
        layout_edges.write_bytes(
            b"".join(struct.pack("<2I", index, index + 1) for index in range(11))
        )
        weights = directory / "layout-weights.f32"
        weights.write_bytes(b"".join(struct.pack("<f", 1.0) for _ in range(11)))
        assets.update(
            layout_edges=_descriptor(layout_edges, 11), layout_weights=_descriptor(weights, 11)
        )
        counts["layout_edges"] = 11
    manifest = {
        "schema": ASSET_SCHEMA,
        "generated_at_epoch_ms": 1,
        "max_label_count": 256,
        "inputs": {
            "entities": {"sha256": "a" * 64, "records": 12},
            "relationships": {"sha256": "b" * 64, "records": 12},
        },
        "parameters": {
            "layout_algorithm": "fixture",
            "random_seed": 7,
            "macro_resolution": 0.5,
            "micro_resolution": 1.0,
            "hub_damping": {"threshold": 2},
        },
        "counts": counts,
        "assets": assets,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_verifies_complete_compact_asset_fixture(tmp_path: Path) -> None:
    result = verify_layout_assets(_fixture(tmp_path / "assets"))
    assert result == {
        "schema": ASSET_SCHEMA,
        "status": "verified",
        "nodes": 12,
        "semantic_edges": 12,
        "layout_edges": 11,
        "community_edges": 11,
        "macro_communities": 12,
        "labels": 13,
    }


def test_layout_projection_is_optional_as_a_pair(tmp_path: Path) -> None:
    assert verify_layout_assets(_fixture(tmp_path / "assets", layout=False))["layout_edges"] == 0


def test_rejects_digest_or_byte_mismatch(tmp_path: Path) -> None:
    assets = _fixture(tmp_path / "assets")
    with (assets / "nodes.f32").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(LayoutAssetVerificationError, match="bytes does not match"):
        verify_layout_assets(assets)


def test_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    assets = _fixture(tmp_path / "assets")
    nodes = assets / "nodes.f32"
    raw = bytearray(nodes.read_bytes())
    raw[:4] = struct.pack("<f", float("nan"))
    nodes.write_bytes(raw)
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["nodes"] = _descriptor(nodes, 12)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LayoutAssetVerificationError, match="coordinate"):
        verify_layout_assets(assets)


def test_rejects_macro_community_count_outside_contract(tmp_path: Path) -> None:
    assets = _fixture(tmp_path / "assets")
    nodes = assets / "nodes.f32"
    raw = bytearray(nodes.read_bytes())
    raw[11 * 32 + 24 : 11 * 32 + 28] = struct.pack("<f", 10.0)
    nodes.write_bytes(raw)
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["nodes"] = _descriptor(nodes, 12)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LayoutAssetVerificationError, match="12 to 30"):
        verify_layout_assets(assets)
