#!/usr/bin/env python3
"""Fail-closed validation for compact, derived Material Graph layout assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ASSET_SCHEMA = "material-graph.layout-assets.v1"
LABEL_SCHEMA = "material-graph.layout-labels.v1"
NODE_RECORD = struct.Struct("<8f")
EDGE_RECORD = struct.Struct("<2I")
WEIGHT_RECORD = struct.Struct("<f")
COMMUNITY_EDGE_RECORD = struct.Struct("<5f")
MAX_ABS_COORDINATE = 1.25
MIN_VISUAL_SIZE = 0.1
MAX_VISUAL_SIZE = 12.0
MAX_LABELS = 256


class LayoutAssetVerificationError(ValueError):
    """A derived layout package cannot be safely published or rendered."""


def _fail(message: str) -> None:
    raise LayoutAssetVerificationError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or (positive and value == 0)
    ):
        _fail(f"{name} must be a {'positive' if positive else 'non-negative'} integer")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        _fail(f"{name} must be {'positive and ' if positive else ''}finite")
    return parsed


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        _fail(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LayoutAssetVerificationError(f"{name} must be a SHA-256 digest") from exc
    return value.lower()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as exc:
        raise LayoutAssetVerificationError(f"{name} is not valid UTF-8 JSON") from exc


def _asset(
    directory: Path,
    assets: Mapping[str, Any],
    name: str,
    filename: str,
    record_bytes: int | None,
) -> tuple[Path, int]:
    entry = _mapping(assets.get(name), f"assets.{name}")
    if entry.get("path") != filename:
        _fail(f"assets.{name}.path must be {filename!r}")
    candidate_value = entry.get("path")
    if not isinstance(candidate_value, str) or Path(candidate_value).is_absolute():
        _fail(f"assets.{name}.path must be a relative path")
    path = (directory / candidate_value).resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise LayoutAssetVerificationError(f"assets.{name}.path escapes asset directory") from exc
    if not path.is_file():
        _fail(f"assets.{name}.path is missing")
    records = _integer(entry.get("records"), f"assets.{name}.records")
    declared_bytes = _integer(entry.get("bytes"), f"assets.{name}.bytes")
    actual_bytes = path.stat().st_size
    if declared_bytes != actual_bytes:
        _fail(f"assets.{name}.bytes does not match file length")
    if record_bytes is not None and actual_bytes != records * record_bytes:
        _fail(f"assets.{name} record count does not match binary byte length")
    if _sha(entry.get("sha256"), f"assets.{name}.sha256") != _digest(path):
        _fail(f"assets.{name}.sha256 does not match file contents")
    return path, records


def _float_id(value: float, name: str) -> int:
    if not math.isfinite(value) or value < 0 or value != math.floor(value):
        _fail(f"{name} must be a non-negative integer float32")
    return int(value)


def _check_parameters(manifest: Mapping[str, Any]) -> None:
    inputs = _mapping(manifest.get("inputs"), "inputs")
    for source in ("entities", "relationships"):
        row = _mapping(inputs.get(source), f"inputs.{source}")
        _sha(row.get("sha256"), f"inputs.{source}.sha256")
        _integer(row.get("records"), f"inputs.{source}.records", positive=True)
    parameters = _mapping(manifest.get("parameters"), "parameters")
    for name in (
        "layout_algorithm",
        "random_seed",
        "macro_resolution",
        "micro_resolution",
        "hub_damping",
    ):
        if name not in parameters:
            _fail(f"parameters.{name} is required")
    if (
        not isinstance(parameters["layout_algorithm"], str)
        or not parameters["layout_algorithm"].strip()
    ):
        _fail("parameters.layout_algorithm must be a non-empty string")
    _integer(parameters["random_seed"], "parameters.random_seed")
    _number(parameters["macro_resolution"], "parameters.macro_resolution", positive=True)
    _number(parameters["micro_resolution"], "parameters.micro_resolution", positive=True)
    _mapping(parameters["hub_damping"], "parameters.hub_damping")


def _check_nodes(path: Path) -> set[int]:
    macro_ids: set[int] = set()
    with path.open("rb") as handle:
        for index, raw in enumerate(iter(lambda: handle.read(NODE_RECORD.size), b"")):
            x, y, size, red, green, blue, macro, flags = NODE_RECORD.unpack(raw)
            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or max(abs(x), abs(y)) > MAX_ABS_COORDINATE
            ):
                _fail(f"nodes.f32 record {index} coordinate is outside normalized bounds")
            if not math.isfinite(size) or not MIN_VISUAL_SIZE <= size <= MAX_VISUAL_SIZE:
                _fail(f"nodes.f32 record {index} size is outside visual bounds")
            if any(not math.isfinite(c) or not 0.0 <= c <= 1.0 for c in (red, green, blue)):
                _fail(f"nodes.f32 record {index} RGB is outside [0, 1]")
            macro_ids.add(_float_id(macro, f"nodes.f32 record {index} macroId"))
            _float_id(flags, f"nodes.f32 record {index} flags")
    return macro_ids


def _check_edges(path: Path, node_count: int, *, layout: bool) -> None:
    seen: set[tuple[int, int]] = set()
    with path.open("rb") as handle:
        for index, raw in enumerate(iter(lambda: handle.read(EDGE_RECORD.size), b"")):
            source, target = EDGE_RECORD.unpack(raw)
            if source >= node_count or target >= node_count:
                _fail(f"{path.name} record {index} endpoint is outside nodes.f32")
            if layout:
                if source >= target:
                    _fail(f"layout-edges.u32 record {index} must use source < target")
                if (source, target) in seen:
                    _fail(f"layout-edges.u32 duplicate pair {(source, target)}")
                seen.add((source, target))


def _check_weights(path: Path, expected: int) -> None:
    with path.open("rb") as handle:
        for index, raw in enumerate(iter(lambda: handle.read(WEIGHT_RECORD.size), b"")):
            (weight,) = WEIGHT_RECORD.unpack(raw)
            if not math.isfinite(weight) or weight <= 0:
                _fail(f"layout-weights.f32 record {index} must be finite and positive")
    if path.stat().st_size // WEIGHT_RECORD.size != expected:
        _fail("layout-weights.f32 record count must equal layout-edges.u32")


def _check_labels(
    path: Path, records: int, macros: set[int], maximum: int
) -> dict[int, tuple[float, float]]:
    payload = _json(path, "labels.json")
    if payload.get("schema") != LABEL_SCHEMA:
        _fail(f"labels.json schema must be {LABEL_SCHEMA!r}")
    if set(payload).difference({"schema", "communities", "nodes"}):
        _fail("labels.json contains unsupported fields")
    communities, nodes = payload.get("communities"), payload.get("nodes")
    if not isinstance(communities, list) or not isinstance(nodes, list):
        _fail("labels.json communities and nodes must be arrays")
    if len(communities) + len(nodes) != records or records > maximum:
        _fail("labels.json record count exceeds cap or differs from manifest")
    centers: dict[int, tuple[float, float]] = {}
    for row in communities:
        row = _mapping(row, "labels.json community")
        if set(row).difference({"macroId", "label", "x", "y", "nodeCount"}):
            _fail("community label contains unsupported raw-content field")
        macro = _integer(row.get("macroId"), "community macroId")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 160:
            _fail("community label must be 1..160 characters")
        x, y = _number(row.get("x"), "community x"), _number(row.get("y"), "community y")
        if max(abs(x), abs(y)) > MAX_ABS_COORDINATE:
            _fail("community center is outside normalized bounds")
        _integer(row.get("nodeCount"), "community nodeCount", positive=True)
        if macro in centers:
            _fail("labels.json has duplicate macro community")
        centers[macro] = (x, y)
    if set(centers) != macros:
        _fail("labels.json must provide exactly one center per macro community")
    node_indices: set[int] = set()
    for row in nodes:
        row = _mapping(row, "labels.json node")
        if set(row).difference({"index", "label", "labelRank"}):
            _fail("node label contains unsupported raw-content field")
        index = _integer(row.get("index"), "node label index")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 160:
            _fail("node label must be 1..160 characters")
        _number(row.get("labelRank"), "node label rank")
        if index in node_indices:
            _fail("labels.json has duplicate node index")
        node_indices.add(index)
    return centers


def _check_community_edges(
    path: Path, macros: set[int], centers: Mapping[int, tuple[float, float]]
) -> None:
    seen: set[tuple[int, int]] = set()
    with path.open("rb") as handle:
        for index, raw in enumerate(iter(lambda: handle.read(COMMUNITY_EDGE_RECORD.size), b"")):
            source, target, weight, x1, y1 = COMMUNITY_EDGE_RECORD.unpack(raw)
            source_id = _float_id(source, f"community-edges.f32 record {index} sourceMacro")
            target_id = _float_id(target, f"community-edges.f32 record {index} targetMacro")
            if source_id not in macros or target_id not in macros or source_id >= target_id:
                _fail(f"community-edges.f32 record {index} has invalid canonical macro pair")
            if (source_id, target_id) in seen:
                _fail("community-edges.f32 has duplicate macro pair")
            seen.add((source_id, target_id))
            if (
                not math.isfinite(weight)
                or weight <= 0
                or not math.isfinite(x1)
                or not math.isfinite(y1)
            ):
                _fail(f"community-edges.f32 record {index} must be finite with positive weight")
            expected_x, expected_y = centers[target_id]
            if not math.isclose(x1, expected_x, abs_tol=1e-5) or not math.isclose(
                y1, expected_y, abs_tol=1e-5
            ):
                _fail(f"community-edges.f32 record {index} target center differs from labels.json")


def verify_layout_assets(asset_dir: str | Path) -> dict[str, Any]:
    """Return counts only after every v1 asset check has passed."""
    directory = Path(asset_dir).resolve()
    if not directory.is_dir():
        _fail(f"Asset directory does not exist: {directory}")
    manifest = _json(directory / "manifest.json", "manifest.json")
    if manifest.get("schema") != ASSET_SCHEMA:
        _fail(f"manifest schema must be {ASSET_SCHEMA!r}")
    _integer(manifest.get("generated_at_epoch_ms"), "generated_at_epoch_ms")
    maximum = _integer(
        manifest.get("max_label_count", MAX_LABELS), "max_label_count", positive=True
    )
    if maximum > MAX_LABELS:
        _fail(f"max_label_count must be <= {MAX_LABELS}")
    _check_parameters(manifest)
    assets, counts = (
        _mapping(manifest.get("assets"), "assets"),
        _mapping(manifest.get("counts"), "counts"),
    )
    nodes, node_count = _asset(directory, assets, "nodes", "nodes.f32", NODE_RECORD.size)
    edges, edge_count = _asset(directory, assets, "edges", "edges.u32", EDGE_RECORD.size)
    community, community_count = _asset(
        directory, assets, "community_edges", "community-edges.f32", COMMUNITY_EDGE_RECORD.size
    )
    labels, label_count = _asset(directory, assets, "labels", "labels.json", None)
    for name, actual in (
        ("nodes", node_count),
        ("semantic_edges", edge_count),
        ("community_edges", community_count),
        ("labels", label_count),
    ):
        if _integer(counts.get(name), f"counts.{name}") != actual:
            _fail(f"counts.{name} does not match asset records")
    macros = _check_nodes(nodes)
    if not 12 <= len(macros) <= 30:
        _fail("layout must contain 12 to 30 macro communities")
    if macros != set(range(len(macros))):
        _fail("nodes.f32 macroId values must be contiguous from zero")
    if _integer(counts.get("macro_communities"), "counts.macro_communities") != len(macros):
        _fail("counts.macro_communities does not match nodes.f32")
    _check_edges(edges, node_count, layout=False)
    optional_edges, optional_weights = "layout_edges" in assets, "layout_weights" in assets
    if optional_edges != optional_weights:
        _fail("layout-edges.u32 and layout-weights.f32 must be provided together")
    layout_count = 0
    if optional_edges:
        layout_edges, layout_count = _asset(
            directory, assets, "layout_edges", "layout-edges.u32", EDGE_RECORD.size
        )
        layout_weights, _ = _asset(
            directory, assets, "layout_weights", "layout-weights.f32", WEIGHT_RECORD.size
        )
        _check_edges(layout_edges, node_count, layout=True)
        _check_weights(layout_weights, layout_count)
        if _integer(counts.get("layout_edges"), "counts.layout_edges") != layout_count:
            _fail("counts.layout_edges does not match layout assets")
    elif "layout_edges" in counts:
        _fail("counts.layout_edges is present without layout projection assets")
    centers = _check_labels(labels, label_count, macros, maximum)
    _check_community_edges(community, macros, centers)
    return {
        "schema": ASSET_SCHEMA,
        "status": "verified",
        "nodes": node_count,
        "semantic_edges": edge_count,
        "layout_edges": layout_count,
        "community_edges": community_count,
        "macro_communities": len(macros),
        "labels": label_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        print(
            json.dumps(
                verify_layout_assets(arguments.asset_dir), ensure_ascii=False, sort_keys=True
            )
        )
    except LayoutAssetVerificationError as exc:
        print(f"layout asset verification failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
