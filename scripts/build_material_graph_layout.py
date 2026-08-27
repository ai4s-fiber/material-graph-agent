#!/usr/bin/env python3
"""Build stable WebGL layout assets from the real material semantic graph.

The canonical JSONL graph remains the source of truth. This script reads it
without modification and creates a separate, undirected, weighted layout graph
only for community detection and coordinates. The output carries no fragment
text or relationship descriptions and is safe to use as a visualisation bundle.

Requires the intentionally separate offline layout environment:
    $env:PYTHONPATH='.graph-layout-deps'
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "data" / "runtime" / "textbook-graph-bundle"
DEFAULT_CONFIG = ROOT / "config" / "material_graph_layout.v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "runtime" / "material-graph-layout"
ASSET_SCHEMA = "material-graph.layout-assets.v1"
NODE_RECORD_FIELDS = ("x", "y", "size", "r", "g", "b", "macro_id", "flags")
NODE_METRIC_FIELDS = (
    "degree",
    "weighted_degree",
    "page_rank",
    "component_id",
    "micro_community",
    "label_rank",
)
FLAG_ISOLATE = 1
FLAG_HUB_OVERLAY = 2
FLAG_SATELLITE = 4

# 19 muted categorical colors for actual Leiden macro-communities, followed by
# two neutral colors for disconnected-component and isolate macro layers.
PALETTE_HEX = (
    "#5D88C9",
    "#D48172",
    "#73A789",
    "#B692C2",
    "#C99A4B",
    "#5EABB0",
    "#A77866",
    "#8C9D57",
    "#788FD0",
    "#C57E9F",
    "#67A18F",
    "#B28C60",
    "#8C79AE",
    "#BF9B7B",
    "#5C9CB6",
    "#9F8B64",
    "#A57186",
    "#748B9E",
    "#A89758",
    "#9EA7B2",
    "#C9CDD1",
)


class LayoutBuildError(RuntimeError):
    """Raised when an input, configuration, or deterministic build gate fails."""


def _require_layout_dependencies() -> tuple[Any, Any, Any]:
    try:
        import igraph  # type: ignore[import-not-found]
        import leidenalg  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment specific
        raise LayoutBuildError(
            "Missing python-igraph, leidenalg, or NumPy. Use the offline "
            ".graph-layout-deps PYTHONPATH documented in this script."
        ) from exc
    return igraph, leidenalg, np


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise LayoutBuildError(f"Required input was not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LayoutBuildError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise LayoutBuildError(f"Expected object at {path}:{line_number}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_unit(seed: int, *parts: object) -> float:
    payload = "\\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / (2**64 - 1)


def _hex_rgb(color: str) -> tuple[float, float, float]:
    raw = color.lstrip("#")
    if len(raw) != 6:
        raise LayoutBuildError(f"Invalid palette color: {color}")
    return tuple(int(raw[index : index + 2], 16) / 255.0 for index in range(0, 6, 2))  # type: ignore[return-value]


def _ordered_type_pair(left: str, right: str) -> str:
    return "|".join(sorted((left or "Unknown", right or "Unknown")))


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LayoutBuildError(f"Could not read layout config: {path}") from exc
    if not isinstance(config, dict) or config.get("schema") != "material-graph.layout-config.v1":
        raise LayoutBuildError("Layout config schema must be material-graph.layout-config.v1")
    macro = config.get("macro_leiden")
    micro = config.get("micro_leiden")
    visual = config.get("visual")
    satellite = config.get("satellite_grouping")
    if not isinstance(macro, dict) or not isinstance(micro, dict) or not isinstance(visual, dict):
        raise LayoutBuildError(
            "Layout config is missing macro_leiden, micro_leiden, or visual sections"
        )
    if not isinstance(satellite, dict):
        raise LayoutBuildError("Layout config is missing top-level satellite_grouping")
    if satellite.get("algorithm") != "dominant-entity-type-by-component-node-mass.v1":
        raise LayoutBuildError(
            "satellite_grouping.algorithm must be dominant-entity-type-by-component-node-mass.v1"
        )
    try:
        satellite_group_count = int(satellite["group_count"])
        top_type_group_count = int(satellite["top_type_group_count"])
        repulsion_iterations = int(satellite["center_repulsion_iterations"])
        minimum_separation = float(satellite["center_minimum_separation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LayoutBuildError(
            "satellite_grouping requires group_count, top_type_group_count, "
            "center_repulsion_iterations, and center_minimum_separation"
        ) from exc
    if not 4 <= satellite_group_count <= 6:
        raise LayoutBuildError("satellite_grouping.group_count must be between 4 and 6")
    if not 1 <= top_type_group_count < satellite_group_count:
        raise LayoutBuildError(
            "satellite_grouping.top_type_group_count must be between 1 and group_count - 1"
        )
    if repulsion_iterations < 1:
        raise LayoutBuildError("satellite_grouping.center_repulsion_iterations must be positive")
    if not math.isfinite(minimum_separation) or minimum_separation <= 0:
        raise LayoutBuildError("satellite_grouping.center_minimum_separation must be positive")
    residual_label = satellite.get("residual_group_label")
    if (
        not isinstance(residual_label, str)
        or not residual_label.strip()
        or len(residual_label) > 160
    ):
        raise LayoutBuildError("satellite_grouping.residual_group_label must be 1..160 characters")
    for section_name, section in (("macro_leiden", macro), ("micro_leiden", micro)):
        try:
            resolution = float(section["resolution"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LayoutBuildError(f"{section_name}.resolution must be a positive number") from exc
        if not math.isfinite(resolution) or resolution <= 0:
            raise LayoutBuildError(f"{section_name}.resolution must be a positive number")
    target = int(macro.get("main_community_target", 0))
    accepted = macro.get("accepted_macro_community_range")
    if not isinstance(accepted, list) or len(accepted) != 2:
        raise LayoutBuildError("accepted_macro_community_range must have two integers")
    if not int(accepted[0]) <= target + satellite_group_count + 1 <= int(accepted[1]):
        raise LayoutBuildError(
            "main_community_target + configured satellite groups + isolate must be inside "
            "accepted macro range"
        )
    return config


def _mention_weight(mentions: object) -> float:
    try:
        count = max(0, int(mentions))
    except (TypeError, ValueError):
        count = 1
    return 1.0 + 0.20 * math.log1p(max(0, count - 1))


def _read_semantic_graph(
    entities_path: Path,
    relationships_path: Path,
    config: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    """Read semantic graph untouched and form the separately weighted layout projection."""

    ids: list[str] = []
    names: list[str] = []
    types: list[str] = []
    id_to_index: dict[str, int] = {}
    duplicate_entities = 0
    for row in _iter_jsonl(entities_path):
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            raise LayoutBuildError("Canonical entity row is missing entity_id")
        if entity_id in id_to_index:
            duplicate_entities += 1
            continue
        id_to_index[entity_id] = len(ids)
        ids.append(entity_id)
        names.append(str(row.get("name") or entity_id))
        types.append(str(row.get("entity_type") or "Unknown"))

    if not ids:
        raise LayoutBuildError("Cannot build a layout from an empty entity input")
    layout_config = config["layout_graph"]
    type_weights = layout_config.get("entity_type_pair_weights", {})
    default_type_weight = float(layout_config.get("default_entity_type_pair_weight", 1.0))
    pair_weights: dict[int, float] = defaultdict(float)
    semantic_edges = array("I")
    semantic_edge_rows = self_loops = missing_endpoints = 0
    semantic_type_pairs: Counter[str] = Counter()
    node_count = len(ids)

    for row in _iter_jsonl(relationships_path):
        semantic_edge_rows += 1
        source_id = str(row.get("source_entity_id", "")).strip()
        target_id = str(row.get("target_entity_id", "")).strip()
        source = id_to_index.get(source_id)
        target = id_to_index.get(target_id)
        if source is None or target is None:
            missing_endpoints += 1
            continue
        semantic_edges.extend((source, target))
        semantic_type_pairs[f"{types[source]} -> {types[target]}"] += 1
        if source == target:
            self_loops += 1
            continue
        left, right = (source, target) if source < target else (target, source)
        type_pair = _ordered_type_pair(types[left], types[right])
        type_weight = float(type_weights.get(type_pair, default_type_weight))
        pair_weights[left * node_count + right] += (
            float(layout_config.get("relation_base_weight", 1.0))
            * type_weight
            * _mention_weight(row.get("mentions", 1))
        )

    if missing_endpoints:
        raise LayoutBuildError(
            f"Refusing layout build: {missing_endpoints} semantic edges have missing endpoints"
        )

    ordered_pairs = sorted(pair_weights)
    layout_edges = np.empty((len(ordered_pairs), 2), dtype=np.uint32)
    base_weights = np.empty(len(ordered_pairs), dtype=np.float64)
    for offset, pair_key in enumerate(ordered_pairs):
        layout_edges[offset, 0] = pair_key // node_count
        layout_edges[offset, 1] = pair_key % node_count
        base_weights[offset] = pair_weights[pair_key]

    degree = np.bincount(layout_edges.reshape(-1), minlength=node_count).astype(np.float64)
    hub_config = layout_config["hub_damping"]
    threshold = float(hub_config["degree_threshold"])
    exponent = float(hub_config["exponent"])
    if bool(hub_config.get("enabled", True)):
        damping = np.minimum(1.0, (threshold / np.maximum(degree, threshold)) ** exponent)
        weights = base_weights * np.sqrt(damping[layout_edges[:, 0]] * damping[layout_edges[:, 1]])
    else:
        damping = np.ones(node_count, dtype=np.float64)
        weights = base_weights.copy()
    weighted_degree = np.bincount(
        layout_edges[:, 0], weights=weights, minlength=node_count
    ) + np.bincount(layout_edges[:, 1], weights=weights, minlength=node_count)
    return {
        "ids": ids,
        "names": names,
        "types": types,
        "semantic_edges": np.asarray(semantic_edges, dtype=np.uint32).reshape(-1, 2),
        "layout_edges": layout_edges,
        "base_weights": base_weights,
        "weights": weights,
        "degree": degree,
        "weighted_degree": weighted_degree,
        "hub_damping": damping,
        "semantic_summary": {
            "nodes": node_count,
            "directed_edges": semantic_edge_rows,
            "valid_directed_edges": int(len(semantic_edges) // 2),
            "self_loops": self_loops,
            "missing_endpoints": missing_endpoints,
            "duplicate_entities": duplicate_entities,
            "relationship_taxonomy_basis": "source entity_type -> target entity_type",
            "relationship_taxonomy": [
                {"source_to_target": pair, "count": count}
                for pair, count in sorted(
                    semantic_type_pairs.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        },
        "layout_summary": {
            "undirected_weighted_edges": int(len(layout_edges)),
            "dropped_self_loops": self_loops,
            "coalesced_duplicate_undirected_rows": int(
                len(semantic_edges) // 2 - self_loops - len(layout_edges)
            ),
        },
    }


def _seeded_initial_coordinates(count: int, seed: int) -> list[list[float]]:
    generator = random.Random(seed)
    return [[generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0)] for _ in range(count)]


def _normalise_cloud(points: Any, np: Any) -> Any:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(values) == 1:
        return np.zeros((1, 2), dtype=np.float64)
    values = values[:, :2]
    values -= np.median(values, axis=0)
    radii = np.hypot(values[:, 0], values[:, 1])
    scale = float(np.quantile(radii, 0.985))
    if not math.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(values)
    return values / scale


def _deterministic_small_cloud(vertices: list[int], seed: int, np: Any) -> Any:
    result = np.zeros((len(vertices), 2), dtype=np.float64)
    if len(vertices) < 2:
        return result
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for local_index, vertex in enumerate(vertices):
        angle = golden * local_index + _stable_unit(seed, "small-angle", vertex) * 0.55
        radius = math.sqrt((local_index + 0.5) / len(vertices))
        result[local_index] = (math.cos(angle) * radius, math.sin(angle) * radius)
    return result


def _force_layout(
    graph: Any,
    vertices: list[int],
    *,
    seed: int,
    options: str,
    minimum_force_nodes: int,
    np: Any,
) -> Any:
    if len(vertices) < minimum_force_nodes:
        return _deterministic_small_cloud(vertices, seed, np)
    subgraph = graph.induced_subgraph(vertices)
    try:
        coordinates = subgraph.layout_drl(
            weights="weight",
            seed=_seeded_initial_coordinates(len(vertices), seed),
            options=options,
        )
    except Exception as exc:  # pragma: no cover - native igraph failure
        raise LayoutBuildError(f"igraph.drl failed for a {len(vertices)}-node community") from exc
    return _normalise_cloud(coordinates, np)


def _macro_centers(
    macro_count: int,
    core_count: int,
    layout_edges: Any,
    weights: Any,
    macro_ids: Any,
    config: Mapping[str, Any],
    igraph: Any,
    np: Any,
) -> tuple[Any, dict[tuple[int, int], float]]:
    inter_weights: dict[tuple[int, int], float] = defaultdict(float)
    for (left, right), weight in zip(layout_edges, weights, strict=True):
        source = int(macro_ids[int(left)])
        target = int(macro_ids[int(right)])
        if source >= core_count or target >= core_count or source == target:
            continue
        if source > target:
            source, target = target, source
        inter_weights[(source, target)] += float(weight)

    centers = np.zeros((macro_count, 2), dtype=np.float64)
    macro_graph = igraph.Graph(n=core_count, edges=list(inter_weights), directed=False)
    if macro_graph.ecount():
        macro_graph.es["weight"] = [inter_weights[edge] for edge in inter_weights]
        raw = macro_graph.layout_drl(
            weights="weight",
            seed=_seeded_initial_coordinates(core_count, int(config["seed"]) + 911),
            options=str(config["multilevel_layout"]["macro_layout_options"]),
        )
        normalized = _normalise_cloud(raw, np)
    else:  # pragma: no cover - the canonical giant component has bridges
        normalized = _deterministic_small_cloud(list(range(core_count)), int(config["seed"]), np)
    extent = float(config["multilevel_layout"]["main_center_extent"])
    centers[:core_count] = normalized * extent
    return centers, inter_weights


def _satellite_component_groups(
    component_nodes: list[list[int]],
    entity_types: list[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Group real non-isolate components by dominant entity-type node mass."""

    component_rows: list[dict[str, Any]] = []
    for component_id, vertices in enumerate(component_nodes):
        if len(vertices) <= 1:
            continue
        type_counts = Counter(entity_types[vertex] or "Unknown" for vertex in vertices)
        dominant_type = min(type_counts, key=lambda item: (-type_counts[item], item))
        component_rows.append(
            {
                "component_id": component_id,
                "vertices": vertices,
                "node_count": len(vertices),
                "dominant_entity_type": dominant_type,
            }
        )
    component_rows.sort(key=lambda item: (-int(item["node_count"]), int(item["component_id"])))
    if not component_rows:
        return []

    grouping = config["satellite_grouping"]
    node_mass_by_type: Counter[str] = Counter()
    components_by_type: Counter[str] = Counter()
    for item in component_rows:
        dominant_type = str(item["dominant_entity_type"])
        node_mass_by_type[dominant_type] += int(item["node_count"])
        components_by_type[dominant_type] += 1
    top_type_count = int(grouping["top_type_group_count"])
    selected_types = sorted(
        node_mass_by_type,
        key=lambda item: (-node_mass_by_type[item], -components_by_type[item], item),
    )[:top_type_count]
    groups = [
        {
            "key": f"dominant-type:{entity_type}",
            "label": f"外围分量 · {entity_type}",
            "dominant_entity_type": entity_type,
            "kind": "dominant_entity_type",
            "components": [],
        }
        for entity_type in selected_types
    ]
    groups.append(
        {
            "key": "residual",
            "label": str(grouping["residual_group_label"]),
            "dominant_entity_type": None,
            "kind": "residual",
            "components": [],
        }
    )
    group_for_type = {entity_type: index for index, entity_type in enumerate(selected_types)}
    residual_index = len(groups) - 1
    for item in component_rows:
        group_index = group_for_type.get(str(item["dominant_entity_type"]), residual_index)
        groups[group_index]["components"].append(item)

    active_groups = [item for item in groups if item["components"]]
    for group in active_groups:
        group["node_count"] = sum(int(item["node_count"]) for item in group["components"])
        group["component_count"] = len(group["components"])
    return active_groups


def _satellite_group_centers(
    groups: list[dict[str, Any]], config: Mapping[str, Any], np: Any
) -> Any:
    """Return seed-stable, eccentric peripheral anchors with deterministic repulsion."""

    centers = np.zeros((len(groups), 2), dtype=np.float64)
    if not groups:
        return centers
    seed = int(config["seed"])
    layout = config["multilevel_layout"]
    grouping = config["satellite_grouping"]
    radius_min = float(layout["satellite_radius_min"])
    radius_max = float(layout["satellite_radius_max"])
    for index, group in enumerate(groups):
        key = str(group["key"])
        angle = math.tau * _stable_unit(seed, "satellite-group-angle", key)
        radius = radius_min + (radius_max - radius_min) * (
            0.14 + 0.72 * _stable_unit(seed, "satellite-group-radius", key)
        )
        horizontal = 0.86 + 0.22 * _stable_unit(seed, "satellite-group-horizontal", key)
        vertical = 0.60 + 0.22 * _stable_unit(seed, "satellite-group-vertical", key)
        centers[index] = (
            math.cos(angle) * radius * horizontal,
            math.sin(angle) * radius * vertical,
        )

    minimum = float(grouping["center_minimum_separation"])
    # The final eccentric-band projection and manifest rounding can reduce a pair
    # by a few ulps, so retain a deterministic margin above the public floor.
    placement_minimum = minimum + max(1e-5, minimum * 1e-4)
    for _ in range(int(grouping["center_repulsion_iterations"])):
        adjustments = np.zeros_like(centers)
        for left in range(len(centers)):
            for right in range(left + 1, len(centers)):
                offset = centers[right] - centers[left]
                distance = float(np.hypot(offset[0], offset[1]))
                if distance >= placement_minimum:
                    continue
                if distance < 1e-9:
                    direction_angle = math.tau * _stable_unit(
                        seed, "satellite-group-repel", groups[left]["key"], groups[right]["key"]
                    )
                    direction = np.asarray(
                        (math.cos(direction_angle), math.sin(direction_angle)), dtype=np.float64
                    )
                else:
                    direction = offset / distance
                adjustment = direction * ((placement_minimum - distance) * 0.5)
                adjustments[left] -= adjustment
                adjustments[right] += adjustment
        centers += adjustments
        for index, center in enumerate(centers):
            elliptical_radius = float(np.hypot(center[0] / 1.04, center[1] / 0.76))
            bounded_radius = min(radius_max, max(radius_min, elliptical_radius))
            if elliptical_radius < 1e-9:
                angle = math.tau * _stable_unit(
                    seed, "satellite-group-fallback", groups[index]["key"]
                )
                centers[index] = (
                    math.cos(angle) * bounded_radius * 1.04,
                    math.sin(angle) * bounded_radius * 0.76,
                )
            else:
                centers[index] *= bounded_radius / elliptical_radius
    return centers


def _place_satellite_component_groups(
    graph: Any,
    groups: list[dict[str, Any]],
    centers: Any,
    positions: Any,
    config: Mapping[str, Any],
    np: Any,
) -> None:
    """Place each actual component around its type-derived irregular group anchor."""

    if not groups:
        return
    seed = int(config["seed"])
    layout = config["multilevel_layout"]
    total_nodes = sum(int(group["node_count"]) for group in groups)
    group_radius_min = float(layout["satellite_group_radius_min"])
    group_radius_max = float(layout["satellite_group_radius_max"])
    for group_rank, group in enumerate(groups):
        group_extent = group_radius_min + (group_radius_max - group_radius_min) * math.sqrt(
            int(group["node_count"]) / max(1, total_nodes)
        )
        components = sorted(
            group["components"],
            key=lambda item: (-int(item["node_count"]), int(item["component_id"])),
        )
        for component_rank, component in enumerate(components):
            component_id = int(component["component_id"])
            vertices = component["vertices"]
            angle = math.tau * _stable_unit(seed, "satellite-component-angle", component_id)
            size_pull = min(0.42, math.log1p(len(vertices)) * 0.060)
            radial = (
                group_extent
                * (
                    0.14
                    + 0.78
                    * math.sqrt(_stable_unit(seed, "satellite-component-radius", component_id))
                )
                * (1.0 - size_pull)
            )
            horizontal = 0.82 + 0.22 * _stable_unit(
                seed, "satellite-component-horizontal", component_id
            )
            vertical = 0.63 + 0.24 * _stable_unit(
                seed, "satellite-component-vertical", component_id
            )
            component_center = centers[group_rank] + np.asarray(
                (math.cos(angle) * radial * horizontal, math.sin(angle) * radial * vertical),
                dtype=np.float64,
            )
            local = _force_layout(
                graph,
                vertices,
                seed=seed + 50_000 + group_rank * 10_000 + component_rank,
                options=str(layout["community_layout_options"]),
                minimum_force_nodes=int(layout["minimum_force_layout_nodes"]),
                np=np,
            )
            rotation_angle = math.tau * _stable_unit(
                seed, "satellite-component-rotation", component_id
            )
            rotation = np.asarray(
                (
                    (math.cos(rotation_angle), -math.sin(rotation_angle)),
                    (math.sin(rotation_angle), math.cos(rotation_angle)),
                ),
                dtype=np.float64,
            )
            local = local @ rotation.T
            local_radius = min(0.042, 0.007 + math.sqrt(len(vertices)) * 0.0011)
            positions[vertices] = component_center + local * local_radius


def _stable_micro_membership(
    graph: Any,
    vertices: list[int],
    *,
    macro_id: int,
    config: Mapping[str, Any],
    leidenalg: Any,
    np: Any,
) -> tuple[Any, int]:
    """Partition one final macro layer with deterministic, independently seeded Leiden."""

    if not vertices:
        return np.empty(0, dtype=np.int64), 0
    if len(vertices) == 1:
        return np.zeros(1, dtype=np.int64), 1

    subgraph = graph.induced_subgraph(vertices)
    if subgraph.ecount() == 0:
        # A disconnected macro layer (normally the isolate overlay) has one real
        # structural micro-community per vertex without invoking a native solver.
        membership = np.arange(len(vertices), dtype=np.int64)
    else:
        micro_config = config["micro_leiden"]
        try:
            partition = leidenalg.find_partition(
                subgraph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=float(micro_config["resolution"]),
                seed=int(config["seed"]) + 20_000 + macro_id,
                n_iterations=-1,
            )
        except Exception as exc:  # pragma: no cover - native leiden failure
            raise LayoutBuildError(
                f"Leiden micro-community detection failed for macro {macro_id}"
            ) from exc
        membership = np.asarray(partition.membership, dtype=np.int64)

    members_by_partition: dict[int, list[int]] = defaultdict(list)
    for local_index, partition_id in enumerate(membership):
        members_by_partition[int(partition_id)].append(local_index)
    ordered_partitions = sorted(
        members_by_partition,
        key=lambda partition_id: (
            -len(members_by_partition[partition_id]),
            min(vertices[index] for index in members_by_partition[partition_id]),
            partition_id,
        ),
    )
    stable_ids = {partition_id: index for index, partition_id in enumerate(ordered_partitions)}
    return (
        np.asarray([stable_ids[int(partition_id)] for partition_id in membership], dtype=np.int64),
        len(ordered_partitions),
    )


def _compute_hierarchical_micro_communities(
    graph: Any,
    macro_ids: Any,
    macro_count: int,
    config: Mapping[str, Any],
    leidenalg: Any,
    np: Any,
) -> tuple[Any, list[dict[str, int]]]:
    """Run a separate Leiden level inside each final macro-community.

    The macro grouping may attach several coarse partitions to one visual cloud.
    Re-running Leiden here is deliberate: it makes microCommunity a true child
    partition of that final macro layer rather than an accidental alias of the
    coarser macro partition.
    """

    micro_ids = np.full(len(macro_ids), -1, dtype=np.int64)
    next_micro_id = 0
    summaries: list[dict[str, int]] = []
    for macro_id in range(macro_count):
        vertices = np.flatnonzero(macro_ids == macro_id).astype(int).tolist()
        local_membership, local_count = _stable_micro_membership(
            graph,
            vertices,
            macro_id=macro_id,
            config=config,
            leidenalg=leidenalg,
            np=np,
        )
        if len(local_membership) != len(vertices):  # pragma: no cover - defensive native gate
            raise LayoutBuildError(f"Micro membership length mismatch for macro {macro_id}")
        if vertices:
            micro_ids[vertices] = local_membership + next_micro_id
        summaries.append(
            {
                "macroId": macro_id,
                "nodes": len(vertices),
                "microCommunities": local_count,
            }
        )
        next_micro_id += local_count
    if (micro_ids < 0).any():  # pragma: no cover - defensive complete coverage gate
        raise LayoutBuildError("Micro-community detection did not cover every layout node")
    return micro_ids, summaries


def _compute_communities_and_positions(
    semantic: Mapping[str, Any],
    config: Mapping[str, Any],
    igraph: Any,
    leidenalg: Any,
    np: Any,
) -> dict[str, Any]:
    node_count = len(semantic["ids"])
    layout_edges = semantic["layout_edges"]
    weights = semantic["weights"]
    graph = igraph.Graph(n=node_count, edges=layout_edges.tolist(), directed=False)
    graph.es["weight"] = weights.tolist()
    components = graph.connected_components()
    component_ids = np.asarray(components.membership, dtype=np.int64)
    component_sizes = np.bincount(component_ids, minlength=len(components)).astype(np.int64)
    giant_component = int(component_sizes.argmax())
    giant_vertices = np.flatnonzero(component_ids == giant_component).astype(int).tolist()
    component_nodes: list[list[int]] = [[] for _ in range(len(components))]
    for vertex, component in enumerate(component_ids):
        if int(component) != giant_component:
            component_nodes[int(component)].append(vertex)
    satellite_groups = _satellite_component_groups(component_nodes, semantic["types"], config)

    giant_graph = graph.induced_subgraph(giant_vertices)
    macro_config = config["macro_leiden"]
    macro_partition = leidenalg.find_partition(
        giant_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(macro_config["resolution"]),
        seed=int(config["seed"]),
        n_iterations=-1,
    )
    giant_macro_partitions = np.asarray(macro_partition.membership, dtype=np.int64)
    members_by_macro_partition: dict[int, list[int]] = defaultdict(list)
    for vertex, macro_partition_id in zip(giant_vertices, giant_macro_partitions, strict=True):
        members_by_macro_partition[int(macro_partition_id)].append(vertex)
    target = int(macro_config["main_community_target"])
    minimum_size = int(macro_config["minimum_main_community_size"])
    ordered_macro_partitions = sorted(
        members_by_macro_partition,
        key=lambda item: (-len(members_by_macro_partition[item]), item),
    )
    eligible_macro_partitions = [
        item
        for item in ordered_macro_partitions
        if len(members_by_macro_partition[item]) >= minimum_size
    ]
    selected_macro_partitions = eligible_macro_partitions[:target]
    if len(selected_macro_partitions) < target:
        raise LayoutBuildError(
            "Leiden produced only "
            f"{len(selected_macro_partitions)} eligible macro communities; target is {target}"
        )
    macro_partition_to_macro = {
        macro_partition_id: macro
        for macro, macro_partition_id in enumerate(selected_macro_partitions)
    }

    # Every smaller coarse partition in the giant component attaches to the major
    # macro cloud with the greatest actual cross-partition layout weight. This
    # creates the visual macro layer only; true child micro communities are
    # independently detected after the final macro memberships are established.
    giant_local = np.full(node_count, -1, dtype=np.int64)
    giant_local[giant_vertices] = np.arange(len(giant_vertices), dtype=np.int64)
    attachment: dict[int, Counter[int]] = defaultdict(Counter)
    for (left, right), weight in zip(layout_edges, weights, strict=True):
        local_left = int(giant_local[int(left)])
        local_right = int(giant_local[int(right)])
        if local_left < 0 or local_right < 0:
            continue
        left_partition = int(giant_macro_partitions[local_left])
        right_partition = int(giant_macro_partitions[local_right])
        if left_partition == right_partition:
            continue
        if left_partition in macro_partition_to_macro:
            attachment[right_partition][macro_partition_to_macro[left_partition]] += float(weight)
        if right_partition in macro_partition_to_macro:
            attachment[left_partition][macro_partition_to_macro[right_partition]] += float(weight)
    for macro_partition_id in ordered_macro_partitions:
        if macro_partition_id in macro_partition_to_macro:
            continue
        candidates = attachment.get(macro_partition_id)
        if candidates:
            macro_partition_to_macro[macro_partition_id] = min(
                candidates,
                key=lambda item: (-candidates[item], item),
            )
        else:
            macro_partition_to_macro[macro_partition_id] = 0

    core_count = target
    for group_index, group in enumerate(satellite_groups):
        group["macro_id"] = core_count + group_index
    isolate_macro = core_count + len(satellite_groups)
    macro_count = isolate_macro + 1
    accepted_min, accepted_max = (
        int(item) for item in macro_config["accepted_macro_community_range"]
    )
    if not accepted_min <= macro_count <= accepted_max:
        raise LayoutBuildError(f"Macro community count {macro_count} is outside configured range")

    macro_ids = np.full(node_count, isolate_macro, dtype=np.int64)
    for vertex, macro_partition_id in zip(giant_vertices, giant_macro_partitions, strict=True):
        macro_ids[vertex] = macro_partition_to_macro[int(macro_partition_id)]
    for group in satellite_groups:
        macro_id = int(group["macro_id"])
        for component in group["components"]:
            macro_ids[component["vertices"]] = macro_id
    micro_ids, micro_communities_by_macro = _compute_hierarchical_micro_communities(
        graph,
        macro_ids,
        macro_count,
        config,
        leidenalg,
        np,
    )

    centers, inter_weights = _macro_centers(
        macro_count, core_count, layout_edges, weights, macro_ids, config, igraph, np
    )
    satellite_centers = _satellite_group_centers(satellite_groups, config, np)
    for group_index, group in enumerate(satellite_groups):
        centers[int(group["macro_id"])] = satellite_centers[group_index]
    positions = np.zeros((node_count, 2), dtype=np.float64)
    layout_settings = config["multilevel_layout"]
    for macro in range(core_count):
        vertices = np.flatnonzero(macro_ids == macro).astype(int).tolist()
        local = _force_layout(
            graph,
            vertices,
            seed=int(config["seed"]) + 1000 + macro,
            options=str(layout_settings["community_layout_options"]),
            minimum_force_nodes=int(layout_settings["minimum_force_layout_nodes"]),
            np=np,
        )
        mass = len(vertices) / max(1, len(giant_vertices))
        radius = float(layout_settings["local_radius_min"]) + (
            float(layout_settings["local_radius_max"]) - float(layout_settings["local_radius_min"])
        ) * math.sqrt(mass)
        angle = _stable_unit(int(config["seed"]), "macro-rotation", macro) * math.tau
        stretch = 0.73 + _stable_unit(int(config["seed"]), "macro-stretch", macro) * 0.54
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=np.float64,
        )
        local = local @ rotation.T
        local[:, 0] *= stretch
        local[:, 1] /= stretch
        positions[vertices] = centers[macro] + local * radius

    _place_satellite_component_groups(
        graph, satellite_groups, satellite_centers, positions, config, np
    )

    isolate_indices = np.flatnonzero(component_sizes[component_ids] == 1).astype(int)
    for vertex in isolate_indices:
        angle = math.tau * _stable_unit(int(config["seed"]), "isolate-angle", int(vertex))
        radius = float(layout_settings["isolate_radius_min"]) + (
            float(layout_settings["isolate_radius_max"])
            - float(layout_settings["isolate_radius_min"])
        ) * _stable_unit(int(config["seed"]), "isolate-radius", int(vertex))
        positions[vertex] = (
            math.cos(angle) * radius * 1.04,
            math.sin(angle) * radius * 0.77,
        )

    # Community labels use medians of true fixed coordinates.
    for macro in range(macro_count):
        vertices = np.flatnonzero(macro_ids == macro)
        if len(vertices):
            centers[macro] = np.median(positions[vertices], axis=0)
    extent = float(config["visual"]["coordinate_extent"])
    positions = np.clip(positions, -extent, extent)
    community_edges = [
        (
            float(source),
            float(target),
            float(weight),
            float(centers[target, 0]),
            float(centers[target, 1]),
        )
        for (source, target), weight in sorted(inter_weights.items())
    ]
    satellite_group_summaries = []
    for group_index, group in enumerate(satellite_groups):
        macro_id = int(group["macro_id"])
        satellite_group_summaries.append(
            {
                "macroId": macro_id,
                "label": str(group["label"]),
                "kind": str(group["kind"]),
                "dominantEntityType": group["dominant_entity_type"],
                "componentCount": int(group["component_count"]),
                "nodeCount": int(group["node_count"]),
                "placementCenter": [
                    round(float(satellite_centers[group_index, 0]), 7),
                    round(float(satellite_centers[group_index, 1]), 7),
                ],
                "coordinateCenter": [
                    round(float(centers[macro_id, 0]), 7),
                    round(float(centers[macro_id, 1]), 7),
                ],
            }
        )
    return {
        "graph": graph,
        "positions": positions,
        "component_ids": component_ids,
        "component_sizes": component_sizes,
        "giant_component": giant_component,
        "macro_ids": macro_ids,
        "micro_ids": micro_ids,
        "macro_count": macro_count,
        "micro_count": sum(item["microCommunities"] for item in micro_communities_by_macro),
        "micro_communities_by_macro": micro_communities_by_macro,
        "core_macro_count": core_count,
        "satellite_groups": satellite_group_summaries,
        "macro_centers": centers,
        "community_edges": np.asarray(community_edges, dtype=np.float32).reshape(-1, 5),
        "community_sizes": np.bincount(macro_ids, minlength=macro_count).astype(np.int64),
    }


def _pagerank(semantic_edges: Any, node_count: int, igraph: Any, np: Any) -> Any:
    if not len(semantic_edges):
        return np.zeros(node_count, dtype=np.float64)
    semantic_graph = igraph.Graph(n=node_count, edges=semantic_edges.tolist(), directed=True)
    values = semantic_graph.pagerank(directed=True, damping=0.85)
    return np.asarray(values, dtype=np.float64)


def _quantile_normalise(values: Any, lower: float, upper: float, np: Any) -> Any:
    low, high = (float(item) for item in np.percentile(values, (lower, upper)))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _write_asset_record(
    path: Path,
    records: int,
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "records": records,
    }
    if fields:
        payload["fields"] = list(fields)
    return payload


def _write_node_index(
    path: Path,
    ids: list[str],
    names: list[str],
    types: list[str],
    macro_ids: Any,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE node_index (entity_id TEXT PRIMARY KEY, node_index INTEGER NOT NULL, "
            "name TEXT NOT NULL, entity_type TEXT NOT NULL, macro_id INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute("CREATE INDEX node_index_name ON node_index(name)")
        connection.executemany(
            "INSERT INTO node_index(entity_id, node_index, name, entity_type, macro_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (entity_id, index, names[index], types[index], int(macro_ids[index]))
                for index, entity_id in enumerate(ids)
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _build_labels(
    names: list[str],
    macro_ids: Any,
    centers: Any,
    community_sizes: Any,
    satellite_groups: list[Mapping[str, Any]],
    label_rank: Any,
    config: Mapping[str, Any],
    np: Any,
) -> dict[str, Any]:
    visual = config["visual"]
    max_labels = int(visual["max_label_count"])
    hub_count = min(int(visual["hub_label_count"]), max_labels)
    macro_count = len(centers)
    satellite_labels = {int(group["macroId"]): str(group["label"]) for group in satellite_groups}
    node_indices = np.argsort(-label_rank, kind="stable")[:hub_count]
    communities = [
        {
            "macroId": macro,
            "label": "断开孤立节点"
            if macro == macro_count - 1
            else satellite_labels.get(macro, f"Leiden 社区 {macro + 1:02d}"),
            "nodeCount": int(community_sizes[macro]),
            "x": round(float(centers[macro, 0]), 7),
            "y": round(float(centers[macro, 1]), 7),
        }
        for macro in range(macro_count)
    ]
    nodes = [
        {
            "index": int(index),
            "label": names[int(index)],
            "labelRank": round(float(label_rank[int(index)]), 7),
        }
        for index in node_indices
    ]
    if len(communities) + len(nodes) > max_labels:
        nodes = nodes[: max(0, max_labels - len(communities))]
    return {"schema": "material-graph.layout-labels.v1", "communities": communities, "nodes": nodes}


def build_layout_assets(
    entities_path: Path,
    relationships_path: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Build immutable derived assets. Existing output directories are never overwritten."""

    started = time.perf_counter()
    igraph, leidenalg, np = _require_layout_dependencies()
    entities_path = Path(entities_path).resolve()
    relationships_path = Path(relationships_path).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    if output_dir.exists():
        raise LayoutBuildError(
            f"Output already exists and will not be overwritten: {output_dir}. "
            "Choose a new derived asset generation directory."
        )
    config = _load_config(config_path)
    set_igraph_rng = getattr(igraph, "set_random_number_generator", None)
    if not callable(set_igraph_rng):  # pragma: no cover - pinned offline dependency contract
        raise LayoutBuildError("python-igraph does not expose a seedable random number generator")
    # layout_drl consumes igraph's global RNG beyond the explicit initial coordinates.
    # Reset it per immutable build so a fixed config seed reproduces all coordinates.
    set_igraph_rng(random.Random(int(config["seed"])))
    semantic = _read_semantic_graph(entities_path, relationships_path, config, np)
    community = _compute_communities_and_positions(semantic, config, igraph, leidenalg, np)
    page_rank = _pagerank(semantic["semantic_edges"], len(semantic["ids"]), igraph, np)

    visual = config["visual"]
    rank_value = np.log1p(semantic["weighted_degree"])
    normalised_rank = _quantile_normalise(
        rank_value,
        float(visual["rank_lower_percentile"]),
        float(visual["rank_upper_percentile"]),
        np,
    )
    page_rank_norm = _quantile_normalise(page_rank, 5.0, 99.5, np)
    visual_sizes = np.clip(
        float(visual["normal_node_size_min"])
        + normalised_rank**3.25
        * (float(visual["normal_node_size_max"]) - float(visual["normal_node_size_min"]))
        + page_rank_norm**4 * 0.65,
        float(visual["normal_node_size_min"]),
        float(visual["normal_node_size_max"]),
    )
    flags = np.zeros(len(semantic["ids"]), dtype=np.uint32)
    flags[semantic["degree"] == 0] |= FLAG_ISOLATE
    flags[
        semantic["degree"]
        >= float(config["layout_graph"]["hub_damping"]["overlay_degree_threshold"])
    ] |= FLAG_HUB_OVERLAY
    flags[community["component_ids"] != community["giant_component"]] |= FLAG_SATELLITE
    colors = np.asarray([_hex_rgb(value) for value in PALETTE_HEX], dtype=np.float64)
    visual_colors = colors[community["macro_ids"] % len(colors)]
    label_rank = 0.78 * normalised_rank + 0.22 * page_rank_norm

    node_records = np.column_stack(
        (
            community["positions"],
            visual_sizes,
            visual_colors,
            community["macro_ids"].astype(np.float64),
            flags.astype(np.float64),
        )
    ).astype("<f4")
    metrics = np.column_stack(
        (
            semantic["degree"],
            semantic["weighted_degree"],
            page_rank,
            community["component_ids"],
            community["micro_ids"],
            label_rank,
        )
    ).astype("<f4")
    labels = _build_labels(
        semantic["names"],
        community["macro_ids"],
        community["macro_centers"],
        community["community_sizes"],
        community["satellite_groups"],
        label_rank,
        config,
        np,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        nodes_path = staging_dir / "nodes.f32"
        metrics_path = staging_dir / "node-metrics.f32"
        semantic_edges_path = staging_dir / "edges.u32"
        layout_edges_path = staging_dir / "layout-edges.u32"
        layout_weights_path = staging_dir / "layout-weights.f32"
        community_edges_path = staging_dir / "community-edges.f32"
        labels_path = staging_dir / "labels.json"
        node_index_path = staging_dir / "node-index.sqlite3"
        node_records.tofile(nodes_path)
        metrics.tofile(metrics_path)
        semantic["semantic_edges"].astype("<u4", copy=False).tofile(semantic_edges_path)
        semantic["layout_edges"].astype("<u4", copy=False).tofile(layout_edges_path)
        semantic["weights"].astype("<f4", copy=False).tofile(layout_weights_path)
        community["community_edges"].astype("<f4", copy=False).tofile(community_edges_path)
        labels_path.write_text(
            json.dumps(labels, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_node_index(
            node_index_path,
            semantic["ids"],
            semantic["names"],
            semantic["types"],
            community["macro_ids"],
        )
        assets = {
            "nodes": _write_asset_record(nodes_path, len(node_records), NODE_RECORD_FIELDS),
            "node_metrics": _write_asset_record(metrics_path, len(metrics), NODE_METRIC_FIELDS),
            "edges": _write_asset_record(
                semantic_edges_path,
                len(semantic["semantic_edges"]),
                ("source_index", "target_index"),
            ),
            "layout_edges": _write_asset_record(
                layout_edges_path, len(semantic["layout_edges"]), ("left_index", "right_index")
            ),
            "layout_weights": _write_asset_record(
                layout_weights_path, len(semantic["weights"]), ("weight",)
            ),
            "community_edges": _write_asset_record(
                community_edges_path,
                len(community["community_edges"]),
                ("source_macro", "target_macro", "weight", "target_x", "target_y"),
            ),
            "labels": _write_asset_record(
                labels_path, len(labels["communities"]) + len(labels["nodes"])
            ),
            "node_index": _write_asset_record(node_index_path, len(semantic["ids"])),
        }
        manifest = {
            "schema": ASSET_SCHEMA,
            "layout_version": output_dir.name,
            "lineage": {
                "parent_layout_version": config.get("layout_lineage", {}).get(
                    "parent_layout_version"
                ),
                "revision": config.get("layout_lineage", {}).get("revision"),
                "semantic_graph_preserved": True,
            },
            "generated_at_epoch_ms": int(time.time() * 1000),
            "max_label_count": int(visual["max_label_count"]),
            "coordinate_system": {
                "x_y_range": [
                    -float(visual["coordinate_extent"]),
                    float(visual["coordinate_extent"]),
                ],
                "endianness": "little",
            },
            "inputs": {
                "entities": {
                    "path": entities_path.name,
                    "sha256": _sha256(entities_path),
                    "records": len(semantic["ids"]),
                },
                "relationships": {
                    "path": relationships_path.name,
                    "sha256": _sha256(relationships_path),
                    "records": int(semantic["semantic_summary"]["directed_edges"]),
                },
            },
            "parameters": {
                "layout_algorithm": "Leiden macro + hierarchical micro RBConfiguration + igraph.drl coarse-to-fine",
                "random_seed": int(config["seed"]),
                "macro_resolution": float(config["macro_leiden"]["resolution"]),
                "micro_resolution": float(config["micro_leiden"]["resolution"]),
                "micro_partitioning": "independent Leiden within each final macroCommunity",
                "hub_damping": config["layout_graph"]["hub_damping"],
                "satellite_grouping": config["satellite_grouping"],
                "config": config,
            },
            "semantic_graph": semantic["semantic_summary"],
            "layout_graph": semantic["layout_summary"],
            "counts": {
                "nodes": len(semantic["ids"]),
                "semantic_edges": len(semantic["semantic_edges"]),
                "layout_edges": len(semantic["layout_edges"]),
                "community_edges": len(community["community_edges"]),
                "macro_communities": int(community["macro_count"]),
                "core_macro_communities": int(community["core_macro_count"]),
                "satellite_macro_communities": len(community["satellite_groups"]),
                "satellite_components": sum(
                    int(group["componentCount"]) for group in community["satellite_groups"]
                ),
                "micro_communities": int(community["micro_count"]),
                "components": int(len(community["component_sizes"])),
                "labels": len(labels["communities"]) + len(labels["nodes"]),
                "isolates": int((flags & FLAG_ISOLATE != 0).sum()),
            },
            "community_layout": {
                "main_component_id": int(community["giant_component"]),
                "main_component_nodes": int(
                    community["component_sizes"][community["giant_component"]]
                ),
                "macro_centers": labels["communities"],
                "micro_communities_by_macro": community["micro_communities_by_macro"],
                "micro_partitioning": (
                    "each final macroCommunity receives an independently seeded Leiden partition; "
                    "isolates are singleton micro communities"
                ),
                "satellite_grouping": {
                    "algorithm": config["satellite_grouping"]["algorithm"],
                    "configuredGroupCount": int(config["satellite_grouping"]["group_count"]),
                    "activeGroupCount": len(community["satellite_groups"]),
                    "groups": community["satellite_groups"],
                },
                "satellite_policy": (
                    "actual non-isolate disconnected components are grouped by dominant "
                    "entity-type node mass around seed-stable irregular peripheral anchors; "
                    "isolates are an explicit outer layer"
                ),
            },
            "visual_encoding": {
                "color_basis": "macroCommunity (not entity type)",
                "size_basis": (
                    "quantile-clipped log1p(weightedDegree) with a small PageRank contribution"
                ),
                "size_below_2px_ratio": round(float((visual_sizes < 2.0).mean()), 6),
                "hub_overlay_degree_threshold": int(
                    config["layout_graph"]["hub_damping"]["overlay_degree_threshold"]
                ),
                "default_isolate_visibility": "off",
            },
            "assets": assets,
            "timing": {"build_seconds": round(time.perf_counter() - started, 3)},
            "provenance": {
                "semantic_graph_preserved": True,
                "raw_corpus_text_included": False,
                "relationship_descriptions_included": False,
                "layout_seed": int(config["seed"]),
            },
        }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
        return manifest
    except Exception:
        # Preserve failed staging output for inspection instead of deleting useful evidence.
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=Path, default=DEFAULT_BUNDLE / "entities.jsonl")
    parser.add_argument(
        "--relationships", type=Path, default=DEFAULT_BUNDLE / "relationships.jsonl"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_layout_assets(
            args.entities, args.relationships, args.output_dir, args.config
        )
    except (LayoutBuildError, OSError, sqlite3.Error) as exc:
        sys.stderr.write(f"error:material_graph_layout_build_failed:{exc}\n")
        return 2
    print(
        json.dumps(
            {"status": "completed", "manifest": manifest}, ensure_ascii=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
