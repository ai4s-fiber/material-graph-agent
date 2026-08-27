#!/usr/bin/env python3
"""Stream and audit the canonical textbook material knowledge graph.

This tool reads the semantic graph as-is. It never rewrites the JSONL inputs
and does not create a layout graph. Its JSON output is a repeatable factual
baseline for the separate community/layout precomputation step.

Canonical relationship rows do not carry a predicate/relation-type field.
``relation_taxonomy`` is therefore source entity_type -> target entity_type,
not a semantic predicate distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = SCRIPT_ROOT / "data" / "runtime" / "textbook-graph-bundle"
DEFAULT_ENTITIES_PATH = DEFAULT_BUNDLE_DIR / "entities.jsonl"
DEFAULT_RELATIONSHIPS_PATH = DEFAULT_BUNDLE_DIR / "relationships.jsonl"
AUDIT_SCHEMA = "material-graph.semantic-audit.v1"


class DisjointSet:
    """Compact union-find for connected components of the semantic graph."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects one line at a time with useful corruption context."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive CLI path
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):  # pragma: no cover - defensive CLI path
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile_sorted(values: list[int], percentile: float) -> float:
    """Linear-interpolated percentile with no optional numerical dependency."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _rounded(value: float, places: int = 6) -> float:
    return round(value, places)


def audit_graph(entities_path: Path, relationships_path: Path) -> dict[str, Any]:
    """Audit canonical JSONL inputs while preserving semantic graph meaning.

    Degree is endpoint incidence across valid raw directed relationships; a
    self-loop contributes two. Components use an undirected projection. The
    audit retains duplicates in degree statistics because semanticGraph itself
    is a raw directed, multirelational graph.
    """

    entities_path = Path(entities_path)
    relationships_path = Path(relationships_path)
    if not entities_path.is_file():
        raise FileNotFoundError(f"Entities JSONL was not found: {entities_path}")
    if not relationships_path.is_file():
        raise FileNotFoundError(f"Relationships JSONL was not found: {relationships_path}")

    entity_index: dict[str, int] = {}
    entity_metadata: list[tuple[str, str, str]] = []
    duplicate_entity_ids = 0
    for row in _iter_jsonl(entities_path):
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            raise ValueError(f"Entity without entity_id in {entities_path}")
        if entity_id in entity_index:
            duplicate_entity_ids += 1
            continue
        entity_index[entity_id] = len(entity_metadata)
        entity_metadata.append(
            (
                entity_id,
                str(row.get("name") or entity_id),
                str(row.get("entity_type") or "Unknown"),
            )
        )

    degrees = [0] * len(entity_metadata)
    components = DisjointSet(len(entity_metadata))
    seen_directed_edges: set[tuple[str, str]] = set()
    relation_taxonomy: Counter[str] = Counter()
    relation_rows = self_loops = duplicate_directed_edges = 0
    missing_endpoint_rows = missing_source_endpoints = missing_target_endpoints = 0

    for row in _iter_jsonl(relationships_path):
        relation_rows += 1
        source_id = str(row.get("source_entity_id", "")).strip()
        target_id = str(row.get("target_entity_id", "")).strip()
        edge_key = (source_id, target_id)
        if edge_key in seen_directed_edges:
            duplicate_directed_edges += 1
        else:
            seen_directed_edges.add(edge_key)

        source_index = entity_index.get(source_id)
        target_index = entity_index.get(target_id)
        if source_index is None or target_index is None:
            missing_endpoint_rows += 1
            missing_source_endpoints += int(source_index is None)
            missing_target_endpoints += int(target_index is None)
            source_type = (
                "MissingEntity" if source_index is None else entity_metadata[source_index][2]
            )
            target_type = (
                "MissingEntity" if target_index is None else entity_metadata[target_index][2]
            )
            relation_taxonomy[f"{source_type} -> {target_type}"] += 1
            continue

        source_type = entity_metadata[source_index][2]
        target_type = entity_metadata[target_index][2]
        relation_taxonomy[f"{source_type} -> {target_type}"] += 1
        degrees[source_index] += 1
        degrees[target_index] += 1
        if source_index == target_index:
            self_loops += 1
        else:
            components.union(source_index, target_index)

    component_sizes = Counter(components.find(index) for index in range(len(entity_metadata)))
    node_count = len(entity_metadata)
    sorted_degrees = sorted(degrees)
    isolated_nodes = sum(degree == 0 for degree in degrees)
    component_count = len(component_sizes)
    largest_component_size = max(component_sizes.values(), default=0)
    average_degree = sum(degrees) / node_count if node_count else 0.0

    top_hubs = sorted(
        (
            {
                "entity_id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "degree": degrees[index],
            }
            for index, (entity_id, name, entity_type) in enumerate(entity_metadata)
        ),
        key=lambda item: (-int(item["degree"]), str(item["name"]), str(item["entity_id"])),
    )[:20]
    taxonomy_rows = [
        {
            "source_to_target": taxonomy,
            "count": count,
            "frequency": _rounded(count / relation_rows if relation_rows else 0.0),
        }
        for taxonomy, count in sorted(
            relation_taxonomy.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    return {
        "schema": AUDIT_SCHEMA,
        "semantic_graph": {
            "directed": True,
            "multirelational": True,
            "inputs": {
                "entities": {"path": str(entities_path), "sha256": _sha256(entities_path)},
                "relationships": {
                    "path": str(relationships_path),
                    "sha256": _sha256(relationships_path),
                },
            },
            "nodes": node_count,
            "directed_edges": relation_rows,
            "duplicate_entity_ids": duplicate_entity_ids,
            "self_loops": self_loops,
            "duplicate_directed_edges": duplicate_directed_edges,
            "unique_directed_endpoint_pairs": len(seen_directed_edges),
            "missing_endpoint_rows": missing_endpoint_rows,
            "missing_source_endpoints": missing_source_endpoints,
            "missing_target_endpoints": missing_target_endpoints,
        },
        "connectivity": {
            "component_definition": "undirected projection of valid semantic relationships",
            "components": component_count,
            "largest_component_nodes": largest_component_size,
            "largest_component_ratio": _rounded(
                largest_component_size / node_count if node_count else 0.0
            ),
            "isolated_nodes": isolated_nodes,
            "isolated_node_ratio": _rounded(isolated_nodes / node_count if node_count else 0.0),
        },
        "degree": {
            "definition": "raw valid directed relationship endpoint incidence; self-loop contributes 2",
            "average": _rounded(average_degree),
            "median": _rounded(_percentile_sorted(sorted_degrees, 50)),
            "p90": _rounded(_percentile_sorted(sorted_degrees, 90)),
            "p99": _rounded(_percentile_sorted(sorted_degrees, 99)),
            "p99_9": _rounded(_percentile_sorted(sorted_degrees, 99.9)),
            "top_hubs": top_hubs,
        },
        "relation_taxonomy": {
            "canonical_predicate_available": False,
            "basis": "source entity_type -> target entity_type",
            "limitation": (
                "Canonical relationship JSONL has no predicate/relation-type field; "
                "this is an entity-type-pair taxonomy, not semantic predicate frequency."
            ),
            "rows": taxonomy_rows,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES_PATH)
    parser.add_argument("--relationships", type=Path, default=DEFAULT_RELATIONSHIPS_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write formatted JSON to this path. Omit to print JSON to standard output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_graph(args.entities, args.relationships)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
