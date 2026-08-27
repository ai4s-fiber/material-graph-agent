"""Focused tests for the read-only semantic graph audit CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_material_graph.py"
SPEC = importlib.util.spec_from_file_location("audit_material_graph", SCRIPT_PATH)
assert SPEC and SPEC.loader
audit_material_graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_material_graph)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_audit_graph_reports_semantic_and_connectivity_metrics(tmp_path: Path) -> None:
    entities = tmp_path / "entities.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    _write_jsonl(
        entities,
        [
            {"entity_id": "a", "name": "Alpha", "entity_type": "Material"},
            {"entity_id": "b", "name": "Beta", "entity_type": "Property"},
            {"entity_id": "c", "name": "Gamma", "entity_type": "Process"},
            {"entity_id": "d", "name": "Delta", "entity_type": "Material"},
        ],
    )
    _write_jsonl(
        relationships,
        [
            {"source_entity_id": "a", "target_entity_id": "b"},
            {"source_entity_id": "a", "target_entity_id": "b"},
            {"source_entity_id": "b", "target_entity_id": "a"},
            {"source_entity_id": "c", "target_entity_id": "c"},
            {"source_entity_id": "a", "target_entity_id": "missing"},
        ],
    )

    report = audit_material_graph.audit_graph(entities, relationships)

    assert report["schema"] == "material-graph.semantic-audit.v1"
    semantic = report["semantic_graph"]
    assert semantic["nodes"] == 4
    assert semantic["directed_edges"] == 5
    assert semantic["self_loops"] == 1
    assert semantic["duplicate_directed_edges"] == 1
    assert semantic["unique_directed_endpoint_pairs"] == 4
    assert semantic["missing_endpoint_rows"] == 1
    assert semantic["missing_target_endpoints"] == 1

    connectivity = report["connectivity"]
    assert connectivity["components"] == 3
    assert connectivity["largest_component_nodes"] == 2
    assert connectivity["largest_component_ratio"] == 0.5
    assert connectivity["isolated_nodes"] == 1
    assert connectivity["isolated_node_ratio"] == 0.25

    degree = report["degree"]
    assert degree["average"] == 2.0
    assert degree["median"] == 2.5
    assert degree["p90"] == 3.0
    assert degree["p99"] == 3.0
    assert degree["p99_9"] == 3.0
    assert degree["top_hubs"][0]["entity_id"] == "a"
    assert degree["top_hubs"][0]["degree"] == 3

    taxonomy = report["relation_taxonomy"]
    assert taxonomy["canonical_predicate_available"] is False
    assert taxonomy["basis"] == "source entity_type -> target entity_type"
    assert taxonomy["rows"][0] == {
        "source_to_target": "Material -> Property",
        "count": 2,
        "frequency": 0.4,
    }


def test_audit_graph_raises_for_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    relationships = tmp_path / "relationships.jsonl"
    _write_jsonl(relationships, [])

    try:
        audit_material_graph.audit_graph(missing, relationships)
    except FileNotFoundError as exc:
        assert "Entities JSONL was not found" in str(exc)
    else:  # pragma: no cover - explicit assertion keeps failure message readable
        raise AssertionError("Expected a missing entities file to be rejected")
