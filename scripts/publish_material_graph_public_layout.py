"""Publish the bounded public WebGL projection of the canonical Material Graph layout.

The source layout bundle remains private and may contain node IDs, a SQLite index,
raw semantic edge endpoints, and other server-only derived files.  This script
only projects the three browser-safe overview assets plus a new public manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .verify_material_graph_layout_assets import (
        LayoutAssetVerificationError,
        verify_layout_assets,
    )
else:  # pragma: no cover - exercised by the direct CLI entrypoint
    from verify_material_graph_layout_assets import (
        LayoutAssetVerificationError,
        verify_layout_assets,
    )

CANONICAL_LAYOUT_VERSION = "textbook-graph-layout-v1-satellite-groups-v1"
PRIVATE_LAYOUT_SCHEMA = "material-graph.layout-assets.v1"
PUBLIC_LAYOUT_SCHEMA = "material-graph.public-layout.v1"
LABEL_SCHEMA = "material-graph.layout-labels.v1"
NODE_STRIDE = 8
COMMUNITY_EDGE_STRIDE = 5
NODE_RECORD_BYTES = NODE_STRIDE * 4
COMMUNITY_EDGE_RECORD_BYTES = COMMUNITY_EDGE_STRIDE * 4
PUBLIC_ASSETS = {
    "nodes": "nodes.f32",
    "community_edges": "community-edges.f32",
    "labels": "labels.json",
}
PUBLIC_FILENAMES = frozenset({"manifest.json", *PUBLIC_ASSETS.values()})
NODE_FIELDS = ["x", "y", "size", "r", "g", "b", "macro_id", "flags"]
COMMUNITY_EDGE_FIELDS = ["source_macro", "target_macro", "weight", "target_x", "target_y"]
COUNT_KEYS = (
    "nodes",
    "semantic_edges",
    "macro_communities",
    "core_macro_communities",
    "micro_communities",
    "components",
    "isolates",
    "satellite_components",
    "satellite_macro_communities",
    "community_edges",
    "labels",
)
REQUIRED_COUNT_KEYS = {
    "nodes",
    "semantic_edges",
    "macro_communities",
    "community_edges",
    "labels",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT / "data" / "runtime" / "material-graph-layout" / CANONICAL_LAYOUT_VERSION
)
DEFAULT_TARGET_DIR = (
    REPO_ROOT.parent
    / "Omnnimat2.0"
    / "product"
    / "app"
    / "public"
    / "material-kg"
    / "constellation"
)


class PublicLayoutPublishError(ValueError):
    """Raised when a source or public projection violates the release boundary."""


def _fail(message: str) -> None:
    raise PublicLayoutPublishError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be a finite number")
    return result


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a non-empty string")
    return value.strip()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label}: {exc}")
    return _mapping(payload, label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_asset(
    source_dir: Path,
    assets: Mapping[str, Any],
    key: str,
    expected_filename: str,
    *,
    record_bytes: int | None,
) -> tuple[Path, dict[str, Any]]:
    descriptor = _mapping(assets.get(key), f"assets.{key}")
    if descriptor.get("path") != expected_filename:
        _fail(f"assets.{key}.path must be exactly {expected_filename!r}")
    path = (source_dir / expected_filename).resolve()
    if path.parent != source_dir or not path.is_file():
        _fail(f"required source asset is missing: {expected_filename}")
    byte_count = _integer(descriptor.get("bytes"), f"assets.{key}.bytes", minimum=1)
    records = _integer(descriptor.get("records"), f"assets.{key}.records", minimum=0)
    digest = _non_empty_string(descriptor.get("sha256"), f"assets.{key}.sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(f"assets.{key}.sha256 must be a lowercase SHA-256 digest")
    if path.stat().st_size != byte_count:
        _fail(f"assets.{key}.bytes does not match {expected_filename}")
    if _sha256(path) != digest:
        _fail(f"assets.{key}.sha256 does not match {expected_filename}")
    if record_bytes is not None:
        if byte_count % record_bytes:
            _fail(f"{expected_filename} has an invalid record boundary")
        if byte_count // record_bytes != records:
            _fail(f"assets.{key}.records does not match {expected_filename}")
    return path, {
        "path": expected_filename,
        "records": records,
        "bytes": byte_count,
        "sha256": digest,
    }


def _validate_labels(path: Path, records: int) -> None:
    labels = _read_json(path, "labels.json")
    if set(labels) != {"schema", "communities", "nodes"}:
        _fail("labels.json must contain only the public label schema fields")
    if labels.get("schema") != LABEL_SCHEMA:
        _fail(f"labels.json schema must be {LABEL_SCHEMA!r}")
    communities = labels.get("communities")
    nodes = labels.get("nodes")
    if not isinstance(communities, list) or not isinstance(nodes, list):
        _fail("labels.json communities and nodes must be arrays")
    if len(communities) + len(nodes) != records:
        _fail("assets.labels.records does not match labels.json")
    for community in communities:
        row = _mapping(community, "labels.json community")
        if set(row) != {"macroId", "label", "x", "y", "nodeCount"}:
            _fail("labels.json community contains a non-public field")
        _integer(row.get("macroId"), "labels.json community macroId")
        _non_empty_string(row.get("label"), "labels.json community label")
        _finite_number(row.get("x"), "labels.json community x")
        _finite_number(row.get("y"), "labels.json community y")
        _integer(row.get("nodeCount"), "labels.json community nodeCount", minimum=1)
    for node in nodes:
        row = _mapping(node, "labels.json node")
        if set(row) != {"index", "label", "labelRank"}:
            _fail("labels.json node contains a non-public field")
        _integer(row.get("index"), "labels.json node index")
        _non_empty_string(row.get("label"), "labels.json node label")
        _finite_number(row.get("labelRank"), "labels.json node labelRank")


def build_public_manifest(source_dir: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate the canonical layout and return its public manifest plus source files."""
    source = Path(source_dir).resolve()
    if not source.is_dir():
        _fail(f"canonical asset directory does not exist: {source}")
    try:
        verify_layout_assets(source)
    except LayoutAssetVerificationError as exc:
        _fail(f"canonical layout verification failed: {exc}")
    manifest = _read_json(source / "manifest.json", "canonical manifest.json")
    if manifest.get("schema") != PRIVATE_LAYOUT_SCHEMA:
        _fail(f"canonical manifest schema must be {PRIVATE_LAYOUT_SCHEMA!r}")
    version = _non_empty_string(manifest.get("layout_version"), "layout_version")
    if version != CANONICAL_LAYOUT_VERSION:
        _fail(
            f"canonical layout version must be {CANONICAL_LAYOUT_VERSION!r}, received {version!r}"
        )
    generated_at = _integer(
        manifest.get("generated_at_epoch_ms"), "generated_at_epoch_ms", minimum=1
    )
    assets = _mapping(manifest.get("assets"), "assets")
    counts = _mapping(manifest.get("counts"), "counts")
    parameters = _mapping(manifest.get("parameters"), "parameters")
    provenance = _mapping(manifest.get("provenance"), "provenance")
    lineage = _mapping(manifest.get("lineage"), "lineage")

    nodes_path, nodes = _source_asset(
        source, assets, "nodes", PUBLIC_ASSETS["nodes"], record_bytes=NODE_RECORD_BYTES
    )
    community_path, community_edges = _source_asset(
        source,
        assets,
        "community_edges",
        PUBLIC_ASSETS["community_edges"],
        record_bytes=COMMUNITY_EDGE_RECORD_BYTES,
    )
    labels_path, labels = _source_asset(
        source, assets, "labels", PUBLIC_ASSETS["labels"], record_bytes=None
    )
    _validate_labels(labels_path, labels["records"])

    public_counts: dict[str, int] = {}
    for key in COUNT_KEYS:
        if key in counts:
            public_counts[key] = _integer(counts[key], f"counts.{key}")
    missing_counts = REQUIRED_COUNT_KEYS.difference(public_counts)
    if missing_counts:
        _fail(f"canonical manifest is missing required counts: {sorted(missing_counts)}")
    if public_counts["nodes"] != nodes["records"]:
        _fail("counts.nodes does not match nodes.f32")
    if public_counts["community_edges"] != community_edges["records"]:
        _fail("counts.community_edges does not match community-edges.f32")
    if public_counts["labels"] != labels["records"]:
        _fail("counts.labels does not match labels.json")

    macro_communities = public_counts["macro_communities"]
    layout = {
        "algorithm": _non_empty_string(
            parameters.get("layout_algorithm"), "parameters.layout_algorithm"
        ),
        "macro_resolution": _finite_number(
            parameters.get("macro_resolution"), "parameters.macro_resolution"
        ),
        "micro_resolution": _finite_number(
            parameters.get("micro_resolution"), "parameters.micro_resolution"
        ),
        "random_seed": _integer(parameters.get("random_seed"), "parameters.random_seed"),
    }
    public_provenance = {
        "source_layout_version": version,
        "lineage": {
            "parent_layout_version": _non_empty_string(
                lineage.get("parent_layout_version"), "lineage.parent_layout_version"
            ),
            "revision": _non_empty_string(lineage.get("revision"), "lineage.revision"),
        },
        "semantic_graph_preserved": provenance.get("semantic_graph_preserved") is True,
        "raw_corpus_text_included": False,
        "relationship_descriptions_included": False,
        "public_assets": list(PUBLIC_ASSETS.values()),
    }
    if not public_provenance["semantic_graph_preserved"]:
        _fail("canonical provenance must preserve the semantic graph")

    public_manifest = {
        "schema": PUBLIC_LAYOUT_SCHEMA,
        "version": version,
        "generatedAtEpochMs": generated_at,
        "nodeCount": nodes["records"],
        "nodeStride": NODE_STRIDE,
        "macroCommunityCount": macro_communities,
        "communityEdgeStride": COMMUNITY_EDGE_STRIDE,
        "assets": {
            "nodes": {**nodes, "fields": NODE_FIELDS},
            "community_edges": {**community_edges, "fields": COMMUNITY_EDGE_FIELDS},
            "labels": labels,
        },
        "counts": public_counts,
        "layout": layout,
        "provenance": public_provenance,
    }
    return public_manifest, {
        "nodes.f32": nodes_path,
        "community-edges.f32": community_path,
        "labels.json": labels_path,
    }


def _validate_target_boundary(target_dir: Path) -> None:
    if target_dir.exists() and not target_dir.is_dir():
        _fail(f"public target is not a directory: {target_dir}")
    if not target_dir.exists():
        return
    unexpected = sorted(
        entry.name for entry in target_dir.iterdir() if entry.name not in PUBLIC_FILENAMES
    )
    if unexpected:
        _fail(f"public target contains forbidden files: {unexpected}")


def _public_asset(
    directory: Path,
    assets: Mapping[str, Any],
    key: str,
    *,
    record_bytes: int | None,
    fields: list[str] | None,
) -> dict[str, Any]:
    expected_filename = PUBLIC_ASSETS[key]
    descriptor = _mapping(assets.get(key), f"public assets.{key}")
    expected_keys = {"path", "records", "bytes", "sha256"}
    if fields is not None:
        expected_keys.add("fields")
    if set(descriptor) != expected_keys:
        _fail(f"public assets.{key} has unsupported fields")
    if descriptor.get("path") != expected_filename:
        _fail(f"public assets.{key}.path must be {expected_filename!r}")
    if fields is not None and descriptor.get("fields") != fields:
        _fail(f"public assets.{key}.fields is not the approved public layout schema")
    path = directory / expected_filename
    if not path.is_file():
        _fail(f"public asset is missing: {expected_filename}")
    byte_count = _integer(descriptor.get("bytes"), f"public assets.{key}.bytes", minimum=1)
    records = _integer(descriptor.get("records"), f"public assets.{key}.records", minimum=0)
    digest = _non_empty_string(descriptor.get("sha256"), f"public assets.{key}.sha256").lower()
    if path.stat().st_size != byte_count or _sha256(path) != digest:
        _fail(f"public assets.{key} does not match its bytes or SHA-256")
    if record_bytes is not None and (
        byte_count % record_bytes or byte_count // record_bytes != records
    ):
        _fail(f"public assets.{key} does not have the declared record count")
    return {"records": records, "bytes": byte_count, "sha256": digest}


def verify_public_layout(target_dir: str | Path) -> dict[str, Any]:
    """Fail closed unless a target contains only the exact four public files."""
    target = Path(target_dir).resolve()
    if not target.is_dir():
        _fail(f"public target directory does not exist: {target}")
    _validate_target_boundary(target)
    manifest = _read_json(target / "manifest.json", "public manifest.json")
    expected_manifest_keys = {
        "schema",
        "version",
        "generatedAtEpochMs",
        "nodeCount",
        "nodeStride",
        "macroCommunityCount",
        "communityEdgeStride",
        "assets",
        "counts",
        "layout",
        "provenance",
    }
    if set(manifest) != expected_manifest_keys:
        _fail("public manifest contains unsupported fields")
    if manifest.get("schema") != PUBLIC_LAYOUT_SCHEMA:
        _fail(f"public manifest schema must be {PUBLIC_LAYOUT_SCHEMA!r}")
    if _non_empty_string(manifest.get("version"), "public version") != CANONICAL_LAYOUT_VERSION:
        _fail("public manifest does not reference the canonical satellite-groups layout")
    _integer(manifest.get("generatedAtEpochMs"), "public generatedAtEpochMs", minimum=1)
    if _integer(manifest.get("nodeStride"), "public nodeStride", minimum=1) != NODE_STRIDE:
        _fail(f"public nodeStride must be {NODE_STRIDE}")
    if (
        _integer(manifest.get("communityEdgeStride"), "public communityEdgeStride", minimum=1)
        != COMMUNITY_EDGE_STRIDE
    ):
        _fail(f"public communityEdgeStride must be {COMMUNITY_EDGE_STRIDE}")
    assets = _mapping(manifest.get("assets"), "public assets")
    if set(assets) != set(PUBLIC_ASSETS):
        _fail("public manifest may expose only nodes, community_edges, and labels")
    nodes = _public_asset(
        target, assets, "nodes", record_bytes=NODE_RECORD_BYTES, fields=NODE_FIELDS
    )
    community_edges = _public_asset(
        target,
        assets,
        "community_edges",
        record_bytes=COMMUNITY_EDGE_RECORD_BYTES,
        fields=COMMUNITY_EDGE_FIELDS,
    )
    labels = _public_asset(target, assets, "labels", record_bytes=None, fields=None)
    _validate_labels(target / "labels.json", labels["records"])
    if _integer(manifest.get("nodeCount"), "public nodeCount", minimum=1) != nodes["records"]:
        _fail("public nodeCount does not match nodes.f32")
    counts = _mapping(manifest.get("counts"), "public counts")
    if _integer(counts.get("nodes"), "public counts.nodes", minimum=1) != nodes["records"]:
        _fail("public counts.nodes does not match nodes.f32")
    if (
        _integer(counts.get("community_edges"), "public counts.community_edges")
        != community_edges["records"]
    ):
        _fail("public counts.community_edges does not match community-edges.f32")
    if _integer(counts.get("labels"), "public counts.labels") != labels["records"]:
        _fail("public counts.labels does not match labels.json")
    if _integer(
        manifest.get("macroCommunityCount"), "public macroCommunityCount", minimum=1
    ) != _integer(counts.get("macro_communities"), "public counts.macro_communities", minimum=1):
        _fail("public macroCommunityCount does not match public counts")
    layout = _mapping(manifest.get("layout"), "public layout")
    if set(layout) != {"algorithm", "macro_resolution", "micro_resolution", "random_seed"}:
        _fail("public layout contains unsupported fields")
    _non_empty_string(layout.get("algorithm"), "public layout.algorithm")
    _finite_number(layout.get("macro_resolution"), "public layout.macro_resolution")
    _finite_number(layout.get("micro_resolution"), "public layout.micro_resolution")
    _integer(layout.get("random_seed"), "public layout.random_seed")
    provenance = _mapping(manifest.get("provenance"), "public provenance")
    if set(provenance) != {
        "source_layout_version",
        "lineage",
        "semantic_graph_preserved",
        "raw_corpus_text_included",
        "relationship_descriptions_included",
        "public_assets",
    }:
        _fail("public provenance contains unsupported fields")
    if provenance.get("source_layout_version") != CANONICAL_LAYOUT_VERSION:
        _fail("public provenance source_layout_version is incorrect")
    if provenance.get("semantic_graph_preserved") is not True:
        _fail("public provenance must preserve the semantic graph")
    if provenance.get("raw_corpus_text_included") is not False:
        _fail("public manifest must not include raw corpus text")
    if provenance.get("relationship_descriptions_included") is not False:
        _fail("public manifest must not include relationship descriptions")
    if provenance.get("public_assets") != list(PUBLIC_ASSETS.values()):
        _fail("public provenance must list only the three permitted public assets")
    lineage = _mapping(provenance.get("lineage"), "public provenance.lineage")
    if set(lineage) != {"parent_layout_version", "revision"}:
        _fail("public provenance lineage contains unsupported fields")
    _non_empty_string(lineage.get("parent_layout_version"), "public lineage.parent_layout_version")
    _non_empty_string(lineage.get("revision"), "public lineage.revision")
    return {
        "schema": PUBLIC_LAYOUT_SCHEMA,
        "version": CANONICAL_LAYOUT_VERSION,
        "nodes": nodes["records"],
        "community_edges": community_edges["records"],
        "labels": labels["records"],
        "target": str(target),
    }


def publish_public_layout(source_dir: str | Path, target_dir: str | Path) -> dict[str, Any]:
    """Safely replace only the four explicitly allowed OmniMat public layout files."""
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    if source == target or source in target.parents or target in source.parents:
        _fail("source and public target directories must be disjoint")
    public_manifest, source_files = build_public_manifest(source)
    _validate_target_boundary(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=True)
    _validate_target_boundary(target)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.publish-", dir=target.parent
    ) as temporary:
        staging = Path(temporary)
        for filename, source_file in source_files.items():
            shutil.copyfile(source_file, staging / filename)
        (staging / "manifest.json").write_text(
            json.dumps(public_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_public_layout(staging)
        for filename in ("nodes.f32", "community-edges.f32", "labels.json", "manifest.json"):
            os.replace(staging / filename, target / filename)
    return verify_public_layout(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    arguments = parser.parse_args()
    try:
        result = publish_public_layout(arguments.source_dir, arguments.target_dir)
    except PublicLayoutPublishError as exc:
        parser.exit(1, f"public layout publish failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
