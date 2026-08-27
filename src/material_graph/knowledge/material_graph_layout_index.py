"""Read-only resolver from semantic graph IDs to compact layout coordinates.

The visualisation bundle deliberately does not expose a 221k-entry ID map to
the browser. This module is the narrow server-side boundary used to turn real
RAG entity IDs into the stable ``layoutIndex``/``macroId`` pair written by the
offline layout builder. Explorer calls read only the derived node-index SQLite
file and ``edges.u32`` endpoint pairs; they never open the canonical corpus,
graph JSONL, vectors, or binary node records.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any


MAX_LAYOUT_LOOKUPS = 128
MAX_LOOKUP_VALUE_LENGTH = 512
MAX_LAYOUT_SEARCH_QUERY_LENGTH = 128
MAX_LAYOUT_SEARCH_RESULTS = 50
MAX_LAYOUT_NEIGHBORHOOD_NODES = 128
MAX_LAYOUT_NEIGHBORHOOD_EDGES = 256
MAX_LAYOUT_HOPS = 2
_MAX_LAYOUT_CACHE_NODES = 1_000_000
_MAX_LAYOUT_CACHE_EDGES = 2_000_000
_MAX_LAYOUT_EDGE_BYTES = 128 * 1024 * 1024
_REQUIRED_COLUMNS = frozenset({"entity_id", "node_index", "name", "macro_id"})
_EXPLORER_REQUIRED_COLUMNS = _REQUIRED_COLUMNS | frozenset({"entity_type"})


class LayoutIndexUnavailableError(RuntimeError):
    """Raised when the verified, derived layout index cannot be read safely."""


def layout_asset_dir_from_environment() -> Path:
    """Return the derived layout asset directory without creating it.

    Operators may bind a read-only asset directory using
    ``MATERIAL_GRAPH_LAYOUT_ASSET_DIR``. The default matches the offline
    builder and deliberately remains within the repository runtime directory.
    """

    configured = os.getenv("MATERIAL_GRAPH_LAYOUT_ASSET_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "data" / "runtime" / "material-graph-layout"


def _clean_values(values: Sequence[str], *, field: str) -> list[str]:
    if len(values) > MAX_LAYOUT_LOOKUPS:
        raise ValueError(f"{field} exceeds {MAX_LAYOUT_LOOKUPS} values")
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field} must contain strings")
        if not value or value != value.strip() or "\x00" in value:
            raise ValueError(f"{field} contains an invalid exact-match value")
        if len(value) > MAX_LOOKUP_VALUE_LENGTH:
            raise ValueError(f"{field} values must be at most {MAX_LOOKUP_VALUE_LENGTH} characters")
        if value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned


def _connect_read_only(
    index_path: Path,
    *,
    required_columns: frozenset[str] = _REQUIRED_COLUMNS,
) -> sqlite3.Connection:
    if not index_path.is_file() or index_path.stat().st_size <= 0:
        raise LayoutIndexUnavailableError("material graph layout index is unavailable")
    try:
        connection = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(node_index)")
            if len(row) > 1
        }
        if not required_columns.issubset(columns):
            connection.close()
            raise LayoutIndexUnavailableError(
                "material graph layout index has an unsupported schema"
            )
        return connection
    except LayoutIndexUnavailableError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise LayoutIndexUnavailableError("material graph layout index is unavailable") from exc


@dataclass(frozen=True, slots=True)
class LayoutNodeMatch:
    """Only the non-corpus fields required by a WebGL highlight overlay."""

    entity_id: str
    layout_index: int
    macro_id: int

    def as_payload(self) -> dict[str, int | str]:
        return {
            "entityId": self.entity_id,
            "layoutIndex": self.layout_index,
            "macroId": self.macro_id,
        }


@dataclass(frozen=True, slots=True)
class LayoutExplorerNode:
    """A small, derived-only node descriptor for search and local inspection."""

    entity_id: str
    layout_index: int
    name: str
    entity_type: str
    macro_id: int

    def as_payload(self) -> dict[str, int | str]:
        return {
            "entityId": self.entity_id,
            "layoutIndex": self.layout_index,
            "name": self.name,
            "entityType": self.entity_type,
            "macroId": self.macro_id,
        }


@dataclass(frozen=True, slots=True)
class _LayoutEdgeIndex:
    """Compact in-memory adjacency over the read-only semantic endpoint asset."""

    node_count: int
    endpoints: array
    offsets: array
    incident_edge_ids: array

    @property
    def edge_count(self) -> int:
        return len(self.endpoints) // 2

    def incident_edges(self, layout_index: int) -> Sequence[int]:
        start = int(self.offsets[layout_index])
        end = int(self.offsets[layout_index + 1])
        return self.incident_edge_ids[start:end]


def resolve_layout_nodes(
    *,
    entity_ids: Sequence[str] = (),
    labels: Sequence[str] = (),
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve real semantic IDs (and uniquely matching exact labels) safely.

    A label is intentionally fail-closed when more than one semantic entity
    has that same exact name. RAG integrations should use canonical IDs;
    labels exist solely for backwards-compatible integrations that can display
    an ambiguity state instead of highlighting a wrong node.
    """

    cleaned_ids = _clean_values(entity_ids, field="entity_ids")
    cleaned_labels = _clean_values(labels, field="labels")
    if len(cleaned_ids) + len(cleaned_labels) > MAX_LAYOUT_LOOKUPS:
        raise ValueError(f"total lookups must not exceed {MAX_LAYOUT_LOOKUPS}")

    directory = asset_dir if asset_dir is not None else layout_asset_dir_from_environment()
    index_path = directory / "node-index.sqlite3"
    matches: list[LayoutNodeMatch] = []
    unresolved_ids: list[str] = []
    unresolved_labels: list[str] = []
    ambiguous_labels: dict[str, int] = {}

    with _connect_read_only(index_path) as connection:
        for entity_id in cleaned_ids:
            row = connection.execute(
                "SELECT entity_id, node_index, macro_id FROM node_index WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                unresolved_ids.append(entity_id)
                continue
            matches.append(LayoutNodeMatch(str(row[0]), int(row[1]), int(row[2])))

        for label in cleaned_labels:
            rows = connection.execute(
                "SELECT entity_id, node_index, macro_id FROM node_index WHERE name = ? "
                "ORDER BY entity_id LIMIT 2",
                (label,),
            ).fetchall()
            if not rows:
                unresolved_labels.append(label)
            elif len(rows) == 1:
                row = rows[0]
                matches.append(LayoutNodeMatch(str(row[0]), int(row[1]), int(row[2])))
            else:
                ambiguous_labels[label] = len(rows)

    # An ID and a unique label may refer to the same node. Return it once,
    # preserving the explicit semantic-ID query precedence.
    deduplicated: list[LayoutNodeMatch] = []
    seen_entity_ids: set[str] = set()
    for match in matches:
        if match.entity_id not in seen_entity_ids:
            deduplicated.append(match)
            seen_entity_ids.add(match.entity_id)
    return {
        "matches": [match.as_payload() for match in deduplicated],
        "unresolvedEntityIds": unresolved_ids,
        "unresolvedLabels": unresolved_labels,
        "ambiguousLabels": ambiguous_labels,
    }


def _clean_optional_text(value: str | None, *, field: str, maximum_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    if len(value) > maximum_length:
        raise ValueError(f"{field} must be at most {maximum_length} characters")
    return value


def _bounded_nonnegative_int(
    value: int | None,
    *,
    field: str,
    maximum: int,
    required: bool = False,
) -> int | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an integer from 0 to {maximum}")
    return value


def _file_identity(path: Path, *, label: str) -> tuple[str, int, int]:
    try:
        metadata = path.stat()
        if not path.is_file() or metadata.st_size <= 0:
            raise LayoutIndexUnavailableError(f"material graph {label} is unavailable")
        return (path.resolve().as_posix(), int(metadata.st_mtime_ns), int(metadata.st_size))
    except LayoutIndexUnavailableError:
        raise
    except OSError as exc:
        raise LayoutIndexUnavailableError(f"material graph {label} is unavailable") from exc


def _like_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _explorer_node_from_row(row: Sequence[object]) -> LayoutExplorerNode:
    return LayoutExplorerNode(
        entity_id=str(row[0]),
        layout_index=int(row[1]),
        name=str(row[2]),
        entity_type=str(row[3]),
        macro_id=int(row[4]),
    )


@lru_cache(maxsize=256)
def _cached_layout_search(
    index_path_text: str,
    index_mtime_ns: int,
    index_size: int,
    query: str | None,
    match_mode: str,
    entity_type: str | None,
    macro_id: int | None,
    layout_index: int | None,
    limit: int,
) -> tuple[tuple[LayoutExplorerNode, ...], bool]:
    """Cache a bounded, derived-only result under the index file identity."""

    del index_mtime_ns, index_size
    clauses: list[str] = []
    parameters: list[int | str] = []
    if query is not None:
        if match_mode == "exact":
            clauses.append("name = ?")
            parameters.append(query)
        else:
            clauses.append("name LIKE ? ESCAPE '\\' COLLATE NOCASE")
            parameters.append(f"%{_like_literal(query)}%")
    if entity_type is not None:
        clauses.append("entity_type = ?")
        parameters.append(entity_type)
    if macro_id is not None:
        clauses.append("macro_id = ?")
        parameters.append(macro_id)
    if layout_index is not None:
        clauses.append("node_index = ?")
        parameters.append(layout_index)
    if not clauses:  # pragma: no cover - public validation prevents broad scans
        raise ValueError("at least one graph layout search filter is required")
    parameters.append(limit + 1)
    sql = (
        "SELECT entity_id, node_index, name, entity_type, macro_id FROM node_index WHERE "
        + " AND ".join(clauses)
        + " ORDER BY name COLLATE NOCASE, entity_id LIMIT ?"
    )
    try:
        with _connect_read_only(
            Path(index_path_text), required_columns=_EXPLORER_REQUIRED_COLUMNS
        ) as connection:
            rows = connection.execute(sql, parameters).fetchall()
    except LayoutIndexUnavailableError:
        raise
    except sqlite3.Error as exc:
        raise LayoutIndexUnavailableError("material graph layout index is unavailable") from exc
    truncated = len(rows) > limit
    return tuple(_explorer_node_from_row(row) for row in rows[:limit]), truncated


def search_layout_nodes(
    *,
    query: str | None = None,
    match_mode: str = "substring",
    entity_type: str | None = None,
    macro_id: int | None = None,
    layout_index: int | None = None,
    limit: int = 20,
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """Search real derived node metadata without a broad or corpus-backed query.

    ``query`` matches the derived entity ``name`` either exactly or as an
    escaped substring. Entity type, macro ID, and layout index are exact filters.
    Every request needs at least one filter and returns at most 50 records.
    """

    cleaned_query = _clean_optional_text(
        query, field="query", maximum_length=MAX_LAYOUT_SEARCH_QUERY_LENGTH
    )
    cleaned_type = _clean_optional_text(entity_type, field="entity_type", maximum_length=128)
    if match_mode not in {"exact", "substring"}:
        raise ValueError("match_mode must be exact or substring")
    cleaned_macro_id = _bounded_nonnegative_int(
        macro_id, field="macro_id", maximum=_MAX_LAYOUT_CACHE_NODES
    )
    cleaned_layout_index = _bounded_nonnegative_int(
        layout_index, field="layout_index", maximum=_MAX_LAYOUT_CACHE_NODES
    )
    cleaned_limit = _bounded_nonnegative_int(
        limit, field="limit", maximum=MAX_LAYOUT_SEARCH_RESULTS, required=True
    )
    if cleaned_limit is None or cleaned_limit < 1:
        raise ValueError(f"limit must be an integer from 1 to {MAX_LAYOUT_SEARCH_RESULTS}")
    if all(
        value is None
        for value in (cleaned_query, cleaned_type, cleaned_macro_id, cleaned_layout_index)
    ):
        raise ValueError("at least one graph layout search filter is required")

    directory = asset_dir if asset_dir is not None else layout_asset_dir_from_environment()
    identity = _file_identity(directory / "node-index.sqlite3", label="layout index")
    rows, truncated = _cached_layout_search(
        *identity,
        cleaned_query,
        match_mode,
        cleaned_type,
        cleaned_macro_id,
        cleaned_layout_index,
        cleaned_limit,
    )
    return {"results": [row.as_payload() for row in rows], "truncated": truncated}


def _validated_node_count(index_path: Path) -> int:
    try:
        with _connect_read_only(
            index_path, required_columns=_EXPLORER_REQUIRED_COLUMNS
        ) as connection:
            row = connection.execute(
                "SELECT COUNT(*), MIN(node_index), MAX(node_index), COUNT(DISTINCT node_index) "
                "FROM node_index"
            ).fetchone()
    except LayoutIndexUnavailableError:
        raise
    except sqlite3.Error as exc:
        raise LayoutIndexUnavailableError("material graph layout index is unavailable") from exc
    if row is None or any(value is None for value in row):
        raise LayoutIndexUnavailableError("material graph layout index is unavailable")
    count, minimum, maximum, distinct_count = (int(value) for value in row)
    if (
        count <= 0
        or count > _MAX_LAYOUT_CACHE_NODES
        or distinct_count != count
        or minimum != 0
        or maximum != count - 1
    ):
        raise LayoutIndexUnavailableError("material graph layout index is unsupported")
    return count


@lru_cache(maxsize=2)
def _cached_edge_index(
    index_path_text: str,
    index_mtime_ns: int,
    index_size: int,
    edges_path_text: str,
    edges_mtime_ns: int,
    edges_size: int,
) -> _LayoutEdgeIndex:
    """Build a compact incident-edge index once per immutable asset version."""

    del index_mtime_ns, index_size, edges_mtime_ns
    if edges_size % 8 or edges_size > _MAX_LAYOUT_EDGE_BYTES:
        raise LayoutIndexUnavailableError("material graph layout edge asset is unavailable")
    edge_count = edges_size // 8
    if edge_count <= 0 or edge_count > _MAX_LAYOUT_CACHE_EDGES:
        raise LayoutIndexUnavailableError("material graph layout edge asset is unavailable")

    node_count = _validated_node_count(Path(index_path_text))
    endpoints = array("I")
    try:
        with Path(edges_path_text).open("rb") as handle:
            endpoints.fromfile(handle, edge_count * 2)
    except (EOFError, OSError, ValueError) as exc:
        raise LayoutIndexUnavailableError(
            "material graph layout edge asset is unavailable"
        ) from exc
    if endpoints.itemsize != 4 or len(endpoints) != edge_count * 2:
        raise LayoutIndexUnavailableError("material graph layout edge asset is unavailable")
    if sys.byteorder != "little":  # pragma: no cover - production targets are little-endian
        endpoints.byteswap()

    degrees = array("I", [0]) * node_count
    for edge_id in range(edge_count):
        source = int(endpoints[2 * edge_id])
        target = int(endpoints[2 * edge_id + 1])
        if source >= node_count or target >= node_count:
            raise LayoutIndexUnavailableError("material graph layout edge asset is unsupported")
        degrees[source] += 1
        degrees[target] += 1

    offsets = array("Q", [0]) * (node_count + 1)
    running_total = 0
    for layout_index_value in range(node_count):
        offsets[layout_index_value] = running_total
        running_total += int(degrees[layout_index_value])
    offsets[node_count] = running_total
    if running_total != len(endpoints):  # pragma: no cover - defensive accounting gate
        raise LayoutIndexUnavailableError("material graph layout edge asset is unsupported")

    incident_edge_ids = array("I", [0]) * running_total
    cursors = array("Q", offsets[:node_count])
    for edge_id in range(edge_count):
        source = int(endpoints[2 * edge_id])
        target = int(endpoints[2 * edge_id + 1])
        source_offset = int(cursors[source])
        incident_edge_ids[source_offset] = edge_id
        if source == target:
            incident_edge_ids[source_offset + 1] = edge_id
            cursors[source] = source_offset + 2
            continue
        target_offset = int(cursors[target])
        incident_edge_ids[target_offset] = edge_id
        cursors[source] = source_offset + 1
        cursors[target] = target_offset + 1
    return _LayoutEdgeIndex(
        node_count=node_count,
        endpoints=endpoints,
        offsets=offsets,
        incident_edge_ids=incident_edge_ids,
    )


def _edge_index_for_asset(directory: Path) -> _LayoutEdgeIndex:
    index_identity = _file_identity(directory / "node-index.sqlite3", label="layout index")
    edges_identity = _file_identity(directory / "edges.u32", label="layout edge asset")
    return _cached_edge_index(*index_identity, *edges_identity)


def _lookup_explorer_node(
    connection: sqlite3.Connection,
    *,
    entity_id: str | None,
    layout_index: int | None,
) -> LayoutExplorerNode | None:
    if entity_id is not None:
        row = connection.execute(
            "SELECT entity_id, node_index, name, entity_type, macro_id FROM node_index "
            "WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT entity_id, node_index, name, entity_type, macro_id FROM node_index "
            "WHERE node_index = ?",
            (layout_index,),
        ).fetchone()
    return None if row is None else _explorer_node_from_row(row)


def _lookup_explorer_nodes(
    connection: sqlite3.Connection,
    layout_indices: Sequence[int],
) -> list[LayoutExplorerNode]:
    unique_indices = sorted(set(layout_indices))
    if not unique_indices:
        return []
    placeholders = ",".join("?" for _ in unique_indices)
    rows = connection.execute(
        "SELECT entity_id, node_index, name, entity_type, macro_id FROM node_index "
        f"WHERE node_index IN ({placeholders}) ORDER BY node_index",
        unique_indices,
    ).fetchall()
    if len(rows) != len(unique_indices):
        raise LayoutIndexUnavailableError("material graph layout index is unsupported")
    return [_explorer_node_from_row(row) for row in rows]


def _collect_bounded_neighborhood(
    edge_index: _LayoutEdgeIndex,
    *,
    center_index: int,
    hops: int,
    max_nodes: int,
    max_edges: int,
) -> tuple[list[int], list[int], bool, bool]:
    selected: set[int] = {center_index}
    selected_order = [center_index]
    frontier = [center_index]
    nodes_truncated = False
    for _depth in range(hops):
        next_frontier: list[int] = []
        for current in frontier:
            for offset in range(
                int(edge_index.offsets[current]), int(edge_index.offsets[current + 1])
            ):
                edge_id = int(edge_index.incident_edge_ids[offset])
                source = int(edge_index.endpoints[2 * edge_id])
                target = int(edge_index.endpoints[2 * edge_id + 1])
                neighbor = target if source == current else source
                if neighbor in selected:
                    continue
                if len(selected) >= max_nodes:
                    nodes_truncated = True
                    break
                selected.add(neighbor)
                selected_order.append(neighbor)
                next_frontier.append(neighbor)
            if nodes_truncated:
                break
        if nodes_truncated:
            break
        frontier = next_frontier
        if not frontier:
            break

    selected_set = set(selected_order)
    selected_edges: list[int] = []
    seen_edge_ids: set[int] = set()
    edges_truncated = False
    for current in sorted(selected_set):
        for offset in range(int(edge_index.offsets[current]), int(edge_index.offsets[current + 1])):
            edge_id = int(edge_index.incident_edge_ids[offset])
            if edge_id in seen_edge_ids:
                continue
            source = int(edge_index.endpoints[2 * edge_id])
            target = int(edge_index.endpoints[2 * edge_id + 1])
            if source not in selected_set or target not in selected_set:
                continue
            if len(selected_edges) >= max_edges:
                edges_truncated = True
                break
            seen_edge_ids.add(edge_id)
            selected_edges.append(edge_id)
        if edges_truncated:
            break
    return selected_order, selected_edges, nodes_truncated, edges_truncated


def get_layout_neighborhood(
    *,
    entity_id: str | None = None,
    layout_index: int | None = None,
    hops: int = 1,
    max_nodes: int = 64,
    max_edges: int = 128,
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """Return a capped, real one- or two-hop semantic endpoint neighborhood."""

    cleaned_entity_id = _clean_optional_text(
        entity_id, field="entity_id", maximum_length=MAX_LOOKUP_VALUE_LENGTH
    )
    cleaned_layout_index = _bounded_nonnegative_int(
        layout_index, field="layout_index", maximum=_MAX_LAYOUT_CACHE_NODES
    )
    if (cleaned_entity_id is None) == (cleaned_layout_index is None):
        raise ValueError("provide exactly one of entity_id or layout_index")
    cleaned_hops = _bounded_nonnegative_int(
        hops, field="hops", maximum=MAX_LAYOUT_HOPS, required=True
    )
    if cleaned_hops not in {1, 2}:
        raise ValueError("hops must be 1 or 2")
    cleaned_max_nodes = _bounded_nonnegative_int(
        max_nodes, field="max_nodes", maximum=MAX_LAYOUT_NEIGHBORHOOD_NODES, required=True
    )
    cleaned_max_edges = _bounded_nonnegative_int(
        max_edges, field="max_edges", maximum=MAX_LAYOUT_NEIGHBORHOOD_EDGES, required=True
    )
    if cleaned_max_nodes is None or cleaned_max_nodes < 1:
        raise ValueError(f"max_nodes must be an integer from 1 to {MAX_LAYOUT_NEIGHBORHOOD_NODES}")
    if cleaned_max_edges is None or cleaned_max_edges < 1:
        raise ValueError(f"max_edges must be an integer from 1 to {MAX_LAYOUT_NEIGHBORHOOD_EDGES}")

    directory = asset_dir if asset_dir is not None else layout_asset_dir_from_environment()
    edge_index = _edge_index_for_asset(directory)
    try:
        with _connect_read_only(
            directory / "node-index.sqlite3", required_columns=_EXPLORER_REQUIRED_COLUMNS
        ) as connection:
            center = _lookup_explorer_node(
                connection,
                entity_id=cleaned_entity_id,
                layout_index=cleaned_layout_index,
            )
            if center is None:
                raise LookupError("material graph layout node was not found")
            if center.layout_index >= edge_index.node_count:
                raise LayoutIndexUnavailableError("material graph layout index is unsupported")
            selected_indices, selected_edge_ids, nodes_truncated, edges_truncated = (
                _collect_bounded_neighborhood(
                    edge_index,
                    center_index=center.layout_index,
                    hops=cleaned_hops,
                    max_nodes=cleaned_max_nodes,
                    max_edges=cleaned_max_edges,
                )
            )
            nodes = _lookup_explorer_nodes(connection, selected_indices)
    except (LayoutIndexUnavailableError, LookupError):
        raise
    except sqlite3.Error as exc:
        raise LayoutIndexUnavailableError("material graph layout index is unavailable") from exc

    return {
        "center": center.as_payload(),
        "hops": cleaned_hops,
        "nodes": [node.as_payload() for node in nodes],
        "edges": [
            {
                "sourceLayoutIndex": int(edge_index.endpoints[2 * edge_id]),
                "targetLayoutIndex": int(edge_index.endpoints[2 * edge_id + 1]),
            }
            for edge_id in selected_edge_ids
        ],
        "nodesTruncated": nodes_truncated,
        "edgesTruncated": edges_truncated,
    }
