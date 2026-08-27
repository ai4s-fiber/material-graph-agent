"""Deterministic canonical graph and LightRAG custom-KG bundle generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import unicodedata
from typing import Any, TextIO

from .textbook_lightrag import iter_textbook_document_batches
from .lightrag_models import build_lightrag_basename


_TYPE_PRIORITY = (
    "Material",
    "Structure",
    "Process",
    "ProcessCondition",
    "Property",
    "TestMethod",
    "Mechanism",
    "Equipment",
    "Application",
    "Standard",
    "Organization",
    "Data",
    "Concept",
    "Other",
)
_TYPE_RANK = {value: index for index, value in enumerate(_TYPE_PRIORITY)}


class TextbookGraphBundleError(RuntimeError):
    """Stable bundle error that does not expose source text."""


@dataclass(frozen=True, slots=True)
class TextbookGraphBundleSettings:
    fragments_path: Path
    extractions_path: Path
    output_dir: Path
    require_complete: bool = True
    max_descriptions: int = 3
    max_sources: int = 32

    def __post_init__(self) -> None:
        if not self.fragments_path.is_file() or not self.extractions_path.is_file():
            raise ValueError("bundle input is unavailable")
        if self.max_descriptions <= 0 or self.max_sources <= 0:
            raise ValueError("bundle limits must be positive")

    @property
    def entities_path(self) -> Path:
        return self.output_dir / "entities.jsonl"

    @property
    def relationships_path(self) -> Path:
        return self.output_dir / "relationships.jsonl"

    @property
    def custom_kg_path(self) -> Path:
        return self.output_dir / "lightrag-custom-kg.json"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"


@dataclass(frozen=True, slots=True)
class TextbookGraphBundleSummary:
    fragment_count: int
    extracted_fragment_count: int
    missing_extraction_count: int
    entity_count: int
    relationship_count: int
    entity_mentions: int
    relationship_mentions: int
    filtered_entity_mentions: int
    filtered_relationship_mentions: int
    elapsed_seconds: float
    status: str


class _EntityAccumulator:
    __slots__ = (
        "citation_paths",
        "descriptions",
        "display_names",
        "mentions",
        "source_ids",
        "types",
    )

    def __init__(self) -> None:
        self.display_names: Counter[str] = Counter()
        self.types: Counter[str] = Counter()
        self.descriptions: Counter[str] = Counter()
        self.source_ids: list[str] = []
        self.citation_paths: list[str] = []
        self.mentions = 0


class _RelationshipAccumulator:
    __slots__ = ("citation_paths", "descriptions", "mentions", "source_ids")

    def __init__(self) -> None:
        self.descriptions: Counter[str] = Counter()
        self.source_ids: list[str] = []
        self.citation_paths: list[str] = []
        self.mentions = 0


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_entity_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = " ".join(normalized.replace("\x00", " ").split())
    normalized = normalized.strip(" \t\r\n\"'“”‘’`")
    return normalized.casefold()


def _evidence_compact(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _fragment_evidence_index(path: Path) -> dict[str, tuple[str, str]]:
    evidence: dict[str, tuple[str, str]] = {}
    for batch in iter_textbook_document_batches(path, batch_size=512):
        for document in batch:
            evidence[str(document.fragment.fragment_id)] = (
                _evidence_compact(document.text),
                build_lightrag_basename(document.fragment),
            )
    return evidence


def _append_unique(values: list[str], value: str, *, maximum: int) -> None:
    if value and value not in values and len(values) < maximum:
        values.append(value)


def _best_display_name(accumulator: _EntityAccumulator) -> str:
    return min(
        accumulator.display_names,
        key=lambda value: (
            -accumulator.display_names[value],
            -len(value),
            value.casefold(),
            value,
        ),
    )


def _best_entity_type(accumulator: _EntityAccumulator) -> str:
    return min(
        accumulator.types or {"Other": 1},
        key=lambda value: (
            -accumulator.types[value],
            _TYPE_RANK.get(value, len(_TYPE_RANK)),
            value,
        ),
    )


def _joined_descriptions(values: Counter[str], *, maximum: int) -> str:
    ranked = sorted(
        values,
        key=lambda value: (-values[value], -len(value), value.casefold(), value),
    )
    return "；".join(ranked[:maximum])[:2_000]


def _entity_id(key: str) -> str:
    return f"textbook-entity:v1:{sha256(key.encode('utf-8')).hexdigest()}"


def _relationship_id(source_key: str, target_key: str) -> str:
    identity = f"{source_key}\x1f{target_key}"
    return f"textbook-relation:v1:{sha256(identity.encode('utf-8')).hexdigest()}"


def _write_json_item(stream: TextIO, item: object, *, first: bool) -> bool:
    if not first:
        stream.write(",")
    stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return False


def _load_extractions(
    settings: TextbookGraphBundleSettings,
    fragment_evidence: dict[str, tuple[str, str]],
) -> tuple[
    dict[str, _EntityAccumulator],
    dict[tuple[str, str], _RelationshipAccumulator],
    set[str],
    Counter[str],
    int,
    int,
]:
    entities: dict[str, _EntityAccumulator] = {}
    relationships: dict[tuple[str, str], _RelationshipAccumulator] = {}
    extracted_fragments: set[str] = set()
    prompt_versions: Counter[str] = Counter()
    filtered_entities = 0
    filtered_relationships = 0

    with settings.extractions_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                fragment_id = str(payload["fragment_id"])
                citation_path = str(payload["citation_path"])
                raw_entities = payload["entities"]
                raw_relationships = payload["relationships"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise TextbookGraphBundleError(
                    f"invalid extraction record at line {line_number}"
                ) from error
            if fragment_id in extracted_fragments:
                raise TextbookGraphBundleError("duplicate fragment extraction")
            evidence_record = fragment_evidence.get(fragment_id)
            if evidence_record is None:
                raise TextbookGraphBundleError("extraction references unknown fragments")
            source_evidence = evidence_record[0]
            extracted_fragments.add(fragment_id)
            prompt_versions[str(payload.get("prompt_version") or "unknown")] += 1

            local_keys: dict[str, str] = {}
            for raw in raw_entities:
                if not isinstance(raw, dict):
                    continue
                name = " ".join(str(raw.get("name") or "").split())
                key = _canonical_entity_key(name)
                description = " ".join(str(raw.get("description") or "").split())
                if not key or not name or not description:
                    continue
                evidence_name = _evidence_compact(name)
                if not evidence_name or evidence_name not in source_evidence:
                    filtered_entities += 1
                    continue
                accumulator = entities.setdefault(key, _EntityAccumulator())
                accumulator.display_names[name] += 1
                accumulator.types[str(raw.get("entity_type") or "Other")] += 1
                accumulator.descriptions[description] += 1
                accumulator.mentions += 1
                _append_unique(
                    accumulator.source_ids,
                    fragment_id,
                    maximum=settings.max_sources,
                )
                _append_unique(
                    accumulator.citation_paths,
                    citation_path,
                    maximum=settings.max_sources,
                )
                local_keys[name.casefold()] = key

            for raw in raw_relationships:
                if not isinstance(raw, dict):
                    continue
                source_key = local_keys.get(str(raw.get("source") or "").casefold())
                target_key = local_keys.get(str(raw.get("target") or "").casefold())
                description = " ".join(str(raw.get("description") or "").split())
                if not source_key or not target_key or source_key == target_key or not description:
                    filtered_relationships += 1
                    continue
                relation_key = tuple(sorted((source_key, target_key)))
                accumulator = relationships.setdefault(
                    relation_key,
                    _RelationshipAccumulator(),
                )
                accumulator.descriptions[description] += 1
                accumulator.mentions += 1
                _append_unique(
                    accumulator.source_ids,
                    fragment_id,
                    maximum=settings.max_sources,
                )
                _append_unique(
                    accumulator.citation_paths,
                    citation_path,
                    maximum=settings.max_sources,
                )

    return (
        entities,
        relationships,
        extracted_fragments,
        prompt_versions,
        filtered_entities,
        filtered_relationships,
    )


def build_textbook_graph_bundle(
    settings: TextbookGraphBundleSettings,
) -> TextbookGraphBundleSummary:
    """Canonicalize raw mentions and create a portable LightRAG custom-KG file."""

    started_at = time.time()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    fragment_evidence = _fragment_evidence_index(settings.fragments_path)
    (
        entities,
        relationships,
        extracted_fragments,
        prompt_versions,
        filtered_entities,
        filtered_relationships,
    ) = _load_extractions(
        settings,
        fragment_evidence,
    )
    entity_rows: dict[str, dict[str, Any]] = {}
    with settings.entities_path.open("w", encoding="utf-8", newline="\n") as stream:
        for key in sorted(entities):
            accumulator = entities[key]
            row = {
                "entity_id": _entity_id(key),
                "canonical_key": key,
                "name": _best_display_name(accumulator),
                "entity_type": _best_entity_type(accumulator),
                "description": _joined_descriptions(
                    accumulator.descriptions,
                    maximum=settings.max_descriptions,
                ),
                "mentions": accumulator.mentions,
                "source_ids": accumulator.source_ids,
                "citation_paths": accumulator.citation_paths,
            }
            entity_rows[key] = row
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )

    relationship_rows: list[dict[str, Any]] = []
    with settings.relationships_path.open("w", encoding="utf-8", newline="\n") as stream:
        for source_key, target_key in sorted(relationships):
            accumulator = relationships[(source_key, target_key)]
            source = entity_rows[source_key]
            target = entity_rows[target_key]
            row = {
                "relationship_id": _relationship_id(source_key, target_key),
                "source_entity_id": source["entity_id"],
                "target_entity_id": target["entity_id"],
                "source_name": source["name"],
                "target_name": target["name"],
                "description": _joined_descriptions(
                    accumulator.descriptions,
                    maximum=settings.max_descriptions,
                ),
                "mentions": accumulator.mentions,
                "source_ids": accumulator.source_ids,
                "citation_paths": accumulator.citation_paths,
            }
            relationship_rows.append(row)
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )

    fragment_count = 0
    seen_fragment_ids: set[str] = set()
    with settings.custom_kg_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"chunks":[')
        first = True
        for batch in iter_textbook_document_batches(
            settings.fragments_path,
            batch_size=512,
        ):
            for document in batch:
                fragment_id = str(document.fragment.fragment_id)
                fragment_count += 1
                seen_fragment_ids.add(fragment_id)
                first = _write_json_item(
                    stream,
                    {
                        "content": document.text,
                        "source_id": fragment_id,
                        "file_path": build_lightrag_basename(document.fragment),
                        "chunk_order_index": int(
                            document.fragment.metadata.get("chunk_index") or 0
                        ),
                    },
                    first=first,
                )
        missing = seen_fragment_ids - extracted_fragments
        if settings.require_complete and missing:
            raise TextbookGraphBundleError("raw extraction is incomplete")

        stream.write('],"entities":[')
        first = True
        for key in sorted(entity_rows):
            row = entity_rows[key]
            first = _write_json_item(
                stream,
                {
                    "entity_name": row["name"],
                    "entity_type": row["entity_type"],
                    "description": row["description"],
                    "source_id": row["source_ids"][0],
                    "file_path": fragment_evidence[row["source_ids"][0]][1],
                },
                first=first,
            )

        stream.write('],"relationships":[')
        first = True
        for row in relationship_rows:
            source_type = entity_rows[_canonical_entity_key(row["source_name"])]["entity_type"]
            target_type = entity_rows[_canonical_entity_key(row["target_name"])]["entity_type"]
            first = _write_json_item(
                stream,
                {
                    "src_id": row["source_name"],
                    "tgt_id": row["target_name"],
                    "description": row["description"],
                    "keywords": f"材料科学,教材知识,{source_type}-{target_type}",
                    "weight": round(1.0 + math.log1p(row["mentions"]), 6),
                    "source_id": row["source_ids"][0],
                    "file_path": fragment_evidence[row["source_ids"][0]][1],
                },
                first=first,
            )
        stream.write("]}\n")

    summary = TextbookGraphBundleSummary(
        fragment_count=fragment_count,
        extracted_fragment_count=len(extracted_fragments),
        missing_extraction_count=len(missing),
        entity_count=len(entity_rows),
        relationship_count=len(relationship_rows),
        entity_mentions=sum(item.mentions for item in entities.values()),
        relationship_mentions=sum(item.mentions for item in relationships.values()),
        filtered_entity_mentions=filtered_entities,
        filtered_relationship_mentions=filtered_relationships,
        elapsed_seconds=round(max(0.001, time.time() - started_at), 3),
        status="completed" if not missing else "completed_with_missing_extractions",
    )
    manifest = {
        "schema_version": 1,
        "summary": asdict(summary),
        "inputs": {
            "fragments_sha256": _file_sha256(settings.fragments_path),
            "extractions_sha256": _file_sha256(settings.extractions_path),
            "prompt_versions": dict(sorted(prompt_versions.items())),
        },
        "artifacts": {
            "entities": {
                "path": settings.entities_path.name,
                "sha256": _file_sha256(settings.entities_path),
            },
            "relationships": {
                "path": settings.relationships_path.name,
                "sha256": _file_sha256(settings.relationships_path),
            },
            "lightrag_custom_kg": {
                "path": settings.custom_kg_path.name,
                "sha256": _file_sha256(settings.custom_kg_path),
            },
        },
    }
    settings.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
