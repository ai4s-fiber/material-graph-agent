from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from material_graph.knowledge.catalog import (
    InMemorySourceCatalog,
    build_source_version_key,
    choose_canonical_source,
    normalize_doi,
)
from material_graph.knowledge.models import SourceCatalogRecord, SourceLocator


def _record(
    *,
    root_id: str = "document_data_3",
    path: str,
    title: str = "A material paper",
    doi: str | None = None,
    digest: str | None = None,
    byte_size: int = 100,
    metadata: dict[str, object] | None = None,
    source_kind: str = "literature",
    legal_status: str = "unknown",
) -> SourceCatalogRecord:
    return SourceCatalogRecord(
        locator=SourceLocator(root_id=root_id, relative_path=path),
        source_kind=source_kind,
        display_title=title,
        normalized_doi=doi,
        sha256=digest,
        byte_size=byte_size,
        legal_status=legal_status,
        metadata=metadata or {},
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" DOI:10.1000/ABC.1 ", "10.1000/abc.1"),
        ("https://doi.org/10.1000%2FABC.1", "10.1000/abc.1"),
        ("http://dx.doi.org/10.1000/ABC.1.", "10.1000/abc.1"),
        ("＜10.1000／ＡＢＣ.1＞", "10.1000/abc.1"),
        (None, None),
        ("  ", None),
    ],
)
def test_normalize_doi_handles_unicode_prefixes_and_trailing_punctuation(
    raw: str | None,
    expected: str | None,
) -> None:
    assert normalize_doi(raw) == expected


def test_normalize_doi_rejects_non_doi_values() -> None:
    with pytest.raises(ValueError, match="valid DOI"):
        normalize_doi("paper-without-a-doi")


def test_source_version_key_is_stable_and_sensitive_to_remote_version() -> None:
    locator = SourceLocator(root_id="document_data_1", relative_path="papers/a.pdf")
    modified = datetime(2026, 7, 1, 8, 30, tzinfo=timezone.utc)

    first = build_source_version_key(
        locator=locator,
        byte_size=42,
        remote_modified_at=modified,
    )
    same = build_source_version_key(
        locator=locator,
        byte_size=42,
        remote_modified_at="2026-07-01T08:30:00+00:00",
    )
    changed = build_source_version_key(
        locator=locator,
        byte_size=43,
        remote_modified_at=modified,
    )

    assert first == same
    assert first.startswith("source-version-v1:")
    assert first != changed


def test_same_logical_path_is_idempotent_and_merges_metadata() -> None:
    catalog = InMemorySourceCatalog()
    first = catalog.upsert(
        _record(path="papers/a.pdf", doi="doi:10.1000/A", metadata={"authors": ["A"]}),
        remote_modified_at="2026-07-01T00:00:00Z",
    )
    second = catalog.upsert(
        _record(path="papers/a.pdf", doi="https://doi.org/10.1000/a", metadata={"year": 2024}),
        remote_modified_at="2026-07-01T00:00:00Z",
    )

    assert first.record.source_id == second.record.source_id
    assert second.created is False
    assert catalog.count() == 1
    assert second.record.normalized_doi == "10.1000/a"
    assert second.record.metadata["authors"] == ["A"]
    assert second.record.metadata["year"] == 2024


def test_identical_sha_records_share_one_canonical_source() -> None:
    catalog = InMemorySourceCatalog()
    digest = "a" * 64
    first = catalog.upsert(
        _record(root_id="document_data_3", path="copy/a.pdf", digest=digest),
    ).record
    preferred = catalog.upsert(
        _record(
            root_id="document_data_1",
            path="library/a.pdf",
            digest=digest,
            metadata={"authors": ["A"], "year": 2025},
        ),
    ).record

    refreshed_first = catalog.get(first.source_id)
    refreshed_preferred = catalog.get(preferred.source_id)

    assert refreshed_first.canonical_source_id == preferred.source_id
    assert refreshed_preferred.canonical_source_id is None
    assert catalog.canonical_for(first.source_id).source_id == preferred.source_id
    assert catalog.canonical_for(preferred.source_id).source_id == preferred.source_id


def test_same_doi_with_different_sha_creates_version_relation_without_merging() -> None:
    catalog = InMemorySourceCatalog()
    older = catalog.upsert(
        _record(path="versions/a-v1.pdf", doi="10.1000/Version", digest="a" * 64),
    ).record
    richer = catalog.upsert(
        _record(
            root_id="document_data_1",
            path="versions/a-v2.pdf",
            doi="https://doi.org/10.1000/version",
            digest="b" * 64,
            metadata={"authors": ["A"], "abstract": "Evidence"},
        ),
    ).record

    relations = catalog.relations("IS_VERSION_OF")

    assert catalog.count() == 2
    assert catalog.get(older.source_id).canonical_source_id is None
    assert catalog.get(richer.source_id).canonical_source_id is None
    assert len(relations) == 1
    assert relations[0].source_id == older.source_id
    assert relations[0].target_source_id == richer.source_id
    assert relations[0].normalized_doi == "10.1000/version"


def test_canonical_choice_prefers_metadata_then_primary_root_then_stable_id() -> None:
    sparse_primary = _record(root_id="document_data_1", path="a.pdf", metadata={})
    rich_secondary = _record(
        root_id="document_data_3",
        path="b.pdf",
        metadata={"authors": ["A"], "year": 2020},
    )
    assert choose_canonical_source([sparse_primary, rich_secondary]) == rich_secondary

    equally_rich_primary = _record(
        root_id="document_data_1",
        path="c.pdf",
        metadata={"authors": ["A"], "year": 2020},
    )
    assert choose_canonical_source([rich_secondary, equally_rich_primary]) == equally_rich_primary

    first_stable = equally_rich_primary.model_copy(update={"source_id": UUID(int=1)})
    second_stable = equally_rich_primary.model_copy(
        update={
            "source_id": UUID(int=2),
            "locator": SourceLocator(root_id="document_data_1", relative_path="d.pdf"),
        }
    )
    assert choose_canonical_source([second_stable, first_stable]) == first_stable


def test_data_2_process_data_is_excluded_without_body_reads() -> None:
    catalog = InMemorySourceCatalog()
    result = catalog.upsert(
        _record(root_id="data_2", path="process_data/mineru/intermediate.json"),
    )

    assert result.record.status == "excluded_process_data"
    assert result.record.metadata["exclusion_reason"] == "process_data_never_open"
    assert catalog.stats.body_read_count == 0


def test_patent_filename_does_not_imply_current_legal_status() -> None:
    catalog = InMemorySourceCatalog()
    result = catalog.upsert(
        _record(
            root_id="document_data_1",
            path="patents/CN111676547B.pdf",
            title="一种材料制备方法",
            source_kind="patent",
        )
    )

    assert result.record.legal_status == "unknown"
