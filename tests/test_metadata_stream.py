from __future__ import annotations

import csv
import io
import json
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from material_graph.knowledge.catalog import InMemorySourceCatalog, build_source_version_key
from material_graph.knowledge.manifest import (
    InMemoryCursorRepository,
    MetadataCursor,
    MetadataCursorKey,
    MetadataManifestIngestor,
    MetadataStreamError,
    MetadataStreamLimits,
)
from material_graph.knowledge.models import SourceLocator
from material_graph.knowledge.remote_reader import (
    DirectoryCursor,
    RemoteEntry,
    RemoteSourceReader,
    RemoteStat,
)


class ManifestReader(RemoteSourceReader):
    def __init__(
        self,
        content: bytes,
        *,
        manifest_path: str = "manifests/catalog.jsonl",
        modified_at: int = 100,
        stream_error: Exception | None = None,
        stat_error: Exception | None = None,
        is_dir: bool = False,
    ) -> None:
        self.content = content
        self.manifest_path = manifest_path
        self.modified_at = modified_at
        self.stream_error = stream_error
        self.stat_error = stat_error
        self.is_dir = is_dir
        self.opened_paths: list[str] = []
        self.requested_offsets: list[int] = []
        self.requested_expectations: list[tuple[int | None, int | None]] = []
        self.closed = False

    def iter_entries(
        self,
        root_id: str,
        slice_id: str,
        *,
        cursor: DirectoryCursor | None = None,
        page_size: int = 500,
    ) -> AsyncIterator[RemoteEntry]:
        del root_id, slice_id, cursor, page_size

        async def empty() -> AsyncIterator[RemoteEntry]:
            if False:  # pragma: no cover - establishes async-generator type
                yield RemoteEntry(
                    root_id="data_2",
                    slice_id="documents",
                    relative_path="unused",
                    name="unused",
                    is_dir=False,
                )

        return empty()

    async def stat(self, root_id: str, slice_id: str, relative_path: str) -> RemoteStat:
        if self.stat_error is not None:
            raise self.stat_error
        assert relative_path == self.manifest_path
        return RemoteStat(
            root_id=root_id,
            slice_id=slice_id,
            relative_path=relative_path,
            is_dir=self.is_dir,
            byte_size=len(self.content),
            modified_at=self.modified_at,
        )

    def open_stream(
        self,
        root_id: str,
        slice_id: str,
        relative_path: str,
        *,
        offset: int = 0,
        expected_size: int | None = None,
        expected_mtime: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        del root_id, slice_id, chunk_size
        self.opened_paths.append(relative_path)
        self.requested_offsets.append(offset)
        self.requested_expectations.append((expected_size, expected_mtime))

        async def chunks() -> AsyncIterator[bytes]:
            if self.stream_error is not None:
                raise self.stream_error
            remaining = self.content[offset:]
            for index in range(0, len(remaining), 7):
                yield remaining[index : index + 7]

        return chunks()

    async def close(self) -> None:
        self.closed = True


class FailOnceCursorRepository(InMemoryCursorRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_save = True

    def save(self, cursor: MetadataCursor) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError("sensitive cursor backend detail")
        super().save(cursor)


class RaisingCursorRepository(InMemoryCursorRepository):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    def load(self, key: MetadataCursorKey) -> MetadataCursor | None:
        if self.operation == "load":
            raise RuntimeError("sensitive load detail")
        return super().load(key)

    def save(self, cursor: MetadataCursor) -> None:
        if self.operation == "save":
            raise RuntimeError("sensitive save detail")
        super().save(cursor)


def _service(
    reader: ManifestReader,
    *,
    catalog: InMemorySourceCatalog | None = None,
    cursors: InMemoryCursorRepository | None = None,
    limits: MetadataStreamLimits | None = None,
) -> tuple[MetadataManifestIngestor, InMemorySourceCatalog, InMemoryCursorRepository]:
    selected_catalog = catalog or InMemorySourceCatalog()
    selected_cursors = cursors or InMemoryCursorRepository()
    return (
        MetadataManifestIngestor(
            reader=reader,
            catalog=selected_catalog,
            cursors=selected_cursors,
            limits=limits,
        ),
        selected_catalog,
        selected_cursors,
    )


async def _ingest(
    service: MetadataManifestIngestor,
    *,
    root_id: str = "data_2",
    manifest_path: str = "manifests/catalog.jsonl",
    manifest_format: str = "jsonl",
):
    return await service.ingest(
        root_id=root_id,
        slice_id="documents",
        manifest_path=manifest_path,
        manifest_format=manifest_format,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_jsonl_catalogues_metadata_doi_first_and_never_opens_referenced_bodies() -> None:
    rows = [
        {
            "path": "papers/polyimide.pdf",
            "title": "Low-k polyimide",
            "doi": " DOI:10.1000/PI.1 ",
            "byte_size": 42,
            "modified_at": 123,
            "source_kind": "literature",
            "authors": ["A"],
        },
        {
            "path": "process_data/mineru/intermediate.pdf",
            "title": "Parser intermediate",
        },
    ]
    content = b"".join(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in rows)
    reader = ManifestReader(content)
    service, catalog, _ = _service(reader)

    result = await _ingest(service)
    replay = await _ingest(service)
    records = {record.locator.relative_path: record for record in catalog.list_records()}

    assert result.records_seen == result.records_created == 2
    assert result.bounded_digest_required == 1
    assert result.cursor.next_byte_offset == len(content)
    assert replay.records_seen == 0
    assert catalog.count() == 2
    assert records["papers/polyimide.pdf"].normalized_doi == "10.1000/pi.1"
    assert records["papers/polyimide.pdf"].metadata["identity_basis"] == "doi"
    assert records["papers/polyimide.pdf"].metadata["bounded_digest_required"] is False
    process_record = records["process_data/mineru/intermediate.pdf"]
    assert process_record.status == "excluded_process_data"
    assert process_record.metadata["bounded_digest_required"] is True
    assert process_record.metadata["exclusion_reason"] == "process_data_never_open"
    assert reader.opened_paths == [reader.manifest_path, reader.manifest_path]
    assert not {row["path"] for row in rows} & set(reader.opened_paths)
    assert reader.requested_offsets == [0, len(content)]
    assert reader.requested_expectations == [(len(content), 100), (len(content), 100)]
    assert catalog.stats.body_read_count == 0


@pytest.mark.asyncio
async def test_csv_supports_rfc4180_multiline_cells_and_resumes_after_header() -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(["path", "title", "doi", "metadata"])
    writer.writerow(
        [
            "papers/a.pdf",
            'Line one\nline two, "quoted"',
            "https://doi.org/10.1000/A",
            json.dumps({"authors": ["A"]}),
        ]
    )
    content = stream.getvalue().encode("utf-8")
    reader = ManifestReader(content, manifest_path="manifests/catalog.csv")
    service, catalog, cursors = _service(reader)

    result = await _ingest(
        service,
        manifest_path="manifests/catalog.csv",
        manifest_format="csv",
    )
    replay = await _ingest(
        service,
        manifest_path="manifests/catalog.csv",
        manifest_format="csv",
    )
    record = catalog.list_records()[0]
    key = MetadataCursorKey("data_2", "documents", "manifests/catalog.csv", "csv")

    assert result.records_seen == result.records_created == 1
    assert result.cursor.csv_fieldnames == ("path", "title", "doi", "metadata")
    assert result.cursor.next_byte_offset == len(content)
    assert cursors.load(key) == result.cursor
    assert replay.records_seen == 0
    assert record.display_title == 'Line one\nline two, "quoted"'
    assert record.metadata["authors"] == ["A"]
    assert record.normalized_doi == "10.1000/a"
    assert reader.opened_paths == [reader.manifest_path, reader.manifest_path]


@pytest.mark.asyncio
async def test_cursor_save_crash_replays_idempotently_without_skip_or_new_duplicate() -> None:
    rows = [
        {"path": "papers/a.pdf", "title": "A"},
        {"path": "papers/b.pdf", "title": "B"},
    ]
    content = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    reader = ManifestReader(content)
    catalog = InMemorySourceCatalog()
    cursors = FailOnceCursorRepository()
    service, _, _ = _service(reader, catalog=catalog, cursors=cursors)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)
    assert captured.value.code == "metadata.cursor.save_failed"
    assert captured.value.failure_class == "cursor"
    assert captured.value.retryable is True
    assert "sensitive" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert catalog.count() == 1

    result = await _ingest(service)

    assert result.records_seen == 2
    assert result.records_created == 1
    assert result.records_updated == 1
    assert catalog.count() == 2
    assert {record.locator.relative_path for record in catalog.list_records()} == {
        "papers/a.pdf",
        "papers/b.pdf",
    }
    assert len({record.source_id for record in catalog.list_records()}) == 2


def test_cursor_checkpoint_is_safe_strict_and_round_trips() -> None:
    cursor = MetadataCursor(
        root_id="data_2",
        slice_id="documents",
        manifest_path="manifests/catalog.csv",
        manifest_format="csv",
        manifest_version_key="source-version-v1:" + "a" * 64,
        next_byte_offset=20,
        records_committed=3,
        csv_fieldnames=("path", "title"),
    )
    checkpoint = cursor.to_checkpoint()

    assert MetadataCursor.from_checkpoint(checkpoint) == cursor
    assert (
        not {
            "endpoint",
            "password",
            "raw_line",
            "session",
            "token",
        }
        & checkpoint.keys()
    )
    assert "secret row value" not in repr(checkpoint)

    with pytest.raises(ValueError, match="invalid metadata cursor fields"):
        MetadataCursor.from_checkpoint({**checkpoint, "token": "must-not-persist"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_line_bytes": 0},
        {"max_fields": True},
        {"max_line_bytes": 9, "max_metadata_bytes": 8},
    ],
)
def test_limits_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MetadataStreamLimits(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content", "limits", "code"),
    [
        (b'{"path":"a.pdf","path":"b.pdf"}\n', None, "metadata.parse.duplicate_field"),
        (b'{"path":"a.pdf","metadata":{"token":"x"}}\n', None, "metadata.parse.forbidden_field"),
        (b'{"path":"a.pdf",}\n', None, "metadata.parse.invalid_json"),
        (b"\xff\n", None, "metadata.parse.invalid_utf8"),
        (b'["a.pdf"]\n', None, "metadata.parse.invalid_schema"),
        (
            b'{"path":"a.pdf","title":"A","extra":1}\n',
            MetadataStreamLimits(max_fields=2),
            "metadata.parse.too_many_fields",
        ),
        (
            b'{"path":"a.pdf","metadata":{"a":{"b":1}}}\n',
            MetadataStreamLimits(max_depth=1),
            "metadata.parse.nesting_too_deep",
        ),
        (
            b'{"path":"a.pdf","title":"01234567890"}\n',
            MetadataStreamLimits(max_cell_bytes=10),
            "metadata.parse.cell_too_large",
        ),
        (
            b'{"path":"a-very-long-name.pdf"}\n',
            MetadataStreamLimits(max_line_bytes=20, max_metadata_bytes=100),
            "metadata.parse.line_too_large",
        ),
        (b'{"path":"../escape.pdf"}\n', None, "metadata.parse.invalid_path"),
        (
            b'{"path":"a.pdf","doi":"not-a-doi"}\n',
            None,
            "metadata.parse.invalid_doi",
        ),
        (
            b'{"path":"a.pdf","value":NaN}\n',
            None,
            "metadata.parse.non_finite_number",
        ),
    ],
)
@pytest.mark.asyncio
async def test_jsonl_parse_failures_have_stable_codes_without_remote_content(
    content: bytes,
    limits: MetadataStreamLimits | None,
    code: str,
) -> None:
    reader = ManifestReader(content)
    service, catalog, _ = _service(reader, limits=limits)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == code
    assert captured.value.failure_class == "parse"
    assert str(captured.value) == code
    decoded_content = content.decode("utf-8", errors="ignore").strip()
    if decoded_content:
        assert decoded_content not in str(captured.value)
    assert catalog.count() == 0


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"path,path\r\na.pdf,a.pdf\r\n", "metadata.parse.duplicate_field"),
        (b"path,api-key\r\na.pdf,value\r\n", "metadata.parse.invalid_csv_header"),
        (b"path,title\r\na.pdf\r\n", "metadata.parse.csv_field_count_mismatch"),
        (b"", "metadata.parse.missing_csv_header"),
        (b'path,title\r\na.pdf,"unterminated\r\n', "metadata.parse.invalid_csv"),
    ],
)
@pytest.mark.asyncio
async def test_csv_rejects_unsafe_or_malformed_records(content: bytes, code: str) -> None:
    reader = ManifestReader(content, manifest_path="manifest.csv")
    service, catalog, _ = _service(reader)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service, manifest_path="manifest.csv", manifest_format="csv")

    assert captured.value.code == code
    assert catalog.count() == 0


@pytest.mark.asyncio
async def test_csv_bounds_multiline_logical_record_without_reading_whole_manifest() -> None:
    content = (
        b"path,title\r\n"
        + b'a.pdf,"'
        + b"a" * 20
        + b"\n"
        + b"b" * 20
        + b"\n"
        + b"c" * 20
        + b'"\r\n'
    )
    limits = MetadataStreamLimits(
        max_line_bytes=30,
        max_cell_bytes=100,
        max_metadata_bytes=50,
    )
    reader = ManifestReader(content, manifest_path="manifest.csv")
    service, catalog, _ = _service(reader, limits=limits)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service, manifest_path="manifest.csv", manifest_format="csv")

    assert captured.value.code == "metadata.parse.record_too_large"
    assert catalog.count() == 0
    assert reader.opened_paths == ["manifest.csv"]


@pytest.mark.parametrize(
    ("reader", "code"),
    [
        (
            ManifestReader(b"", stat_error=RuntimeError("remote secret payload")),
            "metadata.provider.stat_failed",
        ),
        (
            ManifestReader(
                b'{"path":"a.pdf"}\n',
                stream_error=RuntimeError("remote secret payload"),
            ),
            "metadata.provider.stream_failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_provider_failures_are_retryable_sanitized_and_stable(
    reader: ManifestReader,
    code: str,
) -> None:
    service, _, _ = _service(reader)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == code
    assert captured.value.failure_class == "provider"
    assert captured.value.retryable is True
    assert "secret" not in str(captured.value)
    assert captured.value.__cause__ is None


@pytest.mark.parametrize(
    ("operation", "code"),
    [("load", "metadata.cursor.load_failed"), ("save", "metadata.cursor.save_failed")],
)
@pytest.mark.asyncio
async def test_cursor_repository_failures_are_sanitized(operation: str, code: str) -> None:
    reader = ManifestReader(b'{"path":"a.pdf"}\n')
    cursors = RaisingCursorRepository(operation)
    service, _, _ = _service(reader, cursors=cursors)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == code
    assert captured.value.failure_class == "cursor"
    assert captured.value.__cause__ is None


@pytest.mark.asyncio
async def test_sha_metadata_is_used_without_scheduling_digest_and_chinese_aliases_map() -> None:
    row = {
        "路径": "patents/CN-example.pdf",
        "标题": "一种纤维材料",
        "SHA256": "A" * 64,
        "文件大小": "12",
        "年份": "2024",
        "root_id": "data_2",
        "slice_id": "documents",
    }
    reader = ManifestReader(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")
    service, catalog, _ = _service(reader)

    result = await _ingest(service)
    record = catalog.list_records()[0]

    assert result.bounded_digest_required == 0
    assert record.display_title == "一种纤维材料"
    assert record.sha256 == "a" * 64
    assert record.byte_size == 12
    assert record.directory_year == 2024
    assert record.metadata["identity_basis"] == "sha256"
    assert record.metadata["bounded_digest_required"] is False


@pytest.mark.parametrize(
    ("row", "limits", "code"),
    [
        ({"title": "missing"}, None, "metadata.parse.missing_path"),
        ({"path": "/volume1/private/paper.pdf"}, None, "metadata.parse.invalid_path"),
        ({"path": "a.pdf", "root_id": "other"}, None, "metadata.parse.root_mismatch"),
        ({"path": "a.pdf", "slice_id": "other"}, None, "metadata.parse.slice_mismatch"),
        ({"path": "a.pdf", "sha256": "bad"}, None, "metadata.parse.invalid_sha256"),
        ({"path": "a.pdf", "byte_size": True}, None, "metadata.parse.invalid_byte_size"),
        (
            {"path": "a.pdf", "modified_at": {"bad": 1}},
            None,
            "metadata.parse.invalid_modified_at",
        ),
        ({"path": "a.pdf", "metadata": []}, None, "metadata.parse.invalid_metadata"),
        (
            {"path": "a.pdf", "metadata": {"authors": ["A"]}, "authors": ["B"]},
            None,
            "metadata.parse.duplicate_field",
        ),
        ({"path": "a.pdf", "source_kind": "unsafe"}, None, "metadata.parse.invalid_schema"),
        ({"path": "a.pdf", "year": 1700}, None, "metadata.parse.invalid_schema"),
        (
            {"path": "a.pdf", "metadata": {"tags": [1, 2, 3]}},
            MetadataStreamLimits(max_fields=2),
            "metadata.parse.too_many_items",
        ),
    ],
)
@pytest.mark.asyncio
async def test_record_mapping_rejects_invalid_values_without_echo(
    row: dict[str, object],
    limits: MetadataStreamLimits | None,
    code: str,
) -> None:
    raw_line = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
    reader = ManifestReader(raw_line)
    service, catalog, _ = _service(reader, limits=limits)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == code
    assert raw_line.decode("utf-8").strip() not in str(captured.value)
    assert "/volume1/private" not in str(captured.value)
    assert catalog.count() == 0


@pytest.mark.asyncio
async def test_manifest_version_change_is_fail_closed_before_resuming() -> None:
    content = b'{"path":"a.pdf"}\n'
    reader = ManifestReader(content)
    service, catalog, _ = _service(reader)
    await _ingest(service)
    reader.modified_at += 1

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.cursor.version_mismatch"
    assert captured.value.failure_class == "cursor"
    assert captured.value.retryable is False
    assert catalog.count() == 1
    assert reader.opened_paths == [reader.manifest_path]


@pytest.mark.asyncio
async def test_cursor_offset_beyond_manifest_is_rejected_before_streaming() -> None:
    content = b'{"path":"a.pdf"}\n'
    reader = ManifestReader(content)
    cursors = InMemoryCursorRepository()
    locator = SourceLocator(root_id="data_2", relative_path=reader.manifest_path)
    version = build_source_version_key(
        locator=locator,
        byte_size=len(content),
        remote_modified_at=reader.modified_at,
    )
    cursors.save(
        MetadataCursor(
            root_id="data_2",
            slice_id="documents",
            manifest_path=reader.manifest_path,
            manifest_format="jsonl",
            manifest_version_key=version,
            next_byte_offset=len(content) + 1,
        )
    )
    service, _, _ = _service(reader, cursors=cursors)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.cursor.offset_out_of_range"
    assert reader.opened_paths == []


@pytest.mark.asyncio
async def test_directory_and_invalid_manifest_locator_are_rejected_safely() -> None:
    directory_reader = ManifestReader(b"", is_dir=True)
    service, _, _ = _service(directory_reader)
    with pytest.raises(MetadataStreamError) as directory_error:
        await _ingest(service)
    assert directory_error.value.code == "metadata.parse.manifest_is_directory"

    reader = ManifestReader(b"")
    service, _, _ = _service(reader)
    with pytest.raises(MetadataStreamError) as locator_error:
        await _ingest(service, manifest_path="https://nas.invalid/manifest.jsonl")
    assert locator_error.value.code == "metadata.parse.invalid_manifest_locator"
    assert "nas.invalid" not in str(locator_error.value)


@pytest.mark.asyncio
async def test_catalog_failure_is_sanitized_and_does_not_advance_cursor() -> None:
    class RaisingCatalog(InMemorySourceCatalog):
        def upsert(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("/volume1/private/raw-line and provider token")

    reader = ManifestReader(b'{"path":"a.pdf","title":"raw-line"}\n')
    cursors = InMemoryCursorRepository()
    service, _, _ = _service(reader, catalog=RaisingCatalog(), cursors=cursors)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.catalog.upsert_failed"
    assert captured.value.failure_class == "catalog"
    assert captured.value.retryable is True
    assert "volume1" not in str(captured.value)
    assert "raw-line" not in str(captured.value)
    key = MetadataCursorKey("data_2", "documents", reader.manifest_path, "jsonl")
    assert cursors.load(key) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"manifest_format": "xml"},
        {"schema_version": 2},
        {"manifest_version_key": "bad"},
        {"next_byte_offset": -1},
        {"records_committed": True},
        {"manifest_format": "jsonl", "csv_fieldnames": ("path",)},
        {"manifest_format": "csv", "next_byte_offset": 1, "csv_fieldnames": None},
        {"manifest_format": "csv", "csv_fieldnames": ()},
        {"manifest_format": "csv", "csv_fieldnames": ("path", "PATH")},
        {"manifest_format": "csv", "csv_fieldnames": ("path", "token")},
    ],
)
def test_cursor_validation_rejects_unsafe_or_inconsistent_state(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "root_id": "data_2",
        "slice_id": "documents",
        "manifest_path": "manifest.csv",
        "manifest_format": "csv",
        "manifest_version_key": "source-version-v1:" + "a" * 64,
        "csv_fieldnames": ("path",),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        MetadataCursor(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("fieldnames", ["path", ["path", 1]])
def test_cursor_checkpoint_rejects_non_string_fieldname_sequences(fieldnames: object) -> None:
    cursor = MetadataCursor(
        root_id="data_2",
        slice_id="documents",
        manifest_path="manifest.csv",
        manifest_format="csv",
        manifest_version_key="source-version-v1:" + "a" * 64,
        csv_fieldnames=("path",),
    )
    checkpoint = cursor.to_checkpoint()
    checkpoint["csv_fieldnames"] = fieldnames

    with pytest.raises(ValueError, match="invalid CSV field names"):
        MetadataCursor.from_checkpoint(checkpoint)


@pytest.mark.asyncio
async def test_cursor_identity_mismatch_is_rejected_without_streaming() -> None:
    content = b'{"path":"a.pdf"}\n'
    reader = ManifestReader(content)
    locator = SourceLocator(root_id="data_2", relative_path=reader.manifest_path)
    version = build_source_version_key(
        locator=locator,
        byte_size=len(content),
        remote_modified_at=reader.modified_at,
    )
    wrong_cursor = MetadataCursor(
        root_id="data_2",
        slice_id="documents",
        manifest_path="other.jsonl",
        manifest_format="jsonl",
        manifest_version_key=version,
    )

    class WrongIdentityRepository(InMemoryCursorRepository):
        def load(self, key: MetadataCursorKey) -> MetadataCursor | None:
            del key
            return replace(wrong_cursor)

    service, _, _ = _service(reader, cursors=WrongIdentityRepository())

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.cursor.identity_mismatch"
    assert reader.opened_paths == []


@pytest.mark.asyncio
async def test_blank_jsonl_lines_advance_cursor_without_catalog_entries() -> None:
    content = b"\r\n\n" + b'{"path":"a.pdf"}\n'
    reader = ManifestReader(content)
    service, catalog, _ = _service(reader)

    result = await _ingest(service)

    assert result.records_seen == 1
    assert result.cursor.records_committed == 1
    assert result.cursor.next_byte_offset == len(content)
    assert catalog.count() == 1


@pytest.mark.asyncio
async def test_mapped_metadata_size_is_bounded_after_internal_fields_are_added() -> None:
    content = b'{"path":"a","x":"1"}\n'
    limits = MetadataStreamLimits(
        max_line_bytes=40,
        max_cell_bytes=40,
        max_metadata_bytes=40,
    )
    reader = ManifestReader(content)
    service, catalog, _ = _service(reader, limits=limits)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.parse.object_too_large"
    assert catalog.count() == 0


@pytest.mark.asyncio
async def test_escaped_invalid_unicode_is_rejected_as_utf8_failure() -> None:
    reader = ManifestReader(b'{"path":"a.pdf","title":"\\ud800"}\n')
    service, _, _ = _service(reader)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.parse.invalid_utf8"


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ({"path": "a.pdf", "relative_path": "b.pdf"}, "metadata.parse.duplicate_field"),
        ({"path": "a.pdf", "metadata": "[]"}, "metadata.parse.invalid_metadata"),
        ({"path": "a.pdf", "byte_size": "not-an-int"}, "metadata.parse.invalid_byte_size"),
        ({"path": "a.pdf", "byte_size": -1}, "metadata.parse.invalid_byte_size"),
        ({"path": "a.pdf", "title": 1}, "metadata.parse.invalid_schema"),
    ],
)
@pytest.mark.asyncio
async def test_additional_mapping_failures_are_classified(
    row: dict[str, object],
    code: str,
) -> None:
    reader = ManifestReader(json.dumps(row).encode("utf-8") + b"\n")
    service, _, _ = _service(reader)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == code


@pytest.mark.asyncio
async def test_optional_empty_values_are_not_persisted_and_finite_float_is_allowed() -> None:
    row = {
        "path": "a.pdf",
        "modified_at": "",
        "empty_extra": "",
        "score": 1.5,
    }
    reader = ManifestReader(json.dumps(row).encode("utf-8") + b"\n")
    service, catalog, _ = _service(reader)

    await _ingest(service)
    record = catalog.list_records()[0]

    assert "empty_extra" not in record.metadata
    assert record.metadata["score"] == 1.5
    assert "remote_modified_at" not in record.metadata


@pytest.mark.parametrize(
    ("content", "limits", "code"),
    [
        (
            b"path,title,doi\r\na.pdf,A,10.1000/a\r\n",
            MetadataStreamLimits(max_fields=2),
            "metadata.parse.too_many_fields",
        ),
        (
            b"path,title\r\na.pdf,123456\r\n",
            MetadataStreamLimits(max_cell_bytes=5),
            "metadata.parse.cell_too_large",
        ),
        (
            b'path,title\r\na.pdf,"bad"tail\r\n',
            None,
            "metadata.parse.invalid_csv",
        ),
        (b"path,title\r\na.pdf,\xff\r\n", None, "metadata.parse.invalid_utf8"),
    ],
)
@pytest.mark.asyncio
async def test_csv_parser_bounds_and_utf8_failures(
    content: bytes,
    limits: MetadataStreamLimits | None,
    code: str,
) -> None:
    reader = ManifestReader(content, manifest_path="manifest.csv")
    service, _, _ = _service(reader, limits=limits)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service, manifest_path="manifest.csv", manifest_format="csv")

    assert captured.value.code == code


@pytest.mark.asyncio
async def test_non_bytes_reader_output_is_sanitized_as_provider_failure() -> None:
    class InvalidLineReader(ManifestReader):
        async def iter_lines(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            yield "NAS /volume1/private/raw-line"  # type: ignore[misc]

    reader = InvalidLineReader(b'{"path":"a.pdf"}\n')
    service, _, _ = _service(reader)

    with pytest.raises(MetadataStreamError) as captured:
        await _ingest(service)

    assert captured.value.code == "metadata.provider.stream_failed"
    assert "volume1" not in str(captured.value)
    assert "raw-line" not in str(captured.value)
