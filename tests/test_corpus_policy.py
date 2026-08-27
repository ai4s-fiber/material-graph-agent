from pathlib import Path

import pytest

from material_graph.knowledge.policy import CorpusPolicy, SpoolPolicy


POLICY_PATH = Path("config/knowledge/corpus-policy.v1.json")


def test_verified_corpus_policy_matches_inventory():
    policy = CorpusPolicy.load(POLICY_PATH)

    data1 = policy.source("document_data_1")
    assert (data1.verified_size_gb, data1.pdf_count) == (621, 222_169)
    assert (data1.literature_size_gb, data1.patent_size_gb) == (592, 29)
    assert data1.physical_material_type_count == 17
    assert data1.deduplicated_record_count == 216_339
    assert data1.existing_pdf_record_count == 197_866
    assert data1.no_source_record_count == 18_473
    assert set(data1.metadata_formats) == {"csv", "jsonl"}

    data2 = policy.source("data_2")
    assert (data2.verified_size_gb, data2.file_count) == (137, 31_076)
    assert (data2.usable_size_gb, data2.excluded_process_size_gb) == (35.6, 102)

    data3 = policy.source("data_3")
    assert (data3.verified_size_gb, data3.pdf_count) == (16, 3_751)
    assert data3.deduplicate_against == "document_data_1"


def test_policy_contains_no_connection_or_credential_fields(tmp_path):
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        '{"schema_version":1,"server_capacity_gb":130,'
        '"minimum_free_bytes":34359738368,"hard_stop_free_bytes":27917287424,'
        '"derived_data_target_bytes":1,"derived_data_hard_cap_bytes":2,'
        '"filesystem_alert_ratio":0.75,"filesystem_stop_ratio":0.8,'
        '"spool":{"max_total_bytes":1,"max_object_bytes":1,'
        '"max_active_objects":1,"abandoned_ttl_seconds":60},'
        '"sources":[{"root_id":"data","display_name":"data",'
        '"verified_size_gb":1,"password":"must-not-be-here"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden configuration field: password"):
        CorpusPolicy.load(unsafe)


@pytest.mark.parametrize(
    "field",
    [
        "quickconnect_id",
        "device-token",
        "did",
        "relay_cookie",
        "sessionSid",
        "providerCredential",
    ],
)
def test_policy_rejects_transport_credentials_hidden_in_extra_fields(
    tmp_path,
    field: str,
):
    unsafe = tmp_path / "unsafe-extra.json"
    unsafe.write_text(
        '{"schema_version":1,"server_capacity_gb":130,'
        '"minimum_free_bytes":34359738368,'
        '"hard_stop_free_bytes":27917287424,'
        '"derived_data_target_bytes":80530636800,'
        '"derived_data_hard_cap_bytes":85899345920,'
        '"filesystem_alert_ratio":0.75,"filesystem_stop_ratio":0.8,'
        '"spool":{"max_total_bytes":8589934592,'
        '"max_object_bytes":1073741824,"max_active_objects":4,'
        '"abandoned_ttl_seconds":3600},'
        f'"sources":[{{"root_id":"data","display_name":"Data",'
        f'"verified_size_gb":1,"{field}":"must-not-be-here"}}]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden configuration field"):
        CorpusPolicy.load(unsafe)


def test_policy_does_not_misclassify_non_secret_author_metadata(tmp_path):
    safe = tmp_path / "safe-extra.json"
    safe.write_text(
        '{"schema_version":1,"server_capacity_gb":130,'
        '"minimum_free_bytes":34359738368,'
        '"hard_stop_free_bytes":27917287424,'
        '"derived_data_target_bytes":80530636800,'
        '"derived_data_hard_cap_bytes":85899345920,'
        '"filesystem_alert_ratio":0.75,"filesystem_stop_ratio":0.8,'
        '"spool":{"max_total_bytes":8589934592,'
        '"max_object_bytes":1073741824,"max_active_objects":4,'
        '"abandoned_ttl_seconds":3600},'
        '"sources":[{"root_id":"data","display_name":"Data",'
        '"verified_size_gb":1,"authoritative_inventory":true}]}',
        encoding="utf-8",
    )

    assert CorpusPolicy.load(safe).source("data").model_extra == {"authoritative_inventory": True}


def test_policy_rejects_unsafe_capacity_thresholds(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        '{"schema_version":1,"server_capacity_gb":10,'
        '"minimum_free_bytes":10,"hard_stop_free_bytes":20,'
        '"derived_data_target_bytes":1,"derived_data_hard_cap_bytes":2,'
        '"filesystem_alert_ratio":0.75,"filesystem_stop_ratio":0.8,'
        '"spool":{"max_total_bytes":1,"max_object_bytes":1,'
        '"max_active_objects":1,"abandoned_ttl_seconds":60},'
        '"sources":[{"root_id":"data","display_name":"data",'
        '"verified_size_gb":1}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard stop free space"):
        CorpusPolicy.load(invalid)


def _valid_payload() -> dict[str, object]:
    return CorpusPolicy.load(POLICY_PATH).model_dump(mode="json")


def test_spool_policy_rejects_an_object_larger_than_total_capacity() -> None:
    with pytest.raises(ValueError, match="object limit exceeds"):
        SpoolPolicy(
            max_total_bytes=1,
            max_object_bytes=2,
            max_active_objects=1,
            abandoned_ttl_seconds=1,
        )


def test_source_policy_rejects_material_type_count_drift() -> None:
    payload = _valid_payload()
    sources = payload["sources"]
    assert isinstance(sources, list)
    first = sources[0]
    assert isinstance(first, dict)
    first["physical_material_type_count"] = 1

    with pytest.raises(ValueError, match="material type count"):
        CorpusPolicy.model_validate(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"derived_data_target_bytes": 3, "derived_data_hard_cap_bytes": 2},
            "target exceeds",
        ),
        (
            {"filesystem_alert_ratio": 0.9, "filesystem_stop_ratio": 0.8},
            "alert ratio",
        ),
        (
            {
                "server_capacity_gb": 1,
                "minimum_free_bytes": 600_000_000,
                "hard_stop_free_bytes": 500_000_000,
                "derived_data_target_bytes": 500_000_000,
                "derived_data_hard_cap_bytes": 600_000_000,
            },
            "insufficient free space",
        ),
    ],
)
def test_policy_rejects_each_global_capacity_invariant(
    updates: dict[str, object],
    message: str,
) -> None:
    payload = _valid_payload()
    payload.update(updates)

    with pytest.raises(ValueError, match=message):
        CorpusPolicy.model_validate(payload)


def test_policy_rejects_spool_that_consumes_reserved_free_space() -> None:
    payload = _valid_payload()
    payload.update(
        {
            "server_capacity_gb": 1,
            "minimum_free_bytes": 600_000_000,
            "hard_stop_free_bytes": 500_000_000,
            "derived_data_target_bytes": 100_000_000,
            "derived_data_hard_cap_bytes": 200_000_000,
        }
    )
    spool = payload["spool"]
    assert isinstance(spool, dict)
    spool.update({"max_total_bytes": 600_000_000, "max_object_bytes": 1})

    with pytest.raises(ValueError, match="spool quota"):
        CorpusPolicy.model_validate(payload)


def test_policy_rejects_duplicate_roots_and_unknown_lookup() -> None:
    policy = CorpusPolicy.load(POLICY_PATH)
    with pytest.raises(KeyError, match="unknown corpus root"):
        policy.source("missing")

    payload = policy.model_dump(mode="json")
    sources = payload["sources"]
    assert isinstance(sources, list)
    duplicate = dict(sources[0])
    sources.append(duplicate)
    with pytest.raises(ValueError, match="root IDs must be unique"):
        CorpusPolicy.model_validate(payload)
