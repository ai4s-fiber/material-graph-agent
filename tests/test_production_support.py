from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from material_graph.knowledge import production_support
from material_graph.knowledge.mineru_client import MinerUBlock, MinerUParseResult
from material_graph.knowledge.models import SelectionDecision, SourceLocator
from material_graph.knowledge.policy import CorpusPolicy, SpoolPolicy
from material_graph.knowledge.production_support import (
    CapacitySamplingError,
    GenericEvidenceAssessmentProvider,
    LocalCapacitySnapshotProvider,
)
from material_graph.knowledge.spool import SpoolManager


def _corpus_policy(tmp_path: Path) -> CorpusPolicy:
    gib = 1024**3
    return CorpusPolicy(
        schema_version=1,
        server_capacity_gb=130,
        minimum_free_bytes=32 * gib,
        hard_stop_free_bytes=26 * gib,
        derived_data_target_bytes=75 * gib,
        derived_data_hard_cap_bytes=80 * gib,
        filesystem_alert_ratio=0.75,
        filesystem_stop_ratio=0.80,
        spool=SpoolPolicy(
            max_total_bytes=8 * gib,
            max_object_bytes=1 * gib,
            max_active_objects=4,
            abandoned_ttl_seconds=3600,
        ),
        sources=[
            {
                "root_id": "documents",
                "display_name": "Documents",
                "verified_size_gb": 1,
            }
        ],
    )


def test_generic_assessor_is_material_neutral_and_rejects_auxiliary_blocks() -> None:
    parsed = MinerUParseResult(
        batch_id="batch-1",
        task_id="task-1",
        filename="sample.pdf",
        parser_version="3.4.4",
        model_version="vlm",
        blocks=[
            MinerUBlock(
                block_type="text",
                text="The alloy reached 612 MPa after aging at 180 °C for two hours.",
                page=1,
                block_index=1,
                section="Results",
            ),
            MinerUBlock(
                block_type="footer",
                text="Journal footer",
                page=1,
                block_index=2,
            ),
            MinerUBlock(
                block_type="text",
                text="brief",
                page=1,
                block_index=3,
            ),
        ],
    )
    gap_id = uuid4()
    decision = SelectionDecision(
        source_id=uuid4(),
        selected=True,
        reason_code="active_evidence_gap",
        evidence_gap_id=gap_id,
        policy_version="test-v1",
    )

    assessments = asyncio.run(
        GenericEvidenceAssessmentProvider().assess(
            parsed,
            decision=decision,
            source_locator=SourceLocator(root_id="documents", relative_path="sample.pdf"),
        )
    )

    assert [item.block_index for item in assessments] == [1, 2, 3]
    assert assessments[0].accepted is True
    assert assessments[0].confidence >= 0.8
    assert assessments[0].evidence_gap_ids == [str(gap_id)]
    assert assessments[0].metadata == {
        "has_measurement_signal": True,
        "has_scientific_signal": True,
    }
    assert assessments[1].accepted is False
    assert assessments[1].confidence == 0
    assert assessments[2].accepted is False


def test_generic_assessor_rejects_invalid_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="minimum_text_characters"):
        GenericEvidenceAssessmentProvider(minimum_text_characters=0)

    parsed = MinerUParseResult(
        batch_id="batch-1",
        task_id="task-1",
        filename="sample.pdf",
        parser_version="3.4.4",
        model_version="vlm",
        blocks=[],
    )
    unselected = SelectionDecision(
        source_id=uuid4(),
        selected=False,
        reason_code="insufficient_metadata",
        policy_version="test-v1",
    )
    provider = GenericEvidenceAssessmentProvider()
    with pytest.raises(ValueError, match="selected source"):
        asyncio.run(
            provider.assess(
                parsed,
                decision=unselected,
                source_locator=SourceLocator(
                    root_id="documents",
                    relative_path="sample.pdf",
                ),
            )
        )

    selected = unselected.model_copy(update={"selected": True})
    with pytest.raises(TypeError, match="SourceLocator"):
        asyncio.run(
            provider.assess(
                parsed,
                decision=selected,
                source_locator="sample.pdf",  # type: ignore[arg-type]
            )
        )


def test_capacity_provider_samples_disk_without_following_symlinks(tmp_path: Path) -> None:
    derived = tmp_path / "runtime"
    spool_root = derived / "spool"
    spool_root.mkdir(parents=True)
    (derived / "safe.bin").write_bytes(b"x" * 32)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y" * 4096)
    try:
        (derived / "outside-link").symlink_to(outside)
    except OSError:
        pass
    policy = _corpus_policy(tmp_path)
    spool = SpoolManager(
        spool_root,
        policy.spool,
        capacity_probe=lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )
    provider = LocalCapacitySnapshotProvider(
        derived_root=derived,
        spool_root=spool_root,
        spool=spool,
        corpus_policy=policy,
        memory_pressure_probe=lambda: True,
    )

    capacity = provider.capacity_snapshot()
    admission = asyncio.run(provider.snapshot())

    assert 32 <= capacity.derived_active_bytes < 4096
    assert capacity.derived_projected_bytes == (
        capacity.derived_active_bytes + policy.spool.max_object_bytes
    )
    assert capacity.filesystem_total_bytes > 0
    assert admission.spool_max_bytes == policy.spool.max_total_bytes
    assert admission.derived_bytes == capacity.derived_active_bytes
    assert admission.memory_pressure is True


def test_capacity_provider_fails_closed_when_scan_budget_is_exceeded(tmp_path: Path) -> None:
    derived = tmp_path / "runtime"
    spool_root = derived / "spool"
    spool_root.mkdir(parents=True)
    (derived / "one").write_text("1", encoding="utf-8")
    (derived / "two").write_text("2", encoding="utf-8")
    policy = _corpus_policy(tmp_path)
    spool = SpoolManager(
        spool_root,
        policy.spool,
        capacity_probe=lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )
    provider = LocalCapacitySnapshotProvider(
        derived_root=derived,
        spool_root=spool_root,
        spool=spool,
        corpus_policy=policy,
        max_scan_entries=1,
    )

    with pytest.raises(CapacitySamplingError, match="capacity_sampling_failed"):
        provider.capacity_snapshot()


def test_capacity_provider_rejects_unsafe_roots_and_probe_failures(tmp_path: Path) -> None:
    derived = tmp_path / "runtime"
    spool_root = derived / "spool"
    spool_root.mkdir(parents=True)
    policy = _corpus_policy(tmp_path)
    spool = SpoolManager(
        spool_root,
        policy.spool,
        capacity_probe=lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )

    with pytest.raises(ValueError, match="child of derived_root"):
        LocalCapacitySnapshotProvider(
            derived_root=derived,
            spool_root=derived,
            spool=spool,
            corpus_policy=policy,
        )
    with pytest.raises(ValueError, match="capacity roots must exist"):
        LocalCapacitySnapshotProvider(
            derived_root=tmp_path / "missing",
            spool_root=tmp_path / "missing" / "spool",
            spool=spool,
            corpus_policy=policy,
        )
    with pytest.raises(ValueError, match="max_scan_entries"):
        LocalCapacitySnapshotProvider(
            derived_root=derived,
            spool_root=spool_root,
            spool=spool,
            corpus_policy=policy,
            max_scan_entries=0,
        )

    provider = LocalCapacitySnapshotProvider(
        derived_root=derived,
        spool_root=spool_root,
        spool=spool,
        corpus_policy=policy,
        memory_pressure_probe=lambda: (_ for _ in ()).throw(RuntimeError("probe detail")),
    )
    with pytest.raises(CapacitySamplingError, match="capacity_sampling_failed"):
        asyncio.run(provider.snapshot())


def test_capacity_provider_converts_filesystem_errors_to_stable_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    derived = tmp_path / "runtime"
    spool_root = derived / "spool"
    spool_root.mkdir(parents=True)
    policy = _corpus_policy(tmp_path)
    spool = SpoolManager(
        spool_root,
        policy.spool,
        capacity_probe=lambda: (_ for _ in ()).throw(AssertionError("unused")),
    )
    provider = LocalCapacitySnapshotProvider(
        derived_root=derived,
        spool_root=spool_root,
        spool=spool,
        corpus_policy=policy,
    )
    monkeypatch.setattr(
        production_support.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(OSError("filesystem detail")),
    )

    with pytest.raises(CapacitySamplingError, match="capacity_sampling_failed"):
        provider.capacity_snapshot()
