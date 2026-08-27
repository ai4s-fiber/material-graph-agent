from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from material_graph.knowledge.policy import SpoolPolicy
from material_graph.knowledge.spool import (
    CapacitySnapshot,
    SpoolCapacityError,
    SpoolIntegrityError,
    SpoolManager,
)


async def _stream(*chunks: bytes):
    for chunk in chunks:
        yield chunk


def _policy(**overrides: int) -> SpoolPolicy:
    values = {
        "max_total_bytes": 32,
        "max_object_bytes": 16,
        "max_active_objects": 2,
        "abandoned_ttl_seconds": 60,
    }
    values.update(overrides)
    return SpoolPolicy.model_validate(values)


def _healthy_capacity() -> CapacitySnapshot:
    return CapacitySnapshot(
        filesystem_total_bytes=1_000,
        filesystem_used_bytes=500,
        filesystem_free_bytes=500,
        derived_active_bytes=100,
        derived_projected_bytes=100,
        minimum_free_bytes=300,
        hard_stop_free_bytes=200,
        derived_target_bytes=700,
        derived_hard_cap_bytes=800,
        filesystem_alert_ratio=0.75,
        filesystem_stop_ratio=0.80,
    )


def test_materialize_streams_hashes_and_cleans_all_transient_artifacts(tmp_path: Path) -> None:
    manager = SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity)

    async def run() -> str:
        async with manager.materialize(
            source_id="source-1",
            expected_bytes=6,
            chunks=_stream(b"", b"abc", b"def"),
        ) as source:
            assert source.path.read_bytes() == b"abcdef"
            assert source.actual_bytes == 6
            assert source.content_sha256 == (
                "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
            )
            (source.parser_output_dir / "complete-output.json").write_text(
                "transient",
                encoding="utf-8",
            )
            return source.content_sha256

    assert asyncio.run(run())
    assert manager.active_reservations == 0
    assert manager.reserved_bytes == 0
    assert list(tmp_path.iterdir()) == []


def test_materialize_cleans_on_exception_and_cancellation(tmp_path: Path) -> None:
    manager = SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity)

    async def fail() -> None:
        async with manager.materialize(
            source_id="source-fail",
            expected_bytes=3,
            chunks=_stream(b"abc"),
        ):
            raise RuntimeError("parser failed")

    async def cancel() -> None:
        async with manager.materialize(
            source_id="source-cancel",
            expected_bytes=3,
            chunks=_stream(b"abc"),
        ):
            raise asyncio.CancelledError

    with pytest.raises(RuntimeError, match="parser failed"):
        asyncio.run(fail())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel())
    assert manager.active_reservations == 0
    assert list(tmp_path.iterdir()) == []


def test_object_active_and_aggregate_quotas_are_reserved_before_streaming(tmp_path: Path) -> None:
    manager = SpoolManager(
        tmp_path,
        _policy(max_total_bytes=12, max_object_bytes=8, max_active_objects=1),
        capacity_probe=_healthy_capacity,
    )

    async def run() -> None:
        with pytest.raises(SpoolCapacityError, match="object limit"):
            async with manager.materialize(
                source_id="too-large",
                expected_bytes=9,
                chunks=_stream(b"123456789"),
            ):
                pass

        async with manager.materialize(
            source_id="first",
            expected_bytes=8,
            chunks=_stream(b"12345678"),
        ):
            with pytest.raises(SpoolCapacityError, match="active object limit"):
                async with manager.materialize(
                    source_id="second",
                    expected_bytes=5,
                    chunks=_stream(b"12345"),
                ):
                    pass

    asyncio.run(run())


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (
            CapacitySnapshot(
                filesystem_total_bytes=1_000,
                filesystem_used_bytes=810,
                filesystem_free_bytes=190,
                derived_active_bytes=100,
                derived_projected_bytes=100,
                minimum_free_bytes=300,
                hard_stop_free_bytes=200,
                derived_target_bytes=700,
                derived_hard_cap_bytes=800,
                filesystem_alert_ratio=0.75,
                filesystem_stop_ratio=0.80,
            ),
            "filesystem hard stop",
        ),
        (
            CapacitySnapshot(
                filesystem_total_bytes=1_000,
                filesystem_used_bytes=500,
                filesystem_free_bytes=500,
                derived_active_bytes=801,
                derived_projected_bytes=801,
                minimum_free_bytes=300,
                hard_stop_free_bytes=200,
                derived_target_bytes=700,
                derived_hard_cap_bytes=800,
                filesystem_alert_ratio=0.75,
                filesystem_stop_ratio=0.80,
            ),
            "derived-data hard stop",
        ),
        (
            CapacitySnapshot(
                filesystem_total_bytes=1_000,
                filesystem_used_bytes=500,
                filesystem_free_bytes=500,
                derived_active_bytes=600,
                derived_projected_bytes=701,
                minimum_free_bytes=300,
                hard_stop_free_bytes=200,
                derived_target_bytes=700,
                derived_hard_cap_bytes=800,
                filesystem_alert_ratio=0.75,
                filesystem_stop_ratio=0.80,
            ),
            "derived-data target",
        ),
    ],
)
def test_global_capacity_hard_gates_reject_before_body_open(
    tmp_path: Path,
    snapshot: CapacitySnapshot,
    message: str,
) -> None:
    opened = False

    async def body():
        nonlocal opened
        opened = True
        yield b"abc"

    manager = SpoolManager(tmp_path, _policy(), capacity_probe=lambda: snapshot)

    async def run() -> None:
        with pytest.raises(SpoolCapacityError, match=message):
            async with manager.materialize(
                source_id="blocked",
                expected_bytes=3,
                chunks=body(),
            ):
                pass

    asyncio.run(run())
    assert opened is False
    assert list(tmp_path.iterdir()) == []


def test_pressure_warning_is_exposed_but_does_not_cross_hard_gate(tmp_path: Path) -> None:
    snapshot = _healthy_capacity().model_copy(
        update={"filesystem_used_bytes": 760, "filesystem_free_bytes": 240}
    )
    manager = SpoolManager(tmp_path, _policy(), capacity_probe=lambda: snapshot)

    async def run() -> tuple[str, ...]:
        async with manager.materialize(
            source_id="warning",
            expected_bytes=3,
            chunks=_stream(b"abc"),
        ) as source:
            return source.pressure_warnings

    warnings = asyncio.run(run())
    assert "filesystem_alert_ratio" in warnings
    assert "minimum_free_bytes" in warnings


def test_size_or_chunk_contract_violation_cleans_reservation(tmp_path: Path) -> None:
    manager = SpoolManager(
        tmp_path,
        _policy(),
        capacity_probe=_healthy_capacity,
        max_chunk_bytes=4,
    )

    async def oversized_chunk() -> None:
        async with manager.materialize(
            source_id="chunk",
            expected_bytes=5,
            chunks=_stream(b"12345"),
        ):
            pass

    async def short_stream() -> None:
        async with manager.materialize(
            source_id="short",
            expected_bytes=4,
            chunks=_stream(b"123"),
        ):
            pass

    with pytest.raises(SpoolIntegrityError, match="chunk exceeds"):
        asyncio.run(oversized_chunk())
    with pytest.raises(SpoolIntegrityError, match="size changed"):
        asyncio.run(short_stream())
    assert list(tmp_path.iterdir()) == []


def test_sweeper_removes_only_abandoned_owned_reservations(tmp_path: Path) -> None:
    manager = SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity)
    abandoned = tmp_path / "spool-abandoned"
    abandoned.mkdir()
    (abandoned / ".reservation.json").write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "keep-me"
    unrelated.mkdir()
    old = time.time() - 120
    os.utime(abandoned, (old, old))
    os.utime(unrelated, (old, old))

    removed = manager.sweep_abandoned(now=time.time())

    assert removed == ["spool-abandoned"]
    assert not abandoned.exists()
    assert unrelated.exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"filesystem_used_bytes": 800, "filesystem_free_bytes": 300}, "used/free"),
        ({"hard_stop_free_bytes": 400}, "hard-stop"),
        ({"derived_target_bytes": 900}, "target exceeds"),
        ({"filesystem_alert_ratio": 0.9}, "alert ratio"),
        ({"derived_projected_bytes": 50}, "below active"),
    ],
)
def test_capacity_snapshot_rejects_each_inconsistent_threshold(
    updates: dict[str, int | float],
    message: str,
) -> None:
    payload = _healthy_capacity().model_dump(mode="python")
    payload.update(updates)

    with pytest.raises(ValueError, match=message):
        CapacitySnapshot.model_validate(payload)


def test_spool_rejects_invalid_chunk_limit_and_nonpositive_source_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_chunk_bytes"):
        SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity, max_chunk_bytes=0)

    manager = SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity)

    async def run() -> None:
        with pytest.raises(SpoolCapacityError, match="size must be positive"):
            async with manager.materialize(
                source_id="empty",
                expected_bytes=0,
                chunks=_stream(),
            ):
                pass

    asyncio.run(run())


def test_aggregate_quota_is_checked_separately_from_active_limit(tmp_path: Path) -> None:
    manager = SpoolManager(
        tmp_path,
        _policy(max_total_bytes=12, max_object_bytes=8, max_active_objects=2),
        capacity_probe=_healthy_capacity,
    )

    async def run() -> None:
        async with manager.materialize(
            source_id="first",
            expected_bytes=8,
            chunks=_stream(b"12345678"),
        ):
            with pytest.raises(SpoolCapacityError, match="aggregate reservation"):
                async with manager.materialize(
                    source_id="second",
                    expected_bytes=5,
                    chunks=_stream(b"12345"),
                ):
                    pass

    asyncio.run(run())


def test_stream_rejects_non_bytes_and_cumulative_overrun(tmp_path: Path) -> None:
    manager = SpoolManager(
        tmp_path,
        _policy(),
        capacity_probe=_healthy_capacity,
        max_chunk_bytes=4,
    )

    async def non_bytes():
        yield "not-bytes"

    async def reject_non_bytes() -> None:
        async with manager.materialize(
            source_id="wrong-type",
            expected_bytes=1,
            chunks=non_bytes(),  # type: ignore[arg-type]
        ):
            pass

    async def reject_overrun() -> None:
        async with manager.materialize(
            source_id="overrun",
            expected_bytes=4,
            chunks=_stream(b"123", b"45"),
        ):
            pass

    with pytest.raises(SpoolIntegrityError, match="non-bytes"):
        asyncio.run(reject_non_bytes())
    with pytest.raises(SpoolIntegrityError, match="size changed"):
        asyncio.run(reject_overrun())
    assert list(tmp_path.iterdir()) == []


def test_cleanup_refuses_outside_paths_and_ignores_absent_owned_path(tmp_path: Path) -> None:
    root = tmp_path / "spool"
    manager = SpoolManager(root, _policy(), capacity_probe=_healthy_capacity)

    with pytest.raises(SpoolIntegrityError, match="outside the spool root"):
        manager._remove_owned_directory(tmp_path / "outside")

    manager._remove_owned_directory(root / "spool-already-absent")


def test_sweeper_keeps_recent_owned_reservation_with_default_clock(tmp_path: Path) -> None:
    manager = SpoolManager(tmp_path, _policy(), capacity_probe=_healthy_capacity)
    recent = tmp_path / "spool-recent"
    recent.mkdir()
    (recent / ".reservation.json").write_text("{}", encoding="utf-8")

    assert manager.sweep_abandoned() == []
    assert recent.exists()
