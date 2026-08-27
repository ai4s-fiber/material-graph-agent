"""Quota-bound transient storage for selected remote source bodies.

The spool is the only place where an original source may exist locally.  Every
reservation owns one directory and cleanup is enforced by an async context
manager across success, failure, cancellation, and parser-output creation.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .policy import SpoolPolicy


class SpoolError(RuntimeError):
    """Base error for transient source storage."""


class SpoolCapacityError(SpoolError):
    """Raised before source bytes are opened when an admission gate is closed."""


class SpoolIntegrityError(SpoolError):
    """Raised when a remote stream violates its reserved size/chunk contract."""


class CapacitySnapshot(BaseModel):
    """Global capacity facts sampled immediately before a spool reservation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filesystem_total_bytes: int = Field(gt=0)
    filesystem_used_bytes: int = Field(ge=0)
    filesystem_free_bytes: int = Field(ge=0)
    derived_active_bytes: int = Field(ge=0)
    derived_projected_bytes: int = Field(ge=0)
    minimum_free_bytes: int = Field(gt=0)
    hard_stop_free_bytes: int = Field(gt=0)
    derived_target_bytes: int = Field(gt=0)
    derived_hard_cap_bytes: int = Field(gt=0)
    filesystem_alert_ratio: float = Field(gt=0, lt=1)
    filesystem_stop_ratio: float = Field(gt=0, lt=1)

    @property
    def filesystem_used_ratio(self) -> float:
        return self.filesystem_used_bytes / self.filesystem_total_bytes

    @model_validator(mode="after")
    def validate_thresholds(self) -> "CapacitySnapshot":
        if self.filesystem_used_bytes + self.filesystem_free_bytes > self.filesystem_total_bytes:
            raise ValueError("filesystem used/free bytes exceed total bytes")
        if self.hard_stop_free_bytes > self.minimum_free_bytes:
            raise ValueError("hard-stop free bytes exceed warning threshold")
        if self.derived_target_bytes > self.derived_hard_cap_bytes:
            raise ValueError("derived-data target exceeds hard cap")
        if self.filesystem_alert_ratio > self.filesystem_stop_ratio:
            raise ValueError("filesystem alert ratio exceeds stop ratio")
        if self.derived_projected_bytes < self.derived_active_bytes:
            raise ValueError("projected derived bytes are below active bytes")
        return self


@dataclass(frozen=True, slots=True)
class TemporarySource:
    reservation_id: str
    source_id: str
    path: Path
    parser_output_dir: Path
    expected_bytes: int
    actual_bytes: int
    content_sha256: str
    pressure_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Reservation:
    reservation_id: str
    source_id: str
    directory: Path
    source_path: Path
    parser_output_dir: Path
    expected_bytes: int
    pressure_warnings: tuple[str, ...]


class SpoolManager:
    """Atomic admission, streaming materialization, and fail-safe cleanup."""

    _MANIFEST_NAME = ".reservation.json"
    _DIRECTORY_PREFIX = "spool-"

    def __init__(
        self,
        root: str | Path,
        policy: SpoolPolicy,
        *,
        capacity_probe: Callable[[], CapacitySnapshot],
        max_chunk_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self._capacity_probe = capacity_probe
        self._max_chunk_bytes = max_chunk_bytes
        self._lock = asyncio.Lock()
        self._reservations: dict[str, int] = {}

    @property
    def active_reservations(self) -> int:
        return len(self._reservations)

    @property
    def reserved_bytes(self) -> int:
        return sum(self._reservations.values())

    @asynccontextmanager
    async def materialize(
        self,
        *,
        source_id: str,
        expected_bytes: int,
        chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[TemporarySource]:
        """Reserve capacity, stream exactly one source, and always remove it."""

        reservation = await self._reserve(source_id=source_id, expected_bytes=expected_bytes)
        try:
            temporary_source = await self._write_stream(reservation, chunks)
            yield temporary_source
        finally:
            try:
                self._remove_owned_directory(reservation.directory)
            finally:
                await self._release(reservation.reservation_id)

    async def _reserve(self, *, source_id: str, expected_bytes: int) -> _Reservation:
        if expected_bytes <= 0:
            raise SpoolCapacityError("expected source size must be positive")
        if expected_bytes > self.policy.max_object_bytes:
            raise SpoolCapacityError("source exceeds spool object limit")

        snapshot = self._capacity_probe()
        warnings = self._validate_global_capacity(snapshot)

        async with self._lock:
            if len(self._reservations) >= self.policy.max_active_objects:
                raise SpoolCapacityError("spool active object limit reached")
            if self.reserved_bytes + expected_bytes > self.policy.max_total_bytes:
                raise SpoolCapacityError("spool aggregate reservation limit reached")

            reservation_id = uuid4().hex
            directory = self.root / f"{self._DIRECTORY_PREFIX}{reservation_id}"
            directory.mkdir(mode=0o700)
            source_path = directory / "source.bin"
            parser_output_dir = directory / "parser-output"
            parser_output_dir.mkdir(mode=0o700)
            manifest = {
                "reservation_id": reservation_id,
                "source_id": str(source_id),
                "expected_bytes": expected_bytes,
                "created_at_epoch": time.time(),
            }
            (directory / self._MANIFEST_NAME).write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            self._reservations[reservation_id] = expected_bytes

        return _Reservation(
            reservation_id=reservation_id,
            source_id=str(source_id),
            directory=directory,
            source_path=source_path,
            parser_output_dir=parser_output_dir,
            expected_bytes=expected_bytes,
            pressure_warnings=warnings,
        )

    def _validate_global_capacity(self, snapshot: CapacitySnapshot) -> tuple[str, ...]:
        if (
            snapshot.filesystem_used_ratio >= snapshot.filesystem_stop_ratio
            or snapshot.filesystem_free_bytes < snapshot.hard_stop_free_bytes
        ):
            raise SpoolCapacityError("filesystem hard stop prevents source admission")
        if snapshot.derived_active_bytes >= snapshot.derived_hard_cap_bytes:
            raise SpoolCapacityError("derived-data hard stop prevents source admission")
        if snapshot.derived_projected_bytes > snapshot.derived_target_bytes:
            raise SpoolCapacityError("derived-data target would be exceeded")

        warnings: list[str] = []
        if snapshot.filesystem_used_ratio >= snapshot.filesystem_alert_ratio:
            warnings.append("filesystem_alert_ratio")
        if snapshot.filesystem_free_bytes < snapshot.minimum_free_bytes:
            warnings.append("minimum_free_bytes")
        return tuple(warnings)

    async def _write_stream(
        self,
        reservation: _Reservation,
        chunks: AsyncIterator[bytes],
    ) -> TemporarySource:
        digest = sha256()
        actual_bytes = 0
        with reservation.source_path.open("xb") as target:
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise SpoolIntegrityError("source stream yielded a non-bytes chunk")
                if not chunk:
                    continue
                if len(chunk) > self._max_chunk_bytes:
                    raise SpoolIntegrityError("source chunk exceeds configured chunk limit")
                actual_bytes += len(chunk)
                if actual_bytes > reservation.expected_bytes:
                    raise SpoolIntegrityError("remote source size changed while streaming")
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())

        if actual_bytes != reservation.expected_bytes:
            raise SpoolIntegrityError("remote source size changed while streaming")

        return TemporarySource(
            reservation_id=reservation.reservation_id,
            source_id=reservation.source_id,
            path=reservation.source_path,
            parser_output_dir=reservation.parser_output_dir,
            expected_bytes=reservation.expected_bytes,
            actual_bytes=actual_bytes,
            content_sha256=digest.hexdigest(),
            pressure_warnings=reservation.pressure_warnings,
        )

    async def _release(self, reservation_id: str) -> None:
        async with self._lock:
            self._reservations.pop(reservation_id, None)

    def _remove_owned_directory(self, directory: Path) -> None:
        root = self.root.resolve()
        if directory.is_symlink():
            raise SpoolIntegrityError("refusing to follow a spool reservation symlink")
        candidate = directory.resolve(strict=False)
        if candidate == root or root not in candidate.parents:
            raise SpoolIntegrityError("refusing to remove a path outside the spool root")
        if directory.exists():
            shutil.rmtree(directory)

    def sweep_abandoned(self, *, now: float | None = None) -> list[str]:
        """Remove stale owned reservation directories and leave unrelated files alone."""

        current_time = time.time() if now is None else now
        removed: list[str] = []
        for child in sorted(self.root.iterdir(), key=lambda item: item.name):
            if (
                not child.name.startswith(self._DIRECTORY_PREFIX)
                or not child.is_dir()
                or child.is_symlink()
                or not (child / self._MANIFEST_NAME).is_file()
            ):
                continue
            age_seconds = current_time - child.stat().st_mtime
            if age_seconds < self.policy.abandoned_ttl_seconds:
                continue
            self._remove_owned_directory(child)
            removed.append(child.name)
        return removed
