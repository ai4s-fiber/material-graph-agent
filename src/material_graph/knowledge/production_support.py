"""Production-only, content-safe support for the knowledge worker."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
import re
import shutil

from .concurrency import AdmissionSnapshot
from .ingestion import EvidenceAssessmentProvider
from .mineru_client import MinerUParseResult
from .models import SelectionDecision, SourceLocator
from .policy import CorpusPolicy
from .retention import BlockEvidenceAssessment
from .spool import CapacitySnapshot, SpoolManager


_AUXILIARY_BLOCK_TYPES = frozenset({"header", "footer", "page_number", "aside_text"})
_MEASUREMENT_SIGNAL = re.compile(
    r"(?:\d(?:[\d.,]*\d)?\s*(?:%|°\s*[CFK]|MPa|GPa|Pa|kPa|mPa|"
    r"g\s*/\s*cm(?:3|³)|kg\s*/\s*m(?:3|³)|W\s*/\s*m|S\s*/\s*m|"
    r"eV|nm|µm|um|mm|cm|mL|mol|h|hours?|min|s)\b)",
    re.IGNORECASE,
)
_SCIENTIFIC_SIGNAL = re.compile(
    r"\b(?:alloy|ceramic|composite|fiber|material|membrane|metal|polymer|resin|"
    r"sample|specimen|synthesis|anneal|aging|temperature|strength|modulus|"
    r"conductivity|dielectric|viscosity|crystallinity)\b|"
    r"(?:材料|合金|陶瓷|复合|纤维|聚合物|树脂|样品|制备|合成|退火|温度|强度|"
    r"模量|电导率|介电|黏度|粘度|结晶)",
    re.IGNORECASE,
)


class CapacitySamplingError(RuntimeError):
    """Stable fail-closed signal for an incomplete local capacity sample."""

    def __init__(self) -> None:
        super().__init__("capacity_sampling_failed")


class GenericEvidenceAssessmentProvider(EvidenceAssessmentProvider):
    """Retain bounded substantive blocks without assuming one material family."""

    def __init__(self, *, minimum_text_characters: int = 16) -> None:
        if minimum_text_characters < 1:
            raise ValueError("minimum_text_characters must be positive")
        self._minimum_text_characters = minimum_text_characters

    async def assess(
        self,
        parsed: MinerUParseResult,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
    ) -> list[BlockEvidenceAssessment]:
        if not decision.selected:
            raise ValueError("evidence assessment requires a selected source")
        if not isinstance(source_locator, SourceLocator):
            raise TypeError("source_locator must be a SourceLocator")
        gap_ids = [str(decision.evidence_gap_id)] if decision.evidence_gap_id else []
        assessments: list[BlockEvidenceAssessment] = []
        for block in parsed.blocks:
            block_type = block.block_type.casefold()
            if block_type in _AUXILIARY_BLOCK_TYPES:
                assessments.append(
                    BlockEvidenceAssessment(
                        block_index=block.block_index,
                        accepted=False,
                        confidence=0,
                        retention_reason="auxiliary_block",
                        evidence_gap_ids=gap_ids,
                    )
                )
                continue

            text = " ".join(block.text.split())
            substantive = len(text) >= self._minimum_text_characters
            has_measurement = bool(_MEASUREMENT_SIGNAL.search(text))
            has_scientific = bool(_SCIENTIFIC_SIGNAL.search(text))
            confidence = 0.2
            if substantive:
                confidence = 0.65
                confidence += 0.15 if has_measurement else 0
                confidence += 0.10 if has_scientific else 0
                confidence += 0.05 if block_type == "table" else 0
                confidence += 0.05 if block.section else 0
            assessments.append(
                BlockEvidenceAssessment(
                    block_index=block.block_index,
                    accepted=substantive,
                    confidence=min(0.95, confidence),
                    retention_reason=(
                        "substantive_evidence_candidate" if substantive else "insufficient_content"
                    ),
                    evidence_gap_ids=gap_ids,
                    metadata={
                        "has_measurement_signal": has_measurement,
                        "has_scientific_signal": has_scientific,
                    },
                )
            )
        return assessments


def _default_memory_pressure_probe() -> bool:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, separator, raw = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                values[name] = int(raw.strip().split()[0])
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        return total > 0 and available / total < 0.10
    except (OSError, UnicodeError, ValueError):
        return False


class LocalCapacitySnapshotProvider:
    """Sample actual local storage facts without reading file content."""

    def __init__(
        self,
        *,
        derived_root: str | Path,
        spool_root: str | Path,
        spool: SpoolManager,
        corpus_policy: CorpusPolicy,
        max_scan_entries: int = 1_000_000,
        memory_pressure_probe: Callable[[], bool] = _default_memory_pressure_probe,
    ) -> None:
        derived = Path(derived_root).resolve()
        spool_path = Path(spool_root).resolve()
        if derived == spool_path or derived not in spool_path.parents:
            raise ValueError("spool_root must be a child of derived_root")
        if not derived.is_dir() or not spool_path.is_dir():
            raise ValueError("capacity roots must exist")
        if max_scan_entries < 1:
            raise ValueError("max_scan_entries must be positive")
        self._derived_root = derived
        self._spool_root = spool_path
        self._spool = spool
        self._policy = corpus_policy
        self._max_scan_entries = max_scan_entries
        self._memory_pressure_probe = memory_pressure_probe

    def _directory_size(self, root: Path) -> int:
        total = 0
        entries_seen = 0
        pending = [root]
        try:
            while pending:
                current = pending.pop()
                with os.scandir(current) as entries:
                    for entry in entries:
                        entries_seen += 1
                        if entries_seen > self._max_scan_entries:
                            raise CapacitySamplingError()
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
            return total
        except CapacitySamplingError:
            raise
        except OSError:
            raise CapacitySamplingError() from None

    def _sample(self) -> tuple[CapacitySnapshot, int]:
        try:
            usage = shutil.disk_usage(self._derived_root)
            derived_bytes = self._directory_size(self._derived_root)
            spool_bytes = max(
                self._directory_size(self._spool_root),
                self._spool.reserved_bytes,
            )
        except CapacitySamplingError:
            raise
        except OSError:
            raise CapacitySamplingError() from None
        policy = self._policy
        return (
            CapacitySnapshot(
                filesystem_total_bytes=usage.total,
                filesystem_used_bytes=usage.used,
                filesystem_free_bytes=usage.free,
                derived_active_bytes=derived_bytes,
                derived_projected_bytes=derived_bytes + policy.spool.max_object_bytes,
                minimum_free_bytes=policy.minimum_free_bytes,
                hard_stop_free_bytes=policy.hard_stop_free_bytes,
                derived_target_bytes=policy.derived_data_target_bytes,
                derived_hard_cap_bytes=policy.derived_data_hard_cap_bytes,
                filesystem_alert_ratio=policy.filesystem_alert_ratio,
                filesystem_stop_ratio=policy.filesystem_stop_ratio,
            ),
            spool_bytes,
        )

    def capacity_snapshot(self) -> CapacitySnapshot:
        return self._sample()[0]

    async def snapshot(self) -> AdmissionSnapshot:
        capacity, spool_bytes = await asyncio.to_thread(self._sample)
        try:
            memory_pressure = bool(self._memory_pressure_probe())
        except Exception:
            raise CapacitySamplingError() from None
        return AdmissionSnapshot(
            spool_used_bytes=spool_bytes,
            spool_max_bytes=self._policy.spool.max_total_bytes,
            filesystem_used_ratio=capacity.filesystem_used_ratio,
            free_bytes=capacity.filesystem_free_bytes,
            derived_bytes=capacity.derived_active_bytes,
            queues={},
            memory_pressure=memory_pressure,
        )


__all__ = [
    "CapacitySamplingError",
    "GenericEvidenceAssessmentProvider",
    "LocalCapacitySnapshotProvider",
]
