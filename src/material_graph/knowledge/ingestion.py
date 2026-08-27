"""Resumable, evidence-only orchestration for selected remote sources.

Rejected sources never cross the remote-body boundary. For selected sources,
the original and complete parser output exist only inside the spool context;
only deterministic ``EvidenceFragment`` records are durable.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from .lightrag_client import (
    LightRAGClient,
    LightRAGError,
    LightRAGForbiddenOperation,
    LightRAGPollingTimeout,
    LightRAGProtocolError,
    LightRAGRequestError,
    LightRAGSourceMappingConflict,
)
from .lightrag_models import LightRAGInsertResult
from .mineru_client import MinerUClient, MinerUError, MinerUParseResult
from .models import EvidenceFragment, SelectionDecision, SourceLocator
from .processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    ProcessingStateMachine,
    SourceLifecycleStatus,
)
from .remote_reader import RemoteSourceReader, normalize_identifier
from .retention import BlockEvidenceAssessment, EvidenceSelector
from .spool import SpoolCapacityError, SpoolError, SpoolIntegrityError, SpoolManager


_SAFE_CATEGORY = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,99}$")
_TERMINAL_JOBS = frozenset(
    {
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED_PERMANENT,
        IngestionJobStatus.CANCELLED,
    }
)


def _same_evidence_identity(left: EvidenceFragment, right: EvidenceFragment) -> bool:
    """Ignore retry-variant assessment/provider metadata; first durable write wins."""

    return (
        left.fragment_id == right.fragment_id
        and left.source_id == right.source_id
        and left.text == right.text
        and left.locator == right.locator
        and left.content_sha256 == right.content_sha256
        and left.parser_name == right.parser_name
        and left.parser_version == right.parser_version
        and left.embedding_generation_id == right.embedding_generation_id
    )


class CheckpointRepository(Protocol):
    """Durable, credential-free checkpoint boundary."""

    async def load(self, idempotency_key: str) -> ProcessingCheckpoint | None: ...

    async def save(self, checkpoint: ProcessingCheckpoint) -> None: ...


class EvidenceRepository(Protocol):
    """Durable repository accepting retained fragments and nothing raw."""

    async def persist_many(
        self,
        source_id: UUID,
        fragments: Sequence[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> None: ...

    async def list_for_source(
        self,
        source_id: UUID,
        *,
        idempotency_key: str,
    ) -> list[EvidenceFragment]: ...


class EvidenceAssessmentProvider(Protocol):
    """Task-aware assessment boundary over transient normalized blocks."""

    async def assess(
        self,
        parsed: MinerUParseResult,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
    ) -> Sequence[BlockEvidenceAssessment]: ...


class InMemoryCheckpointRepository:
    """Atomic in-memory checkpoint fake used by tests and local dry-runs."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, ProcessingCheckpoint] = {}
        self._history: list[ProcessingCheckpoint] = []
        self._lock = asyncio.Lock()

    async def load(self, idempotency_key: str) -> ProcessingCheckpoint | None:
        async with self._lock:
            value = self._checkpoints.get(idempotency_key)
            return None if value is None else value.model_copy(deep=True)

    async def save(self, checkpoint: ProcessingCheckpoint) -> None:
        candidate = ProcessingCheckpoint.model_validate(checkpoint.model_dump(mode="python"))
        async with self._lock:
            existing = self._checkpoints.get(candidate.idempotency_key)
            if existing is not None and existing.source_id != candidate.source_id:
                raise ValueError("checkpoint source identity conflict")
            stored = candidate.model_copy(deep=True)
            self._checkpoints[candidate.idempotency_key] = stored
            self._history.append(stored)

    async def history(
        self,
        source_id: UUID | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> list[ProcessingCheckpoint]:
        async with self._lock:
            return [
                item.model_copy(deep=True)
                for item in self._history
                if (source_id is None or item.source_id == source_id)
                and (idempotency_key is None or item.idempotency_key == idempotency_key)
            ]


class InMemoryEvidenceRepository:
    """Idempotent in-memory evidence fake with conflict detection."""

    def __init__(self) -> None:
        self._sources: dict[tuple[UUID, str], dict[UUID, EvidenceFragment]] = {}
        self._lock = asyncio.Lock()

    async def persist_many(
        self,
        source_id: UUID,
        fragments: Sequence[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> None:
        if not idempotency_key.strip():
            raise ValueError("evidence idempotency key is required")
        candidates: dict[UUID, EvidenceFragment] = {}
        for fragment in fragments:
            if not isinstance(fragment, EvidenceFragment):
                raise TypeError("evidence repository accepts EvidenceFragment instances only")
            if fragment.source_id != source_id:
                raise ValueError("evidence source_id does not match repository key")
            existing = candidates.get(fragment.fragment_id)
            if existing is not None and not _same_evidence_identity(existing, fragment):
                raise ValueError("conflicting evidence fragment identity")
            candidates[fragment.fragment_id] = fragment

        key = (source_id, idempotency_key)
        async with self._lock:
            current = self._sources.setdefault(key, {})
            for fragment_id, fragment in candidates.items():
                existing = current.get(fragment_id)
                if existing is not None and not _same_evidence_identity(existing, fragment):
                    raise ValueError("conflicting durable evidence fragment")
            for fragment_id, fragment in candidates.items():
                current.setdefault(fragment_id, fragment.model_copy(deep=True))

    async def list_for_source(
        self,
        source_id: UUID,
        *,
        idempotency_key: str,
    ) -> list[EvidenceFragment]:
        async with self._lock:
            values = self._sources.get((source_id, idempotency_key), {})
            return [
                values[fragment_id].model_copy(deep=True)
                for fragment_id in sorted(values, key=lambda item: item.hex)
            ]


class FailureDisposition(StrEnum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    CANCELLED = "cancelled"


class IngestionPipelineError(RuntimeError):
    """Sanitized failure safe for logs, APIs, and execution traces."""

    def __init__(
        self,
        *,
        disposition: FailureDisposition,
        stage: IngestionStage,
        category: str,
    ) -> None:
        self.disposition = disposition
        self.stage = stage
        self.category = _normalize_category(category)
        super().__init__(
            f"knowledge ingestion {disposition.value} at {stage.value}: {self.category}"
        )


class IngestionCancelledError(asyncio.CancelledError):
    """Cancellation classification preserving asyncio cancellation semantics."""

    def __init__(self, *, stage: IngestionStage) -> None:
        self.disposition = FailureDisposition.CANCELLED
        self.stage = stage
        self.category = "cancelled"
        super().__init__(f"knowledge ingestion cancelled at {stage.value}")


class IngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Literal["skipped", "indexed", "parsed_no_value"]
    checkpoint: ProcessingCheckpoint
    evidence_count: int = Field(ge=0)
    index_outcome: Literal["processed", "idempotent_conflict", "no_evidence"] | None = None
    resumed: bool = False


class _PipelineFault(RuntimeError):
    def __init__(self, category: str, disposition: FailureDisposition) -> None:
        self.category = _normalize_category(category)
        self.disposition = disposition
        super().__init__(self.category)


@dataclass(slots=True)
class _RunContext:
    checkpoint: ProcessingCheckpoint
    resumed: bool


def build_ingestion_idempotency_key(
    source_id: UUID,
    *,
    source_version_key: str,
    embedding_generation_id: str,
) -> str:
    """Bind one source version and embedding generation without exposing either."""

    version = source_version_key.strip()
    generation = embedding_generation_id.strip()
    if not version or not generation:
        raise ValueError("source version and embedding generation are required")
    digest = sha256("\x00".join((source_id.hex, version, generation)).encode("utf-8")).hexdigest()
    return f"knowledge-ingestion:v2:{source_id.hex}:{digest}"


def _normalize_category(value: str) -> str:
    candidate = value.strip().casefold().replace(" ", "_")
    return candidate if _SAFE_CATEGORY.fullmatch(candidate) else "dependency_failure"


def _replace_metadata(
    checkpoint: ProcessingCheckpoint,
    **updates: object,
) -> ProcessingCheckpoint:
    payload = checkpoint.model_dump(mode="python")
    metadata = dict(payload["metadata"])
    metadata.update(updates)
    payload["metadata"] = metadata
    return ProcessingCheckpoint.model_validate(payload)


def _replace_error(
    checkpoint: ProcessingCheckpoint,
    category: str,
) -> ProcessingCheckpoint:
    payload = checkpoint.model_dump(mode="python")
    payload["last_error_category"] = _normalize_category(category)
    return ProcessingCheckpoint.model_validate(payload)


class KnowledgeIngestionPipeline:
    """Run the bounded SPOOL -> PARSE -> RETAIN -> INDEX source pipeline."""

    def __init__(
        self,
        *,
        reader: RemoteSourceReader,
        spool: SpoolManager,
        mineru: MinerUClient,
        evidence_selector: EvidenceSelector,
        assessment_provider: EvidenceAssessmentProvider,
        lightrag: LightRAGClient,
        checkpoints: CheckpointRepository,
        evidence: EvidenceRepository,
        state_machine: ProcessingStateMachine | None = None,
    ) -> None:
        self._reader = reader
        self._spool = spool
        self._mineru = mineru
        self._evidence_selector = evidence_selector
        self._assessment_provider = assessment_provider
        self._lightrag = lightrag
        self._checkpoints = checkpoints
        self._evidence = evidence
        self._machine = state_machine or ProcessingStateMachine()

    async def ingest(
        self,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        source_version_key: str,
        embedding_generation_id: str,
    ) -> IngestionResult:
        """Ingest one decision without ever opening a rejected source body."""

        try:
            normalized_slice = normalize_identifier(slice_id, field="slice_id")
            generation = embedding_generation_id.strip()
            version = source_version_key.strip()
            expected_key = build_ingestion_idempotency_key(
                decision.source_id,
                source_version_key=version,
                embedding_generation_id=generation,
            )
        except Exception:
            raise IngestionPipelineError(
                disposition=FailureDisposition.PERMANENT,
                stage=IngestionStage.SELECT,
                category="invalid_ingestion_request",
            ) from None

        version_fingerprint = sha256(version.encode("utf-8")).hexdigest()
        if not decision.selected:
            return self._skipped_result(
                decision=decision,
                source_locator=source_locator,
                slice_id=normalized_slice,
                embedding_generation_id=generation,
                source_version_fingerprint=version_fingerprint,
                idempotency_key=expected_key,
            )
        try:
            loaded = await self._checkpoints.load(expected_key)
        except asyncio.CancelledError:
            raise IngestionCancelledError(stage=IngestionStage.SELECT) from None
        except Exception:
            raise IngestionPipelineError(
                disposition=FailureDisposition.RETRYABLE,
                stage=IngestionStage.SELECT,
                category="checkpoint_read_failed",
            ) from None

        if loaded is None:
            checkpoint = ProcessingCheckpoint(
                source_id=decision.source_id,
                lifecycle_status=SourceLifecycleStatus.PARSE_ELIGIBLE,
                stage=IngestionStage.SELECT,
                job_status=IngestionJobStatus.RUNNING,
                idempotency_key=expected_key,
                selection=decision,
                metadata={
                    "root_id": source_locator.root_id,
                    "slice_id": normalized_slice,
                    "relative_path": source_locator.relative_path,
                    "embedding_generation_id": generation,
                    "source_version_fingerprint": version_fingerprint,
                },
            )
            context = _RunContext(checkpoint=checkpoint, resumed=False)
            try:
                await self._persist_checkpoint(context, checkpoint)
            except Exception:
                raise IngestionPipelineError(
                    disposition=FailureDisposition.RETRYABLE,
                    stage=IngestionStage.SELECT,
                    category="checkpoint_write_failed",
                ) from None
        else:
            context = _RunContext(checkpoint=loaded, resumed=True)
            self._validate_resume(
                loaded,
                decision=decision,
                source_locator=source_locator,
                slice_id=normalized_slice,
                embedding_generation_id=generation,
                source_version_fingerprint=version_fingerprint,
                expected_key=expected_key,
            )
        if context.checkpoint.job_status is IngestionJobStatus.SUCCEEDED:
            return self._terminal_result(context)
        if context.checkpoint.job_status in {
            IngestionJobStatus.FAILED_PERMANENT,
            IngestionJobStatus.CANCELLED,
        }:
            if context.checkpoint.job_status is IngestionJobStatus.CANCELLED:
                raise IngestionCancelledError(stage=context.checkpoint.stage)
            raise IngestionPipelineError(
                disposition=FailureDisposition.PERMANENT,
                stage=context.checkpoint.stage,
                category=context.checkpoint.last_error_category or "terminal_failure",
            )

        try:
            await self._ensure_running(context)
            return await self._ingest_selected(
                context,
                decision=decision,
                source_locator=source_locator,
                slice_id=normalized_slice,
                embedding_generation_id=generation,
            )
        except asyncio.CancelledError:
            await self._record_failure(
                context,
                disposition=FailureDisposition.CANCELLED,
                category="cancelled",
            )
            raise IngestionCancelledError(stage=context.checkpoint.stage) from None
        except Exception as error:
            disposition, category = self._classify(error)
            await self._record_failure(context, disposition=disposition, category=category)
            raise IngestionPipelineError(
                disposition=disposition,
                stage=context.checkpoint.stage,
                category=category,
            ) from None

    def _validate_resume(
        self,
        checkpoint: ProcessingCheckpoint,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        embedding_generation_id: str,
        source_version_fingerprint: str,
        expected_key: str,
    ) -> None:
        expected_metadata = {
            "root_id": source_locator.root_id,
            "slice_id": slice_id,
            "relative_path": source_locator.relative_path,
            "embedding_generation_id": embedding_generation_id,
            "source_version_fingerprint": source_version_fingerprint,
        }
        persisted_selection = checkpoint.selection
        invalid = (
            checkpoint.idempotency_key != expected_key
            or persisted_selection is None
            or not persisted_selection.selected
            or persisted_selection.source_id != decision.source_id
            or any(
                checkpoint.metadata.get(key) != value for key, value in expected_metadata.items()
            )
        )
        if invalid:
            raise IngestionPipelineError(
                disposition=FailureDisposition.PERMANENT,
                stage=checkpoint.stage,
                category="checkpoint_identity_mismatch",
            )

    @staticmethod
    def _skipped_result(
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        embedding_generation_id: str,
        source_version_fingerprint: str,
        idempotency_key: str,
    ) -> IngestionResult:
        lifecycle = {
            "duplicate": SourceLifecycleStatus.DEDUPLICATED,
            "process_data_excluded": SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
        }.get(decision.reason_code, SourceLifecycleStatus.METADATA_INDEXED)
        checkpoint = ProcessingCheckpoint(
            source_id=decision.source_id,
            lifecycle_status=lifecycle,
            stage=IngestionStage.SELECT,
            job_status=IngestionJobStatus.SUCCEEDED,
            idempotency_key=idempotency_key,
            selection=decision,
            metadata={
                "root_id": source_locator.root_id,
                "slice_id": slice_id,
                "relative_path": source_locator.relative_path,
                "embedding_generation_id": embedding_generation_id,
                "source_version_fingerprint": source_version_fingerprint,
                "selection_skipped": True,
            },
        )
        return IngestionResult(
            outcome="skipped",
            checkpoint=checkpoint,
            evidence_count=0,
        )

    async def _ensure_running(self, context: _RunContext) -> None:
        checkpoint = context.checkpoint
        if checkpoint.job_status is IngestionJobStatus.RUNNING:
            return
        if checkpoint.job_status is IngestionJobStatus.QUEUED:
            running = self._machine.transition_job(checkpoint, IngestionJobStatus.RUNNING)
            await self._persist_checkpoint(context, running)
            return
        if checkpoint.job_status is IngestionJobStatus.FAILED_RETRYABLE:
            waiting = self._machine.schedule_retry(checkpoint)
            await self._persist_checkpoint(context, waiting)
            checkpoint = waiting
        if checkpoint.job_status is IngestionJobStatus.RETRY_WAIT:
            running = self._machine.resume_retry(checkpoint)
            await self._persist_checkpoint(context, running)
            return
        raise _PipelineFault("non_resumable_job_status", FailureDisposition.PERMANENT)

    async def _ingest_selected(
        self,
        context: _RunContext,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        embedding_generation_id: str,
    ) -> IngestionResult:
        checkpoint = context.checkpoint
        if checkpoint.metadata.get("index_completed") is True:
            return await self._finalize(context)

        if checkpoint.stage in {IngestionStage.SELECT, IngestionStage.SPOOL, IngestionStage.PARSE}:
            fragments = await self._parse_and_retain(
                context,
                decision=decision,
                source_locator=source_locator,
                slice_id=slice_id,
                embedding_generation_id=embedding_generation_id,
            )
        elif checkpoint.stage in {IngestionStage.RETAIN, IngestionStage.INDEX}:
            fragments = await self._load_durable_fragments(
                context,
                embedding_generation_id=embedding_generation_id,
            )
        else:
            raise _PipelineFault("invalid_resume_stage", FailureDisposition.PERMANENT)

        if context.checkpoint.stage is IngestionStage.RETAIN:
            indexing = self._machine.advance_stage(
                context.checkpoint,
                IngestionStage.INDEX,
            )
            await self._persist_checkpoint(context, indexing)
        elif context.checkpoint.stage is not IngestionStage.INDEX:
            raise _PipelineFault("retain_boundary_missing", FailureDisposition.PERMANENT)

        if context.checkpoint.metadata.get("index_completed") is True:
            return await self._finalize(context)

        if fragments:
            insertion = await self._lightrag.insert_retained_fragments(fragments)
            index_outcome = self._accepted_index_outcome(insertion)
        else:
            index_outcome = "no_evidence"

        indexed = _replace_metadata(
            context.checkpoint,
            index_completed=True,
            index_outcome=index_outcome,
        )
        await self._persist_checkpoint(context, indexed)
        return await self._finalize(context)

    async def _parse_and_retain(
        self,
        context: _RunContext,
        *,
        decision: SelectionDecision,
        source_locator: SourceLocator,
        slice_id: str,
        embedding_generation_id: str,
    ) -> list[EvidenceFragment]:
        if context.checkpoint.stage is IngestionStage.SELECT:
            spooling = self._machine.advance_stage(
                context.checkpoint,
                IngestionStage.SPOOL,
                selection=decision,
            )
            await self._persist_checkpoint(context, spooling)
        if context.checkpoint.stage not in {IngestionStage.SPOOL, IngestionStage.PARSE}:
            raise _PipelineFault("invalid_parse_stage", FailureDisposition.PERMANENT)

        remote_stat = await self._reader.stat(
            source_locator.root_id,
            slice_id,
            source_locator.relative_path,
        )
        if remote_stat.is_dir or remote_stat.byte_size is None or remote_stat.byte_size <= 0:
            raise _PipelineFault("invalid_remote_source", FailureDisposition.PERMANENT)
        if (
            remote_stat.root_id != source_locator.root_id
            or remote_stat.slice_id != slice_id
            or remote_stat.relative_path != source_locator.relative_path
        ):
            raise _PipelineFault("remote_identity_mismatch", FailureDisposition.PERMANENT)

        chunks = self._reader.open_stream(
            source_locator.root_id,
            slice_id,
            source_locator.relative_path,
            expected_size=remote_stat.byte_size,
            expected_mtime=remote_stat.modified_at,
        )
        async with self._spool.materialize(
            source_id=str(decision.source_id),
            expected_bytes=remote_stat.byte_size,
            chunks=chunks,
        ) as temporary:
            if context.checkpoint.stage is IngestionStage.SPOOL:
                parsing = self._machine.advance_stage(
                    context.checkpoint,
                    IngestionStage.PARSE,
                )
                await self._persist_checkpoint(context, parsing)

            parsed = await self._mineru.parse(
                temporary.path,
                file_name=PurePosixPath(source_locator.relative_path).name,
                idempotency_key=context.checkpoint.idempotency_key,
                output_dir=temporary.parser_output_dir,
            )
            assessments = list(
                await self._assessment_provider.assess(
                    parsed,
                    decision=decision,
                    source_locator=source_locator,
                )
            )
            selected = self._evidence_selector.retain(
                parsed,
                decision=decision,
                source_locator=source_locator,
                assessments=assessments,
                embedding_generation_id=embedding_generation_id,
            )
            fragments = self._stable_fragments(
                selected,
                idempotency_key=context.checkpoint.idempotency_key,
            )
            await self._evidence.persist_many(
                decision.source_id,
                fragments,
                idempotency_key=context.checkpoint.idempotency_key,
            )

            retaining = self._machine.advance_stage(
                context.checkpoint,
                IngestionStage.RETAIN,
            )
            retaining = _replace_metadata(
                retaining,
                content_sha256=temporary.content_sha256,
                parse_completed=True,
                evidence_persisted=True,
                fragment_count=len(fragments),
            )
            await self._persist_checkpoint(context, retaining)
            return fragments

    async def _load_durable_fragments(
        self,
        context: _RunContext,
        *,
        embedding_generation_id: str,
    ) -> list[EvidenceFragment]:
        fragments = await self._evidence.list_for_source(
            context.checkpoint.source_id,
            idempotency_key=context.checkpoint.idempotency_key,
        )
        expected_count = context.checkpoint.metadata.get("fragment_count")
        if not isinstance(expected_count, int) or expected_count != len(fragments):
            raise _PipelineFault("durable_evidence_mismatch", FailureDisposition.PERMANENT)
        if any(
            fragment.source_id != context.checkpoint.source_id
            or fragment.embedding_generation_id != embedding_generation_id
            for fragment in fragments
        ):
            raise _PipelineFault("durable_evidence_mismatch", FailureDisposition.PERMANENT)
        return fragments

    @staticmethod
    def _stable_fragments(
        fragments: Sequence[EvidenceFragment],
        *,
        idempotency_key: str,
    ) -> list[EvidenceFragment]:
        stable: list[EvidenceFragment] = []
        for fragment in fragments:
            if not isinstance(fragment, EvidenceFragment):
                raise _PipelineFault("invalid_evidence_fragment", FailureDisposition.PERMANENT)
            locator = fragment.locator
            identity = "|".join(
                (
                    idempotency_key,
                    fragment.content_sha256 or "",
                    str(locator.page or 0),
                    str(locator.block_index or 0),
                    locator.section or "",
                )
            )
            stable.append(
                fragment.model_copy(update={"fragment_id": uuid5(NAMESPACE_URL, identity)})
            )
        return stable

    @staticmethod
    def _accepted_index_outcome(
        insertion: LightRAGInsertResult,
    ) -> Literal["processed", "idempotent_conflict"]:
        if insertion.outcome == "processed":
            return "processed"
        if insertion.outcome == "idempotent_conflict":
            return "idempotent_conflict"
        raise _PipelineFault("lightrag_document_failed", FailureDisposition.PERMANENT)

    async def _finalize(self, context: _RunContext) -> IngestionResult:
        checkpoint = context.checkpoint
        if checkpoint.metadata.get("index_completed") is not True:
            raise _PipelineFault("index_boundary_missing", FailureDisposition.PERMANENT)
        raw_count = checkpoint.metadata.get("fragment_count")
        if not isinstance(raw_count, int) or raw_count < 0:
            raise _PipelineFault("fragment_count_missing", FailureDisposition.PERMANENT)

        if checkpoint.job_status is not IngestionJobStatus.SUCCEEDED:
            lifecycle = (
                SourceLifecycleStatus.EVIDENCE_RETAINED
                if raw_count > 0
                else SourceLifecycleStatus.PARSED_NO_VALUE
            )
            completed = self._machine.transition_lifecycle(checkpoint, lifecycle)
            completed = self._machine.transition_job(completed, IngestionJobStatus.SUCCEEDED)
            await self._persist_checkpoint(context, completed)

        raw_index_outcome = context.checkpoint.metadata.get("index_outcome")
        if raw_index_outcome not in {"processed", "idempotent_conflict", "no_evidence"}:
            raise _PipelineFault("index_outcome_missing", FailureDisposition.PERMANENT)
        return IngestionResult(
            outcome="indexed" if raw_count else "parsed_no_value",
            checkpoint=context.checkpoint,
            evidence_count=raw_count,
            index_outcome=raw_index_outcome,
            resumed=context.resumed,
        )

    def _terminal_result(self, context: _RunContext) -> IngestionResult:
        checkpoint = context.checkpoint
        count = checkpoint.metadata.get("fragment_count", 0)
        if not isinstance(count, int) or count < 0:
            count = 0
        index_outcome = checkpoint.metadata.get("index_outcome")
        if index_outcome not in {"processed", "idempotent_conflict", "no_evidence"}:
            index_outcome = None
        if checkpoint.metadata.get("selection_skipped") is True:
            outcome: Literal["skipped", "indexed", "parsed_no_value"] = "skipped"
        else:
            outcome = "indexed" if count else "parsed_no_value"
        return IngestionResult(
            outcome=outcome,
            checkpoint=checkpoint,
            evidence_count=count,
            index_outcome=index_outcome,
            resumed=context.resumed,
        )

    async def _persist_checkpoint(
        self,
        context: _RunContext,
        checkpoint: ProcessingCheckpoint,
    ) -> None:
        await self._checkpoints.save(checkpoint)
        context.checkpoint = checkpoint

    async def _record_failure(
        self,
        context: _RunContext,
        *,
        disposition: FailureDisposition,
        category: str,
    ) -> None:
        checkpoint = context.checkpoint
        if checkpoint.job_status in _TERMINAL_JOBS:
            return
        try:
            terminal_lifecycles = {
                SourceLifecycleStatus.DEDUPLICATED,
                SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
                SourceLifecycleStatus.EVIDENCE_RETAINED,
                SourceLifecycleStatus.PARSED_NO_VALUE,
                SourceLifecycleStatus.FAILED_PERMANENT,
            }
            if (
                disposition is FailureDisposition.PERMANENT
                and checkpoint.lifecycle_status not in terminal_lifecycles
            ):
                checkpoint = self._machine.transition_lifecycle(
                    checkpoint,
                    SourceLifecycleStatus.FAILED_PERMANENT,
                )
            target = {
                FailureDisposition.RETRYABLE: IngestionJobStatus.FAILED_RETRYABLE,
                FailureDisposition.PERMANENT: IngestionJobStatus.FAILED_PERMANENT,
                FailureDisposition.CANCELLED: IngestionJobStatus.CANCELLED,
            }[disposition]
            checkpoint = self._machine.transition_job(checkpoint, target)
            checkpoint = _replace_error(checkpoint, category)
            await self._persist_checkpoint(context, checkpoint)
        except Exception:
            # Never replace the sanitized primary error with a repository or
            # provider exception that may contain credentials.
            return

    @staticmethod
    def _classify(error: Exception) -> tuple[FailureDisposition, str]:
        if isinstance(error, _PipelineFault):
            return error.disposition, error.category
        if isinstance(error, MinerUError):
            category = _normalize_category(error.category)
            disposition = (
                FailureDisposition.RETRYABLE if error.retryable else FailureDisposition.PERMANENT
            )
            return disposition, f"mineru.{category}"
        if isinstance(error, SpoolCapacityError):
            return FailureDisposition.RETRYABLE, "spool.capacity"
        if isinstance(error, SpoolIntegrityError):
            return FailureDisposition.RETRYABLE, "spool.integrity"
        if isinstance(error, SpoolError):
            return FailureDisposition.RETRYABLE, "spool.failure"
        if isinstance(error, LightRAGRequestError):
            retryable = (
                error.status_code is None or error.status_code == 429 or error.status_code >= 500
            )
            disposition = (
                FailureDisposition.RETRYABLE if retryable else FailureDisposition.PERMANENT
            )
            return disposition, "lightrag.request"
        if isinstance(error, LightRAGPollingTimeout):
            return FailureDisposition.RETRYABLE, "lightrag.poll_timeout"
        if isinstance(
            error,
            (
                LightRAGForbiddenOperation,
                LightRAGProtocolError,
                LightRAGSourceMappingConflict,
            ),
        ):
            return FailureDisposition.PERMANENT, "lightrag.protocol"
        if isinstance(error, LightRAGError):
            return FailureDisposition.RETRYABLE, "lightrag.failure"
        if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
            return FailureDisposition.PERMANENT, "invalid_pipeline_data"
        if isinstance(error, (TimeoutError, ConnectionError, OSError)):
            return FailureDisposition.RETRYABLE, "dependency_io"
        return FailureDisposition.RETRYABLE, "dependency_failure"


__all__ = [
    "CheckpointRepository",
    "EvidenceAssessmentProvider",
    "EvidenceRepository",
    "FailureDisposition",
    "InMemoryCheckpointRepository",
    "InMemoryEvidenceRepository",
    "IngestionCancelledError",
    "IngestionPipelineError",
    "IngestionResult",
    "KnowledgeIngestionPipeline",
    "build_ingestion_idempotency_key",
]
