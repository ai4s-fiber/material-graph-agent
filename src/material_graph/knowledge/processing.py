"""Auditable lifecycle and job-state contracts for knowledge ingestion.

Source lifecycle, pipeline stage, and worker job status are deliberately
separate.  A retry may change the job status and attempt counter without
rewinding the source lifecycle or losing the exact stage being retried.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .models import SelectionDecision


class SourceLifecycleStatus(StrEnum):
    """Durable classification of one canonical source."""

    METADATA_DISCOVERED = "metadata_discovered"
    METADATA_INDEXED = "metadata_indexed"
    DEDUPLICATED = "deduplicated"
    EXCLUDED_PROCESS_DATA = "excluded_process_data"
    PARSE_ELIGIBLE = "parse_eligible"
    EVIDENCE_RETAINED = "evidence_retained"
    PARSED_NO_VALUE = "parsed_no_value"
    FAILED_PERMANENT = "failed_permanent"


class IngestionStage(StrEnum):
    """Ordered work stage within one ingestion job."""

    CATALOG = "catalog"
    HASH = "hash"
    SELECT = "select"
    SPOOL = "spool"
    PARSE = "parse"
    RETAIN = "retain"
    INDEX = "index"


class IngestionJobStatus(StrEnum):
    """Execution status for a resumable worker job."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    CANCELLED = "cancelled"


class ProcessingStateError(ValueError):
    """Raised when a requested processing transition violates the contract."""


_SENSITIVE_FIELD_MARKERS = (
    "endpoint",
    "session",
    "token",
    "credential",
    "password",
    "api_key",
    "secret",
    "username",
    "host",
    "address",
    "cookie",
    "quickconnect",
)
_SENSITIVE_EXACT_FIELDS = frozenset(
    {
        "did",
        "sid",
        "device_id",
        "deviceid",
        "syno_token",
        "synotoken",
    }
)


def _reject_sensitive_fields(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _SENSITIVE_EXACT_FIELDS or any(
                marker in normalized for marker in _SENSITIVE_FIELD_MARKERS
            ):
                raise ValueError(f"sensitive checkpoint field is forbidden: {path}.{key}")
            _reject_sensitive_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_fields(nested, path=f"{path}[{index}]")


_BODY_STAGES = frozenset(
    {
        IngestionStage.SPOOL,
        IngestionStage.PARSE,
        IngestionStage.RETAIN,
        IngestionStage.INDEX,
    }
)

_TERMINAL_SOURCE_STATUSES = frozenset(
    {
        SourceLifecycleStatus.DEDUPLICATED,
        SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
        SourceLifecycleStatus.EVIDENCE_RETAINED,
        SourceLifecycleStatus.PARSED_NO_VALUE,
        SourceLifecycleStatus.FAILED_PERMANENT,
    }
)

_TERMINAL_JOB_STATUSES = frozenset(
    {
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED_PERMANENT,
        IngestionJobStatus.CANCELLED,
    }
)

_NEXT_STAGE = {
    IngestionStage.CATALOG: IngestionStage.HASH,
    IngestionStage.HASH: IngestionStage.SELECT,
    IngestionStage.SELECT: IngestionStage.SPOOL,
    IngestionStage.SPOOL: IngestionStage.PARSE,
    IngestionStage.PARSE: IngestionStage.RETAIN,
    IngestionStage.RETAIN: IngestionStage.INDEX,
}

_SOURCE_TRANSITIONS = {
    SourceLifecycleStatus.METADATA_DISCOVERED: frozenset(
        {
            SourceLifecycleStatus.METADATA_INDEXED,
            SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
            SourceLifecycleStatus.FAILED_PERMANENT,
        }
    ),
    SourceLifecycleStatus.METADATA_INDEXED: frozenset(
        {
            SourceLifecycleStatus.DEDUPLICATED,
            SourceLifecycleStatus.PARSE_ELIGIBLE,
            SourceLifecycleStatus.FAILED_PERMANENT,
        }
    ),
    SourceLifecycleStatus.PARSE_ELIGIBLE: frozenset(
        {
            SourceLifecycleStatus.EVIDENCE_RETAINED,
            SourceLifecycleStatus.PARSED_NO_VALUE,
            SourceLifecycleStatus.FAILED_PERMANENT,
        }
    ),
}

_JOB_TRANSITIONS = {
    IngestionJobStatus.QUEUED: frozenset(
        {IngestionJobStatus.RUNNING, IngestionJobStatus.CANCELLED}
    ),
    IngestionJobStatus.RUNNING: frozenset(
        {
            IngestionJobStatus.RETRY_WAIT,
            IngestionJobStatus.SUCCEEDED,
            IngestionJobStatus.FAILED_RETRYABLE,
            IngestionJobStatus.FAILED_PERMANENT,
            IngestionJobStatus.CANCELLED,
        }
    ),
    IngestionJobStatus.RETRY_WAIT: frozenset(
        {
            IngestionJobStatus.RUNNING,
            IngestionJobStatus.FAILED_PERMANENT,
            IngestionJobStatus.CANCELLED,
        }
    ),
    IngestionJobStatus.FAILED_RETRYABLE: frozenset(
        {
            IngestionJobStatus.RETRY_WAIT,
            IngestionJobStatus.RUNNING,
            IngestionJobStatus.FAILED_PERMANENT,
            IngestionJobStatus.CANCELLED,
        }
    ),
}


class ProcessingCheckpoint(BaseModel):
    """Credential-free durable snapshot used for retry and resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: UUID
    lifecycle_status: SourceLifecycleStatus
    stage: IngestionStage
    job_status: IngestionJobStatus = IngestionJobStatus.QUEUED
    attempt: int = Field(default=0, ge=0)
    idempotency_key: str = Field(min_length=1, max_length=512)
    selection: SelectionDecision | None = None
    cursor: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    last_error_category: str | None = Field(default=None, max_length=100)

    @field_validator("cursor", "metadata", mode="before")
    @classmethod
    def reject_sensitive_checkpoint_fields(cls, value: Any) -> Any:
        _reject_sensitive_fields(value, path="checkpoint")
        return value

    @model_validator(mode="after")
    def validate_source_and_selection(self) -> "ProcessingCheckpoint":
        if self.selection is not None and self.selection.source_id != self.source_id:
            raise ValueError("selection source_id must match checkpoint source_id")
        if self.stage in _BODY_STAGES:
            if self.lifecycle_status is SourceLifecycleStatus.EXCLUDED_PROCESS_DATA:
                raise ValueError("excluded source cannot enter body processing")
            if self.selection is None or not self.selection.selected:
                raise ValueError("body processing requires a selected SelectionDecision")
        return self


def _updated_checkpoint(
    checkpoint: ProcessingCheckpoint,
    **updates: Any,
) -> ProcessingCheckpoint:
    payload = checkpoint.model_dump(mode="python")
    payload.update(updates)
    return ProcessingCheckpoint.model_validate(payload)


class ProcessingStateMachine:
    """Validate every durable source, stage, job, and retry transition."""

    def transition_lifecycle(
        self,
        checkpoint: ProcessingCheckpoint,
        target: SourceLifecycleStatus,
    ) -> ProcessingCheckpoint:
        current = checkpoint.lifecycle_status
        if target is current:
            return checkpoint
        if current in _TERMINAL_SOURCE_STATUSES:
            raise ProcessingStateError(
                f"terminal source lifecycle {current.value} cannot transition"
            )
        if target not in _SOURCE_TRANSITIONS.get(current, frozenset()):
            raise ProcessingStateError(
                f"invalid source lifecycle transition: {current.value} -> {target.value}"
            )
        return _updated_checkpoint(checkpoint, lifecycle_status=target)

    def transition_job(
        self,
        checkpoint: ProcessingCheckpoint,
        target: IngestionJobStatus,
    ) -> ProcessingCheckpoint:
        current = checkpoint.job_status
        if target is current:
            return checkpoint
        if current in _TERMINAL_JOB_STATUSES:
            raise ProcessingStateError(f"terminal job status {current.value} cannot transition")
        if target not in _JOB_TRANSITIONS.get(current, frozenset()):
            raise ProcessingStateError(
                f"invalid job status transition: {current.value} -> {target.value}"
            )
        return _updated_checkpoint(checkpoint, job_status=target)

    def advance_stage(
        self,
        checkpoint: ProcessingCheckpoint,
        target: IngestionStage,
        *,
        selection: SelectionDecision | None = None,
    ) -> ProcessingCheckpoint:
        current = checkpoint.stage
        if checkpoint.lifecycle_status is SourceLifecycleStatus.EXCLUDED_PROCESS_DATA:
            if target in _BODY_STAGES:
                raise ProcessingStateError("excluded source cannot enter body processing")
        if checkpoint.lifecycle_status in _TERMINAL_SOURCE_STATUSES:
            raise ProcessingStateError(
                "terminal source lifecycle cannot advance an ingestion stage"
            )
        if checkpoint.job_status is not IngestionJobStatus.RUNNING:
            raise ProcessingStateError("only a running ingestion job can advance stages")
        expected = _NEXT_STAGE.get(current)
        if expected is not target:
            raise ProcessingStateError(
                f"invalid ingestion stage transition: {current.value} -> {target.value}"
            )

        resolved_selection = selection or checkpoint.selection
        if target is IngestionStage.SPOOL and (
            resolved_selection is None or not resolved_selection.selected
        ):
            raise ProcessingStateError("spooling requires a persisted selected SelectionDecision")
        return _updated_checkpoint(
            checkpoint,
            stage=target,
            selection=resolved_selection,
        )

    def schedule_retry(self, checkpoint: ProcessingCheckpoint) -> ProcessingCheckpoint:
        if checkpoint.job_status in _TERMINAL_JOB_STATUSES:
            raise ProcessingStateError(
                f"terminal job status {checkpoint.job_status.value} cannot retry"
            )
        if checkpoint.job_status not in {
            IngestionJobStatus.RUNNING,
            IngestionJobStatus.FAILED_RETRYABLE,
        }:
            raise ProcessingStateError(
                f"job status {checkpoint.job_status.value} cannot schedule a retry"
            )
        return _updated_checkpoint(
            checkpoint,
            job_status=IngestionJobStatus.RETRY_WAIT,
            attempt=checkpoint.attempt + 1,
        )

    def resume_retry(self, checkpoint: ProcessingCheckpoint) -> ProcessingCheckpoint:
        if checkpoint.job_status is not IngestionJobStatus.RETRY_WAIT:
            raise ProcessingStateError(
                f"job status {checkpoint.job_status.value} is not waiting for retry"
            )
        return _updated_checkpoint(checkpoint, job_status=IngestionJobStatus.RUNNING)


__all__ = [
    "IngestionJobStatus",
    "IngestionStage",
    "ProcessingCheckpoint",
    "ProcessingStateError",
    "ProcessingStateMachine",
    "SourceLifecycleStatus",
]
