from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from material_graph.knowledge.models import SelectionDecision
from material_graph.knowledge.processing import (
    IngestionJobStatus,
    IngestionStage,
    ProcessingCheckpoint,
    ProcessingStateError,
    ProcessingStateMachine,
    SourceLifecycleStatus,
)


MACHINE = ProcessingStateMachine()


def _selection(source_id: UUID, *, selected: bool = True) -> SelectionDecision:
    return SelectionDecision(
        source_id=source_id,
        selected=selected,
        reason_code="active_evidence_gap" if selected else "budget_deferred",
        policy_version="selection-v1",
    )


def _checkpoint(
    *,
    source_id: UUID | None = None,
    lifecycle_status: SourceLifecycleStatus = SourceLifecycleStatus.PARSE_ELIGIBLE,
    stage: IngestionStage = IngestionStage.SELECT,
    job_status: IngestionJobStatus = IngestionJobStatus.RUNNING,
    selection: SelectionDecision | None = None,
    attempt: int = 0,
) -> ProcessingCheckpoint:
    resolved_source_id = source_id or uuid4()
    return ProcessingCheckpoint(
        source_id=resolved_source_id,
        lifecycle_status=lifecycle_status,
        stage=stage,
        job_status=job_status,
        selection=selection,
        attempt=attempt,
        idempotency_key=f"ingestion:{resolved_source_id}",
    )


def test_processing_contracts_are_split_and_json_serializable() -> None:
    checkpoint = _checkpoint()
    payload = json.loads(checkpoint.model_dump_json())

    assert payload["lifecycle_status"] == "parse_eligible"
    assert payload["stage"] == "select"
    assert payload["job_status"] == "running"
    assert "spooling" not in {status.value for status in SourceLifecycleStatus}
    assert "spool" in {stage.value for stage in IngestionStage}


def test_excluded_source_cannot_enter_body_processing() -> None:
    source_id = uuid4()
    excluded = _checkpoint(
        source_id=source_id,
        lifecycle_status=SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
        stage=IngestionStage.SELECT,
        selection=_selection(source_id),
    )

    with pytest.raises(ProcessingStateError, match="excluded"):
        MACHINE.advance_stage(excluded, IngestionStage.SPOOL)
    with pytest.raises(ProcessingStateError, match="terminal source lifecycle"):
        MACHINE.transition_lifecycle(excluded, SourceLifecycleStatus.PARSE_ELIGIBLE)


@pytest.mark.parametrize("selected", [None, False])
def test_selection_decision_is_required_before_spooling(selected: bool | None) -> None:
    source_id = uuid4()
    checkpoint = _checkpoint(
        source_id=source_id,
        selection=None if selected is None else _selection(source_id, selected=selected),
    )

    with pytest.raises(ProcessingStateError, match="selected SelectionDecision"):
        MACHINE.advance_stage(checkpoint, IngestionStage.SPOOL)


def test_selection_must_reference_the_same_source() -> None:
    with pytest.raises(ValidationError, match="selection source_id"):
        _checkpoint(selection=_selection(uuid4()))


def test_stage_transition_cannot_skip_or_move_backwards() -> None:
    checkpoint = _checkpoint(stage=IngestionStage.CATALOG)

    with pytest.raises(ProcessingStateError, match="catalog -> select"):
        MACHINE.advance_stage(checkpoint, IngestionStage.SELECT)

    hashing = MACHINE.advance_stage(checkpoint, IngestionStage.HASH)
    selecting = MACHINE.advance_stage(hashing, IngestionStage.SELECT)
    selected = MACHINE.advance_stage(
        selecting,
        IngestionStage.SPOOL,
        selection=_selection(checkpoint.source_id),
    )

    assert selected.stage is IngestionStage.SPOOL
    with pytest.raises(ProcessingStateError, match="spool -> select"):
        MACHINE.advance_stage(selected, IngestionStage.SELECT)


def test_retry_preserves_stage_attempt_history_and_idempotency_key() -> None:
    source_id = uuid4()
    checkpoint = _checkpoint(
        source_id=source_id,
        stage=IngestionStage.PARSE,
        selection=_selection(source_id),
        attempt=2,
    )

    waiting = MACHINE.schedule_retry(checkpoint)
    resumed = MACHINE.resume_retry(waiting)

    assert waiting.job_status is IngestionJobStatus.RETRY_WAIT
    assert waiting.stage is IngestionStage.PARSE
    assert waiting.attempt == 3
    assert waiting.idempotency_key == checkpoint.idempotency_key
    assert resumed.job_status is IngestionJobStatus.RUNNING
    assert resumed.stage is checkpoint.stage
    assert resumed.attempt == waiting.attempt
    assert resumed.idempotency_key == checkpoint.idempotency_key


@pytest.mark.parametrize(
    "terminal_status",
    [
        IngestionJobStatus.SUCCEEDED,
        IngestionJobStatus.FAILED_PERMANENT,
        IngestionJobStatus.CANCELLED,
    ],
)
def test_terminal_job_status_cannot_transition(
    terminal_status: IngestionJobStatus,
) -> None:
    checkpoint = _checkpoint(job_status=terminal_status)

    with pytest.raises(ProcessingStateError, match="terminal job status"):
        MACHINE.transition_job(checkpoint, IngestionJobStatus.RUNNING)


def test_lifecycle_follows_explicit_transitions_and_then_stops() -> None:
    checkpoint = _checkpoint(
        lifecycle_status=SourceLifecycleStatus.METADATA_DISCOVERED,
        stage=IngestionStage.CATALOG,
    )

    indexed = MACHINE.transition_lifecycle(
        checkpoint,
        SourceLifecycleStatus.METADATA_INDEXED,
    )
    eligible = MACHINE.transition_lifecycle(
        indexed,
        SourceLifecycleStatus.PARSE_ELIGIBLE,
    )
    retained = MACHINE.transition_lifecycle(
        eligible,
        SourceLifecycleStatus.EVIDENCE_RETAINED,
    )

    assert retained.lifecycle_status is SourceLifecycleStatus.EVIDENCE_RETAINED
    with pytest.raises(ProcessingStateError, match="terminal source lifecycle"):
        MACHINE.transition_lifecycle(
            retained,
            SourceLifecycleStatus.FAILED_PERMANENT,
        )


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        {"endpoint": "https://nas.invalid"},
        {"nested": {"session": "session-id"}},
        {"nested": [{"token": "auth-token"}]},
        {"credential": {"username": "operator"}},
        {"api_key": "secret"},
        {"password": "secret"},
        {"did": "device-token"},
        {"device_id": "device-token"},
        {"sid": "session-id"},
        {"synotoken": "session-token"},
        {"relay_cookie": "cookie-value"},
        {"quickconnect_id": "nas-alias"},
    ],
)
def test_checkpoint_rejects_sensitive_fields_recursively(
    unsafe_payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="sensitive checkpoint field"):
        ProcessingCheckpoint(
            source_id=uuid4(),
            lifecycle_status=SourceLifecycleStatus.METADATA_DISCOVERED,
            stage=IngestionStage.CATALOG,
            job_status=IngestionJobStatus.RUNNING,
            idempotency_key="catalog:fixture",
            cursor=unsafe_payload,
        )


def test_checkpoint_forbids_sensitive_top_level_extras() -> None:
    with pytest.raises(ValidationError):
        ProcessingCheckpoint(
            source_id=uuid4(),
            lifecycle_status=SourceLifecycleStatus.METADATA_DISCOVERED,
            stage=IngestionStage.CATALOG,
            job_status=IngestionJobStatus.RUNNING,
            idempotency_key="catalog:fixture",
            endpoint="https://nas.invalid",
        )


def test_safe_cursor_and_metadata_are_retained() -> None:
    checkpoint = ProcessingCheckpoint(
        source_id=uuid4(),
        lifecycle_status=SourceLifecycleStatus.METADATA_DISCOVERED,
        stage=IngestionStage.CATALOG,
        job_status=IngestionJobStatus.RUNNING,
        idempotency_key="catalog:fixture",
        cursor={"row_number": 120, "byte_offset": 4096},
        metadata={"root_id": "document_data_1", "slice_id": "literature"},
    )

    assert checkpoint.cursor["row_number"] == 120
    assert checkpoint.metadata["root_id"] == "document_data_1"


def test_checkpoint_cannot_be_constructed_inside_body_pipeline_without_authority() -> None:
    source_id = uuid4()
    with pytest.raises(ValidationError, match="excluded source"):
        _checkpoint(
            source_id=source_id,
            lifecycle_status=SourceLifecycleStatus.EXCLUDED_PROCESS_DATA,
            stage=IngestionStage.SPOOL,
            selection=_selection(source_id),
        )
    with pytest.raises(ValidationError, match="selected SelectionDecision"):
        _checkpoint(stage=IngestionStage.PARSE)


def test_noop_and_invalid_lifecycle_and_job_transitions() -> None:
    checkpoint = _checkpoint()

    assert MACHINE.transition_lifecycle(checkpoint, checkpoint.lifecycle_status) is checkpoint
    assert MACHINE.transition_job(checkpoint, checkpoint.job_status) is checkpoint

    discovered = _checkpoint(
        lifecycle_status=SourceLifecycleStatus.METADATA_DISCOVERED,
        stage=IngestionStage.CATALOG,
    )
    with pytest.raises(ProcessingStateError, match="invalid source lifecycle"):
        MACHINE.transition_lifecycle(
            discovered,
            SourceLifecycleStatus.EVIDENCE_RETAINED,
        )

    queued = _checkpoint(job_status=IngestionJobStatus.QUEUED)
    with pytest.raises(ProcessingStateError, match="invalid job status"):
        MACHINE.transition_job(queued, IngestionJobStatus.SUCCEEDED)


def test_terminal_source_and_nonrunning_job_cannot_advance_stage() -> None:
    terminal = _checkpoint(
        lifecycle_status=SourceLifecycleStatus.EVIDENCE_RETAINED,
        stage=IngestionStage.CATALOG,
    )
    with pytest.raises(ProcessingStateError, match="terminal source lifecycle"):
        MACHINE.advance_stage(terminal, IngestionStage.HASH)

    queued = _checkpoint(
        stage=IngestionStage.CATALOG,
        job_status=IngestionJobStatus.QUEUED,
    )
    with pytest.raises(ProcessingStateError, match="running ingestion job"):
        MACHINE.advance_stage(queued, IngestionStage.HASH)


def test_retry_rejects_terminal_nonretryable_and_not_waiting_jobs() -> None:
    with pytest.raises(ProcessingStateError, match="terminal job status"):
        MACHINE.schedule_retry(_checkpoint(job_status=IngestionJobStatus.CANCELLED))
    with pytest.raises(ProcessingStateError, match="cannot schedule a retry"):
        MACHINE.schedule_retry(_checkpoint(job_status=IngestionJobStatus.QUEUED))
    with pytest.raises(ProcessingStateError, match="not waiting for retry"):
        MACHINE.resume_retry(_checkpoint(job_status=IngestionJobStatus.RUNNING))
