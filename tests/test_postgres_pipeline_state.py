from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from material_graph.knowledge import postgres_pipeline_state as state
from material_graph.knowledge.extraction import (
    FactExtractionCheckpoint,
    FactExtractionCheckpointConflict,
    build_fact_extraction_idempotency_key,
)
from material_graph.knowledge.facts import (
    EntityRef,
    EvidenceLink,
    ExtractionProvenance,
    FactBatch,
    RelationAssertion,
)
from material_graph.knowledge.models import SourceLocator
from material_graph.knowledge.service import (
    CanaryStage,
    CanaryStageRecord,
    CanaryStatus,
    KnowledgeCanaryError,
)


ROOT = Path(__file__).parents[1]
FRAGMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
SOURCE_ID = UUID("22222222-2222-4222-8222-222222222222")
OTHER_SOURCE_ID = UUID("33333333-3333-4333-8333-333333333333")
GENERATION = "extractor-generation-1"
CONTENT_SHA = "a" * 64
FINGERPRINT = "b" * 64


@dataclass(frozen=True)
class Statement:
    sql: str
    params: tuple[object, ...]


class Script:
    def __init__(self) -> None:
        self.responses: dict[str, deque[list[object]]] = defaultdict(deque)
        self.failures: dict[str, BaseException] = {}

    def add(self, needle: str, *responses: Sequence[object]) -> None:
        for response in responses:
            self.responses[needle].append(list(response))

    def fail(self, needle: str, error: BaseException) -> None:
        self.failures[needle] = error

    def take(self, sql: str) -> list[object]:
        compact = " ".join(sql.split())
        for needle, error in self.failures.items():
            if needle in compact:
                raise error
        for needle, responses in self.responses.items():
            if needle in compact and responses:
                return responses.popleft()
        return []


class Cursor:
    def __init__(self, rows: Sequence[object]) -> None:
        self.rows = list(rows)

    async def fetchone(self) -> object | None:
        return None if not self.rows else self.rows[0]

    async def fetchall(self) -> list[object]:
        return list(self.rows)


class RecordingConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    async def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> Cursor:
        self.statements.append(Statement(sql, tuple(params or ())))
        return Cursor(self.script.take(sql))

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        try:
            yield self
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1


class RecordingPool:
    def __init__(self, connection: RecordingConnection) -> None:
        self.value = connection
        self.connections = 0
        self.dsn = "postgresql://secret:password@internal/db"

    @asynccontextmanager
    async def connection(self):
        self.connections += 1
        yield self.value

    def __repr__(self) -> str:
        return f"RecordingPool({self.dsn!r})"


def extraction() -> ExtractionProvenance:
    return ExtractionProvenance(
        extractor_name="fact-extractor",
        extractor_version="1.0",
        generation_id=GENERATION,
        model_name="reasoning-model",
        model_version="2026-07-27",
    )


def fact_batch() -> FactBatch:
    provenance = extraction()
    material = EntityRef(
        entity_type="material",
        canonical_name="Material X",
        identifiers={"registry": "MX-1"},
    )
    additive = EntityRef(
        entity_type="component",
        canonical_name="Additive Y",
        identifiers={"registry": "AY-1"},
    )
    relation = RelationAssertion(
        subject=additive,
        predicate="improves",
        object=material,
        evidence=(
            EvidenceLink(
                fragment_id=FRAGMENT_ID,
                source_id=SOURCE_ID,
                locator=SourceLocator(
                    root_id="approved_root",
                    relative_path=f"fragments/{FRAGMENT_ID}",
                    page=3,
                    section="Results",
                ),
            ),
        ),
        confidence=0.9,
        evidence_quality="high",
        assertion_status="affirmed",
        extraction=provenance,
    )
    return FactBatch(
        evidence_fragment_id=FRAGMENT_ID,
        extraction=provenance,
        entities=(material, additive),
        relations=(relation,),
    )


def checkpoint(
    *,
    status: str = "running",
    attempts: int = 1,
    request_fingerprint: str = FINGERPRINT,
    content_sha: str = CONTENT_SHA,
) -> FactExtractionCheckpoint:
    completed = status == "completed"
    failed = status in {"retry_wait", "failed_permanent"}
    return FactExtractionCheckpoint(
        idempotency_key=build_fact_extraction_idempotency_key(FRAGMENT_ID, GENERATION),
        fragment_id=FRAGMENT_ID,
        source_id=SOURCE_ID,
        fragment_content_sha256=content_sha,
        request_fingerprint=request_fingerprint,
        extraction=extraction(),
        status=status,
        attempts=attempts,
        batch=fact_batch() if completed else None,
        last_error_code="extraction.provider_unavailable" if failed else None,
    )


def checkpoint_row(
    value: FactExtractionCheckpoint,
    *,
    json_strings: bool = False,
) -> dict[str, object]:
    raw_batch: object = None
    if value.batch is not None:
        raw_batch = state._redacted_batch(value.batch, value.fragment_id)
    raw_extraction: object = value.extraction.model_dump(mode="json")
    if json_strings:
        raw_extraction = json.dumps(raw_extraction)
        if raw_batch is not None:
            raw_batch = json.dumps(raw_batch)
    return {
        "idempotency_key": value.idempotency_key,
        "fragment_id": value.fragment_id,
        "source_id": value.source_id,
        "fragment_content_sha256": value.fragment_content_sha256,
        "request_fingerprint": value.request_fingerprint,
        "extraction": raw_extraction,
        "status": value.status,
        "attempts": value.attempts,
        "batch": raw_batch,
        "last_error_code": value.last_error_code,
    }


def canary(
    *,
    run_id: str = "run-1",
    stage: CanaryStage = CanaryStage.METADATA_ONLY,
    status: CanaryStatus = CanaryStatus.RUNNING,
    attempt: int = 1,
    fingerprint: str = FINGERPRINT,
    source_ids: tuple[UUID, ...] | None = None,
    approval_id: str | None = None,
    code: str | None = None,
    resumed: bool = False,
) -> CanaryStageRecord:
    resolved_sources = source_ids
    resolved_approval = approval_id
    if stage is CanaryStage.METADATA_ONLY:
        resolved_sources = () if source_ids is None else source_ids
        resolved_approval = None
    else:
        resolved_sources = (SOURCE_ID,) if source_ids is None else source_ids
        resolved_approval = "approval:1" if approval_id is None else approval_id
    resolved_code = code
    if resolved_code is None:
        resolved_code = "canary_running" if status is CanaryStatus.RUNNING else "done"
    counts = status is not CanaryStatus.RUNNING
    return CanaryStageRecord(
        run_id=run_id,
        stage=stage,
        status=status,
        attempt=attempt,
        code=resolved_code,
        request_fingerprint=fingerprint,
        source_ids=resolved_sources,
        approval_id=resolved_approval,
        attempted_count=1 if counts else 0,
        completed_count=1 if counts and status is CanaryStatus.SUCCEEDED else 0,
        evidence_count=2 if counts and stage is not CanaryStage.METADATA_ONLY else 0,
        metadata_records=3 if counts and stage is CanaryStage.METADATA_ONLY else 0,
        resumed=resumed,
    )


def canary_row(value: CanaryStageRecord) -> dict[str, object]:
    return {
        "run_id": value.run_id,
        "stage": value.stage.value,
        "status": value.status.value,
        "attempt": value.attempt,
        "code": value.code,
        "request_fingerprint": value.request_fingerprint,
        "source_ids": list(value.source_ids),
        "approval_id": value.approval_id,
        "attempted_count": value.attempted_count,
        "completed_count": value.completed_count,
        "evidence_count": value.evidence_count,
        "metadata_records": value.metadata_records,
        "resumed": value.resumed,
    }


def repositories(
    script: Script | None = None,
) -> tuple[
    state.PostgresFactExtractionCheckpointRepository,
    state.PostgresCanaryRunRepository,
    RecordingPool,
    RecordingConnection,
    Script,
]:
    resolved = script or Script()
    connection = RecordingConnection(resolved)
    pool = RecordingPool(connection)
    return (
        state.PostgresFactExtractionCheckpointRepository(pool),
        state.PostgresCanaryRunRepository(pool),
        pool,
        connection,
        resolved,
    )


def run(awaitable):
    return asyncio.run(awaitable)


def test_migration_is_ordered_idempotent_bounded_and_reversible() -> None:
    forward = (ROOT / "migrations" / "knowledge_0002.sql").read_text(encoding="utf-8")
    down = (ROOT / "migrations" / "knowledge_0002.down.sql").read_text(encoding="utf-8")

    assert "Apply after knowledge_0001" in forward
    assert "VALUES ('knowledge_0002')" in forward
    assert "ON CONFLICT (version) DO NOTHING" in forward
    assert "knowledge_fact_extraction_checkpoints" in forward
    assert "knowledge_canary_runs" in forward
    assert "attempts BETWEEN 1 AND 8" in forward
    assert "octet_length(batch::text) <= 4194304" in forward
    assert "position('\"relative_path\"' IN batch::text) = 0" in forward
    assert "PRIMARY KEY (run_id, stage)" in forward
    assert "attempt = 1 OR resumed" in forward
    assert "Never run automatically" in down
    assert down.index("knowledge_canary_runs") < down.index("knowledge_fact_extraction_checkpoints")
    assert "DELETE FROM schema_migrations WHERE version = 'knowledge_0002'" in down


def test_checkpoint_insert_is_atomic_parameterized_and_returns_defensive_state() -> None:
    candidate = checkpoint()
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [])
    script.add("INSERT INTO knowledge_fact_extraction_checkpoints", [checkpoint_row(candidate)])
    checkpoints, _, pool, connection, _ = repositories(script)

    persisted = run(checkpoints.save(candidate))

    assert persisted == candidate
    assert persisted is not candidate
    assert pool.connections == 1
    assert connection.transactions == connection.commits == 1
    assert connection.rollbacks == 0
    assert any("pg_advisory_xact_lock" in item.sql for item in connection.statements)
    insert = next(
        item
        for item in connection.statements
        if "INSERT INTO knowledge_fact_extraction_checkpoints" in item.sql
    )
    assert "%s" in insert.sql
    assert candidate.idempotency_key not in insert.sql
    assert candidate.idempotency_key in insert.params


def test_completed_checkpoint_redacts_synthetic_path_and_reconstructs_on_load() -> None:
    candidate = checkpoint(status="completed", attempts=2)
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [])
    script.add(
        "INSERT INTO knowledge_fact_extraction_checkpoints",
        [checkpoint_row(candidate, json_strings=True)],
    )
    checkpoints, _, _, connection, _ = repositories(script)

    persisted = run(checkpoints.save(candidate))

    assert persisted == candidate
    insert = next(
        item
        for item in connection.statements
        if "INSERT INTO knowledge_fact_extraction_checkpoints" in item.sql
    )
    batch_parameter = str(insert.params[8])
    assert "relative_path" not in batch_parameter
    assert "private/" not in batch_parameter
    assert "provider_output" not in batch_parameter
    assert f"fragments/{FRAGMENT_ID}" not in batch_parameter
    assert persisted.batch is not None
    link = persisted.batch.relations[0].evidence[0]
    assert link.locator.relative_path == f"fragments/{FRAGMENT_ID}"


def test_checkpoint_load_validates_key_and_stored_contract() -> None:
    candidate = checkpoint(status="retry_wait")
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [checkpoint_row(candidate)])
    checkpoints, _, pool, _, _ = repositories(script)

    assert run(checkpoints.load(candidate.idempotency_key)) == candidate
    before = pool.connections
    with pytest.raises(ValueError, match="invalid extraction checkpoint key"):
        run(checkpoints.load("bad-key"))
    assert pool.connections == before


def test_checkpoint_identical_replay_does_not_issue_update() -> None:
    candidate = checkpoint()
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [checkpoint_row(candidate)])
    checkpoints, _, _, connection, _ = repositories(script)

    assert run(checkpoints.save(candidate)) == candidate
    assert not any(
        "UPDATE knowledge_fact_extraction_checkpoints" in item.sql for item in connection.statements
    )


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [
        (checkpoint(), checkpoint(status="retry_wait")),
        (checkpoint(status="retry_wait"), checkpoint(status="running", attempts=2)),
        (checkpoint(attempts=2), checkpoint(status="completed", attempts=2)),
        (checkpoint(attempts=2), checkpoint(status="failed_permanent", attempts=2)),
    ],
)
def test_checkpoint_valid_transitions_are_compare_and_swap_updated(
    existing: FactExtractionCheckpoint,
    candidate: FactExtractionCheckpoint,
) -> None:
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [checkpoint_row(existing)])
    script.add("UPDATE knowledge_fact_extraction_checkpoints", [checkpoint_row(candidate)])
    checkpoints, _, _, connection, _ = repositories(script)

    assert run(checkpoints.save(candidate)) == candidate
    update = next(
        item
        for item in connection.statements
        if "UPDATE knowledge_fact_extraction_checkpoints" in item.sql
    )
    assert existing.status in update.params
    assert existing.attempts in update.params


@pytest.mark.parametrize(
    ("existing", "candidate"),
    [
        (checkpoint(), checkpoint(request_fingerprint="0" * 64)),
        (checkpoint(), checkpoint(content_sha="0" * 64)),
        (checkpoint(status="completed"), checkpoint()),
        (checkpoint(attempts=2), checkpoint(attempts=1)),
        (checkpoint(), checkpoint(attempts=3)),
        (checkpoint(status="retry_wait"), checkpoint(status="failed_permanent")),
    ],
)
def test_checkpoint_identity_drift_regression_and_terminal_rollback_conflict(
    existing: FactExtractionCheckpoint,
    candidate: FactExtractionCheckpoint,
) -> None:
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [checkpoint_row(existing)])
    checkpoints, _, _, connection, _ = repositories(script)

    with pytest.raises(FactExtractionCheckpointConflict) as error:
        run(checkpoints.save(candidate))

    assert str(error.value) == "extraction.checkpoint_conflict"
    assert connection.rollbacks == 1
    assert not any(
        "UPDATE knowledge_fact_extraction_checkpoints" in item.sql for item in connection.statements
    )


def test_checkpoint_lost_update_and_missing_insert_return_fail_closed() -> None:
    existing = checkpoint()
    candidate = checkpoint(status="retry_wait")
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [checkpoint_row(existing)])
    script.add("UPDATE knowledge_fact_extraction_checkpoints", [])
    checkpoints, _, _, connection, _ = repositories(script)
    with pytest.raises(FactExtractionCheckpointConflict):
        run(checkpoints.save(candidate))
    assert connection.rollbacks == 1

    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [])
    script.add("INSERT INTO knowledge_fact_extraction_checkpoints", [])
    checkpoints, _, _, connection, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError):
        run(checkpoints.save(existing))
    assert connection.rollbacks == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"complete_mineru_output": {}},
        {"provider_output": "full response"},
        {"relative_path": "private/paper.pdf"},
        {"value": "smb://nas/private/paper.pdf"},
        {"value": "https://host.cn6.quickconnect.cn/share"},
        {"value": b"bytes"},
        {"value": float("nan")},
        {"value": "Bearer " + "x" * 24},
    ],
)
def test_pipeline_json_boundary_rejects_body_paths_provider_output_and_secrets(
    payload: dict[str, object],
) -> None:
    with pytest.raises(state.UnsafePipelineStatePayload) as error:
        state._safe_json_object(payload, max_bytes=1024)
    assert str(error.value) == "unsafe_pipeline_state_payload"


def test_unsafe_checkpoint_and_size_limit_are_rejected_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = checkpoint()
    unsafe_provenance = candidate.extraction.model_copy(
        update={"model_name": "Bearer " + "x" * 24},
        deep=True,
    )
    unsafe = candidate.model_copy(update={"extraction": unsafe_provenance}, deep=True)
    checkpoints, _, pool, _, _ = repositories()
    with pytest.raises(state.UnsafePipelineStatePayload):
        run(checkpoints.save(unsafe))
    assert pool.connections == 0

    monkeypatch.setattr(state, "_MAX_EXTRACTION_BYTES", 1)
    with pytest.raises(state.UnsafePipelineStatePayload):
        run(checkpoints.save(candidate))
    assert pool.connections == 0


def test_invalid_stored_checkpoint_and_driver_failure_are_redacted() -> None:
    candidate = checkpoint()
    invalid = checkpoint_row(candidate)
    invalid["extraction"] = "not-json"
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [invalid])
    checkpoints, _, _, connection, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError):
        run(checkpoints.load(candidate.idempotency_key))
    assert connection.rollbacks == 1

    script = Script()
    script.fail("FROM knowledge_fact_extraction_checkpoints", RuntimeError("password=secret"))
    checkpoints, _, _, _, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError) as error:
        run(checkpoints.load(candidate.idempotency_key))
    assert str(error.value) == "pipeline_state_persistence_error"
    assert "password" not in str(error.value)


def test_new_canary_claim_is_atomic_and_parameterized() -> None:
    expected = canary()
    script = Script()
    script.add("FROM knowledge_canary_runs", [])
    script.add("INSERT INTO knowledge_canary_runs", [canary_row(expected)])
    _, canaries, pool, connection, _ = repositories(script)

    claimed = run(
        canaries.begin(
            run_id=expected.run_id,
            stage=expected.stage,
            request_fingerprint=expected.request_fingerprint,
            source_ids=expected.source_ids,
            approval_id=expected.approval_id,
        )
    )

    assert claimed == expected
    assert pool.connections == 1
    assert connection.commits == 1
    assert any("pg_advisory_xact_lock" in item.sql for item in connection.statements)
    insert = next(
        item for item in connection.statements if "INSERT INTO knowledge_canary_runs" in item.sql
    )
    assert "%s" in insert.sql
    assert expected.run_id not in insert.sql
    assert expected.run_id in insert.params


def test_running_canary_claim_and_identity_drift_fail_closed() -> None:
    existing = canary(stage=CanaryStage.SINGLE_PDF)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(existing)])
    _, canaries, _, connection, _ = repositories(script)
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_running"):
        run(
            canaries.begin(
                run_id=existing.run_id,
                stage=existing.stage,
                request_fingerprint=existing.request_fingerprint,
                source_ids=existing.source_ids,
                approval_id=existing.approval_id,
            )
        )
    assert connection.rollbacks == 1

    for changed in (
        {"request_fingerprint": "0" * 64},
        {"source_ids": (OTHER_SOURCE_ID,)},
        {"approval_id": "approval:other"},
    ):
        script = Script()
        script.add("FROM knowledge_canary_runs", [canary_row(existing)])
        _, canaries, _, _, _ = repositories(script)
        request = {
            "run_id": existing.run_id,
            "stage": existing.stage,
            "request_fingerprint": existing.request_fingerprint,
            "source_ids": existing.source_ids,
            "approval_id": existing.approval_id,
            **changed,
        }
        with pytest.raises(KnowledgeCanaryError, match="canary_stage_identity_mismatch"):
            run(canaries.begin(**request))


def test_terminal_canary_restarts_with_incremented_attempt_and_resume_flag() -> None:
    existing = canary(status=CanaryStatus.FAILED)
    expected = canary(attempt=2, resumed=True)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(existing)])
    script.add("UPDATE knowledge_canary_runs SET status", [canary_row(expected)])
    _, canaries, _, connection, _ = repositories(script)

    restarted = run(
        canaries.begin(
            run_id=existing.run_id,
            stage=existing.stage,
            request_fingerprint=existing.request_fingerprint,
            source_ids=existing.source_ids,
            approval_id=existing.approval_id,
        )
    )

    assert restarted == expected
    update = next(
        item
        for item in connection.statements
        if "UPDATE knowledge_canary_runs" in item.sql and "resumed = true" in item.sql
    )
    assert existing.attempt in update.params
    assert existing.status.value in update.params


def test_canary_restart_overflow_and_lost_claim_are_stable() -> None:
    existing = canary(status=CanaryStatus.FAILED).model_copy(
        update={"attempt": state._MAX_CANARY_ATTEMPT, "resumed": True},
        deep=True,
    )
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(existing)])
    _, canaries, _, connection, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError):
        run(
            canaries.begin(
                run_id=existing.run_id,
                stage=existing.stage,
                request_fingerprint=existing.request_fingerprint,
                source_ids=existing.source_ids,
                approval_id=existing.approval_id,
            )
        )
    assert connection.rollbacks == 1

    existing = canary(status=CanaryStatus.FAILED)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(existing)])
    script.add("UPDATE knowledge_canary_runs SET status", [])
    _, canaries, _, _, _ = repositories(script)
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        run(
            canaries.begin(
                run_id=existing.run_id,
                stage=existing.stage,
                request_fingerprint=existing.request_fingerprint,
                source_ids=existing.source_ids,
                approval_id=existing.approval_id,
            )
        )


def test_canary_finish_is_compare_and_swap_and_idempotent() -> None:
    running = canary(stage=CanaryStage.SINGLE_PDF)
    finished = canary(
        stage=CanaryStage.SINGLE_PDF,
        status=CanaryStatus.SUCCEEDED,
        code="single_pdf_succeeded",
    )
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(running)])
    script.add("UPDATE knowledge_canary_runs SET status", [canary_row(finished)])
    _, canaries, _, connection, _ = repositories(script)

    assert run(canaries.finish(finished)) is None
    finish = next(item for item in connection.statements if "AND status = 'running'" in item.sql)
    assert finished.request_fingerprint in finish.params
    assert list(finished.source_ids) in finish.params

    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(finished)])
    _, canaries, _, connection, _ = repositories(script)
    assert run(canaries.finish(finished)) is None
    assert not any("AND status = 'running'" in item.sql for item in connection.statements)


def test_canary_finish_rejects_running_stale_drift_and_terminal_rewrite() -> None:
    running = canary(stage=CanaryStage.SINGLE_PDF)
    _, canaries, pool, _, _ = repositories()
    with pytest.raises(ValueError, match="cannot finish"):
        run(canaries.finish(running))
    assert pool.connections == 0

    candidates = [
        canary(
            stage=CanaryStage.SINGLE_PDF,
            status=CanaryStatus.FAILED,
            attempt=2,
            resumed=True,
        ),
        canary(
            stage=CanaryStage.SINGLE_PDF,
            status=CanaryStatus.FAILED,
            fingerprint="0" * 64,
        ),
    ]
    for candidate in candidates:
        script = Script()
        script.add("FROM knowledge_canary_runs", [canary_row(running)])
        _, canaries, _, connection, _ = repositories(script)
        with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
            run(canaries.finish(candidate))
        assert connection.rollbacks == 1

    terminal = canary(stage=CanaryStage.SINGLE_PDF, status=CanaryStatus.FAILED)
    rewrite = terminal.model_copy(update={"code": "different"}, deep=True)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(terminal)])
    _, canaries, _, _, _ = repositories(script)
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        run(canaries.finish(rewrite))


def test_canary_finish_lost_update_and_missing_claim_fail_closed() -> None:
    running = canary()
    finished = canary(status=CanaryStatus.SUCCEEDED)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(running)])
    script.add("UPDATE knowledge_canary_runs SET status", [])
    _, canaries, _, connection, _ = repositories(script)
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        run(canaries.finish(finished))
    assert connection.rollbacks == 1

    script = Script()
    script.add("FROM knowledge_canary_runs", [])
    _, canaries, _, _, _ = repositories(script)
    with pytest.raises(KnowledgeCanaryError, match="canary_stage_claim_lost"):
        run(canaries.finish(finished))


def test_canary_load_and_invalid_inputs_are_checked_before_connection() -> None:
    record = canary(status=CanaryStatus.SUCCEEDED)
    script = Script()
    script.add("FROM knowledge_canary_runs", [canary_row(record)])
    _, canaries, pool, _, _ = repositories(script)
    assert run(canaries.load(record.run_id, record.stage)) == record
    before = pool.connections
    with pytest.raises(ValueError, match="invalid canary run key"):
        run(canaries.load("INVALID RUN", record.stage))
    with pytest.raises(ValueError, match="invalid canary stage claim"):
        run(
            canaries.begin(
                run_id="body",
                stage=CanaryStage.SINGLE_PDF,
                request_fingerprint=FINGERPRINT,
                source_ids=(SOURCE_ID,),
                approval_id="sk-" + "x" * 24,
            )
        )
    assert pool.connections == before


def test_invalid_stored_canary_and_insert_mismatch_are_safe_failures() -> None:
    invalid = canary_row(canary())
    invalid["attempt"] = 0
    script = Script()
    script.add("FROM knowledge_canary_runs", [invalid])
    _, canaries, _, connection, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError):
        run(canaries.load("run-1", CanaryStage.METADATA_ONLY))
    assert connection.rollbacks == 1

    expected = canary()
    mismatch = canary_row(expected)
    mismatch["code"] = "different"
    script = Script()
    script.add("FROM knowledge_canary_runs", [])
    script.add("INSERT INTO knowledge_canary_runs", [mismatch])
    _, canaries, _, connection, _ = repositories(script)
    with pytest.raises(state.PipelineStatePersistenceError):
        run(
            canaries.begin(
                run_id=expected.run_id,
                stage=expected.stage,
                request_fingerprint=expected.request_fingerprint,
                source_ids=expected.source_ids,
                approval_id=expected.approval_id,
            )
        )
    assert connection.rollbacks == 1


def test_repository_repr_protocol_and_exports_do_not_expose_pool() -> None:
    checkpoints, canaries, _, _, _ = repositories()

    assert repr(checkpoints) == "PostgresFactExtractionCheckpointRepository()"
    assert repr(canaries) == "PostgresCanaryRunRepository()"
    assert "postgresql" not in repr(checkpoints) + repr(canaries)
    assert all(callable(getattr(canaries, name)) for name in ("load", "begin", "finish"))
    assert state.PIPELINE_STATE_SCHEMA_VERSION == "knowledge_0002"
