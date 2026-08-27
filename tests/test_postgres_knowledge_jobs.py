from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from material_graph.knowledge import postgres_jobs as pgjobs
from material_graph.knowledge.jobs import (
    GraphWriteJobPayload,
    KnowledgeJobLease,
    KnowledgeJobLeaseLost,
    KnowledgeJobResult,
    KnowledgeJobStatus,
    MetadataOnlyJobPayload,
    build_knowledge_job_idempotency_key,
)
from material_graph.knowledge.postgres_jobs import (
    KnowledgeJobPersistenceError,
    PostgresKnowledgeJobRepository,
)
from material_graph.knowledge.service import MetadataCanaryRequest


ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 7, 27, tzinfo=UTC)
JOB_ID = UUID("11111111-1111-4111-8111-111111111111")
GRAPH_BATCH_ID = "fact-batch:v1:" + "a" * 64
GRAPH_FACT_KEY = "fact-batch-idempotency:v1:" + "b" * 64


@dataclass(frozen=True)
class Statement:
    sql: str
    params: tuple[object, ...]


class Script:
    def __init__(self) -> None:
        self.responses: dict[str, deque[list[Mapping[str, Any]]]] = defaultdict(deque)

    def add(self, needle: str, *responses: Sequence[Mapping[str, Any]]) -> None:
        for response in responses:
            self.responses[needle].append(list(response))

    def take(self, sql: str) -> list[Mapping[str, Any]]:
        compact = " ".join(sql.split())
        for needle, queued in self.responses.items():
            if needle in compact and queued:
                return queued.popleft()
        return []


class Cursor:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    async def fetchone(self) -> Mapping[str, Any] | None:
        return None if not self.rows else self.rows[0]

    async def fetchall(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


class RecordingConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
        self.transactions = 0

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
        yield self


class Pool:
    def __init__(self, connection: RecordingConnection) -> None:
        self.value = connection
        self.connections = 0

    @asynccontextmanager
    async def connection(self):
        self.connections += 1
        yield self.value


def payload() -> MetadataOnlyJobPayload:
    return MetadataOnlyJobPayload(
        request=MetadataCanaryRequest(
            run_id="metadata-run",
            root_id="document_data_1",
            slice_id="literature",
            manifest_path="private/manifests/catalog.jsonl",
            manifest_format="jsonl",
        )
    )


def result() -> KnowledgeJobResult:
    return KnowledgeJobResult(
        job_type="metadata_only",
        run_id="metadata-run",
        code="metadata_completed",
        metadata_records=4,
    )


def graph_payload() -> GraphWriteJobPayload:
    return GraphWriteJobPayload(
        batch_id=GRAPH_BATCH_ID,
        fact_batch_idempotency_key=GRAPH_FACT_KEY,
        projection_digest="c" * 64,
        approval_digest="graph-approval:v1:" + "d" * 64,
    )


def graph_result(*, status: str = "written") -> KnowledgeJobResult:
    return KnowledgeJobResult(
        job_type="graph_write",
        code="graph_write_completed",
        batch_id=GRAPH_BATCH_ID,
        fact_batch_idempotency_key=GRAPH_FACT_KEY,
        graph_write_status=status,
        node_count=4,
        edge_count=3,
    )


def row(
    status: KnowledgeJobStatus = KnowledgeJobStatus.QUEUED,
    *,
    attempt: int = 0,
    owner: str | None = None,
    token: int = 0,
    lease_until: datetime | None = None,
    stored_result: KnowledgeJobResult | None = None,
    error: str | None = None,
    json_strings: bool = False,
) -> dict[str, object]:
    job_payload: object = payload().model_dump(mode="json")
    result_payload: object = (
        None if stored_result is None else stored_result.model_dump(mode="json")
    )
    if json_strings:
        job_payload = json.dumps(job_payload)
        if result_payload is not None:
            result_payload = json.dumps(result_payload)
    return {
        "job_id": JOB_ID,
        "idempotency_key": build_knowledge_job_idempotency_key(payload()),
        "job_type": "metadata_only",
        "payload": job_payload,
        "status": status.value,
        "attempt": attempt,
        "max_attempts": 4,
        "available_at": NOW,
        "lease_owner": owner,
        "lease_token": token,
        "lease_until": lease_until,
        "result": result_payload,
        "last_error_code": error,
    }


def graph_row(
    status: KnowledgeJobStatus,
    *,
    attempt: int,
    owner: str | None = None,
    token: int = 1,
    lease_until: datetime | None = None,
    stored_result: KnowledgeJobResult | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "job_id": JOB_ID,
        "idempotency_key": build_knowledge_job_idempotency_key(graph_payload()),
        "job_type": "graph_write",
        "payload": graph_payload().model_dump(mode="json"),
        "status": status.value,
        "attempt": attempt,
        "max_attempts": 4,
        "available_at": NOW,
        "lease_owner": owner,
        "lease_token": token,
        "lease_until": lease_until,
        "result": (None if stored_result is None else stored_result.model_dump(mode="json")),
        "last_error_code": error,
    }


def running_row(*, owner: str = "worker-a", token: int = 1) -> dict[str, object]:
    return row(
        KnowledgeJobStatus.RUNNING,
        attempt=1,
        owner=owner,
        token=token,
        lease_until=NOW + timedelta(minutes=1),
    )


def lease_from(raw: Mapping[str, object]) -> KnowledgeJobLease:
    script = Script()
    script.add("WHERE job_id = %s", [raw])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))
    record = asyncio.run(repository.get(JOB_ID))
    return KnowledgeJobLease(
        record=record,
        worker_id=str(raw["lease_owner"]),
        lease_token=int(raw["lease_token"]),
        lease_until=raw["lease_until"],  # type: ignore[arg-type]
    )


def test_enqueue_is_idempotent_and_uses_bounded_json() -> None:
    script = Script()
    queued = row(json_strings=True)
    script.add("ON CONFLICT (idempotency_key) DO NOTHING", [queued], [])
    script.add("WHERE idempotency_key = %s", [queued])
    connection = RecordingConnection(script)
    repository = PostgresKnowledgeJobRepository(Pool(connection))

    async def exercise() -> None:
        first = await repository.enqueue(payload())
        repeated = await repository.enqueue(payload())
        assert first == repeated

    asyncio.run(exercise())
    assert connection.transactions == 2
    insert = connection.statements[0]
    assert insert.params[3] == json.dumps(
        payload().model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_claim_renew_and_complete_are_fenced() -> None:
    script = Script()
    claimed = running_row()
    renewed = {**claimed, "lease_until": NOW + timedelta(minutes=2)}
    succeeded = row(
        KnowledgeJobStatus.SUCCEEDED,
        attempt=1,
        token=1,
        stored_result=result(),
    )
    script.add("FOR UPDATE SKIP LOCKED", [claimed])
    script.add("SET lease_until = now()", [renewed])
    script.add("SET status = 'succeeded'", [succeeded])
    connection = RecordingConnection(script)
    repository = PostgresKnowledgeJobRepository(Pool(connection))

    async def exercise() -> None:
        lease = await repository.claim("worker-a", lease_seconds=60)
        assert lease is not None
        renewed_lease = await repository.renew(lease, lease_seconds=60)
        completed = await repository.complete(renewed_lease, result())
        assert completed.status is KnowledgeJobStatus.SUCCEEDED

    asyncio.run(exercise())
    claim_sql = " ".join(connection.statements[0].sql.split())
    assert "FOR UPDATE SKIP LOCKED" in claim_sql
    for statement in connection.statements[1:]:
        compact = " ".join(statement.sql.split())
        assert "lease_owner = %s" in compact
        assert "lease_token = %s" in compact
        assert "lease_until > now()" in compact


def test_complete_rejects_an_old_fencing_token() -> None:
    stale = lease_from(running_row())
    script = Script()
    script.add("SET status = 'succeeded'", [])
    script.add("WHERE job_id = %s", [running_row(owner="worker-b", token=2)])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))

    with pytest.raises(KnowledgeJobLeaseLost):
        asyncio.run(repository.complete(stale, result()))


def test_complete_reports_lease_loss_when_job_disappears() -> None:
    lease = lease_from(running_row())
    script = Script()
    script.add("SET status = 'succeeded'", [])
    script.add("WHERE job_id = %s", [])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))
    with pytest.raises(KnowledgeJobLeaseLost):
        asyncio.run(repository.complete(lease, result()))


def test_fail_is_idempotent_and_schedules_retry() -> None:
    active = running_row()
    lease = lease_from(active)
    retrying = row(
        KnowledgeJobStatus.RETRY_WAIT,
        attempt=1,
        token=1,
        error="provider.rate_limited",
    )
    script = Script()
    script.add("SET status = %s", [retrying], [])
    script.add("WHERE job_id = %s", [retrying])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))

    async def exercise() -> None:
        first = await repository.fail(
            lease,
            code="provider.rate_limited",
            retryable=True,
            retry_delay_seconds=5,
        )
        repeated = await repository.fail(
            lease,
            code="provider.rate_limited",
            retryable=True,
            retry_delay_seconds=5,
        )
        assert first == repeated
        assert first.status is KnowledgeJobStatus.RETRY_WAIT

    asyncio.run(exercise())


def test_worker_migration_has_claim_and_state_safety_constraints() -> None:
    migration = (ROOT / "migrations" / "knowledge_0003.sql").read_text(encoding="utf-8")
    down = (ROOT / "migrations" / "knowledge_0003.down.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS knowledge_worker_jobs" in migration
    assert "knowledge_worker_jobs_claim_ready_idx" in migration
    assert "knowledge_worker_jobs_claim_expired_idx" in migration
    assert "knowledge_worker_jobs_lease_state_check" in migration
    assert "knowledge_worker_jobs_result_check" in migration
    assert "VALUES ('knowledge_0003')" in migration
    assert "DROP TABLE IF EXISTS knowledge_worker_jobs" in down
    assert "DELETE FROM schema_migrations WHERE version = 'knowledge_0003'" in down


def test_repository_argument_and_payload_guards_fail_closed() -> None:
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(Script())))
    with pytest.raises(ValueError, match="max_attempts"):
        asyncio.run(repository.enqueue(payload(), max_attempts=0))
    with pytest.raises(ValueError, match="invalid worker id"):
        asyncio.run(repository.claim("invalid worker!", lease_seconds=30))
    with pytest.raises(ValueError, match="lease_seconds"):
        asyncio.run(repository.claim("worker-a", lease_seconds=0))
    with pytest.raises(ValueError, match="knowledge_job_payload_too_large"):
        pgjobs._bounded_json({"value": "x" * 300_000}, maximum=10)
    with pytest.raises(ValueError, match="knowledge_job_json_object_required"):
        pgjobs._json_object([])
    with pytest.raises(KnowledgeJobPersistenceError):
        asyncio.run(repository.enqueue(payload()))
    with pytest.raises(KeyError):
        asyncio.run(repository.get(JOB_ID))
    assert asyncio.run(repository.claim("worker-a", lease_seconds=30)) is None


def test_row_contract_mismatch_is_a_stable_persistence_error() -> None:
    mismatched = {**row(), "job_type": "single_document"}
    script = Script()
    script.add("WHERE job_id = %s", [mismatched])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))
    with pytest.raises(KnowledgeJobPersistenceError, match="knowledge_job_persistence_failed"):
        asyncio.run(repository.get(JOB_ID))


def test_renew_without_matching_fence_reports_lease_loss() -> None:
    active = running_row()
    lease = lease_from(active)
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(Script())))
    with pytest.raises(KnowledgeJobLeaseLost):
        asyncio.run(repository.renew(lease, lease_seconds=60))


def test_complete_can_replay_an_identical_result() -> None:
    active = running_row()
    lease = lease_from(active)
    succeeded = row(
        KnowledgeJobStatus.SUCCEEDED,
        attempt=1,
        token=1,
        stored_result=result(),
        json_strings=True,
    )
    script = Script()
    script.add("SET status = 'succeeded'", [])
    script.add("WHERE job_id = %s", [succeeded])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))
    replayed = asyncio.run(repository.complete(lease, result()))
    assert replayed.status is KnowledgeJobStatus.SUCCEEDED
    assert replayed.result == result()


def test_final_attempt_becomes_permanent_and_invalid_delay_is_rejected() -> None:
    active = running_row()
    active["attempt"] = 4
    lease = lease_from(active)
    permanent = row(
        KnowledgeJobStatus.FAILED_PERMANENT,
        attempt=4,
        token=1,
        error="provider.rate_limited",
    )
    script = Script()
    script.add("SET status = %s", [permanent])
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(script)))
    failed = asyncio.run(
        repository.fail(
            lease,
            code="provider.rate_limited",
            retryable=True,
            retry_delay_seconds=5,
        )
    )
    assert failed.status is KnowledgeJobStatus.FAILED_PERMANENT
    with pytest.raises(ValueError, match="retry delay"):
        asyncio.run(
            repository.fail(
                lease,
                code="provider.rate_limited",
                retryable=True,
                retry_delay_seconds=-1,
            )
        )


def _graph_lease(*, attempt: int = 1, token: int = 1) -> KnowledgeJobLease:
    raw = graph_row(
        KnowledgeJobStatus.RUNNING,
        attempt=attempt,
        owner="worker-a",
        token=token,
        lease_until=NOW + timedelta(minutes=1),
    )
    record = pgjobs._record_from_row(raw)
    return KnowledgeJobLease(
        record=record,
        worker_id="worker-a",
        lease_token=token,
        lease_until=raw["lease_until"],  # type: ignore[arg-type]
    )


def test_graph_write_cannot_use_generic_enqueue_boundary() -> None:
    repository = PostgresKnowledgeJobRepository(Pool(RecordingConnection(Script())))
    with pytest.raises(ValueError, match="graph_write_jobs_require_review_outbox"):
        asyncio.run(repository.enqueue(graph_payload()))


def test_graph_write_claim_and_retry_are_audited_in_job_transactions() -> None:
    claimed = graph_row(
        KnowledgeJobStatus.RUNNING,
        attempt=1,
        owner="worker-a",
        token=1,
        lease_until=NOW + timedelta(minutes=1),
    )
    script = Script()
    script.add("FOR UPDATE SKIP LOCKED", [claimed])
    script.add("INSERT INTO knowledge_graph_write_audit", [{"audit_id": 1}])
    connection = RecordingConnection(script)
    repository = PostgresKnowledgeJobRepository(Pool(connection))

    lease = asyncio.run(repository.claim("worker-a", lease_seconds=60))
    assert lease is not None
    claim_audit = next(
        item for item in connection.statements if "knowledge_graph_write_audit" in item.sql
    )
    assert claim_audit.params[2:6] == ("claimed", 1, 1, "claimed")

    retry = graph_row(
        KnowledgeJobStatus.RETRY_WAIT,
        attempt=1,
        token=1,
        error="graph_write.persistence_failed",
    )
    retry_script = Script()
    retry_script.add("SET status = %s", [retry])
    retry_script.add("INSERT INTO knowledge_graph_write_audit", [{"audit_id": 2}])
    retry_connection = RecordingConnection(retry_script)
    retry_repository = PostgresKnowledgeJobRepository(Pool(retry_connection))
    failed = asyncio.run(
        retry_repository.fail(
            _graph_lease(),
            code="graph_write.persistence_failed",
            retryable=True,
            retry_delay_seconds=5,
        )
    )
    assert failed.status is KnowledgeJobStatus.RETRY_WAIT
    retry_audit = next(
        item for item in retry_connection.statements if "knowledge_graph_write_audit" in item.sql
    )
    assert retry_audit.params[2] == "retry_scheduled"
    assert retry_audit.params[5] == "graph_write.persistence_failed"


def test_graph_write_completion_replay_reuses_one_deterministic_audit_identity() -> None:
    succeeded = graph_row(
        KnowledgeJobStatus.SUCCEEDED,
        attempt=1,
        token=1,
        stored_result=graph_result(),
    )
    script = Script()
    script.add("SET status = 'succeeded'", [succeeded], [])
    script.add("WHERE job_id = %s", [succeeded])
    script.add(
        "INSERT INTO knowledge_graph_write_audit",
        [{"audit_id": 1}],
        [],
    )
    script.add("SELECT event_key FROM knowledge_graph_write_audit", [{"event_key": "stored"}])
    connection = RecordingConnection(script)
    repository = PostgresKnowledgeJobRepository(Pool(connection))

    async def exercise() -> None:
        first = await repository.complete(_graph_lease(), graph_result())
        repeated = await repository.complete(_graph_lease(), graph_result())
        assert first == repeated

    asyncio.run(exercise())
    audit_inserts = [
        item
        for item in connection.statements
        if "INSERT INTO knowledge_graph_write_audit" in item.sql
    ]
    assert len(audit_inserts) == 2
    assert audit_inserts[0].params[0] == audit_inserts[1].params[0]
    assert audit_inserts[0].params[2] == "succeeded"


def test_graph_write_final_failure_is_audited_as_permanent() -> None:
    permanent = graph_row(
        KnowledgeJobStatus.FAILED_PERMANENT,
        attempt=4,
        token=4,
        error="graph_write.conflict",
    )
    permanent["max_attempts"] = 4
    script = Script()
    script.add("SET status = %s", [permanent])
    script.add("INSERT INTO knowledge_graph_write_audit", [{"audit_id": 3}])
    connection = RecordingConnection(script)
    repository = PostgresKnowledgeJobRepository(Pool(connection))

    failed = asyncio.run(
        repository.fail(
            _graph_lease(attempt=4, token=4),
            code="graph_write.conflict",
            retryable=False,
            retry_delay_seconds=5,
        )
    )
    assert failed.status is KnowledgeJobStatus.FAILED_PERMANENT
    audit = next(
        item for item in connection.statements if "knowledge_graph_write_audit" in item.sql
    )
    assert audit.params[2] == "failed_permanent"
    assert audit.params[3:6] == (4, 4, "graph_write.conflict")
