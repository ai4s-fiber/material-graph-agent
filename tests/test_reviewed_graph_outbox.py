from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from material_graph.knowledge.age_writer import KnowledgeGraphPersistenceError
from material_graph.knowledge.facts import FactWriteResult
from material_graph.knowledge.jobs import (
    ApprovedFactBatchError,
    GraphWriteJobExecutor,
    GraphWriteJobPayload,
    InMemoryKnowledgeJobRepository,
    KnowledgeJobExecutionError,
)
from material_graph.knowledge.reviewed_graph import (
    FactReviewRecord,
    build_graph_write_audit_key,
)


ROOT = Path(__file__).resolve().parents[1]


def graph_payload() -> GraphWriteJobPayload:
    return GraphWriteJobPayload(
        batch_id="fact-batch:v1:" + "a" * 64,
        fact_batch_idempotency_key="fact-batch-idempotency:v1:" + "b" * 64,
        projection_digest="c" * 64,
        approval_digest="graph-approval:v1:" + "d" * 64,
    )


def test_graph_write_jobs_can_only_enter_through_review_outbox() -> None:
    repository = InMemoryKnowledgeJobRepository()
    with pytest.raises(ValueError, match="graph_write_jobs_require_review_outbox"):
        asyncio.run(repository.enqueue(graph_payload()))


def test_graph_write_executor_returns_idempotent_writer_result() -> None:
    payload = graph_payload()

    class ApprovedBatches:
        async def load_approved_batch(self, value: GraphWriteJobPayload) -> object:
            assert value == payload
            return object()

    class Writer:
        async def write_batch(self, _batch: object) -> FactWriteResult:
            return FactWriteResult(
                batch_id=payload.batch_id,
                idempotency_key=payload.fact_batch_idempotency_key,
                status="already_present",
                node_count=4,
                edge_count=3,
            )

    result = asyncio.run(
        GraphWriteJobExecutor(
            approved_batches=ApprovedBatches(),  # type: ignore[arg-type]
            graph_writer=Writer(),  # type: ignore[arg-type]
        ).execute(payload)
    )
    assert result.job_type == "graph_write"
    assert result.graph_write_status == "already_present"
    assert (result.node_count, result.edge_count) == (4, 3)


def test_graph_write_executor_never_calls_writer_for_pending_review() -> None:
    payload = graph_payload()
    calls = 0

    class PendingBatches:
        async def load_approved_batch(self, _value: GraphWriteJobPayload) -> object:
            raise ApprovedFactBatchError("graph_write.review_pending", retryable=False)

    class Writer:
        async def write_batch(self, _batch: object) -> FactWriteResult:
            nonlocal calls
            calls += 1
            raise AssertionError("writer must not be called")

    executor = GraphWriteJobExecutor(
        approved_batches=PendingBatches(),  # type: ignore[arg-type]
        graph_writer=Writer(),  # type: ignore[arg-type]
    )
    with pytest.raises(KnowledgeJobExecutionError) as raised:
        asyncio.run(executor.execute(payload))
    assert raised.value.code == "graph_write.review_pending"
    assert raised.value.retryable is False
    assert calls == 0


def test_graph_write_persistence_failure_is_retryable_without_changing_payload() -> None:
    payload = graph_payload()

    class ApprovedBatches:
        async def load_approved_batch(self, _value: GraphWriteJobPayload) -> object:
            return object()

    class FailingWriter:
        async def write_batch(self, _batch: object) -> FactWriteResult:
            raise KnowledgeGraphPersistenceError("knowledge_graph_persistence_error")

    executor = GraphWriteJobExecutor(
        approved_batches=ApprovedBatches(),  # type: ignore[arg-type]
        graph_writer=FailingWriter(),  # type: ignore[arg-type]
    )
    with pytest.raises(KnowledgeJobExecutionError) as raised:
        asyncio.run(executor.execute(payload))
    assert raised.value.code == "graph_write.persistence_failed"
    assert raised.value.retryable is True


def test_review_state_and_audit_identity_are_terminal_and_deterministic() -> None:
    payload = graph_payload()
    pending = FactReviewRecord(
        batch_id=payload.batch_id,
        fact_batch_idempotency_key=payload.fact_batch_idempotency_key,
        projection_digest=payload.projection_digest,
        status="pending",
    )
    assert pending.job_id is None

    job_id = UUID("11111111-1111-4111-8111-111111111111")
    reviewed_at = datetime(2026, 7, 27, tzinfo=UTC)
    approved = FactReviewRecord(
        batch_id=payload.batch_id,
        fact_batch_idempotency_key=payload.fact_batch_idempotency_key,
        projection_digest=payload.projection_digest,
        status="approved",
        job_id=job_id,
        approval_digest=payload.approval_digest,
        reviewer_generation_digest="e" * 64,
        audit_generation_digest="f" * 64,
        approval_expires_at=reviewed_at + timedelta(days=30),
        reviewed_at=reviewed_at,
    )
    assert approved.job_id == job_id
    first = build_graph_write_audit_key(
        batch_id=payload.batch_id,
        event_type="review_approved",
        job_id=job_id,
        attempt=0,
        lease_token=0,
        code="approved",
    )
    assert first == build_graph_write_audit_key(
        batch_id=payload.batch_id,
        event_type="review_approved",
        job_id=job_id,
        attempt=0,
        lease_token=0,
        code="approved",
    )


def test_review_outbox_migration_enforces_approved_only_job_binding() -> None:
    forward = (ROOT / "migrations" / "knowledge_0005.sql").read_text(encoding="utf-8")
    down = (ROOT / "migrations" / "knowledge_0005.down.sql").read_text(encoding="utf-8")

    assert "'graph_write'" in forward
    assert "status = 'approved' AND job_id IS NOT NULL" in forward
    assert "status = 'rejected' AND job_id IS NULL" in forward
    assert "knowledge_graph_write_audit" in forward
    assert "DEFERRABLE INITIALLY DEFERRED" in forward
    assert "graph_write job requires an exact approved fact review" in forward
    assert "VALUES ('knowledge_0005')" in forward
    assert "cannot roll back knowledge_0005 while graph_write jobs exist" in down
