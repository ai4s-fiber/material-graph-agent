from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from typing import Any
from uuid import UUID

import pytest

from material_graph.knowledge import postgres_pipeline_state as pipeline_state
from material_graph.knowledge.age_writer import (
    GraphWriteApproval,
    build_graph_write_approval_request,
)
from material_graph.knowledge.facts import (
    EntityRef,
    EvidenceLink,
    ExtractionProvenance,
    FactBatch,
    RelationAssertion,
)
from material_graph.knowledge.jobs import ApprovedFactBatchError, GraphWriteJobPayload
from material_graph.knowledge.models import SourceLocator
from material_graph.knowledge.reviewed_graph import (
    FactReviewCommand,
    FactReviewConflict,
    PostgresFactReviewRepository,
    _review_digests,
)


NOW = datetime(2026, 7, 27, 9, 30, tzinfo=UTC)
FRAGMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
SOURCE_ID = UUID("22222222-2222-4222-8222-222222222222")
JOB_ID = UUID("33333333-3333-4333-8333-333333333333")


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


class RecordingConnection:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.statements: list[Statement] = []
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
        try:
            yield self
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1


class Pool:
    def __init__(self, connection: RecordingConnection) -> None:
        self.connection_value = connection

    @asynccontextmanager
    async def connection(self):
        yield self.connection_value


def fact_batch() -> FactBatch:
    extraction = ExtractionProvenance(
        extractor_name="fact-extractor",
        extractor_version="1.0",
        generation_id="generation-1",
        model_name="reasoning-model",
        model_version="2026-07-27",
    )
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
        extraction=extraction,
    )
    return FactBatch(
        evidence_fragment_id=FRAGMENT_ID,
        extraction=extraction,
        entities=(material, additive),
        relations=(relation,),
    )


def batch_row(batch: FactBatch) -> dict[str, object]:
    return {
        "fragment_id": FRAGMENT_ID,
        "batch": pipeline_state._redacted_batch(batch, FRAGMENT_ID),
    }


def terminal_review_row(
    batch: FactBatch,
    *,
    decision: str,
    reviewer: str = "reviewer-a",
    comment: str = "checked",
) -> dict[str, object]:
    request = build_graph_write_approval_request(batch)
    command = FactReviewCommand(
        decision=decision,
        reviewer=reviewer,
        comment=comment,
    )
    reviewer_digest, audit_digest = _review_digests(request.batch_id, command)
    approval = GraphWriteApproval(
        batch_id=request.batch_id,
        idempotency_key=request.idempotency_key,
        projection_digest=request.projection_digest,
        approved=decision == "approve",
        reviewer_generation_digest=reviewer_digest,
        audit_generation_digest=audit_digest,
        expires_at=NOW + timedelta(days=30),
    )
    return {
        "batch_id": request.batch_id,
        "fact_batch_idempotency_key": request.idempotency_key,
        "projection_digest": request.projection_digest,
        "status": "approved" if decision == "approve" else "rejected",
        "job_id": JOB_ID if decision == "approve" else None,
        "approval_digest": approval.approval_digest,
        "reviewer_generation_digest": reviewer_digest,
        "audit_generation_digest": audit_digest,
        "approval_expires_at": approval.expires_at,
        "reviewed_at": NOW,
    }


def test_approval_atomically_inserts_content_bound_outbox_review_and_audit() -> None:
    batch = fact_batch()
    review_row = terminal_review_row(batch, decision="approve")
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [batch_row(batch)])
    script.add("FROM knowledge_fact_reviews WHERE batch_id = %s FOR UPDATE", [])
    script.add("INSERT INTO knowledge_worker_jobs", [{"job_id": JOB_ID}])
    script.add("INSERT INTO knowledge_fact_reviews", [review_row])
    connection = RecordingConnection(script)
    repository = PostgresFactReviewRepository(Pool(connection), clock=lambda: NOW)

    record = asyncio.run(
        repository.decide(
            str(batch.batch_id),
            FactReviewCommand(
                decision="approve",
                reviewer="reviewer-a",
                comment="checked",
            ),
        )
    )

    assert record.status == "approved"
    assert record.job_id == JOB_ID
    assert connection.commits == 1
    assert connection.rollbacks == 0
    inserts = [statement for statement in connection.statements if "INSERT INTO" in statement.sql]
    assert [
        next(
            line.strip() for line in statement.sql.splitlines() if line.strip().startswith("INSERT")
        )
        for statement in inserts
    ] == [
        "INSERT INTO knowledge_worker_jobs(",
        "INSERT INTO knowledge_fact_reviews(",
        "INSERT INTO knowledge_graph_write_audit(",
    ]
    encoded_payload = json.loads(str(inserts[0].params[2]))
    assert encoded_payload == {
        "job_type": "graph_write",
        "batch_id": record.batch_id,
        "fact_batch_idempotency_key": record.fact_batch_idempotency_key,
        "projection_digest": record.projection_digest,
        "approval_digest": record.approval_digest,
    }


def test_rejection_is_terminal_and_never_creates_graph_write_job() -> None:
    batch = fact_batch()
    review_row = terminal_review_row(batch, decision="reject")
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [batch_row(batch)])
    script.add("FROM knowledge_fact_reviews WHERE batch_id = %s FOR UPDATE", [])
    script.add("INSERT INTO knowledge_fact_reviews", [review_row])
    connection = RecordingConnection(script)
    repository = PostgresFactReviewRepository(Pool(connection), clock=lambda: NOW)

    record = asyncio.run(
        repository.decide(
            str(batch.batch_id),
            FactReviewCommand(
                decision="reject",
                reviewer="reviewer-a",
                comment="checked",
            ),
        )
    )

    assert record.status == "rejected"
    assert record.job_id is None
    assert not any(
        "INSERT INTO knowledge_worker_jobs" in item.sql for item in connection.statements
    )
    assert any("review_rejected" in item.params for item in connection.statements)


def test_duplicate_terminal_decision_is_stable_and_opposite_decision_conflicts() -> None:
    batch = fact_batch()
    review_row = terminal_review_row(batch, decision="approve")
    script = Script()
    script.add(
        "FROM knowledge_fact_extraction_checkpoints",
        [batch_row(batch)],
        [batch_row(batch)],
    )
    script.add(
        "FROM knowledge_fact_reviews WHERE batch_id = %s FOR UPDATE",
        [review_row],
        [review_row],
    )
    connection = RecordingConnection(script)
    repository = PostgresFactReviewRepository(Pool(connection), clock=lambda: NOW)

    async def exercise() -> None:
        repeated = await repository.decide(
            str(batch.batch_id),
            FactReviewCommand(decision="approve", reviewer="another-reviewer"),
        )
        assert repeated.job_id == JOB_ID
        with pytest.raises(FactReviewConflict):
            await repository.decide(
                str(batch.batch_id),
                FactReviewCommand(decision="reject", reviewer="another-reviewer"),
            )

    asyncio.run(exercise())
    assert not any("INSERT INTO" in item.sql for item in connection.statements)


@pytest.mark.parametrize(
    ("review_state", "code"),
    [(None, "graph_write.review_pending"), ("rejected", "graph_write.review_rejected")],
)
def test_pending_and_rejected_batches_cannot_cross_graph_writer_boundary(
    review_state: str | None,
    code: str,
) -> None:
    batch = fact_batch()
    request = build_graph_write_approval_request(batch)
    approved = terminal_review_row(batch, decision="approve")
    payload = GraphWriteJobPayload(
        batch_id=request.batch_id,
        fact_batch_idempotency_key=request.idempotency_key,
        projection_digest=request.projection_digest,
        approval_digest=str(approved["approval_digest"]),
    )
    script = Script()
    script.add("FROM knowledge_fact_extraction_checkpoints", [batch_row(batch)])
    review_rows: list[Mapping[str, Any]] = []
    if review_state == "rejected":
        review_rows = [terminal_review_row(batch, decision="reject")]
    script.add("FROM knowledge_fact_reviews WHERE batch_id = %s FOR UPDATE", review_rows)
    repository = PostgresFactReviewRepository(Pool(RecordingConnection(script)), clock=lambda: NOW)

    with pytest.raises(ApprovedFactBatchError) as raised:
        asyncio.run(repository.load_approved_batch(payload))
    assert raised.value.code == code
    assert raised.value.retryable is False
