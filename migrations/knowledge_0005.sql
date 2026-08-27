-- Human-reviewed fact batches and transactional graph-write outbox.
-- Apply after knowledge_0004 and before enabling reviewed AGE writes.
BEGIN;
SET LOCAL search_path = public;

ALTER TABLE knowledge_worker_jobs
    DROP CONSTRAINT IF EXISTS knowledge_worker_jobs_type_check;
ALTER TABLE knowledge_worker_jobs
    ADD CONSTRAINT knowledge_worker_jobs_type_check
    CHECK (job_type IN ('metadata_only', 'single_document', 'graph_write'));

ALTER TABLE knowledge_worker_jobs
    DROP CONSTRAINT IF EXISTS knowledge_worker_jobs_graph_write_payload_check;
ALTER TABLE knowledge_worker_jobs
    ADD CONSTRAINT knowledge_worker_jobs_graph_write_payload_check
    CHECK (
        job_type <> 'graph_write'
        OR (
            payload ?& ARRAY[
                'job_type', 'batch_id', 'fact_batch_idempotency_key',
                'projection_digest', 'approval_digest'
            ]
            AND payload
                - 'job_type'
                - 'batch_id'
                - 'fact_batch_idempotency_key'
                - 'projection_digest'
                - 'approval_digest' = '{}'::jsonb
            AND payload ->> 'job_type' = 'graph_write'
            AND payload ->> 'batch_id' ~ '^fact-batch:v1:[0-9a-f]{64}$'
            AND payload ->> 'fact_batch_idempotency_key'
                ~ '^fact-batch-idempotency:v1:[0-9a-f]{64}$'
            AND payload ->> 'projection_digest' ~ '^[0-9a-f]{64}$'
            AND payload ->> 'approval_digest'
                ~ '^graph-approval:v1:[0-9a-f]{64}$'
        )
    );

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_fact_extraction_batch_id_idx
    ON knowledge_fact_extraction_checkpoints ((batch ->> 'batch_id'))
    WHERE status = 'completed' AND batch IS NOT NULL;

CREATE TABLE IF NOT EXISTS knowledge_fact_reviews (
    batch_id text PRIMARY KEY,
    fact_batch_idempotency_key text NOT NULL UNIQUE,
    projection_digest char(64) NOT NULL,
    status text NOT NULL,
    job_id uuid UNIQUE REFERENCES knowledge_worker_jobs(job_id),
    approval_digest text NOT NULL,
    reviewer_generation_digest char(64) NOT NULL,
    audit_generation_digest char(64) NOT NULL,
    approval_expires_at timestamptz NOT NULL,
    reviewed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (batch_id ~ '^fact-batch:v1:[0-9a-f]{64}$'),
    CHECK (fact_batch_idempotency_key ~ '^fact-batch-idempotency:v1:[0-9a-f]{64}$'),
    CHECK (projection_digest ~ '^[0-9a-f]{64}$'),
    CHECK (status IN ('approved', 'rejected')),
    CHECK (approval_digest ~ '^graph-approval:v1:[0-9a-f]{64}$'),
    CHECK (reviewer_generation_digest ~ '^[0-9a-f]{64}$'),
    CHECK (audit_generation_digest ~ '^[0-9a-f]{64}$'),
    CHECK (
        (status = 'approved' AND job_id IS NOT NULL)
        OR (status = 'rejected' AND job_id IS NULL)
    )
);

CREATE OR REPLACE FUNCTION enforce_graph_write_job_review_binding()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    IF NEW.job_type = 'graph_write' AND NOT EXISTS (
        SELECT 1
        FROM knowledge_fact_reviews AS review
        WHERE review.job_id = NEW.job_id
          AND review.status = 'approved'
          AND review.batch_id = NEW.payload ->> 'batch_id'
          AND review.fact_batch_idempotency_key
                = NEW.payload ->> 'fact_batch_idempotency_key'
          AND review.projection_digest = NEW.payload ->> 'projection_digest'
          AND review.approval_digest = NEW.payload ->> 'approval_digest'
    ) THEN
        RAISE EXCEPTION 'graph_write job requires an exact approved fact review';
    END IF;
    RETURN NEW;
END
$function$;

CREATE OR REPLACE FUNCTION enforce_fact_review_job_binding()
RETURNS trigger
LANGUAGE plpgsql
AS $function$
BEGIN
    IF NEW.status = 'approved' AND NOT EXISTS (
        SELECT 1
        FROM knowledge_worker_jobs AS job
        WHERE job.job_id = NEW.job_id
          AND job.job_type = 'graph_write'
          AND job.payload ->> 'batch_id' = NEW.batch_id
          AND job.payload ->> 'fact_batch_idempotency_key'
                = NEW.fact_batch_idempotency_key
          AND job.payload ->> 'projection_digest' = NEW.projection_digest
          AND job.payload ->> 'approval_digest' = NEW.approval_digest
    ) THEN
        RAISE EXCEPTION 'approved fact review requires an exact graph_write job';
    END IF;
    RETURN NEW;
END
$function$;

DROP TRIGGER IF EXISTS knowledge_graph_write_job_review_guard
    ON knowledge_worker_jobs;
CREATE CONSTRAINT TRIGGER knowledge_graph_write_job_review_guard
AFTER INSERT OR UPDATE ON knowledge_worker_jobs
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION enforce_graph_write_job_review_binding();

DROP TRIGGER IF EXISTS knowledge_fact_review_job_guard
    ON knowledge_fact_reviews;
CREATE CONSTRAINT TRIGGER knowledge_fact_review_job_guard
AFTER INSERT OR UPDATE ON knowledge_fact_reviews
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION enforce_fact_review_job_binding();

CREATE TABLE IF NOT EXISTS knowledge_graph_write_audit (
    audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_key text NOT NULL UNIQUE,
    batch_id text NOT NULL REFERENCES knowledge_fact_reviews(batch_id),
    job_id uuid REFERENCES knowledge_worker_jobs(job_id),
    event_type text NOT NULL,
    attempt smallint NOT NULL DEFAULT 0,
    lease_token bigint NOT NULL DEFAULT 0,
    code text NOT NULL,
    approval_digest text NOT NULL,
    reviewer_generation_digest char(64) NOT NULL,
    audit_generation_digest char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (event_key ~ '^graph-write-audit:v1:[0-9a-f]{64}$'),
    CHECK (event_type IN (
        'review_approved', 'review_rejected', 'claimed', 'succeeded',
        'retry_scheduled', 'failed_permanent'
    )),
    CHECK (attempt BETWEEN 0 AND 8),
    CHECK (lease_token >= 0),
    CHECK (code ~ '^[a-z0-9][a-z0-9_.-]{0,99}$'),
    CHECK (approval_digest ~ '^graph-approval:v1:[0-9a-f]{64}$'),
    CHECK (reviewer_generation_digest ~ '^[0-9a-f]{64}$'),
    CHECK (audit_generation_digest ~ '^[0-9a-f]{64}$'),
    CHECK (
        (event_type = 'review_rejected' AND job_id IS NULL)
        OR (event_type <> 'review_rejected' AND job_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS knowledge_graph_write_audit_batch_idx
    ON knowledge_graph_write_audit(batch_id, created_at, audit_id);
CREATE INDEX IF NOT EXISTS knowledge_graph_write_audit_job_idx
    ON knowledge_graph_write_audit(job_id, created_at, audit_id)
    WHERE job_id IS NOT NULL;

INSERT INTO schema_migrations(version)
VALUES ('knowledge_0005')
ON CONFLICT (version) DO NOTHING;

COMMIT;
