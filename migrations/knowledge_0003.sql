-- Durable knowledge worker queue with fenced leases and bounded payloads.
-- Apply after knowledge_0002; rollback is kept in knowledge_0003.down.sql.
BEGIN;
SET LOCAL search_path = public;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_worker_jobs (
    job_id uuid PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    job_type text NOT NULL,
    payload jsonb NOT NULL,
    status text NOT NULL,
    attempt smallint NOT NULL DEFAULT 0,
    max_attempts smallint NOT NULL DEFAULT 4,
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_token bigint NOT NULL DEFAULT 0,
    lease_until timestamptz,
    result jsonb,
    last_error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,

    CONSTRAINT knowledge_worker_jobs_key_check
        CHECK (idempotency_key ~ '^knowledge-job:v1:[0-9a-f]{64}$'),
    CONSTRAINT knowledge_worker_jobs_type_check
        CHECK (job_type IN ('metadata_only', 'single_document')),
    CONSTRAINT knowledge_worker_jobs_payload_check
        CHECK (
            jsonb_typeof(payload) = 'object'
            AND payload ->> 'job_type' = job_type
            AND octet_length(payload::text) <= 262144
        ),
    CONSTRAINT knowledge_worker_jobs_status_check
        CHECK (
            status IN (
                'queued', 'running', 'retry_wait', 'succeeded',
                'failed_permanent', 'cancelled'
            )
        ),
    CONSTRAINT knowledge_worker_jobs_attempt_check
        CHECK (
            attempt BETWEEN 0 AND 8
            AND max_attempts BETWEEN 1 AND 8
            AND attempt <= max_attempts
        ),
    CONSTRAINT knowledge_worker_jobs_queue_attempt_check
        CHECK (status <> 'queued' OR attempt = 0),
    CONSTRAINT knowledge_worker_jobs_retry_attempt_check
        CHECK (status <> 'retry_wait' OR attempt < max_attempts),
    CONSTRAINT knowledge_worker_jobs_lease_token_check
        CHECK (lease_token >= 0),
    CONSTRAINT knowledge_worker_jobs_lease_owner_check
        CHECK (
            lease_owner IS NULL
            OR lease_owner ~ '^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$'
        ),
    CONSTRAINT knowledge_worker_jobs_lease_state_check
        CHECK (
            (status = 'running')
            = (lease_owner IS NOT NULL AND lease_until IS NOT NULL)
        ),
    CONSTRAINT knowledge_worker_jobs_running_attempt_check
        CHECK (status <> 'running' OR (attempt >= 1 AND lease_token >= 1)),
    CONSTRAINT knowledge_worker_jobs_result_check
        CHECK (
            (status = 'succeeded') = (result IS NOT NULL)
            AND (
                result IS NULL
                OR (
                    jsonb_typeof(result) = 'object'
                    AND octet_length(result::text) <= 65536
                )
            )
        ),
    CONSTRAINT knowledge_worker_jobs_error_code_check
        CHECK (
            last_error_code IS NULL
            OR last_error_code ~ '^[a-z0-9][a-z0-9_.-]{0,99}$'
        ),
    CONSTRAINT knowledge_worker_jobs_error_state_check
        CHECK (
            (status IN ('retry_wait', 'failed_permanent'))
            = (last_error_code IS NOT NULL)
        ),
    CONSTRAINT knowledge_worker_jobs_completed_state_check
        CHECK (
            (status IN ('succeeded', 'failed_permanent', 'cancelled'))
            = (completed_at IS NOT NULL)
        )
);

CREATE INDEX IF NOT EXISTS knowledge_worker_jobs_claim_ready_idx
    ON knowledge_worker_jobs (available_at, created_at, job_id)
    WHERE status IN ('queued', 'retry_wait');

CREATE INDEX IF NOT EXISTS knowledge_worker_jobs_claim_expired_idx
    ON knowledge_worker_jobs (lease_until, created_at, job_id)
    WHERE status = 'running';

INSERT INTO schema_migrations(version)
VALUES ('knowledge_0003')
ON CONFLICT (version) DO NOTHING;

COMMIT;
