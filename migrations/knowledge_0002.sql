-- Durable extraction checkpoints and staged canary claims.
-- Apply after knowledge_0001; rollback is kept in knowledge_0002.down.sql.
BEGIN;
SET LOCAL search_path = public;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_fact_extraction_checkpoints (
    idempotency_key text PRIMARY KEY,
    fragment_id uuid NOT NULL,
    source_id uuid NOT NULL,
    fragment_content_sha256 char(64) NOT NULL,
    request_fingerprint char(64) NOT NULL,
    extraction jsonb NOT NULL,
    status text NOT NULL,
    attempts integer NOT NULL,
    batch jsonb,
    last_error_code text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (fragment_id, source_id)
        REFERENCES knowledge_evidence_fragments(fragment_id, source_id),
    CHECK (idempotency_key ~ '^fact-batch-idempotency:v1:[0-9a-f]{64}$'),
    CHECK (fragment_content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CHECK (jsonb_typeof(extraction) = 'object'),
    CHECK (octet_length(extraction::text) <= 65536),
    CHECK (status IN ('running', 'retry_wait', 'completed', 'failed_permanent')),
    CHECK (attempts BETWEEN 1 AND 8),
    CHECK (batch IS NULL OR jsonb_typeof(batch) = 'object'),
    CHECK (batch IS NULL OR octet_length(batch::text) <= 4194304),
    CHECK (batch IS NULL OR position('"relative_path"' IN batch::text) = 0),
    CHECK (
        (status = 'completed' AND batch IS NOT NULL AND last_error_code IS NULL)
        OR
        (status IN ('running', 'retry_wait', 'failed_permanent') AND batch IS NULL)
    ),
    CHECK (
        (status IN ('retry_wait', 'failed_permanent')
            AND last_error_code ~ '^extraction\.[a-z][a-z0-9_]*$')
        OR
        (status IN ('running', 'completed') AND last_error_code IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS knowledge_fact_extraction_fragment_idx
    ON knowledge_fact_extraction_checkpoints(fragment_id, updated_at);
CREATE INDEX IF NOT EXISTS knowledge_fact_extraction_status_idx
    ON knowledge_fact_extraction_checkpoints(status, updated_at, idempotency_key);

CREATE TABLE IF NOT EXISTS knowledge_canary_runs (
    run_id text NOT NULL,
    stage text NOT NULL,
    status text NOT NULL,
    attempt integer NOT NULL,
    code text NOT NULL,
    request_fingerprint char(64) NOT NULL,
    source_ids uuid[] NOT NULL DEFAULT '{}',
    approval_id text,
    attempted_count integer NOT NULL DEFAULT 0,
    completed_count integer NOT NULL DEFAULT 0,
    evidence_count integer NOT NULL DEFAULT 0,
    metadata_records integer NOT NULL DEFAULT 0,
    resumed boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, stage),
    CHECK (run_id ~ '^[a-z0-9][a-z0-9_-]{0,95}$'),
    CHECK (stage IN ('metadata_only', 'single_pdf', 'small_batch')),
    CHECK (status IN ('running', 'succeeded', 'blocked', 'failed', 'cancelled')),
    CHECK (attempt >= 1),
    CHECK (code ~ '^[a-z0-9][a-z0-9_.-]{0,99}$'),
    CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CHECK (cardinality(source_ids) <= 64),
    CHECK (approval_id IS NULL OR approval_id ~ '^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$'),
    CHECK (
        (stage = 'metadata_only' AND cardinality(source_ids) = 0 AND approval_id IS NULL)
        OR
        (stage IN ('single_pdf', 'small_batch') AND approval_id IS NOT NULL)
    ),
    CHECK (attempted_count >= 0),
    CHECK (completed_count BETWEEN 0 AND attempted_count),
    CHECK (evidence_count >= 0 AND metadata_records >= 0),
    CHECK (attempt = 1 OR resumed),
    CHECK (
        status <> 'running'
        OR (
            code = 'canary_running'
            AND attempted_count = 0
            AND completed_count = 0
            AND evidence_count = 0
            AND metadata_records = 0
        )
    )
);

CREATE INDEX IF NOT EXISTS knowledge_canary_status_idx
    ON knowledge_canary_runs(status, updated_at, run_id, stage);

INSERT INTO schema_migrations(version)
VALUES ('knowledge_0002')
ON CONFLICT (version) DO NOTHING;

COMMIT;
