-- Durable knowledge repositories. Forward migration only; rollback is separate.
BEGIN;
SET LOCAL search_path = public;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_sources (
    source_id uuid PRIMARY KEY,
    root_id text NOT NULL,
    relative_path text NOT NULL,
    source_version_key text NOT NULL,
    source_kind text NOT NULL,
    display_title text NOT NULL,
    status text NOT NULL,
    directory_year integer,
    normalized_doi text,
    application_number text,
    publication_number text,
    grant_number text,
    legal_status text NOT NULL,
    sha256 char(64),
    byte_size bigint,
    material_category text,
    knowledge_domain text NOT NULL,
    locator jsonb NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    canonical_source_id uuid REFERENCES knowledge_sources(source_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (root_id, relative_path),
    CHECK (root_id ~ '^[a-z0-9][a-z0-9_-]*$'),
    CHECK (relative_path <> ''),
    CHECK (source_version_key ~ '^source-version-v1:[0-9a-f]{64}$'),
    CHECK (source_kind IN (
        'literature', 'patent', 'standard', 'textbook',
        'experiment', 'industrial_data', 'unknown'
    )),
    CHECK (status IN (
        'metadata_discovered', 'metadata_indexed', 'deduplicated',
        'excluded_process_data', 'selected_for_parse', 'spooling', 'parsing',
        'evidence_retained', 'parsed_no_value', 'indexed',
        'failed_retryable', 'failed_permanent'
    )),
    CHECK (directory_year IS NULL OR directory_year BETWEEN 1800 AND 2200),
    CHECK (normalized_doi IS NULL OR normalized_doi = lower(normalized_doi)),
    CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (byte_size IS NULL OR byte_size >= 0),
    CHECK (jsonb_typeof(locator) = 'object'),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (octet_length(metadata::text) <= 262144),
    CHECK (canonical_source_id IS NULL OR canonical_source_id <> source_id)
);

-- SHA is intentionally non-unique: physical duplicates remain auditable and
-- canonical_source_id links them. DOI+SHA distinguishes versions of one work.
CREATE INDEX IF NOT EXISTS knowledge_sources_sha_dedupe_idx
    ON knowledge_sources(sha256, source_id) WHERE sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS knowledge_sources_doi_sha_version_idx
    ON knowledge_sources(normalized_doi, sha256, source_id)
    WHERE normalized_doi IS NOT NULL AND sha256 IS NOT NULL;
CREATE INDEX IF NOT EXISTS knowledge_sources_canonical_idx
    ON knowledge_sources(canonical_source_id)
    WHERE canonical_source_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS knowledge_source_relations (
    relation_type text NOT NULL,
    source_id uuid NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    target_source_id uuid NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    normalized_doi text,
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (relation_type, source_id, target_source_id),
    UNIQUE (relation_type, normalized_doi, source_id),
    CHECK (relation_type IN ('DUPLICATE_OF', 'IS_VERSION_OF')),
    CHECK (source_id <> target_source_id),
    CHECK (relation_type <> 'IS_VERSION_OF' OR normalized_doi IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS knowledge_source_relations_target_idx
    ON knowledge_source_relations(target_source_id, relation_type);

CREATE TABLE IF NOT EXISTS knowledge_ingestion_checkpoints (
    idempotency_key text PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES knowledge_sources(source_id),
    source_version_fingerprint char(64) NOT NULL,
    embedding_generation_id text NOT NULL,
    lifecycle_status text NOT NULL,
    stage text NOT NULL,
    job_status text NOT NULL,
    attempt integer NOT NULL DEFAULT 0,
    selection jsonb,
    cursor jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    last_error_category text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_id, source_version_fingerprint, embedding_generation_id),
    UNIQUE (idempotency_key, source_id),
    CHECK (idempotency_key ~ '^knowledge-ingestion:v2:[0-9a-f]{32}:[0-9a-f]{64}$'),
    CHECK (split_part(idempotency_key, ':', 3) = replace(source_id::text, '-', '')),
    CHECK (source_version_fingerprint ~ '^[0-9a-f]{64}$'),
    CHECK (embedding_generation_id <> ''),
    CHECK (lifecycle_status IN (
        'metadata_discovered', 'metadata_indexed', 'deduplicated',
        'excluded_process_data', 'parse_eligible', 'evidence_retained',
        'parsed_no_value', 'failed_permanent'
    )),
    CHECK (stage IN ('catalog', 'hash', 'select', 'spool', 'parse', 'retain', 'index')),
    CHECK (job_status IN (
        'queued', 'running', 'retry_wait', 'succeeded', 'failed_retryable',
        'failed_permanent', 'cancelled'
    )),
    CHECK (attempt >= 0),
    CHECK (selection IS NULL OR jsonb_typeof(selection) = 'object'),
    CHECK (jsonb_typeof(cursor) = 'object'),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (octet_length(cursor::text) <= 262144),
    CHECK (octet_length(metadata::text) <= 262144),
    CHECK (metadata->>'source_version_fingerprint' = source_version_fingerprint),
    CHECK (metadata->>'embedding_generation_id' = embedding_generation_id)
);

CREATE INDEX IF NOT EXISTS knowledge_checkpoints_resume_idx
    ON knowledge_ingestion_checkpoints(job_status, stage, updated_at);

CREATE TABLE IF NOT EXISTS knowledge_evidence_fragments (
    fragment_id uuid PRIMARY KEY,
    source_id uuid NOT NULL REFERENCES knowledge_sources(source_id),
    idempotency_key text NOT NULL,
    text text NOT NULL,
    locator jsonb NOT NULL,
    content_sha256 char(64) NOT NULL,
    retention_reason text NOT NULL,
    supported_entity_ids text[] NOT NULL DEFAULT '{}',
    supported_relation_ids text[] NOT NULL DEFAULT '{}',
    parser_name text NOT NULL,
    parser_version text NOT NULL,
    embedding_generation_id text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (idempotency_key, source_id)
        REFERENCES knowledge_ingestion_checkpoints(idempotency_key, source_id),
    UNIQUE (fragment_id, source_id),
    UNIQUE (fragment_id, source_id, content_sha256, embedding_generation_id),
    UNIQUE (source_id, idempotency_key, fragment_id),
    CHECK (text <> ''),
    CHECK (char_length(text) <= 65536),
    CHECK (left(ltrim(text), 5) <> '%PDF-'),
    CHECK (jsonb_typeof(locator) = 'object'),
    CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (retention_reason <> ''),
    CHECK (parser_name <> '' AND parser_version <> ''),
    CHECK (embedding_generation_id <> ''),
    CHECK (jsonb_typeof(metadata) = 'object'),
    CHECK (octet_length(metadata::text) <= 262144)
);

CREATE INDEX IF NOT EXISTS knowledge_evidence_source_run_idx
    ON knowledge_evidence_fragments(source_id, idempotency_key, fragment_id);

CREATE TABLE IF NOT EXISTS knowledge_lightrag_source_mappings (
    basename text PRIMARY KEY,
    fragment_id uuid NOT NULL UNIQUE,
    source_id uuid NOT NULL,
    locator jsonb NOT NULL,
    logical_source_uri text NOT NULL,
    content_sha256 char(64) NOT NULL,
    embedding_generation_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (fragment_id, source_id, content_sha256, embedding_generation_id)
        REFERENCES knowledge_evidence_fragments(
            fragment_id, source_id, content_sha256, embedding_generation_id
        ),
    CHECK (basename ~ '^mg_[0-9a-f]{32}_[0-9a-f]{32}_[0-9a-f]{16}\.txt$'),
    CHECK (
        basename = 'mg_' || replace(source_id::text, '-', '') || '_' ||
            replace(fragment_id::text, '-', '') || '_' || left(content_sha256, 16) || '.txt'
    ),
    CHECK (jsonb_typeof(locator) = 'object'),
    CHECK (logical_source_uri ~ '^source://'),
    CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    CHECK (embedding_generation_id <> '')
);

CREATE INDEX IF NOT EXISTS knowledge_lightrag_source_idx
    ON knowledge_lightrag_source_mappings(source_id, fragment_id);

INSERT INTO schema_migrations(version)
VALUES ('knowledge_0001')
ON CONFLICT (version) DO NOTHING;

COMMIT;
