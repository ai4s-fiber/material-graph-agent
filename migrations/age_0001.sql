-- Apache AGE global knowledge graph writer ledger and safe relational mirror.
-- Forward migration only; rollback is intentionally kept in age_0001.down.sql.
BEGIN;

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET LOCAL search_path = ag_catalog, "$user", public;

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'material_graph'
    ) THEN
        PERFORM ag_catalog.create_graph('material_graph');
    END IF;
END
$migration$;

SET LOCAL search_path = public, ag_catalog;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_graph_batches (
    idempotency_key text PRIMARY KEY,
    batch_id text NOT NULL UNIQUE,
    projection_digest char(64) NOT NULL,
    node_count integer NOT NULL,
    edge_count integer NOT NULL,
    approval_status text NOT NULL,
    approval_digest text NOT NULL,
    reviewer_generation_digest char(64) NOT NULL,
    audit_generation_digest char(64) NOT NULL,
    approval_expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (idempotency_key ~ '^fact-batch-idempotency:v1:[0-9a-f]{64}$'),
    CHECK (batch_id ~ '^fact-batch:v1:[0-9a-f]{64}$'),
    CHECK (projection_digest ~ '^[0-9a-f]{64}$'),
    CHECK (node_count >= 0 AND edge_count >= 0),
    CHECK (approval_status = 'approved'),
    CHECK (approval_digest ~ '^graph-approval:v1:[0-9a-f]{64}$'),
    CHECK (reviewer_generation_digest ~ '^[0-9a-f]{64}$'),
    CHECK (audit_generation_digest ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
    node_id text PRIMARY KEY,
    labels jsonb NOT NULL,
    properties jsonb NOT NULL,
    node_digest char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (node_id <> '' AND char_length(node_id) <= 200),
    CHECK (jsonb_typeof(labels) = 'array' AND jsonb_array_length(labels) > 0),
    CHECK (jsonb_typeof(properties) = 'object'),
    CHECK (octet_length(labels::text) <= 8192),
    CHECK (octet_length(properties::text) <= 262144),
    CHECK (node_digest ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
    edge_id text PRIMARY KEY,
    edge_type text NOT NULL,
    source_node_id text NOT NULL REFERENCES knowledge_graph_nodes(node_id),
    target_node_id text NOT NULL REFERENCES knowledge_graph_nodes(node_id),
    properties jsonb NOT NULL,
    edge_digest char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (edge_id ~ '^edge:v1:[0-9a-f]{64}$'),
    CHECK (edge_type IN (
        'ASSERTION_SUBJECT', 'ASSERTION_OBJECT', 'OBSERVED_ON', 'SUPPORTED_BY'
    )),
    CHECK (jsonb_typeof(properties) = 'object'),
    CHECK (octet_length(properties::text) <= 262144),
    CHECK (edge_digest ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS knowledge_graph_edges_source_idx
    ON knowledge_graph_edges(source_node_id, edge_type, edge_id);
CREATE INDEX IF NOT EXISTS knowledge_graph_edges_target_idx
    ON knowledge_graph_edges(target_node_id, edge_type, edge_id);
CREATE INDEX IF NOT EXISTS knowledge_graph_batches_created_idx
    ON knowledge_graph_batches(created_at, idempotency_key);

INSERT INTO schema_migrations(version)
VALUES ('age_0001')
ON CONFLICT (version) DO NOTHING;

COMMIT;
