-- Explicit operator-invoked rollback for knowledge_0001. Never run automatically.
BEGIN;

DROP TABLE IF EXISTS knowledge_lightrag_source_mappings;
DROP TABLE IF EXISTS knowledge_evidence_fragments;
DROP TABLE IF EXISTS knowledge_ingestion_checkpoints;
DROP TABLE IF EXISTS knowledge_source_relations;
DROP TABLE IF EXISTS knowledge_sources;
DELETE FROM schema_migrations WHERE version = 'knowledge_0001';

COMMIT;
