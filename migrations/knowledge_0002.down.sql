-- Destructive rollback. Never run automatically.
BEGIN;
SET LOCAL search_path = public;

DROP TABLE IF EXISTS knowledge_canary_runs;
DROP TABLE IF EXISTS knowledge_fact_extraction_checkpoints;
DELETE FROM schema_migrations WHERE version = 'knowledge_0002';

COMMIT;
