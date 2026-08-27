-- Destructive rollback. Never run automatically.
BEGIN;
SET LOCAL search_path = public;

DROP TABLE IF EXISTS knowledge_worker_jobs;
DELETE FROM schema_migrations WHERE version = 'knowledge_0003';

COMMIT;
