-- Destructive rollback. Never run automatically.
BEGIN;
SET LOCAL search_path = public;

DROP TABLE IF EXISTS knowledge_metadata_cursors;
DELETE FROM schema_migrations WHERE version = 'knowledge_0004';

COMMIT;
