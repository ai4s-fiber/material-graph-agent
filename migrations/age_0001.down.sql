-- Destructive rollback. Never run automatically.
-- The shared material_graph AGE graph is deliberately retained: other graph
-- clients may use it, and dropping it would destroy data outside this writer.
BEGIN;
SET LOCAL search_path = public, ag_catalog;

DROP TABLE IF EXISTS knowledge_graph_edges;
DROP TABLE IF EXISTS knowledge_graph_nodes;
DROP TABLE IF EXISTS knowledge_graph_batches;
DELETE FROM schema_migrations WHERE version = 'age_0001';

COMMIT;
