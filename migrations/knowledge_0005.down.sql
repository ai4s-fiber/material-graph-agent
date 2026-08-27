BEGIN;
SET LOCAL search_path = public;

DO $rollback$
BEGIN
    IF EXISTS (SELECT 1 FROM knowledge_worker_jobs WHERE job_type = 'graph_write') THEN
        RAISE EXCEPTION 'cannot roll back knowledge_0005 while graph_write jobs exist';
    END IF;
END
$rollback$;

DROP TABLE IF EXISTS knowledge_graph_write_audit;
DROP TRIGGER IF EXISTS knowledge_fact_review_job_guard ON knowledge_fact_reviews;
DROP TRIGGER IF EXISTS knowledge_graph_write_job_review_guard ON knowledge_worker_jobs;
DROP FUNCTION IF EXISTS enforce_fact_review_job_binding();
DROP FUNCTION IF EXISTS enforce_graph_write_job_review_binding();
DROP TABLE IF EXISTS knowledge_fact_reviews;
DROP INDEX IF EXISTS knowledge_fact_extraction_batch_id_idx;

ALTER TABLE knowledge_worker_jobs
    DROP CONSTRAINT IF EXISTS knowledge_worker_jobs_graph_write_payload_check;
ALTER TABLE knowledge_worker_jobs
    DROP CONSTRAINT IF EXISTS knowledge_worker_jobs_type_check;
ALTER TABLE knowledge_worker_jobs
    ADD CONSTRAINT knowledge_worker_jobs_type_check
    CHECK (job_type IN ('metadata_only', 'single_document'));

DELETE FROM schema_migrations WHERE version = 'knowledge_0005';
COMMIT;
