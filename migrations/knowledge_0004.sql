-- Durable metadata manifest cursors with exact logical identity.
-- Apply after knowledge_0003; rollback is kept in knowledge_0004.down.sql.
BEGIN;
SET LOCAL search_path = public;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_metadata_cursors (
    root_id text NOT NULL,
    slice_id text NOT NULL,
    manifest_path text NOT NULL,
    manifest_format text NOT NULL,
    manifest_version_key text NOT NULL,
    next_byte_offset bigint NOT NULL DEFAULT 0,
    records_committed bigint NOT NULL DEFAULT 0,
    checkpoint jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (root_id, slice_id, manifest_path, manifest_format),
    CONSTRAINT knowledge_metadata_cursors_root_check
        CHECK (root_id ~ '^[a-z0-9][a-z0-9_-]*$'),
    CONSTRAINT knowledge_metadata_cursors_slice_check
        CHECK (slice_id ~ '^[a-z0-9][a-z0-9_-]*$'),
    CONSTRAINT knowledge_metadata_cursors_path_check
        CHECK (
            manifest_path <> ''
            AND manifest_path !~ '(^/|//|(^|/)\.\.?(/|$))'
        ),
    CONSTRAINT knowledge_metadata_cursors_format_check
        CHECK (manifest_format IN ('jsonl', 'csv')),
    CONSTRAINT knowledge_metadata_cursors_version_check
        CHECK (manifest_version_key ~ '^source-version-v1:[0-9a-f]{64}$'),
    CONSTRAINT knowledge_metadata_cursors_progress_check
        CHECK (next_byte_offset >= 0 AND records_committed >= 0),
    CONSTRAINT knowledge_metadata_cursors_checkpoint_size_check
        CHECK (
            jsonb_typeof(checkpoint) = 'object'
            AND octet_length(checkpoint::text) <= 262144
        ),
    CONSTRAINT knowledge_metadata_cursors_checkpoint_fields_check
        CHECK (
            checkpoint ?& ARRAY[
                'schema_version', 'root_id', 'slice_id', 'manifest_path',
                'manifest_format', 'manifest_version_key', 'next_byte_offset',
                'records_committed', 'csv_fieldnames'
            ]
            AND checkpoint
                - 'schema_version'
                - 'root_id'
                - 'slice_id'
                - 'manifest_path'
                - 'manifest_format'
                - 'manifest_version_key'
                - 'next_byte_offset'
                - 'records_committed'
                - 'csv_fieldnames' = '{}'::jsonb
        ),
    CONSTRAINT knowledge_metadata_cursors_checkpoint_identity_check
        CHECK (
            checkpoint ->> 'root_id' = root_id
            AND checkpoint ->> 'slice_id' = slice_id
            AND checkpoint ->> 'manifest_path' = manifest_path
            AND checkpoint ->> 'manifest_format' = manifest_format
            AND checkpoint ->> 'manifest_version_key' = manifest_version_key
        ),
    CONSTRAINT knowledge_metadata_cursors_checkpoint_progress_check
        CHECK (
            jsonb_typeof(checkpoint -> 'schema_version') = 'number'
            AND (checkpoint ->> 'schema_version')::integer = 1
            AND jsonb_typeof(checkpoint -> 'next_byte_offset') = 'number'
            AND (checkpoint ->> 'next_byte_offset')::bigint = next_byte_offset
            AND jsonb_typeof(checkpoint -> 'records_committed') = 'number'
            AND (checkpoint ->> 'records_committed')::bigint = records_committed
        ),
    CONSTRAINT knowledge_metadata_cursors_checkpoint_csv_check
        CHECK (
            jsonb_typeof(checkpoint -> 'csv_fieldnames') IN ('array', 'null')
            AND (
                manifest_format = 'csv'
                OR jsonb_typeof(checkpoint -> 'csv_fieldnames') = 'null'
            )
            AND (
                manifest_format <> 'csv'
                OR next_byte_offset = 0
                OR jsonb_typeof(checkpoint -> 'csv_fieldnames') = 'array'
            )
        )
);

CREATE INDEX IF NOT EXISTS knowledge_metadata_cursors_updated_idx
    ON knowledge_metadata_cursors(updated_at, root_id, slice_id);

INSERT INTO schema_migrations(version)
VALUES ('knowledge_0004')
ON CONFLICT (version) DO NOTHING;

COMMIT;
