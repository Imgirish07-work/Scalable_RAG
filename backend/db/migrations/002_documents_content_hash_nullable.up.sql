-- Allow content_hash to be NULL while the row is in 'pending' state.
-- With Architecture A (client uploads directly to MinIO), the bytes — and
-- therefore the hash — are only available during finalize, not at
-- upload-session creation. content_hash is filled in during ingestion.

ALTER TABLE documents ALTER COLUMN content_hash DROP NOT NULL;

-- Recreate the dedup unique index so pending rows (content_hash IS NULL)
-- do not all collide on the constraint.
DROP INDEX IF EXISTS uq_documents_user_content_active;

CREATE UNIQUE INDEX uq_documents_user_content_active
    ON documents (user_id, content_hash)
    WHERE deleted_at IS NULL AND content_hash IS NOT NULL;
