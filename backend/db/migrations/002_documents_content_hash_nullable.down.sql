-- Revert content_hash to NOT NULL.
-- WARNING: fails if any rows have content_hash IS NULL — delete pending rows
-- manually before downgrading.

DROP INDEX IF EXISTS uq_documents_user_content_active;

CREATE UNIQUE INDEX uq_documents_user_content_active
    ON documents (user_id, content_hash)
    WHERE deleted_at IS NULL;

ALTER TABLE documents ALTER COLUMN content_hash SET NOT NULL;
