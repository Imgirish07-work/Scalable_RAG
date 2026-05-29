-- Documents: one row per file ingested into the user's corpus.
CREATE TABLE documents (
    id              UUID         PRIMARY KEY,
    user_id         TEXT         NOT NULL,
    content_hash    CHAR(64)     NOT NULL,
    file_name       TEXT         NOT NULL,
    mime_type       TEXT         NOT NULL,
    size_bytes      BIGINT       NOT NULL CHECK (size_bytes >= 0),
    s3_bucket       TEXT         NOT NULL,
    s3_key          TEXT         NOT NULL,
    collection      TEXT         NOT NULL DEFAULT 'default',
    status          TEXT         NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    chunks_count    INTEGER,
    error_message   TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ
);

-- Dedup: same content cannot be uploaded twice by the same user while active.
CREATE UNIQUE INDEX uq_documents_user_content_active
    ON documents (user_id, content_hash)
    WHERE deleted_at IS NULL;

-- S3 key is globally unique; doubles as an integrity check.
CREATE UNIQUE INDEX uq_documents_s3_key
    ON documents (s3_key);

-- Listing endpoint: a user's documents, newest first.
CREATE INDEX ix_documents_user_created
    ON documents (user_id, created_at DESC)
    WHERE deleted_at IS NULL;

-- Listing endpoint scoped to a logical folder.
CREATE INDEX ix_documents_user_collection_created
    ON documents (user_id, collection, created_at DESC)
    WHERE deleted_at IS NULL;

-- Background workers polling for stuck or queued ingestion work.
CREATE INDEX ix_documents_status_active
    ON documents (status)
    WHERE status IN ('pending', 'processing', 'failed');
