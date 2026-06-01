-- Worker-lease column read by the stuck-job sweeper.

ALTER TABLE documents
    ADD COLUMN processing_started_at TIMESTAMPTZ;

-- Partial index keeps the sweeper scan O(stuck rows).
CREATE INDEX ix_documents_processing_started_at
    ON documents (processing_started_at)
    WHERE status = 'processing';
