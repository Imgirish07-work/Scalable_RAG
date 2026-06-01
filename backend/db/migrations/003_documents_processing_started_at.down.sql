DROP INDEX IF EXISTS ix_documents_processing_started_at;
ALTER TABLE documents DROP COLUMN IF EXISTS processing_started_at;
