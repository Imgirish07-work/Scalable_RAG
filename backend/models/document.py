"""SQLAlchemy ORM mapping for the `documents` table.

The class mirrors the schema produced by `001_create_documents.up.sql`. The
`id` column stores the same UUID v7 value that is also written to Qdrant
payload as `metadata.doc_id`, so a single identifier links the metadata row
to its chunks.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, CheckConstraint, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class Document(Base):
    """Single ingested file owned by a user; tracked from upload to ready."""

    __tablename__ = "documents"

    id:             Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    user_id:        Mapped[str] = mapped_column(Text, nullable=False)
    content_hash:   Mapped[str] = mapped_column(String(64), nullable=False)
    file_name:      Mapped[str] = mapped_column(Text, nullable=False)
    mime_type:      Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes:     Mapped[int] = mapped_column(BigInteger, nullable=False)
    s3_bucket:      Mapped[str] = mapped_column(Text, nullable=False)
    s3_key:         Mapped[str] = mapped_column(Text, nullable=False)
    collection:     Mapped[str] = mapped_column(
        Text, nullable=False, server_default="default",
    )
    status:         Mapped[str] = mapped_column(
        Text, nullable=False, server_default="pending",
    )
    chunks_count:   Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message:  Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at:     Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"),
    )
    updated_at:     Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("NOW()"),
    )
    deleted_at:     Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True,
    )

    # Mirror the CHECK constraints defined in the migration so the ORM
    # rejects invalid values without a round-trip to Postgres.
    __table_args__ = (
        CheckConstraint("size_bytes >= 0", name="documents_size_bytes_nonneg"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="documents_status_valid",
        ),
    )
