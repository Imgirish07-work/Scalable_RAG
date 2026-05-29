"""Declarative base for all SQLAlchemy ORM entities."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single declarative base that every ORM model in this package inherits from."""
