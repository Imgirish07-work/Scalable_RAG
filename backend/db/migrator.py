"""SQL migration runner.

Usage:
    python -m backend.db.migrator up      Apply every pending .up.sql migration.
    python -m backend.db.migrator down    Revert the most recently applied migration.
    python -m backend.db.migrator info    Show applied + pending status.

Run inside the backend container so `BACKEND_DATABASE_URL` resolves the
`postgres` compose hostname:

    docker compose exec backend python -m backend.db.migrator up

Each migration is applied in its own transaction; partial failures roll back
that migration alone. The bookkeeping table `_schema_migrations` tracks which
versions are applied and is created on first run.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.repositories.database import dispose_engine, get_engine
from utils.logger import get_logger

logger = get_logger(__name__)


_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_BOOKKEEPING_TABLE = "_schema_migrations"


def _split_statements(sql: str) -> list[str]:
    """Split SQL on `;` boundaries, ignoring `--` line comments."""
    stripped_lines = [line.split("--", 1)[0] for line in sql.splitlines()]
    cleaned = "\n".join(stripped_lines)
    return [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]


async def _ensure_bookkeeping_table(conn: AsyncConnection) -> None:
    """Create the migrations tracking table if it does not already exist."""
    await conn.execute(text(
        f"CREATE TABLE IF NOT EXISTS {_BOOKKEEPING_TABLE} ("
        "  version    TEXT        PRIMARY KEY,"
        "  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
        ")"
    ))


async def _applied_versions(conn: AsyncConnection) -> set[str]:
    """Return the set of versions currently recorded in the bookkeeping table."""
    result = await conn.execute(text(f"SELECT version FROM {_BOOKKEEPING_TABLE}"))
    return {row[0] for row in result.fetchall()}


def _discover_migrations(direction: str) -> list[tuple[str, Path]]:
    """Return [(version, path), ...] for every .{direction}.sql file, sorted."""
    suffix = f".{direction}.sql"
    files = sorted(_MIGRATIONS_DIR.glob(f"*{suffix}"))
    return [(f.name[: -len(suffix)], f) for f in files]


async def _apply_one(conn: AsyncConnection, version: str, path: Path) -> None:
    logger.info("Applying migration | version=%s | file=%s", version, path.name)
    for statement in _split_statements(path.read_text(encoding="utf-8")):
        await conn.execute(text(statement))
    await conn.execute(
        text(f"INSERT INTO {_BOOKKEEPING_TABLE} (version) VALUES (:v)"),
        {"v": version},
    )


async def _revert_one(conn: AsyncConnection, version: str, path: Path) -> None:
    logger.info("Reverting migration | version=%s | file=%s", version, path.name)
    for statement in _split_statements(path.read_text(encoding="utf-8")):
        await conn.execute(text(statement))
    await conn.execute(
        text(f"DELETE FROM {_BOOKKEEPING_TABLE} WHERE version = :v"),
        {"v": version},
    )


async def up() -> None:
    """Apply every pending .up.sql migration in version order."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await _ensure_bookkeeping_table(conn)
            applied = await _applied_versions(conn)

        pending = [(v, p) for v, p in _discover_migrations("up") if v not in applied]
        if not pending:
            logger.info("No pending migrations | applied=%d", len(applied))
            return

        logger.info("Pending migrations | count=%d", len(pending))
        for version, path in pending:
            async with engine.begin() as conn:
                await _apply_one(conn, version, path)
        logger.info("Migration upgrade complete | applied=%d", len(pending))
    finally:
        await dispose_engine()


async def down() -> None:
    """Revert the single most recently applied migration."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await _ensure_bookkeeping_table(conn)
            applied = await _applied_versions(conn)

        if not applied:
            logger.info("No migrations to revert")
            return

        latest = max(applied)
        down_files = dict(_discover_migrations("down"))
        if latest not in down_files:
            raise RuntimeError(
                f"No .down.sql found for applied migration '{latest}' — "
                "cannot revert safely"
            )

        async with engine.begin() as conn:
            await _revert_one(conn, latest, down_files[latest])
        logger.info("Migration downgrade complete | reverted=%s", latest)
    finally:
        await dispose_engine()


async def info() -> None:
    """Print applied and pending migrations."""
    engine = get_engine()
    try:
        async with engine.begin() as conn:
            await _ensure_bookkeeping_table(conn)
            applied = await _applied_versions(conn)

        discovered = [v for v, _ in _discover_migrations("up")]

        print("\nDatabase migration status")
        print(f"  Directory : {_MIGRATIONS_DIR}")
        print(f"  Applied   : {len(applied)}")
        print(f"  Discovered: {len(discovered)}\n")
        for version in discovered:
            marker = "applied" if version in applied else "pending"
            print(f"  [{marker:>7}]  {version}")

        orphans = sorted(applied - set(discovered))
        if orphans:
            print("\n  WARNING: applied versions with no matching .up.sql file:")
            for version in orphans:
                print(f"    !!  {version}")
        print()
    finally:
        await dispose_engine()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="backend.db.migrator",
        description="Apply or revert raw-SQL database migrations.",
    )
    parser.add_argument(
        "command",
        choices=("up", "down", "info"),
        help="up: apply pending | down: revert latest | info: show status",
    )
    args = parser.parse_args()

    handlers = {"up": up, "down": down, "info": info}
    try:
        asyncio.run(handlers[args.command]())
        return 0
    except Exception as exc:
        logger.exception("Migrator failed | command=%s | error=%s", args.command, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
