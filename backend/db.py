"""
Neon PostgreSQL storage layer.

Manages a shared asyncpg pool and exposes async functions for sessions,
messages, and checkpoints.  All IDs are plain text (nanoid-style) so
they round-trip cleanly with the AI SDK frontend.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pydantic

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class Session(pydantic.BaseModel):
    """Serialisable session record."""

    id: str
    title: str | None = None
    created_at: str
    updated_at: str


class StoredMessage(pydantic.BaseModel):
    """A message as stored in the DB (parts already parsed)."""

    id: str
    role: str
    parts: list[dict[str, Any]]
    created_at: str


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


def _read_schema() -> str:
    """Read the canonical schema from scripts/001_create_tables.sql."""
    return (_REPO_ROOT / "scripts" / "001_create_tables.sql").read_text()


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return _pool


async def ensure_schema() -> None:
    """Run ``CREATE TABLE IF NOT EXISTS`` for every table."""
    pool = await get_pool()
    await pool.execute(_read_schema())


async def close_pool() -> None:
    """Gracefully close the pool (call from FastAPI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_session(row: asyncpg.Record) -> Session:
    """Convert an asyncpg row to a Session model."""
    return Session(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
    )


def _parse_jsonb(val: Any) -> Any:
    """Ensure a JSONB value is a Python object, not a raw JSON string."""
    if isinstance(val, str):
        return json.loads(val)
    return val


def _row_to_message(row: asyncpg.Record) -> StoredMessage:
    """Convert an asyncpg row to a StoredMessage model."""
    return StoredMessage(
        id=row["id"],
        role=row["role"],
        parts=_parse_jsonb(row["parts"]),
        created_at=row["created_at"].isoformat(),
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

_SESSION_COLS = "id, title, created_at, updated_at"


async def create_session(session_id: str, title: str | None = None) -> Session:
    """Insert a new session and return it.  No-ops if the ID exists."""
    pool = await get_pool()
    row = await pool.fetchrow(
        f"INSERT INTO sessions (id, title) VALUES ($1, $2) "
        f"ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id "
        f"RETURNING {_SESSION_COLS}",
        session_id,
        title,
    )
    return _row_to_session(row)


async def list_sessions() -> list[Session]:
    """Return all sessions ordered by most-recently-updated first."""
    pool = await get_pool()
    rows = await pool.fetch(
        f"SELECT {_SESSION_COLS} FROM sessions ORDER BY updated_at DESC",
    )
    return [_row_to_session(r) for r in rows]


async def get_session(session_id: str) -> Session | None:
    """Return a single session or ``None``."""
    pool = await get_pool()
    row = await pool.fetchrow(
        f"SELECT {_SESSION_COLS} FROM sessions WHERE id = $1",
        session_id,
    )
    return _row_to_session(row) if row else None


async def update_session_title(session_id: str, title: str) -> Session | None:
    """Set the title (and bump ``updated_at``)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        f"UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 "
        f"RETURNING {_SESSION_COLS}",
        session_id,
        title,
    )
    return _row_to_session(row) if row else None


async def delete_session(session_id: str) -> bool:
    """Delete a session (messages + checkpoint cascade). Return True if found."""
    pool = await get_pool()
    result = await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
    return bool(result == "DELETE 1")


async def touch_session(session_id: str) -> None:
    """Bump ``updated_at`` without changing other fields."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE sessions SET updated_at = now() WHERE id = $1", session_id
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


async def get_messages(session_id: str) -> list[StoredMessage]:
    """Return all messages for a session in chronological order."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, role, parts, created_at "
        "FROM messages WHERE session_id = $1 ORDER BY created_at",
        session_id,
    )
    return [_row_to_message(r) for r in rows]


async def save_message(
    message_id: str,
    session_id: str,
    role: str,
    parts: list[dict[str, Any]],
) -> None:
    """Insert or update a single message (upsert on id)."""
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO messages (id, session_id, role, parts) "
        "VALUES ($1, $2, $3, $4::jsonb) "
        "ON CONFLICT (id) DO UPDATE SET parts = EXCLUDED.parts",
        message_id,
        session_id,
        role,
        json.dumps(parts),
    )


async def save_messages_batch(
    messages: list[tuple[str, str, str, list[dict[str, Any]]]],
) -> None:
    """Batch-upsert messages.  Each tuple is (id, session_id, role, parts).

    Duplicates by message ID are deduplicated (last occurrence wins)
    because PostgreSQL's ON CONFLICT DO UPDATE cannot touch the same
    row twice in a single statement.
    """
    if not messages:
        return
    # Deduplicate: keep last occurrence per message ID.
    seen: dict[str, tuple[str, str, str, list[dict[str, Any]]]] = {}
    for row in messages:
        seen[row[0]] = row
    deduped = list(seen.values())

    pool = await get_pool()
    # Build a single VALUES clause for all messages.
    args: list[Any] = []
    placeholders: list[str] = []
    for i, (mid, sid, role, parts) in enumerate(deduped):
        base = i * 4
        placeholders.append(
            f"(${base + 1}, ${base + 2}, ${base + 3}, ${base + 4}::jsonb)"
        )
        args.extend([mid, sid, role, json.dumps(parts)])
    sql = (
        "INSERT INTO messages (id, session_id, role, parts) VALUES "
        + ", ".join(placeholders)
        + " ON CONFLICT (id) DO UPDATE SET parts = EXCLUDED.parts"
    )
    await pool.execute(sql, *args)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


async def get_checkpoint(session_id: str) -> dict[str, Any] | None:
    """Return the checkpoint data dict, or ``None``."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT data FROM checkpoints WHERE session_id = $1", session_id
    )
    if row is None:
        return None
    data = row["data"]
    return json.loads(data) if isinstance(data, str) else data  # type: ignore[no-any-return]


async def save_checkpoint(session_id: str, data: dict[str, Any]) -> None:
    """Upsert the checkpoint for a session (one per session)."""
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO checkpoints (session_id, data) "
        "VALUES ($1, $2::jsonb) "
        "ON CONFLICT (session_id) "
        "DO UPDATE SET data = EXCLUDED.data, updated_at = now()",
        session_id,
        json.dumps(data),
    )
