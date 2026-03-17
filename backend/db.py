"""
Neon PostgreSQL storage layer.

Manages a shared asyncpg pool and exposes async functions for sessions,
messages, and checkpoints.  All IDs are plain text (nanoid-style) so
they round-trip cleanly with the AI SDK frontend.
"""

from __future__ import annotations

import json
import os
from typing import Any

import asyncpg

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    parts       JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS checkpoints (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    data        JSONB NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def get_pool() -> asyncpg.Pool:
    """Return the shared pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return _pool


async def ensure_schema() -> None:
    """Run ``CREATE TABLE IF NOT EXISTS`` for every table."""
    pool = await get_pool()
    await pool.execute(_SCHEMA)


async def close_pool() -> None:
    """Gracefully close the pool (call from FastAPI shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def create_session(session_id: str, title: str | None = None) -> dict[str, Any]:
    """Insert a new session and return it.  No-ops if the ID exists."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "INSERT INTO sessions (id, title) VALUES ($1, $2) "
        "ON CONFLICT (id) DO NOTHING "
        "RETURNING id, title, created_at, updated_at",
        session_id,
        title,
    )
    if row is None:
        # Already existed -- just fetch it.
        row = await pool.fetchrow(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = $1",
            session_id,
        )
    return dict(row)  # type: ignore[arg-type]


async def list_sessions() -> list[dict[str, Any]]:
    """Return all sessions ordered by most-recently-updated first."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, title, created_at, updated_at "
        "FROM sessions ORDER BY updated_at DESC",
    )
    return [dict(r) for r in rows]


async def get_session(session_id: str) -> dict[str, Any] | None:
    """Return a single session or ``None``."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE id = $1",
        session_id,
    )
    return dict(row) if row else None


async def update_session_title(session_id: str, title: str) -> dict[str, Any] | None:
    """Set the title (and bump ``updated_at``)."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "UPDATE sessions SET title = $2, updated_at = now() WHERE id = $1 "
        "RETURNING id, title, created_at, updated_at",
        session_id,
        title,
    )
    return dict(row) if row else None


async def delete_session(session_id: str) -> bool:
    """Delete a session (messages + checkpoint cascade). Return True if found."""
    pool = await get_pool()
    result = await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)
    return result == "DELETE 1"


async def touch_session(session_id: str) -> None:
    """Bump ``updated_at`` without changing other fields."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE sessions SET updated_at = now() WHERE id = $1", session_id
    )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _parse_jsonb(val: Any) -> Any:
    """Ensure a JSONB value is a Python object, not a raw JSON string."""
    if isinstance(val, str):
        return json.loads(val)
    return val


async def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Return all messages for a session in chronological order."""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT id, session_id, role, parts, created_at "
        "FROM messages WHERE session_id = $1 ORDER BY created_at",
        session_id,
    )
    result = []
    for r in rows:
        d = dict(r)
        d["parts"] = _parse_jsonb(d["parts"])
        result.append(d)
    return result


async def save_message(
    message_id: str,
    session_id: str,
    role: str,
    parts: list[dict[str, Any]],
) -> None:
    """Insert or update a message (upsert on id)."""
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
