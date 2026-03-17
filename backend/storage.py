"""
Pluggable storage for checkpoints and session data.

Provides a minimal Storage protocol and implementations for both
FileStorage (dev) and NeonStorage (production) backends.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import db


@runtime_checkable
class Storage(Protocol):
    """Async key-value storage interface."""

    async def get(self, key: str) -> dict[str, Any] | None: ...
    async def put(self, key: str, value: dict[str, Any]) -> None: ...
    async def delete(self, key: str) -> None: ...


class FileStorage:
    """
    JSON-file-per-key storage backend.

    Each key is stored as ``{directory}/{key}.json``. Good enough for
    local development; replace with a real database for production.
    """

    def __init__(self, directory: str | pathlib.Path | None = None) -> None:
        if directory is None:
            directory = "./data"
        self._dir = pathlib.Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        safe = key.replace("/", "__").replace(":", "_")
        return self._dir / f"{safe}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())  # type: ignore[no-any-return]

    async def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value, indent=2))

    async def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Neon PostgreSQL Storage
# ---------------------------------------------------------------------------


class NeonStorage:
    """
    PostgreSQL-backed storage using Neon.
    
    Provides methods for managing users, sessions, messages, and checkpoints.
    Uses asyncpg with parameterized queries for security and performance.
    """

    async def _get_pool(self):
        """Lazily import db module and get pool."""
        import db
        return await db.get_pool()

    # -------------------------------------------------------------------------
    # User Management
    # -------------------------------------------------------------------------

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT id, external_id, display_name, created_at, updated_at FROM users WHERE id = $1",
            user_id,
        )
        return dict(row) if row else None

    async def create_user(self, user_id: str | None = None, display_name: str | None = None) -> dict[str, Any]:
        """Create a new user. If user_id is provided, use it; otherwise generate."""
        pool = await self._get_pool()
        if user_id:
            row = await pool.fetchrow(
                """INSERT INTO users (id, display_name) VALUES ($1, $2) 
                   ON CONFLICT (id) DO UPDATE SET updated_at = now()
                   RETURNING id, external_id, display_name, created_at, updated_at""",
                user_id,
                display_name,
            )
        else:
            row = await pool.fetchrow(
                """INSERT INTO users (display_name) VALUES ($1) 
                   RETURNING id, external_id, display_name, created_at, updated_at""",
                display_name,
            )
        return dict(row)  # type: ignore[arg-type]

    async def get_or_create_user(self, user_id: str) -> dict[str, Any]:
        """Get an existing user or create a new one with the given ID."""
        user = await self.get_user(user_id)
        if user:
            return user
        return await self.create_user(user_id)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    async def get_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get all sessions for a user, ordered by most recent first."""
        pool = await self._get_pool()
        rows = await pool.fetch(
            """SELECT id, user_id, title, created_at, updated_at 
               FROM sessions WHERE user_id = $1 
               ORDER BY updated_at DESC""",
            user_id,
        )
        return [dict(r) for r in rows]

    async def create_session(self, user_id: str, title: str | None = None) -> dict[str, Any]:
        """Create a new session for a user."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """INSERT INTO sessions (user_id, title) VALUES ($1, $2) 
               RETURNING id, user_id, title, created_at, updated_at""",
            user_id,
            title,
        )
        return dict(row)  # type: ignore[arg-type]

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a session by ID."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = $1",
            session_id,
        )
        return dict(row) if row else None

    async def update_session(self, session_id: str, title: str | None = None) -> dict[str, Any] | None:
        """Update a session's title and updated_at timestamp."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """UPDATE sessions SET title = COALESCE($2, title), updated_at = now() 
               WHERE id = $1 
               RETURNING id, user_id, title, created_at, updated_at""",
            session_id,
            title,
        )
        return dict(row) if row else None

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages/checkpoints (CASCADE)."""
        pool = await self._get_pool()
        await pool.execute("DELETE FROM sessions WHERE id = $1", session_id)

    async def touch_session(self, session_id: str) -> None:
        """Update the session's updated_at timestamp."""
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE sessions SET updated_at = now() WHERE id = $1",
            session_id,
        )

    # -------------------------------------------------------------------------
    # Message Management
    # -------------------------------------------------------------------------

    async def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages in a session, ordered by creation time."""
        pool = await self._get_pool()
        rows = await pool.fetch(
            """SELECT id, session_id, role, content, created_at 
               FROM messages WHERE session_id = $1 
               ORDER BY created_at ASC""",
            session_id,
        )
        return [dict(r) for r in rows]

    async def add_message(
        self, session_id: str, role: str, content: dict[str, Any]
    ) -> dict[str, Any]:
        """Add a message to a session."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3) 
               RETURNING id, session_id, role, content, created_at""",
            session_id,
            role,
            json.dumps(content),
        )
        result = dict(row)  # type: ignore[arg-type]
        # Parse JSON content back to dict
        if isinstance(result.get("content"), str):
            result["content"] = json.loads(result["content"])
        return result

    async def add_messages_batch(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add multiple messages to a session in a single transaction."""
        pool = await self._get_pool()
        results = []
        async with pool.acquire() as conn:
            async with conn.transaction():
                for msg in messages:
                    row = await conn.fetchrow(
                        """INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3) 
                           RETURNING id, session_id, role, content, created_at""",
                        session_id,
                        msg["role"],
                        json.dumps(msg["content"]),
                    )
                    result = dict(row)  # type: ignore[arg-type]
                    if isinstance(result.get("content"), str):
                        result["content"] = json.loads(result["content"])
                    results.append(result)
        return results

    # -------------------------------------------------------------------------
    # Checkpoint Management
    # -------------------------------------------------------------------------

    async def get_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Get the latest checkpoint for a session."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """SELECT id, session_id, checkpoint_data, created_at 
               FROM checkpoints WHERE session_id = $1 
               ORDER BY created_at DESC LIMIT 1""",
            session_id,
        )
        if not row:
            return None
        result = dict(row)
        if isinstance(result.get("checkpoint_data"), str):
            result["checkpoint_data"] = json.loads(result["checkpoint_data"])
        return result

    async def save_checkpoint(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Save a checkpoint for a session (creates new, keeps history)."""
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """INSERT INTO checkpoints (session_id, checkpoint_data) VALUES ($1, $2) 
               RETURNING id, session_id, checkpoint_data, created_at""",
            session_id,
            json.dumps(data),
        )
        result = dict(row)  # type: ignore[arg-type]
        if isinstance(result.get("checkpoint_data"), str):
            result["checkpoint_data"] = json.loads(result["checkpoint_data"])
        return result

    async def delete_checkpoints(self, session_id: str) -> None:
        """Delete all checkpoints for a session."""
        pool = await self._get_pool()
        await pool.execute("DELETE FROM checkpoints WHERE session_id = $1", session_id)

    # -------------------------------------------------------------------------
    # Legacy Storage Protocol Methods (for compatibility)
    # -------------------------------------------------------------------------

    async def get(self, key: str) -> dict[str, Any] | None:
        """Legacy get method - treats key as session_id for checkpoint."""
        checkpoint = await self.get_checkpoint(key)
        return checkpoint["checkpoint_data"] if checkpoint else None

    async def put(self, key: str, value: dict[str, Any]) -> None:
        """Legacy put method - treats key as session_id for checkpoint."""
        await self.save_checkpoint(key, value)

    async def delete(self, key: str) -> None:
        """Legacy delete method - treats key as session_id for checkpoint."""
        await self.delete_checkpoints(key)


# Default storage instance
storage = NeonStorage()
