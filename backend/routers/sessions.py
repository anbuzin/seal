"""Session management endpoints."""

from __future__ import annotations

import fastapi
import pydantic

import agent
import db

router = fastapi.APIRouter()


class CreateSessionRequest(pydantic.BaseModel):
    """Body for ``POST /sessions``."""

    id: str
    title: str | None = None


@router.get("/sessions")
async def list_sessions() -> list[db.Session]:
    """Return all sessions, most recent first."""
    return await db.list_sessions()


@router.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> db.Session:
    """Create a new session with a client-generated ID."""
    return await db.create_session(body.id, body.title)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:  # type: ignore[type-arg]
    """Return a session with its messages."""
    session = await db.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    messages = await db.get_messages(session_id)
    result = session.model_dump()
    result["messages"] = [
        {
            "id": m.id,
            "role": m.role,
            "parts": m.parts,
            "createdAt": m.created_at,
        }
        for m in messages
    ]
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session (cascades to messages + checkpoint)."""
    found = await db.delete_session(session_id)
    if not found:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


def _extract_first_user_text(messages: list[db.StoredMessage]) -> str | None:
    """Return the text of the first user text-part, or None."""
    for msg in messages:
        if msg.role != "user":
            continue
        for part in msg.parts:
            if part.get("type") == "text" and part.get("text"):
                return str(part["text"])
    return None


@router.get("/sessions/{session_id}/steering")
async def get_steering_queue(session_id: str) -> list[db.SteeringItem]:
    """Return all pending steering messages for a session."""
    return await db.get_steering(session_id)


@router.post("/sessions/{session_id}/title")
async def generate_title(session_id: str) -> db.Session:
    """Generate an LLM title for a session from its first message."""
    session = await db.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")

    if session.title:
        return session

    messages = await db.get_messages(session_id)
    first_text = _extract_first_user_text(messages)
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )

    title = await agent.generate_title(first_text)
    row = await db.update_session_title(session_id, title)
    return row  # type: ignore[return-value]
