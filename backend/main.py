"""FastAPI application entry point."""

from __future__ import annotations

import base64
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic
import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui
from vercel.blob import AsyncBlobClient

import agent
import storage as storage_module

# Prefix used by proxy URLs returned from the upload endpoint.
# Includes /api so the browser can fetch directly (Vercel routes /api/* to
# the backend and strips the prefix before forwarding).
FILES_PREFIX = "/api/files/"

# Cookie name for anonymous user tracking
USER_COOKIE_NAME = "seal_user_id"

app = fastapi.FastAPI(
    title="seal",
    description="Seal – personal AI assistant",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage instance
storage = storage_module.NeonStorage()


# ---------------------------------------------------------------------------
# User identification helpers
# ---------------------------------------------------------------------------


def get_user_id_from_cookie(request: fastapi.Request) -> str | None:
    """Extract user_id from cookie if present."""
    return request.cookies.get(USER_COOKIE_NAME)


def set_user_cookie(response: fastapi.Response, user_id: str) -> None:
    """Set the user_id cookie on the response."""
    response.set_cookie(
        key=USER_COOKIE_NAME,
        value=user_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 365,  # 1 year
    )


async def get_or_create_user(request: fastapi.Request, response: fastapi.Response) -> str:
    """Get existing user ID from cookie or create a new user."""
    user_id = get_user_id_from_cookie(request)
    if user_id:
        # Verify user exists in DB
        user = await storage.get_user(user_id)
        if user:
            return user_id
    
    # Create new user
    new_user = await storage.create_user()
    user_id = str(new_user["id"])
    set_user_cookie(response, user_id)
    return user_id


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# File upload & serving
# ---------------------------------------------------------------------------


class UploadResponse(pydantic.BaseModel):
    """Response from the file upload endpoint."""

    url: str
    media_type: str = pydantic.Field(serialization_alias="mediaType")
    filename: str


@app.post("/upload")
async def upload(file: fastapi.UploadFile) -> UploadResponse:
    """Upload a file to Vercel Blob storage (private)."""
    content = await file.read()
    media_type = file.content_type or "application/octet-stream"
    filename = file.filename or "attachment"

    async with AsyncBlobClient() as client:
        result = await client.put(
            f"attachments/{filename}",
            content,
            access="private",
            content_type=media_type,
            add_random_suffix=True,
        )

    # Return a proxy URL so the browser fetches through our backend,
    # keeping the blob private.
    return UploadResponse(
        url=f"{FILES_PREFIX}{result.pathname}",
        media_type=media_type,
        filename=filename,
    )


@app.get("/files/{pathname:path}")
async def get_file(pathname: str) -> fastapi.responses.Response:
    """Proxy a private Vercel Blob file to the browser."""
    async with AsyncBlobClient() as client:
        result = await client.get(pathname, access="private")

    return fastapi.responses.Response(
        content=result.content,
        media_type=result.content_type or "application/octet-stream",
        headers={
            # Blob pathnames include a random suffix so each upload is unique.
            # Aggressive caching avoids re-fetching on every message re-render.
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


class SessionResponse(pydantic.BaseModel):
    """Response model for a session."""
    id: str
    user_id: str
    title: str | None
    created_at: str
    updated_at: str


class SessionWithMessagesResponse(pydantic.BaseModel):
    """Response model for a session with messages."""
    id: str
    user_id: str
    title: str | None
    created_at: str
    updated_at: str
    messages: list[dict[str, Any]]


class CreateSessionRequest(pydantic.BaseModel):
    """Request body for creating a session."""
    title: str | None = None


class UpdateSessionRequest(pydantic.BaseModel):
    """Request body for updating a session."""
    title: str | None = None


def serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    """Convert a session dict to JSON-serializable format."""
    return {
        "id": str(session["id"]),
        "user_id": str(session["user_id"]),
        "title": session["title"],
        "created_at": session["created_at"].isoformat() if hasattr(session["created_at"], "isoformat") else str(session["created_at"]),
        "updated_at": session["updated_at"].isoformat() if hasattr(session["updated_at"], "isoformat") else str(session["updated_at"]),
    }


def serialize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a message dict to JSON-serializable format."""
    return {
        "id": str(msg["id"]),
        "session_id": str(msg["session_id"]),
        "role": msg["role"],
        "content": msg["content"],
        "created_at": msg["created_at"].isoformat() if hasattr(msg["created_at"], "isoformat") else str(msg["created_at"]),
    }


@app.get("/sessions")
async def list_sessions(
    request: fastapi.Request,
    response: fastapi.Response,
) -> list[dict[str, Any]]:
    """List all sessions for the current user."""
    user_id = await get_or_create_user(request, response)
    sessions = await storage.get_sessions(user_id)
    return [serialize_session(s) for s in sessions]


@app.post("/sessions")
async def create_session(
    request: fastapi.Request,
    response: fastapi.Response,
    body: CreateSessionRequest | None = None,
) -> dict[str, Any]:
    """Create a new session for the current user."""
    user_id = await get_or_create_user(request, response)
    title = body.title if body else None
    session = await storage.create_session(user_id, title)
    return serialize_session(session)


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: fastapi.Request,
    response: fastapi.Response,
) -> dict[str, Any]:
    """Get a session with its message history."""
    user_id = await get_or_create_user(request, response)
    
    session = await storage.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    
    # Verify ownership
    if str(session["user_id"]) != user_id:
        raise fastapi.HTTPException(status_code=403, detail="Access denied")
    
    messages = await storage.get_messages(session_id)
    
    result = serialize_session(session)
    result["messages"] = [serialize_message(m) for m in messages]
    return result


@app.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    body: UpdateSessionRequest,
    request: fastapi.Request,
    response: fastapi.Response,
) -> dict[str, Any]:
    """Update a session's title."""
    user_id = await get_or_create_user(request, response)
    
    session = await storage.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    
    if str(session["user_id"]) != user_id:
        raise fastapi.HTTPException(status_code=403, detail="Access denied")
    
    updated = await storage.update_session(session_id, body.title)
    if not updated:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    
    return serialize_session(updated)


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: fastapi.Request,
    response: fastapi.Response,
) -> dict[str, str]:
    """Delete a session."""
    user_id = await get_or_create_user(request, response)
    
    session = await storage.get_session(session_id)
    if not session:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    
    if str(session["user_id"]) != user_id:
        raise fastapi.HTTPException(status_code=403, detail="Access denied")
    
    await storage.delete_session(session_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str | None = None


def _extract_blob_pathname(url: str) -> str | None:
    """Extract the blob pathname from a proxy URL, or return None."""
    if url.startswith(FILES_PREFIX):
        return url[len(FILES_PREFIX) :]
    return None


async def _inline_file_parts(
    messages: list[ai.Message],
) -> list[ai.Message]:
    """Replace proxy-URL file parts with inline base64 data URLs.

    The AI Gateway requires file content as data URLs (not raw HTTP URLs).
    Our proxy URLs (``/api/files/...``) aren't reachable from the gateway,
    so we fetch the blob content here and inline it before sending.
    """
    result: list[ai.Message] = []
    for msg in messages:
        new_parts: list[ai.core.messages.Part] = []
        for part in msg.parts:
            pathname = (
                _extract_blob_pathname(part.data)
                if isinstance(part, ai.FilePart) and isinstance(part.data, str)
                else None
            )
            if isinstance(part, ai.FilePart) and pathname is not None:
                async with AsyncBlobClient() as client:
                    blob = await client.get(pathname, access="private")

                b64 = base64.b64encode(blob.content).decode("ascii")
                media_type = blob.content_type or part.media_type
                data_url = f"data:{media_type};base64,{b64}"

                new_parts.append(
                    ai.FilePart(
                        data=data_url,
                        media_type=media_type,
                        filename=part.filename,
                    )
                )
            else:
                new_parts.append(part)

        result.append(msg.model_copy(update={"parts": new_parts}))
    return result


def ui_message_to_storage_format(msg: ai.ai_sdk_ui.UIMessage) -> dict[str, Any]:
    """Convert a UIMessage to storage format."""
    return {
        "role": msg.role,
        "content": {
            "id": msg.id,
            "parts": [part.model_dump() if hasattr(part, "model_dump") else dict(part) for part in msg.parts],
        },
    }


@app.post("/chat")
async def chat(
    chat_request: fastapi.Request,
    response: fastapi.Response,
) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    # Parse body manually to handle session persistence
    body = await chat_request.json()
    request = ChatRequest(**body)
    
    messages = ai.ai_sdk_ui.to_messages(request.messages)

    # Inline any proxy-URL file parts so the LLM receives base64 data.
    messages = await _inline_file_parts(messages)

    # Prepend the system prompt
    system = ai.Message(
        role="system",
        parts=[ai.TextPart(text=agent.SYSTEM)],
    )
    all_messages = [system, *messages]

    llm = agent.get_llm()

    result = ai.run(
        agent.graph,
        llm,
        all_messages,
        agent.TOOLS,
    )

    # Handle session persistence
    session_id = request.session_id
    user_id = await get_or_create_user(chat_request, response)
    
    # Auto-create session if not provided
    if not session_id:
        session = await storage.create_session(user_id)
        session_id = str(session["id"])
    else:
        # Verify session exists and belongs to user
        session = await storage.get_session(session_id)
        if not session or str(session["user_id"]) != user_id:
            # Create new session if invalid
            session = await storage.create_session(user_id)
            session_id = str(session["id"])
    
    # Store the user message (last one in the list)
    if request.messages:
        last_user_msg = request.messages[-1]
        if last_user_msg.role == "user":
            await storage.add_message(
                session_id,
                "user",
                {
                    "id": last_user_msg.id,
                    "parts": [part.model_dump() if hasattr(part, "model_dump") else dict(part) for part in last_user_msg.parts],
                },
            )
            
            # Auto-generate title from first message if session has no title
            if session and not session.get("title"):
                # Use first 50 chars of first text part as title
                for part in last_user_msg.parts:
                    if hasattr(part, "text") and part.text:
                        title = part.text[:50] + ("..." if len(part.text) > 50 else "")
                        await storage.update_session(session_id, title)
                        break

    async def stream_response() -> AsyncGenerator[str]:
        collected_parts: list[Any] = []
        
        async for chunk in ai.ai_sdk_ui.to_sse_stream(result):
            yield chunk
            
        # After streaming completes, save the assistant response
        # We need to collect the final response separately
        # For now, we'll update the session timestamp
        await storage.touch_session(session_id)

    streaming_response = fastapi.responses.StreamingResponse(
        stream_response(),
        headers={
            **ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
            "X-Session-ID": session_id,
        },
    )
    
    # Set user cookie on response
    set_user_cookie(streaming_response, user_id)
    
    return streaming_response
