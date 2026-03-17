"""FastAPI application entry point."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic
import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui
from vercel.blob import AsyncBlobClient

import agent
import db

# Prefix used by proxy URLs returned from the upload endpoint.
# Includes /api so the browser can fetch directly (Vercel routes /api/* to
# the backend and strips the prefix before forwarding).
FILES_PREFIX = "/api/files/"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):  # noqa: ANN201
    """Create the DB pool + tables on startup, close on shutdown."""
    await db.ensure_schema()
    yield
    await db.close_pool()


app = fastapi.FastAPI(
    title="seal",
    description="Seal – personal AI assistant",
    lifespan=lifespan,
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# Sessions
# ---------------------------------------------------------------------------


class CreateSessionRequest(pydantic.BaseModel):
    """Body for ``POST /sessions``."""

    id: str
    title: str | None = None


@app.get("/sessions")
async def list_sessions() -> list[db.Session]:
    """Return all sessions, most recent first."""
    return await db.list_sessions()


@app.post("/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> db.Session:
    """Create a new session with a client-generated ID."""
    return await db.create_session(body.id, body.title)


@app.get("/sessions/{session_id}")
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


@app.delete("/sessions/{session_id}")
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
                return part["text"]
    return None


@app.post("/sessions/{session_id}/title")
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


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str


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


# ---------------------------------------------------------------------------
# Message serialisation (UI <-> SDK <-> DB)
# ---------------------------------------------------------------------------


def _ui_parts_to_dicts(
    parts: list,  # type: ignore[type-arg]
) -> list[dict]:  # type: ignore[type-arg]
    """Serialize UIMessage parts to plain dicts for DB storage."""
    return [
        part.model_dump() if hasattr(part, "model_dump") else dict(part)
        for part in parts
    ]


def _sdk_parts_to_ui_dicts(
    parts: list[ai.core.messages.Part],
) -> list[dict]:  # type: ignore[type-arg]
    """Convert internal SDK parts to the UI-compatible dict format.

    The frontend expects the AI SDK UI protocol shape (``type``, ``text``,
    ``toolCallId``, ``toolName``, ``input``, ``output``, ``state``, etc.)
    which differs from the internal SDK model.
    """
    result: list[dict] = []  # type: ignore[type-arg]
    for part in parts:
        if isinstance(part, ai.TextPart):
            if part.text:
                result.append({"type": "text", "text": part.text})
        elif isinstance(part, ai.core.messages.ToolPart):
            state = {
                "result": "output-available",
                "error": "output-error",
                "pending": "call",
            }.get(part.status, "call")
            result.append(
                {
                    "type": f"tool-{part.tool_name}",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "state": state,
                    "input": part.tool_args,
                    "output": part.result,
                }
            )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _persist_request_messages(
    session_id: str, ui_messages: list[ai.ai_sdk_ui.UIMessage]
) -> None:
    """Batch-upsert all incoming UI messages into the DB."""
    rows = [
        (msg.id, session_id, msg.role, _ui_parts_to_dicts(msg.parts))
        for msg in ui_messages
    ]
    await db.save_messages_batch(rows)


async def _persist_assistant_messages(
    session_id: str, messages: list[ai.Message]
) -> None:
    """Save completed assistant messages to the DB."""
    rows: list[tuple[str, str, str, list[dict]]] = []  # type: ignore[type-arg]
    for msg in messages:
        ui_parts = _sdk_parts_to_ui_dicts(msg.parts)
        if ui_parts:
            rows.append((msg.id, session_id, "assistant", ui_parts))
    await db.save_messages_batch(rows)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    session_id = request.session_id

    # Ensure the session exists (create if somehow missing).
    session = await db.get_session(session_id)
    if not session:
        await db.create_session(session_id)

    # Batch-upsert all incoming messages.
    await _persist_request_messages(session_id, request.messages)

    # Convert UI messages to SDK messages and inline file parts.
    messages = ai.ai_sdk_ui.to_messages(request.messages)
    messages = await _inline_file_parts(messages)

    # Prepend the system prompt.
    system = ai.Message(
        role="system",
        parts=[ai.TextPart(text=agent.SYSTEM)],
    )

    llm = agent.get_llm()
    run_result = ai.run(agent.graph, llm, [system, *messages], agent.TOOLS)

    # Tap the stream to capture completed assistant messages for persistence.
    assistant_messages: list[ai.Message] = []

    async def _tap_messages() -> AsyncGenerator[ai.Message]:
        async for msg in run_result:
            if msg.role == "assistant" and msg.is_done:
                assistant_messages.append(msg.model_copy(deep=True))
            yield msg

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(_tap_messages()):
            yield chunk

        # Post-stream persistence.
        await _persist_assistant_messages(session_id, assistant_messages)
        if run_result.checkpoint:
            await db.save_checkpoint(session_id, run_result.checkpoint.model_dump())
        await db.touch_session(session_id)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
