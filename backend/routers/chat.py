"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from typing import Any

import fastapi
import fastapi.responses
import pydantic
import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui
import vercel_ai_sdk.ai_sdk_ui.adapter as sse_adapter
import vercel_ai_sdk.ai_sdk_ui.protocol as protocol
from vercel.blob import AsyncBlobClient

import agent
import db

router = fastapi.APIRouter()

# Prefix used by proxy URLs returned from the upload endpoint.
# Includes /api so the browser can fetch directly (Vercel routes /api/* to
# the backend and strips the prefix before forwarding).
FILES_PREFIX = "/api/files/"


# ---------------------------------------------------------------------------
# File upload & serving
# ---------------------------------------------------------------------------


class UploadResponse(pydantic.BaseModel):
    """Response from the file upload endpoint."""

    url: str
    media_type: str = pydantic.Field(serialization_alias="mediaType")
    filename: str


@router.post("/upload")
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


@router.get("/files/{pathname:path}")
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
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str


class SteerRequest(pydantic.BaseModel):
    """Request body for the steering endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str


@router.post("/chat/steer")
async def steer(request: SteerRequest) -> dict[str, Any]:
    """Inject steering messages into the DB-backed queue.

    The agent graph's ``Steering`` hook pops these at the next loop
    iteration (or at the start of the next turn if the stream is idle).
    """
    items: list[tuple[str, str, str, list[dict[str, Any]]]] = [
        (msg.id, request.session_id, msg.role, _ui_parts_to_dicts(msg.parts))
        for msg in request.messages
    ]
    await db.push_steering(items)
    await db.save_messages_batch(items)
    return {"ok": True, "queued": len(items)}


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
    rows: list[tuple[str, str, str, list[dict[str, Any]]]] = [
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


def _is_steering_hook(msg: ai.Message) -> bool:
    """True if *msg* is a single pending Steering HookPart."""
    return (
        len(msg.parts) == 1
        and isinstance(msg.parts[0], ai.HookPart)
        and msg.parts[0].hook_type == "Steering"
    )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@router.post("/chat")
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

    # Load checkpoint so we can resume after a tool approval round-trip.
    checkpoint_data = await db.get_checkpoint(session_id)
    checkpoint = (
        ai.Checkpoint.model_validate(checkpoint_data) if checkpoint_data else None
    )

    # When resuming from a checkpoint, the frontend already has an assistant
    # message from the first (interrupted) stream.  We must ensure the
    # resumed stream re-uses that same message ID so the frontend SDK
    # *replaces* the existing message instead of pushing a duplicate.
    resume_message_id: str | None = None
    if checkpoint is not None:
        for ui_msg in reversed(request.messages):
            if ui_msg.role == "assistant":
                resume_message_id = ui_msg.id
                break

    run_result = ai.run(
        agent.graph,
        llm,
        [system, *messages],
        agent.TOOLS,
        checkpoint=checkpoint,
    )

    # Tap the stream to capture completed assistant messages for persistence.
    assistant_messages: list[ai.Message] = []
    # Buffer of steering items consumed during the current stream.
    consumed_steering: list[dict[str, Any]] = []

    async def _tap_messages() -> AsyncGenerator[ai.Message]:
        pinned = False
        segment = 0
        async for msg in run_result:
            # Auto-resolve Steering hooks: pop any queued steering
            # messages from the DB and pass them through the hook so
            # the graph can inject them into the conversation.
            if _is_steering_hook(msg):
                hook_part: ai.HookPart = msg.parts[0]  # type: ignore[assignment]
                items = await db.pop_steering(session_id)
                agent.Steering.resolve(  # type: ignore[attr-defined]
                    hook_part.hook_id,
                    {"messages": [item.model_dump() for item in items]},
                )
                if items:
                    # Persist assistant messages accumulated so far so their
                    # created_at precedes the steering user message.
                    if assistant_messages:
                        await _persist_assistant_messages(
                            session_id, assistant_messages
                        )
                        assistant_messages.clear()
                    segment += 1
                    # Buffer consumed items so stream_response can emit a
                    # DataPart before the next assistant chunk.
                    consumed_steering.extend(
                        {
                            "id": item.id,
                            "role": item.role,
                            "parts": item.parts,
                        }
                        for item in items
                    )
                continue

            # Force a message boundary after steering: a label change
            # triggers FinishPart + StartPart in the SSE stream.
            if segment > 0 and msg.role == "assistant":
                msg = msg.model_copy(update={"label": f"s{segment}"})

            # Pin the first yielded message to the original assistant ID so
            # the SSE StartPart carries the same ID the frontend already has.
            if not pinned and resume_message_id and msg.role == "assistant":
                msg = msg.model_copy(update={"id": resume_message_id})
                pinned = True
            if msg.role == "assistant" and msg.is_done:
                assistant_messages.append(msg.model_copy(deep=True))
            yield msg

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(_tap_messages()):
            # Emit a transient DataPart for any steering items consumed
            # since the last chunk.  The frontend's onData handler picks
            # this up and appends the user messages inline.
            if consumed_steering:
                part = protocol.DataPart(
                    data_type="steering-consumed",
                    data={"messages": list(consumed_steering)},
                    transient=True,
                )
                yield sse_adapter.format_sse(part)
                consumed_steering.clear()
            yield chunk

        # Post-stream persistence.
        await _persist_assistant_messages(session_id, assistant_messages)

        # Filter out Steering hooks from the checkpoint — they are
        # auto-resolved and must not persist across requests.
        cp = run_result.checkpoint
        real_pending = [h for h in cp.pending_hooks if h.hook_type != "Steering"]
        if real_pending:
            cleaned = cp.model_copy(update={"pending_hooks": real_pending})
            await db.save_checkpoint(session_id, cleaned.model_dump())
        else:
            await db.delete_checkpoint(session_id)
        await db.touch_session(session_id)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
