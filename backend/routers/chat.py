"""Chat, file upload, and file serving endpoints."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any

import ai
import fastapi
import fastapi.responses
import pydantic
from ai.adapters.ai_sdk_ui import (
    UI_MESSAGE_STREAM_HEADERS,
    UIMessage,
    to_messages,
    to_sse_stream,
)
from vercel.blob import AsyncBlobClient

import agent
import db
from replay import ReplayMiddleware, ReplayMismatchError, compute_replay_fingerprint

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

    messages: list[UIMessage]
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
        new_parts: list[
            ai.TextPart | ai.ToolCallPart | ai.ToolResultPart | ai.FilePart | Any
        ] = []
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
    parts: list[Any],
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
        elif isinstance(part, ai.ToolCallPart):
            result.append(
                {
                    "type": f"tool-{part.tool_name}",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "state": "input-available",
                    "input": part.tool_args,
                }
            )
        elif isinstance(part, ai.ToolResultPart):
            state = "output-error" if part.is_error else "output-available"
            result.append(
                {
                    "type": f"tool-{part.tool_name}",
                    "toolCallId": part.tool_call_id,
                    "toolName": part.tool_name,
                    "state": state,
                    "output": part.result,
                }
            )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _persist_request_messages(
    session_id: str, ui_messages: list[UIMessage]
) -> None:
    """Batch-upsert all incoming UI messages into the DB."""
    rows: list[tuple[str, str, str, list[dict[str, Any]]]] = [
        (msg.id, session_id, msg.role, _ui_parts_to_dicts(msg.parts))
        for msg in ui_messages
    ]
    await db.save_messages_batch(rows)


async def _persist_assistant_messages(
    rows: list[tuple[str, str, str, list[dict[str, Any]]]]
) -> None:
    """Save completed assistant UI messages to the DB."""
    await db.save_messages_batch(rows)


def _tool_call_id_from_approval_id(approval_id: str) -> str | None:
    """Extract the tool_call_id from a ToolApproval hook label."""
    prefix = "approve_"
    if approval_id.startswith(prefix):
        return approval_id[len(prefix) :]
    return None


def _normalize_tool_input(raw: str) -> str | dict[str, Any]:
    """Persist tool input in the UI's accepted string-or-dict shape."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed if isinstance(parsed, dict) else raw


class _AssistantTurnBuilder:
    """Accumulate runtime messages into one persisted AI SDK UI assistant turn."""

    def __init__(self, message_id: str | None = None) -> None:
        self.message_id = message_id
        self.parts: list[dict[str, Any]] = []
        self._tool_indexes: dict[str, int] = {}

    def ingest(self, message: ai.Message) -> None:
        """Consume one runtime message."""
        if message.role == "assistant" and message.is_done:
            self._ingest_assistant(message)
        elif message.role == "tool":
            self._ingest_tool(message)
        elif message.role == "signal":
            self._ingest_signal(message)

    def build_row(
        self,
        *,
        session_id: str,
    ) -> tuple[str, str, str, list[dict[str, Any]]] | None:
        """Return one DB row for the assistant turn, if anything was accumulated."""
        if not self.parts or self.message_id is None:
            return None
        return (self.message_id, session_id, "assistant", self.parts)

    def _ingest_assistant(self, message: ai.Message) -> None:
        if self.message_id is None:
            self.message_id = message.id
        for part in message.parts:
            if isinstance(part, ai.ReasoningPart) and part.text:
                self.parts.append({"type": "reasoning", "reasoning": part.text})
            elif isinstance(part, ai.TextPart) and part.text:
                self.parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ai.ToolCallPart):
                if part.tool_call_id in self._tool_indexes:
                    continue
                self._tool_indexes[part.tool_call_id] = len(self.parts)
                self.parts.append(
                    {
                        "type": f"tool-{part.tool_name}",
                        "toolCallId": part.tool_call_id,
                        "toolName": part.tool_name,
                        "state": "input-available",
                        "input": _normalize_tool_input(part.tool_args),
                    }
                )

    def _ingest_tool(self, message: ai.Message) -> None:
        for part in message.parts:
            if not isinstance(part, ai.ToolResultPart):
                continue
            index = self._tool_indexes.get(part.tool_call_id)
            if index is None:
                continue
            tool_part = dict(self.parts[index])
            if tool_part.get("state") != "output-denied":
                tool_part["state"] = (
                    "output-error" if part.is_error else "output-available"
                )
            tool_part["output"] = part.result
            self.parts[index] = tool_part

    def _ingest_signal(self, message: ai.Message) -> None:
        hook_part = message.get_hook_part()
        if hook_part is None:
            return
        tool_call_id = _tool_call_id_from_approval_id(hook_part.hook_id)
        if tool_call_id is None:
            return
        index = self._tool_indexes.get(tool_call_id)
        if index is None:
            return

        tool_part = dict(self.parts[index])
        if hook_part.status == "pending":
            tool_part["state"] = "approval-requested"
            tool_part["approval"] = {"id": hook_part.hook_id}
        elif hook_part.status == "resolved":
            resolution = hook_part.resolution or {}
            tool_part["approval"] = {
                "id": hook_part.hook_id,
                "approved": resolution.get("granted"),
                "reason": resolution.get("reason"),
            }
            if resolution.get("granted", False):
                tool_part["state"] = "approval-responded"
            else:
                tool_part["state"] = "output-denied"
        elif hook_part.status == "cancelled":
            tool_part["state"] = "output-error"
            tool_part["errorText"] = "Hook cancelled"
        self.parts[index] = tool_part


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
    # NOTE: to_messages() has a side-effect of calling resolve_hook() for
    # any tool parts in "approval-responded" state. This pre-registers
    # resolutions so the agent loop can pass through hooks on replay.
    messages = to_messages(request.messages)
    messages = await _inline_file_parts(messages)

    # Prepend the system prompt.
    system = ai.system_message(agent.SYSTEM)

    model = agent.get_model()
    fingerprint = compute_replay_fingerprint(
        session_id=session_id,
        system=system,
        messages=messages,
        model=model,
        tools=agent.TOOLS,
    )
    saved_replay = await db.get_replay(session_id)
    replay_middleware = ReplayMiddleware(
        session_id=session_id,
        fingerprint=fingerprint,
        model=model,
        tools=agent.TOOLS,
        input_message_count=len(messages) + 1,
        replay=saved_replay,
    )

    # When resuming from a tool approval round-trip, the frontend already
    # has an assistant message from the first (interrupted) stream. We must
    # ensure the resumed stream re-uses that same message ID so the
    # frontend SDK *replaces* the existing message instead of pushing a
    # duplicate.
    resume_message_id: str | None = None
    for ui_msg in reversed(request.messages):
        if ui_msg.role == "assistant":
            resume_message_id = ui_msg.id
            break

    run_result = agent.seal.run(
        model,
        [system, *messages],
        middleware=[replay_middleware],
    )

    # Accumulate one persisted UI assistant message for this turn.
    turn_builder = _AssistantTurnBuilder(message_id=resume_message_id)

    async def _tap_messages() -> AsyncGenerator[ai.Message]:
        pinned = False
        async for msg in run_result:
            turn_builder.ingest(msg)

            # Replayed prefix reconstructs agent state but should not be
            # emitted back to the UI; the client already has it locally.
            if replay_middleware.consume_replayed_outbound(msg):
                continue

            # Pin the first non-replayed message to the original assistant ID
            # so resumed tool/result updates stay attached to the same UI turn.
            if (
                not pinned
                and resume_message_id
                and msg.role in {"assistant", "tool", "signal"}
            ):
                msg = msg.model_copy(update={"id": resume_message_id})
                pinned = True
            yield msg

    async def stream_response() -> AsyncGenerator[str]:
        token = agent.activate_session(session_id)
        try:
            async for chunk in to_sse_stream(_tap_messages()):
                yield chunk

            # Post-stream persistence.
            if replay_middleware.should_persist():
                await db.save_replay(session_id, replay_middleware.build_state())
            else:
                await db.delete_replay(session_id)
            assistant_row = turn_builder.build_row(session_id=session_id)
            if assistant_row is not None:
                await _persist_assistant_messages([assistant_row])
            await db.touch_session(session_id)
        except ReplayMismatchError:
            await db.delete_replay(session_id)
            raise
        finally:
            agent.deactivate_session(token)

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=UI_MESSAGE_STREAM_HEADERS,
    )
