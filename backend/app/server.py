"""FastAPI app for the seal durable agent — the UI-facing surface.

Endpoints:

  POST /api/chat                     run a turn, stream the AI SDK UI message stream
  POST /api/chat/{id}/stop           interrupt the in-flight turn (preempt)
  GET  /api/chat/{id}/stream         resume an in-flight stream
  GET  /api/sessions                 list sessions
  POST /api/sessions                 create a session
  GET  /api/sessions/{id}            session metadata + UI message history
  POST /api/sessions/{id}/title      generate a title from the first user message
  DELETE /api/sessions/{id}          delete a session
  POST /api/upload, GET /api/files/{p}   private blob upload + proxy

This service is a pure control plane: it starts sessions, sends messages,
resolves approval hooks, and observes streams. Handlers run only on the
rotor worker (``worker.py``).
"""

from __future__ import annotations

import collections.abc
import contextlib
import logging

logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

from agent import telemetry  # noqa: E402

_telemetry = telemetry.install("seal-backend")

import ai.ui.ai_sdk as ai_sdk  # noqa: E402
import fastapi  # noqa: E402
import fastapi.middleware.cors  # noqa: E402
import fastapi.responses  # noqa: E402
import pydantic  # noqa: E402
from rotor import MailboxClosed, ProcessNotFound  # noqa: E402
from vercel.blob import AsyncBlobClient  # noqa: E402

from agent import proto  # noqa: E402
from agent.runtime import client  # noqa: E402
from app import attachments, chat, sessions  # noqa: E402


@contextlib.asynccontextmanager
async def lifespan(_app: fastapi.FastAPI) -> collections.abc.AsyncIterator[None]:
    await sessions.ensure_schema()
    try:
        yield
    finally:
        if _telemetry is not None:
            _telemetry.shutdown()


app = fastapi.FastAPI(title="seal-durable-agent", lifespan=lifespan)
app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# --- chat ---------------------------------------------------------------------


class ChatRequest(pydantic.BaseModel):
    session_id: str
    messages: list[ai_sdk.UIMessage]


@app.post("/api/chat")
async def post_chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    messages, approvals = ai_sdk.to_messages(request.messages)
    await sessions.touch(request.session_id)

    # ``sendAutomaticallyWhen`` resubmits the full history after the user
    # responds to an approval, so the trailing message is the assistant turn
    # holding the answered tool part — not a new user message. That case
    # resolves the durable hooks; a trailing user message starts (or queues
    # into) a turn.
    is_approval_resume = bool(approvals) and not (
        messages and messages[-1].role == "user"
    )

    try:
        if is_approval_resume:
            await chat.submit_approvals(request.session_id, list(approvals))
        else:
            prompt = next(
                (m.text for m in reversed(messages) if m.role == "user" and m.text),
                None,
            )
            if prompt is None:
                raise fastapi.HTTPException(
                    status_code=400, detail="No user message to run"
                )
            await chat.start_or_resume(request.session_id, prompt)
    except MailboxClosed as exc:
        raise fastapi.HTTPException(status_code=410, detail=str(exc)) from exc

    return fastapi.responses.StreamingResponse(
        chat.to_sse(request.session_id),
        headers=ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )


@app.post("/api/chat/{session_id}/stop")
async def stop_chat(session_id: str) -> dict[str, str]:
    """Interrupt the in-flight turn: preempt revokes the lease, the running
    activation aborts uncommitted, and its provisional tokens are retracted.
    (The workflow port had no interruption story.)"""
    try:
        await client().send(session_id, proto.Interrupt(), preempt=True)
    except (ProcessNotFound, MailboxClosed) as exc:
        raise fastapi.HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "stopping"}


@app.get("/api/chat/{session_id}/stream")
async def resume_chat(session_id: str) -> fastapi.responses.Response:
    # ``useChat({ resume: true })`` GETs this on mount. The spool replays the
    # in-flight turn's tokens; 204 when nothing is running or parked.
    if not await chat.in_flight(session_id):
        return fastapi.responses.Response(status_code=204)
    return fastapi.responses.StreamingResponse(
        chat.to_sse(session_id),
        headers=ai_sdk.UI_MESSAGE_STREAM_HEADERS,
    )


# --- sessions -----------------------------------------------------------------


class CreateSessionRequest(pydantic.BaseModel):
    id: str
    title: str | None = None


@app.get("/api/sessions")
async def list_sessions() -> list[sessions.SessionMeta]:
    return await sessions.list_sessions()


@app.post("/api/sessions", status_code=201)
async def create_session(body: CreateSessionRequest) -> sessions.SessionMeta:
    return await sessions.create_session(body.id, title=body.title)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, object]:
    meta = await sessions.get_session(session_id)
    if meta is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    # committed turns only; an in-flight turn is rebuilt from the resumed
    # stream (GET /chat/.../stream), token-for-token via the spool replay.
    ui_messages = ai_sdk.to_ui_messages(await sessions.history(session_id))
    serialized = [
        message.model_dump(mode="json", by_alias=True) for message in ui_messages
    ]
    return {**meta.model_dump(), "messages": serialized}


@app.post("/api/sessions/{session_id}/title")
async def generate_title(session_id: str) -> sessions.SessionMeta:
    meta = await sessions.get_session(session_id)
    if meta is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    if meta.title:
        return meta

    first_text = await sessions.first_user_text(session_id)
    if not first_text:
        raise fastapi.HTTPException(
            status_code=400, detail="No user message to generate title from"
        )
    updated = await sessions.set_title(
        session_id, await sessions.generate_title(session_id, first_text)
    )
    if updated is None:
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    return updated


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await sessions.delete_session(session_id):
        raise fastapi.HTTPException(status_code=404, detail="Session not found")
    # the durable process (and its non-detached subagents) go with the chat
    with contextlib.suppress(ProcessNotFound):
        await client().cancel(session_id, by="delete_session")
    return {"status": "deleted"}


# --- attachments --------------------------------------------------------------


class UploadResponse(pydantic.BaseModel):
    url: str
    media_type: str = pydantic.Field(serialization_alias="mediaType")
    filename: str


@app.post("/api/upload")
async def upload(file: fastapi.UploadFile) -> UploadResponse:
    content = await file.read()
    media_type = file.content_type or "application/octet-stream"
    filename = file.filename or "attachment"
    async with AsyncBlobClient() as client_:
        result = await client_.put(
            f"attachments/{filename}",
            content,
            access="private",
            content_type=media_type,
            add_random_suffix=True,
        )
    return UploadResponse(
        url=f"{attachments.FILES_PREFIX}{result.pathname}",
        media_type=media_type,
        filename=filename,
    )


@app.get("/api/files/{pathname:path}")
async def get_file(pathname: str) -> fastapi.responses.Response:
    async with AsyncBlobClient() as client_:
        result = await client_.get(pathname, access="private")
    return fastapi.responses.Response(
        content=result.content,
        media_type=result.content_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )
