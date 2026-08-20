"""Bridge the durable session protocol to the AI SDK UI message stream.

The durable agent persists ``ai.events.AgentEvent | proto.LifecycleEvent`` to a
per-session stream. The browser's ``useChat`` speaks the AI SDK UI protocol, so
we tail one turn of the durable stream, hand the ``AgentEvent``s to the SDK's
``to_stream`` adapter (lifecycle events stay server-side), and drive control
flow off the lifecycle events.

Two lifecycle features surface to the UI:

  * tool approvals — a gated tool emits a ``tool-approval-request`` part (built by
    the SDK adapter from the pending hook). The turn parks; the browser replies
    with ``addToolApprovalResponse`` which arrives on the next ``POST /chat`` and
    is forwarded back into the durable hook by :func:`submit_approvals`.
  * subagents — a delegated child runs as background work owned by the session.
    The tool returns an immediate acknowledgement; when the child finishes, the
    session injects its report as a new user message and starts another root turn.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import typing

import ai
import ai.ui.ai_sdk as ai_sdk
import ai.ui.ai_sdk.outbound_stream as outbound_stream
import ai.ui.ai_sdk.ui_events as ui_events
import vercel.workflow

from agent import driver, proto, session, stream

_SESSION_TERMINAL = {
    proto.SESSION_COMPLETED,
    proto.SESSION_FAILED,
}


class SessionUnavailableError(Exception):
    """The session cannot accept a new user message right now."""


async def active_run_start_index(session_id: str) -> int | None:
    """Return the stream index to resume the in-flight run from, else ``None``.

    A *run* is the whole agent turn the SDK adapter folds into a single UI
    message (one ``UIStartEvent``). It begins at its opener (``session.started``
    / ``turn.started``) and ends only at a *terminal* boundary (``session.*``) —
    a ``tool_approval.requested`` park is mid-run, not the end of one, because the
    same ``run_turn`` resumes in place once the approval lands.

    Used only by the cold-reload path (``GET /chat/{id}/stream``): there is no
    submitted message to continue, so re-tailing from the opener replays the
    assistant message's stable id and the SDK rebuilds the same UI message. (The
    live approval POST doesn't need this — its resubmit already carries the
    assistant message, so its continuation folds in; see :func:`submit_approvals`.)

    The run is in flight (resumable) when its opener has no terminal after it.
    """
    run_start: int | None = None
    seen_boundary = True
    session_opener = False
    index = -1
    async for event in stream.replay(session_id):
        index += 1
        if not isinstance(event, proto.LifecycleEvent):
            continue
        if event.type == proto.TURN_STARTED and (seen_boundary or session_opener):
            run_start = index
            seen_boundary = False
            session_opener = False
        elif event.type == proto.SESSION_STARTED and seen_boundary:
            # Compatibility for a first turn captured before turn.started existed.
            run_start = index
            seen_boundary = False
            session_opener = True
        elif event.type == proto.SESSION_WAITING:
            seen_boundary = event.data.get("active_background_tasks", 0) == 0
        elif event.type in _SESSION_TERMINAL:
            seen_boundary = True
    return None if seen_boundary else run_start


async def start_or_resume(session_id: str, prompt: str) -> int:
    """Start a new session or resume a parked one.

    Returns the stream index to tail from so only the new turn reaches the
    client.
    """
    if await active_run_start_index(session_id) is not None:
        raise SessionUnavailableError("A turn is already running")

    start_index = await stream.tail_index(session_id) + 1

    if await session.read_session(session_id) is None:
        await vercel.workflow.start(
            driver.run_session,
            proto.SessionInput(session_id=session_id),
        )
        await _wait_for_lifecycle(session_id, proto.SESSION_STARTED)

    await _resume(
        proto.session_inbox_token(session_id),
        proto.SessionInboxHook(command=proto.NewUserMessage(prompt=prompt)),
    )
    return start_index


async def submit_approvals(
    session_id: str, approvals: list[proto.ToolApprovalResponse]
) -> int:
    """Send each UI approval decision through the session's public inbox.

    Resume after the durable approval marker, not at the current tail: background
    child events may arrive while the human is deciding and must still reach the
    one backend-reconciled UI stream.
    """
    if await active_run_start_index(session_id) is None:
        raise SessionUnavailableError("No turn is waiting for approval")

    start_index = (
        await _latest_event_index(session_id, proto.TOOL_APPROVAL_REQUESTED) + 1
    )
    token = proto.session_inbox_token(session_id)
    for approval in approvals:
        await _resume(
            token,
            proto.SessionInboxHook(command=proto.SubmitToolApproval(response=approval)),
        )
    return start_index


def background_task_output(
    messages: collections.abc.Sequence[ai.messages.Message],
) -> list[dict[str, object]]:
    """Return child assistant bubbles in the nested UI shape the tool card renders."""
    return [
        message.model_dump(mode="json", by_alias=True)
        for message in ai_sdk.to_ui_messages(list(messages))
        if message.role == "assistant"
    ]


def project_background_tasks(
    ui_messages: list[dict[str, object]],
    tasks: collections.abc.Mapping[str, proto.BackgroundTaskState],
) -> list[dict[str, object]]:
    """Overlay durable task presentation state without changing model history."""
    for message in ui_messages:
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for raw_part in parts:
            if not isinstance(raw_part, dict):
                continue
            part = typing.cast(dict[str, object], raw_part)
            task_id = part.get("toolCallId")
            task = tasks.get(task_id) if isinstance(task_id, str) else None
            if task is None:
                continue
            if task.status == "failed":
                part["state"] = "output-error"
                part["errorText"] = task.error or "Unknown subagent error"
                part.pop("output", None)
                part.pop("preliminary", None)
            else:
                part["state"] = "output-available"
                if task.status == "completed" or task.messages:
                    part["output"] = background_task_output(task.messages)
                part["preliminary"] = task.status == "running"
    return ui_messages


def _upsert_message(
    messages: list[ai.messages.Message], message: ai.messages.Message
) -> None:
    for index, existing in enumerate(messages):
        if existing.id == message.id:
            messages[index] = message
            return
    messages.append(message)


async def to_sse(
    session_id: str, start_index: int
) -> collections.abc.AsyncIterator[str]:
    """Stream one logical UI run as AI SDK UI SSE chunks.

    A UI run starts with a user turn and remains open across background subagent
    progress and the automatic root turn that reports the results. All durable
    events are already ordered on the parent session stream, so the backend can
    reconcile them before the browser sees a single AI SDK stream.
    """
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None] = asyncio.Queue()

    async def pump_adapter() -> None:
        events = _ui_run_events(session_id, start_index, queue)
        async for event in ai_sdk.to_stream(events):
            await queue.put(event)
        await queue.put(None)

    adapter_task = asyncio.create_task(pump_adapter())
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            yield outbound_stream.format_sse(line)
        yield outbound_stream.format_done_sse()
    finally:
        adapter_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await adapter_task


async def _ui_run_events(
    session_id: str,
    start_index: int,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | None],
) -> collections.abc.AsyncIterator[ai.events.AgentEvent]:
    """Yield one causally connected UI run from the parent durable stream.

    Root-turn ``AgentEvent`` objects pass through the stock Python AI SDK adapter.
    Proxied child events update the original subagent tool call directly on the
    same UI queue. A root turn ending is not a presentation boundary while its
    background work or automatic follow-up remains active; quiescent
    ``session.waiting`` is the boundary.
    """
    active_tasks: set[str] = set()
    task_messages: dict[str, list[ai.messages.Message]] = {}

    async for event in stream.get_readable(session_id, start_index=start_index):
        if not isinstance(event, proto.LifecycleEvent):
            yield event
            continue

        task_id = event.data.get("tool_call_id")
        if event.type == proto.SUBAGENT_CALLED and isinstance(task_id, str):
            active_tasks.add(task_id)
            task_messages.setdefault(task_id, [])
        elif event.type == proto.SUBAGENT_EVENT and isinstance(task_id, str):
            child_event = proto.STREAM_EVENT_ADAPTER.validate_python(
                event.data.get("event")
            )
            if (
                isinstance(child_event, proto.LifecycleEvent)
                and child_event.type == proto.RELOAD_REQUESTED
            ):
                task_messages[task_id] = []
            message = getattr(child_event, "message", None)
            if isinstance(message, ai.messages.Message):
                _upsert_message(task_messages.setdefault(task_id, []), message)
            output = background_task_output(task_messages.get(task_id, []))
            if output:
                await queue.put(
                    ui_events.UIToolOutputAvailableEvent(
                        tool_call_id=task_id,
                        output=output,
                        preliminary=True,
                    )
                )
        elif event.type == proto.SUBAGENT_COMPLETED and isinstance(task_id, str):
            active_tasks.discard(task_id)
            if event.data.get("is_error"):
                await queue.put(
                    ui_events.UIToolOutputErrorEvent(
                        tool_call_id=task_id,
                        error_text=str(
                            event.data.get("error") or "Unknown subagent error"
                        ),
                    )
                )
            else:
                messages = [
                    ai.messages.Message.model_validate(message)
                    for message in event.data.get("messages", [])
                ]
                task_messages[task_id] = messages
                await queue.put(
                    ui_events.UIToolOutputAvailableEvent(
                        tool_call_id=task_id,
                        output=background_task_output(messages),
                        preliminary=False,
                    )
                )
        elif event.type == proto.TOOL_APPROVAL_REQUESTED:
            return
        elif event.type == proto.RELOAD_REQUESTED:
            await queue.put(ui_events.UIFinishStepEvent())
            await queue.put(ui_events.UIDataEvent(data_type="reload", data={}))
            await queue.put(ui_events.UIStartStepEvent())
        elif event.type == proto.SESSION_WAITING:
            remaining = event.data.get("active_background_tasks")
            if remaining == 0 or (remaining is None and not active_tasks):
                return
        elif event.type in _SESSION_TERMINAL:
            return


async def _latest_event_index(session_id: str, type_: str) -> int:
    found: int | None = None
    index = -1
    async for event in stream.replay(session_id):
        index += 1
        if isinstance(event, proto.LifecycleEvent) and event.type == type_:
            found = index
    if found is None:
        raise SessionUnavailableError(f"No {type_} event to resume from")
    return found


async def _wait_for_lifecycle(session_id: str, type_: str) -> None:
    while True:
        async for event in stream.replay(session_id):
            if isinstance(event, proto.LifecycleEvent) and event.type == type_:
                return
        await asyncio.sleep(0.05)


async def _resume(token: str, hook: vercel.workflow.BaseHook) -> None:
    """Resolve a workflow hook, retrying while the driver registers it."""
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)
