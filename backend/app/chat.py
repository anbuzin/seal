"""Bridge the durable agent's channels to the AI SDK UI message stream.

The agent has three channels, and this module is the only place they meet
the browser's ``useChat`` protocol:

* **live** (``client.live(scope=session_id, replay_inflight=True)``) — the
  provisional token stream. One subscription covers the session *and every
  subagent under it* (children inherit scope). ``replay_inflight`` replays
  the current model turn's spooled tokens, so reconnecting mid-generation
  resumes exactly; the workflow port's run-opener stream scanning
  (``active_run_start_index``) has no equivalent because there is nothing
  to reconstruct.
* **records** (``client.tail``) — durable facts: turn boundaries and the
  approval capabilities (hook id + token) the resume path needs.
* **state** (``client.query(Session.history)``) — the committed transcript
  the reload path serves.

Two retraction signals map to the frontend's existing ``data-reload``
handler (`getFreshParts`): ``Settled(discarded)`` (a retried model turn
took its provisional tokens back) and ``Gap`` (the lossy channel dropped
chunks; trust the durable record instead). The frontend is unchanged.

Gone from the workflow port: the ``_resume`` 40×50ms hook-registration
retry loop (mailboxes and hook rows cannot race), ``_waiting_turn_index``
and ``active_run_start_index`` (state is readable, not reconstructed), and
the per-child stream tails (scope covers the tree).
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import json
from typing import Any

import ai
import ai.types.events as events_
import ai.ui.ai_sdk as ai_sdk
import ai.ui.ai_sdk.outbound_stream as outbound_stream
import ai.ui.ai_sdk.ui_events as ui_events
import pydantic
from rotor import Chunk, Gap, HookNotPending, ProcessNotFound, Settled

from agent import proto
from agent.processes import Session
from agent.runtime import client

# TODO(draft): confirm the exported union name for model stream events; the
# conftest MockProvider yields ``events_.Event``, so a TypeAdapter over it is
# the least-surprise decode.
_EVENT: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(events_.Event)


# ── control plane ────────────────────────────────────────────────────────


async def start_or_resume(session_id: str, prompt: str) -> None:
    """Get-or-create the session process, then deliver the prompt.

    Both verbs are race-free: ``start`` is atomic get-or-create, and a send
    to a mid-turn session just queues in its mailbox.
    """
    try:
        await client().snapshot(session_id)
    except ProcessNotFound:
        await client().start(
            Session, input=prompt, id=session_id, scope=session_id
        )
        return
    await client().send(session_id, proto.UserMessage(text=prompt))


async def submit_approvals(
    session_id: str, approvals: list[Any]
) -> None:
    """Resolve each answered approval's durable hook.

    The hook id and token were recorded when the gate was minted; the token
    is the capability, so resolution needs no session-side bookkeeping and
    re-submissions land as ``HookNotPending`` (already answered) — ignored.
    """
    minted: dict[str, dict[str, Any]] = {}
    async for event in client().tail(process_id=session_id, follow=False):
        if event.kind == proto.APPROVAL_RECORD:
            minted[event.data["tool_call_id"]] = event.data
    for approval in approvals:
        info = minted.get(approval.tool_call_id)
        if info is None:
            continue
        with contextlib.suppress(HookNotPending):
            await client().resolve_hook(
                info["hook_id"],
                proto.Approval(
                    granted=approval.granted, reason=approval.reason or ""
                ),
                info["token"],
            )


async def in_flight(session_id: str) -> bool:
    """Whether there is anything for a resumed stream to attach to: a
    running/queued turn, or a turn parked on approvals."""
    try:
        snap = await client().snapshot(session_id)
    except ProcessNotFound:
        return False
    if snap.phase in ("ready", "leased") or snap.mailbox_depth > 0:
        return True
    return bool(snap.pending_hooks)


# ── the stream ───────────────────────────────────────────────────────────


async def to_sse(session_id: str) -> collections.abc.AsyncIterator[str]:
    """Stream one turn of the session as AI SDK UI SSE chunks.

    Model events go through the SDK adapter; tool results, approvals,
    retractions, and subagent progress are injected directly (the same
    bypass the workflow port used for subagent progress). All lines funnel
    through one queue so the merge is sequential.
    """
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | dict[str, Any] | None] = (
        asyncio.Queue()
    )

    async def pump_adapter() -> None:
        try:
            async for event in ai_sdk.to_stream(_model_events(session_id, queue)):
                await queue.put(event)
        finally:
            await queue.put(None)

    adapter_task = asyncio.create_task(pump_adapter())
    try:
        while True:
            line = await queue.get()
            if line is None:
                break
            if isinstance(line, dict):
                yield f"data: {json.dumps(line, separators=(',', ':'))}\n\n"
            else:
                yield outbound_stream.format_sse(line)
        yield outbound_stream.format_done_sse()
    finally:
        adapter_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await adapter_task


async def _model_events(
    session_id: str,
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | dict[str, Any] | None],
) -> collections.abc.AsyncIterator[Any]:
    """Yield this turn's model events for the SDK adapter; drive everything
    else onto ``queue`` directly. Ends at the turn's park or completion."""
    subagents: dict[str, str] = {}  # child process_id -> parent tool_call_id
    nested: dict[str, list[ai.messages.Message]] = {}  # tool_call_id -> messages
    approvals_pending = False

    async for item in client().live(scope=session_id, replay_inflight=True):
        match item:
            case Gap(process_id=pid) if pid == session_id:
                # the lossy channel dropped chunks: retract the provisional
                # step; the durable record is authoritative on reload.
                await _put_reload(queue)

            case Settled(process_id=pid, outcome=outcome) if pid == session_id:
                if outcome == "discarded":
                    # a retried model turn took its tokens back
                    await _put_reload(queue)
                elif approvals_pending:
                    return  # the turn parked on its approval gates

            case Chunk(process_id=pid, data=data) if pid == session_id:
                envelope = data or {}
                if envelope.get("kind") == "model":
                    yield _EVENT.validate_python(envelope["event"])
                    continue
                lifecycle_type = envelope.get("type")
                payload = envelope.get("data") or {}
                if lifecycle_type == proto.SESSION_WAITING:
                    return
                if lifecycle_type == proto.TOOL_APPROVAL_REQUESTED:
                    approvals_pending = True
                    await queue.put(_approval_request(payload["tool_call_id"]))
                elif lifecycle_type == proto.TOOL_RESULT:
                    await queue.put(
                        ui_events.UIToolOutputAvailableEvent(
                            tool_call_id=payload["tool_call_id"],
                            output=payload["result"].get("result"),
                        )
                    )
                elif lifecycle_type == proto.SUBAGENT_CALLED:
                    subagents[payload["child_id"]] = payload["tool_call_id"]
                    nested[payload["tool_call_id"]] = []
                elif lifecycle_type == proto.SUBAGENT_COMPLETED:
                    messages = [
                        ai.messages.Message.model_validate(m)
                        for m in payload.get("messages", [])
                    ]
                    wire = bundle_to_wire(messages)
                    if wire is not None:
                        await queue.put(
                            ui_events.UIToolOutputAvailableEvent(
                                tool_call_id=payload["tool_call_id"],
                                output=wire,
                            )
                        )

            case Chunk(process_id=pid, data=data) if pid in subagents:
                # a subagent's progress, folded into a growing nested
                # UIMessage as preliminary output on the parent's tool call
                envelope = data or {}
                if (
                    envelope.get("kind") == "lifecycle"
                    and envelope.get("type") == proto.ASSISTANT_MESSAGE
                ):
                    tool_call_id = subagents[pid]
                    message = ai.messages.Message.model_validate(
                        (envelope.get("data") or {})["message"]
                    )
                    _upsert(nested[tool_call_id], message)
                    wire = bundle_to_wire(nested[tool_call_id])
                    if wire is not None:
                        await queue.put(
                            ui_events.UIToolOutputAvailableEvent(
                                tool_call_id=tool_call_id,
                                output=wire,
                                preliminary=True,
                            )
                        )

            case _:
                continue


async def _put_reload(
    queue: asyncio.Queue[ui_events.UIMessageStreamEvent | dict[str, Any] | None],
) -> None:
    """The frontend's existing retraction protocol: finish the step, signal
    ``data-reload`` (getFreshParts drops the step), start fresh."""
    await queue.put(ui_events.UIFinishStepEvent())
    await queue.put(ui_events.UIDataEvent(data_type="reload", data={}))
    await queue.put(ui_events.UIStartStepEvent())


def _approval_request(tool_call_id: str) -> dict[str, Any]:
    # TODO(draft): emit through the SDK's typed event once its name is
    # confirmed; this is the wire shape the frontend's approval UI consumes
    # (approval id ``approve_{tool_call_id}`` — see tests/app/test_server.py).
    return {
        "type": "tool-approval-request",
        "toolCallId": tool_call_id,
        "approvalId": f"approve_{tool_call_id}",
    }


# ── reload helpers ───────────────────────────────────────────────────────


async def history(session_id: str) -> list[ai.messages.Message]:
    """The committed transcript, straight off the checkpoint (lease-free)."""
    try:
        value, _revision = await client().query(session_id, Session.history)
    except ProcessNotFound:
        return []
    return [ai.messages.Message.model_validate(m) for m in value]


def bundle_to_wire(
    messages: collections.abc.Sequence[ai.messages.Message],
) -> dict[str, object] | None:
    """Flatten a subagent transcript into one nested wire ``UIMessage``.

    The child may take several turns; fold all of its assistant bubbles into
    a single ``UIMessage`` (anchored on the first) so the whole trajectory
    renders under the parent's ``subagent`` tool call.
    """
    bubbles = [
        bubble
        for bubble in ai_sdk.to_ui_messages(list(messages))
        if bubble.role == "assistant"
    ]
    if not bubbles:
        return None
    nested = bubbles[0].model_dump(mode="json", by_alias=True)
    nested["parts"] = [
        part
        for bubble in bubbles
        for part in bubble.model_dump(mode="json", by_alias=True)["parts"]
    ]
    return nested


def _upsert(messages: list[ai.messages.Message], message: ai.messages.Message) -> None:
    """Replace the message with the same id, else append it."""
    for index, existing in enumerate(messages):
        if existing.id == message.id:
            messages[index] = message
            return
    messages.append(message)
