"""Wire protocol of the durable agent.

Three planes, matching rotor's channels:

* **Mailbox messages** (``UserMessage``, ``Generate``) — how turns are
  driven. A user prompt is just a message: sending to an idle session
  starts the next turn, sending mid-turn queues it for the turn after.
  There is no "session is busy" error path and no hook-registration race.
* **The hook model** (``Approval``) — the durable gate one gated tool call
  parks on. Resolution requires the minted token, so a ``HookResolved``
  arriving in an arm *is* the security check.
* **The provisional stream envelope** — everything published with
  ``rotor.stream()`` while an activation runs. The stream is the rumor;
  the transcript checkpoint and the ``record()`` ledger are the facts.
  Chunks carry either one model event or one lifecycle marker.

What this file replaced from the workflow port: ``TurnHook``, ``SessionHook``,
``ApprovalHook``, the hook token naming conventions, ``TurnInput``/``TurnOutput``
(state now lives on the process), and ``ToolCallContext`` (arms have ``self``).
"""

from __future__ import annotations

from typing import Any

import ai
from rotor import Resolution, message

# ── mailbox messages ─────────────────────────────────────────────────────


@message
class UserMessage:
    """One user prompt for a session that already exists."""

    text: str


@message
class Generate:
    """One model turn, self-sent. ``attempt`` rides the message so model
    retries are an ``except`` clause plus a timer, not retry machinery."""

    attempt: int = 0


@message
class Interrupt:
    """Stop generating. Sent with ``preempt=True``: the in-flight model turn
    aborts uncommitted (its tokens are retracted), this jumps the queue, and
    the arm swallows the turn that would otherwise re-run."""


@message
class Approval(Resolution):
    """The resolution model for a gated tool call's hook."""

    granted: bool
    reason: str = ""


# ── provisional stream envelope ──────────────────────────────────────────

# lifecycle marker types (rumor plane, drives the SSE connection)
TURN_STARTED = "turn.started"
SESSION_WAITING = "session.waiting"
ASSISTANT_MESSAGE = "assistant.message"
TOOL_APPROVAL_REQUESTED = "tool_approval.requested"
TOOL_RESULT = "tool.result"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_COMPLETED = "subagent.completed"

# durable record kinds (fact plane, ``record()`` / ``client.tail()``)
TURN_RECORD = "turn"
APPROVAL_RECORD = "approval_requested"
MODEL_RETRY_RECORD = "model_retry"


def model_chunk(event: Any) -> dict[str, Any]:
    """Envelope one AI SDK model stream event for ``rotor.stream()``."""
    return {"kind": "model", "event": event.model_dump(mode="json")}


def lifecycle_chunk(type_: str, **data: Any) -> dict[str, Any]:
    """Envelope one lifecycle marker for ``rotor.stream()``."""
    return {"kind": "lifecycle", "type": type_, "data": data}


# ── tool result shaping ──────────────────────────────────────────────────
# A verdict fact carries the reason, never the question; the Fanout deposit
# (the original ToolCallPart dump) supplies tool_call_id/tool_name here.


def tool_result(call: dict[str, Any], value: Any, *, kind: str = "json") -> dict[str, Any]:
    return ai.messages.ToolResultPart(
        tool_call_id=call["tool_call_id"],
        tool_name=call["tool_name"],
        result=value,
        result_kind=kind,
    ).model_dump(mode="json")


def error_result(call: dict[str, Any], reason: Any) -> dict[str, Any]:
    return tool_result(call, str(reason), kind="error")


def denied_result(call: dict[str, Any], reason: str | None) -> dict[str, Any]:
    return tool_result(
        call, f"Denied by user: {reason or 'no reason given'}", kind="error"
    )
