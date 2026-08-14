from __future__ import annotations

from typing import Annotated, Any, Literal

import ai
import pydantic
import vercel.workflow


# external decision for a single gated tool call
class ToolApprovalResponse(pydantic.BaseModel):
    tool_call_id: str
    granted: bool
    reason: str | None = None


# ai sdk gates a tool behind a hook labelled ``approve_{tool_call_id}``.
TOOL_APPROVAL_HOOK_PREFIX = "approve_"


# turn inbox: turn workflow dispatches work in a task, and repeatedly
# suspends on the inbox hook. the hook is resumed with InboxCommand.


def inbox_token(session_id: str) -> str:
    return f"seal-inbox:{session_id}"


def control_scope(session_id: str, turn_index: int) -> str:
    """Namespace for control signals for an agent turn.

    Control signals ride the durable stream to get into llm_step
    and other steps, and this is their agent-specific namespace.
    """
    return f"{session_id}#turn-{turn_index}"


# a human's decision for one gated tool call.
class Approval(pydantic.BaseModel):
    kind: Literal["approval"] = "approval"
    response: ToolApprovalResponse


# sent by the agent.run task when the run is finished
class AgentFinished(pydantic.BaseModel):
    kind: Literal["agent_finished"] = "agent_finished"


class Cancel(pydantic.BaseModel):
    kind: Literal["cancel"] = "cancel"


type InboxCommand = Approval | AgentFinished | Cancel


class InboxHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    command: Annotated[InboxCommand, pydantic.Field(discriminator="kind")]


# current state saved to disk


class SessionState(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    # index of the last turn that ran (the next turn is turn_index + 1).
    turn_index: int = 0


# turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]

    # subagent behavior is different from root agent
    is_subagent: bool = False
    # set for subagent turns: the parent's hook to resume with the TurnOutput.
    parent_hook_token: str | None = None

    # index of this turn within its session (always 0 for subagent turns).
    turn_index: int = 0
    # turn's root span. llm_steps and child turns nest under it.
    turn_span: ai.experimental_telemetry.Span | None = None


# in-process context of the running tool call, set by the agent loop around
# each schedule so a tool can reach it without smuggling args. never journaled.
class ToolCallContext(pydantic.BaseModel):
    session_id: str
    tool_call_id: str
    # the enclosing turn's root span; a spawned child turn nests under it.
    turn_span: ai.experimental_telemetry.Span | None = None
    # the enclosing agent's control message scope on the durable stream
    control_scope: str | None = None


# result of one llm_step: the (possibly partial) assistant message and
# whether the step exited on the turn's cancel flag.
class LlmStepResult(pydantic.BaseModel):
    cancelled: bool
    message: ai.messages.Message


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error", "cancelled"]
    messages: list[ai.messages.Message]
    error: str | None = None


# resumed by a subagent turn to deliver its result to the waiting parent.
class SubagentHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    output: TurnOutput


# Durable stream

SESSION_NAMESPACE = "session"

SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
TURN_STARTED = "turn.started"
TURN_COMPLETED = "turn.completed"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_COMPLETED = "subagent.completed"
TOOL_APPROVAL_REQUESTED = "tool_approval.requested"
RELOAD_REQUESTED = "reload.requested"
DEFAULT_STREAM_NAMESPACE = "default"
DEFAULT_STREAM_POLL_INTERVAL = 0.05
WRITABLE_STREAM_HANDLE_TYPE = "seal.durable_agent.writable_stream"


class LifecycleEvent(pydantic.BaseModel):
    kind: Literal["lifecycle"] = "lifecycle"
    type: str
    data: dict[str, Any] = pydantic.Field(default_factory=dict)
    # ISO 8601 UTC string. None when constructed inside a workflow body
    # (datetime is sandbox-restricted); stamped by the write function.
    at: str | None = None


type StreamEvent = ai.events.AgentEvent | LifecycleEvent

STREAM_EVENT_ADAPTER: pydantic.TypeAdapter[StreamEvent] = pydantic.TypeAdapter(
    StreamEvent
)
