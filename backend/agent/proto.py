from __future__ import annotations

from typing import Annotated, Any, Literal

import ai
import pydantic
import vercel.workflow

# Session inputs / outputs


# external decision for a single gated tool call.
class ToolApprovalResponse(pydantic.BaseModel):
    tool_call_id: str
    granted: bool
    reason: str | None = None


# ai SDK gates a tool behind a hook labelled ``approve_{tool_call_id}``.
TOOL_APPROVAL_HOOK_PREFIX = "approve_"


def session_inbox_token(session_id: str) -> str:
    return f"seal-session-inbox:{session_id}"


def turn_inbox_token(session_id: str, turn_index: int) -> str:
    return f"seal-turn-inbox:{session_id}:{turn_index}"


class SessionInput(pydantic.BaseModel):
    session_id: str


class SessionOutput(pydantic.BaseModel):
    session_id: str
    output: str
    is_error: bool = False


class NewUserMessage(pydantic.BaseModel):
    kind: Literal["new_user_message"] = "new_user_message"
    prompt: str | None = None
    close: bool = False


class SubmitToolApproval(pydantic.BaseModel):
    kind: Literal["submit_tool_approval"] = "submit_tool_approval"
    response: ToolApprovalResponse


class BackgroundTaskState(pydantic.BaseModel):
    task_id: str
    child_session_id: str
    name: str
    status: Literal["running", "completed", "failed"] = "running"
    messages: list[ai.messages.Message] = pydantic.Field(default_factory=list)
    error: str | None = None


class SessionState(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    background_tasks: dict[str, BackgroundTaskState] = pydantic.Field(
        default_factory=dict
    )
    hidden_ui_message_ids: set[str] = pydantic.Field(default_factory=set)
    # While a logical UI run is active, GET /sessions exposes only messages before
    # this index and the reconnect stream rebuilds the assistant response from its
    # durable opener. The triggering user message is therefore still visible.
    active_ui_run_message_index: int | None = None


# Turn inputs / outputs


class TurnInput(pydantic.BaseModel):
    session_id: str
    messages: list[ai.messages.Message]
    # gated turns expose bash behind approval + subagent; ungated (subagent
    # children) run bash directly and cannot delegate further.
    gated: bool = True
    # Background turns report completion to the owning root session.
    parent_session_id: str | None = None
    background_task_id: str | None = None
    # index of this turn within its session (always 0 for subagent turns).
    turn_index: int = 0
    # turn's root span. llm_steps and child turns nest under it.
    turn_span: ai.experimental_telemetry.Span | None = None


# in-process context of the running tool call, set by the agent loop around
# each schedule so a tool can reach it without smuggling args. never journaled.
class ToolCallContext(pydantic.BaseModel):
    session_id: str
    turn_index: int
    tool_call_id: str
    # the enclosing turn's root span; a spawned child turn nests under it.
    turn_span: ai.experimental_telemetry.Span | None = None


class TurnOutput(pydantic.BaseModel):
    kind: Literal["suspend", "error"]
    messages: list[ai.messages.Message]
    error: str | None = None


class TurnApproval(pydantic.BaseModel):
    kind: Literal["turn_approval"] = "turn_approval"
    response: ToolApprovalResponse


class AgentFinished(pydantic.BaseModel):
    kind: Literal["agent_finished"] = "agent_finished"
    output: TurnOutput


type TurnCommand = TurnApproval | AgentFinished


# all work entering one active turn is delivered through its private inbox.
class TurnInboxHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    command: Annotated[TurnCommand, pydantic.Field(discriminator="kind")]


class TurnFinished(pydantic.BaseModel):
    kind: Literal["turn_finished"] = "turn_finished"
    turn_index: int
    output: TurnOutput


class StartBackgroundTask(pydantic.BaseModel):
    kind: Literal["start_background_task"] = "start_background_task"
    task_id: str
    task_kind: Literal["subagent"] = "subagent"
    prompt: str
    name: str
    parent_turn_index: int
    parent_span: ai.experimental_telemetry.Span | None = None


class BackgroundTaskFinished(pydantic.BaseModel):
    kind: Literal["background_task_finished"] = "background_task_finished"
    task_id: str
    task_kind: Literal["subagent"] = "subagent"
    output: TurnOutput


type SessionCommand = (
    NewUserMessage
    | SubmitToolApproval
    | TurnFinished
    | StartBackgroundTask
    | BackgroundTaskFinished
)


# all work entering the long-lived session is delivered through this inbox.
class SessionInboxHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    command: Annotated[SessionCommand, pydantic.Field(discriminator="kind")]


# Durable stream

SESSION_STARTED = "session.started"
SESSION_WAITING = "session.waiting"
SESSION_COMPLETED = "session.completed"
SESSION_FAILED = "session.failed"
TURN_STARTED = "turn.started"
TURN_COMPLETED = "turn.completed"
SUBAGENT_CALLED = "subagent.called"
SUBAGENT_EVENT = "subagent.event"
SUBAGENT_COMPLETED = "subagent.completed"
TOOL_APPROVAL_REQUESTED = "tool_approval.requested"
RELOAD_REQUESTED = "reload.requested"


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
