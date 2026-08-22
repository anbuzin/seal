"""The durable agent: one process per conversation.

The workflow port split the agent into a ``run_session`` driver workflow and
one ``run_turn`` child workflow per turn, joined by hooks, because a parked
workflow run is a held resource. A rotor process is a row: idle is free, so
the whole conversation is **one keyed ``Session`` process** whose arms are
the agent's states:

    Start ──▶ Generate ──▶ (tool verdicts | approvals | subagent) ──▶ Generate
                 │
                 └─ no tool calls ──▶ idle (the mailbox wakes us)

Commit boundaries: each model turn is one activation (assistant message,
tool fan-out, hooks, and spawns land atomically); each tool verdict is one
activation. A worker dying mid-conversation resumes from the last committed
message — no replay and no determinism contract, so handlers stream tokens,
run eager tools, and call telemetry directly.

Streaming: ``spool = True`` gives every ``stream()`` chunk write-through
durability for the in-flight activation, so a browser reconnecting
mid-generation replays the current turn's tokens
(``client.live(replay_inflight=True)``). A retried model turn is a new
activation: ``Settled(discarded)`` retracts the old tokens on the live
channel and the spool's validity facts make the discarded attempt
unreachable — the workflow port's ``reload.requested`` patch protocol has no
equivalent here because the failure it patched cannot happen.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import field
from typing import Any, ClassVar

import ai
from rotor import (
    ChildDone,
    ChildFailed,
    DurableProcess,
    HookExpired,
    HookResolved,
    Journal,
    Start,
    on,
    query,
    record,
    stream,
)
from rotor.patterns import Fanout

from agent import proto
from agent.tools import (
    EAGER,
    GATED,
    MODEL_ID,
    SUBAGENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    bash,
    execute_tool,
    generate_image,
    run_tool,
    subagent,
    web_fetch,
)


class SessionState:
    # the conversation, as AI SDK message dumps; Journal seals spans into
    # immutable blob chunks so a long chat never rewrites its full history
    transcript: Journal[dict] = field(default_factory=Journal)
    tools: Fanout = field(default_factory=Fanout)  # expected verdicts, this round
    results: list[dict] = field(default_factory=list)  # settled ToolResultParts
    queued: list[str] = field(default_factory=list)  # prompts that arrived mid-turn
    turn: int = 0
    turn_queued: bool = False  # a Generate is staged or enqueued
    interrupted: bool = False  # swallow the next Generate (user hit stop)


async def _run_eager(call: dict[str, Any]) -> dict[str, Any]:
    """Eager execution may be cancelled (turn rollback) but never raises:
    a failure becomes an error-kind result the model reads."""
    try:
        return await execute_tool(call)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        return proto.error_result(call, error)


def _transient(error: BaseException) -> bool:
    """Policy is code: what counts as a retryable model failure."""
    import httpx

    return isinstance(
        error, httpx.TransportError | TimeoutError | asyncio.TimeoutError
    )


class AgentBase(DurableProcess[SessionState]):
    """The shared loop. Never registered or spawned itself — ``Session``
    and ``Subagent`` are the concrete processes."""

    spool = True  # durable replay of the in-flight turn's tokens
    handle_timeout = "10m"  # one model turn's wall-clock budget
    retention = "30d"

    SYSTEM: ClassVar[str] = SYSTEM_PROMPT
    GATES: ClassVar[frozenset[str]] = GATED
    TOOLSET: ClassVar[list] = [bash, web_fetch, generate_image, subagent]

    # ── lifecycle ────────────────────────────────────────────────────────

    @on
    async def start(self, msg: Start):
        self.state.transcript.extend(
            [
                ai.system_message(self.SYSTEM).model_dump(mode="json"),
                ai.user_message(msg.input or "").model_dump(mode="json"),
            ]
        )
        self.state.turn_queued = True
        self.send(self.ref, proto.Generate())

    @on
    async def user(self, msg: proto.UserMessage):
        """A prompt is a message. Idle: starts the next turn. Mid-turn: is
        queued and folded in after the pending tool results, keeping the
        tool-call/tool-result adjacency the model requires."""
        if self.state.tools.settled and not self.state.turn_queued:
            self.state.transcript.append(
                ai.user_message(msg.text).model_dump(mode="json")
            )
            self.state.turn_queued = True
            self.send(self.ref, proto.Generate())
        else:
            self.state.queued.append(msg.text)

    @on
    async def interrupt(self, msg: proto.Interrupt):
        """User hit stop. The preempt already aborted the in-flight turn
        uncommitted; this swallows its redelivered Generate and abandons the
        round (in-flight verdicts redeem as stale)."""
        self.state.interrupted = True
        self.state.tools.clear()
        self.state.results.clear()
        await stream(
            proto.lifecycle_chunk(proto.SESSION_WAITING, turn_index=self.state.turn)
        )

    # ── one model turn ───────────────────────────────────────────────────

    async def _model_turn(self, msg: proto.Generate) -> ai.messages.Message | None:
        state = self.state
        if state.interrupted:
            state.interrupted = False
            return None

        # fold the settled round's verdicts, then any prompts that queued up
        if state.results:
            parts = [
                ai.messages.ToolResultPart.model_validate(result)
                for result in state.results
            ]
            state.transcript.append(ai.tool_message(*parts).model_dump(mode="json"))
            state.results.clear()
        for text in state.queued:
            state.transcript.append(ai.user_message(text).model_dump(mode="json"))
        state.queued.clear()

        await stream(proto.lifecycle_chunk(proto.TURN_STARTED, turn_index=state.turn))
        history = [
            ai.messages.Message.model_validate(m) async for m in state.transcript
        ]

        # Eager tools launch the moment the model finishes emitting the call,
        # concurrent with the rest of the model stream. In-process tasks: a
        # turn that rolls back cancels them; nothing durable was staged.
        eager: dict[str, asyncio.Task[dict[str, Any]]] = {}
        try:
            async with asyncio.TaskGroup() as tg:
                async with ai.stream(
                    ai.get_model(MODEL_ID), history, tools=self.TOOLSET
                ) as response:
                    async for event in response:
                        if getattr(event, "replay", False):
                            continue
                        await stream(proto.model_chunk(event))
                        if (
                            isinstance(event, ai.types.events.ToolEnd)
                            and event.tool_call.tool_name in EAGER
                        ):
                            call = event.tool_call.model_dump(mode="json")
                            eager[event.tool_call.tool_call_id] = tg.create_task(
                                _run_eager(call)
                            )
                # harvest inside the TaskGroup; _run_eager never raises
                harvested = {cid: await task for cid, task in eager.items()}
            reply = response.message
        except Exception as error:
            if msg.attempt < 2 and _transient(error):
                # catch = commit: the retry timer and the discard verdict land
                # atomically; live subscribers retract this turn's tokens.
                stream.discard()
                record(proto.MODEL_RETRY_RECORD, {"attempt": msg.attempt + 1})
                self.schedule(proto.Generate(attempt=msg.attempt + 1), delay="5s")
                return None
            raise  # → HandlingFailed → this process's floor / the parent's arm

        state.transcript.append(reply.model_dump(mode="json"))
        await stream(
            proto.lifecycle_chunk(
                proto.ASSISTANT_MESSAGE, message=reply.model_dump(mode="json")
            )
        )
        state.turn += 1
        record(proto.TURN_RECORD, {"turn": state.turn})

        for call_part in reply.tool_calls:
            call = call_part.model_dump(mode="json")
            cid = call_part.tool_call_id
            if cid in harvested:
                # already answered in-handler; no child, no expectation
                await self._finish_call(harvested[cid])
            elif call_part.tool_name in self.GATES:
                key = state.tools.expect(data=call)
                hook = self.create_hook(proto.Approval, key=key, data=call)
                record(
                    proto.APPROVAL_RECORD,
                    {"tool_call_id": cid, "hook_id": hook.id, "token": hook.token},
                )
                await stream(
                    proto.lifecycle_chunk(
                        proto.TOOL_APPROVAL_REQUESTED, tool_call_id=cid
                    )
                )
            elif call_part.tool_name == "subagent":
                key = state.tools.expect(data=call)
                args = json.loads(call["tool_args"] or "{}")
                ref = self.spawn(Subagent, input=args.get("prompt", ""), key=key)
                await stream(
                    proto.lifecycle_chunk(
                        proto.SUBAGENT_CALLED,
                        tool_call_id=cid,
                        child_id=ref.id,
                        name=args.get("name") or "subagent",
                    )
                )
            else:
                key = state.tools.expect(data=call)
                self.spawn(run_tool, input={"call": call}, key=key)

        if reply.tool_calls and state.tools.settled and not state.turn_queued:
            # every call was eager and already harvested: next turn now
            state.turn_queued = True
            self.send(self.ref, proto.Generate())
        return reply

    # ── verdicts: each arm is one commit ─────────────────────────────────

    @on(proto.Approval.Resolved)
    async def approved(self, msg: HookResolved):
        call = self.state.tools.pending.get(msg.key)
        if call is None:
            return  # a cleared round's straggler
        if msg.payload.granted:
            # keep the deposit: the verdict that settles it is the tool's
            self.spawn(run_tool, input={"call": call}, key=msg.key)
        else:
            self.state.tools.settle(key=msg.key)
            await self._finish_call(proto.denied_result(call, msg.payload.reason))

    @on(proto.Approval.Expired)
    async def approval_expired(self, msg: HookExpired):
        call = self.state.tools.settle(key=msg.key)
        if call is not None:
            await self._finish_call(proto.error_result(call, "approval timed out"))

    @on(run_tool.Done)
    async def tool_done(self, msg: ChildDone):
        if self.state.tools.settle(key=msg.key) is None:
            return
        await self._finish_call(msg.output)

    @on(run_tool.Failed)
    async def tool_failed(self, msg: ChildFailed):
        call = self.state.tools.settle(key=msg.key)
        if call is None:
            return
        # The fact carries the reason, never the question — the deposit does.
        # An error-kind result lets the model read the failure and correct
        # course instead of waiting for an answer that never comes.
        await self._finish_call(proto.error_result(call, msg.reason))

    async def _finish_call(self, result: dict[str, Any]) -> None:
        self.state.results.append(result)
        await stream(
            proto.lifecycle_chunk(
                proto.TOOL_RESULT,
                tool_call_id=result["tool_call_id"],
                result=result,
            )
        )
        if self.state.tools.settled and not self.state.turn_queued:
            self.state.turn_queued = True
            self.send(self.ref, proto.Generate())

    # ── reload path: the committed fact, lease-free ──────────────────────

    @query
    async def history(self) -> list[dict]:
        return [m async for m in self.state.transcript]


class Subagent(AgentBase):
    """One focused delegation: the same loop, ungated, terminal at rest.

    A subagent cannot surface an approval UI, so bash runs ungated (the
    demo's rule, kept — the workflow port needed a ``dataclasses.replace``
    copy of the tool for this; here it is one dispatch-set difference). It
    inherits the parent's scope, so the one live subscription covering the
    session already carries its tokens; and it is a non-detached child, so
    cancelling the session cancels it. The workflow port had no story for
    either.
    """

    SYSTEM = SUBAGENT_SYSTEM_PROMPT
    GATES = frozenset()
    TOOLSET = [bash, web_fetch, generate_image]  # no further delegation

    @on
    async def generate(self, msg: proto.Generate):
        self.state.turn_queued = False
        reply = await self._model_turn(msg)
        if reply is None or reply.tool_calls:
            return
        # done: the transcript tail rides ChildDone, atomic with this commit
        tail = [
            m
            async for m in self.state.transcript
            if m.get("role") in ("assistant", "tool")
        ]
        self.stop(output={"text": reply.text or "", "messages": tail})


class Session(AgentBase):
    """The main agent: gated bash, subagent delegation, keyed per chat."""

    @on
    async def generate(self, msg: proto.Generate):
        self.state.turn_queued = False
        reply = await self._model_turn(msg)
        if reply is not None and not reply.tool_calls:
            await stream(
                proto.lifecycle_chunk(
                    proto.SESSION_WAITING, turn_index=self.state.turn
                )
            )
            # idle. Free. The mailbox wakes us.

    # ── subagent joins (child type in class position) ────────────────────

    @on(Subagent.Done)
    async def subagent_done(self, msg: ChildDone):
        call = self.state.tools.settle(key=msg.key)
        if call is None:
            return
        output = msg.output or {}
        await stream(
            proto.lifecycle_chunk(
                proto.SUBAGENT_COMPLETED,
                tool_call_id=call["tool_call_id"],
                is_error=False,
                messages=output.get("messages", []),
            )
        )
        await self._finish_call(
            proto.tool_result(call, output.get("text") or "(subagent gave no answer)")
        )

    @on(Subagent.Failed)
    async def subagent_failed(self, msg: ChildFailed):
        call = self.state.tools.settle(key=msg.key)
        if call is None:
            return
        await stream(
            proto.lifecycle_chunk(
                proto.SUBAGENT_COMPLETED,
                tool_call_id=call["tool_call_id"],
                is_error=True,
                messages=[],
            )
        )
        await self._finish_call(proto.error_result(call, msg.reason))
