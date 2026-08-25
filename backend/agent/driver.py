import asyncio
import collections.abc
import contextlib

import ai
import vercel.workflow

import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step(max_retries=0)
async def spawn_turn_workflow(turn_input: proto.TurnInput) -> str:
    if ai.experimental_telemetry.is_enabled():
        turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        turn_input = turn_input.model_copy(update={"turn_span": turn_span})
    started = await vercel.workflow.start(turn.run_turn, turn_input)
    return started.run_id


@workflow.step
async def load_session(session_id: str) -> proto.SessionState | None:
    return await session.read_session(session_id)


@workflow.step
async def save_session(state: proto.SessionState) -> None:
    await session.write_session(state)


@workflow.step
async def forward_tool_approval(
    session_id: str,
    turn_index: int,
    response: proto.ToolApprovalResponse,
) -> None:
    hook = proto.TurnInboxHook(command=proto.TurnApproval(response=response))
    token = proto.turn_inbox_token(session_id, turn_index)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


async def _buffer_inbox(
    inbox: collections.abc.AsyncIterator[proto.SessionInboxHook],
) -> collections.abc.AsyncGenerator[proto.SessionInboxHook]:
    queue: asyncio.Queue[proto.SessionInboxHook | Exception | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for received in inbox:
                await queue.put(received)
        except Exception as error:
            await queue.put(error)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            received = await queue.get()
            if received is None:
                return
            if isinstance(received, Exception):
                raise received
            yield received
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@workflow.workflow
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_session(session_input: proto.SessionInput) -> proto.SessionOutput:
    session_id = session_input.session_id

    restored = await load_session(session_id)
    state = (
        restored
        if restored is not None
        else proto.SessionState(
            session_id=session_id,
            messages=[ai.system_message(turn.SYSTEM_PROMPT)],
        )
    )
    await save_session(state)
    await turn.write_event(session_id, stream.session_started())

    inbox = proto.SessionInboxHook.wait(token=proto.session_inbox_token(session_id))
    active_turn_index: int | None = None
    next_turn_index = 0

    async for received in _buffer_inbox(inbox):
        command = received.command
        completed_turn_index: int | None = None

        match command:
            case proto.NewUserMessage():
                if active_turn_index is not None:
                    # The HTTP boundary rejects this today. Keep the workflow
                    # defensive until queued followups are implemented.
                    continue
                if command.close:
                    await turn.write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    inbox.dispose()
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    )

                state.messages.append(ai.user_message(command.prompt or ""))
                await save_session(state)

                active_turn_index = next_turn_index
                next_turn_index += 1
                await turn.write_event(
                    session_id,
                    stream.turn_started(turn_index=active_turn_index),
                )
                await spawn_turn_workflow(
                    proto.TurnInput(
                        session_id=session_id,
                        messages=state.messages,
                        turn_index=active_turn_index,
                    )
                )

            case proto.SubmitToolApproval():
                if active_turn_index is None:
                    continue
                await forward_tool_approval(
                    session_id,
                    active_turn_index,
                    command.response,
                )

            case proto.TurnFinished():
                if command.turn_index != active_turn_index:
                    continue

                turn_result = command.output
                state.messages = turn_result.messages
                await save_session(state)
                await turn.write_event(
                    session_id,
                    stream.turn_completed(
                        turn_index=command.turn_index, kind=turn_result.kind
                    ),
                )
                completed_turn_index = command.turn_index
                active_turn_index = None

                if turn_result.kind == "error":
                    await turn.write_event(
                        session_id, stream.session_completed(is_error=True)
                    )
                    await turn.close_stream(session_id)
                    inbox.dispose()
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=turn_result.error or _last_text(state.messages),
                        is_error=True,
                    )

        if completed_turn_index is not None:
            await turn.write_event(
                session_id,
                stream.session_waiting(turn_index=completed_turn_index),
            )

    raise RuntimeError("Session inbox closed without a terminal command")
