import asyncio
import collections.abc
import contextlib
from typing import Any

import ai
import vercel.workflow

import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step
async def write_event(
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


@workflow.step(max_retries=0)
async def spawn_turn_workflow(turn_input: dict[str, object]) -> dict[str, object]:
    payload = dict(turn_input)
    if ai.experimental_telemetry.is_enabled():
        turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        payload["turn_span"] = turn_span.model_dump(mode="json")
    started = await vercel.workflow.start(turn.run_turn, payload)
    return {"run_id": started.run_id}


@workflow.step
async def load_session(session_id: str) -> dict[str, Any] | None:
    state = await session.read_session(session_id)
    return state.model_dump(mode="json") if state is not None else None


@workflow.step
async def save_session(state_data: dict[str, Any]) -> None:
    await session.write_session(proto.SessionState.model_validate(state_data))


@workflow.step
async def forward_tool_approval(
    session_id: str,
    turn_index: int,
    response_data: dict[str, Any],
) -> None:
    hook = proto.TurnInboxHook(
        command=proto.TurnApproval(
            response=proto.ToolApprovalResponse.model_validate(response_data)
        )
    )
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
async def run_session(session_input: dict[str, Any]) -> dict[str, Any]:
    _session_input = proto.SessionInput.model_validate(session_input)
    session_id = _session_input.session_id

    restored = await load_session(session_id)
    state = (
        proto.SessionState.model_validate(restored)
        if restored is not None
        else proto.SessionState(
            session_id=session_id,
            messages=[ai.system_message(turn.SYSTEM_PROMPT)],
        )
    )
    await save_session(state.model_dump(mode="json"))
    await write_event(session_id, stream.session_started())

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
                    await write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    inbox.dispose()
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    ).model_dump(mode="json")

                state.messages.append(ai.user_message(command.prompt or ""))
                await save_session(state.model_dump(mode="json"))

                active_turn_index = next_turn_index
                next_turn_index += 1
                await write_event(
                    session_id,
                    stream.turn_started(turn_index=active_turn_index),
                )
                await spawn_turn_workflow(
                    proto.TurnInput(
                        session_id=session_id,
                        messages=state.messages,
                        turn_index=active_turn_index,
                    ).model_dump(mode="json")
                )

            case proto.SubmitToolApproval():
                if active_turn_index is None:
                    continue
                await forward_tool_approval(
                    session_id,
                    active_turn_index,
                    command.response.model_dump(mode="json"),
                )

            case proto.TurnFinished():
                if command.turn_index != active_turn_index:
                    continue

                turn_result = command.output
                state.messages = turn_result.messages
                await save_session(state.model_dump(mode="json"))
                await write_event(
                    session_id,
                    stream.turn_completed(
                        turn_index=command.turn_index, kind=turn_result.kind
                    ),
                )
                completed_turn_index = command.turn_index
                active_turn_index = None

                if turn_result.kind == "error":
                    await write_event(
                        session_id, stream.session_completed(is_error=True)
                    )
                    await turn.close_stream(session_id)
                    inbox.dispose()
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=turn_result.error or _last_text(state.messages),
                        is_error=True,
                    ).model_dump(mode="json")

        if completed_turn_index is not None:
            await write_event(
                session_id,
                stream.session_waiting(turn_index=completed_turn_index),
            )

    raise RuntimeError("Session inbox closed without a terminal command")
