import asyncio
import collections.abc
import contextlib

import ai
import vercel.workflow

import agent.control as control
import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step(max_retries=0)
async def spawn_turn_workflow(
    turn_input: proto.TurnInput,
    parent_span: ai.experimental_telemetry.Span | None = None,
) -> str:
    if ai.experimental_telemetry.is_enabled():
        turn_span = ai.experimental_telemetry.create_span(
            "turn", parent=parent_span
        ).stamp_start()
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
async def next_stream_index(session_id: str) -> int:
    return await stream.tail_index(session_id) + 1


@workflow.step
async def forward_turn_command(
    session_id: str,
    turn_index: int,
    command: proto.TurnCommand,
) -> None:
    hook = proto.TurnInboxHook(command=command)
    token = proto.turn_inbox_token(session_id, turn_index)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


@workflow.step
async def forward_interrupt(session_id: str, turn_index: int) -> None:
    await control.interrupt(session_id, turn_index)
    hook = proto.TurnInboxHook(command=proto.InterruptTurn())
    token = proto.turn_inbox_token(session_id, turn_index)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt < 39:
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
    background_tasks = {
        task_id: (task.name, task.child_session_id)
        for task_id, task in state.background_tasks.items()
        if task.status == "running"
    }
    pending_user_messages: list[tuple[str, bool]] = []
    pending_background_updates: list[str] = []
    interrupting = False

    async for received in _buffer_inbox(inbox):
        command = received.command
        completed_turn_index: int | None = None

        match command:
            case proto.NewUserMessage():
                if command.close:
                    if active_turn_index is not None or background_tasks:
                        continue
                    await turn.write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    inbox.dispose()
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    )

                pending_user_messages.append((command.prompt or "", False))

            case proto.SubmitToolApproval():
                if active_turn_index is None:
                    continue
                await forward_turn_command(
                    session_id,
                    active_turn_index,
                    proto.TurnApproval(response=command.response),
                )

            case proto.InterruptSession():
                if interrupting:
                    continue
                interrupting = True
                pending_user_messages.clear()
                pending_background_updates.clear()
                targets = [
                    (child_session_id, 0)
                    for _, child_session_id in background_tasks.values()
                ]
                if active_turn_index is not None:
                    targets.append((session_id, active_turn_index))
                await asyncio.gather(
                    *(
                        forward_interrupt(target_session_id, target_turn_index)
                        for target_session_id, target_turn_index in targets
                    )
                )
                if not targets:
                    interrupting = False
                    state.active_ui_run_message_index = None
                    await save_session(state)
                    await turn.write_event(session_id, stream.session_interrupted())

            case proto.StartBackgroundTask():
                if interrupting or command.task_id in background_tasks:
                    continue
                child_session_id = f"{session_id}:child:{command.task_id}"
                background_tasks[command.task_id] = (command.name, child_session_id)
                state.background_tasks[command.task_id] = proto.BackgroundTaskState(
                    task_id=command.task_id,
                    child_session_id=child_session_id,
                    name=command.name,
                )
                await save_session(state)
                await turn.write_event(
                    session_id,
                    stream.subagent_called(
                        tool_call_id=command.task_id,
                        child_session_id=child_session_id,
                        name=command.name,
                    ),
                )
                child_stream_start_index = await next_stream_index(child_session_id)
                await spawn_turn_workflow(
                    proto.TurnInput(
                        session_id=child_session_id,
                        messages=[
                            ai.system_message(turn.SUBAGENT_SYSTEM_PROMPT),
                            ai.user_message(command.prompt),
                        ],
                        gated=False,
                        parent_session_id=session_id,
                        background_task_id=command.task_id,
                        stream_start_index=child_stream_start_index,
                    ),
                    command.parent_span,
                )

            case proto.BackgroundTaskFinished():
                task = background_tasks.pop(command.task_id, None)
                if task is None:
                    continue
                name, child_session_id = task
                is_error = command.output.kind == "error"
                is_interrupted = command.output.kind == "interrupted"
                error = command.output.error if is_error else None
                child_messages = [
                    message
                    for message in command.output.messages
                    if message.role in ("assistant", "tool")
                ]
                state.background_tasks[command.task_id] = proto.BackgroundTaskState(
                    task_id=command.task_id,
                    child_session_id=child_session_id,
                    name=name,
                    status=(
                        "interrupted"
                        if is_interrupted
                        else "failed"
                        if is_error
                        else "completed"
                    ),
                    messages=child_messages,
                    error=error,
                )
                await save_session(state)
                await turn.write_event(
                    session_id,
                    stream.subagent_completed(
                        tool_call_id=command.task_id,
                        is_error=is_error,
                        messages=[
                            message.model_dump(mode="json")
                            for message in child_messages
                        ],
                        error="Interrupted by user" if is_interrupted else error,
                    ),
                )
                await turn.close_stream(child_session_id)
                if not is_interrupted:
                    if is_error:
                        report = error or "Unknown subagent error"
                        pending_background_updates.append(
                            f'Background subagent "{name}" failed:\n\n{report}'
                        )
                    else:
                        pending_background_updates.append(
                            f'Background subagent "{name}" finished:\n\n'
                            f"{_last_text(command.output.messages)}"
                        )
                if not background_tasks and not interrupting:
                    pending_user_messages.extend(
                        (update, True) for update in pending_background_updates
                    )
                    pending_background_updates.clear()

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

        if interrupting and active_turn_index is None and not background_tasks:
            interrupting = False
            pending_background_updates.clear()
            state.active_ui_run_message_index = None
            await save_session(state)
            await turn.write_event(session_id, stream.session_interrupted())
            continue

        if active_turn_index is None and pending_user_messages:
            queued_messages = pending_user_messages.copy()
            pending_user_messages.clear()
            user_message = ai.user_message(
                "\n\n".join(text for text, _ in queued_messages)
            )
            state.messages.append(user_message)
            is_background_followup = all(
                background for _, background in queued_messages
            )
            if is_background_followup:
                state.hidden_ui_message_ids.add(user_message.id)
            else:
                # The user message is committed immediately; everything appended by
                # this turn and its background continuation is rebuilt by resume.
                state.active_ui_run_message_index = len(state.messages)
            await save_session(state)

            active_turn_index = next_turn_index
            next_turn_index += 1
            await turn.write_event(
                session_id,
                stream.turn_started(
                    turn_index=active_turn_index,
                    background=all(background for _, background in queued_messages),
                ),
            )
            turn_stream_start_index = await next_stream_index(session_id)
            await spawn_turn_workflow(
                proto.TurnInput(
                    session_id=session_id,
                    messages=state.messages,
                    turn_index=active_turn_index,
                    stream_start_index=turn_stream_start_index,
                )
            )
        elif completed_turn_index is not None:
            if not background_tasks:
                state.active_ui_run_message_index = None
                await save_session(state)
            await turn.write_event(
                session_id,
                stream.session_waiting(
                    turn_index=completed_turn_index,
                    active_background_tasks=len(background_tasks),
                ),
            )

    raise RuntimeError("Session inbox closed without a terminal command")
