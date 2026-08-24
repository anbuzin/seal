import ai
import vercel.workflow

import agent.proto as proto
import agent.session as session
import agent.stream as stream
import agent.turn as turn
from agent import workflow


@workflow.step(max_retries=0)
async def spawn_turn_workflow(turn_input: proto.TurnInput) -> str:
    # TODO: making retry for this safe requires cooperation on the workflow side
    # ts docs suggest using a hook and checking uniqueness!
    # fires child workflow for an agent turn
    if ai.experimental_telemetry.is_enabled():
        # mint the span for the turn and pass it in. this way
        # whatever is going on inside will be able to nest under it.
        turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        turn_input = turn_input.model_copy(update={"turn_span": turn_span})
    started = await vercel.workflow.start(turn.run_turn, turn_input)
    return started.run_id


@workflow.step
async def load_session(session_id: str) -> proto.SessionState | None:
    # restores the latest persisted session snapshot, if any
    return await session.read_session(session_id)


@workflow.step
async def save_session(state: proto.SessionState) -> None:
    # appends the current session state as the latest snapshot
    await session.write_session(state)


def _last_text(messages: list[ai.messages.Message]) -> str:
    for message in reversed(messages):
        if message.role == "assistant" and message.text:
            return message.text
    return ""


@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay.
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_session(session_input: proto.SessionInput) -> proto.SessionOutput:
    # prepare the session
    session_id = session_input.session_id

    state = await load_session(session_id)
    if state is not None:
        # resume a persisted session with the new user message appended.
        state.messages.append(ai.user_message(session_input.prompt))
    else:
        state = proto.SessionState(
            session_id=session_id,
            messages=[
                ai.system_message(turn.SYSTEM_PROMPT),
                ai.user_message(session_input.prompt),
            ],
        )
    await save_session(state)
    await turn.write_event(session_id, stream.session_started())

    turn_index = 0
    while True:
        # run turn workflow and suspend on a hook until it completes
        await turn.write_event(session_id, stream.turn_started(turn_index=turn_index))
        turn_hook_token = f"seal-turn:{session_id}:{turn_index}"
        turn_hook = proto.TurnHook.wait(token=turn_hook_token)
        turn_input = proto.TurnInput(
            session_id=session_id,
            messages=state.messages,
            turn_hook_token=turn_hook_token,
            turn_index=turn_index,
        )
        await spawn_turn_workflow(turn_input)
        turn_resolution = await turn_hook
        turn_hook.dispose()
        assert turn_resolution is not None
        turn_result = turn_resolution.output

        # process turn results
        state.messages = turn_result.messages
        await save_session(state)
        await turn.write_event(
            session_id,
            stream.turn_completed(turn_index=turn_index, kind=turn_result.kind),
        )

        match turn_result.kind:
            case "suspend":
                # we are currently in the main session. wait for the next user message.
                await turn.write_event(
                    session_id, stream.session_waiting(turn_index=turn_index)
                )
                hook = proto.SessionHook.wait(
                    token=f"seal-session:{session_id}:{turn_index}"
                )
                resolution = await hook
                hook.dispose()
                message = resolution.payload if resolution is not None else None

                if not isinstance(message, proto.NewUserMessage) or message.close:
                    await turn.write_event(session_id, stream.session_completed())
                    await turn.close_stream(session_id)
                    return proto.SessionOutput(
                        session_id=session_id,
                        output=_last_text(state.messages),
                    )

                state.messages.append(ai.user_message(message.prompt or ""))

            case "error":
                await turn.write_event(
                    session_id, stream.session_completed(is_error=True)
                )
                await turn.close_stream(session_id)
                return proto.SessionOutput(
                    session_id=session_id,
                    output=turn_result.error or _last_text(state.messages),
                    is_error=True,
                )

        # persist post-turn mutations (resume prompt / subagent results) so the
        # next turn resumes from the latest state after a crash.
        await save_session(state)
        turn_index += 1
