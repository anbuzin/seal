import asyncio
import contextvars
import dataclasses
import traceback
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar

import ai
import pydantic
import vercel.workflow

from agent import proto, session, stream, util, workflow

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
IMAGE_MODEL_ID = "gateway:google/gemini-3.1-flash-image"
SYSTEM_PROMPT = (
    "You are Seal, a coding assistant. Use bash, web_fetch, and subagent to "
    "inspect the environment, gather information, and delegate focused work. "
    "Use generate_image to create images."
)
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Use bash, web_fetch, and generate_image "
    "when useful, then answer the delegated task directly."
)
IMAGE_SYSTEM_PROMPT = (
    "You are an image generator. Generate an image for the user's prompt."
)


class EagerToolHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: ai.messages.ToolCallPart


@workflow.step
async def llm_step(
    model_id: str,
    messages_data: list[dict[str, object]],
    tools_data: list[dict[str, object]],
    session_id: str | None,
    tool_token: str | None = None,
    turn_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    model = ai.get_model(model_id)
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]
    tools = [ai.Tool.model_validate(tool) for tool in tools_data]

    writer = await stream.get_writable(session_id) if session_id else None
    metadata = vercel.workflow.get_step_metadata()

    # On a retry, emit a message requesting a reload. The will trigger
    # the client to drop everything from the last step.
    if writer is not None and metadata.attempt > 1:
        await writer.write(stream.reload_requested())

    # parent this step's spans under the turn's span
    turn_span = (
        ai.experimental_telemetry.Span.model_validate(turn_span_data)
        if turn_span_data
        else None
    )
    async with (
        ai.experimental_telemetry.use_span(turn_span),
        ai.stream(model, messages, tools=tools) as model_stream,
    ):
        async for e in model_stream:
            if e.replay:
                continue

            if writer is not None:
                await writer.write(e)
            if tool_token and isinstance(e, ai.types.events.ToolEnd):
                await EagerToolHook(payload=e.tool_call).resume(tool_token)

    return model_stream.message.model_dump(mode="json")


@workflow.step
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream
    session_id: str,
    event_data: dict[str, object],
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event_data)


# closes a durable event stream once the owning session is terminal.
@workflow.step
async def close_stream(session_id: str) -> None:
    writer = await stream.get_writable(session_id)
    await writer.close()


@ai.tool(require_approval=True)
@workflow.step(max_retries=0)
async def bash(command: str, timeout: int | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.communicate()
        return f"Command timed out after {timeout}s."

    output = stdout.decode() if stdout else ""
    if proc.returncode != 0:
        return f"[exit code {proc.returncode}]\n{output}"
    return output


# subagent (task) sessions cannot surface tool approvals to a human and would
# deadlock on a gated tool, so they run an ungated copy of the same tool.
bash_ungated = dataclasses.replace(
    bash, tool=bash.tool.model_copy(update={"require_approval": False})
)


@ai.tool
@workflow.step
async def web_fetch(
    url: str,
    method: str = "GET",
    headers: str = "",
    body: str = "",
) -> str:
    import httpx

    parsed_headers: dict[str, str] = {}
    for line in headers.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            parsed_headers[key.strip()] = value.strip()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=parsed_headers or None,
            content=body or None,
        )

    parts = [
        f"HTTP {response.status_code}",
        *(f"{key}: {value}" for key, value in response.headers.items()),
        "",
        response.text[:50_000],
    ]
    return "\n".join(parts)


@workflow.step
async def image_step(prompt: str) -> dict[str, object]:
    """Generate an image from a text prompt. Describe the desired image in
    detail, including subject, style, and composition."""

    # the ai library has no direct image-generation API yet, so this
    # runs a model that emits images inline with its response
    # (FileParts on the message).
    model = ai.get_model(IMAGE_MODEL_ID)
    messages = [ai.system_message(IMAGE_SYSTEM_PROMPT), ai.user_message(prompt)]
    async with ai.stream(model, messages) as model_stream:
        async for _ in model_stream:
            pass
    message = model_stream.message

    if not message.images:
        return ai.content_output(
            message.text or "The image model returned no image.",
        ).model_dump(mode="json")
    # keep any caption text the model emitted alongside its images
    return ai.content_output(
        *(
            part
            for part in message.parts
            if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
        )
    ).model_dump(mode="json")


@ai.tool
async def generate_image(prompt: str) -> ai.messages.ContentOutput:
    # TODO: annoyingly we have to have a model_validate in a tool outside the step
    return ai.messages.ContentOutput.model_validate(await image_step(prompt))


@workflow.step(max_retries=0)
async def spawn_subagent_turn(
    turn_input: dict[str, object],
    parent_span_data: dict[str, object] | None = None,
) -> dict[str, object]:
    # a subagent is just one ungated turn writing to its own stream. its span
    payload = dict(turn_input)
    if ai.experimental_telemetry.is_enabled():
        # create and nest the span for the subagent turn
        parent = (
            ai.experimental_telemetry.Span.model_validate(parent_span_data)
            if parent_span_data
            else None
        )
        turn_span = ai.experimental_telemetry.create_span(
            "turn", parent=parent
        ).stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        payload["turn_span"] = turn_span.model_dump(mode="json")
    started = await vercel.workflow.start(run_turn, payload)
    return {"run_id": started.run_id}


# the running tool call's context, set by the loop around each schedule so a
# tool can reach it without smuggling args. tasks copy the contextvars at
# creation, so each tool sees its own call.
tool_call_context: contextvars.ContextVar[proto.ToolCallContext] = (
    contextvars.ContextVar("tool_call_context")
)


# hack: the only way the library currently supports transforming a
# tool result before sending it to the model is by using an
# aggregator, so we use MessageAggregator without actually being a
# generator.
@ai.tool(aggregator=ai.agents.MessageAggregator)  # type: ignore
@util.print_traceback
async def subagent(prompt: str, name: str | None = None) -> ai.agents.MessageBundle:
    """Delegate a focused task to a child agent and return its answer."""
    call = tool_call_context.get()
    session_id, tool_call_id = call.session_id, call.tool_call_id
    name = name or "subagent"
    child_session_id = f"{session_id}:child:{tool_call_id}"
    token = f"seal-subagent:{child_session_id}:0"
    await write_event(
        session_id,
        stream.subagent_called(
            tool_call_id=tool_call_id, child_session_id=child_session_id, name=name
        ),
    )
    hook = proto.SubagentHook.wait(token=token)
    await spawn_subagent_turn(
        proto.TurnInput(
            session_id=child_session_id,
            messages=[
                ai.system_message(SUBAGENT_SYSTEM_PROMPT),
                ai.user_message(prompt),
            ],
            is_subagent=True,
            parent_hook_token=token,
        ).model_dump(mode="json"),
        # the child turn's root span nests under this turn's root span.
        call.turn_span.model_dump(mode="json") if call.turn_span else None,
    )
    resolution = await hook
    hook.dispose()
    assert resolution is not None
    output = resolution.output
    await write_event(
        session_id,
        stream.subagent_completed(
            tool_call_id=tool_call_id, is_error=output.kind == "error"
        ),
    )
    await close_stream(child_session_id)
    return ai.agents.MessageBundle(
        messages=tuple(m for m in output.messages if m.role in ("assistant", "tool"))
    )


# Tools that we can run eagerly, before the llm call generating them
# has completed. These should be non-effectful (because they might get
# cancelled) and non-streaming (because that would take some extra
# thought).
EAGER_TOOLS = {"generate_image", "web_fetch"}


class DurableAgent(ai.Agent):
    # bash is gated/ungated per mode, so it is supplied via tools=, not here.
    TOOLS: ClassVar[list[ai.AgentTool]] = [web_fetch, generate_image]

    def __init__(
        self,
        *,
        tg: asyncio.TaskGroup,
        tools: Sequence[ai.AgentTool | ai.Tool] | None = None,
        session_id: str | None = None,
        turn_span: ai.experimental_telemetry.Span | None = None,
    ) -> None:
        super().__init__(tools=tools)
        self.tg = tg
        self.session_id = session_id
        self.turn_span = turn_span

        # eager tool dispatch bits
        self.watcher_task: asyncio.Task[None] | None = None
        self.live_tool_calls: dict[str, asyncio.Task[Any]] = {}

    def cancel_leftovers(self) -> None:
        """Cancel eager tool-related tasks; no-op after a clean run."""
        if self.watcher_task is not None:
            self.watcher_task.cancel()
        for task in self.live_tool_calls.values():
            task.cancel()

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        model_id = context.model.id
        session_id = self.session_id
        turn_span_data = (
            self.turn_span.model_dump(mode="json") if self.turn_span else None
        )

        tool_token = f"seal-early-tool:{session_id}"

        def launch_tool(tool_call: ai.messages.ToolCallPart) -> None:
            # Launch a tool in a task under the right context, track
            # it in the live call table.
            token = tool_call_context.set(
                proto.ToolCallContext(
                    session_id=session_id or "",
                    tool_call_id=tool_call.tool_call_id,
                    turn_span=self.turn_span,
                )
            )
            self.live_tool_calls[tool_call.tool_call_id] = self.tg.create_task(
                context.resolve(tool_call)()
            )
            tool_call_context.reset(token)

        eager_tool_hook = EagerToolHook.wait(token=tool_token)

        async def watcher() -> None:
            # Wait on our eager tool hook. For EAGER_TOOLS, trigger
            # them now, from the watcher thread.
            #
            # Once llm_step returns, the tool runner will schedule a
            # ToolRunner task that waits on them.
            async for ev in eager_tool_hook:
                tool_call = ev.payload
                if tool_call.tool_name in EAGER_TOOLS:
                    launch_tool(tool_call)

        self.watcher_task = self.tg.create_task(watcher())

        while context.keep_running():
            self.live_tool_calls.clear()

            result = await llm_step(
                model_id,
                [message.model_dump(mode="json") for message in context.messages],
                [tool.model_dump(mode="json") for tool in context.tools],
                session_id,
                tool_token,
                turn_span_data,
            )

            assistant_message = ai.messages.Message.model_validate(result)
            context.add(assistant_message)
            # llm_step streamed this turn out-of-band (straight to the durable
            # stream), so yield the final StreamEnd here for run-blocked
            # tracking, which counts the turn's tool calls from it.
            yield ai.events.StreamEnd(message=assistant_message)

            async with ai.ToolRunner() as runner:
                # Cancel eager tool calls that are not legit -- that
                # is, ones that are from a retried llm call. They
                # won't actually get stopped if they are steps, unless
                # the cancellation happens before the step was
                # launched, but it will stop us from waiting on them.
                legit_call_ids = {
                    tc.tool_call_id for tc in assistant_message.tool_calls
                }
                for id, task in list(self.live_tool_calls.items()):
                    if id not in legit_call_ids:
                        task.cancel()
                        del self.live_tool_calls[id]

                for tool_call in assistant_message.tool_calls:
                    # Launch the tool if it isn't running already
                    if tool_call.tool_call_id not in self.live_tool_calls:
                        launch_tool(tool_call)

                    # Wait on it
                    async def _wait(tc: ai.messages.ToolCallPart = tool_call) -> Any:
                        return await self.live_tool_calls[tc.tool_call_id]

                    runner.schedule(_wait)

                async for event in runner.events():
                    # write tool-running events from the producer side so they land
                    # in loop order (results before the next turn's answer); run_turn
                    # only writes HookEvents, which ride the runtime queue instead.
                    if session_id is not None:
                        await write_event(session_id, event.model_dump(mode="json"))
                    yield event

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                # HACK: TODO(sully)
                # the library computes aggregator model_input only for
                # generator tools (and backfills only at run start), so a
                # MessageBundle from the subagent hack would reach the next
                # llm_step raw and fail JSON encoding. backfill it here.
                for part in tool_message.tool_results:
                    if (
                        isinstance(part.result, ai.agents.MessageBundle)
                        and not part.has_model_input
                        and not part.is_error
                    ):
                        part.set_model_input(
                            ai.agents.MessageAggregator.to_model_input(part.result)
                        )
                context.add(tool_message)

        # live tool tasks are all done by now, so this only stops the watcher.
        self.cancel_leftovers()
        eager_tool_hook.dispose()


@workflow.step
async def ship_spans(spans_data: list[dict[str, Any]]) -> None:
    # re-deliver spans collected in the workflow body to the real adapters.
    await ai.experimental_telemetry.push_all(spans_data)


@workflow.step
async def notify_parent(token: str, output_data: dict[str, Any]) -> None:
    # when subagent got done working, notify the parent that is suspended on
    # a hook
    hook = proto.SubagentHook(output=proto.TurnOutput.model_validate(output_data))
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


@workflow.step
async def save_session(state_data: dict[str, Any]) -> None:
    # appends the current session state as the latest snapshot
    await session.write_session(proto.SessionState.model_validate(state_data))


@workflow.step
async def notify_agent_finished(token: str) -> None:
    # when the agent task finishes, post a signal to turn's inbox
    await proto.InboxHook(command=proto.AgentFinished()).resume(token)


async def build_turn_input(session_id: str, prompt: str) -> dict[str, Any]:
    """Assemble the next root turn's input from the latest session snapshot."""
    # called by the server before starting a turn
    # maybe bad: read-then-start without a lock
    state = await session.read_session(session_id)
    if state is None:
        messages = [ai.system_message(SYSTEM_PROMPT), ai.user_message(prompt)]
        turn_index = 0
    else:
        messages = [*state.messages, ai.user_message(prompt)]
        turn_index = state.turn_index + 1
    payload = proto.TurnInput(
        session_id=session_id, messages=messages, turn_index=turn_index
    ).model_dump(mode="json")

    # mint the turn's root span
    if ai.experimental_telemetry.is_enabled():
        turn_span = ai.experimental_telemetry.create_span("turn").stamp_start()
        turn_span.set_attrs({"openinference.span.kind": "AGENT"})
        payload["turn_span"] = turn_span.model_dump(mode="json")
    return payload


# run one agent turn
# has an inbox hook for commands such as Approval and AgentFinished, and in the
# future also steering, cancellation, and others.
# the workflow dispatches work in a task and repeatedly suspends on the inbox
# hook until it receives an AgentFinished.
@workflow.workflow
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_turn(turn_input: dict[str, Any]) -> dict[str, Any]:
    _turn_input = proto.TurnInput.model_validate(turn_input)
    messages = _turn_input.messages
    session_id = _turn_input.session_id
    turn_index = _turn_input.turn_index
    # root turns own the session snapshot + lifecycle stream; subagent turns
    # only write agent events to their own stream.
    root = not _turn_input.is_subagent

    if root:
        # commit the user message before the model runs, so a crashed turn
        # resumes from a snapshot that already has it.
        await save_session(
            proto.SessionState(
                session_id=session_id, messages=messages, turn_index=turn_index
            ).model_dump(mode="json")
        )
        if turn_index == 0:
            await write_event(session_id, stream.session_started())
        await write_event(session_id, stream.turn_started(turn_index=turn_index))

    extra_tools = [bash_ungated] if _turn_input.is_subagent else [bash, subagent]

    # collect spans that happen inside the workflow body, and send them
    # once in a separate step.
    span_sink = (
        ai.experimental_telemetry.DictSink()
        if _turn_input.turn_span is not None
        else None
    )

    inbox = proto.InboxHook.wait(token=proto.inbox_token(session_id))
    output: proto.TurnOutput | None = None

    try:
        model = ai.get_model(MODEL_ID)
        tg = ai.util.TaskGroup()
        agent = DurableAgent(
            tg=tg,
            tools=extra_tools,
            session_id=session_id,
            turn_span=_turn_input.turn_span,
        )

        async with (
            ai.experimental_telemetry.use_sink(span_sink),
            ai.experimental_telemetry.use_span(_turn_input.turn_span),
            agent.run(model, messages) as run,
            tg,
        ):

            async def drive_agent_run() -> None:
                # iterate the agent run in a task, call notify_agent_finished
                # to send AgentFinished signal to the inbox when done.
                nonlocal output
                try:
                    async for event in run:
                        if (
                            isinstance(event, ai.events.HookEvent)
                            and event.hook.status == "pending"
                            and event.hook.hook_type
                            == ai.agents.TOOL_APPROVAL_HOOK_TYPE
                            and event.hook.tool_call_id is not None
                        ):
                            # report hook event to the client
                            await write_event(session_id, event.model_dump(mode="json"))
                        elif isinstance(event, ai.events.RunBlocked):
                            # the run is blocked on approvals; tell the client
                            # we're waiting on a human.
                            await write_event(
                                session_id,
                                stream.tool_approval_requested(turn_index=turn_index),
                            )
                    output = proto.TurnOutput(kind="suspend", messages=run.messages)
                except Exception as error:
                    output = proto.TurnOutput(
                        kind="error",
                        messages=messages,
                        error=f"{type(error).__name__}: {error}",
                    )
                    print(
                        f"[seal] error in run_turn:\n{traceback.format_exc()}",
                        flush=True,
                    )
                finally:
                    # close the stream in this task, on every exit
                    await run.aclose()

                await notify_agent_finished(proto.inbox_token(session_id))

            # dispatch agent.run iteration into a task
            tg.create_task(drive_agent_run())

            async for received in inbox:
                command = received.command
                if isinstance(command, proto.Approval):
                    response = command.response
                    # hack: using derived labels
                    ai.resolve_hook(
                        f"{proto.TOOL_APPROVAL_HOOK_PREFIX}{response.tool_call_id}",
                        {"granted": response.granted, "reason": response.reason},
                    )
                elif isinstance(command, proto.AgentFinished):
                    # stop reading the inbox and wrap up the run
                    break

            # clean up eager tool-related machinery
            agent.cancel_leftovers()
    except Exception as error:
        # package and report the error instead of crashing
        print(f"[seal] error in run_turn:\n{traceback.format_exc()}", flush=True)
        if output is None or output.kind != "error":
            output = proto.TurnOutput(
                kind="error",
                messages=messages,
                error=f"{type(error).__name__}: {error}",
            )

    inbox.dispose()
    assert output is not None

    # send off telemetry spans collected inside the workflow body
    if span_sink is not None:
        finished = [s.model_dump(mode="json") for s in span_sink.finished_spans]
        if _turn_input.turn_span is not None:
            # complete the turn span here (pure data ops on workflow time) so
            # it ships with the rest instead of riding the resume step.
            turn_span = _turn_input.turn_span.stamp_end(
                error=ai.experimental_telemetry.SpanError(
                    type="TurnError", message=output.error
                )
                if output.kind == "error" and output.error
                else None
            )
            turn_span.set_attrs({"session.id": session_id, "turn_index": turn_index})
            finished.append(turn_span.model_dump(mode="json"))
        if finished:
            await ship_spans(finished)

    # if this is a subagent, notify the parent
    if _turn_input.is_subagent:
        if _turn_input.parent_hook_token is None:
            # soft-assert: without the token the parent hangs on its hook
            print(
                f"[seal] subagent turn {session_id} has no parent_hook_token; "
                "the parent will not be notified",
                flush=True,
            )
        else:
            await notify_parent(
                _turn_input.parent_hook_token, output.model_dump(mode="json")
            )

    if root:
        # settle the session: commit the turn's messages, then write the
        # boundary events the UI stream and the next POST /chat key off.
        await save_session(
            proto.SessionState(
                session_id=session_id, messages=output.messages, turn_index=turn_index
            ).model_dump(mode="json")
        )
        await write_event(
            session_id,
            stream.turn_completed(turn_index=turn_index, kind=output.kind),
        )
        if output.kind == "suspend":
            await write_event(session_id, stream.session_waiting(turn_index=turn_index))
        else:
            await write_event(session_id, stream.session_completed(is_error=True))
            await close_stream(session_id)

    return output.model_dump(mode="json")
