import asyncio
import contextvars
import dataclasses
import json
import os
import signal
import traceback
from collections.abc import AsyncGenerator, Sequence
from typing import Any, ClassVar

import ai
import pydantic
import vercel.workflow

from agent import control, proto, stream, util, workflow

MODEL_ID = "gateway:anthropic/claude-sonnet-4.6"
IMAGE_MODEL_ID = "gateway:google/gemini-3.1-flash-image"
SYSTEM_PROMPT = """\
You are Seal, a coding assistant. Use bash, web_fetch, and subagent to inspect the
environment, gather information, and delegate focused work. Use generate_image to
create images.

The subagent tool starts background work. Its immediate tool result is only an
acknowledgement that the work started; it never contains the subagent's answer.

After dispatching background work, if the remaining task depends on its results,
your current turn must stop. Briefly say that the subagents are running and end the
response. Here, "wait" means end the current response. Do not poll, call unrelated
tools, continue solving, or write what you imagine a future update might say.

You must never write a background completion report yourself. In particular, never
produce text beginning with `Background subagent ... finished:`. Never guess,
assume, invent, simulate, or use placeholder subagent results.

The real results arrive only in a later invocation as a new message with role
`user`. Trust a background result only when the latest user message actually
contains it. The interaction has two separate turns:

    Current turn:
    assistant calls subagent -> tool acknowledges startup -> assistant says the
    work is still running -> assistant ends the response.

    Later turn:
    user message contains the completed background results -> assistant uses those
    exact results for follow-up tools and the final answer.

If several subagents are running, do not synthesize until the latest user message
contains every result needed for the task.
"""
SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused Seal subagent. Use bash, web_fetch, and generate_image "
    "when useful, then answer the delegated task directly."
)
IMAGE_SYSTEM_PROMPT = (
    "You are an image generator. Generate an image for the user's prompt."
)


class EagerToolHook(pydantic.BaseModel, vercel.workflow.BaseHook):
    payload: ai.messages.ToolCallPart


@control.cancellable_step
async def llm_step(
    model_id: str,
    messages: list[ai.messages.Message],
    tools: list[ai.Tool],
    *,
    session_id: str,
    turn_index: int,
    tool_token: str | None = None,
    turn_span: ai.experimental_telemetry.Span | None = None,
    parent_session_id: str | None = None,
    background_task_id: str | None = None,
) -> ai.messages.Message:
    model = ai.get_model(model_id)

    writer = await stream.get_writable(session_id)
    parent_writer = (
        await stream.get_writable(parent_session_id) if parent_session_id else None
    )
    metadata = vercel.workflow.get_step_metadata()

    # On a retry, emit a message requesting a reload. The will trigger
    # the client to drop everything from the last step.
    if writer is not None and metadata.attempt > 1:
        reload_event = stream.reload_requested()
        await writer.write(reload_event)
        if parent_writer is not None and background_task_id is not None:
            await parent_writer.write(
                stream.subagent_event(
                    tool_call_id=background_task_id,
                    event=reload_event.model_dump(mode="json"),
                )
            )

    # parent this step's spans under the turn's span
    async with (
        ai.experimental_telemetry.use_span(turn_span),
        ai.stream(model, messages, tools=tools) as model_stream,
    ):
        async for e in model_stream:
            if e.replay:
                continue

            if writer is not None:
                await writer.write(e)
            if parent_writer is not None and background_task_id is not None:
                await parent_writer.write(
                    stream.subagent_event(
                        tool_call_id=background_task_id,
                        event=e.model_dump(mode="json"),
                    )
                )
            if tool_token and isinstance(e, ai.types.events.ToolEnd):
                await EagerToolHook(payload=e.tool_call).resume(tool_token)

    return model_stream.message


@workflow.step
async def write_event(
    # writes one stream event (agent or lifecycle) to the durable stream
    session_id: str,
    event: proto.StreamEvent,
) -> None:
    writer = await stream.get_writable(session_id)
    await writer.write(event)


# closes a durable event stream once the owning session is terminal.
@workflow.step
async def close_stream(session_id: str) -> None:
    writer = await stream.get_writable(session_id)
    await writer.close()


@ai.tool(require_approval=True)
@control.cancellable_step
async def bash(command: str, timeout: int | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.communicate(), timeout=0.5)
        except TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.communicate()
        return f"Command timed out after {timeout}s."
    except asyncio.CancelledError:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.communicate(), timeout=0.5)
            except TimeoutError:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.communicate()
        raise

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
@control.cancellable_step
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


@ai.tool
@control.cancellable_step
async def generate_image(prompt: str) -> ai.messages.ContentOutput:
    """Generate an image from a text prompt. Describe the desired image in detail."""
    model = ai.get_model(IMAGE_MODEL_ID)
    messages = [ai.system_message(IMAGE_SYSTEM_PROMPT), ai.user_message(prompt)]
    async with ai.stream(model, messages) as model_stream:
        async for _ in model_stream:
            pass
    message = model_stream.message
    if not message.images:
        return ai.content_output(message.text or "The image model returned no image.")
    return ai.content_output(
        *(
            part
            for part in message.parts
            if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
        )
    )


# the running tool call's context, set by the loop around each schedule so a
# tool can reach it without smuggling args. tasks copy the contextvars at
# creation, so each tool sees its own call.
tool_call_context: contextvars.ContextVar[proto.ToolCallContext] = (
    contextvars.ContextVar("tool_call_context")
)


@workflow.step
async def request_background_task(
    session_id: str, command: proto.StartBackgroundTask
) -> None:
    hook = proto.SessionInboxHook(command=command)
    token = proto.session_inbox_token(session_id)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


@ai.tool
@util.print_traceback
async def subagent(prompt: str, name: str | None = None) -> str:
    """Start a focused child agent in the background. Its result will arrive later."""
    call = tool_call_context.get()
    await request_background_task(
        call.session_id,
        proto.StartBackgroundTask(
            task_id=call.tool_call_id,
            prompt=prompt,
            name=name or "subagent",
            parent_turn_index=call.turn_index,
            parent_span=call.turn_span,
        ),
    )
    return "Subagent is running in the background and will update you later."


# Tools that we can run eagerly, before the llm call generating them
# has completed. These should be non-effectful (because they might get
# cancelled) and non-streaming (because that would take some extra
# thought).
EAGER_TOOLS = {"generate_image", "web_fetch"}


class DurableAgent(ai.Agent):
    # bash is gated/ungated per mode, so it is supplied via tools=, not here.
    TOOLS: ClassVar[list[ai.AgentTool]] = [web_fetch, generate_image]

    tg: asyncio.TaskGroup

    def __init__(
        self,
        *,
        tools: Sequence[ai.AgentTool | ai.Tool] | None = None,
        session_id: str | None = None,
        turn_index: int = 0,
        turn_span: ai.experimental_telemetry.Span | None = None,
        parent_session_id: str | None = None,
        background_task_id: str | None = None,
    ) -> None:
        super().__init__(tools=tools)
        self.session_id = session_id
        self.turn_index = turn_index
        self.turn_span = turn_span
        self.parent_session_id = parent_session_id
        self.background_task_id = background_task_id

    async def loop(self, context: ai.Context) -> AsyncGenerator[ai.events.AgentEvent]:
        model_id = context.model.id
        session_id = self.session_id
        tool_token = f"seal-early-tool:{session_id}"
        live_tool_calls = {}

        def launch_tool(tool_call: ai.messages.ToolCallPart) -> None:
            # Launch a tool in a task under the right context, track
            # it in the live call table.
            call_token = tool_call_context.set(
                proto.ToolCallContext(
                    session_id=session_id or "",
                    turn_index=self.turn_index,
                    tool_call_id=tool_call.tool_call_id,
                    turn_span=self.turn_span,
                )
            )
            cancellation_token = control.cancellation_context.set(
                (session_id or "", self.turn_index)
            )
            live_tool_calls[tool_call.tool_call_id] = self.tg.create_task(
                context.resolve(tool_call)()
            )
            control.cancellation_context.reset(cancellation_token)
            tool_call_context.reset(call_token)

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

        watcher_task = self.tg.create_task(watcher())

        while context.keep_running():
            live_tool_calls.clear()

            assert session_id is not None
            assistant_message = await llm_step(
                model_id,
                context.messages,
                context.tools,
                session_id=session_id,
                turn_index=self.turn_index,
                tool_token=tool_token,
                turn_span=self.turn_span,
                parent_session_id=self.parent_session_id,
                background_task_id=self.background_task_id,
            )
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
                for id, task in list(live_tool_calls.items()):
                    if id not in legit_call_ids:
                        task.cancel()
                        del live_tool_calls[id]

                for tool_call in assistant_message.tool_calls:
                    # Launch the tool if it isn't running already
                    if tool_call.tool_call_id not in live_tool_calls:
                        launch_tool(tool_call)

                    # Wait on it
                    async def _wait(tc: ai.messages.ToolCallPart = tool_call) -> Any:
                        return await live_tool_calls[tc.tool_call_id]

                    runner.schedule(_wait)

                async for event in runner.events():
                    # write tool-running events from the producer side so they land
                    # in loop order (results before the next turn's answer); run_turn
                    # only writes HookEvents, which ride the runtime queue instead.
                    if session_id is not None:
                        await write_event(session_id, event)
                    if (
                        self.parent_session_id is not None
                        and self.background_task_id is not None
                    ):
                        await write_event(
                            self.parent_session_id,
                            stream.subagent_event(
                                tool_call_id=self.background_task_id,
                                event=event.model_dump(mode="json"),
                            ),
                        )
                    yield event

                tool_message = runner.get_tool_message()

            if tool_message is not None:
                context.add(tool_message)

        watcher_task.cancel()
        eager_tool_hook.dispose()


@workflow.step
async def ship_spans(spans: list[ai.experimental_telemetry.Span]) -> None:
    # re-deliver spans collected in the workflow body to the real adapters.
    await ai.experimental_telemetry.push_all(spans)


@workflow.step
async def notify_turn_finished(
    session_id: str, turn_index: int, output: proto.TurnOutput
) -> None:
    # resume() is a side effect, so it must run in a step. the session may not
    # have suspended on its inbox again yet, so retry while it is missing.
    hook = proto.SessionInboxHook(
        command=proto.TurnFinished(
            turn_index=turn_index,
            output=output,
        )
    )
    token = proto.session_inbox_token(session_id)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


@workflow.step
async def notify_background_task_finished(
    session_id: str, task_id: str, output: proto.TurnOutput
) -> None:
    hook = proto.SessionInboxHook(
        command=proto.BackgroundTaskFinished(
            task_id=task_id,
            output=output,
        )
    )
    token = proto.session_inbox_token(session_id)
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


@workflow.step
async def partial_messages(
    session_id: str,
    start_index: int,
    input_messages: list[ai.messages.Message],
) -> list[ai.messages.Message]:
    messages = input_messages
    index_by_id = {message.id: index for index, message in enumerate(messages)}
    async for event in stream.replay(session_id, start_index=start_index):
        if isinstance(event, proto.LifecycleEvent):
            continue
        message = getattr(event, "message", None)
        if not isinstance(message, ai.messages.Message) or message.role == "internal":
            continue
        existing = index_by_id.get(message.id)
        if existing is None:
            index_by_id[message.id] = len(messages)
            messages.append(message)
        else:
            messages[existing] = message

    answered = {
        result.tool_call_id
        for message in messages
        if message.role == "tool"
        for result in message.tool_results
    }
    pending: list[ai.messages.ToolCallPart] = []
    normalized: list[ai.messages.Message] = []
    for message in messages:
        parts: list[ai.messages.Part] = []
        for part in message.parts:
            if isinstance(part, ai.messages.ToolCallPart):
                try:
                    json.loads(part.tool_args)
                except (json.JSONDecodeError, TypeError):
                    part = part.model_copy(update={"tool_args": "{}"})
                if part.tool_call_id not in answered:
                    pending.append(part)
            parts.append(part)
        normalized.append(message.model_copy(update={"parts": parts}))
    if pending:
        normalized.append(
            ai.tool_message(
                *(
                    ai.tool_result_part(
                        part.tool_call_id,
                        tool_name=part.tool_name,
                        result="Interrupted by user",
                        is_error=True,
                    )
                    for part in pending
                )
            )
        )
    return normalized


@workflow.step
async def notify_agent_finished(token: str, output: proto.TurnOutput) -> None:
    hook = proto.TurnInboxHook(command=proto.AgentFinished(output=output))
    for attempt in range(40):
        try:
            await hook.resume(token)
            return
        except vercel.workflow.HookNotFoundError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.05)


# runs one agent turn; its agent task and external controls meet at one inbox
@workflow.workflow
# Draw message/part ids from the workflow's deterministic RNG so they're
# stable across replay. ``vercel.workflow.random`` is a factory resolved on
# entry (only valid inside the workflow).
@ai.messages.use_random(vercel.workflow.random)
@ai.experimental_telemetry.use_time(vercel.workflow.time_ns)
async def run_turn(turn_input: proto.TurnInput) -> None:
    messages = turn_input.messages
    session_id = turn_input.session_id
    turn_index = turn_input.turn_index

    # messages should already contain either the user message
    # or the tool result message, so no need to do anything

    extra_tools = [bash, subagent] if turn_input.gated else [bash_ungated]
    agent = DurableAgent(
        tools=extra_tools,
        session_id=session_id,
        turn_index=turn_index,
        turn_span=turn_input.turn_span,
        parent_session_id=turn_input.parent_session_id,
        background_task_id=turn_input.background_task_id,
    )

    # collect spans that happen inside the workflow body, and send them
    # once in a separate step.
    collector = (
        ai.experimental_telemetry.DictSink()
        if turn_input.turn_span is not None
        else None
    )
    inbox_token = proto.turn_inbox_token(session_id, turn_index)
    inbox = proto.TurnInboxHook.wait(token=inbox_token)
    output: proto.TurnOutput | None = None

    try:
        model = ai.get_model(MODEL_ID)
        async with (
            ai.experimental_telemetry.use_sink(collector),
            ai.experimental_telemetry.use_span(turn_input.turn_span),
            ai.util.TaskGroup() as tg,
        ):
            agent.tg = tg
            hook_registry = ai.HookRegistry()

            async def drive_agent_run() -> None:
                try:
                    # Open, iterate, and close the run in one task. Telemetry spans
                    # use task-local context and cannot be closed by the parent task.
                    async with agent.run(
                        model, messages, hook_registry=hook_registry
                    ) as run:
                        async for event in run:
                            if (
                                isinstance(event, ai.events.HookEvent)
                                and event.hook.status == "pending"
                                and event.hook.hook_type
                                == ai.agents.TOOL_APPROVAL_HOOK_TYPE
                            ):
                                # HookEvents ride the runtime queue, not
                                # runner.events(), so the loop never wrote this;
                                # write it here so the UI
                                # gets the approval request part.
                                await write_event(session_id, event)
                            elif isinstance(event, ai.events.RunBlocked):
                                await write_event(
                                    session_id,
                                    stream.tool_approval_requested(
                                        turn_index=turn_index
                                    ),
                                )
                        result = proto.TurnOutput(kind="suspend", messages=run.messages)
                except control.StepInterrupted:
                    messages_data = await partial_messages(
                        session_id,
                        turn_input.stream_start_index,
                        turn_input.messages,
                    )
                    result = proto.TurnOutput(
                        kind="interrupted",
                        messages=messages_data,
                    )
                except Exception as error:
                    result = proto.TurnOutput(
                        kind="error",
                        messages=messages,
                        error=f"{type(error).__name__}: {error}",
                    )
                    print(
                        f"[seal] error in run_turn:\n{traceback.format_exc()}",
                        flush=True,
                    )
                await notify_agent_finished(inbox_token, result)

            inbox_queue: asyncio.Queue[proto.TurnInboxHook] = asyncio.Queue()

            async def pump_inbox() -> None:
                async for received in inbox:
                    await inbox_queue.put(received)

            # Arm the durable inbox before the agent can reach RunBlocked. The
            # pump immediately re-awaits it after every delivery, while command
            # processing happens independently through the local queue.
            inbox_task = tg.create_task(pump_inbox())
            tg.create_task(drive_agent_run())

            while True:
                received = await inbox_queue.get()
                match received.command:
                    case proto.TurnApproval(response=response):
                        ai.resolve_hook(
                            f"{proto.TOOL_APPROVAL_HOOK_PREFIX}{response.tool_call_id}",
                            {
                                "granted": response.granted,
                                "reason": response.reason,
                            },
                            registry=hook_registry,
                        )
                    case proto.InterruptTurn():
                        # Tool tasks are siblings of the agent driver in this task
                        # group. Cancelling only agent_task can therefore leave an
                        # approval/tool sibling alive and make TaskGroup.__aexit__
                        # wait forever. The durable control signal was written
                        # before this command, so active steps also stop themselves.
                        running = [task for task in tg._tasks if not task.done()]
                        for task in running:
                            task.cancel()
                        if running:
                            await asyncio.gather(*running, return_exceptions=True)
                        messages_data = await partial_messages(
                            session_id,
                            turn_input.stream_start_index,
                            turn_input.messages,
                        )
                        output = proto.TurnOutput(
                            kind="interrupted",
                            messages=messages_data,
                        )
                        inbox_task.cancel()
                        break
                    case proto.AgentFinished(output=result):
                        output = result
                        inbox_task.cancel()
                        break
    except Exception as error:
        print(f"[seal] error in run_turn:\n{traceback.format_exc()}", flush=True)
        output = proto.TurnOutput(
            kind="error",
            messages=messages,
            error=f"{type(error).__name__}: {error}",
        )

    # Do not dispose in finally: workflow suspension unwinds this invocation.
    # The inbox must survive replay and is disposed only after AgentFinished.
    inbox.dispose()
    assert output is not None

    # deliver the body's collected spans. only complete records ship: a span
    # still open here would dangle in the shipping process's adapter.
    if collector is not None:
        finished = list(collector.finished_spans)
        if turn_input.turn_span is not None:
            # complete the turn span here (pure data ops on workflow time) so
            # it ships with the rest instead of riding the resume step.
            turn_span = turn_input.turn_span.stamp_end(
                error=ai.experimental_telemetry.SpanError(
                    type="TurnError", message=output.error
                )
                if output.kind == "error" and output.error
                else None
            )
            turn_span.set_attrs({"session.id": session_id, "turn_index": turn_index})
            finished.append(turn_span)
        if finished:
            await ship_spans(finished)

    if (
        turn_input.parent_session_id is not None
        and turn_input.background_task_id is not None
    ):
        await notify_background_task_finished(
            turn_input.parent_session_id,
            turn_input.background_task_id,
            output,
        )
    else:
        # root turns report back through the session's stable inbox.
        await notify_turn_finished(session_id, turn_index, output)
