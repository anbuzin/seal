"""Seal's tools, and the durable child that runs one call.

Tool *bodies* are unchanged from the workflow port — they are plain async
functions again, with no ``@workflow.step`` wrapper.

Execution splits three ways, decided by the Session's dispatch arm (not by
per-tool config):

* **Eager** tools (non-effectful: ``web_fetch``, ``generate_image``) run as
  plain asyncio tasks *inside* the model-turn activation, started the moment
  the model finishes emitting the call — before the whole turn is done. This
  replaces the workflow port's EagerToolHook/watcher contraption: in-process
  tasks are cancellable, and a turn that rolls back takes its provisional
  eager work with it.
* **Gated** tools (``bash`` on the main session) park on a durable Approval
  hook; the granted call is then spawned as a ``run_tool`` child.
* Everything else spawns a keyed ``run_tool`` child: its own mailbox, lease,
  retry clock, and terminal verdict. One failed call never re-runs a sibling.

``subagent`` is schema-only: the Session intercepts it and spawns a real
child process instead of invoking a function.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import ai
from rotor import TerminalFailure
from rotor.patterns import task

from agent import proto

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


@ai.tool
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


@ai.tool
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
async def generate_image(prompt: str) -> dict[str, Any]:
    """Generate an image from a text prompt. Describe the desired image in
    detail, including subject, style, and composition."""
    # the ai library has no direct image-generation API yet, so this runs a
    # model that emits images inline with its response (FileParts).
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
    return ai.content_output(
        *(
            part
            for part in message.parts
            if isinstance(part, ai.messages.TextPart | ai.messages.FilePart)
        )
    ).model_dump(mode="json")


@ai.tool
async def subagent(prompt: str, name: str | None = None) -> str:
    """Delegate a focused task to a child agent and return its answer."""
    raise RuntimeError(
        "subagent is dispatched as a durable child process, never invoked "
        "in-process — see Session.generate"
    )


TOOLS = [bash, web_fetch, generate_image, subagent]

# bash is approval-gated on the main session only; a subagent cannot surface
# an approval UI, so its Session subclass simply has an empty gate set (the
# workflow port needed a dataclasses.replace() copy of the tool for this).
GATED = frozenset({"bash"})

# Non-effectful, non-streaming tools that may start before the model turn
# finishes. They run in-handler; a rolled-back turn discards their work.
EAGER = frozenset({"web_fetch", "generate_image"})


async def execute_tool(call: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call dict and shape its ToolResultPart dump."""
    tool = next((t for t in TOOLS if t.name == call["tool_name"]), None)
    if tool is None:
        raise ToolError(f"unknown tool {call['tool_name']!r}")
    args = json.loads(call["tool_args"] or "{}")
    value = await tool.fn(**args)
    return proto.tool_result(call, value)


class ToolError(TerminalFailure):
    """A conclusion, not a crash: bypasses retry, lands in the parent's
    ``run_tool.Failed`` arm as-is."""


def retry_tool(error: Exception, attempt: int) -> str | None:
    """Policy is code: retry transient transport errors, briefly."""
    import httpx

    transient = isinstance(error, httpx.TransportError | asyncio.TimeoutError)
    return "2s" if transient and attempt <= 3 else None


@task(retry=retry_tool)
async def run_tool(call: dict[str, Any]) -> dict[str, Any]:
    """One tool call, one durable child, its own retry clock.

    The AI SDK's stable ``tool_call_id`` doubles as the provider idempotency
    key for side-effecting tools: it names the same semantic call across
    crash redelivery and policy retries alike.
    """
    return await execute_tool(call)
