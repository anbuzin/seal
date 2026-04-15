"""Agent definition and tool declarations."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator
from typing import Any

import ai
import httpx


@ai.tool
async def bash(command: str, timeout: int | None = None) -> str:
    """Execute a bash command.

    Use timeout (seconds) to limit long-running commands.
    """
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
    url: str, method: str = "GET", headers: str = "", body: str = ""
) -> str:
    """Fetch a URL and return the response.

    Args:
        url: The URL to fetch.
        method: HTTP method (GET, POST, PUT, DELETE, etc.).
        headers: Optional headers as newline-separated "Key: Value" pairs.
        body: Optional request body for POST/PUT.
    """
    parsed_headers: dict[str, str] = {}
    for line in headers.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed_headers[k.strip()] = v.strip()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.request(
            method,
            url,
            headers=parsed_headers or None,
            content=body or None,
        )

    parts = [
        f"HTTP {response.status_code}",
        *(f"{k}: {v}" for k, v in response.headers.items()),
        "",
        response.text[:50_000],
    ]
    return "\n".join(parts)


SYSTEM = """You are a helpful assistant with access to a bash shell and the internet."""

TOOLS: list[ai.Tool[..., Any]] = [bash, web_fetch]

_TITLE_PROMPT = (
    "Generate a concise 3-6 word title for a conversation that starts with "
    "the following message. Reply with ONLY the title, no quotes or punctuation."
)

_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "seal_current_session_id",
    default=None,
)


def activate_session(session_id: str) -> contextvars.Token[str | None]:
    """Bind a session id to the current task for hook namespacing."""
    return _current_session_id.set(session_id)


def deactivate_session(token: contextvars.Token[str | None]) -> None:
    """Restore the previous session binding."""
    _current_session_id.reset(token)


def make_approval_label(tool_call_id: str) -> str:
    """Construct a ToolApproval hook label compatible with the UI adapter."""
    return f"approve_{tool_call_id}"


def get_model() -> ai.Model:
    """Create the primary LLM instance."""
    return ai.ai_gateway("anthropic/claude-opus-4.6")


def _get_fast_model() -> ai.Model:
    """Cheap / fast model for lightweight tasks like title generation."""
    return ai.ai_gateway("anthropic/claude-sonnet-4-20250514")


async def generate_title(first_message: str) -> str:
    """Generate a short title for a session using a cheap LLM call."""
    model = _get_fast_model()
    messages = [
        ai.system_message(_TITLE_PROMPT),
        ai.user_message(first_message),
    ]
    stream = await ai.stream(model, messages)
    async for _ in stream:
        pass
    return stream.text.strip()


# ---------------------------------------------------------------------------
# Agent with human-in-the-loop tool approval
# ---------------------------------------------------------------------------

seal = ai.agent(tools=TOOLS)


async def _execute_with_approval(tc: ai.ToolCall) -> ai.Message | None:
    """Resolve one tool approval and execute the tool if approved.

    Returns ``None`` when the approval is still pending and the run should
    suspend for serverless re-entry.
    """
    try:
        approval: ai.ToolApproval = await ai.hook(
            make_approval_label(tc.id),
            payload=ai.ToolApproval,
            metadata={
                "session_id": _current_session_id.get(),
                "tool_name": tc.name,
                "tool_args": tc.kwargs,
            },
            interrupt_loop=True,
        )
    except asyncio.CancelledError:
        return None

    if approval.granted:
        return await tc()

    return ai.tool_message(
        ai.tool_result(
            tc.id,
            tool_name=tc.name,
            result="Tool call was denied by the user.",
            is_error=True,
        )
    )


@seal.loop
async def _loop(context: ai.Context) -> AsyncGenerator[ai.Message]:
    """Agent loop with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The hook suspends execution and emits an approval-request event on
    the SSE stream. The frontend displays Approve / Reject buttons and
    sends the decision back on the next request.
    """
    while True:
        stream = await ai.stream(context.model, context.messages, tools=context.tools)
        async for msg in stream:
            yield msg

        tool_calls = context.resolve(stream.tool_calls)
        if not tool_calls:
            return

        # Gate tool calls behind concurrent approvals so every pending tool
        # from the current model step can surface in one round-trip.
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_execute_with_approval(tc)) for tc in tool_calls]

        results = [task.result() for task in tasks]
        completed = [result for result in results if result is not None]
        if completed:
            yield ai.tool_message(*completed)

        if any(result is None for result in results):
            return
