"""Agent graph and tool definitions."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import vercel_ai_sdk as ai


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


def get_llm() -> ai.LanguageModel:
    """Create the LLM instance."""
    return ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6")


def _get_fast_llm() -> ai.LanguageModel:
    """Cheap / fast model for lightweight tasks like title generation."""
    return ai.ai_gateway.GatewayModel(model="anthropic/claude-sonnet-4-20250514")


async def generate_title(first_message: str) -> str:
    """Generate a short title for a session using a cheap LLM call."""
    llm = _get_fast_llm()
    msg = await llm.buffer(
        messages=ai.make_messages(system=_TITLE_PROMPT, user=first_message),
    )
    return msg.text.strip()


async def _execute_with_approval(
    tc: ai.ToolPart, message: ai.Message | None = None
) -> None:
    """Gate a single tool call behind user approval.

    Creates a ``ToolApproval`` hook that suspends execution until the
    frontend responds with an approve/reject decision.
    """
    approval = await ai.ToolApproval.create(  # type: ignore[attr-defined]
        f"approve_{tc.tool_call_id}",
        metadata={"tool_name": tc.tool_name, "tool_args": tc.tool_args},
    )
    if approval.granted:
        await ai.execute_tool(tc, message=message)
    else:
        tc.set_error("Tool call was denied by the user.")


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool[..., Any]],
) -> ai.StreamResult:
    """Agent graph with human-in-the-loop tool approval.

    Loops: stream LLM -> request approval -> execute tools -> repeat.
    The ToolApproval hook suspends execution and emits an approval-
    request event on the SSE stream.  The frontend displays Approve /
    Reject buttons and sends the decision back on the next request.
    """
    local_messages = list(messages)

    while True:
        result = await ai.stream_step(llm, local_messages, tools)

        if not result.tool_calls:
            return result

        last_msg = result.last_message
        assert last_msg is not None
        local_messages.append(last_msg)

        await asyncio.gather(
            *(_execute_with_approval(tc, message=last_msg) for tc in result.tool_calls)
        )
