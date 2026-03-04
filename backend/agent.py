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


def get_llm() -> ai.LanguageModel:
    """Create the LLM instance."""
    return ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6")


async def graph(
    llm: ai.LanguageModel,
    messages: list[ai.Message],
    tools: list[ai.Tool[..., Any]],
) -> ai.StreamResult:
    """Agent graph: stream_loop with no approval gates.

    Tools execute immediately. Loops until the model stops
    issuing tool calls.
    """
    return await ai.stream_loop(llm, messages, tools)
