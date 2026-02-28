---
name: ai-sdk
description: Vercel AI SDK (Python) - patterns for building LLM-powered apps with streaming, tools, hooks, and structured output
---

# Vercel AI SDK (Python)

Import as `import vercel_ai_sdk as ai`. Everything runs inside `ai.run()`.

## Core Loop

```python
result = ai.run(my_agent, llm, "query")
async for msg in result:
    print(msg.text_delta, end="")
```

`ai.run(root_fn, *args, checkpoint=None, cancel_on_hooks=False)` -> `RunResult`. The root function is any async function. If it declares a param typed `ai.Runtime`, it's auto-injected.

## Providers

```python
# Gateway (recommended) - routes to appropriate backend
llm = ai.ai_gateway.GatewayModel(model="anthropic/claude-opus-4.6", thinking=True, budget_tokens=10000)

# Direct
llm = ai.openai.OpenAIModel(model="gpt-4o")
llm = ai.anthropic.AnthropicModel(model="claude-opus-4-6-20250916", thinking=True, budget_tokens=10000)
```

All implement `LanguageModel` with `stream()` (async generator of `Message`) and `buffer()` (returns final `Message`).

## Messages

```python
messages = ai.make_messages(system="You are helpful.", user="Hello")
```

`Message` has `role`, `parts` (list of `TextPart`, `ToolPart`, `ReasoningPart`, `HookPart`, `StructuredOutputPart`), `label`, `usage`. Key properties: `msg.text_delta`, `msg.reasoning_delta`, `msg.tool_deltas`, `msg.text`, `msg.tool_calls`, `msg.is_done`, `msg.output`.

## Streaming

`@ai.stream` wraps an async generator into the Runtime queue:

```python
@ai.stream
async def my_step(llm, messages):
    async for msg in llm.stream(messages):
        yield msg
```

Built-in helpers:
- `ai.stream_step(llm, messages, tools=None, label=None, output_type=None)` -> `StreamResult` (single LLM call)
- `ai.stream_loop(llm, messages, tools, label=None, output_type=None)` -> `StreamResult` (full agent loop: call + tool exec + repeat)

`StreamResult` has `.text`, `.tool_calls`, `.output`, `.usage`, `.last_message`.

## Tools

```python
@ai.tool
async def search(query: str, limit: int = 10) -> list[str]:
    """Search the database."""
    return [...]
```

Schema extracted from type hints + docstring. `Runtime`-typed params are auto-injected, excluded from schema. Execute with `ai.execute_tool(tool_part, message=msg)`.

## Agent Patterns

Simple (stream_loop handles the tool loop):
```python
async def agent(llm, query):
    return await ai.stream_loop(llm, ai.make_messages(system="...", user=query), tools=[search])
```

Manual loop:
```python
async def agent(llm, query):
    messages = ai.make_messages(system="...", user=query)
    while True:
        result = await ai.stream_step(llm, messages, tools=[search])
        if not result.tool_calls:
            return result
        messages.append(result.last_message)
        await asyncio.gather(*(ai.execute_tool(tc, message=result.last_message) for tc in result.tool_calls))
```

Multi-agent with labels:
```python
async def multi(llm, query):
    r1, r2 = await asyncio.gather(
        ai.stream_loop(llm, msgs1, tools=[t1], label="agent-a"),
        ai.stream_loop(llm, msgs2, tools=[t2], label="agent-b"),
    )
    return await ai.stream_loop(llm, ai.make_messages(user=f"{r1.text}\n{r2.text}"), tools=[], label="summary")
```

## Structured Output

Pass a Pydantic model as `output_type`:
```python
class Forecast(pydantic.BaseModel):
    city: str
    temperature: float

result = await ai.stream_step(llm, messages, output_type=Forecast)
print(result.output.city)  # validated Pydantic instance
```

## Hooks (Human-in-the-Loop)

```python
@ai.hook
class Approval(pydantic.BaseModel):
    granted: bool

# Inside agent:
approval = await Approval.create("approve-action", metadata={"tool": "send_email"})
if approval.granted: ...

# Outside (resolve from API/UI):
Approval.resolve("approve-action", {"granted": True})
```

With `cancel_on_hooks=True`, run stops at hooks; save `result.checkpoint`, collect resolutions, re-enter with `checkpoint=`.

## Checkpoints

`ai.Checkpoint` records steps/tools/hooks for replay. Serialize with `.model_dump()`, restore with `ai.Checkpoint.model_validate(data)`. Pass to `ai.run(..., checkpoint=cp)` to skip already-completed work.

## MCP Tools

```python
tools = await ai.mcp.get_http_tools("https://mcp.example.com/mcp", headers={...}, tool_prefix="docs")
tools = await ai.mcp.get_stdio_tools("npx", "-y", "@anthropic/mcp-server-filesystem", "/tmp", tool_prefix="fs")
```

Returns native `Tool` objects usable in `stream_step`/`stream_loop`.

## AI SDK UI Adapter

```python
from vercel_ai_sdk.ai_sdk_ui import to_sse_stream, to_messages, UI_MESSAGE_STREAM_HEADERS

messages = to_messages(request.messages)
return StreamingResponse(to_sse_stream(ai.run(agent, llm, query)), headers=UI_MESSAGE_STREAM_HEADERS)
```

## Pre-built Agent

```python
import vercel_ai_sdk.agent as agent

a = agent.Agent(model=llm, filesystem=agent.local.LocalFilesystem(), system="...")
async for msg in a.run(messages):
    print(msg.text_delta, end="")
```

Built-in tools: `read`, `write`, `edit`, `ls`, `glob`, `grep`, `bash`. All gated by `agent.ToolApproval` hook.

## Key Imports

```python
import vercel_ai_sdk as ai

# Core
ai.run, ai.stream_step, ai.stream_loop, ai.execute_tool
ai.tool, ai.stream, ai.hook
ai.make_messages, ai.Message, ai.StreamResult, ai.RunResult
ai.Runtime, ai.Checkpoint, ai.Tool, ai.ToolSchema, ai.LanguageModel

# Parts
ai.TextPart, ai.ToolPart, ai.ReasoningPart, ai.HookPart, ai.StructuredOutputPart

# Providers
ai.ai_gateway.GatewayModel, ai.openai.OpenAIModel, ai.anthropic.AnthropicModel

# MCP
ai.mcp.get_http_tools, ai.mcp.get_stdio_tools

# UI
ai.ai_sdk_ui.to_sse_stream, ai.ai_sdk_ui.to_messages

# Telemetry
ai.telemetry.enable()
```
