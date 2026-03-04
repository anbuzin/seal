"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import fastapi
import fastapi.middleware.cors
import fastapi.responses
import pydantic

from backend import agent

import vercel_ai_sdk as ai
import vercel_ai_sdk.ai_sdk_ui

app = fastapi.FastAPI(
    title="seal",
    description="Seal – personal AI assistant",
)

app.add_middleware(
    fastapi.middleware.cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


class ChatRequest(pydantic.BaseModel):
    """Request body for the chat endpoint."""

    messages: list[ai.ai_sdk_ui.UIMessage]
    session_id: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest) -> fastapi.responses.StreamingResponse:
    """Handle chat requests and stream responses."""
    messages = ai.ai_sdk_ui.to_messages(request.messages)

    # Prepend the system prompt
    system = ai.Message(
        role="system",
        parts=[ai.TextPart(text=agent.SYSTEM)],
    )
    all_messages = [system, *messages]

    llm = agent.get_llm()

    result = ai.run(
        agent.graph,
        llm,
        all_messages,
        agent.TOOLS,
    )

    async def stream_response() -> AsyncGenerator[str]:
        async for chunk in ai.ai_sdk_ui.to_sse_stream(result):
            yield chunk

    return fastapi.responses.StreamingResponse(
        stream_response(),
        headers=ai.ai_sdk_ui.UI_MESSAGE_STREAM_HEADERS,
    )
