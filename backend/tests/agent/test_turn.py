"""Tests for components used by turn."""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any, cast

import ai
import ai.types.events as events_
import ai.types.messages as messages_

from agent import control, stream, turn


def _assistant_with_tool_calls() -> messages_.Message:
    return messages_.Message(
        role="assistant",
        parts=[
            messages_.TextPart(text="running it"),
            messages_.ToolCallPart(
                tool_call_id="tc-1", tool_name="bash", tool_args='{"command": "ls"}'
            ),
            messages_.ToolCallPart(
                tool_call_id="tc-2",
                tool_name="web_fetch",
                tool_args='{"url": "https://example.com"}',
            ),
        ],
    )


def test_round_trip_preserves_replay_and_cached_result() -> None:
    cached = ai.tool_result_part("tc-1", tool_name="bash", result="file.txt")
    message = _assistant_with_tool_calls()
    message = message.model_copy(
        update={
            "replay": True,
            "parts": [
                part.model_copy(update={"cached_result": cached})
                if isinstance(part, messages_.ToolCallPart)
                and part.tool_call_id == "tc-1"
                else part
                for part in message.parts
            ],
        }
    )

    restored = ai.types.messages.Message.model_validate(message.model_dump(mode="json"))

    assert restored.replay is True
    by_id = {part.tool_call_id: part for part in restored.tool_calls}
    assert by_id["tc-1"].cached_result is not None
    assert by_id["tc-1"].cached_result.result == "file.txt"
    assert by_id["tc-2"].cached_result is None
    # the visible content is untouched
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


async def test_partial_messages_closes_interrupted_tool_call() -> None:
    user = ai.user_message("run")
    assistant = ai.messages.Message(
        role="assistant",
        parts=[
            messages_.TextPart(text="starting"),
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="bash",
                tool_args='{"command": "sleep 30"}',
            ),
        ],
    )
    writer = await stream.get_writable("s1")
    await writer.write(events_.StreamEnd(message=assistant))

    messages_data = await turn.partial_messages.func(
        "s1",
        0,
        [user.model_dump(mode="json")],
    )
    messages = [
        ai.messages.Message.model_validate(message) for message in messages_data
    ]

    assert [message.role for message in messages] == ["user", "assistant", "tool"]
    [result] = messages[-1].tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "Interrupted by user"
    assert result.result_kind == "error"


def test_cancellable_steps_stack_with_tools() -> None:
    assert cast(Any, turn.bash.fn).__name__ == "bash"
    assert cast(Any, turn.web_fetch.fn).__name__ == "web_fetch"
    assert cast(Any, turn.image_step).__name__ == "image_step"
    assert turn.bash.validator is not None
    assert set(turn.bash.validator.model_fields) == {"command", "timeout"}


async def test_bash_tool_kills_process_group_on_interrupt(
    tmp_path: pathlib.Path,
) -> None:
    marker = str(tmp_path / "done")
    token = control.cancellation_context.set(("s1", 0))
    step = cast(Any, turn.bash.fn)
    task = asyncio.create_task(step.func(f"sleep 5; touch {marker}"))
    control.cancellation_context.reset(token)
    await asyncio.sleep(0.05)
    await control.interrupt("s1", 0)
    try:
        await task
    except control.StepInterrupted:
        pass
    else:
        raise AssertionError("bash step was not interrupted")
    await asyncio.sleep(0.1)
    assert not pathlib.Path(marker).exists()


def test_round_trip_of_plain_message_stays_plain() -> None:
    message = _assistant_with_tool_calls()

    restored = ai.types.messages.Message.model_validate(message.model_dump(mode="json"))

    assert restored.replay is False
    assert all(part.cached_result is None for part in restored.tool_calls)
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")
