"""The durable-stream → AI SDK UI bridge: where duplicate/missing-message
regressions live.

``active_run_start_index`` decides where a reload resumes (wrong answer =
duplicated assistant message in the UI), ``to_sse`` decides when a stream
terminates (wrong answer = hang or truncated turn).

All tests run against real jsonl streams and the real ai SDK UI adapter.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import ai
import ai.types.events as events_
import ai.types.messages as messages_

from agent import proto, stream
from app import chat


async def _write(session_id: str, *events: proto.StreamEvent) -> None:
    writer = await stream.get_writable(session_id)
    for event in events:
        await writer.write(event)


def _text_events(text: str, *, block: str = "b") -> list[proto.StreamEvent]:
    message = ai.messages.Message(
        role="assistant", parts=[messages_.TextPart(text=text)]
    )
    return [
        events_.StreamStart(),
        events_.TextStart(block_id=block),
        events_.TextDelta(block_id=block, chunk=text),
        events_.TextEnd(block_id=block),
        events_.StreamEnd(message=message),
    ]


# --- active_run_start_index -------------------------------------------------------


async def test_no_stream_means_nothing_to_resume() -> None:
    assert await chat.active_run_start_index("s1") is None


async def test_completed_run_is_not_resumable() -> None:
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("hi"),
        stream.turn_completed(turn_index=0, kind="suspend"),
        stream.session_waiting(turn_index=0),
    )
    assert await chat.active_run_start_index("s1") is None


async def test_in_flight_run_resumes_from_its_opener() -> None:
    await _write(
        "s1",
        stream.session_started(),  # 0
        stream.turn_started(turn_index=0),  # 1
        *_text_events("hi"),  # 2-6
        stream.turn_completed(turn_index=0, kind="suspend"),  # 7
        stream.session_waiting(turn_index=0),  # 8
        stream.turn_started(turn_index=1),  # 9 ← in-flight run opens here
        events_.StreamStart(),  # 10
    )
    assert await chat.active_run_start_index("s1") == 9


async def test_background_followup_resumes_from_the_logical_run_opener() -> None:
    # A fast child can wake the next root turn without a quiescent wait in between;
    # reconnect must rebuild the whole logical response from the original opener.
    await _write(
        "s1",
        stream.session_started(),  # 0
        stream.turn_started(turn_index=0),  # 1
        *_text_events("delegating"),  # 2-6
        stream.turn_completed(turn_index=0, kind="suspend"),  # 7
        stream.subagent_completed(tool_call_id="tc-1", is_error=False),  # 8
        stream.turn_started(turn_index=1, background=True),  # 9
        events_.StreamStart(),  # 10
    )
    assert await chat.active_run_start_index("s1") == 1


async def test_run_parked_on_approval_is_resumable_from_run_start() -> None:
    # a turn parked on an approval is mid-run, so a cold reload re-tails from the
    # opener to rebuild the same UI message (the live POST resume folds instead).
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("need approval"),
        stream.tool_approval_requested(turn_index=0),
    )
    assert await chat.active_run_start_index("s1") == 1


# --- to_sse ----------------------------------------------------------------------


async def _collect_sse(session_id: str, start_index: int = 0) -> list[str]:
    async def drain() -> list[str]:
        return [line async for line in chat.to_sse(session_id, start_index)]

    return await asyncio.wait_for(drain(), timeout=5)


def _sse_payloads(lines: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for line in lines:
        body = line.removeprefix("data: ").strip()
        if body and body != "[DONE]":
            payloads.append(json.loads(body))
    return payloads


async def test_to_sse_streams_background_followup_in_one_ui_run() -> None:
    # A root turn may finish while its background work is still active. The AI SDK
    # response stays open and folds the automatic follow-up into the same message.
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        *_text_events("hello world"),
        stream.turn_completed(turn_index=0, kind="suspend"),
        stream.session_waiting(turn_index=0, active_background_tasks=1),
        stream.turn_started(turn_index=1, background=True),
        *_text_events("next turn", block="next"),
        stream.turn_completed(turn_index=1, kind="suspend"),
        stream.session_waiting(turn_index=1),
    )
    lines = await _collect_sse("s1")

    deltas = [
        payload
        for payload in _sse_payloads(lines)
        if payload.get("type") == "text-delta"
    ]
    assert [delta["delta"] for delta in deltas] == ["hello world", "next turn"]
    assert lines[-1].startswith("data:")
    assert "[DONE]" in lines[-1]


async def test_to_sse_parks_at_a_deferred_approval() -> None:
    hook: messages_.HookPart[Any] = messages_.HookPart(
        hook_id="approve_tc-1",
        hook_type="ToolApproval",
        tool_call_id="tc-1",
        status="pending",
        metadata={"tool": "bash", "kwargs": {"command": "rm -rf /tmp/x"}},
    )
    tool_call = ai.messages.Message(
        role="assistant",
        parts=[
            messages_.ToolCallPart(
                tool_call_id="tc-1",
                tool_name="bash",
                tool_args='{"command": "rm -rf /tmp/x"}',
            )
        ],
    )
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-1", tool_name="bash"),
        events_.ToolEnd(tool_call_id="tc-1", tool_call=tool_call.tool_calls[0]),
        events_.StreamEnd(message=tool_call),
        events_.HookEvent(
            message=ai.messages.Message(role="internal", parts=[hook]), hook=hook
        ),
        stream.tool_approval_requested(turn_index=0),
    )
    lines = await _collect_sse("s1")

    kinds = [payload.get("type") for payload in _sse_payloads(lines)]
    assert "tool-approval-request" in kinds
    assert "[DONE]" in lines[-1]


async def test_submit_approvals_replays_events_after_the_approval_marker(
    monkeypatch: Any,
) -> None:
    await _write(
        "s1",
        stream.turn_started(turn_index=0),
        stream.tool_approval_requested(turn_index=0),
        stream.subagent_completed(tool_call_id="tc-sub", is_error=False),
    )

    async def resume(_token: str, _hook: Any) -> None:
        return None

    monkeypatch.setattr(chat, "_resume", resume)
    start_index = await chat.submit_approvals(
        "s1", [proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)]
    )
    assert start_index == 2


async def test_approval_resume_continuation_opens_id_less() -> None:
    # an approval resume tails the continuation, which opens with the tool result
    # (run_turn writes results before the answer). Its message has no turn_id, so
    # the adapter emits an id-less ``start`` and the client folds the output +
    # answer into the assistant message it resubmitted (no new message).
    tool_msg = ai.tool_message(tool_call_id="tc-1", tool_name="bash", result="out")
    await _write(
        "s1",
        events_.ToolCallResult(message=tool_msg, results=tool_msg.tool_results),
        *_text_events("done"),
        stream.session_waiting(turn_index=0),
    )

    payloads = _sse_payloads(await _collect_sse("s1"))
    starts = [p for p in payloads if p.get("type") == "start"]
    assert starts and "messageId" not in starts[0]
    assert any(p.get("type") == "tool-output-available" for p in payloads)
    assert any(p.get("type") == "text-delta" for p in payloads)


async def test_to_sse_reloads_the_current_step_and_continues() -> None:
    # a retried llm_step wipes its own aborted attempt's events and asks the
    # client to discard the current step. The stream stays open so events from
    # the retry can continue over the same connection.
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-aborted", tool_name="bash"),
        stream.reload_requested(),
        events_.ToolStart(tool_call_id="tc-retry", tool_name="bash"),
        stream.session_waiting(turn_index=0),
    )
    lines = await _collect_sse("s1")

    payloads = _sse_payloads(lines)
    assert any(p.get("type") == "data-reload" for p in payloads)
    tool_starts = [p for p in payloads if p.get("type") == "tool-input-start"]
    assert [p["toolCallId"] for p in tool_starts] == ["tc-aborted", "tc-retry"]
    assert "[DONE]" in lines[-1]


async def test_to_sse_replays_reload_marker() -> None:
    # ``reload.requested`` is durable history, so a fresh connection sees the
    # marker too. It forwards it and continues with the retried step's events.

    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        stream.reload_requested(),
        *_text_events("done"),
        stream.turn_completed(turn_index=0, kind="suspend"),
        stream.session_waiting(turn_index=0),
    )
    lines = await _collect_sse("s1")

    payloads = _sse_payloads(lines)
    assert any(p.get("type") == "data-reload" for p in payloads)
    deltas = [p for p in payloads if p.get("type") == "text-delta"]
    assert [delta["delta"] for delta in deltas] == ["done"]
    assert "[DONE]" in lines[-1]


def test_project_background_task_updates_original_tool_part() -> None:
    ui_messages: list[dict[str, Any]] = [
        {
            "id": "a1",
            "role": "assistant",
            "parts": [
                {
                    "type": "tool-subagent",
                    "toolCallId": "tc-1",
                    "state": "output-available",
                    "input": {"prompt": "go"},
                    "output": (
                        "Subagent is running in the background and will update you "
                        "later."
                    ),
                }
            ],
        },
        {"id": "u2", "role": "user", "parts": [{"type": "text", "text": "later"}]},
    ]
    task = proto.BackgroundTaskState(
        task_id="tc-1",
        child_session_id="s1:child:tc-1",
        name="helper",
        status="completed",
        messages=[ai.assistant_message("child answer")],
    )

    projected = chat.project_background_tasks(ui_messages, {"tc-1": task})
    parts = projected[0]["parts"]
    assert isinstance(parts, list)
    part = cast(dict[str, Any], parts[0])
    assert part["toolCallId"] == "tc-1"
    output = cast(list[dict[str, Any]], part["output"])
    output_parts = cast(list[dict[str, Any]], output[0]["parts"])
    assert output_parts[0]["text"] == "child answer"
    assert part["preliminary"] is False
    user_parts = cast(list[dict[str, Any]], projected[1]["parts"])
    assert user_parts[0]["text"] == "later"


async def test_to_sse_stitches_background_subagent_into_the_ui_run() -> None:
    parent_call = messages_.ToolCallPart(
        tool_call_id="tc-1", tool_name="subagent", tool_args='{"prompt":"go"}'
    )
    acknowledgement = "Subagent is running in the background and will update you later."
    await _write(
        "s1",
        stream.session_started(),
        stream.turn_started(turn_index=0),
        events_.StreamStart(),
        events_.ToolStart(tool_call_id="tc-1", tool_name="subagent"),
        events_.ToolEnd(tool_call_id="tc-1", tool_call=parent_call),
        events_.StreamEnd(
            message=ai.messages.Message(role="assistant", parts=[parent_call])
        ),
        stream.subagent_called(
            tool_call_id="tc-1", child_session_id="s1:child:tc-1", name="helper"
        ),
        events_.ToolCallResult(
            message=ai.tool_message(
                tool_call_id="tc-1",
                tool_name="subagent",
                result=acknowledgement,
            ),
            results=[
                ai.tool_result_part(
                    "tc-1",
                    tool_name="subagent",
                    result=acknowledgement,
                )
            ],
        ),
        stream.turn_completed(turn_index=0, kind="suspend"),
        stream.session_waiting(turn_index=0, active_background_tasks=1),
        stream.subagent_event(
            tool_call_id="tc-1",
            event=events_.TextDelta(
                block_id="child",
                chunk="child",
                message=ai.assistant_message("child"),
            ).model_dump(mode="json"),
        ),
        stream.subagent_completed(
            tool_call_id="tc-1",
            is_error=False,
            messages=[ai.assistant_message("child answer").model_dump(mode="json")],
        ),
        stream.turn_started(turn_index=1, background=True),
        *_text_events("final answer", block="final"),
        stream.turn_completed(turn_index=1, kind="suspend"),
        stream.session_waiting(turn_index=1),
    )

    lines = [line async for line in chat.to_sse("s1", 0)]
    payloads = _sse_payloads(lines)
    outputs = [
        payload
        for payload in payloads
        if payload.get("type") == "tool-output-available"
    ]
    assert outputs[0]["output"] == acknowledgement
    assert outputs[-1]["output"][0]["parts"][0]["text"] == "child answer"
    assert any(payload.get("preliminary") for payload in outputs)
    assert any(
        payload.get("type") == "text-delta" and payload.get("delta") == "final answer"
        for payload in payloads
    )
    assert "[DONE]" in lines[-1]
