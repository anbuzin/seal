"""End-to-end driver tests: the real durable engine, in-process.

The harness (``tests/harness.py``) runs ``run_session`` on the real workflow
engine; the only test double is the scripted model (``scripted_model``).

These are the regression net for the failure modes that matter here:
duplicated or missing messages after replay, unanswered tool calls, and
deadlocks (every wait is bounded, so a deadlock is a fast red test).
"""

from __future__ import annotations

import asyncio
import base64
import os

import ai
import pytest
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg
from harness import (
    InProcessWorld,
)
from harness import (
    interrupt_session as _interrupt,
)
from harness import (
    lifecycle as _lifecycle,
)
from harness import (
    resume_approval as _resume_approval,
)
from harness import (
    resume_session as _resume,
)
from harness import (
    start_session as _start,
)
from harness import (
    wait_for_lifecycle as _wait_for_lifecycle,
)
from harness import (
    wait_run as _wait_run,
)

from agent import proto, session, storage, stream


async def test_single_turn_suspends_then_closes(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    run = await _start("s1", "hi")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == ["system", "user", "assistant"]
    assert state.messages[-1].text == "hello there"
    assert_message_invariants(state.messages)

    await _resume(proto.session_inbox_token("s1"), proto.NewUserMessage(close=True))
    output = await _wait_run(run)
    assert output.output == "hello there"
    assert not output.is_error

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
        proto.SESSION_COMPLETED,
    ]
    _, closed = await storage.store().info("s1", "default")
    assert closed


async def test_interrupt_waiting_turn_preserves_history_and_allows_resume(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first answer")], [text_msg("second answer")]]

    await _start("s1", "one")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)
    await _interrupt("s1")
    await _wait_for_lifecycle("s1", proto.SESSION_INTERRUPTED)

    interrupted = await session.read_session("s1")
    assert interrupted is not None
    assert interrupted.messages[-1].text == "first answer"
    assert interrupted.active_ui_run_message_index is None

    await _resume(proto.session_inbox_token("s1"), proto.NewUserMessage(prompt="two"))
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    resumed = await session.read_session("s1")
    assert resumed is not None
    assert [message.text for message in resumed.messages if message.role == "user"] == [
        "one",
        "two",
    ]
    assert resumed.messages[-1].text == "second answer"
    assert_message_invariants(resumed.messages)


async def test_resume_appends_user_message_without_duplicating_history(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first answer")], [text_msg("second answer")]]

    await _start("s1", "one")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    await _resume(proto.session_inbox_token("s1"), proto.NewUserMessage(prompt="two"))
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m.text for m in state.messages if m.role == "user"] == ["one", "two"]
    assert state.messages[-1].text == "second answer"
    assert_message_invariants(state.messages)
    assert scripted_model.call_count == 2


async def test_gated_tool_approval_runs_in_one_turn(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command": "echo approved-run"}',
                text="running it",
            )
        ],
        [text_msg("done")],
    ]

    await _start("s1", "run it")
    # the turn parks on the approval hook and emits tool_approval.requested; the
    # gated tool has not run yet, so the model was called exactly once.
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    # the bash subprocess really ran, exactly once, after the approval landed
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "approved-run\n"
    assert_message_invariants(state.messages)
    # one model call for the gated turn, one more for the final answer
    assert scripted_model.call_count == 2

    # the whole exchange is a single turn that parked once on the approval.
    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TOOL_APPROVAL_REQUESTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]


async def test_interrupt_approval_wait_reconciles_history_without_running_tool(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command": "touch /tmp/seal-should-not-run"}',
                text="waiting for approval",
            )
        ]
    ]

    await _start("s1", "run it")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)
    calls_before_interrupt = scripted_model.call_count

    await _interrupt("s1")
    await _wait_for_lifecycle("s1", proto.SESSION_INTERRUPTED)

    state = await session.read_session("s1")
    assert state is not None
    assert state.active_ui_run_message_index is None
    assert scripted_model.call_count == calls_before_interrupt == 1
    assert [message.role for message in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    [result] = state.messages[-1].tool_results
    assert result.tool_call_id == "tc-1"
    assert result.result == "Interrupted by user"
    assert result.result_kind == "error"
    assert_message_invariants(state.messages)


async def test_parallel_gated_tools_park_then_run(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="running both"),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-a",
                        tool_name="bash",
                        tool_args='{"command": "echo a"}',
                    ),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-b",
                        tool_name="bash",
                        tool_args='{"command": "echo b"}',
                    ),
                ],
            )
        ],
        [text_msg("done")],
    ]

    await _start("s2", "run both")
    # both gated calls park on their own hook before the turn parks.
    await _wait_for_lifecycle("s2", proto.TOOL_APPROVAL_REQUESTED)
    assert scripted_model.call_count == 1

    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-a", granted=True)
    )
    await _resume_approval(
        "s2", proto.ToolApprovalResponse(tool_call_id="tc-b", granted=True)
    )
    await _wait_for_lifecycle("s2", proto.SESSION_WAITING)

    state = await session.read_session("s2")
    assert state is not None
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r.result for r in tool_message.tool_results}
    assert results == {"tc-a": "a\n", "tc-b": "b\n"}
    assert_message_invariants(state.messages)
    assert scripted_model.call_count == 2
    assert proto.TOOL_APPROVAL_REQUESTED in await _lifecycle("s2")


async def test_subagent_returns_immediately_then_wakes_parent_with_user_message(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-sub",
                name="subagent",
                args='{"prompt": "say hi", "name": "helper"}',
                text="delegating",
            )
        ],
        [text_msg("working in the background")],
        [text_msg("final answer")],
    ]
    scripted_model.keyed_responses = {"say hi": [text_msg("child answer")]}

    await _start("s1", "delegate")
    # The parent completes a turn without waiting for the child tool call.
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)
    first_state = await session.read_session("s1")
    assert first_state is not None
    assert first_state.messages[-1].text == "working in the background"

    await _wait_for_lifecycle("s1", proto.SUBAGENT_COMPLETED)
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    state = await session.read_session("s1")
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert_message_invariants(state.messages)
    assert state.messages[-2].text == (
        'Background subagent "helper" finished:\n\nchild answer'
    )
    assert state.messages[-1].text == "final answer"

    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-sub"
    assert result.result == (
        "Subagent is running in the background and will update you later."
    )

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.SUBAGENT_CALLED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
        proto.SUBAGENT_COMPLETED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]
    # The child ran on its own stream and its progress was durably proxied onto
    # the parent stream for the UI without changing the parent's model history.
    assert await _lifecycle("s1:child:tc-sub") == []
    parent_events = [event async for event in stream.replay("s1")]
    proxied = [
        event
        for event in parent_events
        if isinstance(event, proto.LifecycleEvent)
        and event.type == proto.SUBAGENT_EVENT
    ]
    assert proxied
    assert any(event.data["event"].get("kind") == "text_delta" for event in proxied)
    assert scripted_model.call_count == 4


async def test_interrupt_background_child_suppresses_parent_followup(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-sub",
                name="subagent",
                args='{"prompt": "child-blocked", "name": "helper"}',
                text="delegating",
            )
        ],
        [text_msg("working in the background")],
    ]
    scripted_model.keyed_responses = {"child-blocked": [text_msg("child answer")]}

    await _start("s1", "delegate")
    await _wait_for_lifecycle("s1", proto.SUBAGENT_CALLED)
    await _interrupt("s1")
    await _wait_for_lifecycle("s1", proto.SESSION_INTERRUPTED)
    calls_after_ack = scripted_model.call_count
    await asyncio.sleep(0.1)

    state = await session.read_session("s1")
    assert state is not None
    assert state.background_tasks["tc-sub"].status == "interrupted"
    assert state.active_ui_run_message_index is None
    assert scripted_model.call_count == calls_after_ack
    assert not any(
        message.role == "user"
        and message.text
        and message.text.startswith('Background subagent "helper"')
        for message in state.messages
    )
    assert_message_invariants(state.messages)


async def test_generate_image_returns_multipart_result(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    png_b64 = base64.b64encode(b"\x89PNG fake image bytes").decode()
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-img",
                name="generate_image",
                args='{"prompt": "a cat in cherry blossoms"}',
                text="drawing it",
            )
        ],
        # the image model's turn: an inline image alongside text
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="here it is"),
                    ai.messages.FilePart(data=png_b64, media_type="image/png"),
                ],
            )
        ],
        [text_msg("done drawing")],
    ]

    await _start("s1", "draw a cat")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert_message_invariants(state.messages)
    assert state.messages[-1].text == "done drawing"

    # the tool result is a multipart ContentOutput carrying the image, so the
    # model (and the UI adapter) sees the actual media.
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-img"
    assert result.result_kind == "special"
    content = result.result
    assert isinstance(content, ai.messages.ContentOutput)
    [text_part, file_part] = content.value
    assert isinstance(text_part, ai.messages.TextPart)
    assert text_part.text == "here it is"
    assert isinstance(file_part, ai.messages.FilePart)
    assert file_part.media_type == "image/png"
    assert file_part.data == png_b64
    assert scripted_model.call_count == 3

    # what the follow-up model call actually saw: the tool result's
    # model-facing value must still be typed after the step JSON round-trip,
    # or providers JSON-encode it and the image goes up as base64 text.
    final_call = scripted_model.calls[-1]
    [seen_result] = [part for m in final_call for part in m.tool_results]
    assert isinstance(seen_result.get_model_input(), ai.messages.ContentOutput)


# How many times to repeat the parallel-subagent stress. Kept low by default
# (each iteration re-imports `ai` per delivery, which is slow); bump for a
# heavier determinism sweep, e.g. SEAL_PARALLEL_SUBAGENT_ITERS=24.
_PARALLEL_SUBAGENT_ITERS = int(os.environ.get("SEAL_PARALLEL_SUBAGENT_ITERS", "2"))


@pytest.mark.parametrize("iteration", range(_PARALLEL_SUBAGENT_ITERS))
async def test_parallel_subagents_land_deterministically(
    world: InProcessWorld, scripted_model: MockProvider, iteration: int
) -> None:
    # Two subagents scheduled from one assistant turn run concurrently: their
    # tool coroutines and the agent loop all issue durable ``write_event`` steps,
    # so the engine must deliver recorded completions one at a time (fully
    # draining each before the next) or the two coroutines interleave their
    # writes differently across replays -> NondeterminismError. Repeated to catch
    # the flaky ordering.
    session_id = f"s{iteration}"
    scripted_model.responses = [
        [
            ai.messages.Message(
                role="assistant",
                parts=[
                    ai.messages.TextPart(text="delegating both"),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-a",
                        tool_name="subagent",
                        tool_args='{"prompt": "task-alpha", "name": "alpha"}',
                    ),
                    ai.messages.ToolCallPart(
                        tool_call_id="tc-b",
                        tool_name="subagent",
                        tool_args='{"prompt": "task-beta", "name": "beta"}',
                    ),
                ],
            )
        ],
        [text_msg("both are running")],
        [text_msg("wrapped up")],  # parent's follow-up after both children report
    ]
    scripted_model.keyed_responses = {
        "task-alpha": [text_msg("alpha-report")],
        "task-beta": [text_msg("beta-report")],
    }

    await _start(session_id, "delegate both")
    await _wait_for_lifecycle(session_id, proto.SESSION_WAITING)
    first_state = await session.read_session(session_id)
    assert first_state is not None
    assert first_state.messages[-1].text == "both are running"

    await _wait_for_lifecycle(session_id, proto.SUBAGENT_COMPLETED, count=2)
    await _wait_for_lifecycle(session_id, proto.SESSION_WAITING, count=2)

    state = await session.read_session(session_id)
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert state.messages[-1].text == "wrapped up"
    updates = state.messages[-2].text
    assert 'Background subagent "alpha" finished:\n\nalpha-report' in updates
    assert 'Background subagent "beta" finished:\n\nbeta-report' in updates
    assert_message_invariants(state.messages)

    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r for r in tool_message.tool_results}
    assert set(results) == {"tc-a", "tc-b"}
    assert all(
        r.result == "Subagent is running in the background and will update you later."
        for r in results.values()
    )
    # Parent issues calls, acknowledges them, then consumes the update.
    # Each child runs once.
    assert scripted_model.call_count == 5
