"""End-to-end turn tests: the real durable engine, in-process. The only test
double is the scripted model (``scripted_model``)."""

from __future__ import annotations

import base64
import os

import ai
import ai.types.messages as messages_
import pytest
from conftest import MockProvider, assert_message_invariants, text_msg, tool_call_msg
from harness import (
    InProcessWorld,
)
from harness import (
    lifecycle as _lifecycle,
)
from harness import (
    resume_approval as _resume_approval,
)
from harness import (
    start_turn as _start,
)
from harness import (
    wait_for_lifecycle as _wait_for_lifecycle,
)
from harness import (
    wait_run as _wait_run,
)

from agent import proto, session, storage

# --- message round trips ----------------------------------------------------------


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


def test_round_trip_of_plain_message_stays_plain() -> None:
    message = _assistant_with_tool_calls()

    restored = ai.types.messages.Message.model_validate(message.model_dump(mode="json"))

    assert restored.replay is False
    assert all(part.cached_result is None for part in restored.tool_calls)
    assert restored.model_dump(mode="json") == message.model_dump(mode="json")


# --- the durable turn on the real engine -------------------------------------------


async def test_single_turn_settles_and_session_waits(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    run = await _start("s1", "hi")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    assert state.turn_index == 0
    assert [m.role for m in state.messages] == ["system", "user", "assistant"]
    assert state.messages[-1].text == "hello there"
    assert_message_invariants(state.messages)

    output = proto.TurnOutput.model_validate(await _wait_run(run))
    assert output.kind == "suspend"
    assert output.messages[-1].text == "hello there"

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]
    # the session stays open for the next turn; nothing is left running.
    _, closed = await storage.store().info("s1", "default")
    assert not closed


async def test_next_turn_appends_user_message_without_duplicating_history(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first answer")], [text_msg("second answer")]]

    await _start("s1", "one")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    await _start("s1", "two")
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING, count=2)

    state = await session.read_session("s1")
    assert state is not None
    assert state.turn_index == 1
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

    # one session opener; each turn wrote its own boundaries.
    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
        proto.TURN_STARTED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]


async def test_error_turn_completes_the_session(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    # no responses scripted: every model call raises, llm_step exhausts its
    # retries, and the turn settles as an error instead of failing the run.
    scripted_model.responses = []

    run = await _start("s1", "hi")
    output = proto.TurnOutput.model_validate(await _wait_run(run))
    assert output.kind == "error"
    assert output.error is not None

    lifecycle_events = await _lifecycle("s1")
    assert lifecycle_events[-2:] == [proto.TURN_COMPLETED, proto.SESSION_COMPLETED]
    _, closed = await storage.store().info("s1", "default")
    assert closed


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
    # the turn parks on the approval and emits tool_approval.requested; the
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


async def test_stale_approval_is_ignored(
    world: InProcessWorld, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-1",
                name="bash",
                args='{"command": "echo ok"}',
                text="running it",
            )
        ],
        [text_msg("done")],
    ]

    await _start("s1", "run it")
    await _wait_for_lifecycle("s1", proto.TOOL_APPROVAL_REQUESTED)

    # an approval for a call this turn never made pre-registers harmlessly;
    # the real one still unblocks the tool.
    await _resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-unknown", granted=False)
    )
    await _resume_approval(
        "s1", proto.ToolApprovalResponse(tool_call_id="tc-1", granted=True)
    )
    await _wait_for_lifecycle("s1", proto.SESSION_WAITING)

    state = await session.read_session("s1")
    assert state is not None
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.result == "ok\n"
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
    # both gated calls park before the turn parks.
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


async def test_subagent_result_lands_on_the_trailing_tool_message(
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
        [text_msg("child answer")],  # the child session's single turn
        [text_msg("final answer")],  # the parent's follow-up llm call
    ]

    await _start("s1", "delegate")
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
    assert_message_invariants(state.messages)
    assert state.messages[-1].text == "final answer"

    # the child's full transcript (a MessageBundle) is the tool result
    [tool_message] = [m for m in state.messages if m.role == "tool"]
    [result] = tool_message.tool_results
    assert result.tool_call_id == "tc-sub"
    bundle = ai.agents.MessageBundle.model_validate(result.result)
    assert [m.role for m in bundle.messages] == ["assistant"]
    assert bundle.messages[-1].text == "child answer"

    assert await _lifecycle("s1") == [
        proto.SESSION_STARTED,
        proto.TURN_STARTED,
        proto.SUBAGENT_CALLED,
        proto.SUBAGENT_COMPLETED,
        proto.TURN_COMPLETED,
        proto.SESSION_WAITING,
    ]
    # the child ran as a single ungated turn on its own stream (no session
    # lifecycle, no snapshot)
    assert await _lifecycle("s1:child:tc-sub") == []
    assert await session.read_session("s1:child:tc-sub") is None
    assert scripted_model.call_count == 3


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
        [text_msg("wrapped up")],  # parent's follow-up llm call after both children
    ]
    scripted_model.keyed_responses = {
        "task-alpha": [text_msg("alpha-report")],
        "task-beta": [text_msg("beta-report")],
    }

    await _start(session_id, "delegate both")
    await _wait_for_lifecycle(session_id, proto.SESSION_WAITING)

    state = await session.read_session(session_id)
    assert state is not None
    assert [m.role for m in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert state.messages[-1].text == "wrapped up"
    assert_message_invariants(state.messages)

    [tool_message] = [m for m in state.messages if m.role == "tool"]
    results = {r.tool_call_id: r for r in tool_message.tool_results}
    assert set(results) == {"tc-a", "tc-b"}
    bundle_a = ai.agents.MessageBundle.model_validate(results["tc-a"].result)
    bundle_b = ai.agents.MessageBundle.model_validate(results["tc-b"].result)
    assert bundle_a.messages[-1].text == "alpha-report"
    assert bundle_b.messages[-1].text == "beta-report"
    # parent: 1 llm call issuing both + 1 follow-up; each child: 1 call
    assert scripted_model.call_count == 4
