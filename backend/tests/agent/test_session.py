"""The agent loop as durable-process arms.

These replace the workflow port's ``test_driver.py``/``test_turn.py``. There
is no bridge harness: ``rt.drain()`` runs the real engine (claims, commits,
verdict facts, hooks) against an in-memory store with a virtual clock.
"""

from __future__ import annotations

import ai
import pytest
from conftest import (
    MockProvider,
    assert_message_invariants,
    drain_until_idle,
    text_msg,
    tool_call_msg,
)
from rotor.testing import LocalRuntime

from agent import proto
from agent.processes import Session


async def _start(rt: LocalRuntime, session_id: str, prompt: str) -> None:
    await rt.client.start(Session, input=prompt, id=session_id, scope=session_id)


async def _history(rt: LocalRuntime, session_id: str) -> list[ai.messages.Message]:
    value, _revision = await rt.client.query(session_id, Session.history)
    return [ai.messages.Message.model_validate(m) for m in value]


async def _approval_records(rt: LocalRuntime, session_id: str) -> list[dict]:
    return [
        event.data
        async for event in rt.client.tail(process_id=session_id, follow=False)
        if event.kind == proto.APPROVAL_RECORD
    ]


async def test_direct_answer(rt: LocalRuntime, scripted_model: MockProvider) -> None:
    scripted_model.responses = [[text_msg("hello there")]]

    await _start(rt, "s1", "hi")
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    assert history[-1].role == "assistant"
    assert history[-1].text == "hello there"

    snap = await rt.client.snapshot("s1")
    assert snap.phase == "idle"  # waiting costs nothing: one row, no lease


async def test_followup_message_runs_a_second_turn(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [[text_msg("first")], [text_msg("second")]]

    await _start(rt, "s1", "one")
    await drain_until_idle(rt)
    await rt.client.send("s1", proto.UserMessage(text="two"))
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    assert [m.text for m in history if m.role == "assistant"] == ["first", "second"]
    assert scripted_model.call_count == 2


async def test_gated_bash_parks_then_runs_on_grant(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [tool_call_msg(tc_id="tc-1", name="bash", args='{"command": "echo hi"}')],
        [text_msg("done")],
    ]

    await _start(rt, "s1", "run echo")
    await drain_until_idle(rt)

    # the turn is parked on a durable hook; the process holds no lease
    snap = await rt.client.snapshot("s1")
    assert snap.phase == "idle"
    assert snap.pending_hooks
    assert scripted_model.call_count == 1

    [minted] = await _approval_records(rt, "s1")
    assert minted["tool_call_id"] == "tc-1"
    await rt.client.resolve_hook(
        minted["hook_id"],
        proto.Approval(granted=True, reason="ok"),
        minted["token"],
    )
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    [result] = [p for m in history for p in m.tool_results]
    assert result.tool_call_id == "tc-1"
    assert "hi" in str(result.result)
    assert history[-1].text == "done"


async def test_denied_bash_becomes_an_error_result(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    scripted_model.responses = [
        [tool_call_msg(tc_id="tc-1", name="bash", args='{"command": "rm -rf /"}')],
        [text_msg("understood, not running it")],
    ]

    await _start(rt, "s1", "run something scary")
    await drain_until_idle(rt)
    [minted] = await _approval_records(rt, "s1")
    await rt.client.resolve_hook(
        minted["hook_id"],
        proto.Approval(granted=False, reason="no thanks"),
        minted["token"],
    )
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    [result] = [p for m in history for p in m.tool_results]
    assert result.is_error
    assert "no thanks" in str(result.result)


async def test_wrong_token_cannot_resolve_the_gate(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    import rotor

    scripted_model.responses = [
        [tool_call_msg(tc_id="tc-1", name="bash", args='{"command": "echo hi"}')],
    ]
    await _start(rt, "s1", "run echo")
    await drain_until_idle(rt)
    [minted] = await _approval_records(rt, "s1")
    with pytest.raises(rotor.HookError):
        await rt.client.resolve_hook(
            minted["hook_id"],
            proto.Approval(granted=True),
            "forged-token",
        )


async def test_subagent_transcript_lands_on_the_parent_call(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    scripted_model.keyed_responses = {
        "count the files": [text_msg("there are 3 files")],
    }
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-sub",
                name="subagent",
                args='{"prompt": "count the files", "name": "counter"}',
            )
        ],
        [text_msg("the subagent says: 3 files")],
    ]

    await _start(rt, "s1", "delegate the counting")
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    [result] = [p for m in history for p in m.tool_results]
    assert result.tool_call_id == "tc-sub"
    assert "3 files" in str(result.result)
    assert history[-1].text == "the subagent says: 3 files"


async def test_prompt_sent_mid_turn_queues_for_the_next_turn(
    rt: LocalRuntime, scripted_model: MockProvider
) -> None:
    # park the turn on an approval, then send a prompt while it is parked
    scripted_model.responses = [
        [tool_call_msg(tc_id="tc-1", name="bash", args='{"command": "echo hi"}')],
        [text_msg("done with both")],
    ]
    await _start(rt, "s1", "run echo")
    await drain_until_idle(rt)

    await rt.client.send("s1", proto.UserMessage(text="also, hello"))
    await drain_until_idle(rt)

    [minted] = await _approval_records(rt, "s1")
    await rt.client.resolve_hook(
        minted["hook_id"], proto.Approval(granted=True), minted["token"]
    )
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    # the queued prompt landed after the tool results, keeping the
    # call/result adjacency the model requires
    roles = [m.role for m in history]
    assert roles.index("tool") < roles.index("user", roles.index("tool"))
    assert history[-1].text == "done with both"


async def test_eager_tool_runs_inside_the_turn(
    rt: LocalRuntime,
    scripted_model: MockProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(call: dict) -> dict:
        return proto.tool_result(call, "fetched: ok")

    import agent.processes as processes

    monkeypatch.setattr(processes, "execute_tool", fake_execute)
    scripted_model.responses = [
        [
            tool_call_msg(
                tc_id="tc-f", name="web_fetch", args='{"url": "http://x.test"}'
            )
        ],
        [text_msg("summarized")],
    ]

    await _start(rt, "s1", "fetch it")
    await drain_until_idle(rt)

    history = await _history(rt, "s1")
    assert_message_invariants(history)
    [result] = [p for m in history for p in m.tool_results]
    assert str(result.result) == "fetched: ok"
    # no run_tool child existed: the eager call answered inside the turn
    assert history[-1].text == "summarized"
