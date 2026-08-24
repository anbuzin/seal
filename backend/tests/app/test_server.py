"""The inbound boundary ``POST /chat`` branches on.

``server.post_chat`` decides between "resume a parked approval" and "start a
new turn" with::

    is_approval_resume = bool(approvals) and not (
        messages and messages[-1].role == "user")

after running the request history through ``ai_sdk.to_messages``. These tests
pin the SDK outputs that decision relies on, with the exact UIMessage shapes
the browser sends (``sendAutomaticallyWhen`` resubmits the full history after
an approval response).
"""

from __future__ import annotations

import asyncio
from typing import Any

import ai.ui.ai_sdk as ai_sdk
import pytest
import vercel.workflow

from app import server


def _ui(role: str, *parts: dict[str, Any], id: str = "m1") -> ai_sdk.UIMessage:
    return ai_sdk.UIMessage.model_validate(
        {"id": id, "role": role, "parts": list(parts)}
    )


def _answered_approval(granted: bool = True) -> dict[str, Any]:
    return {
        "type": "tool-bash",
        "toolCallId": "tc-1",
        "state": "approval-responded",
        "input": {"command": "ls"},
        "approval": {"id": "approve_tc-1", "approved": granted, "reason": "ok"},
    }


def test_approval_resubmission_resumes_the_parked_turn() -> None:
    # trailing message is the assistant turn holding the answered approval
    messages, approvals = ai_sdk.to_messages(
        [
            _ui("user", {"type": "text", "text": "run ls"}, id="u1"),
            _ui("assistant", _answered_approval(), id="a1"),
        ]
    )
    assert messages[-1].role != "user"
    assert [a.tool_call_id for a in approvals] == ["tc-1"]
    assert approvals[0].granted is True
    assert approvals[0].reason == "ok"


def test_trailing_user_message_starts_a_new_turn_even_with_past_approvals() -> None:
    messages, approvals = ai_sdk.to_messages(
        [
            _ui("user", {"type": "text", "text": "run ls"}, id="u1"),
            _ui("assistant", _answered_approval(), id="a1"),
            _ui("user", {"type": "text", "text": "now do more"}, id="u2"),
        ]
    )
    assert messages[-1].role == "user"
    assert messages[-1].text == "now do more"


async def test_create_session_waits_for_initial_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Run:
        run_id = "session-1"

        async def status(self) -> str:
            return "pending"

    async def start(*args: object, **kwargs: object) -> Run:
        return Run()

    attempts = 0

    async def get_hook_by_token(token: str) -> object:
        nonlocal attempts
        assert token == "seal-session:session-1:0"
        attempts += 1
        if attempts <= 40:
            raise vercel.workflow.HookNotFoundError
        return object()

    async def sleep(delay: float) -> None:
        assert delay == 0.05

    monkeypatch.setattr(vercel.workflow, "start", start)
    monkeypatch.setattr(vercel.workflow, "get_hook_by_token", get_hook_by_token)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    created = await server.create_session(server.CreateSessionRequest())

    assert created.id == "session-1"
    assert attempts == 41
