from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import contextvars
import functools
from typing import Any

import pydantic

from agent import storage, workflow

CONTROL_NAMESPACE = "control"
CONTROL_POLL_INTERVAL = 0.05

cancellation_context: contextvars.ContextVar[tuple[str, int]] = contextvars.ContextVar(
    "cancellation_context"
)


class InterruptSignal(pydantic.BaseModel):
    kind: str = "interrupt"
    turn_index: int


class StepInterrupted(Exception):
    pass


async def interrupt(session_id: str, turn_index: int) -> int:
    await storage.ensure_ready()
    return await storage.store().append(
        session_id,
        CONTROL_NAMESPACE,
        InterruptSignal(turn_index=turn_index).model_dump(mode="json"),
    )


async def wait_for_interrupt(session_id: str, turn_index: int) -> None:
    await storage.ensure_ready()
    backend = storage.store()
    next_index = 0
    while True:
        records = await backend.read(session_id, CONTROL_NAMESPACE, next_index)
        for index, data in records:
            next_index = index + 1
            signal = InterruptSignal.model_validate(data)
            if signal.turn_index == turn_index:
                return
        await asyncio.sleep(CONTROL_POLL_INTERVAL)


def cancellable_step(
    func: collections.abc.Callable[..., collections.abc.Coroutine[Any, Any, Any]],
) -> collections.abc.Callable[..., collections.abc.Coroutine[Any, Any, Any]]:
    async def run_step(
        *args: Any,
        _cancel_session_id: str,
        _cancel_turn_index: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body = asyncio.create_task(func(*args, **kwargs))
        watcher = asyncio.create_task(
            wait_for_interrupt(_cancel_session_id, _cancel_turn_index)
        )
        try:
            done, _ = await asyncio.wait(
                {body, watcher}, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                body.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await body
                return {"interrupted": True, "value": None}

            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            return {"interrupted": False, "value": await body}
        finally:
            for task in (body, watcher):
                if not task.done():
                    task.cancel()

    # Workflow argument binding must see the private cancellation arguments.
    # Copy only the registration identity; functools.wraps would hide them.
    run_step.__module__ = func.__module__
    run_step.__name__ = getattr(func, "__name__", "cancellable_step")
    run_step.__qualname__ = getattr(func, "__qualname__", run_step.__name__)
    durable_step = workflow.step(max_retries=0)(run_step)

    @functools.wraps(func)
    async def dispatch(*args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id")
        turn_index = kwargs.get("turn_index")
        if not isinstance(session_id, str) or not isinstance(turn_index, int):
            try:
                session_id, turn_index = cancellation_context.get()
            except LookupError:
                raise TypeError(
                    "cancellable steps require session_id and turn_index keyword "
                    "arguments or a cancellation context"
                ) from None

        result = await durable_step(
            *args,
            _cancel_session_id=session_id,
            _cancel_turn_index=turn_index,
            **kwargs,
        )
        if result["interrupted"]:
            raise StepInterrupted
        return result["value"]

    async def run_locally(*args: Any, **kwargs: Any) -> Any:
        try:
            session_id, turn_index = cancellation_context.get()
        except LookupError:
            raise TypeError(
                "local cancellable step requires a cancellation context"
            ) from None
        result = await run_step(
            *args,
            _cancel_session_id=session_id,
            _cancel_turn_index=turn_index,
            **kwargs,
        )
        if result["interrupted"]:
            raise StepInterrupted
        return result["value"]

    # Match workflow Step's useful local testing seam.
    setattr(dispatch, "func", run_locally)  # noqa: B010
    return dispatch
