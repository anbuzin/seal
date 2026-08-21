from __future__ import annotations

import asyncio
import collections.abc
import datetime
from typing import Literal

import pydantic

from agent import proto, storage

__all__ = ["get_readable", "get_writable"]


# lifecycle event constructors


def session_started() -> proto.LifecycleEvent:
    return proto.LifecycleEvent(type=proto.SESSION_STARTED)


def session_waiting(*, turn_index: int) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SESSION_WAITING, data={"turn_index": turn_index}
    )


def session_completed(*, is_error: bool = False) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SESSION_COMPLETED, data={"is_error": is_error}
    )


def turn_started(*, turn_index: int) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.TURN_STARTED, data={"turn_index": turn_index}
    )


def turn_completed(*, turn_index: int, kind: str) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.TURN_COMPLETED, data={"turn_index": turn_index, "kind": kind}
    )


def subagent_called(
    *, tool_call_id: str, child_session_id: str, name: str
) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SUBAGENT_CALLED,
        data={
            "tool_call_id": tool_call_id,
            "child_session_id": child_session_id,
            "name": name,
        },
    )


def subagent_completed(*, tool_call_id: str, is_error: bool) -> proto.LifecycleEvent:
    return proto.LifecycleEvent(
        type=proto.SUBAGENT_COMPLETED,
        data={"tool_call_id": tool_call_id, "is_error": is_error},
    )


def tool_approval_requested(*, turn_index: int) -> proto.LifecycleEvent:
    # the turn parked: every scheduled tool has finished or is awaiting approval.
    return proto.LifecycleEvent(
        type=proto.TOOL_APPROVAL_REQUESTED, data={"turn_index": turn_index}
    )


def reload_requested() -> proto.LifecycleEvent:
    # a retried step just wiped its own aborted first attempt's partial
    # output; a client that already streamed that output needs to reload
    # rather than trust it.
    return proto.LifecycleEvent(type=proto.RELOAD_REQUESTED)


DEFAULT_STREAM_NAMESPACE = "default"
DEFAULT_STREAM_POLL_INTERVAL = 0.05


class WritableStreamHandle(pydantic.BaseModel):
    type: Literal["seal.durable_agent.writable_stream"] = (
        "seal.durable_agent.writable_stream"
    )
    stream_id: str
    namespace: str = DEFAULT_STREAM_NAMESPACE

    async def write(self, event: proto.StreamEvent) -> int:
        data = event.model_dump(mode="json")

        # stamp the dump, not the caller's event
        if isinstance(event, proto.LifecycleEvent) and event.at is None:
            data["at"] = datetime.datetime.now(datetime.UTC).isoformat()
        return await storage.store().append(self.stream_id, self.namespace, data)

    async def close(self) -> None:
        await storage.store().close(self.stream_id, self.namespace)


async def get_writable(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
) -> WritableStreamHandle:
    """Return the writable handle for ``stream_id``."""
    await storage.ensure_ready()
    return WritableStreamHandle(
        stream_id=stream_id,
        namespace=namespace,
    )


async def get_readable(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
    start_index: int = 0,
    poll_interval: float = DEFAULT_STREAM_POLL_INTERVAL,
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    """Yield stream events from ``start_index`` until the stream is closed."""
    if poll_interval < 0:
        raise ValueError("poll_interval must be non-negative")

    await storage.ensure_ready()
    backend = storage.store()

    next_index = start_index
    if next_index < 0:
        tail_index, _ = await backend.info(stream_id, namespace)
        next_index = max(0, tail_index + 1 + next_index)

    while True:
        records = await backend.read(stream_id, namespace, next_index)
        for index, data in records:
            next_index = index + 1
            yield proto.STREAM_EVENT_ADAPTER.validate_python(data)

        if records:
            continue

        _, closed = await backend.info(stream_id, namespace)
        if closed:
            # drain anything written between the read and the close check.
            for index, data in await backend.read(stream_id, namespace, next_index):
                next_index = index + 1
                yield proto.STREAM_EVENT_ADAPTER.validate_python(data)
            return

        await asyncio.sleep(poll_interval)


async def tail_index(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
) -> int:
    """Return the last written index (``-1`` when the stream is empty)."""
    await storage.ensure_ready()
    index, _ = await storage.store().info(stream_id, namespace)
    return index


async def replay(
    stream_id: str,
    *,
    namespace: str = DEFAULT_STREAM_NAMESPACE,
    start_index: int = 0,
) -> collections.abc.AsyncIterator[proto.StreamEvent]:
    """Yield already-written events once, without tailing for new ones."""
    await storage.ensure_ready()
    for _, data in await storage.store().read(stream_id, namespace, start_index):
        yield proto.STREAM_EVENT_ADAPTER.validate_python(data)


# current design implies that get_writable gets called inside a step with session_id
# if there's encryption involved, writable handle will have to be passed all the way
# down session -> step -> turn -> step, in order for the turn step (e.g. llm call) to
# write to session's stream.
