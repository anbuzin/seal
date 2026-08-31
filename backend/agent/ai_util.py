import asyncio
from collections.abc import AsyncIterable, Callable, Collection
from typing import Any, Self

import ai


class TrackingToolRunner(ai.ToolRunner):
    def __init__(self) -> None:
        super().__init__()
        self.id_to_task: dict[str, asyncio.Task[ai.events.ToolCallResult]] = {}
        self.task_to_id: dict[asyncio.Task[ai.events.ToolCallResult], str] = {}

    def discard(self, task: asyncio.Task[ai.events.ToolCallResult]) -> None:
        if task in self.task_to_id:
            del self.id_to_task[self.task_to_id[task]]
            del self.task_to_id[task]
        super().discard(task)

    def discard_except(self, ok_calls: Collection[ai.ToolCall]) -> None:
        ok_ids = {tc.id for tc in ok_calls}
        for id, task in list(self.id_to_task.items()):
            if id not in ok_ids:
                self.discard(task)

    def schedule(
        self, tc: ai.agents.agent.ToolCallCallable
    ) -> asyncio.Task[ai.events.ToolCallResult]:
        id = getattr(tc, "id", None)
        if id and id in self.id_to_task:
            return self.id_to_task[id]

        task = super().schedule(tc)
        if id:
            self.id_to_task[id] = task
            self.task_to_id[task] = id
        return task


class SpeculativeToolRunner(TrackingToolRunner):
    def __init__(
        self,
        *,
        eager_tools: Collection[str] = (),
        resolver: Callable[[ai.messages.ToolCallPart], ai.ToolCall],
        tool_stream: AsyncIterable[ai.messages.ToolCallPart],
    ) -> None:
        super().__init__()
        self.eager_tools = frozenset(eager_tools)
        self.resolver = resolver
        self.tool_stream = tool_stream

    async def __aenter__(self) -> Self:
        res = await super().__aenter__()
        self.worker = asyncio.create_task(self.watcher())
        return res

    async def __aexit__(self, *args: Any) -> None:
        self.worker.cancel()
        return await super().__aexit__(*args)

    async def watcher(self) -> None:
        async for tool_call in self.tool_stream:
            if tool_call.tool_name in self.eager_tools:
                self.schedule(self.resolver(tool_call))
