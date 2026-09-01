from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, Protocol, TypeVar, cast

import vercel.workflow
from vercel.workflow._internal import core

HookFunction = TypeVar("HookFunction", bound=Callable[..., Awaitable[None]])
HookToken = Callable[[Any], str]
P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)

_HOOK_TYPE = "__workflow_hook_type__"
_HOOK_TOKEN = "__workflow_hook_token__"
_WORKFLOW_INIT = "__workflow_init__"


def hook(
    hook_type: type[vercel.workflow.BaseHook],
    token: HookToken | None = None,
) -> Callable[[HookFunction], HookFunction]:
    """Mark a hook handler, optionally setting its token from the instance."""
    if not inspect.isclass(hook_type) or not issubclass(
        hook_type, vercel.workflow.BaseHook
    ):
        raise TypeError("hook() expects a BaseHook subclass")
    if token is not None and not callable(token):
        raise TypeError("hook token must be a callable")

    def decorate(func: HookFunction) -> HookFunction:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("hook handlers must be async functions")
        setattr(func, _HOOK_TYPE, hook_type)
        setattr(func, _HOOK_TOKEN, token)
        return cast(HookFunction, func)

    return decorate


def init[F: Callable[..., None]](func: F) -> F:
    """Mark the real initializer to receive the workflow arguments."""
    if func.__name__ != "__init__":  # ty: ignore[unresolved-attribute]
        raise TypeError("@init may only decorate __init__")
    if inspect.iscoroutinefunction(func):
        raise TypeError("@init must decorate a synchronous function")
    setattr(func, _WORKFLOW_INIT, True)
    return func


def hook_token(run_id: str, hook_name: str) -> str:
    return run_id + "$signal$" + hook_name


def _hook_methods(
    cls: type[Any],
) -> tuple[
    tuple[
        str,
        type[vercel.workflow.BaseHook],
        HookToken | None,
    ],
    ...,
]:
    methods: list[
        tuple[
            str,
            type[vercel.workflow.BaseHook],
            HookToken | None,
        ]
    ] = []
    seen: set[str] = set()
    for base in cls.__mro__:
        for name, member in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            hook_type = getattr(member, _HOOK_TYPE, None)
            if hook_type is not None:
                methods.append(
                    (
                        name,
                        cast(type[vercel.workflow.BaseHook], hook_type),
                        cast(
                            HookToken | None,
                            getattr(member, _HOOK_TOKEN, None),
                        ),
                    )
                )
    return tuple(methods)


class WorkflowClass(Protocol[P, R_co]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R_co]: ...


def workflow_class[**Params, Result](
    registry: vercel.workflow.Workflows,
    cls: Callable[..., WorkflowClass[Params, Result]],
) -> core.Workflow[Params, Result]:
    """Register a class using run as its typed workflow entrypoint."""
    constructor = cls
    if not inspect.isclass(cls):
        raise TypeError("workflow_class() expects a class")

    class_type = cls
    prototype = cast(WorkflowClass[Params, Result], object.__new__(class_type))
    bound_run = prototype.run
    init_accepts_workflow_args = hasattr(class_type.__init__, _WORKFLOW_INIT)
    hook_methods = _hook_methods(class_type)

    async def listen_for_hook(
        instance: WorkflowClass[Params, Result],
        func_name: str,
        hook_name: str,
        hook_type: type[vercel.workflow.BaseHook],
        token: str,
        active_hooks: list[vercel.workflow.HookEvent[Any]],
    ) -> None:
        events = hook_type.wait(
            token=token,
            metadata={
                "kind": "hook",
                "handler": func_name,
                "name": hook_name,
            },
        )
        active_hooks.append(events)
        async for payload in events:
            handler = cast(
                Callable[[vercel.workflow.BaseHook], Awaitable[None]],
                getattr(instance, func_name),
            )
            await handler(payload)

    @functools.wraps(bound_run)
    async def generated_workflow(*args: Params.args, **kwargs: Params.kwargs) -> Result:
        instance = (
            constructor(*args, **kwargs)
            if init_accepts_workflow_args
            else constructor()
        )
        run_id = vercel.workflow.get_workflow_metadata().run_id
        active_hooks: list[vercel.workflow.HookEvent[Any]] = []

        hook_tasks: list[asyncio.Task[None]] = []
        async with asyncio.TaskGroup() as tasks:
            for func_name, hook_type, token_for in hook_methods:
                token = (
                    token_for(instance)
                    if token_for is not None
                    else hook_token(run_id, func_name)
                )
                if not isinstance(token, str):
                    raise TypeError(f"hook token for {func_name} must be a string")
                hook_tasks.append(
                    tasks.create_task(
                        listen_for_hook(
                            instance,
                            func_name,
                            func_name,
                            hook_type,
                            token,
                            active_hooks,
                        ),
                        name=token,
                    )
                )

            result = await instance.run(*args, **kwargs)
            for events in active_hooks:
                events.dispose()
            for task in hook_tasks:
                task.cancel()

        return result

    generated_workflow.__name__ = "workflow"
    generated_workflow.__qualname__ = f"{class_type.__qualname__}.workflow"
    registered = registry.workflow(generated_workflow)
    class_type.workflow = registered  # type: ignore
    return registered
