import asyncio
import contextlib
import functools
import inspect
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from typing import Any, overload


async def hook_retries(
    retry_count: int = 40, sleep_seconds: float = 0.05
) -> AsyncIterator[bool]:
    """Yield hook attempts, marking the last one, with a delay between retries."""
    if retry_count < 1:
        raise ValueError("retry_count must be positive")
    for attempt in range(retry_count):
        if attempt:
            await asyncio.sleep(sleep_seconds)
        yield attempt == retry_count - 1


@contextlib.contextmanager
def log_traceback(message: str) -> Iterator[None]:
    # prints the traceback of any exception, labeled with message, and re-raises
    try:
        yield
    except Exception:
        print(f"[seal] {message}:\n{traceback.format_exc()}", flush=True)
        raise


@overload
def print_traceback[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Coroutine[Any, Any, R]]: ...
@overload
def print_traceback[**P, R](func: Callable[P, R]) -> Callable[P, R]: ...
def print_traceback(func: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_traceback(
                f"{getattr(func, '__name__', type(func).__name__)} failed"
            ):
                return await func(*args, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with log_traceback(f"{getattr(func, '__name__', type(func).__name__)} failed"):
            return func(*args, **kwargs)

    return wrapper
