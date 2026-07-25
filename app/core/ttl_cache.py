"""Tiny in-memory TTL cache for expensive, rarely-changing aggregate
endpoints. Same single-instance scope assumption as RateLimiter (see
app/core/rate_limit.py) — would need Redis if this ever ran on more than
one Render instance."""

import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


def async_ttl_cache(ttl_seconds: float):
    def decorator(fn: Callable[[], Awaitable[T]]) -> Callable[[], Awaitable[T]]:
        cache: dict[str, tuple[float, T]] = {}

        @wraps(fn)
        async def wrapper() -> T:
            cached = cache.get("value")
            if cached is not None:
                cached_at, value = cached
                if time.monotonic() - cached_at < ttl_seconds:
                    return value
            value = await fn()
            cache["value"] = (time.monotonic(), value)
            return value

        return wrapper

    return decorator
