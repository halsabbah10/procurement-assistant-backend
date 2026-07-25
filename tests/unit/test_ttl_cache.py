import asyncio

from app.core.ttl_cache import async_ttl_cache


def test_ttl_cache_returns_cached_value_within_ttl():
    calls = 0

    @async_ttl_cache(ttl_seconds=60)
    async def fetch():
        nonlocal calls
        calls += 1
        return calls

    async def run():
        first = await fetch()
        second = await fetch()
        assert first == 1
        assert second == 1  # cached, not recomputed
        assert calls == 1

    asyncio.run(run())


def test_ttl_cache_recomputes_after_expiry():
    calls = 0

    @async_ttl_cache(ttl_seconds=0.05)
    async def fetch():
        nonlocal calls
        calls += 1
        return calls

    async def run():
        first = await fetch()
        await asyncio.sleep(0.1)
        second = await fetch()
        assert first == 1
        assert second == 2
        assert calls == 2

    asyncio.run(run())
