import pytest

from app.db import client as db_client


@pytest.fixture(autouse=True)
def _reset_cached_motor_client():
    """app/db/client.py's get_database()/get_checkpoint_database() are
    backed by an @lru_cache'd AsyncIOMotorClient — a deliberate production
    singleton bound to the app's one running event loop. Pytest-asyncio
    gives each test function its own fresh event loop by default, so a
    Motor client cached from an earlier test is bound to a now-closed loop
    and raises "RuntimeError: Event loop is closed" the moment a later test
    tries to use it. Clearing the cache after every test forces a fresh
    client (and thus a fresh loop binding) per test, matching the
    per-test-function loop pytest-asyncio actually gives us."""
    yield
    db_client._get_client.cache_clear()
