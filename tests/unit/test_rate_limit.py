import time

from app.core.rate_limit import RateLimiter


def test_rate_limiter_allows_requests_under_limit():
    limiter = RateLimiter(per_minute=3, daily_cap=100)
    ip = "1.2.3.4"
    for _ in range(3):
        assert limiter.check(ip) is True
        limiter.record(ip)


def test_rate_limiter_blocks_requests_over_per_minute_limit():
    limiter = RateLimiter(per_minute=2, daily_cap=100)
    ip = "1.2.3.4"
    limiter.record(ip)
    limiter.record(ip)
    assert limiter.check(ip) is False


def test_rate_limiter_daily_cap_blocks_all_ips_once_tripped():
    limiter = RateLimiter(per_minute=1000, daily_cap=2)
    limiter.record("1.1.1.1")
    limiter.record("2.2.2.2")
    assert limiter.check("3.3.3.3") is False


def test_rate_limiter_window_resets_after_60_seconds(monkeypatch):
    limiter = RateLimiter(per_minute=1, daily_cap=100)
    ip = "1.2.3.4"
    limiter.record(ip)
    assert limiter.check(ip) is False

    real_time = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_time() + 61)
    assert limiter.check(ip) is True
