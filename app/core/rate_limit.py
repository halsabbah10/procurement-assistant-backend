"""In-memory, single-instance rate limiting — a deliberate scope choice for
a demo-scale deployment (see project_architecture.md). Would need Redis if
this ever ran on more than one Render instance."""
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, per_minute: int, daily_cap: int):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._daily_total = 0
        self._day_started_at = time.monotonic()

    def _prune(self, ip: str) -> None:
        cutoff = time.monotonic() - 60
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

    def _maybe_reset_daily(self) -> None:
        if time.monotonic() - self._day_started_at > 86400:
            self._daily_total = 0
            self._day_started_at = time.monotonic()

    def check(self, ip: str) -> bool:
        self._maybe_reset_daily()
        if self._daily_total >= self.daily_cap:
            return False
        self._prune(ip)
        return len(self._requests[ip]) < self.per_minute

    def record(self, ip: str) -> None:
        self._requests[ip].append(time.monotonic())
        self._daily_total += 1
