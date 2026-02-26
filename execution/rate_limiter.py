import asyncio
import time
import logging
from contextlib import asynccontextmanager
import aiohttp

logger = logging.getLogger("arb_scanner.ratelimit")


class TokenBucketLimiter:
    def __init__(self, rps=3.0, burst=8):
        self._rps = rps
        self._original_rps = rps
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._requests = 0
        self._wait_total = 0.0
        self._429_count = 0
        self._last_429 = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._burst, self._tokens + elapsed * self._rps)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rps
                self._wait_total += wait
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0
            self._requests += 1
            if self._rps < self._original_rps and time.time() - self._last_429 > 60:
                self._rps = self._original_rps
                logger.info(f"RPS restored to {self._rps:.1f}")

    def on_429(self):
        self._429_count += 1
        self._last_429 = time.time()
        old = self._rps
        self._rps = max(0.5, self._rps * 0.5)
        logger.warning(f"429 #{self._429_count}: RPS {old:.1f} -> {self._rps:.1f}")

    def stats(self):
        return {
            "rps": self._rps,
            "requests": self._requests,
            "wait_total_s": round(self._wait_total, 2),
            "429s": self._429_count,
        }


class RateLimitedSession:
    def __init__(self, rps=3.0, burst=8, **kwargs):
        self._limiter = TokenBucketLimiter(rps, burst)
        self._session = None
        self._kwargs = kwargs

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(**self._kwargs)

    @asynccontextmanager
    async def get(self, url, **kwargs):
        await self._limiter.acquire()
        await self._ensure_session()
        async with self._session.get(url, **kwargs) as resp:
            if resp.status == 429:
                self._limiter.on_429()
            yield resp

    @asynccontextmanager
    async def post(self, url, **kwargs):
        await self._limiter.acquire()
        await self._ensure_session()
        async with self._session.post(url, **kwargs) as resp:
            if resp.status == 429:
                self._limiter.on_429()
            yield resp

    def stats(self):
        return self._limiter.stats()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
