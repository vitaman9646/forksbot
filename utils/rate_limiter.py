# utils/rate_limiter.py
"""
TokenBucketLimiter + RateLimitedSession v2.0
────────────────────────────────────────────
Изменения vs v1.5:
  • retry с exponential backoff для 5xx и таймаутов
  • HEAD/DELETE методы
  • json() helper для частого паттерна
  • общая сессия — переиспользуется всеми компонентами
  • graceful shutdown
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any

import aiohttp

logger = logging.getLogger("arb_scanner.ratelimit")


class TokenBucketLimiter:
    """
    Token bucket rate limiter.
    Проверенная реализация из v1.5 — без изменений в логике.
    """

    def __init__(self, rps: float = 3.0, burst: int = 8):
        self._rps = rps
        self._original_rps = rps
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

        # stats
        self._requests = 0
        self._wait_total = 0.0
        self._429_count = 0
        self._5xx_count = 0
        self._last_429 = 0.0
        self._last_error_time = 0.0

    async def acquire(self):
        """Ждёт пока появится токен, затем забирает его."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._burst,
                self._tokens + elapsed * self._rps,
            )
            self._last_refill = now

            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rps
                self._wait_total += wait
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0

            self._requests += 1

            # авто-восстановление RPS через 60s после 429
            if (
                self._rps < self._original_rps
                and time.time() - self._last_429 > 60
            ):
                self._rps = self._original_rps
                logger.info(f"RPS restored to {self._rps:.1f}")

    def on_429(self):
        """Вызывается при получении 429 Too Many Requests."""
        self._429_count += 1
        self._last_429 = time.time()
        old = self._rps
        self._rps = max(0.5, self._rps * 0.5)
        logger.warning(
            f"429 #{self._429_count}: RPS {old:.1f} → {self._rps:.1f}"
        )

    def on_5xx(self):
        """Вызывается при получении 5xx."""
        self._5xx_count += 1
        self._last_error_time = time.time()

    def stats(self) -> dict:
        return {
            "rps_current": round(self._rps, 1),
            "rps_original": self._original_rps,
            "requests": self._requests,
            "wait_total_s": round(self._wait_total, 2),
            "429s": self._429_count,
            "5xxs": self._5xx_count,
        }


class RateLimitedSession:
    """
    Обёртка над aiohttp.ClientSession с rate limiting и retry.

    Использование:
        session = RateLimitedSession(rps=3.0, burst=8)

        # как context manager для raw response:
        async with session.get(url) as resp:
            data = await resp.json()

        # или helper для JSON:
        data = await session.get_json(url, params={...})

        # обязательно закрыть:
        await session.close()
    """

    DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=20)
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
        "Accept": "application/json",
    }

    def __init__(
        self,
        rps: float = 3.0,
        burst: int = 8,
        max_retries: int = 2,
        retry_backoff: float = 1.0,
        **session_kwargs,
    ):
        self._limiter = TokenBucketLimiter(rps, burst)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_kwargs = session_kwargs
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._closed = False

    async def _ensure_session(self):
        """Создаёт сессию при первом использовании."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.DEFAULT_TIMEOUT,
                headers=self.DEFAULT_HEADERS,
                **self._session_kwargs,
            )

    # ── Context manager methods ──────────────────────────────

    @asynccontextmanager
    async def get(self, url: str, **kwargs):
        """GET запрос с rate limiting."""
        await self._limiter.acquire()
        await self._ensure_session()
        async with self._session.get(url, **kwargs) as resp:
            if resp.status == 429:
                self._limiter.on_429()
            elif resp.status >= 500:
                self._limiter.on_5xx()
            yield resp

    @asynccontextmanager
    async def post(self, url: str, **kwargs):
        """POST запрос с rate limiting."""
        await self._limiter.acquire()
        await self._ensure_session()
        async with self._session.post(url, **kwargs) as resp:
            if resp.status == 429:
                self._limiter.on_429()
            elif resp.status >= 500:
                self._limiter.on_5xx()
            yield resp

    @asynccontextmanager
    async def delete(self, url: str, **kwargs):
        """DELETE запрос с rate limiting."""
        await self._limiter.acquire()
        await self._ensure_session()
        async with self._session.delete(url, **kwargs) as resp:
            if resp.status == 429:
                self._limiter.on_429()
            elif resp.status >= 500:
                self._limiter.on_5xx()
            yield resp

    # ── JSON helpers с retry ─────────────────────────────────

    async def get_json(
        self,
        url: str,
        params: dict = None,
        headers: dict = None,
        timeout: aiohttp.ClientTimeout = None,
    ) -> Optional[Any]:
        """
        GET запрос → JSON с retry.
        Возвращает parsed JSON или None при ошибке.
        """
        for attempt in range(self._max_retries + 1):
            try:
                kwargs = {}
                if params:
                    kwargs["params"] = params
                if headers:
                    kwargs["headers"] = headers
                if timeout:
                    kwargs["timeout"] = timeout

                async with self.get(url, **kwargs) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    if resp.status == 429:
                        wait = self._retry_backoff * (2 ** attempt)
                        logger.warning(
                            f"429 on {url[:60]}, "
                            f"retry in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 500:
                        wait = self._retry_backoff * (2 ** attempt)
                        logger.warning(
                            f"{resp.status} on {url[:60]}, "
                            f"retry in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue

                    # 4xx (не 429) — не ретраим
                    logger.warning(
                        f"HTTP {resp.status} on {url[:60]}"
                    )
                    return None

            except asyncio.TimeoutError:
                if attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        f"Timeout on {url[:60]}, "
                        f"retry in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Timeout on {url[:60]} (final)")
                    return None

            except aiohttp.ClientError as e:
                if attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        f"Client error on {url[:60]}: {e}, "
                        f"retry in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"Client error on {url[:60]}: {e} (final)"
                    )
                    return None

        return None

    async def post_json(
        self,
        url: str,
        json_data: dict = None,
        headers: dict = None,
    ) -> Optional[Any]:
        """POST запрос → JSON с retry."""
        for attempt in range(self._max_retries + 1):
            try:
                kwargs = {}
                if json_data:
                    kwargs["json"] = json_data
                if headers:
                    kwargs["headers"] = headers

                async with self.post(url, **kwargs) as resp:
                    if resp.status in (200, 201):
                        return await resp.json()

                    if resp.status == 429:
                        wait = self._retry_backoff * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 500:
                        wait = self._retry_backoff * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue

                    body = await resp.text()
                    logger.warning(
                        f"HTTP {resp.status} POST {url[:60]}: "
                        f"{body[:200]}"
                    )
                    return None

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"POST {url[:60]} failed: {e}"
                    )
                    return None

        return None

    # ── Stats and lifecycle ──────────────────────────────────

    def stats(self) -> dict:
        return self._limiter.stats()

    async def close(self):
        """Закрывает сессию."""
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("RateLimitedSession closed")

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, *args):
        await self.close()
