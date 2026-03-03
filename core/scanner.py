# core/scanner.py
"""
ForkScanner v2 — сканер с реальными стаканами.

Этапы:
1. Получаем события из Gamma API (как раньше)
2. Фильтруем по negRisk / title
3. Для кандидатов запрашиваем РЕАЛЬНЫЙ orderbook из CLOB
4. Считаем edge с учётом глубины стакана и slippage
5. Возвращаем только реально исполнимые вилки
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("arb_scanner.scanner")

# ── Константы ─────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com/events"
CLOB_API = "https://clob.polymarket.com"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# Polymarket fee: 2% на выигрышную сторону
POLY_FEE_PCT = 0.02

# Фильтры реалистичности
MAX_SUM_DEVIATION = 0.08    # sum не дальше 8% от 1.0
MIN_SUM_DEVIATION = 0.005   # меньше 0.5% — не вилка

# Anti-fake: edge больше 15% — точно ошибка данных
MAX_REALISTIC_EDGE = 15.0


# ══════════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class OrderBookLevel:
    """Один уровень стакана."""
    price: float
    size: float  # в shares


@dataclass
class OutcomeBook:
    """Один исход с реальным стаканом."""
    question: str
    token_id: str
    market_id: str
    volume_24h: float

    # mid-price из Gamma (для быстрого фильтра)
    mid_price: float

    # реальный стакан (заполняется позже)
    asks: List[OrderBookLevel] = field(default_factory=list)
    bids: List[OrderBookLevel] = field(default_factory=list)

    # рассчитанные поля
    best_ask: Optional[float] = None
    best_bid: Optional[float] = None
    ask_depth_usd: float = 0.0  # сколько $ можно купить

    def has_book(self) -> bool:
        return len(self.asks) > 0

    def cost_to_buy(self, amount_usd: float) -> Optional[dict]:
        """
        Считает РЕАЛЬНУЮ стоимость покупки на $amount_usd
        с учётом глубины стакана.

        Идём по ask levels от лучшего к худшему,
        "съедая" ликвидность.
        """
        if not self.asks:
            return None

        sorted_asks = sorted(self.asks, key=lambda x: x.price)

        total_cost = 0.0
        shares_acquired = 0.0
        levels_consumed = 0
        remaining = amount_usd

        for level in sorted_asks:
            level_value = level.size * level.price
            if level_value <= remaining:
                # забираем весь уровень
                total_cost += level_value
                shares_acquired += level.size
                remaining -= level_value
                levels_consumed += 1
            else:
                # частично забираем этот уровень
                shares_at_level = remaining / level.price
                total_cost += remaining
                shares_acquired += shares_at_level
                remaining = 0
                levels_consumed += 1
                break

        if shares_acquired == 0:
            return None

        return {
            "total_cost": total_cost,
            "shares": shares_acquired,
            "avg_price": total_cost / shares_acquired,
            "levels_consumed": levels_consumed,
            "fully_filled": remaining < 0.01,
            "unfilled_usd": remaining,
        }


@dataclass
class ForkCandidate:
    """Кандидат на вилку (до проверки стакана)."""
    event_title: str
    event_id: str
    is_neg_risk: bool
    verification: str  # "neg_risk" / "title"
    outcomes: List[OutcomeBook] = field(default_factory=list)

    # быстрая оценка по mid-price
    mid_sum: float = 0.0
    mid_deviation_pct: float = 0.0


@dataclass
class RealFork:
    """Подтверждённая вилка с реальными данными стакана."""
    event_title: str
    event_id: str
    fork_type: str  # "under" / "over"
    is_neg_risk: bool
    verification: str

    outcomes: List[OutcomeBook] = field(default_factory=list)
    entry_size_usd: float = 0.0

    # mid-price данные
    mid_sum: float = 0.0

    # реальные данные стакана
    real_sum: float = 0.0              # сумма avg_price при входе
    real_avg_prices: List[float] = field(default_factory=list)

    # edge расчёты
    gross_edge_pct: float = 0.0        # до комиссий
    fee_cost_pct: float = 0.0          # комиссия в %
    net_edge_pct: float = 0.0          # после комиссий
    slippage_pct: float = 0.0          # разница mid vs real

    # ликвидность
    min_depth_usd: float = 0.0         # мин. глубина среди исходов
    all_filled: bool = False           # все ноги полностью заполнены

    # валидация
    is_valid: bool = False
    reject_reason: str = ""

    # token_ids для исполнения
    token_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        outcomes_str = "\n".join(
            f"    {o.question[:40]}: "
            f"mid={o.mid_price:.3f} "
            f"ask={o.best_ask or 0:.3f} "
            f"depth=${o.ask_depth_usd:.0f}"
            for o in self.outcomes[:6]
        )
        return (
            f"{'✅' if self.is_valid else '❌'} "
            f"Fork [{self.fork_type}] "
            f"{'negRisk' if self.is_neg_risk else 'title'}\n"
            f"  Event: {self.event_title[:50]}\n"
            f"  Outcomes:\n{outcomes_str}\n"
            f"  Mid sum: {self.mid_sum:.4f}\n"
            f"  Real sum: {self.real_sum:.4f}\n"
            f"  Slippage: {self.slippage_pct:.2f}%\n"
            f"  Gross edge: {self.gross_edge_pct:.2f}%\n"
            f"  Fees: {self.fee_cost_pct:.2f}%\n"
            f"  Net edge: {self.net_edge_pct:.2f}%\n"
            f"  Min depth: ${self.min_depth_usd:.0f}\n"
            f"  {self.reject_reason if not self.is_valid else 'VALID'}"
        )

    def format_alert(self) -> str:
        tag = "✅ negRisk" if self.is_neg_risk else "⚠️ title-match"
        outcomes_str = "\n".join(
            f"   {o.question[:40]}: "
            f"ask={o.best_ask or 0:.4f} "
            f"(mid={o.mid_price:.3f})"
            for o in self.outcomes[:8]
        )
        return (
            f"🔀 REAL FORK [{self.fork_type.upper()}] {tag}\n\n"
            f"📋 {self.event_title}\n\n"
            f"📊 Outcomes ({len(self.outcomes)}):\n{outcomes_str}\n\n"
            f"📈 Mid sum: {self.mid_sum:.4f}\n"
            f"📈 Real sum: {self.real_sum:.4f}\n"
            f"📈 Slippage: {self.slippage_pct:.2f}%\n"
            f"💰 Gross edge: {self.gross_edge_pct:.2f}%\n"
            f"💸 Fees: {self.fee_cost_pct:.2f}%\n"
            f"✅ Net edge: {self.net_edge_pct:.2f}%\n"
            f"💧 Min depth: ${self.min_depth_usd:.0f}\n"
            f"💵 Entry: ${self.entry_size_usd:.2f}"
        )

    def to_dict(self) -> dict:
        return {
            "event_title": self.event_title,
            "event_id": self.event_id,
            "fork_type": self.fork_type,
            "is_neg_risk": self.is_neg_risk,
            "mid_sum": round(self.mid_sum, 4),
            "real_sum": round(self.real_sum, 4),
            "gross_edge_pct": round(self.gross_edge_pct, 2),
            "net_edge_pct": round(self.net_edge_pct, 2),
            "slippage_pct": round(self.slippage_pct, 2),
            "min_depth_usd": round(self.min_depth_usd, 0),
            "entry_size_usd": self.entry_size_usd,
            "num_outcomes": len(self.outcomes),
            "is_valid": self.is_valid,
            "reject_reason": self.reject_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (из твоего кода, с доработками)
# ══════════════════════════════════════════════════════════════

def parse_prices(raw) -> List[float]:
    """Парсит outcomePrices из Gamma API."""
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def check_neg_risk(event: dict) -> bool:
    """Проверяет negRisk флаг — золотой стандарт."""
    markets = event.get("markets", [])
    if not markets:
        return False
    first = markets[0]
    return first.get("negRisk", False) or first.get("neg_risk", False)


def check_exclusive_by_title(event: dict) -> bool:
    """Фallback: проверка по заголовку."""
    title = (event.get("title") or "").lower()

    non_exclusive = [
        "above", "below", "over", "under",
        "how many", "how much", "gdp", "growth",
        "more markets", "total", "spread",
        "close above", "close below", "temperature",
    ]
    exclusive = [
        "champion", "winner", "who will win",
        "next president", "nominee", "mvp",
        "ballon d'or", "which team wins",
        "who will be the next",
    ]

    for pat in non_exclusive:
        if pat in title:
            return False
    for pat in exclusive:
        if pat in title:
            return True
    return False


# ══════════════════════════════════════════════════════════════
#  FORK SCANNER CLASS
# ══════════════════════════════════════════════════════════════

class ForkScanner:
    """
    Двухэтапный сканер вилок:
      1. Быстрый фильтр по mid-price (Gamma API)
      2. Глубокая проверка по реальному стакану (CLOB API)
    """

    def __init__(self, clob_client, config):
        """
        Args:
            clob_client: PolymarketClient instance
            config: Settings instance
        """
        self.clob = clob_client
        self.config = config

        # счётчики для статистики
        self.stats = {
            "scans": 0,
            "events_checked": 0,
            "candidates_found": 0,
            "books_fetched": 0,
            "valid_forks": 0,
            "rejected": {
                "no_neg_risk": 0,
                "few_outcomes": 0,
                "sum_too_far": 0,
                "no_deviation": 0,
                "no_book": 0,
                "insufficient_depth": 0,
                "slippage_killed": 0,
                "fees_killed": 0,
                "edge_too_high": 0,
            },
        }

    async def find_forks(
        self,
        entry_size: float = 5.0,
        min_edge: float = 0.5,
        max_edge: float = 15.0,
        min_volume: float = 500,
    ) -> List[RealFork]:
        """
        Главный метод: находит реально исполнимые вилки.

        Args:
            entry_size: размер входа в USD
            min_edge: минимальный net edge %
            max_edge: максимальный edge % (anti-fake)
            min_volume: минимальный объём рынка

        Returns:
            Список RealFork с is_valid=True/False
        """
        self.stats["scans"] += 1

        # ── Этап 1: Получаем события из Gamma ────────────────────
        events = await self._fetch_events()
        if not events:
            return []

        # ── Этап 2: Быстрый фильтр по mid-price ─────────────────
        candidates = self._fast_filter(events, min_volume)
        logger.info(
            f"Stage 1: {len(events)} events → "
            f"{len(candidates)} candidates"
        )

        if not candidates:
            return []

        # ── Этап 3: Получаем реальные стаканы ────────────────────
        forks = []
        for candidate in candidates:
            fork = await self._deep_check(
                candidate, entry_size, min_edge, max_edge
            )
            forks.append(fork)

        valid = [f for f in forks if f.is_valid]
        invalid = [f for f in forks if not f.is_valid]

        logger.info(
            f"Stage 2: {len(candidates)} candidates → "
            f"{len(valid)} valid forks"
        )

        for f in invalid:
            logger.debug(
                f"Rejected: {f.event_title[:40]} | "
                f"{f.reject_reason}"
            )

        for f in valid:
            logger.info(f"VALID FORK:\n{f.summary()}")

        return forks

    # ── Этап 1: Fetch Events ─────────────────────────────────────

    async def _fetch_events(self, limit: int = 200) -> List[dict]:
        """Получает события из Gamma API."""
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession() as session:
                params = {
                    "active": "true",
                    "closed": "false",
                    "limit": str(limit),
                    "order": "volume24hr",
                }
                async with session.get(
                    GAMMA_API,
                    params=params,
                    headers=DEFAULT_HEADERS,
                    timeout=timeout,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Gamma API: {resp.status}")
                        return []

                    events = await resp.json()

                    # фильтруем: >= 3 открытых рынка
                    multi = []
                    for e in events:
                        markets = e.get("markets", [])
                        open_mkts = [
                            m for m in markets
                            if not m.get("closed", False)
                        ]
                        if len(open_mkts) >= 3:
                            e["markets"] = open_mkts
                            multi.append(e)

                    self.stats["events_checked"] += len(events)
                    logger.info(
                        f"Gamma: {len(events)} events, "
                        f"{len(multi)} multi-outcome"
                    )
                    return multi

        except Exception as e:
            logger.error(f"Gamma fetch error: {e}")
            return []

    # ── Этап 2: Быстрый фильтр ──────────────────────────────────

    def _fast_filter(
        self,
        events: List[dict],
        min_volume: float,
    ) -> List[ForkCandidate]:
        """
        Быстрый фильтр по mid-price.
        Не обращается к CLOB — только Gamma данные.
        """
        candidates = []

        for event in events:
            # ── проверка exclusivity ─────────────────────────
            is_neg = check_neg_risk(event)
            is_title = check_exclusive_by_title(event)

            if not is_neg and not is_title:
                self.stats["rejected"]["no_neg_risk"] += 1
                continue

            verification = "neg_risk" if is_neg else "title"

            # ── парсим исходы ────────────────────────────────
            outcomes = []
            for m in event.get("markets", []):
                if m.get("closed", False):
                    continue

                prices = parse_prices(m.get("outcomePrices"))
                if len(prices) < 2:
                    continue

                yes_price = prices[0]
                if yes_price < 0.005:
                    continue

                vol = float(
                    m.get("volume24hr")
                    or m.get("volumeNum")
                    or 0
                )

                # ── КЛЮЧЕВОЕ: извлекаем token_id ─────────────
                # Polymarket CLOB использует clobTokenIds
                clob_ids = m.get("clobTokenIds")
                if isinstance(clob_ids, str):
                    try:
                        clob_ids = json.loads(clob_ids)
                    except json.JSONDecodeError:
                        clob_ids = []

                if not clob_ids or len(clob_ids) < 1:
                    continue

                # clobTokenIds[0] = YES token, [1] = NO token
                yes_token_id = clob_ids[0]

                outcomes.append(OutcomeBook(
                    question=m.get("question", ""),
                    token_id=yes_token_id,
                    market_id=m.get("id", ""),
                    volume_24h=vol,
                    mid_price=yes_price,
                ))

            if len(outcomes) < 3:
                self.stats["rejected"]["few_outcomes"] += 1
                continue

            # ── volume фильтр ────────────────────────────────
            min_vol = min(o.volume_24h for o in outcomes)
            if min_vol < min_volume:
                continue

            # ── mid-price sum check ──────────────────────────
            mid_sum = sum(o.mid_price for o in outcomes)
            deviation = abs(mid_sum - 1.0)

            if deviation > MAX_SUM_DEVIATION:
                self.stats["rejected"]["sum_too_far"] += 1
                continue

            if deviation < MIN_SUM_DEVIATION:
                self.stats["rejected"]["no_deviation"] += 1
                continue

            candidate = ForkCandidate(
                event_title=event.get("title", "Unknown"),
                event_id=event.get("id", ""),
                is_neg_risk=is_neg,
                verification=verification,
                outcomes=outcomes,
                mid_sum=mid_sum,
                mid_deviation_pct=((1.0 - mid_sum) / mid_sum) * 100,
            )
            candidates.append(candidate)
            self.stats["candidates_found"] += 1

        return candidates

    # ── Этап 3: Глубокая проверка со стаканами ───────────────────

    async def _deep_check(
        self,
        candidate: ForkCandidate,
        entry_size: float,
        min_edge: float,
        max_edge: float,
    ) -> RealFork:
        """
        Получает реальные стаканы и считает
        настоящий edge с учётом slippage.
        """
        fork = RealFork(
            event_title=candidate.event_title,
            event_id=candidate.event_id,
            fork_type="under" if candidate.mid_sum < 1.0 else "over",
            is_neg_risk=candidate.is_neg_risk,
            verification=candidate.verification,
            outcomes=candidate.outcomes,
            entry_size_usd=entry_size,
            mid_sum=candidate.mid_sum,
        )

        # ── Получаем стаканы для всех исходов ────────────────
        per_outcome_usd = entry_size / len(candidate.outcomes)

        for outcome in fork.outcomes:
            book = await self._fetch_orderbook(outcome.token_id)
            if not book:
                fork.reject_reason = (
                    f"no_orderbook: {outcome.question[:30]}"
                )
                self.stats["rejected"]["no_book"] += 1
                return fork

            outcome.asks = book["asks"]
            outcome.bids = book["bids"]
            outcome.best_ask = (
                min(a.price for a in book["asks"])
                if book["asks"] else None
            )
            outcome.best_bid = (
                max(b.price for b in book["bids"])
                if book["bids"] else None
            )
            outcome.ask_depth_usd = sum(
                a.price * a.size for a in book["asks"]
            )

            self.stats["books_fetched"] += 1

            # маленькая пауза чтобы не забить API
            await asyncio.sleep(0.3)

        # ── Проверяем глубину ─────────────────────────────────
        fork.min_depth_usd = min(
            o.ask_depth_usd for o in fork.outcomes
        )

        if fork.min_depth_usd < per_outcome_usd:
            fork.reject_reason = (
                f"depth={fork.min_depth_usd:.0f} "
                f"< needed={per_outcome_usd:.0f}"
            )
            self.stats["rejected"]["insufficient_depth"] += 1
            return fork

        # ── Считаем реальный edge ─────────────────────────────
        if fork.fork_type == "under":
            self._calc_under_fork(fork, per_outcome_usd)
        else:
            self._calc_over_fork(fork, per_outcome_usd)

        # ── Финальные проверки ────────────────────────────────
        if fork.net_edge_pct > max_edge:
            fork.is_valid = False
            fork.reject_reason = (
                f"edge_too_high={fork.net_edge_pct:.1f}% "
                f"(max={max_edge}%)"
            )
            self.stats["rejected"]["edge_too_high"] += 1
            return fork

        if fork.net_edge_pct < min_edge:
            fork.is_valid = False
            fork.reject_reason = (
                f"net_edge={fork.net_edge_pct:.2f}% "
                f"< min={min_edge}%"
            )
            self.stats["rejected"]["fees_killed"] += 1
            return fork

        fork.is_valid = True
        self.stats["valid_forks"] += 1
        return fork

    def _calc_under_fork(self, fork: RealFork, per_outcome_usd: float):
        """
        Under fork: sum(YES) < 1.0
        Стратегия: купить все YES, получить $1 при settlement.
        """
        real_prices = []
        all_filled = True

        for outcome in fork.outcomes:
            fill = outcome.cost_to_buy(per_outcome_usd)
            if not fill:
                fork.reject_reason = (
                    f"cant_fill: {outcome.question[:30]}"
                )
                return

            if not fill["fully_filled"]:
                all_filled = False

            real_prices.append(fill["avg_price"])

        fork.real_avg_prices = real_prices
        fork.real_sum = sum(real_prices)
        fork.all_filled = all_filled

        # slippage: насколько real хуже mid
        if fork.mid_sum > 0:
            fork.slippage_pct = (
                (fork.real_sum - fork.mid_sum) / fork.mid_sum
            ) * 100

        # gross edge (до комиссий)
        if fork.real_sum > 0 and fork.real_sum < 1.0:
            fork.gross_edge_pct = (
                (1.0 - fork.real_sum) / fork.real_sum
            ) * 100
        else:
            fork.gross_edge_pct = 0
            fork.reject_reason = (
                f"real_sum={fork.real_sum:.4f} >= 1.0 after slippage"
            )
            self.stats["rejected"]["slippage_killed"] += 1
            return

        # fee: 2% на выигрышную сторону
        # В multi-outcome, одна сторона выигрывает
        # Мы платим 2% от выигрыша с ОДНОГО исхода
        # Упрощённо: ~2% от (1/N) позиции, но от всей $1 выплаты
        # Точнее: fee = 0.02 * payout_per_share
        fork.fee_cost_pct = POLY_FEE_PCT * 100  # 2%

        fork.net_edge_pct = fork.gross_edge_pct - fork.fee_cost_pct

    def _calc_over_fork(self, fork: RealFork, per_outcome_usd: float):
        """
        Over fork: sum(YES) > 1.0
        Стратегия: mint полный набор за $1, продать все YES.
        """
        # Для over-fork нам нужны BIDS (мы продаём)
        real_prices = []
        all_filled = True

        for outcome in fork.outcomes:
            if not outcome.bids:
                fork.reject_reason = (
                    f"no_bids: {outcome.question[:30]}"
                )
                return

            # считаем сколько получим при продаже
            sell_result = self._calc_sell(
                outcome.bids, per_outcome_usd
            )
            if not sell_result:
                fork.reject_reason = (
                    f"cant_sell: {outcome.question[:30]}"
                )
                return

            if not sell_result["fully_filled"]:
                all_filled = False

            real_prices.append(sell_result["avg_price"])

        fork.real_avg_prices = real_prices
        fork.real_sum = sum(real_prices)
        fork.all_filled = all_filled

        if fork.mid_sum > 0:
            fork.slippage_pct = (
                (fork.mid_sum - fork.real_sum) / fork.mid_sum
            ) * 100

        if fork.real_sum > 1.0:
            fork.gross_edge_pct = (
                (fork.real_sum - 1.0) / 1.0
            ) * 100
        else:
            fork.gross_edge_pct = 0
            fork.reject_reason = (
                f"real_sum={fork.real_sum:.4f} <= 1.0 after slippage"
            )
            self.stats["rejected"]["slippage_killed"] += 1
            return

        fork.fee_cost_pct = POLY_FEE_PCT * 100
        fork.net_edge_pct = fork.gross_edge_pct - fork.fee_cost_pct

    def _calc_sell(
        self, bids: List[OrderBookLevel], amount_usd: float
    ) -> Optional[dict]:
        """Считает реальную выручку при продаже."""
        sorted_bids = sorted(bids, key=lambda x: -x.price)

        total_revenue = 0.0
        shares_sold = 0.0
        remaining = amount_usd

        for level in sorted_bids:
            level_value = level.size * level.price
            if level_value <= remaining:
                total_revenue += level_value
                shares_sold += level.size
                remaining -= level_value
            else:
                shares_at_level = remaining / level.price
                total_revenue += remaining
                shares_sold += shares_at_level
                remaining = 0
                break

        if shares_sold == 0:
            return None

        return {
            "total_revenue": total_revenue,
            "shares": shares_sold,
            "avg_price": total_revenue / shares_sold,
            "fully_filled": remaining < 0.01,
        }

    # ── CLOB Orderbook ───────────────────────────────────────────

    async def _fetch_orderbook(
        self, token_id: str
    ) -> Optional[dict]:
        """
        Получает реальный стакан из CLOB API.

        Returns:
            {"asks": [OrderBookLevel], "bids": [OrderBookLevel]}
        """
        try:
            url = f"{CLOB_API}/book"
            params = {"token_id": token_id}

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=DEFAULT_HEADERS,
                    timeout=timeout,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"CLOB book {token_id[:16]}...: "
                            f"{resp.status}"
                        )
                        return None

                    data = await resp.json()

                    asks = [
                        OrderBookLevel(
                            price=float(a["price"]),
                            size=float(a["size"]),
                        )
                        for a in data.get("asks", [])
                        if float(a.get("price", 0)) > 0
                    ]

                    bids = [
                        OrderBookLevel(
                            price=float(b["price"]),
                            size=float(b["size"]),
                        )
                        for b in data.get("bids", [])
                        if float(b.get("price", 0)) > 0
                    ]

                    return {"asks": asks, "bids": bids}

        except Exception as e:
            logger.error(
                f"CLOB book error {token_id[:16]}...: {e}"
            )
            return None

    def get_stats(self) -> dict:
        return dict(self.stats)

    def reset_stats(self):
        for key in self.stats["rejected"]:
            self.stats["rejected"][key] = 0
        self.stats["candidates_found"] = 0
        self.stats["books_fetched"] = 0
        self.stats["valid_forks"] = 0
