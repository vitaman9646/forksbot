# core/scanner.py
"""
ForkScanner v2.0 — сканер с реальными стаканами.

Двухэтапная проверка:
  1. Быстрый фильтр по mid-price (Gamma API) — отсеивает 95%
  2. Глубокая проверка по реальному orderbook (CLOB API) — точный edge

Учитывает:
  • Реальную глубину стакана (не mid-price)
  • Slippage при конкретном размере входа
  • Комиссию Polymarket 2% на выигрышную сторону
  • Anti-fake фильтр (edge > 15% = ошибка данных)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

logger = logging.getLogger("arb_scanner.scanner")

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

POLY_FEE_PCT = 0.02          # 2% на выигрышную сторону
MAX_REALISTIC_EDGE = 15.0    # выше = ошибка данных / фейк


# ══════════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class OrderBookLevel:
    """Один уровень в стакане (цена + размер)."""
    price: float
    size: float  # в shares


@dataclass
class OutcomeBook:
    """Один исход рынка с данными стакана."""
    question: str
    token_id: str
    market_id: str
    volume_24h: float
    mid_price: float        # из Gamma API (для быстрого фильтра)

    # реальный стакан (заполняется на этапе 2)
    asks: List[OrderBookLevel] = field(default_factory=list)
    bids: List[OrderBookLevel] = field(default_factory=list)

    # рассчитанные поля
    best_ask: Optional[float] = None
    best_bid: Optional[float] = None
    ask_depth_usd: float = 0.0

    def has_book(self) -> bool:
        return len(self.asks) > 0

    def cost_to_buy(self, amount_usd: float) -> Optional[dict]:
        """
        Считает РЕАЛЬНУЮ стоимость покупки на $amount_usd
        с учётом глубины стакана.

        Идём по ask levels от лучшего к худшему,
        "съедая" ликвидность на каждом уровне.

        Returns:
            {
                "total_cost": float,      # сколько потратим USD
                "shares": float,          # сколько shares получим
                "avg_price": float,       # средняя цена
                "levels_consumed": int,   # сколько уровней съели
                "fully_filled": bool,     # хватило ли ликвидности
                "unfilled_usd": float,    # сколько не смогли купить
            }
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

    def revenue_from_sell(self, amount_usd: float) -> Optional[dict]:
        """
        Считает РЕАЛЬНУЮ выручку при продаже shares на ~$amount_usd.

        Идём по bid levels от лучшего к худшему.

        Returns:
            {
                "total_revenue": float,
                "shares": float,
                "avg_price": float,
                "levels_consumed": int,
                "fully_filled": bool,
                "unfilled_usd": float,
            }
        """
        if not self.bids:
            return None

        sorted_bids = sorted(self.bids, key=lambda x: -x.price)

        total_revenue = 0.0
        shares_sold = 0.0
        levels_consumed = 0
        remaining = amount_usd

        for level in sorted_bids:
            level_value = level.size * level.price

            if level_value <= remaining:
                total_revenue += level_value
                shares_sold += level.size
                remaining -= level_value
                levels_consumed += 1
            else:
                shares_at_level = remaining / level.price
                total_revenue += remaining
                shares_sold += shares_at_level
                remaining = 0
                levels_consumed += 1
                break

        if shares_sold == 0:
            return None

        return {
            "total_revenue": total_revenue,
            "shares": shares_sold,
            "avg_price": total_revenue / shares_sold,
            "levels_consumed": levels_consumed,
            "fully_filled": remaining < 0.01,
            "unfilled_usd": remaining,
        }


@dataclass
class ForkCandidate:
    """Кандидат на вилку (после быстрого фильтра, до проверки стакана)."""
    event_title: str
    event_id: str
    is_neg_risk: bool
    verification: str       # "neg_risk" / "title"
    outcomes: List[OutcomeBook] = field(default_factory=list)

    mid_sum: float = 0.0
    mid_deviation_pct: float = 0.0


@dataclass
class RealFork:
    """Подтверждённая (или отклонённая) вилка с реальными данными стакана."""
    event_title: str
    event_id: str
    fork_type: str           # "under" / "over"
    is_neg_risk: bool
    verification: str

    outcomes: List[OutcomeBook] = field(default_factory=list)
    entry_size_usd: float = 0.0

    # mid-price данные (из Gamma)
    mid_sum: float = 0.0

    # реальные данные из стакана
    real_sum: float = 0.0
    real_avg_prices: List[float] = field(default_factory=list)

    # edge расчёты
    gross_edge_pct: float = 0.0
    fee_cost_pct: float = 0.0
    net_edge_pct: float = 0.0
    slippage_pct: float = 0.0

    # ликвидность
    min_depth_usd: float = 0.0
    all_filled: bool = False

    # валидация
    is_valid: bool = False
    reject_reason: str = ""

    # token_ids для исполнения
    token_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ REJECTED"
        tag = "negRisk" if self.is_neg_risk else "title-match"

        outcomes_str = "\n".join(
            f"    {o.question[:40]:40s} "
            f"mid={o.mid_price:.3f}  "
            f"ask={o.best_ask or 0:.3f}  "
            f"depth=${o.ask_depth_usd:.0f}"
            for o in self.outcomes[:8]
        )
        if len(self.outcomes) > 8:
            outcomes_str += f"\n    ... +{len(self.outcomes) - 8} more"

        return (
            f"{status} Fork [{self.fork_type.upper()}] ({tag})\n"
            f"  Event: {self.event_title[:60]}\n"
            f"  Outcomes ({len(self.outcomes)}):\n{outcomes_str}\n"
            f"  Mid sum:    {self.mid_sum:.4f}\n"
            f"  Real sum:   {self.real_sum:.4f}\n"
            f"  Slippage:   {self.slippage_pct:+.2f}%\n"
            f"  Gross edge: {self.gross_edge_pct:.2f}%\n"
            f"  Fees:       {self.fee_cost_pct:.2f}%\n"
            f"  Net edge:   {self.net_edge_pct:.2f}%\n"
            f"  Min depth:  ${self.min_depth_usd:.0f}\n"
            f"  Entry size: ${self.entry_size_usd:.2f}\n"
            f"  {self.reject_reason if not self.is_valid else 'READY TO TRADE'}"
        )

    def format_alert(self) -> str:
        tag = "✅ negRisk" if self.is_neg_risk else "⚠️ title"
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
            f"📈 Slippage: {self.slippage_pct:+.2f}%\n"
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
            "verification": self.verification,
            "num_outcomes": len(self.outcomes),
            "mid_sum": round(self.mid_sum, 4),
            "real_sum": round(self.real_sum, 4),
            "gross_edge_pct": round(self.gross_edge_pct, 2),
            "fee_cost_pct": round(self.fee_cost_pct, 2),
            "net_edge_pct": round(self.net_edge_pct, 2),
            "slippage_pct": round(self.slippage_pct, 2),
            "min_depth_usd": round(self.min_depth_usd, 0),
            "entry_size_usd": self.entry_size_usd,
            "all_filled": self.all_filled,
            "is_valid": self.is_valid,
            "reject_reason": self.reject_reason,
            "outcomes": [
                {
                    "question": o.question[:60],
                    "token_id": o.token_id,
                    "mid_price": round(o.mid_price, 4),
                    "best_ask": round(o.best_ask, 4) if o.best_ask else None,
                    "best_bid": round(o.best_bid, 4) if o.best_bid else None,
                    "depth_usd": round(o.ask_depth_usd, 0),
                    "volume_24h": round(o.volume_24h, 0),
                }
                for o in self.outcomes
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def parse_prices(raw) -> List[float]:
    """Парсит outcomePrices из Gamma API (может быть list или JSON string)."""
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except (ValueError, TypeError):
            return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            return []
    return []


def parse_clob_token_ids(raw) -> List[str]:
    """Парсит clobTokenIds из Gamma API (может быть list или JSON string)."""
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def check_neg_risk(event: dict) -> bool:
    """
    Проверяет negRisk флаг — золотой стандарт.
    negRisk = True означает исходы взаимоисключающие,
    shares можно mint/merge через контракт.
    """
    markets = event.get("markets", [])
    if not markets:
        return False
    first = markets[0]
    return bool(first.get("negRisk", False) or first.get("neg_risk", False))


def check_exclusive_by_title(event: dict) -> bool:
    """
    Fallback: проверка взаимоисключительности по заголовку.
    Менее надёжна чем negRisk.
    """
    title = (event.get("title") or "").lower()

    non_exclusive = [
        "above", "below", "over", "under",
        "how many", "how much", "gdp", "growth",
        "more markets", "total", "spread",
        "close above", "close below", "temperature",
        "at least", "fewer than", "higher than",
        "lower than", "between", "range",
    ]

    exclusive = [
        "champion", "winner", "who will win",
        "next president", "nominee", "mvp",
        "ballon d'or", "which team wins",
        "who will be the next",
        "which country", "which party",
        "who will replace",
    ]

    for pat in non_exclusive:
        if pat in title:
            return False
    for pat in exclusive:
        if pat in title:
            return True
    return False


# ══════════════════════════════════════════════════════════════
#  FORK SCANNER
# ══════════════════════════════════════════════════════════════

class ForkScanner:
    """
    Двухэтапный сканер вилок:
      Этап 1: Быстрый фильтр по mid-price из Gamma API
      Этап 2: Глубокая проверка по реальному стакану из CLOB API
    """

    def __init__(self, http_session, config):
        """
        Args:
            http_session: RateLimitedSession — общая HTTP сессия
            config: Settings — конфигурация
        """
        self.http = http_session
        self.cfg = config

        # статистика
        self.stats = {
            "scans": 0,
            "events_checked": 0,
            "candidates_found": 0,
            "books_fetched": 0,
            "valid_forks": 0,
            "rejected": {
                "no_neg_risk": 0,
                "few_outcomes": 0,
                "no_token_id": 0,
                "low_volume": 0,
                "sum_too_far": 0,
                "no_deviation": 0,
                "no_book": 0,
                "insufficient_depth": 0,
                "cant_fill": 0,
                "partial_fill": 0,
                "slippage_killed": 0,
                "fees_killed": 0,
                "edge_too_high": 0,
                "edge_negative": 0,
            },
        }

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    async def find_forks(
        self,
        entry_size: float = 5.0,
        min_edge: float = 0.5,
        max_edge: float = 15.0,
        min_volume: float = 500.0,
    ) -> List[RealFork]:
        """
        Главный метод: находит вилки с реальной проверкой стаканов.

        Args:
            entry_size: размер входа в USD (делится поровну между исходами)
            min_edge: минимальный net edge % после комиссий
            max_edge: максимальный edge % (anti-fake)
            min_volume: минимальный volume24hr для каждого исхода

        Returns:
            Список RealFork (и valid, и rejected — для анализа)
        """
        self.stats["scans"] += 1

        # ── Этап 1: события из Gamma API ─────────────────────
        events = await self._fetch_events()
        if not events:
            logger.info("No events fetched")
            return []

        # ── Этап 2: быстрый фильтр по mid-price ─────────────
        candidates = self._fast_filter(events, min_volume)
        logger.info(
            f"Stage 1 (mid-price): "
            f"{len(events)} events → {len(candidates)} candidates"
        )

        if not candidates:
            return []

        # ── Этап 3: глубокая проверка со стаканами ───────────
        forks = []
        for candidate in candidates:
            fork = await self._deep_check(
                candidate, entry_size, min_edge, max_edge
            )
            forks.append(fork)

        valid_count = sum(1 for f in forks if f.is_valid)
        logger.info(
            f"Stage 2 (orderbook): "
            f"{len(candidates)} candidates → {valid_count} valid forks"
        )

        # логируем каждый valid fork
        for f in forks:
            if f.is_valid:
                logger.info(f"\n{f.summary()}")

        return forks

    def get_stats(self) -> dict:
        """Возвращает статистику сканера."""
        return {
            "scans": self.stats["scans"],
            "events_checked": self.stats["events_checked"],
            "candidates_found": self.stats["candidates_found"],
            "books_fetched": self.stats["books_fetched"],
            "valid_forks": self.stats["valid_forks"],
            "rejected": dict(self.stats["rejected"]),
        }

    def reset_stats(self):
        """Сбрасывает статистику."""
        for key in self.stats["rejected"]:
            self.stats["rejected"][key] = 0
        self.stats["scans"] = 0
        self.stats["events_checked"] = 0
        self.stats["candidates_found"] = 0
        self.stats["books_fetched"] = 0
        self.stats["valid_forks"] = 0

    # ══════════════════════════════════════════════════════════
    #  ЭТАП 1: ПОЛУЧЕНИЕ СОБЫТИЙ ИЗ GAMMA API
    # ══════════════════════════════════════════════════════════

    async def _fetch_events(self) -> List[dict]:
        """
        Запрашивает активные события из Gamma API.
        Фильтрует: только события с >= MIN_OUTCOMES открытых рынков.
        """
        data = await self.http.get_json(
            self.cfg.GAMMA_EVENTS_API,
            params={
                "active": "true",
                "closed": "false",
                "limit": str(self.cfg.GAMMA_FETCH_LIMIT),
                "order": "volume24hr",
            },
        )

        if not data:
            logger.warning("Gamma API returned no data")
            return []

        if not isinstance(data, list):
            logger.warning(f"Gamma API unexpected format: {type(data)}")
            return []

        multi = []
        for event in data:
            markets = event.get("markets", [])
            open_markets = [
                m for m in markets
                if not m.get("closed", False)
            ]
            if len(open_markets) >= self.cfg.MIN_OUTCOMES:
                event["markets"] = open_markets
                multi.append(event)

        self.stats["events_checked"] += len(data)
        logger.info(
            f"Gamma: {len(data)} events total, "
            f"{len(multi)} with {self.cfg.MIN_OUTCOMES}+ outcomes"
        )
        return multi

    # ══════════════════════════════════════════════════════════
    #  ЭТАП 2: БЫСТРЫЙ ФИЛЬТР ПО MID-PRICE
    # ══════════════════════════════════════════════════════════

    def _fast_filter(
        self,
        events: List[dict],
        min_volume: float,
    ) -> List[ForkCandidate]:
        """
        Быстрый фильтр: отсеивает ~95% событий
        без обращения к CLOB API.

        Проверяет:
        1. negRisk или title-match
        2. Наличие token_id для каждого исхода
        3. Объём >= min_volume
        4. Сумма mid-prices отклоняется от 1.0
        """
        candidates = []

        for event in events:
            # ── 1. Проверка взаимоисключительности ────────────
            is_neg = check_neg_risk(event)
            is_title = check_exclusive_by_title(event)

            if not is_neg and not is_title:
                self.stats["rejected"]["no_neg_risk"] += 1
                continue

            verification = "neg_risk" if is_neg else "title"

            # ── 2. Парсим исходы ─────────────────────────────
            outcomes = []
            skip_event = False

            for market in event.get("markets", []):
                if market.get("closed", False):
                    continue

                # prices
                prices = parse_prices(market.get("outcomePrices"))
                if len(prices) < 2:
                    continue

                yes_price = prices[0]
                if yes_price < 0.005:
                    continue

                # volume
                vol = float(
                    market.get("volume24hr")
                    or market.get("volumeNum")
                    or 0
                )

                # token_id — КРИТИЧЕСКИ ВАЖНО для CLOB запросов
                clob_ids = parse_clob_token_ids(
                    market.get("clobTokenIds")
                )
                if not clob_ids:
                    self.stats["rejected"]["no_token_id"] += 1
                    skip_event = True
                    break

                yes_token_id = clob_ids[0]

                outcomes.append(OutcomeBook(
                    question=market.get("question", ""),
                    token_id=yes_token_id,
                    market_id=market.get("id", ""),
                    volume_24h=vol,
                    mid_price=yes_price,
                ))

            if skip_event:
                continue

            # ── 3. Достаточно исходов? ───────────────────────
            if len(outcomes) < self.cfg.MIN_OUTCOMES:
                self.stats["rejected"]["few_outcomes"] += 1
                continue

            # ── 4. Объём ─────────────────────────────────────
            min_vol = min(o.volume_24h for o in outcomes)
            if min_vol < min_volume:
                self.stats["rejected"]["low_volume"] += 1
                continue

            # ── 5. Сумма mid-prices ──────────────────────────
            mid_sum = sum(o.mid_price for o in outcomes)
            deviation = abs(mid_sum - 1.0)

            if deviation > self.cfg.MAX_SUM_DEVIATION:
                self.stats["rejected"]["sum_too_far"] += 1
                continue

            if deviation < self.cfg.MIN_SUM_DEVIATION:
                self.stats["rejected"]["no_deviation"] += 1
                continue

            # ── Кандидат прошёл быстрый фильтр ───────────────
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

            logger.info(
                f"Candidate: {candidate.event_title[:50]} | "
                f"mid_sum={mid_sum:.4f} | "
                f"dev={deviation:.4f} | "
                f"{verification} | "
                f"{len(outcomes)} outcomes"
            )

        return candidates

    # ══════════════════════════════════════════════════════════
    #  ЭТАП 3: ГЛУБОКАЯ ПРОВЕРКА СО СТАКАНАМИ
    # ══════════════════════════════════════════════════════════

    async def _deep_check(
        self,
        candidate: ForkCandidate,
        entry_size: float,
        min_edge: float,
        max_edge: float,
    ) -> RealFork:
        """
        Получает реальные стаканы из CLOB API и считает
        настоящий edge с учётом глубины и slippage.
        """
        fork_type = "under" if candidate.mid_sum < 1.0 else "over"

        fork = RealFork(
            event_title=candidate.event_title,
            event_id=candidate.event_id,
            fork_type=fork_type,
            is_neg_risk=candidate.is_neg_risk,
            verification=candidate.verification,
            outcomes=candidate.outcomes,
            entry_size_usd=entry_size,
            mid_sum=candidate.mid_sum,
            token_ids=[o.token_id for o in candidate.outcomes],
        )

        per_outcome_usd = entry_size / len(candidate.outcomes)

        # ── Получаем стакан для каждого исхода ────────────────
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

            # пауза между запросами стаканов
            await asyncio.sleep(self.cfg.BOOK_FETCH_DELAY)

        # ── Проверяем минимальную глубину ─────────────────────
        depths = [o.ask_depth_usd for o in fork.outcomes]
        fork.min_depth_usd = min(depths) if depths else 0

        if fork.min_depth_usd < per_outcome_usd:
            fork.reject_reason = (
                f"insufficient_depth: "
                f"min_depth=${fork.min_depth_usd:.0f} "
                f"< needed=${per_outcome_usd:.0f}"
            )
            self.stats["rejected"]["insufficient_depth"] += 1
            return fork

        # ── Считаем реальный edge ────────────────────────────
        if fork_type == "under":
            self._calc_under_fork(fork, per_outcome_usd)
        else:
            self._calc_over_fork(fork, per_outcome_usd)

        # если расчёт уже отклонил — возвращаем
        if fork.reject_reason:
            return fork

        # ── Финальная проверка edge ──────────────────────────
        if fork.net_edge_pct > max_edge:
            fork.is_valid = False
            fork.reject_reason = (
                f"edge_too_high: {fork.net_edge_pct:.1f}% "
                f"> max {max_edge}%"
            )
            self.stats["rejected"]["edge_too_high"] += 1
            return fork

        if fork.net_edge_pct < min_edge:
            fork.is_valid = False
            fork.reject_reason = (
                f"edge_too_low: {fork.net_edge_pct:.2f}% "
                f"< min {min_edge}%"
            )
            self.stats["rejected"]["fees_killed"] += 1
            return fork

        # ── VALID! ───────────────────────────────────────────
        fork.is_valid = True
        self.stats["valid_forks"] += 1

        logger.info(
            f"🎯 VALID FORK: {fork.event_title[:45]} | "
            f"type={fork_type} | "
            f"net_edge={fork.net_edge_pct:.2f}% | "
            f"slippage={fork.slippage_pct:+.2f}% | "
            f"depth=${fork.min_depth_usd:.0f}"
        )

        return fork

    # ══════════════════════════════════════════════════════════
    #  EDGE CALCULATION
    # ══════════════════════════════════════════════════════════

    def _calc_under_fork(self, fork: RealFork, per_outcome_usd: float):
        """
        Under fork: sum(YES prices) < 1.0
        Стратегия: купить ВСЕ YES исходы → при settlement один из них = $1.

        Прибыль = $1 (payout) - sum(cost of all YES) - fees
        """
        real_prices = []
        all_filled = True

        for outcome in fork.outcomes:
            fill = outcome.cost_to_buy(per_outcome_usd)

            if not fill:
                fork.reject_reason = (
                    f"cant_fill: {outcome.question[:30]} "
                    f"(no asks)"
                )
                self.stats["rejected"]["cant_fill"] += 1
                return

            if not fill["fully_filled"]:
                all_filled = False
                fork.reject_reason = (
                    f"partial_fill: {outcome.question[:30]} "
                    f"unfilled=${fill['unfilled_usd']:.2f}"
                )
                self.stats["rejected"]["partial_fill"] += 1
                return

            real_prices.append(fill["avg_price"])

        fork.real_avg_prices = real_prices
        fork.real_sum = sum(real_prices)
        fork.all_filled = all_filled

        # slippage: насколько реальная сумма хуже mid-price
        if fork.mid_sum > 0:
            fork.slippage_pct = (
                (fork.real_sum - fork.mid_sum) / fork.mid_sum
            ) * 100

        # gross edge (до комиссий)
        if fork.real_sum >= 1.0:
            fork.reject_reason = (
                f"slippage_killed: real_sum={fork.real_sum:.4f} >= 1.0 "
                f"(was mid_sum={fork.mid_sum:.4f})"
            )
            fork.gross_edge_pct = ((1.0 - fork.real_sum) / fork.real_sum) * 100
            self.stats["rejected"]["slippage_killed"] += 1
            return

        fork.gross_edge_pct = ((1.0 - fork.real_sum) / fork.real_sum) * 100

        if fork.gross_edge_pct <= 0:
            fork.reject_reason = (
                f"edge_negative: gross={fork.gross_edge_pct:.2f}%"
            )
            self.stats["rejected"]["edge_negative"] += 1
            return

        # fee: 2% на выигрышную сторону
        # при settlement одна нога выигрывает, платим 2% от payout
        # payout = num_shares * $1, cost ≈ num_shares * real_sum/N
        # fee ≈ 2% * (entry_per_outcome / winning_price) ≈ 2% от payout
        # упрощённо: fee_pct ≈ 2% (от gross payout)
        fork.fee_cost_pct = POLY_FEE_PCT * 100  # 2.0

        fork.net_edge_pct = fork.gross_edge_pct - fork.fee_cost_pct

    def _calc_over_fork(self, fork: RealFork, per_outcome_usd: float):
        """
        Over fork: sum(YES prices) > 1.0
        Стратегия: mint полный набор за $1, продать все YES.

        Прибыль = sum(sell revenue) - $1 (mint cost) - fees
        """
        real_prices = []
        all_filled = True

        for outcome in fork.outcomes:
            sell = outcome.revenue_from_sell(per_outcome_usd)

            if not sell:
                fork.reject_reason = (
                    f"cant_sell: {outcome.question[:30]} "
                    f"(no bids)"
                )
                self.stats["rejected"]["cant_fill"] += 1
                return

            if not sell["fully_filled"]:
                all_filled = False
                fork.reject_reason = (
                    f"partial_sell: {outcome.question[:30]} "
                    f"unfilled=${sell['unfilled_usd']:.2f}"
                )
                self.stats["rejected"]["partial_fill"] += 1
                return

            real_prices.append(sell["avg_price"])

        fork.real_avg_prices = real_prices
        fork.real_sum = sum(real_prices)
        fork.all_filled = all_filled

        # slippage (для over-fork: slippage уменьшает revenue)
        if fork.mid_sum > 0:
            fork.slippage_pct = (
                (fork.mid_sum - fork.real_sum) / fork.mid_sum
            ) * 100

        # gross edge
        if fork.real_sum <= 1.0:
            fork.reject_reason = (
                f"slippage_killed: real_sum={fork.real_sum:.4f} <= 1.0 "
                f"(was mid_sum={fork.mid_sum:.4f})"
            )
            fork.gross_edge_pct = ((fork.real_sum - 1.0) / 1.0) * 100
            self.stats["rejected"]["slippage_killed"] += 1
            return

        fork.gross_edge_pct = ((fork.real_sum - 1.0) / 1.0) * 100

        if fork.gross_edge_pct <= 0:
            fork.reject_reason = (
                f"edge_negative: gross={fork.gross_edge_pct:.2f}%"
            )
            self.stats["rejected"]["edge_negative"] += 1
            return

        fork.fee_cost_pct = POLY_FEE_PCT * 100
        fork.net_edge_pct = fork.gross_edge_pct - fork.fee_cost_pct

    # ══════════════════════════════════════════════════════════
    #  CLOB ORDERBOOK
    # ══════════════════════════════════════════════════════════

    async def _fetch_orderbook(self, token_id: str) -> Optional[dict]:
        """
        Получает реальный стакан из Polymarket CLOB API.

        Endpoint: GET https://clob.polymarket.com/book?token_id=...

        Returns:
            {"asks": [OrderBookLevel, ...], "bids": [OrderBookLevel, ...]}
            или None при ошибке
        """
        data = await self.http.get_json(
            self.cfg.CLOB_BOOK_URL,
            params={"token_id": token_id},
            timeout=aiohttp.ClientTimeout(total=10),
        )

        if not data:
            logger.debug(f"No orderbook for {token_id[:20]}...")
            return None

        # парсим asks
        asks = []
        for entry in data.get("asks", []):
            try:
                price = float(entry.get("price", 0))
                size = float(entry.get("size", 0))
                if price > 0 and size > 0:
                    asks.append(OrderBookLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue

        # парсим bids
        bids = []
        for entry in data.get("bids", []):
            try:
                price = float(entry.get("price", 0))
                size = float(entry.get("size", 0))
                if price > 0 and size > 0:
                    bids.append(OrderBookLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue

        logger.debug(
            f"Book {token_id[:16]}...: "
            f"{len(asks)} asks, {len(bids)} bids"
        )

        return {"asks": asks, "bids": bids}
