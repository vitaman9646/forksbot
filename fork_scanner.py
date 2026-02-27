"""
Fork Scanner v1.6
─────────────────
Изменения vs v1.5:
  • orderbook depth check через CLOB API
  • token_id парсится из clobTokenIds
  • depth_checks.jsonl — лог для анализа реальной ликвидности
  • real_edge заменяет gamma_edge при исполнимых вилках
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional

from config import CFG
from execution.orderbook_checker import OrderbookChecker

logger = logging.getLogger("arb_scanner.forks")

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

POLY_TAKER_FEE = 0.02
MERGE_GAS_USD = 0.10
SWAP_GAS_PER_OUTCOME = 0.02

MAX_SUM_OVER = 1.08
MIN_SUM_UNDER = 0.92


@dataclass
class Outcome:
    question: str
    yes_price: float
    volume: float
    market_id: str
    liquidity: float = 0.0      # v1.5
    token_id: str = ""          # v1.6: для CLOB orderbook check


@dataclass
class ForkOpportunity:
    event_title: str
    event_id: str
    fork_type: str
    outcomes: list
    sum_yes: float
    raw_profit_pct: float
    net_profit_pct: float
    cost_to_execute: float
    expected_profit: float
    is_neg_risk: bool = False
    timestamp: str = ""
    min_volume: float = 0.0     # v1.5: минимальный volume среди исходов
    min_liquidity: float = 0.0  # v1.5

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def format_alert(self):
        verified = "✅ negRisk" if self.is_neg_risk else "⚠️ unverified"
        outcomes_str = "\n".join(
            f"   {o.question[:45]}: ${o.yes_price:.3f} (vol ${o.volume:.0f})"
            for o in self.outcomes[:8]
        )
        if len(self.outcomes) > 8:
            outcomes_str += f"\n   ...+{len(self.outcomes)-8} more"
        return (
            f"🔱 FORK ({self.fork_type.upper()}) {verified}\n\n"
            f"📋 {self.event_title}\n\n"
            f"📊 Outcomes ({len(self.outcomes)}):\n{outcomes_str}\n\n"
            f"📐 Sum: ${self.sum_yes:.4f}\n"
            f"💰 Net profit: {self.net_profit_pct:.2f}%\n"
            f"📦 Min volume: ${self.min_volume:.0f}\n"
            f"💵 Cost (100sh): ${self.cost_to_execute:.2f}\n"
            f"💎 Profit: ${self.expected_profit:.2f}"
        )

    def to_dict(self):
        return {
            "event": self.event_title,
            "event_id": self.event_id,
            "type": self.fork_type,
            "sum_yes": round(self.sum_yes, 4),
            "net_profit_pct": round(self.net_profit_pct, 2),
            "cost": round(self.cost_to_execute, 2),
            "expected_profit": round(self.expected_profit, 2),
            "num_outcomes": len(self.outcomes),
            "is_neg_risk": self.is_neg_risk,
            "min_volume": round(self.min_volume, 0),
            "timestamp": self.timestamp,
        }


def parse_prices(raw):
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def estimate_fees(num_outcomes: int, total_cost: float, fork_type: str) -> float:
    trading_fees = total_cost * POLY_TAKER_FEE
    gas = (num_outcomes * SWAP_GAS_PER_OUTCOME) + MERGE_GAS_USD
    return trading_fees + gas


def check_neg_risk(event: dict) -> bool:
    markets = event.get("markets", [])
    if not markets:
        return False
    first = markets[0]
    return bool(first.get("negRisk") or first.get("neg_risk"))


def check_exclusive_by_title(event: dict) -> bool:
    title = (event.get("title") or "").lower()
    exclusive = [
        "champion", "winner", "who will win", "next president",
        "nominee", "mvp", "ballon d'or", "which team wins",
        "who will be the next",
    ]
    non_exclusive = [
        "above", "below", "over", "under", "how many", "how much",
        "gdp", "growth", "more markets", "total", "spread",
        "close above", "close below", "temperature",
    ]
    for pat in non_exclusive:
        if pat in title:
            return False
    return any(pat in title for pat in exclusive)


def check_staleness(market: dict, max_age_hours: int = 48) -> bool:
    """True если рынок обновлялся недавно."""
    updated = market.get("updatedAt") or market.get("updated_at")
    if not updated:
        return True
    try:
        from datetime import timedelta
        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - dt
        return age < timedelta(hours=max_age_hours)
    except Exception:
        return True


async def fetch_events(session, limit: int = 200) -> List[dict]:
    try:
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(limit),
            "order": "volume24hr",
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(
            CFG.GAMMA_EVENTS_API, params=params,
            headers=DEFAULT_HEADERS, timeout=timeout,
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Events API: {resp.status}")
                return []
            events = await resp.json()

        multi = []
        for e in events:
            markets = e.get("markets", [])
            open_mkts = [
                m for m in markets
                if not m.get("closed", False) and check_staleness(m)
            ]
            if len(open_mkts) >= 3:
                e["markets"] = open_mkts
                multi.append(e)

        logger.info(
            f"Events: {len(events)} total, {len(multi)} multi-outcome"
        )
        return multi
    except Exception as e:
        logger.error(f"Events fetch: {e}")
        return []


def parse_event_outcomes(event: dict) -> tuple[str, str, List[Outcome]]:
    title = event.get("title", "Unknown")
    event_id = event.get("id", "")
    outcomes = []
    for m in event.get("markets", []):
        if m.get("closed", False):
            continue
        question = m.get("question", "")
        market_id = m.get("id", "")
        vol = float(m.get("volume24hr") or m.get("volumeNum") or 0)
        liquidity = float(m.get("liquidity") or m.get("liquidityNum") or 0)
        prices = parse_prices(m.get("outcomePrices"))
        if len(prices) < 2:
            continue
        yes_price = prices[0]
        if yes_price > 0.001:
            # v1.6: достаём token_id для CLOB API
            clob_ids = m.get("clobTokenIds") or m.get("clob_token_ids") or []
            token_id = clob_ids[0] if clob_ids else ""
            outcomes.append(Outcome(question, yes_price, vol, market_id, liquidity, token_id))
    return title, event_id, outcomes


def detect_fork(
    title: str,
    event_id: str,
    outcomes: List[Outcome],
    is_neg_risk: bool,
    min_profit_pct: float = 0.5,
    shares: int = 100,
) -> Optional[ForkOpportunity]:

    if len(outcomes) < 3:
        return None

    active = [o for o in outcomes if o.yes_price > 0.005]
    if len(active) < 3:
        return None

    sum_yes = sum(o.yes_price for o in active)
    if sum_yes > MAX_SUM_OVER or sum_yes < MIN_SUM_UNDER:
        return None
    if abs(sum_yes - 1.0) < 0.005:
        return None

    min_vol = min(o.volume for o in active)
    if min_vol < CFG.MIN_VOLUME:
        logger.debug(
            f"Fork skipped (low volume ${min_vol:.0f}): {title[:40]}"
        )
        return None

    min_liq = min(o.liquidity for o in active)

    if sum_yes < 1.0:
        raw_pct = ((1.0 - sum_yes) / sum_yes) * 100
        cost = shares * sum_yes
        fees = estimate_fees(len(active), cost, "under")
        net_cost = cost + fees
        net_profit = (shares * 1.0) - net_cost
        net_pct = (net_profit / net_cost) * 100
        if net_pct >= min_profit_pct:
            return ForkOpportunity(
                title, event_id, "under", active,
                sum_yes, raw_pct, net_pct,
                net_cost, net_profit, is_neg_risk,
                min_volume=min_vol, min_liquidity=min_liq,
            )
    else:
        raw_pct = ((sum_yes - 1.0) / 1.0) * 100
        mint_cost = shares * 1.0
        sell_revenue = shares * sum_yes
        fees = estimate_fees(len(active), sell_revenue, "over")
        net_revenue = sell_revenue - fees
        net_profit = net_revenue - mint_cost
        net_pct = (net_profit / mint_cost) * 100
        if net_pct >= min_profit_pct:
            return ForkOpportunity(
                title, event_id, "over", active,
                sum_yes, raw_pct, net_pct,
                mint_cost + fees, net_profit, is_neg_risk,
                min_volume=min_vol, min_liquidity=min_liq,
            )

    return None


def _log_depth_check(fork: ForkOpportunity, depth_result, position_usd: float):
    """Запись результата depth check в лог для анализа."""
    import time
    entry = {
        "ts": time.time(),
        "market_id": fork.event_id,
        "title": fork.event_title[:60],
        "fork_type": fork.fork_type,
        "gamma_edge_pct": round(fork.net_profit_pct, 3),
        "real_edge_pct": round(depth_result.real_edge_pct, 3),
        "min_depth_usd": round(depth_result.min_executable_usd, 2),
        "is_executable": depth_result.is_executable,
        "reject_reason": depth_result.reject_reason,
        "fetch_ms": round(depth_result.fetch_time_ms, 1),
        "position_usd": position_usd,
        "legs": [
            {
                "name": leg.outcome_name[:30],
                "mid": round(leg.mid_price, 4),
                "best_bid": leg.best_bid,
                "depth_usd": round(leg.ask_depth_usd, 2),
                "slippage_pct": round(leg.slippage_pct, 3),
                "error": leg.error,
            }
            for leg in depth_result.legs
        ],
    }
    try:
        with open("data/depth_checks.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"depth log write error: {e}")


async def scan_forks(session, min_profit: float = 0.5) -> List[ForkOpportunity]:
    events = await fetch_events(session)
    if not events:
        return []

    forks = []
    stats = {
        "total": 0, "neg_risk": 0, "title_match": 0, "skipped": 0,
        "depth_checked": 0, "depth_passed": 0, "depth_failed": 0,
    }

    # v1.6: инициализируем depth checker один раз на скан
    depth_checker = OrderbookChecker(
        session=session,
        min_executable_usd=float(getattr(CFG, "MIN_EXECUTABLE_USD", 5.0)),
        max_slippage_pct=float(getattr(CFG, "MAX_SLIPPAGE_PCT", 0.5)),
        depth_ticks=3,
        max_position_to_depth_ratio=0.3,
    )

    for event in events:
        stats["total"] += 1
        is_neg = check_neg_risk(event)
        is_title = check_exclusive_by_title(event)

        if is_neg:
            stats["neg_risk"] += 1
        elif is_title:
            stats["title_match"] += 1
        else:
            stats["skipped"] += 1
            continue

        title, event_id, outcomes = parse_event_outcomes(event)
        if len(outcomes) < 3:
            continue

        fork = detect_fork(title, event_id, outcomes, is_neg, min_profit_pct=min_profit)
        if not fork:
            continue

        # ── v1.6: depth check ──────────────────────────────────
        has_token_ids = any(o.token_id for o in fork.outcomes)

        if has_token_ids:
            stats["depth_checked"] += 1
            position_usd = float(getattr(CFG, "MIN_POSITION_USD", 8.0))

            depth_result = await depth_checker.check_fork(
                outcomes=[
                    {
                        "token_id": o.token_id,
                        "name": o.question,
                        "price": o.yes_price,
                    }
                    for o in fork.outcomes
                ],
                position_size_usd=position_usd,
                market_id=fork.event_id,
            )

            _log_depth_check(fork, depth_result, position_usd)

            if depth_result.is_executable:
                stats["depth_passed"] += 1
                fork.net_profit_pct = depth_result.real_edge_pct
                fork.expected_profit = depth_result.real_edge_usd
            else:
                stats["depth_failed"] += 1
                logger.info(
                    f"DEPTH FAIL | {title[:35]} | "
                    f"Gamma: {fork.net_profit_pct:.2f}% | "
                    f"Reject: {depth_result.reject_reason}"
                )
                continue
        else:
            logger.debug(f"No token_id for {title[:35]} — skipping depth check")
        # ── конец depth check ───────────────────────────────────

        forks.append(fork)
        verified = "negRisk" if is_neg else "title"
        logger.info(
            f"FORK | {fork.fork_type.upper()} | "
            f"{title[:40]} | Sum: ${fork.sum_yes:.4f} | "
            f"Net: {fork.net_profit_pct:.2f}% | "
            f"MinVol: ${fork.min_volume:.0f} | {verified}"
        )

    logger.info(
        f"Forks: {stats['total']} scanned, "
        f"{stats['neg_risk']} negRisk, "
        f"{stats['title_match']} title, "
        f"{stats['skipped']} skipped | "
        f"Depth: {stats['depth_checked']} checked, "
        f"{stats['depth_passed']} passed, "
        f"{stats['depth_failed']} failed | "
        f"{len(forks)} final"
    )
    return forks
