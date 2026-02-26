#!/bin/bash
set -e
cd ~/arb-bot

echo ">>> Creating config.py"
cat > config.py << 'PYEOF'
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    TELEGRAM_TOKEN = os.getenv("ARB_TG_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("ARB_TG_CHAT", "")
    KALSHI_API_KEY = os.getenv("ARB_KALSHI_KEY", "")

    POLYMARKET_API = "https://gamma-api.polymarket.com/markets"
    GAMMA_EVENTS_API = "https://gamma-api.polymarket.com/events"
    KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2/markets"
    POLYMARKET_CLOB = "https://clob.polymarket.com/book"

    SCAN_INTERVAL = 60
    SIMILARITY_THRESHOLD = 0.65
    MIN_PROFIT_PCT = 0.3
    MIN_VOLUME = 10000
    MIN_DAYS_TO_CLOSE = 1
    MIN_LIQUIDITY = 50
    RETRY_ATTEMPTS = 3
    FETCH_TIMEOUT = 15

    MANUAL_MAPPINGS_FILE = "data/manual_mappings.json"
    OPPORTUNITIES_FILE = "data/opportunities.jsonl"
    FORKS_FILE = "data/forks.jsonl"
    SPORTS_FILE = "data/sports_arbs.jsonl"
    LOG_FILE = "logs/arbitrage.log"

    POLY_TAKER_FEE = 0.02
    POLY_MAKER_FEE = 0.00
    KALSHI_FEE = 0.03
    POLY_WIN_FEE = 0.02

    MIN_PRICE = 0.05
    MAX_PRICE = 0.95


CFG = Config()
PYEOF

echo ">>> Creating matching/__init__.py"
cat > matching/__init__.py << 'PYEOF'
PYEOF

echo ">>> Creating matching/normalizer.py"
cat > matching/normalizer.py << 'PYEOF'
import re

NORMALIZE_MAP = {
    "bitcoin": "btc", "ethereum": "eth", "solana": "sol",
    "donald trump": "trump", "joe biden": "biden",
    "federal reserve": "fed", "united states": "us",
    "interest rate": "rate", "rate cut": "cut",
    "above": ">", "below": "<", "exceed": ">",
    "over": ">", "under": "<", "at least": ">=",
    "more than": ">", "less than": "<",
    "thousand": "k", ",000": "k",
    "million": "m", "billion": "b",
    "january": "jan", "february": "feb", "march": "mar",
    "april": "apr", "june": "jun", "july": "jul",
    "august": "aug", "september": "sep", "october": "oct",
    "november": "nov", "december": "dec",
    "super bowl": "superbowl", "nba finals": "nbafinals",
    "world cup": "worldcup",
}

STOP_WORDS = frozenset({
    "will", "the", "a", "an", "be", "is", "are", "to", "of",
    "in", "on", "at", "for", "this", "that", "it", "by", "or",
    "and", "do", "does", "did", "has", "have", "what", "when",
    "how", "much", "many", "next", "new",
})

_cache = {}


def normalize(text):
    if text in _cache:
        return _cache[text]
    result = text.lower().strip()
    result = re.sub(r"[?!.,;:'\"\-()\[\]{}]", " ", result)
    for old, new in NORMALIZE_MAP.items():
        result = result.replace(old, new)
    result = re.sub(r"\s+", " ", result).strip()
    tokens = [t for t in result.split() if t not in STOP_WORDS and len(t) > 1]
    normalized = " ".join(sorted(tokens))
    _cache[text] = normalized
    return normalized
PYEOF

echo ">>> Creating matching/fuzzy.py"
cat > matching/fuzzy.py << 'PYEOF'
import json
import logging
from difflib import SequenceMatcher
from config import CFG
from matching.normalizer import normalize

logger = logging.getLogger("arb_scanner.matcher")


class EventMatcher:
    def __init__(self):
        self.manual_map = {}
        self._load_manual()

    def _load_manual(self):
        try:
            with open(CFG.MANUAL_MAPPINGS_FILE) as f:
                raw = json.load(f)
            self.manual_map = {normalize(k): normalize(v) for k, v in raw.items()}
            logger.info(f"Loaded {len(self.manual_map)} manual mappings")
        except FileNotFoundError:
            logger.info("No manual mappings file")
        except Exception as e:
            logger.error(f"Manual mappings error: {e}")

    def find_match(self, poly_question, kalshi_markets, threshold=CFG.SIMILARITY_THRESHOLD):
        norm_poly = normalize(poly_question)

        if norm_poly in self.manual_map:
            target = self.manual_map[norm_poly]
            for k_key, k_mkt in kalshi_markets.items():
                if normalize(k_mkt.question) == target:
                    return k_key, 1.0, "manual"

        best_key = None
        best_score = 0.0
        poly_tokens = set(norm_poly.split())

        if not poly_tokens:
            return None, 0.0, "none"

        for k_key, k_mkt in kalshi_markets.items():
            norm_k = normalize(k_mkt.question)
            k_tokens = set(norm_k.split())
            if not k_tokens:
                continue

            intersection = poly_tokens & k_tokens
            union = poly_tokens | k_tokens
            jaccard = len(intersection) / len(union)
            seq = SequenceMatcher(None, norm_poly, norm_k).ratio()
            combined = jaccard * 0.6 + seq * 0.4

            if combined > best_score:
                best_score = combined
                best_key = k_key

        min_tokens = min(len(poly_tokens), 3)
        adj = threshold - (0.05 * max(0, 5 - min_tokens))
        adj = max(adj, 0.50)

        if best_score >= adj and best_key is not None:
            return best_key, best_score, "fuzzy"

        return None, 0.0, "none"
PYEOF

echo ">>> Creating strategy/__init__.py"
cat > strategy/__init__.py << 'PYEOF'
PYEOF

echo ">>> Creating strategy/fork_scanner.py"
cat > strategy/fork_scanner.py << 'PYEOF'
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from config import CFG

logger = logging.getLogger("arb_scanner.forks")

POLY_TAKER_FEE = 0.02
MERGE_GAS_USD = 0.10
SWAP_GAS_PER_OUTCOME = 0.02


@dataclass
class Outcome:
    question: str
    token_id: str
    yes_price: float
    no_price: float
    volume: float
    market_id: str


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
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def format_alert(self):
        outcomes_str = "\n".join(
            f"   {o.question[:50]}: YES ${o.yes_price:.3f} (vol ${o.volume:,.0f})"
            for o in self.outcomes
        )
        return (
            f"?? FORK ({self.fork_type.upper()})\n\n"
            f"?? {self.event_title}\n\n"
            f"?? Outcomes:\n{outcomes_str}\n\n"
            f"?? Sum YES: ${self.sum_yes:.4f}\n"
            f"?? Net profit: {self.net_profit_pct:.2f}%\n"
            f"?? Cost: ${self.cost_to_execute:.2f}\n"
            f"?? Profit: ${self.expected_profit:.2f}"
        )

    def to_dict(self):
        return {
            "event": self.event_title, "event_id": self.event_id,
            "type": self.fork_type, "sum_yes": self.sum_yes,
            "raw_profit_pct": self.raw_profit_pct,
            "net_profit_pct": self.net_profit_pct,
            "cost": self.cost_to_execute,
            "expected_profit": self.expected_profit,
            "num_outcomes": len(self.outcomes),
            "timestamp": self.timestamp,
        }


def estimate_fees(num_outcomes, total_cost, fork_type):
    trading_fees = total_cost * POLY_TAKER_FEE
    if fork_type == "under":
        gas = (num_outcomes * SWAP_GAS_PER_OUTCOME) + MERGE_GAS_USD
    else:
        gas = MERGE_GAS_USD + (num_outcomes * SWAP_GAS_PER_OUTCOME)
    return trading_fees + gas


async def fetch_events(session, limit=100):
    try:
        params = {"active": "true", "limit": limit, "order": "volume", "ascending": "false"}
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(CFG.GAMMA_EVENTS_API, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning(f"Events API: {resp.status}")
                return []
            events = await resp.json()
            multi = [e for e in events if len(e.get("markets", [])) >= 3]
            logger.info(f"Events: {len(events)} total, {len(multi)} multi-outcome")
            return multi
    except Exception as e:
        logger.error(f"Events fetch: {e}")
        return []


def parse_outcomes(event):
    title = event.get("title", "Unknown")
    event_id = event.get("id", "")
    outcomes = []
    for m in event.get("markets", []):
        question = m.get("question", "")
        market_id = m.get("id", "")
        tokens = m.get("tokens", [])
        volume = float(m.get("volume") or 0)
        yes_price = no_price = None
        token_id = ""
        for tok in tokens:
            outcome = (tok.get("outcome") or "").lower()
            price = float(tok.get("price") or 0)
            if outcome == "yes":
                yes_price = price
                token_id = tok.get("token_id", "")
            elif outcome == "no":
                no_price = price
        if yes_price is not None and yes_price > 0.001:
            if no_price is None:
                no_price = 1.0 - yes_price
            outcomes.append(Outcome(question, token_id, yes_price, no_price, volume, market_id))
    return title, event_id, outcomes


def detect_fork(title, event_id, outcomes, min_profit_pct=0.3, min_volume=5000, shares=100):
    if len(outcomes) < 3:
        return None
    active = [o for o in outcomes if o.yes_price > 0.005]
    if len(active) < 3:
        return None
    low_vol = [o for o in active if o.volume < min_volume]
    if len(low_vol) > len(active) * 0.5:
        return None

    sum_yes = sum(o.yes_price for o in active)

    if sum_yes < 1.0:
        raw_pct = ((1.0 - sum_yes) / sum_yes) * 100
        cost = shares * sum_yes
        fees = estimate_fees(len(active), cost, "under")
        net_cost = cost + fees
        payout = shares * 1.0
        net_profit = payout - net_cost
        net_pct = (net_profit / net_cost) * 100
        if net_pct >= min_profit_pct:
            return ForkOpportunity(title, event_id, "under", active, sum_yes, raw_pct, net_pct, net_cost, net_profit)

    elif sum_yes > 1.0:
        raw_pct = ((sum_yes - 1.0) / 1.0) * 100
        mint_cost = shares * 1.0
        sell_revenue = shares * sum_yes
        fees = estimate_fees(len(active), sell_revenue, "over")
        net_revenue = sell_revenue - fees
        net_profit = net_revenue - mint_cost
        net_pct = (net_profit / mint_cost) * 100
        if net_pct >= min_profit_pct:
            return ForkOpportunity(title, event_id, "over", active, sum_yes, raw_pct, net_pct, mint_cost + fees, net_profit)

    return None


async def scan_forks(session, min_profit=0.3):
    events = await fetch_events(session)
    if not events:
        return []
    forks = []
    for event in events:
        title, event_id, outcomes = parse_outcomes(event)
        if len(outcomes) < 3:
            continue
        fork = detect_fork(title, event_id, outcomes, min_profit_pct=min_profit)
        if fork:
            forks.append(fork)
            logger.info(f"FORK | {fork.fork_type.upper()} | {title[:50]} | Sum: ${fork.sum_yes:.4f} | Net: {fork.net_profit_pct:.2f}%")
    return forks
PYEOF

echo ">>> Creating strategy/sports_binary.py"
cat > strategy/sports_binary.py << 'PYEOF'
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from config import CFG

logger = logging.getLogger("arb_scanner.sports")

SPORTS_TAGS = [
    "nba", "nfl", "mlb", "nhl", "ncaa", "soccer", "football",
    "premier league", "champions league", "mls", "tennis",
    "olympics", "boxing", "mma", "ufc", "cricket", "rugby",
    "hockey", "basketball", "baseball",
]

YES_ARB_THRESHOLD = 0.98
NO_ARB_THRESHOLD = 0.98
POLY_WIN_FEE = 0.02
MIN_EDGE = 0.002


@dataclass
class BinaryMatch:
    event_title: str
    event_id: str
    team_a: str
    team_b: str
    yes_a: float
    yes_b: float
    no_a: float
    no_b: float
    token_a: str
    token_b: str
    market_a_id: str
    market_b_id: str
    volume_a: float
    volume_b: float


@dataclass
class SportsArb:
    match: BinaryMatch
    arb_type: str
    sum_prices: float
    raw_edge: float
    net_edge: float
    net_edge_pct: float
    cost_per_share: float
    payout_per_share: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def format_alert(self):
        m = self.match
        return (
            f"⚽ SPORTS ARB ({self.arb_type.upper()})\n\n"
            f"?? {m.event_title}\n\n"
            f"{m.team_a[:30]}: YES ${m.yes_a:.3f}\n"
            f"{m.team_b[:30]}: YES ${m.yes_b:.3f}\n\n"
            f"?? Sum: ${self.sum_prices:.4f}\n"
            f"?? Net edge: {self.net_edge_pct:.2f}%\n"
            f"?? Cost/share: ${self.cost_per_share:.4f}\n"
            f"?? Payout/share: ${self.payout_per_share:.4f}"
        )

    def to_dict(self):
        return {
            "event": self.match.event_title,
            "team_a": self.match.team_a, "team_b": self.match.team_b,
            "arb_type": self.arb_type, "yes_a": self.match.yes_a,
            "yes_b": self.match.yes_b, "sum": self.sum_prices,
            "net_edge_pct": self.net_edge_pct,
            "volume_a": self.match.volume_a, "volume_b": self.match.volume_b,
            "timestamp": self.timestamp,
        }


def is_sports(event):
    title = (event.get("title") or "").lower()
    desc = (event.get("description") or "").lower()
    tags = [t.lower() for t in (event.get("tags") or [])]
    text = f"{title} {desc} {' '.join(tags)}"
    return any(s in text for s in SPORTS_TAGS)


def parse_binary(event):
    markets = event.get("markets", [])
    if len(markets) != 2:
        return None
    title = event.get("title", "")
    event_id = event.get("id", "")
    sides = []
    for m in markets:
        question = m.get("question", "")
        market_id = m.get("id", "")
        volume = float(m.get("volume") or 0)
        tokens = m.get("tokens", [])
        yes_price = no_price = None
        token_id = ""
        for tok in tokens:
            outcome = (tok.get("outcome") or "").lower()
            price = float(tok.get("price") or 0)
            if outcome == "yes":
                yes_price = price
                token_id = tok.get("token_id", "")
            elif outcome == "no":
                no_price = price
        if yes_price is None or yes_price <= 0.01:
            return None
        if no_price is None:
            no_price = 1.0 - yes_price
        sides.append({"q": question, "yes": yes_price, "no": no_price, "tid": token_id, "mid": market_id, "vol": volume})
    return BinaryMatch(title, event_id, sides[0]["q"], sides[1]["q"],
                       sides[0]["yes"], sides[1]["yes"], sides[0]["no"], sides[1]["no"],
                       sides[0]["tid"], sides[1]["tid"], sides[0]["mid"], sides[1]["mid"],
                       sides[0]["vol"], sides[1]["vol"])


def detect_arb(match):
    sum_yes = match.yes_a + match.yes_b
    if sum_yes < YES_ARB_THRESHOLD:
        cost = sum_yes
        payout = 1.0 * (1 - POLY_WIN_FEE)
        edge = payout - cost
        if edge > MIN_EDGE:
            return SportsArb(match, "yes", sum_yes, 1.0 - sum_yes, edge, edge / cost * 100, cost, payout)

    sum_no = match.no_a + match.no_b
    if sum_no < NO_ARB_THRESHOLD:
        cost = sum_no
        payout = 1.0 * (1 - POLY_WIN_FEE)
        edge = payout - cost
        if edge > MIN_EDGE:
            return SportsArb(match, "no", sum_no, 1.0 - sum_no, edge, edge / cost * 100, cost, payout)

    return None


async def scan_sports(session):
    try:
        params = {"active": "true", "limit": 200, "order": "volume", "ascending": "false"}
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(CFG.GAMMA_EVENTS_API, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning(f"Events API: {resp.status}")
                return []
            events = await resp.json()
    except Exception as e:
        logger.error(f"Fetch: {e}")
        return []

    sports_binary = [e for e in events if is_sports(e) and len(e.get("markets", [])) == 2]
    logger.info(f"Sports binary: {len(sports_binary)} out of {len(events)}")

    results = []
    for event in sports_binary:
        match = parse_binary(event)
        if not match:
            continue
        arb = detect_arb(match)
        if arb:
            results.append(arb)
            logger.info(f"SPORTS ARB | {arb.arb_type.upper()} | {match.team_a[:20]} vs {match.team_b[:20]} | Sum: ${arb.sum_prices:.4f} | Edge: {arb.net_edge_pct:.2f}%")
    return results
PYEOF

echo ">>> Creating execution/__init__.py"
cat > execution/__init__.py << 'PYEOF'
PYEOF

echo ">>> Creating execution/compounder.py"
cat > execution/compounder.py << 'PYEOF'
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("arb_scanner.compounder")


class CompoundingManager:
    def __init__(self, initial_deposit=80.0, max_risk_pct=5.0, max_drawdown_pct=20.0, state_file="data/compound_state.json"):
        self.initial_deposit = initial_deposit
        self.max_risk_pct = max_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.state_file = Path(state_file)
        self.total_deposited = initial_deposit
        self.bankroll = initial_deposit
        self.total_profit = 0.0
        self.peak_bankroll = initial_deposit
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_trades = 0
        self.last_date = ""
        self.is_stopped = False
        self.stop_reason = ""
        self._load_state()

    def add_deposit(self, amount):
        self.total_deposited += amount
        self.bankroll += amount
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        logger.info(f"DEPOSIT +${amount:.2f} | Total: ${self.total_deposited:.2f} | Bankroll: ${self.bankroll:.2f}")
        self._save_state()

    def get_position_size(self):
        if self.is_stopped:
            return 0
        size = self.bankroll * (self.max_risk_pct / 100)
        if size < 2.0:
            return 0
        return size

    def record_trade(self, profit, details=None):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_date:
            self.daily_trades = 0
            self.last_date = today

        self.trade_count += 1
        self.daily_trades += 1
        self.total_profit += profit
        self.bankroll += profit

        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        dd = ((self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100) if self.peak_bankroll > 0 else 0
        if dd > self.max_drawdown_pct:
            self.is_stopped = True
            self.stop_reason = f"Drawdown {dd:.1f}%"
            logger.warning(f"STOP: {self.stop_reason}")

        wr = (self.winning_trades / (self.winning_trades + self.losing_trades) * 100) if (self.winning_trades + self.losing_trades) > 0 else 0
        logger.info(f"Trade #{self.trade_count} | PnL: ${profit:+.4f} | Bank: ${self.bankroll:.2f} | Total: ${self.total_profit:+.2f} | WR: {wr:.0f}%")

        if details:
            record = {"n": self.trade_count, "pnl": round(profit, 4), "bank": round(self.bankroll, 2), "ts": datetime.now(timezone.utc).isoformat()}
            record.update(details)
            try:
                with open("data/compound_trades.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

        self._save_state()

    def get_stats(self):
        total = self.winning_trades + self.losing_trades
        dd = ((self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100) if self.peak_bankroll > 0 else 0
        return {
            "bankroll": round(self.bankroll, 2),
            "deposited": round(self.total_deposited, 2),
            "profit": round(self.total_profit, 2),
            "roi": round((self.total_profit / self.total_deposited * 100) if self.total_deposited > 0 else 0, 1),
            "trades": self.trade_count,
            "win_rate": round((self.winning_trades / total * 100) if total > 0 else 0, 1),
            "position": round(self.get_position_size(), 2),
            "drawdown": round(dd, 1),
            "growth": round((self.bankroll / self.total_deposited) if self.total_deposited > 0 else 1, 2),
            "stopped": self.is_stopped,
        }

    def print_stats(self):
        s = self.get_stats()
        st = "STOPPED" if s["stopped"] else "ACTIVE"
        logger.info(
            f"\n{'='*45}\n"
            f" PORTFOLIO — {st}\n"
            f"{'='*45}\n"
            f" Bankroll:  ${s['bankroll']:>10,.2f}\n"
            f" Deposited: ${s['deposited']:>10,.2f}\n"
            f" Profit:    ${s['profit']:>+10,.2f}\n"
            f" ROI:       {s['roi']:>+9.1f}%\n"
            f" Growth:    {s['growth']:>9.2f}x\n"
            f" Trades:    {s['trades']:>10}\n"
            f" Win rate:  {s['win_rate']:>9.1f}%\n"
            f" Position:  ${s['position']:>10.2f}\n"
            f" Drawdown:  {s['drawdown']:>9.1f}%\n"
            f"{'='*45}"
        )

    def reset_stop(self):
        self.is_stopped = False
        self.stop_reason = ""
        self._save_state()

    def _save_state(self):
        state = {
            "bankroll": self.bankroll, "total_deposited": self.total_deposited,
            "total_profit": self.total_profit, "peak_bankroll": self.peak_bankroll,
            "trade_count": self.trade_count, "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades, "daily_trades": self.daily_trades,
            "last_date": self.last_date, "is_stopped": self.is_stopped,
            "stop_reason": self.stop_reason,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Save: {e}")

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            s = json.loads(self.state_file.read_text())
            self.bankroll = s["bankroll"]
            self.total_deposited = s["total_deposited"]
            self.total_profit = s["total_profit"]
            self.peak_bankroll = s["peak_bankroll"]
            self.trade_count = s["trade_count"]
            self.winning_trades = s["winning_trades"]
            self.losing_trades = s["losing_trades"]
            self.daily_trades = s.get("daily_trades", 0)
            self.last_date = s.get("last_date", "")
            self.is_stopped = s.get("is_stopped", False)
            self.stop_reason = s.get("stop_reason", "")
            logger.info(f"Loaded: bank=${self.bankroll:.2f} profit=${self.total_profit:+.2f}")
        except Exception as e:
            logger.error(f"Load: {e}")
PYEOF

echo ">>> Creating notifications/__init__.py"
cat > notifications/__init__.py << 'PYEOF'
PYEOF

echo ">>> Creating notifications/telegram.py"
cat > notifications/telegram.py << 'PYEOF'
import aiohttp
import logging
from config import CFG

logger = logging.getLogger("arb_scanner.telegram")


async def send_telegram(session, text):
    if not CFG.TELEGRAM_TOKEN or not CFG.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CFG.TELEGRAM_CHAT_ID, "text": text}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning(f"TG failed: {resp.status}")
    except Exception as e:
        logger.warning(f"TG error: {e}")


async def check_commands(session, compounder):
    if not CFG.TELEGRAM_TOKEN:
        return
    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/getUpdates?offset=-1&timeout=1"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return
            data = await resp.json()
        results = data.get("result", [])
        if not results:
            return
        msg = results[-1].get("message", {})
        text = (msg.get("text") or "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        msg_id = msg.get("message_id", 0)

        if chat_id != CFG.TELEGRAM_CHAT_ID:
            return

        # Deduplicate
        last_file = "data/.last_msg_id"
        try:
            with open(last_file) as f:
                last_id = int(f.read().strip())
            if msg_id <= last_id:
                return
        except Exception:
            pass
        with open(last_file, "w") as f:
            f.write(str(msg_id))

        if text == "/stats":
            s = compounder.get_stats()
            reply = (
                f"?? Portfolio\n\n"
                f"?? Bank: ${s['bankroll']:,.2f}\n"
                f"?? Deposited: ${s['deposited']:,.2f}\n"
                f"?? Profit: ${s['profit']:+,.2f}\n"
                f"?? ROI: {s['roi']:+.1f}%\n"
                f"?? Growth: {s['growth']}x\n"
                f"?? Position: ${s['position']:,.2f}\n"
                f"?? WR: {s['win_rate']:.0f}% ({s['trades']} trades)\n"
                f"?? DD: {s['drawdown']:.1f}%"
            )
            await send_telegram(session, reply)

        elif text.startswith("/deposit"):
            try:
                amount = float(text.split()[1])
                compounder.add_deposit(amount)
                await send_telegram(session, f"✅ +${amount:.2f}\nBank: ${compounder.bankroll:.2f}")
            except Exception:
                await send_telegram(session, "Usage: /deposit 160")

        elif text == "/reset":
            compounder.reset_stop()
            await send_telegram(session, "✅ Stop reset")

        elif text == "/size":
            await send_telegram(session, f"Position: ${compounder.get_position_size():.2f}\nBank: ${compounder.bankroll:.2f}")

    except Exception:
        pass
PYEOF

echo ">>> Creating main.py"
cat > main.py << 'PYEOF'
import asyncio
import aiohttp
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

from config import CFG
from matching.normalizer import normalize
from matching.fuzzy import EventMatcher
from strategy.fork_scanner import scan_forks
from strategy.sports_binary import scan_sports
from execution.compounder import CompoundingManager
from notifications.telegram import send_telegram, check_commands


def setup_logging():
    log = logging.getLogger("arb_scanner")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(CFG.LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


logger = setup_logging()


@dataclass
class Market:
    platform: str
    question: str
    yes_price: float
    volume: float
    market_id: str
    close_time: str = ""


@dataclass
class ArbResult:
    profit: float
    profit_pct: float
    variant: int
    total_cost: float
    is_profitable: bool
    poly_side: str = ""
    kalshi_side: str = ""


def is_tradeable(volume, close_time=""):
    if volume < CFG.MIN_VOLUME:
        return False
    if close_time:
        try:
            dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            if (dt - datetime.now(timezone.utc)).days < CFG.MIN_DAYS_TO_CLOSE:
                return False
        except Exception:
            pass
    return True


def calculate_arb(poly_yes, kalshi_yes):
    pf = CFG.POLY_TAKER_FEE
    kf = CFG.KALSHI_FEE
    c1 = poly_yes * (1 + pf) + (1 - kalshi_yes) + kf
    c2 = (1 - poly_yes) * (1 + pf) + kalshi_yes + kf
    p1, p2 = 1.0 - c1, 1.0 - c2
    if p1 >= p2:
        return ArbResult(p1, p1*100, 1, c1, p1 > CFG.MIN_PROFIT_PCT/100, "YES", "NO")
    return ArbResult(p2, p2*100, 2, c2, p2 > CFG.MIN_PROFIT_PCT/100, "NO", "YES")


async def _retry(coro, *args, attempts=CFG.RETRY_ATTEMPTS):
    for i in range(attempts):
        try:
            return await coro(*args)
        except Exception as e:
            logger.warning(f"Attempt {i+1}/{attempts}: {e}")
            if i == attempts - 1:
                return {}
            await asyncio.sleep(2 ** i)


async def _fetch_poly(session):
    params = {"active": "true", "limit": 100, "order": "volume"}
    async with session.get(CFG.POLYMARKET_API, params=params, timeout=aiohttp.ClientTimeout(total=CFG.FETCH_TIMEOUT)) as r:
        if r.status != 200:
            raise ConnectionError(f"Poly: {r.status}")
        data = await r.json()
    markets = {}
    for m in data:
        q = (m.get("question") or "").strip()
        toks = m.get("tokens") or []
        vol = float(m.get("volume") or 0)
        end = m.get("end_date_iso") or ""
        mid = m.get("id") or ""
        if not q or len(toks) < 2 or not is_tradeable(vol, end):
            continue
        yp = None
        for t in toks:
            if (t.get("outcome") or "").lower() == "yes":
                yp = float(t.get("price") or 0)
                break
        if yp is None:
            for t in toks:
                if (t.get("outcome") or "").lower() == "no":
                    p = float(t.get("price") or 0)
                    if 0 < p < 1:
                        yp = 1.0 - p
                    break
        if yp and CFG.MIN_PRICE < yp < CFG.MAX_PRICE:
            markets[q.lower()] = Market("polymarket", q, yp, vol, mid, end)
    logger.info(f"Poly: {len(markets)} markets")
    return markets


async def _fetch_kalshi(session):
    headers = {}
    if CFG.KALSHI_API_KEY:
        headers["Authorization"] = f"Bearer {CFG.KALSHI_API_KEY}"
    params = {"status": "open", "limit": 100}
    async with session.get(CFG.KALSHI_API, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=CFG.FETCH_TIMEOUT)) as r:
        if r.status != 200:
            raise ConnectionError(f"Kalshi: {r.status}")
        data = await r.json()
    markets = {}
    for m in data.get("markets", []):
        title = (m.get("title") or "").strip()
        ticker = m.get("ticker") or ""
        yb = m.get("yes_bid")
        vol = m.get("volume") or 0
        ct = m.get("close_time") or ""
        if not title or yb is None or not is_tradeable(vol, ct):
            continue
        yp = yb / 100.0
        if CFG.MIN_PRICE < yp < CFG.MAX_PRICE:
            markets[title.lower()] = Market("kalshi", title, yp, vol, ticker, ct)
    logger.info(f"Kalshi: {len(markets)} markets")
    return markets


async def scan_cross_platform(session, matcher):
    poly, kalshi = await asyncio.gather(
        _retry(_fetch_poly, session),
        _retry(_fetch_kalshi, session),
    )
    if not poly or not kalshi:
        return []
    opps = []
    for pk, pm in poly.items():
        kk, conf, mtype = matcher.find_match(pm.question, kalshi)
        if kk is None:
            continue
        km = kalshi[kk]
        arb = calculate_arb(pm.yes_price, km.yes_price)
        if not arb.is_profitable:
            continue
        opps.append({"type": "cross", "poly": pm.question, "kalshi": km.question,
                      "poly_yes": pm.yes_price, "kalshi_yes": km.yes_price,
                      "profit_pct": arb.profit_pct, "confidence": conf, "match_type": mtype})
        logger.info(f"CROSS ARB | {pm.question[:50]} | Profit: {arb.profit_pct:.2f}% | {conf:.0%} ({mtype})")
    return opps


async def main():
    logger.info("=" * 50)
    logger.info("Arbitrage Scanner v1.0 — COMPOUND MODE")
    logger.info(f"Interval: {CFG.SCAN_INTERVAL}s")
    logger.info("Strategies: Cross-platform + Forks + Sports")
    logger.info("=" * 50)

    matcher = EventMatcher()
    compounder = CompoundingManager(initial_deposit=80.0)

    n = 0
    total_opps = 0

    try:
        async with aiohttp.ClientSession() as session:
            # Startup alert
            await send_telegram(session, "?? Arbitrage Scanner started!\nStrategies: Cross + Forks + Sports\nMode: DRY RUN (virtual)")

            while True:
                n += 1
                logger.info(f"--- Scan #{n} ---")

                try:
                    # Check Telegram commands
                    await check_commands(session, compounder)

                    # 1. Cross-platform arb
                    cross_opps = await scan_cross_platform(session, matcher)
                    for opp in cross_opps:
                        pos = compounder.get_position_size()
                        if pos > 0:
                            profit = pos * (opp["profit_pct"] / 100)
                            compounder.record_trade(profit, opp)
                            await send_telegram(session,
                                f"?? CROSS ARB\n{opp['poly'][:50]}\nProfit: {opp['profit_pct']:.2f}%\nVirtual PnL: ${profit:+.4f}")

                    # 2. Multi-outcome forks
                    forks = await scan_forks(session)
                    for fork in forks:
                        pos = compounder.get_position_size()
                        if pos > 0:
                            profit = pos * (fork.net_profit_pct / 100)
                            compounder.record_trade(profit, fork.to_dict())
                            await send_telegram(session, fork.format_alert())
                        try:
                            with open(CFG.FORKS_FILE, "a") as f:
                                f.write(json.dumps(fork.to_dict()) + "\n")
                        except Exception:
                            pass

                    # 3. Sports binary arb
                    sports = await scan_sports(session)
                    for arb in sports:
                        pos = compounder.get_position_size()
                        if pos > 0:
                            profit = pos * (arb.net_edge_pct / 100)
                            compounder.record_trade(profit, arb.to_dict())
                            await send_telegram(session, arb.format_alert())
                        try:
                            with open(CFG.SPORTS_FILE, "a") as f:
                                f.write(json.dumps(arb.to_dict()) + "\n")
                        except Exception:
                            pass

                    total_found = len(cross_opps) + len(forks) + len(sports)
                    total_opps += total_found
                    logger.info(f"Found: {len(cross_opps)} cross + {len(forks)} forks + {len(sports)} sports = {total_found} (total: {total_opps})")

                except Exception as e:
                    logger.error(f"Scan error: {e}", exc_info=True)

                # Stats every 10 scans
                if n % 10 == 0:
                    compounder.print_stats()

                await asyncio.sleep(CFG.SCAN_INTERVAL)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Stopped")
    finally:
        compounder.print_stats()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
PYEOF

echo ">>> Creating manual mappings"
cat > data/manual_mappings.json << 'JSONEOF'
{
    "Will Bitcoin exceed $100,000 by December 31, 2026?": "Bitcoin above $100,000 on 12/31/26",
    "Will the Fed cut rates in July 2026?": "Fed rate cut July 2026",
    "Will Trump win the 2028 presidential election?": "2028 Presidential Election: Republican nominee wins",
    "Will BTC reach $150k by end of 2026?": "Bitcoin above $150,000 on 12/31/26",
    "Will ETH reach $10,000 by 2026?": "Ethereum above $10,000 on 12/31/26",
    "Will S&P 500 close above 6000 by June 2026?": "S&P 500 above 6000 on 6/30/26"
}
JSONEOF

echo ">>> Creating requirements.txt"
cat > requirements.txt << 'REQEOF'
aiohttp>=3.9,<4.0
REQEOF

echo ">>> Creating Dockerfile"
cat > Dockerfile << 'DOCKEOF'
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/logs
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "main.py"]
DOCKEOF

echo ">>> Creating docker-compose.yml"
cat > docker-compose.yml << 'COMPEOF'
version: "3.8"
services:
  scanner:
    build: .
    container_name: arb-scanner
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "0.5"
COMPEOF

echo ">>> Creating .env.example"
cat > .env.example << 'ENVEOF'
ARB_TG_TOKEN=your_telegram_bot_token
ARB_TG_CHAT=your_telegram_chat_id
ARB_KALSHI_KEY=
ENVEOF

echo ""
echo "=========================================="
echo "  PROJECT CREATED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env"
echo "  2. nano .env        (fill TG token + chat ID)"
echo "  3. docker compose up -d --build"
echo "  4. docker logs -f arb-scanner"
echo ""

