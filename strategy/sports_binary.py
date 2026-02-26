import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
from config import CFG

logger = logging.getLogger("arb_scanner.sports")

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

SPORTS_TAGS = [
    "nba", "nfl", "mlb", "nhl", "ncaa", "soccer", "football",
    "premier league", "champions league", "mls", "tennis",
    "olympics", "boxing", "mma", "ufc", "cricket", "rugby",
    "hockey", "basketball", "baseball", "f1", "formula",
    "nascar", "golf", "esports", "lol", "dota", "ipl",
]

# Skip events with these in title (not simple win/lose)
SKIP_KEYWORDS = [
    "more markets", "total goals", "total points",
    "over/under", "correct score", "handicap",
    "spread", "both teams to score", "btts",
    "first half", "second half", "draw",
]

YES_ARB_THRESHOLD = 0.98
NO_ARB_THRESHOLD = 0.98
POLY_WIN_FEE = 0.02
MIN_EDGE = 0.002


def parse_prices(raw):
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


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


def extract_tag_strings(tags_raw):
    if not tags_raw:
        return []
    result = []
    for tag in tags_raw:
        if isinstance(tag, str):
            result.append(tag.lower())
        elif isinstance(tag, dict):
            for key in ("label", "name", "slug", "value"):
                val = tag.get(key)
                if val and isinstance(val, str):
                    result.append(val.lower())
                    break
    return result


def is_sports(event):
    title = (event.get("title") or "").lower()
    desc = (event.get("description") or "").lower()
    tags = extract_tag_strings(event.get("tags"))
    slug = (event.get("slug") or "").lower()
    text = f"{title} {desc} {slug} {' '.join(tags)}"

    # Skip non-simple markets
    for skip in SKIP_KEYWORDS:
        if skip in title:
            return False

    return any(s in text for s in SPORTS_TAGS)


def parse_binary(event):
    markets = event.get("markets", [])
    open_markets = [m for m in markets if not m.get("closed", False)]
    if len(open_markets) != 2:
        return None

    title = event.get("title", "")
    event_id = event.get("id", "")
    sides = []

    for m in open_markets:
        question = m.get("question", "")
        vol = float(m.get("volume24hr") or m.get("volumeNum") or 0)
        prices = parse_prices(m.get("outcomePrices"))

        if len(prices) < 2:
            return None

        yes_price = prices[0]
        no_price = prices[1] if len(prices) > 1 else 1.0 - yes_price

        if yes_price <= 0.01:
            return None

        sides.append({"q": question, "yes": yes_price, "no": no_price, "vol": vol})

    return BinaryMatch(
        title, event_id,
        sides[0]["q"], sides[1]["q"],
        sides[0]["yes"], sides[1]["yes"],
        sides[0]["no"], sides[1]["no"],
        sides[0]["vol"], sides[1]["vol"],
    )


def detect_arb(match):
    sum_yes = match.yes_a + match.yes_b
    if sum_yes < YES_ARB_THRESHOLD:
        cost = sum_yes
        payout = 1.0 * (1 - POLY_WIN_FEE)
        edge = payout - cost
        if edge > MIN_EDGE:
            return SportsArb(
                match, "yes", sum_yes, 1.0 - sum_yes,
                edge, edge / cost * 100, cost, payout
            )

    sum_no = match.no_a + match.no_b
    if sum_no < NO_ARB_THRESHOLD:
        cost = sum_no
        payout = 1.0 * (1 - POLY_WIN_FEE)
        edge = payout - cost
        if edge > MIN_EDGE:
            return SportsArb(
                match, "no", sum_no, 1.0 - sum_no,
                edge, edge / cost * 100, cost, payout
            )

    return None


async def scan_sports(session):
    try:
        params = {
            "active": "true",
            "closed": "false",
            "limit": "200",
            "order": "volume24hr",
        }
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(
            CFG.GAMMA_EVENTS_API, params=params,
            headers=DEFAULT_HEADERS, timeout=timeout
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Events API: {resp.status}")
                return []
            events = await resp.json()
    except Exception as e:
        logger.error(f"Fetch: {e}")
        return []

    sports_candidates = []
    for e in events:
        if not is_sports(e):
            continue
        open_markets = [m for m in e.get("markets", []) if not m.get("closed", False)]
        if len(open_markets) == 2:
            e["markets"] = open_markets
            sports_candidates.append(e)

    logger.info(f"Sports binary: {len(sports_candidates)} candidates out of {len(events)} events")

    results = []
    for event in sports_candidates:
        match = parse_binary(event)
        if not match:
            continue
        arb = detect_arb(match)
        if arb:
            results.append(arb)
            logger.info(
                f"SPORTS ARB | {arb.arb_type.upper()} | "
                f"{match.team_a[:20]} vs {match.team_b[:20]} | "
                f"Sum: ${arb.sum_prices:.4f} | Edge: {arb.net_edge_pct:.2f}%"
            )
    return results
