import os
import re
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger("arb_scanner.config")

# ── параметры, которые можно менять без рестарта ──────────────────
RELOADABLE = {
    "MIN_PROFIT_PCT", "MIN_LIQUIDITY", "MIN_VOLUME",
    "SCAN_INTERVAL", "MAX_EDGE_PCT", "MAX_POSITION_PCT",
    "MAX_DAILY_LOSS_PCT", "COOLDOWN_AFTER_LOSSES",
}


class Config:
    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("ARB_TG_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("ARB_TG_CHAT", "")
    KALSHI_API_KEY: str = os.getenv("ARB_KALSHI_KEY", "")

    # API
    POLYMARKET_API: str = "https://gamma-api.polymarket.com/markets"
    GAMMA_EVENTS_API: str = "https://gamma-api.polymarket.com/events"
    KALSHI_API: str = "https://trading-api.kalshi.com/trade-api/v2/markets"
    POLYMARKET_CLOB: str = "https://clob.polymarket.com/book"

    # Scanning
    SCAN_INTERVAL: int = 60
    SIMILARITY_THRESHOLD: float = 0.65
    MIN_PROFIT_PCT: float = 0.5          # мин. профит после fees
    MIN_VOLUME: float = 0              # мин. объём рынка $
    MIN_DAYS_TO_CLOSE: int = 1
    MIN_LIQUIDITY: float = 50            # мин. ликвидность $
    RETRY_ATTEMPTS: int = 3
    FETCH_TIMEOUT: int = 15

    # Risk / новое в v1.5
    MAX_EDGE_PCT: float = 12.0           # выше → ложная вилка, пропускаем
    MAX_POSITION_PCT: float = 5.0        # % банкролла на сделку
    MAX_DAILY_LOSS_PCT: float = 10.0     # дневной стоп
    MAX_TRADES_PER_HOUR: int = 20        # защита от дублей
    COOLDOWN_AFTER_LOSSES: int = 3       # пауза (в сканах) после N убытков подряд
    MAX_DRAWDOWN_PCT: float = 20.0

    # Liquidity check / новое в v1.5
    MAX_POSITION_TO_VOLUME_RATIO: float = 999.0   # позиция / volume24hr
    MIN_ORDERBOOK_DEPTH: float = 20.0            # мин. глубина по каждому исходу $

    # Fees
    POLY_TAKER_FEE: float = 0.02
    POLY_MAKER_FEE: float = 0.00
    KALSHI_FEE: float = 0.03
    POLY_WIN_FEE: float = 0.02

    # Price guards
    MIN_PRICE: float = 0.05
    MAX_PRICE: float = 0.95

    # Files
    MANUAL_MAPPINGS_FILE: str = "data/manual_mappings.json"
    OPPORTUNITIES_FILE: str = "data/opportunities.jsonl"
    FORKS_FILE: str = "data/forks.jsonl"
    SPORTS_FILE: str = "data/sports_arbs.jsonl"
    LOG_FILE: str = "logs/arbitrage.log"
    DOTENV_PATH: str = ".env"


CFG = Config()


# ── DotEnv loader ─────────────────────────────────────────────────

class DotEnvLoader:
    SECRET_PATTERNS = {"KEY", "SECRET", "TOKEN", "PASSWORD"}

    def __init__(self, path: str = ".env"):
        self.path = Path(path)

    def load(self, override: bool = False):
        for k, v in self._parse().items():
            if override or k not in os.environ:
                os.environ[k] = v

    def read_raw(self) -> dict:
        return self._parse()

    def get_mtime(self) -> float:
        return self.path.stat().st_mtime if self.path.exists() else 0.0

    def _parse(self) -> dict:
        result = {}
        if not self.path.exists():
            return result
        try:
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if val and val[0] not in ('"', "'"):
                    val = re.sub(r"\s+#.*$", "", val).strip()
                if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                    val = val[1:-1]
                if key:
                    result[key] = val
        except Exception as e:
            logger.error(f"DotEnv parse error: {e}")
        return result


# ── Config Watchdog ───────────────────────────────────────────────

class ConfigWatchdog:
    """Перечитывает .env каждые 30 секунд и применяет RELOADABLE параметры."""

    def __init__(self, config: Config, loader: DotEnvLoader):
        self.config = config
        self.loader = loader
        self._last_mtime = loader.get_mtime()
        self._lock = Lock()
        self._reload_count = 0

    async def watch(self):
        import asyncio
        logger.info("ConfigWatchdog started")
        while True:
            try:
                await asyncio.sleep(30)
                mtime = self.loader.get_mtime()
                if mtime > self._last_mtime:
                    self._last_mtime = mtime
                    self._reload()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    def _reload(self):
        new_vars = self.loader.read_raw()
        changed = []
        with self._lock:
            for key, val_str in new_vars.items():
                if key not in RELOADABLE:
                    continue
                attr = getattr(self.config, key, None)
                if attr is None:
                    continue
                try:
                    t = type(attr)
                    if t == float:
                        new_val = float(val_str)
                    elif t == int:
                        new_val = int(val_str)
                    elif t == bool:
                        new_val = val_str.lower() in ("true", "1", "yes")
                    else:
                        new_val = val_str
                    if attr != new_val:
                        setattr(self.config, key, new_val)
                        changed.append(f"{key}: {attr} → {new_val}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Bad reload value {key}={val_str}: {e}")
        if changed:
            self._reload_count += 1
            logger.info(
                f"Config reloaded #{self._reload_count}:\n"
                + "\n".join(f"  {c}" for c in changed)
            )
