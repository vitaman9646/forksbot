# config/settings.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── Polymarket CLOB ──────────────────────────────────────────
    CLOB_HOST: str = "https://clob.polymarket.com"
    CHAIN_ID: int = 137  # Polygon
    PRIVATE_KEY: str = os.getenv("POLY_PRIVATE_KEY", "")
    FUNDER_ADDRESS: str = os.getenv("POLY_FUNDER_ADDRESS", "")

    # ── Telegram ─────────────────────────────────────────────────
    TG_TOKEN: str = os.getenv("TG_TOKEN", "")
    TG_CHAT_ID: str = os.getenv("TG_CHAT_ID", "")

    # ── Scanning ─────────────────────────────────────────────────
    SCAN_INTERVAL: int = 15           # секунды между сканами
    MIN_NET_EDGE_PCT: float = 0.5     # мин. edge ПОСЛЕ комиссий
    MAX_EDGE_PCT: float = 8.0         # макс. edge (fake arb guard)
    MIN_VOLUME: float = 500           # мин. объём рынка $

    # ── Execution ────────────────────────────────────────────────
    MAX_POSITION_USD: float = 5.0     # НАЧИНАЕМ С $5 !!!
    ORDER_TTL: int = 30               # секунд жизни ордера
    MAX_FILL_WAIT: float = 30.0       # макс. ожидание fill
    MAX_SLIPPAGE_PCT: float = 0.3     # макс. допустимый slippage

    # ── Risk ─────────────────────────────────────────────────────
    MAX_CONCURRENT: int = 1           # макс. одновременных позиций
    MAX_DAILY_LOSS: float = 10.0      # макс. дневной убыток $
    MAX_DRAWDOWN_PCT: float = 15.0    # макс. просадка %
    COOLDOWN_AFTER_LOSS: int = 300    # секунд паузы после убытка

    # ── Paper Trading Mode ───────────────────────────────────────
    PAPER_TRADING: bool = True        # ВКЛЮЧЕНО по умолчанию!

    # ── Files ────────────────────────────────────────────────────
    LOG_FILE: str = "arb_bot.log"
    TRADES_FILE: str = "trades.jsonl"


CFG = Settings()
