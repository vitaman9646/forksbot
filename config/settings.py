# config/settings.py
"""
Конфигурация бота v2.0
Разделена на секции: API, Scanning, Execution, Risk, Fees, Files
"""

import os
import logging

logger = logging.getLogger("arb_scanner.config")

# Параметры, которые можно менять без рестарта
RELOADABLE = {
    # scanning
    "MIN_NET_EDGE_PCT", "MAX_EDGE_PCT", "MIN_VOLUME",
    "SCAN_INTERVAL",
    # execution
    "MAX_POSITION_USD", "ORDER_TTL_SEC", "MAX_SLIPPAGE_PCT",
    "MAX_FILL_WAIT_SEC",
    # risk
    "MAX_CONCURRENT", "MAX_DAILY_LOSS_USD", "COOLDOWN_AFTER_LOSS_SEC",
    "MAX_POSITION_PCT",
    # режим
    "PAPER_TRADING",
}


class Settings:
    """
    Все параметры бота в одном месте.
    Значения по умолчанию — БЕЗОПАСНЫЕ (paper mode, малые размеры).
    """

    # ══════════════════════════════════════════════════════════
    #  CREDENTIALS
    # ══════════════════════════════════════════════════════════

    # Telegram
    TELEGRAM_TOKEN: str = os.getenv("ARB_TG_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("ARB_TG_CHAT", "")

    # Polymarket CLOB (для реального исполнения)
    POLY_PRIVATE_KEY: str = os.getenv("POLY_PRIVATE_KEY", "")
    POLY_FUNDER_ADDRESS: str = os.getenv("POLY_FUNDER_ADDRESS", "")
    POLY_API_KEY: str = os.getenv("POLY_API_KEY", "")
    POLY_API_SECRET: str = os.getenv("POLY_API_SECRET", "")
    POLY_PASSPHRASE: str = os.getenv("POLY_PASSPHRASE", "")

    # ══════════════════════════════════════════════════════════
    #  API ENDPOINTS
    # ══════════════════════════════════════════════════════════

    GAMMA_EVENTS_API: str = "https://gamma-api.polymarket.com/events"
    GAMMA_MARKETS_API: str = "https://gamma-api.polymarket.com/markets"
    CLOB_HOST: str = "https://clob.polymarket.com"
    CLOB_BOOK_URL: str = "https://clob.polymarket.com/book"
    CHAIN_ID: int = 137  # Polygon mainnet

    # ══════════════════════════════════════════════════════════
    #  MODE
    # ══════════════════════════════════════════════════════════

    # PAPER_TRADING = True → бот сканирует и логирует,
    # но НЕ размещает ордера
    PAPER_TRADING: bool = True

    # ══════════════════════════════════════════════════════════
    #  SCANNING — поиск вилок
    # ══════════════════════════════════════════════════════════

    SCAN_INTERVAL: int = 30              # секунды между сканами

    # Edge фильтры
    MIN_NET_EDGE_PCT: float = 0.5        # мин. edge ПОСЛЕ комиссий и slippage
    MAX_EDGE_PCT: float = 12.0           # макс. edge (выше = фейк/ошибка данных)
    MIN_GROSS_EDGE_PCT: float = 2.0      # мин. edge ДО комиссий (быстрый фильтр)

    # Объём и ликвидность
    MIN_VOLUME: float = 500              # мин. volume24hr для каждого исхода $
    MIN_ORDERBOOK_DEPTH: float = 20.0    # мин. глубина стакана $
    MIN_OUTCOMES: int = 3                # мин. количество исходов

    # Фильтры mid-price (быстрый этап)
    MAX_SUM_DEVIATION: float = 0.08      # сумма не дальше 8% от 1.0
    MIN_SUM_DEVIATION: float = 0.005     # меньше 0.5% — не вилка

    # API
    GAMMA_FETCH_LIMIT: int = 200         # сколько событий запрашивать
    FETCH_TIMEOUT: int = 20              # таймаут HTTP запроса
    BOOK_FETCH_DELAY: float = 0.3        # пауза между запросами стаканов

    # ══════════════════════════════════════════════════════════
    #  EXECUTION — исполнение ордеров
    # ══════════════════════════════════════════════════════════

    MAX_POSITION_USD: float = 5.0        # НАЧИНАЕМ С $5!
    ORDER_TTL_SEC: int = 30              # время жизни лимитного ордера
    MAX_FILL_WAIT_SEC: float = 30.0      # макс. ожидание исполнения
    MAX_SLIPPAGE_PCT: float = 0.3        # макс. slippage при лимитке
    FILL_CHECK_INTERVAL: float = 2.0     # интервал проверки fill

    # Unwind параметры
    UNWIND_PRICE_DISCOUNT: float = 0.01  # на сколько ниже bid при unwind
    UNWIND_TIMEOUT_SEC: float = 60.0     # таймаут на unwind

    # ══════════════════════════════════════════════════════════
    #  RISK — управление рисками
    # ══════════════════════════════════════════════════════════

    MAX_CONCURRENT: int = 1              # макс. одновременных позиций
    MAX_POSITION_PCT: float = 5.0        # % от баланса на одну позицию
    MAX_DAILY_LOSS_USD: float = 10.0     # дневной стоп-лосс в $
    MAX_DRAWDOWN_PCT: float = 15.0       # макс. просадка от пика %
    COOLDOWN_AFTER_LOSS_SEC: int = 300   # пауза после убытка (секунды)
    MAX_CONSECUTIVE_LOSSES: int = 3      # макс. убытков подряд до паузы
    MAX_TRADES_PER_HOUR: int = 10        # защита от цикла
    MAX_UNWIND_RATE: float = 0.3         # >30% unwind → пауза

    # ══════════════════════════════════════════════════════════
    #  FEES — комиссии Polymarket
    # ══════════════════════════════════════════════════════════

    POLY_TAKER_FEE: float = 0.02         # 2% на выигрышные shares
    POLY_MAKER_FEE: float = 0.00         # 0% для maker
    # Упрощённый расчёт: fee от всей позиции
    EFFECTIVE_FEE_PCT: float = 1.0       # ~1% от total position

    # Price guards
    MIN_PRICE: float = 0.02              # не торговать shares дешевле
    MAX_PRICE: float = 0.98              # не торговать shares дороже

    # ══════════════════════════════════════════════════════════
    #  FILES
    # ══════════════════════════════════════════════════════════

    LOG_FILE: str = "logs/arb_bot.log"
    TRADES_FILE: str = "data/trades.jsonl"
    FORKS_LOG_FILE: str = "data/forks_found.jsonl"
    SCANNER_STATS_FILE: str = "data/scanner_stats.jsonl"
    DOTENV_PATH: str = ".env"

    # ══════════════════════════════════════════════════════════
    #  VALIDATION
    # ══════════════════════════════════════════════════════════

    def validate(self) -> list:
        """
        Проверяет конфигурацию.
        Возвращает список ошибок (пустой = всё ок).
        """
        errors = []

        # Telegram
        if not self.TELEGRAM_TOKEN:
            errors.append("ARB_TG_TOKEN not set")
        if not self.TELEGRAM_CHAT_ID:
            errors.append("ARB_TG_CHAT not set")

        # Execution credentials (только для live mode)
        if not self.PAPER_TRADING:
            if not self.POLY_PRIVATE_KEY:
                errors.append(
                    "POLY_PRIVATE_KEY required for live trading"
                )
            if not self.POLY_FUNDER_ADDRESS:
                errors.append(
                    "POLY_FUNDER_ADDRESS required for live trading"
                )

        # Sanity checks
        if self.MAX_POSITION_USD > 100:
            errors.append(
                f"MAX_POSITION_USD={self.MAX_POSITION_USD} "
                f"too high for initial testing"
            )

        if self.MIN_NET_EDGE_PCT < 0.1:
            errors.append(
                f"MIN_NET_EDGE_PCT={self.MIN_NET_EDGE_PCT} "
                f"dangerously low"
            )

        if self.MAX_EDGE_PCT < self.MIN_NET_EDGE_PCT:
            errors.append(
                f"MAX_EDGE_PCT ({self.MAX_EDGE_PCT}) < "
                f"MIN_NET_EDGE_PCT ({self.MIN_NET_EDGE_PCT})"
            )

        if self.MAX_CONCURRENT > 3:
            errors.append(
                f"MAX_CONCURRENT={self.MAX_CONCURRENT} "
                f"too high for initial testing"
            )

        if self.SCAN_INTERVAL < 10:
            errors.append(
                f"SCAN_INTERVAL={self.SCAN_INTERVAL} "
                f"too fast, will hit rate limits"
            )

        return errors

    def print_config(self):
        """Выводит конфигурацию (маскируя секреты)."""
        lines = [
            "═" * 55,
            "  Bot Configuration",
            "═" * 55,
            f"  Mode: {'📝 PAPER' if self.PAPER_TRADING else '💰 LIVE'}",
            "",
            "  Scanning:",
            f"    Interval: {self.SCAN_INTERVAL}s",
            f"    Min net edge: {self.MIN_NET_EDGE_PCT}%",
            f"    Max edge: {self.MAX_EDGE_PCT}%",
            f"    Min volume: ${self.MIN_VOLUME}",
            "",
            "  Execution:",
            f"    Max position: ${self.MAX_POSITION_USD}",
            f"    Order TTL: {self.ORDER_TTL_SEC}s",
            f"    Max slippage: {self.MAX_SLIPPAGE_PCT}%",
            "",
            "  Risk:",
            f"    Max concurrent: {self.MAX_CONCURRENT}",
            f"    Max daily loss: ${self.MAX_DAILY_LOSS_USD}",
            f"    Max drawdown: {self.MAX_DRAWDOWN_PCT}%",
            f"    Cooldown: {self.COOLDOWN_AFTER_LOSS_SEC}s",
            "",
            "  Credentials:",
            f"    Telegram: {'✅' if self.TELEGRAM_TOKEN else '❌'}",
            f"    CLOB key: {'✅' if self.POLY_PRIVATE_KEY else '❌'}",
            f"    Funder: {'✅' if self.POLY_FUNDER_ADDRESS else '❌'}",
            "═" * 55,
        ]
        for line in lines:
            logger.info(line)


# Глобальный экземпляр
CFG = Settings()
