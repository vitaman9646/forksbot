"""
RiskEngine v1.5
───────────────
Портирован из HFT v7.4, адаптирован под REST-сканер арбитража.

Защищает от:
  • Превышения дневного лимита убытков
  • Слишком высокого edge (признак ложной вилки)
  • Слишком низкой ликвидности относительно размера позиции
  • Слишком частых сделок (дубли, баги)
  • Серии убытков подряд (cooldown)
  • Общего drawdown
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("arb_scanner.risk")


@dataclass
class RiskSnapshot:
    """Текущее состояние риск-движка (для логов и Telegram)."""
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    trades_last_hour: int = 0
    loss_streak: int = 0
    circuit_breaker: bool = False
    circuit_reason: str = ""
    cooldown_remaining: int = 0


class RiskEngine:
    """
    Принимает/отклоняет предложенную сделку.
    Вызывай can_trade() перед каждой виртуальной/реальной сделкой.
    """

    def __init__(self, config):
        self.cfg = config
        self.initial_deposit: float = 80.0  # будет обновлён из CompoundingManager

        # дневной P&L
        self._daily_pnl: float = 0.0
        self._day_key: str = ""  # "YYYY-MM-DD"

        # trades per hour
        self._trade_ts: deque = deque(maxlen=500)

        # loss streak → cooldown
        self._loss_streak: int = 0
        self._cooldown_remaining: int = 0   # в единицах «сканов»

        # circuit breaker
        self.circuit_breaker: bool = False
        self.circuit_reason: str = ""

        # peak для drawdown
        self._peak_balance: float = 80.0

        # per-strategy счётчики (для логов)
        self._strategy_counts: Dict[str, int] = {}

    # ── public API ────────────────────────────────────────────────

    def set_balance(self, balance: float):
        """Синхронизируй с CompoundingManager перед can_trade()."""
        if balance > self._peak_balance:
            self._peak_balance = balance

    def can_trade(
        self,
        balance: float,
        edge_pct: float,
        position_size: float,
        volume24hr: float,
        strategy: str = "unknown",
    ) -> tuple[bool, str]:
        """
        Возвращает (True, "") если сделка разрешена,
        иначе (False, причина).
        """
        self.set_balance(balance)
        self._refresh_day()

        # 0. circuit breaker
        if self.circuit_breaker:
            return False, f"circuit_breaker: {self.circuit_reason}"

        # 1. баланс
        if balance < 2.0:
            return False, "balance_too_low"

        # 2. дневной лимит убытков
        max_loss = self.initial_deposit * self.cfg.MAX_DAILY_LOSS_PCT / 100
        if self._daily_pnl <= -max_loss:
            self._trigger_circuit("daily_loss_limit")
            return False, "daily_loss_limit"

        # 3. общий drawdown
        if self._peak_balance > 0:
            dd = (self._peak_balance - balance) / self._peak_balance * 100
            if dd > self.cfg.MAX_DRAWDOWN_PCT:
                self._trigger_circuit(f"drawdown_{dd:.1f}pct")
                return False, f"drawdown_{dd:.1f}pct"

        # 4. слишком высокий edge → ложная вилка
        if edge_pct > self.cfg.MAX_EDGE_PCT:
            return False, f"edge_too_high_{edge_pct:.1f}pct"

        # 5. ликвидность vs размер позиции
        if volume24hr > 0:
            ratio = position_size / volume24hr
            if ratio > self.cfg.MAX_POSITION_TO_VOLUME_RATIO:
                return False, f"low_liquidity_ratio_{ratio:.3f}"

        # 6. rate limit (сделок в час)
        now = time.time()
        self._trade_ts.append(now)
        hour_count = sum(1 for t in self._trade_ts if now - t < 3600)
        if hour_count > self.cfg.MAX_TRADES_PER_HOUR:
            return False, f"too_many_trades_{hour_count}_per_hour"

        # 7. cooldown после серии убытков
        if self._cooldown_remaining > 0:
            return False, f"cooldown_{self._cooldown_remaining}_scans_left"

        return True, ""

    def record_result(self, pnl: float, strategy: str = "unknown"):
        """Вызывай после каждой сделки."""
        self._refresh_day()
        self._daily_pnl += pnl
        self._strategy_counts[strategy] = self._strategy_counts.get(strategy, 0) + 1

        if pnl < 0:
            self._loss_streak += 1
            if self._loss_streak >= self.cfg.COOLDOWN_AFTER_LOSSES:
                self._cooldown_remaining = self._loss_streak * 2
                logger.warning(
                    f"RiskEngine: loss streak {self._loss_streak}, "
                    f"cooldown {self._cooldown_remaining} scans"
                )
        else:
            self._loss_streak = 0

    def on_scan_complete(self):
        """Вызывай в конце каждого скана — уменьшает cooldown."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                logger.info("RiskEngine: cooldown expired, trading resumed")

    def reset_circuit_breaker(self):
        self.circuit_breaker = False
        self.circuit_reason = ""
        self._daily_pnl = 0.0
        self._loss_streak = 0
        self._cooldown_remaining = 0
        logger.info("RiskEngine: circuit breaker reset")

    def get_snapshot(self) -> RiskSnapshot:
        now = time.time()
        hour_count = sum(1 for t in self._trade_ts if now - t < 3600)
        dd = 0.0
        if self._peak_balance > 0:
            # нужен текущий баланс — приближение через peak и pnl
            dd = max(0.0, -self._daily_pnl / self._peak_balance * 100)
        return RiskSnapshot(
            daily_pnl=round(self._daily_pnl, 4),
            drawdown_pct=round(dd, 2),
            trades_last_hour=hour_count,
            loss_streak=self._loss_streak,
            circuit_breaker=self.circuit_breaker,
            circuit_reason=self.circuit_reason,
            cooldown_remaining=self._cooldown_remaining,
        )

    def format_status(self) -> str:
        s = self.get_snapshot()
        status = "🔴 STOPPED" if s.circuit_breaker else (
            "🟡 COOLDOWN" if s.cooldown_remaining > 0 else "🟢 ACTIVE"
        )
        return (
            f"Risk Engine: {status}\n"
            f"  Daily PnL: ${s.daily_pnl:+.4f}\n"
            f"  Trades/hr: {s.trades_last_hour}\n"
            f"  Loss streak: {s.loss_streak}\n"
            f"  Cooldown: {s.cooldown_remaining} scans\n"
            + (f"  Reason: {s.circuit_reason}" if s.circuit_breaker else "")
        )

    # ── internal ──────────────────────────────────────────────────

    def _refresh_day(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day_key:
            if self._day_key:
                logger.info(f"RiskEngine: new day, daily_pnl reset (was ${self._daily_pnl:+.4f})")
            self._day_key = today
            self._daily_pnl = 0.0

    def _trigger_circuit(self, reason: str):
        if not self.circuit_breaker:
            self.circuit_breaker = True
            self.circuit_reason = reason
            logger.error(f"RiskEngine: CIRCUIT BREAKER — {reason}")
