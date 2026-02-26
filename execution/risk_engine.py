import time
import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger("arb_scanner.risk")


@dataclass
class RiskSnapshot:
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0
    trades_last_hour: int = 0
    loss_streak: int = 0
    circuit_breaker: bool = False
    circuit_reason: str = ""
    cooldown_remaining: int = 0


class RiskEngine:
    def __init__(self, config):
        self.cfg = config
        self.initial_deposit = 80.0
        self._daily_pnl = 0.0
        self._day_key = ""
        self._trade_ts = deque(maxlen=500)
        self._loss_streak = 0
        self._cooldown_remaining = 0
        self.circuit_breaker = False
        self.circuit_reason = ""
        self._peak_balance = 80.0

    def set_balance(self, balance):
        if balance > self._peak_balance:
            self._peak_balance = balance

    def can_trade(self, balance, edge_pct, position_size, volume24hr, strategy="unknown"):
        self.set_balance(balance)
        self._refresh_day()
        if self.circuit_breaker:
            return False, f"circuit_breaker: {self.circuit_reason}"
        if balance < 2.0:
            return False, "balance_too_low"
        max_loss = self.initial_deposit * self.cfg.MAX_DAILY_LOSS_PCT / 100
        if self._daily_pnl <= -max_loss:
            self._trigger_circuit("daily_loss_limit")
            return False, "daily_loss_limit"
        if self._peak_balance > 0:
            dd = (self._peak_balance - balance) / self._peak_balance * 100
            if dd > self.cfg.MAX_DRAWDOWN_PCT:
                self._trigger_circuit(f"drawdown_{dd:.1f}pct")
                return False, f"drawdown_{dd:.1f}pct"
        if edge_pct > self.cfg.MAX_EDGE_PCT:
            return False, f"edge_too_high_{edge_pct:.1f}pct"
        if volume24hr > 0:
            ratio = position_size / volume24hr
            if ratio > self.cfg.MAX_POSITION_TO_VOLUME_RATIO:
                return False, f"low_liquidity_ratio_{ratio:.3f}"
        now = time.time()
        self._trade_ts.append(now)
        hour_count = sum(1 for t in self._trade_ts if now - t < 3600)
        if hour_count > self.cfg.MAX_TRADES_PER_HOUR:
            return False, f"too_many_trades_{hour_count}_per_hour"
        if self._cooldown_remaining > 0:
            return False, f"cooldown_{self._cooldown_remaining}_scans_left"
        return True, ""

    def record_result(self, pnl, strategy="unknown"):
        self._refresh_day()
        self._daily_pnl += pnl
        if pnl < 0:
            self._loss_streak += 1
            if self._loss_streak >= self.cfg.COOLDOWN_AFTER_LOSSES:
                self._cooldown_remaining = self._loss_streak * 2
                logger.warning(f"Loss streak {self._loss_streak}, cooldown {self._cooldown_remaining} scans")
        else:
            self._loss_streak = 0

    def on_scan_complete(self):
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                logger.info("RiskEngine: cooldown expired")

    def reset_circuit_breaker(self):
        self.circuit_breaker = False
        self.circuit_reason = ""
        self._daily_pnl = 0.0
        self._loss_streak = 0
        self._cooldown_remaining = 0
        logger.info("RiskEngine: reset")

    def get_snapshot(self):
        now = time.time()
        hour_count = sum(1 for t in self._trade_ts if now - t < 3600)
        dd = max(0.0, -self._daily_pnl / self._peak_balance * 100) if self._peak_balance > 0 else 0.0
        return RiskSnapshot(
            daily_pnl=round(self._daily_pnl, 4),
            drawdown_pct=round(dd, 2),
            trades_last_hour=hour_count,
            loss_streak=self._loss_streak,
            circuit_breaker=self.circuit_breaker,
            circuit_reason=self.circuit_reason,
            cooldown_remaining=self._cooldown_remaining,
        )

    def _refresh_day(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day_key:
            self._day_key = today
            self._daily_pnl = 0.0

    def _trigger_circuit(self, reason):
        if not self.circuit_breaker:
            self.circuit_breaker = True
            self.circuit_reason = reason
            logger.error(f"CIRCUIT BREAKER: {reason}")
