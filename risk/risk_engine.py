# risk/risk_engine.py
"""
RiskEngine v2.0
───────────────
Адаптирован под реальное исполнение.

Изменения vs v1.5:
  • Cooldown по времени (секунды), а не по сканам
  • Отслеживание unwind rate
  • force_stop / resume для Telegram
  • Реальный баланс вместо виртуального
  • Трекинг по стратегиям
  • Все проверки через can_trade() — единая точка входа
"""

import time
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("arb_scanner.risk")


class RiskEngineV2:
    """
    Единая точка принятия решения: торговать или нет.

    Использование:
        risk = RiskEngineV2(config)

        # перед сделкой:
        if risk.can_trade():
            ... execute ...
            risk.record_trade(pnl, was_unwound=False)
        else:
            reason = risk.block_reason
    """

    def __init__(self, config):
        self.cfg = config

        # ── состояние ─────────────────────────────────────────
        self._daily_pnl: float = 0.0
        self._day_key: str = ""

        # баланс и drawdown
        self._current_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._initial_balance: float = 0.0

        # trade rate limiting
        self._trade_timestamps: deque = deque(maxlen=500)

        # loss streak и cooldown
        self._consecutive_losses: int = 0
        self._cooldown_until: float = 0  # unix timestamp

        # circuit breaker (полная остановка)
        self._stopped: bool = False
        self._stop_reason: str = ""

        # unwind tracking
        self._total_trades: int = 0
        self._total_unwinds: int = 0
        self._trades_today: int = 0
        self._unwinds_today: int = 0

        # per-strategy stats
        self._strategy_stats: dict = {}

        # причина последнего отказа
        self.block_reason: str = ""

        # история для анализа
        self._pnl_history: deque = deque(maxlen=100)

    # ══════════════════════════════════════════════════════════
    #  BALANCE SYNC
    # ══════════════════════════════════════════════════════════

    def set_balance(self, balance: float):
        """
        Синхронизация с реальным балансом.
        Вызывать при старте и периодически.
        """
        self._current_balance = balance

        if self._initial_balance == 0:
            self._initial_balance = balance

        if balance > self._peak_balance:
            self._peak_balance = balance

    # ══════════════════════════════════════════════════════════
    #  MAIN CHECK
    # ══════════════════════════════════════════════════════════

    def can_trade(
        self,
        edge_pct: float = 0.0,
        position_size: float = 0.0,
        min_depth_usd: float = 0.0,
    ) -> bool:
        """
        Единственный метод проверки: можно ли торговать.

        Возвращает True/False.
        Причина отказа в self.block_reason.
        """
        self._refresh_day()
        self.block_reason = ""

        # ── 0. Manual stop ────────────────────────────────────
        if self._stopped:
            self.block_reason = f"manually_stopped: {self._stop_reason}"
            return False

        # ── 1. Balance check ─────────────────────────────────
        if self._current_balance < 2.0:
            self.block_reason = (
                f"balance_too_low: ${self._current_balance:.2f}"
            )
            return False

        # ── 2. Daily loss limit ──────────────────────────────
        if self._daily_pnl <= -self.cfg.MAX_DAILY_LOSS_USD:
            self.block_reason = (
                f"daily_loss_limit: ${self._daily_pnl:+.2f} "
                f"(max: -${self.cfg.MAX_DAILY_LOSS_USD})"
            )
            return False

        # ── 3. Drawdown ──────────────────────────────────────
        dd_pct = self._drawdown_pct()
        if dd_pct > self.cfg.MAX_DRAWDOWN_PCT:
            self.block_reason = (
                f"max_drawdown: {dd_pct:.1f}% "
                f"(max: {self.cfg.MAX_DRAWDOWN_PCT}%)"
            )
            return False

        # ── 4. Cooldown ──────────────────────────────────────
        now = time.time()
        if now < self._cooldown_until:
            remaining = int(self._cooldown_until - now)
            self.block_reason = (
                f"cooldown: {remaining}s remaining "
                f"(after {self._consecutive_losses} losses)"
            )
            return False

        # ── 5. Edge guard (fake arb) ─────────────────────────
        if edge_pct > 0 and edge_pct > self.cfg.MAX_EDGE_PCT:
            self.block_reason = (
                f"edge_too_high: {edge_pct:.1f}% "
                f"(max: {self.cfg.MAX_EDGE_PCT}%)"
            )
            return False

        # ── 6. Position size vs balance ──────────────────────
        if position_size > 0 and self._current_balance > 0:
            pct = (position_size / self._current_balance) * 100
            if pct > self.cfg.MAX_POSITION_PCT:
                self.block_reason = (
                    f"position_too_large: {pct:.1f}% of balance "
                    f"(max: {self.cfg.MAX_POSITION_PCT}%)"
                )
                return False

        # ── 7. Position size vs depth ────────────────────────
        if position_size > 0 and min_depth_usd > 0:
            if position_size > min_depth_usd * 0.5:
                self.block_reason = (
                    f"position_vs_depth: ${position_size:.2f} > "
                    f"50% of depth ${min_depth_usd:.2f}"
                )
                return False

        # ── 8. Trade rate limit ──────────────────────────────
        hour_count = self._trades_last_hour()
        if hour_count >= self.cfg.MAX_TRADES_PER_HOUR:
            self.block_reason = (
                f"rate_limit: {hour_count} trades/hour "
                f"(max: {self.cfg.MAX_TRADES_PER_HOUR})"
            )
            return False

        # ── 9. Unwind rate ───────────────────────────────────
        unwind_rate = self._unwind_rate()
        if (
            self._total_trades >= 5
            and unwind_rate > self.cfg.MAX_UNWIND_RATE
        ):
            self.block_reason = (
                f"high_unwind_rate: {unwind_rate:.0%} "
                f"(max: {self.cfg.MAX_UNWIND_RATE:.0%})"
            )
            return False

        return True

    # ══════════════════════════════════════════════════════════
    #  TRADE RECORDING
    # ══════════════════════════════════════════════════════════

    def record_trade(
        self,
        pnl: float,
        was_unwound: bool = False,
        strategy: str = "fork",
    ):
        """
        Вызывать ПОСЛЕ каждой завершённой сделки.

        Args:
            pnl: реальный P&L в USD
            was_unwound: была ли сделка unwind-нута
            strategy: тип стратегии
        """
        self._refresh_day()
        now = time.time()

        # daily PnL
        self._daily_pnl += pnl

        # trade count
        self._trade_timestamps.append(now)
        self._total_trades += 1
        self._trades_today += 1

        # unwind tracking
        if was_unwound:
            self._total_unwinds += 1
            self._unwinds_today += 1

        # loss streak
        if pnl < 0:
            self._consecutive_losses += 1
            if (
                self._consecutive_losses
                >= self.cfg.MAX_CONSECUTIVE_LOSSES
            ):
                cooldown_sec = self.cfg.COOLDOWN_AFTER_LOSS_SEC
                self._cooldown_until = now + cooldown_sec
                logger.warning(
                    f"RiskEngine: {self._consecutive_losses} "
                    f"consecutive losses → "
                    f"cooldown {cooldown_sec}s"
                )
        else:
            self._consecutive_losses = 0

        # PnL history
        self._pnl_history.append({
            "pnl": pnl,
            "strategy": strategy,
            "unwound": was_unwound,
            "timestamp": now,
        })

        # per-strategy
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {
                "trades": 0,
                "pnl": 0.0,
                "wins": 0,
                "losses": 0,
                "unwinds": 0,
            }
        s = self._strategy_stats[strategy]
        s["trades"] += 1
        s["pnl"] += pnl
        if pnl > 0:
            s["wins"] += 1
        elif pnl < 0:
            s["losses"] += 1
        if was_unwound:
            s["unwinds"] += 1

        logger.info(
            f"RiskEngine: trade recorded | "
            f"PnL=${pnl:+.4f} | "
            f"Daily=${self._daily_pnl:+.4f} | "
            f"Streak={self._consecutive_losses} | "
            f"Unwound={was_unwound}"
        )

    # ══════════════════════════════════════════════════════════
    #  MANUAL CONTROLS
    # ══════════════════════════════════════════════════════════

    def force_stop(self, reason: str = "manual"):
        """Аварийная остановка торговли."""
        self._stopped = True
        self._stop_reason = reason
        logger.error(f"RiskEngine: FORCE STOP — {reason}")

    def resume(self):
        """Возобновление торговли."""
        self._stopped = False
        self._stop_reason = ""
        self._consecutive_losses = 0
        self._cooldown_until = 0
        logger.info("RiskEngine: RESUMED")

    def reset(self):
        """Полный сброс."""
        self._stopped = False
        self._stop_reason = ""
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._cooldown_until = 0
        self._total_unwinds = 0
        self._total_trades = 0
        self._trades_today = 0
        self._unwinds_today = 0
        logger.info("RiskEngine: FULL RESET")

    # ══════════════════════════════════════════════════════════
    #  SNAPSHOT (для Telegram / логов)
    # ══════════════════════════════════════════════════════════

    def get_snapshot(self) -> dict:
        """Возвращает текущее состояние для отображения."""
        now = time.time()
        cooldown_remaining = max(0, self._cooldown_until - now)

        return {
            "is_active": not self._stopped and self.can_trade(),
            "stopped": self._stopped,
            "stop_reason": self._stop_reason,
            "daily_pnl": round(self._daily_pnl, 4),
            "drawdown_pct": round(self._drawdown_pct(), 2),
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": (
                datetime.fromtimestamp(
                    self._cooldown_until, tz=timezone.utc
                ).isoformat()
                if cooldown_remaining > 0 else None
            ),
            "cooldown_remaining_sec": int(cooldown_remaining),
            "trades_today": self._trades_today,
            "trades_last_hour": self._trades_last_hour(),
            "total_trades": self._total_trades,
            "unwind_rate": round(self._unwind_rate(), 3),
            "unwinds_today": self._unwinds_today,
            "peak_balance": round(self._peak_balance, 2),
            "current_balance": round(self._current_balance, 2),
            "strategy_stats": dict(self._strategy_stats),
        }

    def format_status(self) -> str:
        """Форматированный статус для логов."""
        snap = self.get_snapshot()
        status = (
            "🔴 STOPPED" if snap["stopped"]
            else "🟡 COOLDOWN" if snap["cooldown_remaining_sec"] > 0
            else "🟢 ACTIVE"
        )
        lines = [
            f"Risk Engine: {status}",
            f"  Daily PnL: ${snap['daily_pnl']:+.4f}",
            f"  Drawdown: {snap['drawdown_pct']:.1f}%",
            f"  Trades today: {snap['trades_today']}",
            f"  Trades/hr: {snap['trades_last_hour']}",
            f"  Loss streak: {snap['consecutive_losses']}",
            f"  Unwind rate: {snap['unwind_rate']:.0%}",
        ]
        if snap["stopped"]:
            lines.append(f"  Reason: {snap['stop_reason']}")
        if snap["cooldown_remaining_sec"] > 0:
            lines.append(
                f"  Cooldown: {snap['cooldown_remaining_sec']}s"
            )
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _drawdown_pct(self) -> float:
        """Текущая просадка от пика."""
        if self._peak_balance <= 0:
            return 0.0
        dd = (
            (self._peak_balance - self._current_balance)
            / self._peak_balance * 100
        )
        return max(0.0, dd)

    def _trades_last_hour(self) -> int:
        """Количество сделок за последний час."""
        now = time.time()
        return sum(
            1 for t in self._trade_timestamps
            if now - t < 3600
        )

    def _unwind_rate(self) -> float:
        """Доля unwind-нутых сделок."""
        if self._total_trades == 0:
            return 0.0
        return self._total_unwinds / self._total_trades

    def _refresh_day(self):
        """Сброс дневных счётчиков при смене дня."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._day_key:
            if self._day_key:
                logger.info(
                    f"RiskEngine: new day | "
                    f"yesterday PnL: ${self._daily_pnl:+.4f} | "
                    f"trades: {self._trades_today} | "
                    f"unwinds: {self._unwinds_today}"
                )
            self._day_key = today
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._unwinds_today = 0
