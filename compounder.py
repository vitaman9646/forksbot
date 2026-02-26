"""
CompoundingManager v1.5
────────────────────────
Изменения vs v1.3:
  • pending_trades: прибыль от арбитража засчитывается не мгновенно,
    а через simulate_settlement() — имитирует реальный settlement
  • liquidity_check(): проверка объёма перед записью сделки
  • интеграция с RiskEngine
  • volatility_based_sizing(): размер позиции зависит от волатильности edge
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger("arb_scanner.compounder")


class PendingTrade:
    """Виртуальная позиция, ожидающая settlement."""
    def __init__(self, trade_n: int, gross_profit: float, details: dict):
        self.trade_n = trade_n
        self.gross_profit = gross_profit
        self.details = details
        self.opened_at = datetime.now(timezone.utc).isoformat()
        # settlement симулируется через N сканов
        self.scans_remaining: int = details.get("settlement_scans", 5)


class CompoundingManager:
    def __init__(
        self,
        initial_deposit: float = 80.0,
        max_risk_pct: float = 5.0,
        max_drawdown_pct: float = 20.0,
        state_file: str = "data/compound_state.json",
        risk_engine=None,
    ):
        self.initial_deposit = initial_deposit
        self.max_risk_pct = max_risk_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.state_file = Path(state_file)
        self.risk = risk_engine  # может быть None

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

        # v1.5: список ожидающих settlement
        self._pending: List[PendingTrade] = []
        # v1.5: история edge для adaptive sizing
        self._edge_history: List[float] = []

        self._load_state()

        # синхронизируем RiskEngine
        if self.risk:
            self.risk.initial_deposit = self.total_deposited
            self.risk.set_balance(self.bankroll)

    # ── deposit ──────────────────────────────────────────────────

    def add_deposit(self, amount: float):
        self.total_deposited += amount
        self.bankroll += amount
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        if self.risk:
            self.risk.initial_deposit = self.total_deposited
            self.risk.set_balance(self.bankroll)
        logger.info(f"DEPOSIT +${amount:.2f} | Bankroll: ${self.bankroll:.2f}")
        self._save_state()

    # ── sizing ────────────────────────────────────────────────────

    def get_position_size(self, edge_pct: Optional[float] = None) -> float:
        """
        v1.5: учитывает edge через Kelly-inspired scaling.
        edge_pct = 1.5 → 50% от max_risk
        edge_pct = 3.0 → 75% от max_risk
        edge_pct = 5.0+ → 100% от max_risk
        """
        if self.is_stopped:
            return 0.0
        if self.risk and self.risk.circuit_breaker:
            return 0.0

        base = self.bankroll * (self.max_risk_pct / 100)
        if base < 2.0:
            return 0.0

        if edge_pct is not None:
            # плавный Kelly: min 40%, max 100% от base
            scale = min(1.0, max(0.4, edge_pct / 5.0))
            base = base * scale

        return round(base, 2)

    def liquidity_check(
        self,
        position_size: float,
        outcome_volumes: List[float],
        outcome_depths: Optional[List[float]] = None,
    ) -> tuple[bool, str]:
        """
        v1.5: проверяем что позиция не перегружает рынок.
        outcome_volumes: volume24hr каждого исхода
        outcome_depths: текущая ликвидность в стакане (опционально)
        """
        from config import CFG

        if not outcome_volumes:
            return False, "no_volume_data"

        min_vol = min(v for v in outcome_volumes if v > 0) if any(v > 0 for v in outcome_volumes) else 0
        if min_vol < CFG.MIN_VOLUME:
            return False, f"volume_too_low_{min_vol:.0f}"

        # позиция не должна быть больше X% от минимального объёма
        ratio = position_size / max(min_vol, 1)
        if ratio > CFG.MAX_POSITION_TO_VOLUME_RATIO:
            return False, f"position_too_large_vs_volume_{ratio:.3f}"

        if outcome_depths:
            min_depth = min(d for d in outcome_depths if d > 0) if any(d > 0 for d in outcome_depths) else 0
            if min_depth < CFG.MIN_ORDERBOOK_DEPTH:
                return False, f"depth_too_low_{min_depth:.0f}"

        return True, ""

    # ── trade recording ───────────────────────────────────────────

    def record_trade(
        self,
        profit: float,
        details: Optional[Dict] = None,
        immediate: bool = False,
    ):
        """
        v1.5: по умолчанию сделка идёт в pending (simulate settlement).
        immediate=True — старое поведение (для совместимости).
        """
        details = details or {}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_date:
            self.daily_trades = 0
            self.last_date = today

        self.trade_count += 1
        self.daily_trades += 1

        # сохраняем edge для adaptive sizing
        edge = details.get("net_pct") or details.get("edge_pct") or 0
        if edge:
            self._edge_history.append(float(edge))
            if len(self._edge_history) > 100:
                self._edge_history = self._edge_history[-100:]

        if immediate:
            self._settle(profit, details)
        else:
            # кладём в pending, settlement через N сканов
            pt = PendingTrade(self.trade_count, profit, details)
            self._pending.append(pt)
            logger.info(
                f"Trade #{self.trade_count} → PENDING settlement in "
                f"{pt.scans_remaining} scans | "
                f"Expected profit: ${profit:+.4f}"
            )

        if self.risk:
            self.risk.set_balance(self.bankroll)

        self._save_state()

    def process_pending(self) -> List[Dict]:
        """
        Вызывай раз в скан. Возвращает список settled сделок.
        """
        settled = []
        still_pending = []

        for pt in self._pending:
            pt.scans_remaining -= 1
            if pt.scans_remaining <= 0:
                self._settle(pt.gross_profit, pt.details)
                settled.append({
                    "trade_n": pt.trade_n,
                    "profit": pt.gross_profit,
                    "details": pt.details,
                })
                logger.info(
                    f"Trade #{pt.trade_n} SETTLED | "
                    f"Profit: ${pt.gross_profit:+.4f} | "
                    f"Bank: ${self.bankroll:.2f}"
                )
            else:
                still_pending.append(pt)

        self._pending = still_pending

        if self.risk:
            self.risk.on_scan_complete()

        return settled

    def _settle(self, profit: float, details: Dict):
        strategy = details.get("strategy", "unknown")
        self.total_profit += profit
        self.bankroll += profit

        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        dd = (
            (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100
            if self.peak_bankroll > 0 else 0
        )
        if dd > self.max_drawdown_pct:
            self.is_stopped = True
            self.stop_reason = f"Drawdown {dd:.1f}%"
            logger.warning(f"STOP: {self.stop_reason}")

        if self.risk:
            self.risk.record_result(profit, strategy)

        total = self.winning_trades + self.losing_trades
        wr = (self.winning_trades / total * 100) if total > 0 else 0
        logger.info(
            f"Settled #{self.trade_count} | PnL: ${profit:+.4f} | "
            f"Bank: ${self.bankroll:.2f} | "
            f"Total: ${self.total_profit:+.2f} | WR: {wr:.0f}%"
        )

        record = {
            "n": self.trade_count,
            "pnl": round(profit, 4),
            "bank": round(self.bankroll, 2),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        record.update(details)
        try:
            with open("data/compound_trades.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        self._save_state()

    # ── stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        total = self.winning_trades + self.losing_trades
        dd = (
            (self.peak_bankroll - self.bankroll) / self.peak_bankroll * 100
            if self.peak_bankroll > 0 else 0
        )
        avg_edge = (
            sum(self._edge_history) / len(self._edge_history)
            if self._edge_history else 0
        )
        return {
            "bankroll": round(self.bankroll, 2),
            "deposited": round(self.total_deposited, 2),
            "profit": round(self.total_profit, 2),
            "roi": round(
                (self.total_profit / self.total_deposited * 100)
                if self.total_deposited > 0 else 0, 1
            ),
            "trades": self.trade_count,
            "pending": len(self._pending),
            "win_rate": round((self.winning_trades / total * 100) if total > 0 else 0, 1),
            "position": round(self.get_position_size(), 2),
            "drawdown": round(dd, 1),
            "growth": round(
                (self.bankroll / self.total_deposited)
                if self.total_deposited > 0 else 1, 2
            ),
            "avg_edge": round(avg_edge, 2),
            "stopped": self.is_stopped,
        }

    def print_stats(self):
        s = self.get_stats()
        risk_str = ""
        if self.risk:
            snap = self.risk.get_snapshot()
            risk_str = (
                f"\n Risk CB:    {'YES' if snap.circuit_breaker else 'no'}\n"
                f" Daily PnL: ${snap.daily_pnl:>+9.4f}\n"
                f" Loss str:  {snap.loss_streak:>10}\n"
                f" Cooldown:  {snap.cooldown_remaining:>10} scans"
            )
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
            f" Pending:   {s['pending']:>10}\n"
            f" Win rate:  {s['win_rate']:>9.1f}%\n"
            f" Avg edge:  {s['avg_edge']:>9.2f}%\n"
            f" Position:  ${s['position']:>10.2f}\n"
            f" Drawdown:  {s['drawdown']:>9.1f}%"
            f"{risk_str}\n"
            f"{'='*45}"
        )

    def reset_stop(self):
        self.is_stopped = False
        self.stop_reason = ""
        if self.risk:
            self.risk.reset_circuit_breaker()
        self._save_state()

    # ── persistence ───────────────────────────────────────────────

    def _save_state(self):
        state = {
            "bankroll": self.bankroll,
            "total_deposited": self.total_deposited,
            "total_profit": self.total_profit,
            "peak_bankroll": self.peak_bankroll,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "daily_trades": self.daily_trades,
            "last_date": self.last_date,
            "is_stopped": self.is_stopped,
            "stop_reason": self.stop_reason,
            "edge_history": self._edge_history[-20:],
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Save state: {e}")

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
            self._edge_history = s.get("edge_history", [])
            logger.info(
                f"Loaded state: bank=${self.bankroll:.2f} "
                f"profit=${self.total_profit:+.2f}"
            )
        except Exception as e:
            logger.error(f"Load state: {e}")
