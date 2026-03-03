# core/daily_report.py
"""
Ежедневные и почасовые отчёты.
Собирает статистику и отправляет в Telegram.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger("arb_scanner.reports")


@dataclass
class ScanRecord:
    """Запись одного скана."""
    timestamp: str
    candidates: int
    valid_forks: int
    books_fetched: int
    rejections: Dict[str, int]
    best_edge: float = 0.0
    best_event: str = ""


@dataclass
class ForkSighting:
    """Запись одной увиденной вилки (valid или нет)."""
    timestamp: str
    event_id: str
    event_title: str
    fork_type: str
    is_valid: bool
    mid_sum: float
    real_sum: float
    gross_edge: float
    net_edge: float
    slippage: float
    min_depth: float
    reject_reason: str = ""


class ReportCollector:
    """
    Собирает данные каждого скана.
    Генерирует почасовые и ежедневные отчёты.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # текущий день
        self._today: str = ""
        self._hour: int = -1

        # накопители за день
        self._scans_today: int = 0
        self._candidates_today: int = 0
        self._valid_today: int = 0
        self._books_today: int = 0

        # все вилки за день (для анализа)
        self._forks_today: List[ForkSighting] = []

        # rejection reasons за день
        self._rejections_today: Dict[str, int] = {}

        # edge distribution
        self._edges_seen: List[float] = []

        # почасовые данные
        self._hourly_valid: Dict[int, int] = {}
        self._hourly_candidates: Dict[int, int] = {}

        # исторические данные
        self._daily_history: List[dict] = []

        # загружаем историю
        self._load_history()

    def record_scan(self, forks: list):
        """
        Вызывается после каждого скана.

        Args:
            forks: список RealFork (и valid, и rejected)
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        hour = now.hour

        # новый день — сохраняем отчёт и сбрасываем
        if today != self._today:
            if self._today:
                self._save_daily_report()
            self._reset_day(today)

        self._today = today
        self._hour = hour
        self._scans_today += 1

        candidates = len(forks)
        valid = [f for f in forks if f.is_valid]
        rejected = [f for f in forks if not f.is_valid]

        self._candidates_today += candidates
        self._valid_today += len(valid)

        # почасовые
        self._hourly_candidates[hour] = (
            self._hourly_candidates.get(hour, 0) + candidates
        )
        self._hourly_valid[hour] = (
            self._hourly_valid.get(hour, 0) + len(valid)
        )

        # записываем каждую вилку
        for fork in forks:
            sighting = ForkSighting(
                timestamp=now.isoformat(),
                event_id=fork.event_id,
                event_title=fork.event_title,
                fork_type=fork.fork_type,
                is_valid=fork.is_valid,
                mid_sum=fork.mid_sum,
                real_sum=fork.real_sum,
                gross_edge=fork.gross_edge_pct,
                net_edge=fork.net_edge_pct,
                slippage=fork.slippage_pct,
                min_depth=fork.min_depth_usd,
                reject_reason=fork.reject_reason,
            )
            self._forks_today.append(sighting)

            if fork.gross_edge_pct > 0:
                self._edges_seen.append(fork.gross_edge_pct)

        # rejection reasons
        for fork in rejected:
            reason = fork.reject_reason.split(":")[0].strip()
            self._rejections_today[reason] = (
                self._rejections_today.get(reason, 0) + 1
            )

    def get_hourly_report(self) -> str:
        """Генерирует почасовой отчёт."""
        now = datetime.now(timezone.utc)
        hour = now.hour

        candidates_this_hour = self._hourly_candidates.get(hour, 0)
        valid_this_hour = self._hourly_valid.get(hour, 0)

        # edge stats
        edges = [
            f.gross_edge for f in self._forks_today
            if f.gross_edge > 0
        ]
        avg_edge = sum(edges) / len(edges) if edges else 0
        max_edge = max(edges) if edges else 0

        return (
            f"⏰ Hourly Report [{now.strftime('%H:00')} UTC]\n\n"
            f"📊 This hour:\n"
            f"  Candidates: {candidates_this_hour}\n"
            f"  Valid forks: {valid_this_hour}\n\n"
            f"📊 Today total:\n"
            f"  Scans: {self._scans_today}\n"
            f"  Candidates: {self._candidates_today}\n"
            f"  Valid forks: {self._valid_today}\n\n"
            f"📈 Edge distribution:\n"
            f"  Avg gross edge: {avg_edge:.2f}%\n"
            f"  Max gross edge: {max_edge:.2f}%\n"
            f"  Edges seen: {len(edges)}"
        )

    def get_daily_report(self) -> str:
        """Генерирует дневной отчёт."""
        now = datetime.now(timezone.utc)

        # edge distribution
        edges = [
            f.gross_edge for f in self._forks_today
            if f.gross_edge > 0
        ]
        net_edges = [
            f.net_edge for f in self._forks_today
            if f.net_edge > 0
        ]

        # unique events
        unique_events = set(f.event_id for f in self._forks_today)

        # fork types
        under = sum(
            1 for f in self._forks_today if f.fork_type == "under"
        )
        over = sum(
            1 for f in self._forks_today if f.fork_type == "over"
        )

        # top rejection reasons
        sorted_rejections = sorted(
            self._rejections_today.items(),
            key=lambda x: -x[1],
        )[:5]
        rejection_str = "\n".join(
            f"    {reason}: {count}"
            for reason, count in sorted_rejections
        )

        # hourly heatmap
        heatmap_lines = []
        for h in range(24):
            v = self._hourly_valid.get(h, 0)
            c = self._hourly_candidates.get(h, 0)
            bar = "█" * v + "░" * max(0, c - v)
            if c > 0:
                heatmap_lines.append(
                    f"    {h:02d}:00  {bar} ({v}/{c})"
                )

        heatmap_str = "\n".join(heatmap_lines) if heatmap_lines else "    no data"

        # best forks of the day
        valid_forks = [f for f in self._forks_today if f.is_valid]
        best_str = ""
        if valid_forks:
            best = sorted(
                valid_forks, key=lambda x: -x.net_edge
            )[:3]
            best_lines = []
            for f in best:
                best_lines.append(
                    f"    {f.event_title[:40]}\n"
                    f"      edge={f.net_edge:.2f}% "
                    f"slip={f.slippage:.2f}% "
                    f"depth=${f.min_depth:.0f}"
                )
            best_str = "\n".join(best_lines)

        # history trend
        trend_str = ""
        if self._daily_history:
            last_days = self._daily_history[-7:]
            trend_lines = []
            for d in last_days:
                trend_lines.append(
                    f"    {d['date']}: "
                    f"{d['valid_forks']} valid / "
                    f"{d['candidates']} candidates"
                )
            trend_str = "\n".join(trend_lines)

        report = (
            f"📋 DAILY REPORT — {now.strftime('%Y-%m-%d')}\n"
            f"{'═' * 40}\n\n"
            f"📊 Overview:\n"
            f"  Scans: {self._scans_today}\n"
            f"  Candidates checked: {self._candidates_today}\n"
            f"  Valid forks: {self._valid_today}\n"
            f"  Unique events: {len(unique_events)}\n"
            f"  Under/Over: {under}/{over}\n\n"
            f"📈 Edge Distribution:\n"
            f"  Gross edges seen: {len(edges)}\n"
        )

        if edges:
            report += (
                f"  Avg gross: {sum(edges)/len(edges):.2f}%\n"
                f"  Max gross: {max(edges):.2f}%\n"
                f"  Min gross: {min(edges):.2f}%\n"
            )

        if net_edges:
            report += (
                f"  Avg net: {sum(net_edges)/len(net_edges):.2f}%\n"
                f"  Max net: {max(net_edges):.2f}%\n"
            )

        report += (
            f"\n❌ Top Rejections:\n{rejection_str}\n\n"
            f"🕐 Hourly Heatmap (valid/candidates):\n"
            f"{heatmap_str}\n"
        )

        if best_str:
            report += f"\n🏆 Best Forks:\n{best_str}\n"

        if trend_str:
            report += f"\n📅 7-Day Trend:\n{trend_str}\n"

        # вердикт
        if self._valid_today == 0:
            verdict = (
                "⚠️ VERDICT: No valid forks today. "
                "Market is efficient at current scan speed."
            )
        elif self._valid_today < 3:
            verdict = (
                "🟡 VERDICT: Few forks. Marginal opportunity. "
                "Need more data before real trading."
            )
        else:
            verdict = (
                "🟢 VERDICT: Forks detected. "
                "Consider micro-testing with $5-10."
            )

        report += f"\n{verdict}"

        return report

    def _reset_day(self, new_day: str):
        """Сбрасывает дневные счётчики."""
        self._today = new_day
        self._scans_today = 0
        self._candidates_today = 0
        self._valid_today = 0
        self._books_today = 0
        self._forks_today = []
        self._rejections_today = {}
        self._edges_seen = []
        self._hourly_valid = {}
        self._hourly_candidates = {}

    def _save_daily_report(self):
        """Сохраняет дневную статистику в файл."""
        edges = [
            f.gross_edge for f in self._forks_today
            if f.gross_edge > 0
        ]

        day_data = {
            "date": self._today,
            "scans": self._scans_today,
            "candidates": self._candidates_today,
            "valid_forks": self._valid_today,
            "unique_events": len(
                set(f.event_id for f in self._forks_today)
            ),
            "avg_gross_edge": (
                round(sum(edges) / len(edges), 3) if edges else 0
            ),
            "max_gross_edge": round(max(edges), 3) if edges else 0,
            "rejections": dict(self._rejections_today),
            "hourly_valid": dict(self._hourly_valid),
        }

        self._daily_history.append(day_data)

        # сохраняем в файл
        history_file = self.data_dir / "daily_history.jsonl"
        try:
            with open(history_file, "a") as f:
                f.write(json.dumps(day_data) + "\n")
            logger.info(f"Daily report saved: {self._today}")
        except Exception as e:
            logger.error(f"Failed to save daily report: {e}")

        # сохраняем все вилки дня
        forks_file = self.data_dir / f"forks_{self._today}.jsonl"
        try:
            with open(forks_file, "w") as f:
                for fork in self._forks_today:
                    f.write(json.dumps({
                        "timestamp": fork.timestamp,
                        "event_id": fork.event_id,
                        "event_title": fork.event_title,
                        "fork_type": fork.fork_type,
                        "is_valid": fork.is_valid,
                        "mid_sum": fork.mid_sum,
                        "real_sum": fork.real_sum,
                        "gross_edge": fork.gross_edge,
                        "net_edge": fork.net_edge,
                        "slippage": fork.slippage,
                        "min_depth": fork.min_depth,
                        "reject_reason": fork.reject_reason,
                    }) + "\n")
        except Exception as e:
            logger.error(f"Failed to save forks: {e}")

    def _load_history(self):
        """Загружает историю при старте."""
        history_file = self.data_dir / "daily_history.jsonl"
        if not history_file.exists():
            return
        try:
            with open(history_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._daily_history.append(json.loads(line))
            logger.info(
                f"Loaded {len(self._daily_history)} days of history"
            )
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
