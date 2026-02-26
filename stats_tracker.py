"""
StatsTracker v1.5
──────────────────
Добавлено: почасовая статистика, edge distribution, rejected count.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("arb_scanner.stats")


class StatsTracker:
    def __init__(self, file: str = "data/daily_stats.jsonl"):
        self.file = Path(file)
        self.today = ""
        self.daily = self._empty_day()

    def _empty_day(self):
        return {
            "scans": 0,
            "forks_found": 0,
            "sports_found": 0,
            "rejected_by_risk": 0,          # v1.5
            "fork_details": [],
            "sports_details": [],
            "edge_distribution": [],         # v1.5: все edge%
            "hourly_counts": defaultdict(int),  # v1.5: час → кол-во находок
            "neg_risk_events": 0,
            "total_events": 0,
            "best_fork_pct": 0.0,
            "best_sport_pct": 0.0,
        }

    def _check_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.today:
            if self.today:
                self._save_day()
            self.today = today
            self.daily = self._empty_day()

    def record_scan(
        self,
        forks,
        sports,
        total_events: int = 0,
        neg_risk_events: int = 0,
        rejected_count: int = 0,
    ):
        self._check_day()
        hour = datetime.now(timezone.utc).hour
        self.daily["scans"] += 1
        self.daily["total_events"] = total_events
        self.daily["neg_risk_events"] = neg_risk_events
        self.daily["rejected_by_risk"] += rejected_count  # v1.5

        for f in forks:
            self.daily["forks_found"] += 1
            self.daily["hourly_counts"][str(hour)] += 1   # v1.5
            self.daily["edge_distribution"].append(        # v1.5
                round(f.net_profit_pct, 2)
            )
            detail = {
                "event": f.event_title[:60],
                "type": f.fork_type,
                "sum": round(f.sum_yes, 4),
                "net_pct": round(f.net_profit_pct, 2),
                "outcomes": len(f.outcomes),
                "neg_risk": f.is_neg_risk,
                "min_vol": round(getattr(f, "min_volume", 0), 0),
                "time": datetime.now(timezone.utc).strftime("%H:%M"),
            }
            self.daily["fork_details"].append(detail)
            if f.net_profit_pct > self.daily["best_fork_pct"]:
                self.daily["best_fork_pct"] = f.net_profit_pct

        for s in sports:
            self.daily["sports_found"] += 1
            self.daily["hourly_counts"][str(hour)] += 1
            self.daily["edge_distribution"].append(round(s.net_edge_pct, 2))
            detail = {
                "event": s.match.event_title[:60],
                "type": s.arb_type,
                "sum": round(s.sum_prices, 4),
                "edge_pct": round(s.net_edge_pct, 2),
                "time": datetime.now(timezone.utc).strftime("%H:%M"),
            }
            self.daily["sports_details"].append(detail)
            if s.net_edge_pct > self.daily["best_sport_pct"]:
                self.daily["best_sport_pct"] = s.net_edge_pct

    def get_summary(self) -> str:
        self._check_day()
        d = self.daily
        unique_forks = len(set(f["event"] for f in d["fork_details"]))
        unique_sports = len(set(s["event"] for s in d["sports_details"]))

        # edge distribution
        edges = d["edge_distribution"]
        edge_str = ""
        if edges:
            avg_e = sum(edges) / len(edges)
            edge_str = f"   Avg edge: {avg_e:.2f}% | Max: {max(edges):.2f}%\n"

        # top hour
        hourly = d["hourly_counts"]
        top_hour_str = ""
        if hourly:
            top_h = max(hourly, key=lambda h: hourly[h])
            top_hour_str = f"   Peak hour: {top_h}:00 UTC ({hourly[top_h]} finds)\n"

        return (
            f"📊 Daily Stats ({self.today})\n\n"
            f"🔍 Scans: {d['scans']}\n"
            f"📋 Events: {d['total_events']}\n"
            f"✅ negRisk: {d['neg_risk_events']}\n"
            f"🚫 Rejected by risk: {d['rejected_by_risk']}\n\n"
            f"🔱 Forks: {d['forks_found']} ({unique_forks} unique)\n"
            f"   Best: {d['best_fork_pct']:.2f}%\n\n"
            f"⚽ Sports: {d['sports_found']} ({unique_sports} unique)\n"
            f"   Best: {d['best_sport_pct']:.2f}%\n\n"
            f"{edge_str}{top_hour_str}"
        )

    def _save_day(self):
        d = self.daily
        record = {
            "date": self.today,
            "scans": d["scans"],
            "forks_found": d["forks_found"],
            "sports_found": d["sports_found"],
            "rejected_by_risk": d["rejected_by_risk"],
            "best_fork_pct": d["best_fork_pct"],
            "best_sport_pct": d["best_sport_pct"],
            "edge_distribution": d["edge_distribution"][:200],
            "hourly_counts": dict(d["hourly_counts"]),
            "fork_details": d["fork_details"][:50],
            "sports_details": d["sports_details"][:50],
        }
        try:
            with open(self.file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Stats save: {e}")
