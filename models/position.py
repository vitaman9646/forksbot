# models/position.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from models.order import Order


class PositionStatus(Enum):
    OPENING = "OPENING"              # размещаем ноги
    LEG_A_FILLED = "LEG_A_FILLED"    # первая нога исполнена
    BOTH_FILLED = "BOTH_FILLED"      # обе ноги исполнены (успех)
    PARTIAL = "PARTIAL"              # частичное исполнение
    UNWINDING = "UNWINDING"          # разворачиваем неудачную позу
    CLOSED = "CLOSED"                # позиция закрыта
    FAILED = "FAILED"                # полный провал


@dataclass
class ArbPosition:
    position_id: str = ""
    event_id: str = ""
    event_title: str = ""
    strategy: str = "fork"           # fork / sports

    # ноги арбитража
    leg_a: Optional[Order] = None    # YES ордер
    leg_b: Optional[Order] = None    # NO ордер

    # расчётные параметры
    expected_edge_pct: float = 0.0   # ожидаемый edge ДО комиссий
    expected_net_pct: float = 0.0    # ожидаемый edge ПОСЛЕ комиссий
    target_size: float = 0.0        # целевой размер позиции

    status: PositionStatus = PositionStatus.OPENING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # результат
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    close_reason: str = ""

    # для unwind
    unwind_orders: List[Order] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            PositionStatus.BOTH_FILLED,
            PositionStatus.CLOSED,
            PositionStatus.FAILED,
        )

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def actual_cost(self) -> float:
        """Реальная стоимость позиции."""
        cost = 0.0
        if self.leg_a and self.leg_a.filled_size > 0:
            cost += self.leg_a.filled_size * self.leg_a.avg_fill_price
        if self.leg_b and self.leg_b.filled_size > 0:
            cost += self.leg_b.filled_size * self.leg_b.avg_fill_price
        return cost

    @property
    def actual_edge_pct(self) -> float:
        """Реальный edge после исполнения."""
        cost = self.actual_cost
        if cost == 0:
            return 0
        # В идеальном арбитраже мы получаем $1 за пару YES+NO
        min_filled = min(
            self.leg_a.filled_size if self.leg_a else 0,
            self.leg_b.filled_size if self.leg_b else 0,
        )
        if min_filled == 0:
            return 0
        revenue = min_filled  # $1 за каждую пару
        return ((revenue - cost) / cost) * 100

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "event_id": self.event_id,
            "event_title": self.event_title,
            "strategy": self.strategy,
            "status": self.status.value,
            "expected_edge_pct": self.expected_edge_pct,
            "expected_net_pct": self.expected_net_pct,
            "target_size": self.target_size,
            "actual_cost": self.actual_cost,
            "actual_edge_pct": self.actual_edge_pct,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "close_reason": self.close_reason,
            "leg_a": self.leg_a.to_dict() if self.leg_a else None,
            "leg_b": self.leg_b.to_dict() if self.leg_b else None,
            "created_at": self.created_at.isoformat(),
        }
