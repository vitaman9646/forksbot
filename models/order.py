# models/order.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"          # создан, ещё не отправлен
    SUBMITTED = "SUBMITTED"      # отправлен в CLOB
    LIVE = "LIVE"                # в стакане
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


@dataclass
class Order:
    order_id: Optional[str] = None       # ID от Polymarket
    internal_id: str = ""                 # наш внутренний ID
    token_id: str = ""                    # CLOB token ID
    market_slug: str = ""
    side: OrderSide = OrderSide.BUY
    price: float = 0.0                   # лимитная цена
    size: float = 0.0                    # размер в долларах
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cancel_reason: str = ""
    ttl_seconds: int = 60                # время жизни ордера

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
            OrderStatus.EXPIRED,
        )

    @property
    def unfilled_size(self) -> float:
        return max(0, self.size - self.filled_size)

    @property
    def fill_pct(self) -> float:
        if self.size == 0:
            return 0
        return (self.filled_size / self.size) * 100

    @property
    def age_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        return self.age_seconds > self.ttl_seconds

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "internal_id": self.internal_id,
            "token_id": self.token_id,
            "market_slug": self.market_slug,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "status": self.status.value,
            "filled_size": self.filled_size,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
        }
