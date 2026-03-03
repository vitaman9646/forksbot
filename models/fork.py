# models/fork.py

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OrderBookLevel:
    """Один уровень стакана."""
    price: float
    size: float        # в shares


@dataclass
class TokenBook:
    """Стакан одного токена (YES или NO)."""
    token_id: str
    outcome: str       # "Yes" / "No"
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_ask(self) -> Optional[float]:
        """Лучшая цена продажи (мы покупаем по ask)."""
        if not self.asks:
            return None
        return min(a.price for a in self.asks)

    @property
    def best_bid(self) -> Optional[float]:
        if not self.bids:
            return None
        return max(b.price for b in self.bids)

    def cost_to_buy(self, amount_usd: float) -> Optional[dict]:
        """
        Считает реальную стоимость покупки на $amount_usd
        с учётом глубины стакана.

        Returns:
            {
                "total_cost": float,
                "shares_acquired": float,
                "avg_price": float,
                "levels_consumed": int,
                "fully_filled": bool,
            }
        """
        if not self.asks:
            return None

        sorted_asks = sorted(self.asks, key=lambda x: x.price)

        total_cost = 0.0
        shares_acquired = 0.0
        levels_consumed = 0
        remaining = amount_usd

        for level in sorted_asks:
            level_value = level.size * level.price
            if level_value <= remaining:
                total_cost += level_value
                shares_acquired += level.size
                remaining -= level_value
                levels_consumed += 1
            else:
                # частичное заполнение этого уровня
                shares_at_level = remaining / level.price
                total_cost += remaining
                shares_acquired += shares_at_level
                remaining = 0
                levels_consumed += 1
                break

        if shares_acquired == 0:
            return None

        return {
            "total_cost": total_cost,
            "shares_acquired": shares_acquired,
            "avg_price": total_cost / shares_acquired,
            "levels_consumed": levels_consumed,
            "fully_filled": remaining <= 0.001,
            "unfilled_usd": remaining,
        }


@dataclass
class RealFork:
    """Вилка с реальными данными стакана."""
    event_id: str
    event_title: str
    condition_id: str

    yes_book: TokenBook
    no_book: TokenBook

    # расчётные поля (заполняются после анализа)
    best_ask_yes: float = 0.0
    best_ask_no: float = 0.0
    raw_sum: float = 0.0             # сумма лучших ask
    gross_edge_pct: float = 0.0      # до комиссий
    net_edge_pct: float = 0.0        # после комиссий
    fee_pct: float = 2.0             # комиссия Polymarket

    # данные при конкретном размере входа
    entry_size_usd: float = 0.0
    real_avg_yes: float = 0.0
    real_avg_no: float = 0.0
    real_sum: float = 0.0
    real_net_edge_pct: float = 0.0
    slippage_pct: float = 0.0

    is_valid: bool = False
    reject_reason: str = ""

    def analyze(self, entry_size_usd: float, min_edge_pct: float = 0.3):
        """
        Полный анализ вилки для конкретного размера входа.
        Учитывает глубину стакана, slippage, комиссии.
        """
        self.entry_size_usd = entry_size_usd
        half = entry_size_usd / 2

        # лучшие ask
        self.best_ask_yes = self.yes_book.best_ask or 999
        self.best_ask_no = self.no_book.best_ask or 999
        self.raw_sum = self.best_ask_yes + self.best_ask_no

        # быстрая проверка
        if self.raw_sum >= 1.0:
            self.reject_reason = f"raw_sum={self.raw_sum:.4f} >= 1.0"
            return

        # считаем реальную стоимость с глубиной стакана
        yes_fill = self.yes_book.cost_to_buy(half)
        no_fill = self.no_book.cost_to_buy(half)

        if not yes_fill or not no_fill:
            self.reject_reason = "insufficient_liquidity"
            return

        if not yes_fill["fully_filled"] or not no_fill["fully_filled"]:
            self.reject_reason = (
                f"partial_fill: yes={yes_fill['unfilled_usd']:.2f} "
                f"no={no_fill['unfilled_usd']:.2f}"
            )
            return

        self.real_avg_yes = yes_fill["avg_price"]
        self.real_avg_no = no_fill["avg_price"]
        self.real_sum = self.real_avg_yes + self.real_avg_no

        # slippage
        self.slippage_pct = ((self.real_sum - self.raw_sum) / self.raw_sum) * 100

        # edge после slippage
        self.gross_edge_pct = (1 - self.real_sum) / self.real_sum * 100

        # edge после комиссий
        # комиссия 2% на выигрышную сторону = ~1% от общей позиции
        effective_fee = self.fee_pct / 2   # ~1% от total
        self.real_net_edge_pct = self.gross_edge_pct - effective_fee

        self.net_edge_pct = self.real_net_edge_pct

        if self.real_net_edge_pct < min_edge_pct:
            self.reject_reason = (
                f"net_edge={self.real_net_edge_pct:.2f}% "
                f"< min={min_edge_pct}%"
            )
            return

        self.is_valid = True

    def summary(self) -> str:
        return (
            f"Fork: {self.event_title[:50]}\n"
            f"  Best ask: YES={self.best_ask_yes:.4f} "
            f"NO={self.best_ask_no:.4f} "
            f"sum={self.raw_sum:.4f}\n"
            f"  Real avg: YES={self.real_avg_yes:.4f} "
            f"NO={self.real_avg_no:.4f} "
            f"sum={self.real_sum:.4f}\n"
            f"  Slippage: {self.slippage_pct:.2f}%\n"
            f"  Gross edge: {self.gross_edge_pct:.2f}%\n"
            f"  Net edge: {self.real_net_edge_pct:.2f}%\n"
            f"  Valid: {self.is_valid} "
            f"{'| ' + self.reject_reason if not self.is_valid else ''}"
        )
