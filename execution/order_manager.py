# execution/order_manager.py

import asyncio
import logging
import uuid
from typing import Optional, Tuple
from datetime import datetime, timezone

from models.order import Order, OrderSide, OrderStatus
from models.position import ArbPosition, PositionStatus
from models.fork import RealFork
from execution.polymarket_client import PolymarketClient

logger = logging.getLogger("arb_scanner")


class OrderManager:
    """
    Управляет жизненным циклом арбитражных позиций.

    Ключевые принципы:
    1. Обе ноги размещаются как LIMIT ордера
    2. Если одна нога не исполняется — UNWIND
    3. Все состояния отслеживаются
    """

    def __init__(
        self,
        client: PolymarketClient,
        max_position_usd: float = 5.0,
        order_ttl: int = 30,
        fill_check_interval: float = 2.0,
        max_fill_wait: float = 30.0,
        max_slippage_pct: float = 0.5,
    ):
        self.client = client
        self.max_position_usd = max_position_usd
        self.order_ttl = order_ttl
        self.fill_check_interval = fill_check_interval
        self.max_fill_wait = max_fill_wait
        self.max_slippage_pct = max_slippage_pct

        self.active_positions: dict[str, ArbPosition] = {}
        self.closed_positions: list[ArbPosition] = []
        self.total_pnl: float = 0.0
        self.total_fees: float = 0.0

    async def execute_fork(self, fork: RealFork) -> Optional[ArbPosition]:
        """
        Исполняет арбитражную вилку.

        Стратегия:
        1. Размещаем обе ноги одновременно как LIMIT
        2. Мониторим исполнение
        3. Если одна нога не fill — отменяем обе
        4. Если частичный fill — unwind
        """
        position_id = f"pos_{uuid.uuid4().hex[:8]}"

        position = ArbPosition(
            position_id=position_id,
            event_id=fork.event_id,
            event_title=fork.event_title,
            condition_id=fork.condition_id,
            strategy="fork",
            expected_edge_pct=fork.gross_edge_pct,
            expected_net_pct=fork.real_net_edge_pct,
            target_size=fork.entry_size_usd,
            status=PositionStatus.OPENING,
        )

        half_size = fork.entry_size_usd / 2

        # ── Рассчитываем размеры в shares ────────────────────────
        yes_shares = half_size / fork.real_avg_yes
        no_shares = half_size / fork.real_avg_no

        # ── Создаём ордера ────────────────────────────────────────
        # Используем ЛИМИТНЫЕ ордера чуть ВЫШЕ best ask
        # чтобы повысить вероятность fill, но ограничить slippage
        yes_limit = min(
            fork.best_ask_yes * (1 + self.max_slippage_pct / 100),
            0.99
        )
        no_limit = min(
            fork.best_ask_no * (1 + self.max_slippage_pct / 100),
            0.99
        )

        # Проверяем: сумма лимитов всё ещё даёт прибыль?
        if yes_limit + no_limit >= 1.0:
            logger.warning(
                f"Fork {position_id}: limit sum "
                f"{yes_limit + no_limit:.4f} >= 1.0, skip"
            )
            return None

        # ── Размещаем ноги ────────────────────────────────────────
        logger.info(
            f"Executing fork {position_id}: "
            f"YES {yes_shares:.1f}@{yes_limit:.4f} + "
            f"NO {no_shares:.1f}@{no_limit:.4f}"
        )

        leg_a = Order(
            internal_id=f"{position_id}_yes",
            token_id=fork.yes_book.token_id,
            market_slug=fork.event_title[:30],
            side=OrderSide.BUY,
            price=round(yes_limit, 2),
            size=round(yes_shares, 2),
            ttl_seconds=self.order_ttl,
        )

        leg_b = Order(
            internal_id=f"{position_id}_no",
            token_id=fork.no_book.token_id,
            market_slug=fork.event_title[:30],
            side=OrderSide.BUY,
            price=round(no_limit, 2),
            size=round(no_shares, 2),
            ttl_seconds=self.order_ttl,
        )

        # ── Отправляем обе ноги ───────────────────────────────────
        yes_order_id = self.client.place_limit_order(
            token_id=leg_a.token_id,
            side=leg_a.side,
            price=leg_a.price,
            size=leg_a.size,
        )

        no_order_id = self.client.place_limit_order(
            token_id=leg_b.token_id,
            side=leg_b.side,
            price=leg_b.price,
            size=leg_b.size,
        )

        if not yes_order_id:
            logger.error(f"{position_id}: YES order failed")
            if no_order_id:
                self.client.cancel_order(no_order_id)
            position.status = PositionStatus.FAILED
            position.close_reason = "yes_order_failed"
            self.closed_positions.append(position)
            return position

        if not no_order_id:
            logger.error(f"{position_id}: NO order failed")
            self.client.cancel_order(yes_order_id)
            position.status = PositionStatus.FAILED
            position.close_reason = "no_order_failed"
            self.closed_positions.append(position)
            return position

        leg_a.order_id = yes_order_id
        leg_a.status = OrderStatus.SUBMITTED
        leg_b.order_id = no_order_id
        leg_b.status = OrderStatus.SUBMITTED

        position.leg_a = leg_a
        position.leg_b = leg_b

        self.active_positions[position_id] = position

        # ── Мониторим исполнение ──────────────────────────────────
        result = await self._monitor_fill(position)

        return result

    async def _monitor_fill(self, position: ArbPosition) -> ArbPosition:
        """
        Мониторит исполнение обеих ног.
        Таймаут: max_fill_wait секунд.
        """
        start = datetime.now(timezone.utc)
        pid = position.position_id

        while True:
            elapsed = (datetime.now(timezone.utc) - start).total_seconds()

            if elapsed > self.max_fill_wait:
                logger.warning(f"{pid}: fill timeout after {elapsed:.0f}s")
                await self._handle_timeout(position)
                break

            # проверяем статус обеих ног
            a_filled = await self._check_order_fill(position.leg_a)
            b_filled = await self._check_order_fill(position.leg_b)

            if a_filled and b_filled:
                position.status = PositionStatus.BOTH_FILLED
                self._calculate_pnl(position)
                logger.info(
                    f"{pid}: BOTH FILLED | "
                    f"cost={position.actual_cost:.4f} "
                    f"edge={position.actual_edge_pct:.2f}%"
                )
                break

            await asyncio.sleep(self.fill_check_interval)

        # переносим в closed
        if position.is_terminal:
            self.active_positions.pop(pid, None)
            self.closed_positions.append(position)

        return position

    async def _check_order_fill(self, order: Order) -> bool:
        """Проверяет исполнение ордера через API."""
        if order.status == OrderStatus.FILLED:
            return True

        status = self.client.get_order_status(order.order_id)
        if not status:
            return False

        # Парсим ответ CLOB API
        api_status = status.get("status", "").upper()
        filled_size = float(status.get("size_matched", 0))

        if api_status == "MATCHED" or filled_size >= order.size * 0.98:
            order.status = OrderStatus.FILLED
            order.filled_size = filled_size
            order.avg_fill_price = float(
                status.get("price", order.price)
            )
            return True

        elif filled_size > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.filled_size = filled_size
            order.avg_fill_price = float(
                status.get("price", order.price)
            )

        return False

    async def _handle_timeout(self, position: ArbPosition):
        """
        Обработка таймаута — самая важная часть.
        Если одна нога исполнена а другая нет — UNWIND.
        """
        pid = position.position_id
        leg_a = position.leg_a
        leg_b = position.leg_b

        a_filled = leg_a.status in (
            OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED
        )
        b_filled = leg_b.status in (
            OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED
        )

        if not a_filled and not b_filled:
            # ни одна нога не исполнена — просто отменяем
            self.client.cancel_order(leg_a.order_id)
            self.client.cancel_order(leg_b.order_id)
            position.status = PositionStatus.FAILED
            position.close_reason = "timeout_no_fills"
            logger.info(f"{pid}: no fills, cancelled both")
            return

        if a_filled and not b_filled:
            # YES исполнен, NO нет — отменяем NO и продаём YES
            self.client.cancel_order(leg_b.order_id)
            position.status = PositionStatus.UNWINDING
            await self._unwind_leg(position, leg_a, "yes")
            return

        if b_filled and not a_filled:
            # NO исполнен, YES нет — отменяем YES и продаём NO
            self.client.cancel_order(leg_a.order_id)
            position.status = PositionStatus.UNWINDING
            await self._unwind_leg(position, leg_b, "no")
            return

        if a_filled and b_filled:
            # оба частично исполнены
            position.status = PositionStatus.BOTH_FILLED
            self._calculate_pnl(position)

    async def _unwind_leg(
        self, position: ArbPosition, filled_leg: Order, leg_name: str
    ):
        """
        Продаёт исполненную ногу обратно (UNWIND).
        Это УБЫТОЧНАЯ операция — мы теряем на спреде.
        """
        pid = position.position_id
        logger.warning(
            f"{pid}: UNWINDING {leg_name} leg | "
            f"filled={filled_leg.filled_size}@{filled_leg.avg_fill_price}"
        )

        # Продаём по рынку (агрессивная цена)
        # Ставим цену чуть ниже best bid чтобы быстро продать
        book = self.client.get_orderbook(filled_leg.token_id)
        if book and book.get("bids"):
            best_bid = max(float(b["price"]) for b in book["bids"])
            sell_price = round(best_bid - 0.01, 2)  # на 1 цент ниже
        else:
            sell_price = round(filled_leg.avg_fill_price * 0.95, 2)

        sell_price = max(0.01, sell_price)

        unwind_order_id = self.client.place_limit_order(
            token_id=filled_leg.token_id,
            side=OrderSide.SELL,
            price=sell_price,
            size=filled_leg.filled_size,
        )

        if unwind_order_id:
            # Ждём исполнения unwind (с коротким таймаутом)
            unwind = Order(
                order_id=unwind_order_id,
                internal_id=f"{pid}_unwind_{leg_name}",
                token_id=filled_leg.token_id,
                side=OrderSide.SELL,
                price=sell_price,
                size=filled_leg.filled_size,
                ttl_seconds=60,
            )
            position.unwind_orders.append(unwind)

            await asyncio.sleep(10)
            await self._check_order_fill(unwind)

            # Считаем убыток от unwind
            loss = filled_leg.filled_size * (
                filled_leg.avg_fill_price - sell_price
            )
            position.realized_pnl = -loss
            position.close_reason = f"unwind_{leg_name}_leg"
        else:
            position.close_reason = f"unwind_failed_{leg_name}"
            # КРИТИЧЕСКАЯ СИТУАЦИЯ: у нас открытая позиция
            # которую не можем закрыть
            logger.critical(
                f"{pid}: UNWIND FAILED! Manual intervention needed!"
            )

        position.status = PositionStatus.CLOSED

    def _calculate_pnl(self, position: ArbPosition):
        """Считает P&L для полностью исполненной позиции."""
        a = position.leg_a
        b = position.leg_b

        min_shares = min(a.filled_size, b.filled_size)
        cost = (
            a.filled_size * a.avg_fill_price +
            b.filled_size * b.avg_fill_price
        )

        # Выручка = $1 за каждую пару (YES + NO = $1 при settlement)
        revenue = min_shares

        # Комиссия: 2% на выигрышную сторону
        fee = min_shares * 0.02
        position.total_fees = fee

        position.realized_pnl = revenue - cost - fee
        self.total_pnl += position.realized_pnl
        self.total_fees += fee

    def get_stats(self) -> dict:
        return {
            "active_positions": len(self.active_positions),
            "closed_positions": len(self.closed_positions),
            "total_pnl": round(self.total_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "wins": sum(
                1 for p in self.closed_positions
                if p.realized_pnl > 0
            ),
            "losses": sum(
                1 for p in self.closed_positions
                if p.realized_pnl < 0
            ),
            "failed": sum(
                1 for p in self.closed_positions
                if p.status == PositionStatus.FAILED
            ),
            "unwound": sum(
                1 for p in self.closed_positions
                if "unwind" in p.close_reason
            ),
      }
