# execution/polymarket_client.py

import logging
from typing import Optional, List, Dict
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

from config.settings import Settings
from models.order import Order, OrderSide, OrderStatus

logger = logging.getLogger("arb_scanner")


class PolymarketClient:
    """
    Обёртка над py-clob-client.
    Единственное место, где происходит взаимодействие с Polymarket.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = ClobClient(
            host=settings.CLOB_HOST,
            key=settings.PRIVATE_KEY,
            chain_id=settings.CHAIN_ID,
            signature_type=2,          # POLY_GNOSIS_SAFE
            funder=settings.FUNDER_ADDRESS,
        )
        # деривируем API key при первом запуске
        self._api_creds = None
        self._init_api_key()

    def _init_api_key(self):
        """Получаем API credentials от CLOB."""
        try:
            self._api_creds = self.client.derive_api_key()
            self.client.set_api_creds(self._api_creds)
            logger.info("CLOB API key derived successfully")
        except Exception as e:
            # Если ключ уже существует
            try:
                self._api_creds = self.client.create_or_derive_api_creds()
                self.client.set_api_creds(self._api_creds)
                logger.info("CLOB API creds set successfully")
            except Exception as e2:
                logger.error(f"Failed to get API creds: {e2}")
                raise

    # ─── Order Book ──────────────────────────────────────────────

    def get_orderbook(self, token_id: str) -> Optional[Dict]:
        """
        Получает реальный стакан для токена.

        Returns:
            {
                "bids": [{"price": "0.45", "size": "100"}, ...],
                "asks": [{"price": "0.47", "size": "50"}, ...],
            }
        """
        try:
            book = self.client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error(f"Failed to get orderbook for {token_id}: {e}")
            return None

    def get_market_info(self, condition_id: str) -> Optional[Dict]:
        """Получает информацию о рынке."""
        try:
            market = self.client.get_market(condition_id)
            return market
        except Exception as e:
            logger.error(f"Failed to get market {condition_id}: {e}")
            return None

    # ─── Orders ──────────────────────────────────────────────────

    def place_limit_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> Optional[str]:
        """
        Размещает лимитный ордер.

        Args:
            token_id: ID токена
            side: BUY или SELL
            price: цена (0.01 - 0.99)
            size: количество shares

        Returns:
            order_id или None
        """
        try:
            clob_side = BUY if side == OrderSide.BUY else SELL

            order_args = OrderArgs(
                price=price,
                size=size,
                side=clob_side,
                token_id=token_id,
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(
                signed_order,
                orderType=OrderType.GTC,  # Good Till Cancelled
            )

            order_id = response.get("orderID") or response.get("id")
            if order_id:
                logger.info(
                    f"Order placed: {order_id} "
                    f"{side.value} {size}@{price} "
                    f"token={token_id[:16]}..."
                )
                return order_id
            else:
                logger.error(f"No order_id in response: {response}")
                return None

        except Exception as e:
            logger.error(
                f"Failed to place order: {side.value} "
                f"{size}@{price} token={token_id[:16]}... "
                f"Error: {e}"
            )
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Отменяет ордер."""
        try:
            result = self.client.cancel(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Аварийная отмена всех ордеров."""
        try:
            result = self.client.cancel_all()
            logger.warning("ALL ORDERS CANCELLED")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Получает статус ордера."""
        try:
            order = self.client.get_order(order_id)
            return order
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """Получает все открытые ордера."""
        try:
            orders = self.client.get_orders(
                # open_only=True  # зависит от версии клиента
            )
            return orders or []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    # ─── Balance ─────────────────────────────────────────────────

    def get_balance(self) -> Optional[float]:
        """Получает баланс USDC."""
        try:
            balance = self.client.get_balance()
            return float(balance) if balance else None
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return None

    # ─── Positions ───────────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        """Получает текущие позиции."""
        try:
            positions = self.client.get_positions()
            return positions or []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
