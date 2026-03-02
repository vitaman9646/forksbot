# execution/orderbook_checker.py

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import aiohttp

from config import CFG

logger = logging.getLogger("arb_scanner.orderbook")


@dataclass
class DepthLegResult:
    outcome_name: str
    mid_price: float
    best_bid: float
    best_ask: float
    ask_depth_usd: float
    slippage_pct: float
    error: Optional[str] = None


@dataclass
class DepthCheckResult:
    is_executable: bool
    real_edge_pct: float
    real_edge_usd: float
    min_executable_usd: float
    reject_reason: str
    fetch_time_ms: float
    legs: List[DepthLegResult] = field(default_factory=list)


class OrderbookChecker:
    """
    Профессиональный модуль проверки стакана для вилок.

    Задачи:
      • получить стакан по каждому исходу (через CLOB API),
      • смоделировать исполнение заданного объёма (USD),
      • оценить slippage и реальный edge,
      • решить: вилка исполнима или нет.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        min_executable_usd: float = 5.0,
        max_slippage_pct: float = 0.5,
        depth_ticks: int = 3,
        max_position_to_depth_ratio: float = 0.3,
    ):
        self.session = session
        self.min_executable_usd = float(min_executable_usd)
        self.max_slippage_pct = float(max_slippage_pct)
        self.depth_ticks = int(depth_ticks)
        self.max_position_to_depth_ratio = float(max_position_to_depth_ratio)

        self._timeout = aiohttp.ClientTimeout(total=getattr(CFG, "FETCH_TIMEOUT", 15))

    async def _fetch_orderbook(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Получает стакан для одного токена через POLYMARKET_CLOB.

        Ожидаемый формат (примерный):
        {
          "bids": [[price, size], ...] или [{"p": price, "s": size}, ...],
          "asks": [[price, size], ...] или [{"p": price, "s": size}, ...]
        }
        """
        if not token_id:
            return None

        url = CFG.POLYMARKET_CLOB
        params = {"token_id": token_id}

        try:
            async with self.session.get(
                url, params=params, timeout=self._timeout
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"CLOB {token_id}: HTTP {resp.status}")
                    return None
                data = await resp.json()
                return data
        except asyncio.TimeoutError:
            logger.debug(f"CLOB {token_id}: timeout")
        except Exception as e:
            logger.debug(f"CLOB {token_id}: error {e}")
        return None

    @staticmethod
    def _normalize_book(raw: Dict[str, Any]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Приводит стакан к виду:
          bids = [(price, size), ...] по убыванию цены
          asks = [(price, size), ...] по возрастанию цены
        """
        def _parse_side(side):
            if not side:
                return []
            out = []
            for lvl in side:
                if isinstance(lvl, dict):
                    p = float(lvl.get("p") or lvl.get("price") or 0)
                    s = float(lvl.get("s") or lvl.get("size") or 0)
                elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    p = float(lvl[0])
                    s = float(lvl[1])
                else:
                    continue
                if p > 0 and s > 0:
                    out.append((p, s))
            return out

        bids = _parse_side(raw.get("bids") or raw.get("bid") or [])
        asks = _parse_side(raw.get("asks") or raw.get("ask") or [])

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks

    @staticmethod
    def _mid_price(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        if not bids and not asks:
            return 0.0
        if not bids:
            return asks[0][0]
        if not asks:
            return bids[0][0]
        return (bids[0][0] + asks[0][0]) / 2.0

    def _simulate_leg_execution(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        target_usd: float,
        ref_price: float,
    ) -> Tuple[Optional[float], float, float, Optional[str]]:
        """
        Симулирует покупку target_usd по стороне ask.

        Возвращает:
          avg_fill_price, depth_usd, slippage_pct, error
        """
        if target_usd <= 0:
            return None, 0.0, 0.0, "zero_target"

        remaining = target_usd
        cost = 0.0
        depth_usd = 0.0

        for price, size in asks[: self.depth_ticks]:
            level_usd = price * size
            if level_usd <= 0:
                continue
            take_usd = min(level_usd, remaining)
            cost += take_usd
            depth_usd += level_usd
            remaining -= take_usd
            if remaining <= 1e-6:
                break

        if cost <= 0:
            return None, depth_usd, 0.0, "no_liquidity"

        if remaining > 1e-6:
            # Недостаточно глубины для полного исполнения
            return None, depth_usd, 0.0, "insufficient_depth"

        avg_fill_price = cost / target_usd
        if ref_price <= 0:
            slippage_pct = 0.0
        else:
            slippage_pct = (avg_fill_price - ref_price) / ref_price * 100.0

        return avg_fill_price, depth_usd, slippage_pct, None

    async def check_fork(
        self,
        outcomes: List[Dict[str, Any]],
        position_size_usd: float,
        market_id: str,
    ) -> DepthCheckResult:
        """
        Главный метод: проверяет, можно ли исполнить вилку на заданный объём.

        outcomes: [
          {"token_id": str, "name": str, "price": float},
          ...
        ]
        """
        t0 = time.time()
        legs: List[DepthLegResult] = []

        if position_size_usd <= 0:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=0.0,
                real_edge_usd=0.0,
                min_executable_usd=0.0,
                reject_reason="zero_position",
                fetch_time_ms=0.0,
                legs=[],
            )

        n_legs = len(outcomes)
        if n_legs == 0:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=0.0,
                real_edge_usd=0.0,
                min_executable_usd=0.0,
                reject_reason="no_outcomes",
                fetch_time_ms=0.0,
                legs=[],
            )

        # равномерно делим позицию по ногам вилки
        per_leg_usd = position_size_usd / n_legs

        # сначала собираем стаканы по всем исходам
        books = {}
        for o in outcomes:
            token_id = o.get("token_id") or ""
            if not token_id:
                legs.append(
                    DepthLegResult(
                        outcome_name=o.get("name", "")[:30],
                        mid_price=o.get("price", 0.0),
                        best_bid=0.0,
                        best_ask=0.0,
                        ask_depth_usd=0.0,
                        slippage_pct=0.0,
                        error="no_token_id",
                    )
                )
                continue

            ob = await self._fetch_orderbook(token_id)
            if not ob:
                legs.append(
                    DepthLegResult(
                        outcome_name=o.get("name", "")[:30],
                        mid_price=o.get("price", 0.0),
                        best_bid=0.0,
                        best_ask=0.0,
                        ask_depth_usd=0.0,
                        slippage_pct=0.0,
                        error="no_orderbook",
                    )
                )
                continue

            bids, asks = self._normalize_book(ob)
            mid = self._mid_price(bids, asks)
            best_bid = bids[0][0] if bids else 0.0
            best_ask = asks[0][0] if asks else 0.0

            avg_fill_price, depth_usd, slippage_pct, err = self._simulate_leg_execution(
                bids=bids,
                asks=asks,
                target_usd=per_leg_usd,
                ref_price=float(o.get("price") or mid or best_ask or 0.0),
            )

            legs.append(
                DepthLegResult(
                    outcome_name=o.get("name", "")[:30],
                    mid_price=mid,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    ask_depth_usd=depth_usd,
                    slippage_pct=slippage_pct if avg_fill_price is not None else 0.0,
                    error=err,
                )
            )

        fetch_ms = (time.time() - t0) * 1000.0

        # базовые проверки по ногам
        # 1) ошибки / отсутствие ликвидности
        for leg in legs:
            if leg.error in ("no_orderbook", "no_liquidity", "insufficient_depth"):
                return DepthCheckResult(
                    is_executable=False,
                    real_edge_pct=0.0,
                    real_edge_usd=0.0,
                    min_executable_usd=0.0,
                    reject_reason=leg.error or "depth_error",
                    fetch_time_ms=fetch_ms,
                    legs=legs,
                )

        # 2) проверка глубины и ratio
        min_depth_usd = min((leg.ask_depth_usd for leg in legs if leg.ask_depth_usd > 0), default=0.0)
        if min_depth_usd <= 0:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=0.0,
                real_edge_usd=0.0,
                min_executable_usd=0.0,
                reject_reason="zero_depth",
                fetch_time_ms=fetch_ms,
                legs=legs,
            )

        # отношение позиции к глубине
        depth_ratio = position_size_usd / min_depth_usd
        if depth_ratio > self.max_position_to_depth_ratio:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=0.0,
                real_edge_usd=0.0,
                min_executable_usd=min_depth_usd,
                reject_reason="position_too_large_vs_depth",
                fetch_time_ms=fetch_ms,
                legs=legs,
            )

        # 3) проверка slippage
        max_leg_slippage = max((abs(leg.slippage_pct) for leg in legs), default=0.0)
        if max_leg_slippage > self.max_slippage_pct:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=0.0,
                real_edge_usd=0.0,
                min_executable_usd=min_depth_usd,
                reject_reason=f"slippage_too_high({max_leg_slippage:.3f}%)",
                fetch_time_ms=fetch_ms,
                legs=legs,
            )

        # 4) оценка реального edge на основе скорректированных цен
        #    (грубая модель: считаем, что slippage уже учтён в ref_price → edge ≈ gamma_edge)
        #    Здесь мы можем приблизительно пересчитать sum_yes по mid/ask, но для v1
        #    достаточно считать, что edge не ухудшился сильнее max_slippage_pct.
        #    Поэтому просто возвращаем "нулевой" real_edge, а реальный edge
        #    будет скорректирован в сканере через fork.net_profit_pct.
        real_edge_pct = 0.0
        real_edge_usd = 0.0

        # Но чтобы было полезно для анализа, попробуем оценить "эффективную" сумму цен:
        effective_prices = []
        for leg in legs:
            # если mid_price есть — используем его, иначе best_ask
            base_price = leg.mid_price or leg.best_ask
            if base_price <= 0:
                continue
            # slippage уже в процентах относительно ref_price, но для простоты
            # считаем, что эффективная цена ≈ base_price * (1 + slippage_pct/100)
            eff_price = base_price * (1.0 + leg.slippage_pct / 100.0)
            effective_prices.append(eff_price)

        if len(effective_prices) == n_legs:
            sum_eff = sum(effective_prices)
            # грубая оценка edge (без комиссий), как в detect_fork
            if sum_eff < 1.0:
                raw_pct = ((1.0 - sum_eff) / sum_eff) * 100.0
            else:
                raw_pct = ((sum_eff - 1.0) / 1.0) * 100.0
            real_edge_pct = raw_pct
            real_edge_usd = position_size_usd * (real_edge_pct / 100.0)

        # финальное решение: исполнимо, если глубина >= min_executable_usd
        if min_depth_usd < self.min_executable_usd:
            return DepthCheckResult(
                is_executable=False,
                real_edge_pct=real_edge_pct,
                real_edge_usd=real_edge_usd,
                min_executable_usd=min_depth_usd,
                reject_reason="depth_below_min_executable",
                fetch_time_ms=fetch_ms,
                legs=legs,
            )

        return DepthCheckResult(
            is_executable=True,
            real_edge_pct=real_edge_pct,
            real_edge_usd=real_edge_usd,
            min_executable_usd=min_depth_usd,
            reject_reason="",
            fetch_time_ms=fetch_ms,
            legs=legs,
      )
