# main.py

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import CFG
from execution.polymarket_client import PolymarketClient
from execution.order_manager import OrderManager
from core.scanner import ForkScanner
from risk.risk_engine import RiskEngineV2
from notifications.telegram import TelegramNotifier

logger = logging.getLogger("arb_scanner")


async def main():
    print("=" * 60)
    print("  Polymarket Arbitrage Bot v2.0")
    print(f"  Mode: {'📝 PAPER' if CFG.PAPER_TRADING else '💰 LIVE'}")
    print(f"  Max position: ${CFG.MAX_POSITION_USD}")
    print(f"  Min net edge: {CFG.MIN_NET_EDGE_PCT}%")
    print("=" * 60)

    if not CFG.PAPER_TRADING:
        print("\n⚠️  LIVE TRADING MODE ⚠️")
        print("Starting in 10 seconds... Ctrl+C to cancel")
        await asyncio.sleep(10)

    # ── Инициализация ─────────────────────────────────────────
    client = PolymarketClient(CFG)
    tg = TelegramNotifier(CFG.TG_TOKEN, CFG.TG_CHAT_ID)

    # Проверяем баланс
    balance = client.get_balance()
    if balance is None:
        logger.error("Cannot get balance. Check credentials.")
        return

    logger.info(f"USDC Balance: ${balance:.2f}")

    if balance < CFG.MAX_POSITION_USD and not CFG.PAPER_TRADING:
        logger.error(
            f"Insufficient balance: ${balance:.2f} "
            f"< ${CFG.MAX_POSITION_USD}"
        )
        return

    order_mgr = OrderManager(
        client=client,
        max_position_usd=CFG.MAX_POSITION_USD,
        order_ttl=CFG.ORDER_TTL,
        max_fill_wait=CFG.MAX_FILL_WAIT,
        max_slippage_pct=CFG.MAX_SLIPPAGE_PCT,
    )

    risk = RiskEngineV2(CFG)
    scanner = ForkScanner(client, CFG)

    await tg.send(
        f"🤖 Bot v2.0 Started\n"
        f"Mode: {'📝 PAPER' if CFG.PAPER_TRADING else '💰 LIVE'}\n"
        f"Balance: ${balance:.2f}\n"
        f"Max position: ${CFG.MAX_POSITION_USD}\n"
        f"Min edge: {CFG.MIN_NET_EDGE_PCT}%"
    )

    scan_n = 0
    while True:
        try:
            scan_n += 1
            logger.info(f"─── Scan #{scan_n} ───")

            # ── Проверка risk limits ──────────────────────────
            if not risk.can_trade():
                logger.info("Risk: trading paused")
                await asyncio.sleep(CFG.SCAN_INTERVAL)
                continue

            # ── Проверка активных позиций ─────────────────────
            if len(order_mgr.active_positions) >= CFG.MAX_CONCURRENT:
                logger.info("Max concurrent positions reached")
                await asyncio.sleep(CFG.SCAN_INTERVAL)
                continue

            # ── Сканируем рынки ───────────────────────────────
            forks = await scanner.find_forks(
                entry_size=CFG.MAX_POSITION_USD,
                min_edge=CFG.MIN_NET_EDGE_PCT,
                max_edge=CFG.MAX_EDGE_PCT,
            )

            valid_forks = [f for f in forks if f.is_valid]
            logger.info(
                f"Scanned: {len(forks)} total, "
                f"{len(valid_forks)} valid forks"
            )

            for fork in valid_forks:
                logger.info(fork.summary())

                if CFG.PAPER_TRADING:
                    # В paper mode — логируем, но не торгуем
                    await tg.send(
                        f"📊 PAPER: Fork found\n"
                        f"{fork.summary()}\n"
                        f"(No real orders placed)"
                    )
                    continue

                # ── LIVE EXECUTION ────────────────────────────
                await tg.send(
                    f"🎯 Executing fork:\n"
                    f"Event: {fork.event_title[:50]}\n"
                    f"Net edge: {fork.real_net_edge_pct:.2f}%\n"
                    f"Size: ${fork.entry_size_usd:.2f}"
                )

                position = await order_mgr.execute_fork(fork)

                if position:
                    stats = order_mgr.get_stats()
                    emoji = (
                        "✅" if position.realized_pnl > 0
                        else "❌" if position.realized_pnl < 0
                        else "⏳"
                    )

                    await tg.send(
                        f"{emoji} Position {position.position_id}\n"
                        f"Status: {position.status.value}\n"
                        f"PnL: ${position.realized_pnl:+.4f}\n"
                        f"Fees: ${position.total_fees:.4f}\n"
                        f"Reason: {position.close_reason}\n\n"
                        f"Total PnL: ${stats['total_pnl']:+.4f}\n"
                        f"W/L: {stats['wins']}/{stats['losses']}\n"
                        f"Unwound: {stats['unwound']}"
                    )

                    risk.record_trade(position.realized_pnl)

                # Только одна сделка за скан
                break

            # ── Периодические отчёты ──────────────────────────
            if scan_n % 100 == 0:
                stats = order_mgr.get_stats()
                balance = client.get_balance() or 0
                await tg.send(
                    f"📊 Report #{scan_n}\n"
                    f"Balance: ${balance:.2f}\n"
                    f"PnL: ${stats['total_pnl']:+.4f}\n"
                    f"Trades: {stats['closed_positions']}\n"
                    f"Wins: {stats['wins']} | "
                    f"Losses: {stats['losses']}\n"
                    f"Failed: {stats['failed']} | "
                    f"Unwound: {stats['unwound']}"
                )

        except Exception as e:
            logger.error(f"Scan error: {e}", exc_info=True)

        await asyncio.sleep(CFG.SCAN_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
