"""
Arbitrage Scanner v1.5
───────────────────────
Изменения vs v1.3:
  • RiskEngine: circuit breaker, cooldown, edge guard, liquidity check
  • CompoundingManager: pending settlement, adaptive sizing, liquidity_check
  • TokenBucketLimiter: rate limiting API запросов
  • ConfigWatchdog: hot-reload .env без рестарта
  • StatsTracker: hourly stats, edge distribution, rejected count
  • Telegram: /risk, /pending, /config команды
  • Логирование: причины отклонения сделок
"""

import asyncio
import aiohttp
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

from config import CFG, DotEnvLoader, ConfigWatchdog
from strategy.fork_scanner import scan_forks
from strategy.sports_binary import scan_sports
from strategy.stats_tracker import StatsTracker
from execution.compounder import CompoundingManager
from execution.risk_engine import RiskEngine
from execution.rate_limiter import RateLimitedSession
from notifications.telegram import (
    send_telegram, check_commands,
)


def setup_logging():
    log = logging.getLogger("arb_scanner")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = RotatingFileHandler(CFG.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


logger = setup_logging()


async def process_fork(fork, compounder, risk, session, seen_events):
    """Обрабатывает одну найденную вилку."""
    event_key = f"fork_{fork.event_id}"
    if event_key in seen_events:
        return False
    seen_events.add(event_key)

    pos = compounder.get_position_size(edge_pct=fork.net_profit_pct)
    if pos <= 0:
        return False

    # ── liquidity check ──────────────────────────────────────────
    volumes = [o.volume for o in fork.outcomes]
    ok, liq_reason = compounder.liquidity_check(pos, volumes)
    if not ok:
        logger.info(f"Fork liquidity fail ({liq_reason}): {fork.event_title[:40]}")
        return False

    # ── risk check ───────────────────────────────────────────────
    min_vol = min(volumes) if volumes else 0
    allowed, risk_reason = risk.can_trade(
        balance=compounder.bankroll,
        edge_pct=fork.net_profit_pct,
        position_size=pos,
        volume24hr=min_vol,
        strategy="fork",
    )
    if not allowed:
        logger.info(f"Fork risk rejected ({risk_reason}): {fork.event_title[:40]}")
        if risk.circuit_breaker:
            await alert_circuit_breaker(session, risk_reason)
        return False

    # ── record trade ─────────────────────────────────────────────
    profit = pos * (fork.net_profit_pct / 100)
    compounder.record_trade(profit, {
        "strategy": "fork",
        "event": fork.event_title[:60],
        "type": fork.fork_type,
        "sum": round(fork.sum_yes, 4),
        "net_pct": round(fork.net_profit_pct, 2),
        "neg_risk": fork.is_neg_risk,
        "min_vol": round(getattr(fork, "min_volume", 0), 0),
        "settlement_scans": 5,
    })

    s = compounder.get_stats()
    alert = (
        f"{fork.format_alert()}\n\n"
        f"💼 VIRTUAL TRADE\n"
        f"Position: ${pos:.2f}\n"
        f"Expected profit: ${profit:+.4f}\n"
        f"Bank: ${s['bankroll']:,.2f} ({s['roi']:+.1f}%)\n"
        f"⏳ Settlement in ~5 scans"
    )
    await send_telegram(session, alert)

    try:
        with open(CFG.FORKS_FILE, "a") as f:
            f.write(json.dumps(fork.to_dict()) + "\n")
    except Exception:
        pass

    return True


async def process_sport(arb, compounder, risk, session, seen_events):
    """Обрабатывает один спортивный арбитраж."""
    event_key = f"sport_{arb.match.event_id}"
    if event_key in seen_events:
        return False
    seen_events.add(event_key)

    pos = compounder.get_position_size(edge_pct=arb.net_edge_pct)
    if pos <= 0:
        return False

    # ── liquidity check ──────────────────────────────────────────
    volumes = [arb.match.volume_a, arb.match.volume_b]
    ok, liq_reason = compounder.liquidity_check(pos, volumes)
    if not ok:
        logger.info(f"Sport liquidity fail ({liq_reason}): {arb.match.event_title[:40]}")
        return False

    # ── risk check ───────────────────────────────────────────────
    min_vol = min(volumes)
    allowed, risk_reason = risk.can_trade(
        balance=compounder.bankroll,
        edge_pct=arb.net_edge_pct,
        position_size=pos,
        volume24hr=min_vol,
        strategy="sports",
    )
    if not allowed:
        logger.info(f"Sport risk rejected ({risk_reason}): {arb.match.event_title[:40]}")
        if risk.circuit_breaker:
            await alert_circuit_breaker(session, risk_reason)
        return False

    # ── record trade ─────────────────────────────────────────────
    profit = pos * (arb.net_edge_pct / 100)
    compounder.record_trade(profit, {
        "strategy": "sports",
        "event": arb.match.event_title[:60],
        "type": arb.arb_type,
        "sum": round(arb.sum_prices, 4),
        "edge_pct": round(arb.net_edge_pct, 2),
        "settlement_scans": 5,
    })

    s = compounder.get_stats()
    alert = (
        f"{arb.format_alert()}\n\n"
        f"💼 VIRTUAL TRADE\n"
        f"Position: ${pos:.2f}\n"
        f"Expected profit: ${profit:+.4f}\n"
        f"Bank: ${s['bankroll']:,.2f} ({s['roi']:+.1f}%)\n"
        f"⏳ Settlement in ~5 scans"
    )
    await send_telegram(session, alert)

    try:
        with open(CFG.SPORTS_FILE, "a") as f:
            f.write(json.dumps(arb.to_dict()) + "\n")
    except Exception:
        pass

    return True


async def main():
    logger.info("=" * 55)
    logger.info("Arbitrage Scanner v1.5 — VIRTUAL TRADING")
    logger.info(f"Interval: {CFG.SCAN_INTERVAL}s")
    logger.info(f"Min profit: {CFG.MIN_PROFIT_PCT}%")
    logger.info(f"Max edge (fake arb guard): {CFG.MAX_EDGE_PCT}%")
    logger.info(f"Min volume: ${CFG.MIN_VOLUME}")
    logger.info("=" * 55)

    # ── инициализация ─────────────────────────────────────────────
    risk = RiskEngine(CFG)
    compounder = CompoundingManager(
        initial_deposit=80.0,
        max_risk_pct=CFG.MAX_POSITION_PCT,
        max_drawdown_pct=CFG.MAX_DRAWDOWN_PCT,
        risk_engine=risk,
    )
    stats = StatsTracker()

    loader = DotEnvLoader(CFG.DOTENV_PATH)
    watchdog = ConfigWatchdog(CFG, loader)

    # ── rate-limited session (3 rps, burst 8) ─────────────────────
    rl_session = RateLimitedSession(rps=3.0, burst=8)

    n = 0
    seen_events: set = set()
    prev_cb = False                      # отслеживаем смену состояния CB

    try:
        async with aiohttp.ClientSession() as tg_session:

            # watchdog запускаем как фоновую задачу
            watchdog_task = asyncio.create_task(watchdog.watch())

            s = compounder.get_stats()
            await send_telegram(tg_session,
                f"🚀 Scanner v1.5 — VIRTUAL TRADING\n"
                f"💰 Bankroll: ${s['bankroll']:,.2f}\n"
                f"📊 Trades: {s['trades']}\n"
                f"🛡 RiskEngine: ON\n"
                f"⚡ RateLimit: 3 rps\n"
                f"🔄 Hot-reload: ON"
            )

            while True:
                n += 1
                logger.info(f"─── Scan #{n} ───")

                try:
                    await check_commands(
                        tg_session, compounder,
                        risk_engine=risk, config=CFG,
                    )

                    # ── settlement pending trades ─────────────────
                    settled = compounder.process_pending()
                    for st in settled:
                        pnl = st["profit"]
                        emoji = "✅" if pnl > 0 else "❌"
                        s = compounder.get_stats()
                        await send_telegram(tg_session,
                            f"{emoji} SETTLED #{st['trade_n']}\n"
                            f"PnL: ${pnl:+.4f}\n"
                            f"Bank: ${s['bankroll']:,.2f} ({s['roi']:+.1f}%)"
                        )

                    # ── circuit breaker alert ─────────────────────
                    if risk.circuit_breaker and not prev_cb:
                        await alert_circuit_breaker(tg_session, risk.circuit_reason)
                    prev_cb = risk.circuit_breaker

                    # ── scan forks ────────────────────────────────
                    rejected_count = 0
                    forks = await scan_forks(rl_session, min_profit=CFG.MIN_PROFIT_PCT)
                    for fork in forks:
                        traded = await process_fork(
                            fork, compounder, risk, tg_session, seen_events
                        )
                        if not traded and f"fork_{fork.event_id}" in seen_events:
                            # seen — не rejected
                            pass
                        elif not traded:
                            rejected_count += 1

                    # ── scan sports ───────────────────────────────
                    sports = await scan_sports(rl_session)
                    for arb in sports:
                        traded = await process_sport(
                            arb, compounder, risk, tg_session, seen_events
                        )
                        if not traded:
                            rejected_count += 1

                    # ── trim seen_events ──────────────────────────
                    if len(seen_events) > 1000:
                        seen_events = set(list(seen_events)[-500:])

                    # ── record scan stats ─────────────────────────
                    stats.record_scan(
                        forks, sports,
                        rejected_count=rejected_count,
                    )

                    api_stats = rl_session.stats()
                    risk_snap = risk.get_snapshot()
                    logger.info(
                        f"Found: {len(forks)} forks + {len(sports)} sports | "
                        f"Rejected: {rejected_count} | "
                        f"Bank: ${compounder.bankroll:.2f} | "
                        f"API: {api_stats['requests']} reqs "
                        f"429s={api_stats['429s']}"
                    )
                    if risk_snap.cooldown_remaining > 0:
                        logger.info(
                            f"RiskEngine cooldown: {risk_snap.cooldown_remaining} scans"
                        )

                except Exception as e:
                    logger.error(f"Scan error: {e}", exc_info=True)

                # ── periodic reports ──────────────────────────────
                if n % 10 == 0:
                    compounder.print_stats()

                if n % 360 == 0:   # каждые 6 часов
                    s = compounder.get_stats()
                    summary = stats.get_summary()
                    api_stats = rl_session.stats()
                    await send_telegram(tg_session,
                        f"{summary}\n"
                        f"💰 Bank: ${s['bankroll']:,.2f}\n"
                        f"📈 Profit: ${s['profit']:+,.2f} ({s['roi']:+.1f}%)\n"
                        f"🔄 Trades: {s['trades']} (pending: {s['pending']})\n"
                        f"📡 API: {api_stats['requests']} reqs"
                    )

                await asyncio.sleep(CFG.SCAN_INTERVAL)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Stopped by user")
    finally:
        watchdog_task.cancel()
        await rl_session.close()
        compounder.print_stats()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
