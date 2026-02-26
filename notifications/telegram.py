import aiohttp
import logging
from config import CFG

logger = logging.getLogger("arb_scanner.telegram")

_last_processed_id = 0


async def send_telegram(session, text):
    if not CFG.TELEGRAM_TOKEN or not CFG.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CFG.TELEGRAM_CHAT_ID, "text": text[:4096]}
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning(f"TG send failed: {resp.status}")
    except Exception as e:
        logger.warning(f"TG error: {e}")


async def alert_circuit_breaker(session, reason):
    await send_telegram(session, f"CIRCUIT BREAKER\nReason: {reason}\nUse /reset to resume.")


async def check_commands(session, compounder, risk_engine=None, config=None):
    global _last_processed_id
    if not CFG.TELEGRAM_TOKEN:
        return None
    url = f"https://api.telegram.org/bot{CFG.TELEGRAM_TOKEN}/getUpdates?offset={_last_processed_id + 1}&timeout=1"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        results = data.get("result", [])
        if not results:
            return None
        for update in results:
            update_id = update.get("update_id", 0)
            _last_processed_id = max(_last_processed_id, update_id)
            msg = update.get("message", {})
            text = (msg.get("text") or "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if chat_id != CFG.TELEGRAM_CHAT_ID:
                continue
            if text == "/stats":
                s = compounder.get_stats()
                reply = (
                    f"Portfolio\n\n"
                    f"Bank: ${s['bankroll']:,.2f}\n"
                    f"Deposited: ${s['deposited']:,.2f}\n"
                    f"Profit: ${s['profit']:+,.2f}\n"
                    f"ROI: {s['roi']:+.1f}%\n"
                    f"WR: {s['win_rate']:.0f}% ({s['trades']} trades)\n"
                    f"Pending: {s['pending']}\n"
                    f"Avg edge: {s['avg_edge']:.2f}%\n"
                    f"Drawdown: {s['drawdown']:.1f}%"
                )
                await send_telegram(session, reply)
            elif text == "/risk":
                if risk_engine:
                    snap = risk_engine.get_snapshot()
                    status = "STOPPED" if snap.circuit_breaker else ("COOLDOWN" if snap.cooldown_remaining > 0 else "ACTIVE")
                    reply = (
                        f"Risk Engine: {status}\n\n"
                        f"Daily PnL: ${snap.daily_pnl:+.4f}\n"
                        f"Trades/hr: {snap.trades_last_hour}\n"
                        f"Loss streak: {snap.loss_streak}\n"
                        f"Cooldown: {snap.cooldown_remaining} scans"
                        + (f"\nReason: {snap.circuit_reason}" if snap.circuit_breaker else "")
                    )
                else:
                    reply = "Risk engine not available"
                await send_telegram(session, reply)
            elif text == "/pending":
                pending = compounder._pending
                if not pending:
                    reply = "No pending trades"
                else:
                    lines = [f"Pending trades ({len(pending)}):"]
                    for pt in pending[:10]:
                        lines.append(f"  #{pt.trade_n} ${pt.gross_profit:+.4f} - {pt.scans_remaining} scans left")
                    reply = "\n".join(lines)
                await send_telegram(session, reply)
            elif text == "/config":
                cfg = config or CFG
                reply = (
                    f"Config\n\n"
                    f"MIN_PROFIT_PCT: {cfg.MIN_PROFIT_PCT}%\n"
                    f"MAX_EDGE_PCT: {cfg.MAX_EDGE_PCT}%\n"
                    f"MIN_VOLUME: ${cfg.MIN_VOLUME}\n"
                    f"MAX_POSITION_PCT: {cfg.MAX_POSITION_PCT}%\n"
                    f"MAX_DAILY_LOSS_PCT: {cfg.MAX_DAILY_LOSS_PCT}%\n"
                    f"SCAN_INTERVAL: {cfg.SCAN_INTERVAL}s"
                )
                await send_telegram(session, reply)
            elif text.startswith("/deposit"):
                try:
                    amount = float(text.split()[1])
                    compounder.add_deposit(amount)
                    await send_telegram(session, f"Deposit +${amount:.2f}\nBank: ${compounder.bankroll:.2f}")
                except Exception:
                    await send_telegram(session, "Usage: /deposit 160")
            elif text == "/reset":
                compounder.reset_stop()
                await send_telegram(session, "Stop and circuit breaker reset")
            elif text == "/help":
                await send_telegram(session,
                    "Commands v1.5:\n"
                    "/stats - portfolio\n"
                    "/risk - risk engine\n"
                    "/pending - pending settlements\n"
                    "/config - current config\n"
                    "/deposit 160 - add deposit\n"
                    "/reset - reset emergency stop\n"
                    "/help - this message"
                )
    except Exception:
        pass
    return None
