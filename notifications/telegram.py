# notifications/telegram.py
"""
Telegram Notifier v2.0
──────────────────────
Изменения vs v1.5:
  • Класс вместо глобальных функций
  • Новые команды: /positions, /kill, /pause, /resume, /balance
  • Команды адаптированы под реальный трейдинг
  • Rate limiting отправки (не больше 20 msg/min)
  • Подтверждение опасных действий (/kill)
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timezone
from typing import Optional, Callable, Any

logger = logging.getLogger("arb_scanner.telegram")


class TelegramNotifier:
    """
    Единственный класс для всего общения с Telegram.
    Используется и для отправки алертов, и для приёма команд.
    """

    TG_API = "https://api.telegram.org/bot{token}"
    MAX_MSG_LEN = 4096
    MAX_MESSAGES_PER_MIN = 20

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = self.TG_API.format(token=token)
        self._last_update_id = 0
        self._msg_count = 0
        self._msg_reset_time = datetime.now(timezone.utc)

        # для подтверждения опасных действий
        self._pending_confirm: Optional[str] = None
        self._confirm_expires: float = 0

        # callback-и на команды (регистрируются из main)
        self._handlers: dict[str, Callable] = {}

        self.enabled = bool(token and chat_id)
        if not self.enabled:
            logger.warning("Telegram not configured")

    # ══════════════════════════════════════════════════════════
    #  SENDING
    # ══════════════════════════════════════════════════════════

    async def send(self, text: str, parse_mode: str = None):
        """Отправляет сообщение в Telegram."""
        if not self.enabled:
            return

        # rate limiting
        now = datetime.now(timezone.utc)
        if (now - self._msg_reset_time).total_seconds() > 60:
            self._msg_count = 0
            self._msg_reset_time = now

        if self._msg_count >= self.MAX_MESSAGES_PER_MIN:
            logger.warning("TG rate limit reached, skipping message")
            return

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text[:self.MAX_MSG_LEN],
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        self._msg_count += 1
                    else:
                        body = await resp.text()
                        logger.warning(
                            f"TG send failed: {resp.status} {body[:100]}"
                        )
        except Exception as e:
            logger.warning(f"TG send error: {e}")

    async def send_alert(self, text: str):
        """Алерт с эмодзи-префиксом."""
        await self.send(f"🔔 {text}")

    async def send_error(self, text: str):
        """Отправка ошибки."""
        await self.send(f"🚨 {text}")

    async def send_trade(self, text: str):
        """Отправка о сделке."""
        await self.send(f"💰 {text}")

    # ══════════════════════════════════════════════════════════
    #  COMMAND REGISTRATION
    # ══════════════════════════════════════════════════════════

    def register_handler(self, command: str, handler: Callable):
        """
        Регистрирует обработчик команды.

        Usage:
            tg.register_handler("/stats", my_stats_handler)

        Handler signature:
            async def handler() -> str:
                return "reply text"
        """
        self._handlers[command] = handler

    # ══════════════════════════════════════════════════════════
    #  COMMAND POLLING
    # ══════════════════════════════════════════════════════════

    async def poll_commands(self):
        """
        Проверяет входящие команды.
        Вызывается из main loop каждый скан.
        """
        if not self.enabled:
            return

        url = (
            f"{self.base_url}/getUpdates"
            f"?offset={self._last_update_id + 1}&timeout=1"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return
                    data = await resp.json()

            for update in data.get("result", []):
                update_id = update.get("update_id", 0)
                self._last_update_id = max(
                    self._last_update_id, update_id
                )

                msg = update.get("message", {})
                text = (msg.get("text") or "").strip()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                # проверяем что сообщение из нашего чата
                if chat_id != self.chat_id:
                    continue

                if text:
                    await self._dispatch(text)

        except Exception as e:
            logger.debug(f"TG poll error: {e}")

    async def _dispatch(self, text: str):
        """Роутинг команд."""

        # ── проверка подтверждения ────────────────────────────
        if self._pending_confirm:
            if text.lower() == "yes":
                cmd = self._pending_confirm
                self._pending_confirm = None
                await self._execute_handler(cmd)
                return
            else:
                self._pending_confirm = None
                await self.send("❌ Cancelled")
                return

        # ── парсим команду ────────────────────────────────────
        parts = text.split()
        cmd = parts[0].lower() if parts else ""

        # ── опасные команды требуют подтверждения ─────────────
        dangerous = {"/kill", "/reset", "/live"}
        if cmd in dangerous:
            self._pending_confirm = text
            await self.send(
                f"⚠️ Confirm '{text}'?\n"
                f"Reply 'yes' to confirm or anything else to cancel."
            )
            return

        # ── обычные команды ───────────────────────────────────
        if cmd == "/help":
            await self._cmd_help()
        elif cmd in self._handlers:
            await self._execute_handler(text)
        else:
            # попробуем без /
            if f"/{cmd}" in self._handlers:
                await self._execute_handler(f"/{text}")

    async def _execute_handler(self, text: str):
        """Выполняет зарегистрированный обработчик."""
        parts = text.split()
        cmd = parts[0].lower()

        handler = self._handlers.get(cmd)
        if not handler:
            await self.send(f"Unknown command: {cmd}")
            return

        try:
            # handler может принимать аргументы
            if len(parts) > 1:
                result = await handler(" ".join(parts[1:]))
            else:
                result = await handler()

            if result:
                await self.send(str(result))

        except Exception as e:
            logger.error(f"Command handler error: {cmd}: {e}")
            await self.send(f"❌ Error: {e}")

    async def _cmd_help(self):
        """Список доступных команд."""
        builtin = [
            "/help — this message",
        ]

        registered = []
        for cmd in sorted(self._handlers.keys()):
            registered.append(f"{cmd}")

        text = (
            "📋 Commands v2.0:\n\n"
            "Built-in:\n"
            + "\n".join(f"  {c}" for c in builtin)
            + "\n\nRegistered:\n"
            + "\n".join(f"  {c}" for c in registered)
            + "\n\n⚠️ Dangerous commands (/kill, /reset, /live) "
            "require confirmation."
        )
        await self.send(text)


# ══════════════════════════════════════════════════════════════
#  COMMAND BUILDERS — регистрируются из main.py
# ══════════════════════════════════════════════════════════════

def build_commands(
    tg: TelegramNotifier,
    order_manager=None,
    risk_engine=None,
    scanner=None,
    polymarket_client=None,
    config=None,
):
    """
    Регистрирует все команды бота.
    Вызывается из main.py после инициализации всех компонентов.
    """

    # ── /stats ────────────────────────────────────────────────
    async def cmd_stats(args: str = ""):
        if not order_manager:
            return "Order manager not available"
        s = order_manager.get_stats()
        return (
            f"📊 Trading Stats\n\n"
            f"Total PnL: ${s['total_pnl']:+.4f}\n"
            f"Total fees: ${s['total_fees']:.4f}\n"
            f"Wins: {s['wins']} | Losses: {s['losses']}\n"
            f"Failed: {s['failed']}\n"
            f"Unwound: {s['unwound']}\n"
            f"Active: {s['active_positions']}\n"
            f"Closed: {s['closed_positions']}"
        )
    tg.register_handler("/stats", cmd_stats)

    # ── /balance ──────────────────────────────────────────────
    async def cmd_balance(args: str = ""):
        if not polymarket_client:
            return "Client not available"
        bal = polymarket_client.get_balance()
        if bal is not None:
            return f"💰 USDC Balance: ${bal:.2f}"
        return "❌ Failed to get balance"
    tg.register_handler("/balance", cmd_balance)

    # ── /positions ────────────────────────────────────────────
    async def cmd_positions(args: str = ""):
        if not order_manager:
            return "Order manager not available"
        active = order_manager.active_positions
        if not active:
            return "📭 No active positions"
        lines = [f"📊 Active Positions ({len(active)}):"]
        for pid, pos in active.items():
            lines.append(
                f"\n  {pid}:\n"
                f"    {pos.event_title[:40]}\n"
                f"    Status: {pos.status.value}\n"
                f"    Cost: ${pos.actual_cost:.4f}\n"
                f"    Age: {pos.age_seconds:.0f}s"
            )
        return "\n".join(lines)
    tg.register_handler("/positions", cmd_positions)

    # ── /risk ─────────────────────────────────────────────────
    async def cmd_risk(args: str = ""):
        if not risk_engine:
            return "Risk engine not available"
        snap = risk_engine.get_snapshot()
        can = risk_engine.can_trade()
        status = "🟢 ACTIVE" if can else "🔴 BLOCKED"
        return (
            f"🛡 Risk Engine: {status}\n\n"
            f"Daily PnL: ${snap['daily_pnl']:+.4f}\n"
            f"Drawdown: {snap['drawdown_pct']:.1f}%\n"
            f"Loss streak: {snap['consecutive_losses']}\n"
            f"Trades today: {snap['trades_today']}\n"
            f"Unwind rate: {snap['unwind_rate']:.0%}\n"
            f"Cooldown until: {snap['cooldown_until'] or 'none'}"
        )
    tg.register_handler("/risk", cmd_risk)

    # ── /config ───────────────────────────────────────────────
    async def cmd_config(args: str = ""):
        cfg = config
        if not cfg:
            return "Config not available"
        return (
            f"⚙️ Config\n\n"
            f"Mode: {'📝 PAPER' if cfg.PAPER_TRADING else '💰 LIVE'}\n"
            f"Min edge: {cfg.MIN_NET_EDGE_PCT}%\n"
            f"Max edge: {cfg.MAX_EDGE_PCT}%\n"
            f"Min volume: ${cfg.MIN_VOLUME}\n"
            f"Max position: ${cfg.MAX_POSITION_USD}\n"
            f"Max concurrent: {cfg.MAX_CONCURRENT}\n"
            f"Max daily loss: ${cfg.MAX_DAILY_LOSS_USD}\n"
            f"Scan interval: {cfg.SCAN_INTERVAL}s\n"
            f"Order TTL: {cfg.ORDER_TTL_SEC}s"
        )
    tg.register_handler("/config", cmd_config)

    # ── /scanner ──────────────────────────────────────────────
    async def cmd_scanner(args: str = ""):
        if not scanner:
            return "Scanner not available"
        s = scanner.get_stats()
        return (
            f"🔍 Scanner Stats\n\n"
            f"Scans: {s['scans']}\n"
            f"Events checked: {s['events_checked']}\n"
            f"Candidates: {s['candidates_found']}\n"
            f"Books fetched: {s['books_fetched']}\n"
            f"Valid forks: {s['valid_forks']}\n\n"
            f"Rejections:\n"
            + "\n".join(
                f"  {k}: {v}"
                for k, v in s['rejected'].items()
                if v > 0
            )
        )
    tg.register_handler("/scanner", cmd_scanner)

    # ── /kill — аварийная отмена ВСЕХ ордеров ─────────────────
    async def cmd_kill(args: str = ""):
        results = []

        # отменяем через order manager
        if order_manager:
            for pid, pos in list(
                order_manager.active_positions.items()
            ):
                if pos.leg_a and pos.leg_a.order_id:
                    polymarket_client.cancel_order(
                        pos.leg_a.order_id
                    )
                if pos.leg_b and pos.leg_b.order_id:
                    polymarket_client.cancel_order(
                        pos.leg_b.order_id
                    )
                results.append(f"Cancelled position {pid}")

        # на всякий случай cancel_all через клиент
        if polymarket_client:
            polymarket_client.cancel_all_orders()
            results.append("cancel_all sent to CLOB")

        if risk_engine:
            risk_engine.force_stop("manual /kill command")

        return (
            "🔴 EMERGENCY KILL EXECUTED\n\n"
            + "\n".join(results)
            + "\n\nAll trading stopped."
        )
    tg.register_handler("/kill", cmd_kill)

    # ── /pause — приостановить торговлю ───────────────────────
    async def cmd_pause(args: str = ""):
        if risk_engine:
            risk_engine.force_stop("manual /pause")
            return "⏸ Trading paused. Use /resume to continue."
        return "Risk engine not available"
    tg.register_handler("/pause", cmd_pause)

    # ── /resume — возобновить торговлю ────────────────────────
    async def cmd_resume(args: str = ""):
        if risk_engine:
            risk_engine.resume()
            return "▶️ Trading resumed"
        return "Risk engine not available"
    tg.register_handler("/resume", cmd_resume)

    # ── /live — переключиться на live (ОПАСНО) ────────────────
    async def cmd_live(args: str = ""):
        if not config:
            return "Config not available"

        if not config.POLY_PRIVATE_KEY:
            return "❌ Cannot go live: POLY_PRIVATE_KEY not set"

        errors = config.validate()
        if errors:
            return "❌ Config errors:\n" + "\n".join(errors)

        config.PAPER_TRADING = False
        return (
            "🔴 LIVE TRADING ENABLED\n\n"
            "Bot will now place REAL orders.\n"
            "Use /pause or /kill to stop."
        )
    tg.register_handler("/live", cmd_live)

    # ── /paper — вернуться в paper mode ───────────────────────
    async def cmd_paper(args: str = ""):
        if config:
            config.PAPER_TRADING = True
            return "📝 Paper trading mode enabled"
        return "Config not available"
    tg.register_handler("/paper", cmd_paper)

    logger.info(
        f"Registered {len(tg._handlers)} Telegram commands"
    )
