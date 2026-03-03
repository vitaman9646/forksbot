# config/watchdog.py
"""
ConfigWatchdog — из твоего кода с мелкими доработками.
Добавлено: логирование маскирует секреты.
"""

import asyncio
import logging
from threading import Lock

from config.settings import Settings, RELOADABLE
from config.dotenv_loader import DotEnvLoader

logger = logging.getLogger("arb_scanner.config")


class ConfigWatchdog:
    """Перечитывает .env каждые 30 секунд."""

    def __init__(self, config: Settings, loader: DotEnvLoader):
        self.config = config
        self.loader = loader
        self._last_mtime = loader.get_mtime()
        self._lock = Lock()
        self._reload_count = 0

    async def watch(self, interval: float = 30.0):
        logger.info("ConfigWatchdog started")
        while True:
            try:
                await asyncio.sleep(interval)
                mtime = self.loader.get_mtime()
                if mtime > self._last_mtime:
                    self._last_mtime = mtime
                    changes = self._reload()
                    if changes:
                        logger.info(
                            f"Config reloaded #{self._reload_count}:\n"
                            + "\n".join(f"  {c}" for c in changes)
                        )
            except asyncio.CancelledError:
                logger.info("ConfigWatchdog stopped")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    def _reload(self) -> list:
        new_vars = self.loader.read_raw()
        changed = []

        with self._lock:
            for key, val_str in new_vars.items():
                if key not in RELOADABLE:
                    continue

                attr = getattr(self.config, key, None)
                if attr is None:
                    continue

                try:
                    t = type(attr)
                    if t == float:
                        new_val = float(val_str)
                    elif t == int:
                        new_val = int(val_str)
                    elif t == bool:
                        new_val = val_str.lower() in (
                            "true", "1", "yes"
                        )
                    else:
                        new_val = val_str

                    if attr != new_val:
                        setattr(self.config, key, new_val)
                        # маскируем секреты
                        if self.loader.is_secret(key):
                            changed.append(f"{key}: ***")
                        else:
                            changed.append(
                                f"{key}: {attr} → {new_val}"
                            )
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Bad reload value {key}={val_str}: {e}"
                    )

        if changed:
            self._reload_count += 1

        return changed

    @property
    def reload_count(self) -> int:
        return self._reload_count
