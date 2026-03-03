# config/dotenv_loader.py
"""
DotEnvLoader — без изменений из твоего кода.
Проверенная реализация, работает хорошо.
"""

import os
import re
import logging
from pathlib import Path

logger = logging.getLogger("arb_scanner.config")


class DotEnvLoader:
    SECRET_PATTERNS = {"KEY", "SECRET", "TOKEN", "PASSWORD", "PASSPHRASE"}

    def __init__(self, path: str = ".env"):
        self.path = Path(path)

    def load(self, override: bool = False):
        for k, v in self._parse().items():
            if override or k not in os.environ:
                os.environ[k] = v

    def read_raw(self) -> dict:
        return self._parse()

    def get_mtime(self) -> float:
        return self.path.stat().st_mtime if self.path.exists() else 0.0

    def is_secret(self, key: str) -> bool:
        return any(p in key.upper() for p in self.SECRET_PATTERNS)

    def _parse(self) -> dict:
        result = {}
        if not self.path.exists():
            return result
        try:
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if val and val[0] not in ('"', "'"):
                    val = re.sub(r"\s+#.*$", "", val).strip()
                if (
                    len(val) >= 2
                    and val[0] in ('"', "'")
                    and val[-1] == val[0]
                ):
                    val = val[1:-1]
                if key:
                    result[key] = val
        except Exception as e:
            logger.error(f"DotEnv parse error: {e}")
        return result
