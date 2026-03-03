# test_scanner.py
"""
Тестовый скрипт — только сканирование.
Никаких ордеров, никакого кошелька.
Просто смотрим: есть ли реальные вилки?
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("arb_scanner")


class MockClobClient:
    """Заглушка — стаканы берём напрямую из HTTP."""
    pass


class MockConfig:
    pass


async def main():
    from core.scanner import ForkScanner

    scanner = ForkScanner(
        clob_client=MockClobClient(),
        config=MockConfig(),
    )

    print("=" * 60)
    print("  SCANNER TEST — no trading, just scanning")
    print("=" * 60)

    scan_n = 0
    total_valid = 0

    while True:
        scan_n += 1
        print(f"\n{'─' * 40} Scan #{scan_n}")

        forks = await scanner.find_forks(
            entry_size=5.0,       # $5 входа
            min_edge=0.3,         # 0.3% минимум
            max_edge=15.0,        # >15% = фейк
            min_volume=500,       # $500 объём
        )

        valid = [f for f in forks if f.is_valid]
        total_valid += len(valid)

        for f in valid:
            print(f"\n{'🎯' * 20}")
            print(f.summary())
            print(f"{'🎯' * 20}\n")

        stats = scanner.get_stats()
        print(
            f"\nStats: "
            f"events={stats['events_checked']} "
            f"candidates={stats['candidates_found']} "
            f"books={stats['books_fetched']} "
            f"valid={total_valid}"
        )
        print(f"Rejected: {stats['rejected']}")

        # ждём 30 секунд
        await asyncio.sleep(30)


if __name__ == "__main__":
    asyncio.run(main())
