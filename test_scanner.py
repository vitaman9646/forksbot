# test_scanner.py
"""
Тестовый скрипт — сканирование без торговли.
Использует общую RateLimitedSession.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("arb_scanner")


async def main():
    from config.settings import CFG
    from core.scanner import ForkScanner
    from utils.rate_limiter import RateLimitedSession

    # валидация конфига
    errors = CFG.validate()
    for e in errors:
        # в paper mode не все ошибки критичны
        if "PRIVATE_KEY" not in e and "FUNDER" not in e:
            logger.error(f"Config error: {e}")

    CFG.print_config()

    # общая сессия с rate limiting
    http = RateLimitedSession(rps=3.0, burst=8, max_retries=2)

    scanner = ForkScanner(
        http_session=http,
        config=CFG,
    )

    print("\n" + "=" * 60)
    print("  FORK SCANNER TEST — no trading")
    print("  Watching real orderbooks on Polymarket")
    print("=" * 60 + "\n")

    scan_n = 0
    total_valid = 0
    all_forks = []

    try:
        while True:
            scan_n += 1
            print(f"\n{'─' * 50} Scan #{scan_n} "
                  f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]")

            forks = await scanner.find_forks(
                entry_size=CFG.MAX_POSITION_USD,
                min_edge=CFG.MIN_NET_EDGE_PCT,
                max_edge=CFG.MAX_EDGE_PCT,
                min_volume=CFG.MIN_VOLUME,
            )

            valid = [f for f in forks if f.is_valid]
            rejected = [f for f in forks if not f.is_valid]
            total_valid += len(valid)

            # показываем valid
            for f in valid:
                print(f"\n{'🎯' * 25}")
                print(f.summary())
                print(f"{'🎯' * 25}")

                # сохраняем
                all_forks.append(f.to_dict())

            # показываем причины отказа (кратко)
            if rejected:
                reasons = {}
                for f in rejected:
                    r = f.reject_reason.split(":")[0]
                    reasons[r] = reasons.get(r, 0) + 1
                reason_str = ", ".join(
                    f"{k}={v}" for k, v in reasons.items()
                )
                print(f"  Rejected ({len(rejected)}): {reason_str}")

            # scanner stats
            stats = scanner.get_stats()
            api_stats = http.stats()
            print(
                f"\n  Scanner: "
                f"candidates={stats['candidates_found']} "
                f"books={stats['books_fetched']} "
                f"valid_total={total_valid}"
            )
            print(
                f"  API: "
                f"reqs={api_stats['requests']} "
                f"429s={api_stats['429s']} "
                f"5xx={api_stats['5xxs']} "
                f"rps={api_stats['rps_current']}"
            )

            # каждые 10 сканов — summary
            if scan_n % 10 == 0:
                print(f"\n{'═' * 50}")
                print(f"  SUMMARY after {scan_n} scans")
                print(f"  Valid forks found: {total_valid}")
                print(f"  Events checked: {stats['events_checked']}")
                print(f"  Rejections: {stats['rejected']}")
                print(f"{'═' * 50}")

            # сохраняем в файл
            if all_forks:
                with open("data/test_forks.jsonl", "w") as f:
                    for fork in all_forks:
                        f.write(json.dumps(fork) + "\n")

            await asyncio.sleep(CFG.SCAN_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        await http.close()

        print(f"\n{'═' * 60}")
        print(f"  FINAL RESULTS")
        print(f"  Scans: {scan_n}")
        print(f"  Valid forks: {total_valid}")
        print(f"  API requests: {http.stats()['requests']}")
        print(f"{'═' * 60}")

        if all_forks:
            print(f"\n  Saved {len(all_forks)} forks to "
                  f"data/test_forks.jsonl")


if __name__ == "__main__":
    # создаём папки
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    asyncio.run(main())
