import json
import logging
from difflib import SequenceMatcher
from config import CFG
from matching.normalizer import normalize

logger = logging.getLogger("arb_scanner.matcher")


class EventMatcher:
    def __init__(self):
        self.manual_map = {}
        self._load_manual()

    def _load_manual(self):
        try:
            with open(CFG.MANUAL_MAPPINGS_FILE) as f:
                raw = json.load(f)
            self.manual_map = {normalize(k): normalize(v) for k, v in raw.items()}
            logger.info(f"Loaded {len(self.manual_map)} manual mappings")
        except FileNotFoundError:
            logger.info("No manual mappings file")
        except Exception as e:
            logger.error(f"Manual mappings error: {e}")

    def find_match(self, poly_question, kalshi_markets, threshold=CFG.SIMILARITY_THRESHOLD):
        norm_poly = normalize(poly_question)

        if norm_poly in self.manual_map:
            target = self.manual_map[norm_poly]
            for k_key, k_mkt in kalshi_markets.items():
                if normalize(k_mkt.question) == target:
                    return k_key, 1.0, "manual"

        best_key = None
        best_score = 0.0
        poly_tokens = set(norm_poly.split())

        if not poly_tokens:
            return None, 0.0, "none"

        for k_key, k_mkt in kalshi_markets.items():
            norm_k = normalize(k_mkt.question)
            k_tokens = set(norm_k.split())
            if not k_tokens:
                continue

            intersection = poly_tokens & k_tokens
            union = poly_tokens | k_tokens
            jaccard = len(intersection) / len(union)
            seq = SequenceMatcher(None, norm_poly, norm_k).ratio()
            combined = jaccard * 0.6 + seq * 0.4

            if combined > best_score:
                best_score = combined
                best_key = k_key

        min_tokens = min(len(poly_tokens), 3)
        adj = threshold - (0.05 * max(0, 5 - min_tokens))
        adj = max(adj, 0.50)

        if best_score >= adj and best_key is not None:
            return best_key, best_score, "fuzzy"

        return None, 0.0, "none"
