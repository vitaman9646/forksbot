import re

NORMALIZE_MAP = {
    "bitcoin": "btc", "ethereum": "eth", "solana": "sol",
    "donald trump": "trump", "joe biden": "biden",
    "federal reserve": "fed", "united states": "us",
    "interest rate": "rate", "rate cut": "cut",
    "above": ">", "below": "<", "exceed": ">",
    "over": ">", "under": "<", "at least": ">=",
    "more than": ">", "less than": "<",
    "thousand": "k", ",000": "k",
    "million": "m", "billion": "b",
    "january": "jan", "february": "feb", "march": "mar",
    "april": "apr", "june": "jun", "july": "jul",
    "august": "aug", "september": "sep", "october": "oct",
    "november": "nov", "december": "dec",
    "super bowl": "superbowl", "nba finals": "nbafinals",
    "world cup": "worldcup",
}

STOP_WORDS = frozenset({
    "will", "the", "a", "an", "be", "is", "are", "to", "of",
    "in", "on", "at", "for", "this", "that", "it", "by", "or",
    "and", "do", "does", "did", "has", "have", "what", "when",
    "how", "much", "many", "next", "new",
})

_cache = {}


def normalize(text):
    if text in _cache:
        return _cache[text]
    result = text.lower().strip()
    result = re.sub(r"[?!.,;:'\"\-()\[\]{}]", " ", result)
    for old, new in NORMALIZE_MAP.items():
        result = result.replace(old, new)
    result = re.sub(r"\s+", " ", result).strip()
    tokens = [t for t in result.split() if t not in STOP_WORDS and len(t) > 1]
    normalized = " ".join(sorted(tokens))
    _cache[text] = normalized
    return normalized
