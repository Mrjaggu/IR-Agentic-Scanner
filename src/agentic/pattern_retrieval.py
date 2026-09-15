"""
pattern_retrieval.py — the retrieval step that runs WHILE a question is being
framed: given (analyst, topic), pull the relevant mined cognitive pattern(s)
from scripts/extract_analyst_patterns.py's output and hand them to
question_framer as extra grounding, instead of just the single most recent
question (_find_precedent's old job).

Mirrors the existing overall_layer.py vs analyst_layer.py split:
  - retrieve_overall(...)      cross-analyst signal: what reasoning pattern
                                tends to fire industry-wide when THIS kind of
                                number moves, regardless of who's asking.
                                A weak prior, used only as a fallback when the
                                target analyst has no pattern of their own for
                                this topic -- the same role composite_scores()
                                plays before reweight_for_analyst narrows the
                                topic list down to one analyst.
  - retrieve_for_analyst(...)  THIS analyst's own mined pattern for this
                                topic. The strong signal, when present.

Matching is deterministic keyword overlap against src.graphs.compiler.TOPICS
(the same keyword list _narration_for_topic already uses in
question_framer.py) -- no LLM call, no invented relevance score. A pattern
with zero keyword overlap against this topic's vocabulary is never returned,
so "the framer used a mined pattern" always means the pattern's own trigger
text is actually about this topic, not a coincidence of ranking.

Degrades honestly: if this bank has no mined-patterns file (Kotak/IndusInd
today -- see scripts/extract_analyst_patterns.py's docstring on why), both
functions return None/[] rather than raising, same discipline as
src.search.embeddings.is_available().
"""

import json
import os
import re

from src.config.settings import BASE_DIR
from src.graphs.compiler import TOPICS as TOPIC_KEYWORDS

_cache: dict[str, dict] = {}


def _patterns_path(bank_id: str) -> str:
    return os.path.join(BASE_DIR, "data", "analysis", f"{bank_id}_analyst_patterns.json")


def _load(bank_id: str) -> dict:
    if bank_id in _cache:
        return _cache[bank_id]
    path = _patterns_path(bank_id)
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path))
        except Exception:
            data = {}
    _cache[bank_id] = data
    return data


def is_available(bank_id: str = "axis") -> bool:
    return bool(_load(bank_id))


def _topic_keywords(topic: str) -> list[str]:
    kws = [k.lower() for k in TOPIC_KEYWORDS.get(topic, [])]
    kws += [w for w in re.split(r"[^a-z]+", topic.lower()) if len(w) > 3]
    return kws


def _score(pattern: dict, kws: list[str]) -> int:
    text = f"{pattern.get('trigger', '')} {pattern.get('reasoning_pattern', '')}".lower()
    return sum(1 for k in kws if k in text)


def retrieve_for_analyst(analyst: str, topic: str, bank_id: str = "axis") -> dict | None:
    """This analyst's own mined pattern whose trigger text is actually about
    this topic (keyword overlap > 0). Highest-scoring pattern wins; None if
    this analyst has no mined patterns at all, or none of them are about
    this topic."""
    entry = _load(bank_id).get(analyst)
    if not entry:
        return None
    kws = _topic_keywords(topic)
    scored = [(_score(p, kws), p) for p in entry.get("patterns", [])]
    scored = [sp for sp in scored if sp[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda sp: -sp[0])
    return scored[0][1]


def retrieve_overall(topic: str, exclude_analyst: str | None = None,
                     bank_id: str = "axis", top_k: int = 2) -> list[dict]:
    """Cross-analyst fallback: the best-matching mined patterns for this
    topic, pooled across every OTHER analyst in this bank's history. Used
    only when the target analyst has no pattern of their own for this topic
    -- a weak prior ("this is the kind of reasoning this scenario tends to
    provoke, industry-wide"), never presented as this specific analyst's own
    behavior."""
    kws = _topic_keywords(topic)
    scored = []
    for name, entry in _load(bank_id).items():
        if name == exclude_analyst:
            continue
        for p in entry.get("patterns", []):
            s = _score(p, kws)
            if s > 0:
                scored.append((s, name, p))
    scored.sort(key=lambda t: -t[0])
    return [{"analyst": name, **p} for _, name, p in scored[:top_k]]
