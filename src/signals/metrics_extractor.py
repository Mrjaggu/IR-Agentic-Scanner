"""
metrics_extractor.py — deterministic (regex-based, no LLM) structured metrics
extraction from earnings-call narration.

Motivation: the engine's current narration signal (q4_narr_topics_count in
engine.py) is a raw count of keyword-tagged sentences per topic — it has no
sense of MAGNITUDE. A passing one-sentence mention and a 30bp guidance miss
score identically as long as both get one sentence. Axis's CFO narration is
heavily templated quarter to quarter ("NIM at 3.99%, declined 6 bps QOQ",
"GNPA at 1.44%, declined 12 bps YOY") — verified across 4 quarters this
session — so a regex extractor can pull structured (metric, value, delta,
direction, period) tuples reliably, without any LLM call.

Usage:
    python3 run.py metrics

Output: data/db/metrics_timeseries.json
    {"q1fy27": {"NIM": {"value": 3.5, "value_unit": "%",
                          "delta": 16.0, "delta_unit": "bps",
                          "direction": "decline", "period": "QOQ"}}}
"""

import json
import os
import re

from src.config.settings import GRAPH_PATH, METRICS_TIMESERIES_PATH

# metric label -> regex alternation of how it appears in narration
METRIC_PATTERNS = {
    "NIM": r"\bNIMs?\b",
    "GNPA": r"\bGNPA\b",
    "NNPA": r"\bNNPA\b",
    "PCR": r"\bPCR%?\b",
    "PAT": r"\bPAT\b",
    "ROA": r"\bROA%?\b",
    "ROE": r"\bROE%?\b",
    "CET1": r"\bCET\s*-?\s*1\b",
    "Cost_to_Income": r"\bCost\s*to\s*income\b",
    "Cost_to_Assets": r"\bCost\s*to\s*assets\b",
    "Net_Credit_Cost": r"\bNet\s*credit\s*cost\b",
}

# Subsidiaries report their own PAT/ROE/etc. -- skip these sentences when
# looking for BANK-LEVEL figures, otherwise "first occurrence" can grab a
# subsidiary's number instead of the headline one (found via spot-check:
# q4fy26 PAT was extracted as Axis Finance's 806cr, not the bank's 7,071cr).
SUBSIDIARY_MARKERS = re.compile(
    r"\bAxis\s+(Finance|AMC|Capital|Securities)\b|\bMax\s+Life\b|\bdomestic\s+subsidiar",
    re.IGNORECASE,
)

# "at"/"was"/"was at"/"to" + a number, optionally "Rs."/"₹" + crores, or a %.
# Search window is deliberately short (see _extract_from_sentence) so this can't
# skip past the intended value into a later, unrelated clause.
_VALUE_RE = re.compile(
    r"(?:at|was(?:\s+at)?|to)\s+(?:Rs\.?\s*|₹\s*)?(?P<value>[\d,]+\.?\d*)\s*(?P<unit>%|crores?|cr\b|bps)?",
    re.IGNORECASE,
)
_VALUE_SEARCH_WINDOW = 40
_DELTA_SEARCH_WINDOW = 90

_DIRECTION_WORDS = {
    "declin": "decline", "improv": "improve", "increas": "increase",
    "grew": "increase", "grow": "increase", "up": "increase",
    "down": "decrease", "reduc": "decrease", "flat": "flat",
}

_PERIOD_ALT = r"QOQ|YOY|Q-o-Q|Y-o-Y|sequentially|quarter[\s-]on[\s-]quarter|year[\s-]on[\s-]year"
_DIR_ALT = r"declin\w*|improv\w*|increas\w*|grew|grow\w*|up|down|reduc\w*|broadly\s+flat|flat"

# "declined 6 bps QOQ" (direction ... delta ... period)
_DELTA_RE = re.compile(
    rf"(?P<dirword>{_DIR_ALT})"
    rf"[^.]{{0,20}}?(?P<delta>[\d,]+\.?\d*)\s*(?P<delta_unit>bps|basis points|%)"
    rf"[^.]{{0,15}}?(?P<period>{_PERIOD_ALT})",
    re.IGNORECASE,
)
# "QoQ growth of 9%" (period ... direction ... delta) -- the reverse ordering,
# found via spot-check on q4fy26's PAT line ("PAT at Rs 7,071 cr, QoQ growth of 9%")
_DELTA_RE_REV = re.compile(
    rf"(?P<period>{_PERIOD_ALT})"
    rf"[^.]{{0,10}}?(?P<dirword>{_DIR_ALT})[^.]{{0,10}}?(?:of\s+)?"
    rf"(?P<delta>[\d,]+\.?\d*)\s*(?P<delta_unit>bps|basis points|%)",
    re.IGNORECASE,
)


def _direction_from_word(word: str) -> str:
    w = word.lower()
    for prefix, direction in _DIRECTION_WORDS.items():
        if w.startswith(prefix):
            return direction
    return "flat" if "flat" in w else "unknown"


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    u = unit.lower()
    if u == "bps":
        return "bps"
    if u.startswith("cr"):
        return "crore"
    return u


def _normalize_period(p: str) -> str:
    p = p.lower().replace("-", "").replace(" ", "")
    if p in ("qoq", "quarteronquarter", "sequentially"):
        return "QOQ"
    if p in ("yoy", "yearonyear"):
        return "YOY"
    return p.upper()


def _split_sentences(text: str) -> list[str]:
    # Narration is one long run-on string per segment. Split on ". " AND on the
    # bullet/list markers used for subsidiary breakdowns (o, ▪, •) -- without
    # this, "Q1FY27 PAT grew 29%... o Strong asset quality... ▪ Axis AMC: PAT..."
    # stays one giant sentence and a metric mention bleeds into unrelated bullets.
    text = re.sub(r"\s+[▪•]\s+", ". ", text)
    text = re.sub(r"(?<=[a-z%\d])\s+o\s+(?=[A-Z])", ". ", text)
    return re.split(r"(?<=[a-z%\d])\.\s+(?=[A-Z])", text)


def _extract_all_from_sentence(sentence: str) -> list[dict]:
    """A merged/run-on sentence can legitimately contain several metrics back
    to back (sentence-splitting on periods/bullets is heuristic, not perfect)
    -- extract every metric found, not just the first in METRIC_PATTERNS'
    definition order, otherwise later metrics in the same run-on silently
    vanish (found via spot-check: Cost_to_Assets dropped whenever GNPA
    appeared later in the same merged sentence)."""
    records = []
    for metric, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, sentence, re.IGNORECASE)
        if not m:
            continue
        rest = sentence[m.end(): m.end() + _DELTA_SEARCH_WINDOW]
        vm = _VALUE_RE.search(rest[:_VALUE_SEARCH_WINDOW])
        if not vm:
            continue
        dm = _DELTA_RE.search(rest) or _DELTA_RE_REV.search(rest)
        if not dm:
            continue
        try:
            value = float(vm.group("value").replace(",", ""))
            delta = float(dm.group("delta").replace(",", ""))
        except ValueError:
            continue
        records.append({
            "metric": metric,
            "value": value,
            "value_unit": _normalize_unit(vm.group("unit")),
            "delta": delta,
            "delta_unit": "bps" if "bp" in dm.group("delta_unit").lower() else "%",
            "direction": _direction_from_word(dm.group("dirword")),
            "period": _normalize_period(dm.group("period")),
        })
    return records


def extract_quarter_metrics(narration_text: str) -> dict[str, dict]:
    """Returns {metric: {value, value_unit, delta, delta_unit, direction, period}}.
    Keeps the FIRST match per metric per quarter (the bank-level summary-line
    mention, which appears before the detailed walk-through in every transcript
    observed) -- skipping subsidiary-context sentences, which report their own
    PAT/ROE and would otherwise be mistaken for the bank-level figure."""
    result = {}
    for sentence in _split_sentences(narration_text):
        if SUBSIDIARY_MARKERS.search(sentence):
            continue
        for rec in _extract_all_from_sentence(sentence):
            if rec["metric"] not in result:
                result[rec["metric"]] = {k: v for k, v in rec.items() if k != "metric"}
    return result


def build_metrics_timeseries(path: str = METRICS_TIMESERIES_PATH, graph_path: str = None) -> dict:
    graph_path = graph_path or GRAPH_PATH
    with open(graph_path) as f:
        graph = json.load(f)

    by_quarter: dict[str, list[str]] = {}
    for n in graph["nodes"]:
        if n["type"] == "NarrationSegment":
            by_quarter.setdefault(n["properties"]["quarter"], []).append(n["properties"]["text"])

    result = {}
    for quarter, segs in by_quarter.items():
        full_text = " ".join(segs)
        result[quarter] = extract_quarter_metrics(full_text)

    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Extracted metrics for {len(result)} quarters -> {path}")
    for q in sorted(result, key=lambda x: (x[-2:], x[:2])):
        found = list(result[q].keys())
        print(f"  {q}: {len(found)}/{len(METRIC_PATTERNS)} metrics found — {found}")
    return result


# metric -> topic it signals. Metrics with no natural topic mapping (e.g. a
# generic operating-profit line) are simply omitted -- topics with no mapped
# metric fall back to the existing keyword-count narration signal in engine.py.
METRIC_TOPIC_MAP = {
    "NIM": "NIM & Yields",
    "GNPA": "Slippages & Asset Quality",
    "NNPA": "Slippages & Asset Quality",
    "PCR": "Slippages & Asset Quality",
    "PAT": "Profitability & Returns",
    "ROA": "Profitability & Returns",
    "ROE": "Profitability & Returns",
    "CET1": "Capital Adequacy",
    "Cost_to_Income": "Opex & Cost-to-Income",
    "Cost_to_Assets": "Opex & Cost-to-Income",
    "Net_Credit_Cost": "Credit Cost & Provisions",
}


def compute_topic_anomaly_scores(val_quarter: str, quarter_order: list[str],
                                 timeseries_path: str = METRICS_TIMESERIES_PATH) -> dict[str, float]:
    """For each topic with a mapped metric, score how unusual val_quarter's
    |delta| is relative to that metric's own historical distribution (quarters
    strictly before val_quarter -- same train/VAL_QUARTER split discipline as
    the rest of the engine, so this can't leak the answer into the signal used
    to predict it). Score = percentile rank in [0, 1]; 1.0 = the biggest move
    that metric has ever had. Topic score = max across its mapped metrics.

    2026-09: returns {} (no anomaly signal, not a crash) if timeseries_path
    doesn't exist -- a newly-registered bank (see src.config.banks) that
    hasn't had `python run.py metrics --bank <id>` run for it yet still
    needs build_initial_state() to succeed, just with this one signal
    absent, same discipline as apply_ask_patterns()'s missing-file handling."""
    if not os.path.exists(timeseries_path):
        return {}
    with open(timeseries_path) as f:
        timeseries = json.load(f)

    vi = quarter_order.index(val_quarter)
    prior_quarters = set(quarter_order[:vi])

    metric_history: dict[str, list[float]] = {}
    for q, metrics in timeseries.items():
        if q not in prior_quarters:
            continue
        for metric, rec in metrics.items():
            metric_history.setdefault(metric, []).append(abs(rec["delta"]))

    val_metrics = timeseries.get(val_quarter, {})
    topic_scores: dict[str, float] = {}
    for metric, rec in val_metrics.items():
        topic = METRIC_TOPIC_MAP.get(metric)
        if not topic:
            continue
        history = metric_history.get(metric, [])
        if not history:
            score = 0.5  # no history to compare against -- moderate, not extreme
        else:
            cur = abs(rec["delta"])
            score = sum(1 for h in history if h <= cur) / len(history)
        topic_scores[topic] = max(topic_scores.get(topic, 0.0), score)

    return topic_scores


if __name__ == "__main__":
    build_metrics_timeseries()
