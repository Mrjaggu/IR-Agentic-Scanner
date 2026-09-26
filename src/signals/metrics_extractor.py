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

2026-09 (Kotak/IndusInd tuning): Axis's "value, then an explicit delta
number" template is NOT how Kotak or IndusInd narrate. Read closely across
6 real transcripts (q1fy27/q3fy26/q4fy26 for both banks), their dominant
shape is a TWO-VALUE COMPARISON instead of a stated delta: "Gross NPA at
1.30% versus 1.39%", "NIM for the quarter is 4.67% as against 4.54% for
Q3", "reduced from 93 bps in Q1 to 46 bps in Q3", "PAT ... at INR 240 crore
against INR 250 crore in Q3". The delta has to be COMPUTED from the two
values, not read as a stated number — see _comparison_delta() below. They
also use value-introducing words Axis's phrasing never needed ("is", "of",
"stood", "remained") and spell some metrics out in full ("Gross NPA",
"Net NPA", "profit after tax") rather than the abbreviation Axis always
uses. Both extensions are additive (broader alternations, a new fallback
delta path tried only when the explicit-delta regex finds nothing) so
Axis's own extraction is unaffected -- verified by an unchanged-output
regression check after this change, not just assumed.

Usage:
    python3 run.py metrics --bank <bank_id>

Output: data/db/<bank_id>/metrics_timeseries.json
    {"q1fy27": {"NIM": {"value": 3.5, "value_unit": "%",
                          "delta": 16.0, "delta_unit": "bps",
                          "direction": "decline", "period": "QOQ"}}}
"""

import json
import os
import re

from src.config.banks import BANKS, DEFAULT_BANK
from src.config.settings import GRAPH_PATH, METRICS_TIMESERIES_PATH

# metric label -> regex alternation of how it appears in narration.
# Kotak/IndusInd sometimes spell GNPA/NNPA/PAT out in full instead of using
# the abbreviation Axis always uses (see module docstring) -- added as
# additional alternatives, so Axis's own abbreviation-only mentions are
# still matched exactly as before.
METRIC_PATTERNS = {
    "NIM": r"\bNIMs?\b|\bNet\s+Interest\s+Margins?\b",
    "GNPA": r"\bGNPA\b|\bGross\s+NPAs?\b",
    "NNPA": r"\bNNPA\b|\bNet\s+NPAs?\b",
    "PCR": r"\bPCR%?\b|\bprovision\s+coverage\s+ratio\b",
    "PAT": r"\bPAT\b|\bprofit\s+after\s+tax\b",
    "ROA": r"\bROAs?%?\b|\breturn\s+on\s+assets?\b",
    "ROE": r"\bROEs?%?\b|\breturn\s+on\s+equity\b",
    "CET1": r"\bCET\s*-?\s*1\b",
    "Cost_to_Income": r"\bCost[\s-]to[\s-]income\b",
    "Cost_to_Assets": r"\bCost[\s-]to[\s-]assets?\b",
    # "Net" is now optional -- Kotak narrates this as plain "Credit cost(s)"
    # far more often than "Net credit cost" (Axis's own phrasing, still
    # matched first since METRIC_PATTERNS order doesn't affect this).
    "Net_Credit_Cost": r"\b(?:Net\s+)?[Cc]redit\s+costs?\b",
}

# PASS 1 uses this byte-identical-to-pre-2026-09 dict instead of the
# broadened METRIC_PATTERNS above -- found via axis regression testing that
# the broadened NAME patterns are themselves a regression source, same
# mechanism as _VALUE_RE_BROAD (see its comment): a broadened alternative
# like "profit after tax" or "Net Interest Margin" can match in an earlier
# SENTENCE than the one containing the literal "PAT"/"NIM" Axis's own
# narration actually uses, so the "first match across sentences" that
# extract_quarter_metrics keeps silently shifts to the wrong sentence even
# though _VALUE_RE itself never changed for that match. Concrete confirmed
# cases this fixes: q4fy22 PAT (13,025cr -> wrongly 4,118cr via "profit
# after tax"), q4fy22 NNPA (0.73% -> wrongly 218/no-unit via "Net NPA"),
# q3fy23/q2fy23 NIM (4.26%/3.96% -> wrongly 4.0/3.96-no-unit via "Net
# Interest Margin"), q4fy24 Net_Credit_Cost (via the now-optional "Net").
# PASS 2 (broadened, see _extract_pass2) keeps using METRIC_PATTERNS above
# -- it only ever runs for metrics PASS 1 found nothing for at all, so it
# can reach a "profit after tax"-only mention without risk.
METRIC_PATTERNS_NARROW = {
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

# Sentences reporting a SUBSIDIARY's own PAT/ROE/etc. must be skipped when
# looking for BANK-LEVEL figures, otherwise "first occurrence" can grab a
# subsidiary's number instead of the headline one (found via spot-check:
# Axis q4fy26 PAT was extracted as Axis Finance's 806cr, not the bank's
# 7,071cr). This used to be a single Axis-hardcoded regex; that meant
# Kotak's OWN subsidiary sentences (Kotak Securities, Kotak Prime, Kotak
# AMC, Kotak Life, Kotak Mahindra Capital -- all named explicitly in its
# narration, see build_metrics_timeseries' bank-scoping) were never
# filtered at all. Now built per-bank from BANKS[bank_id].subsidiary_keywords
# (src/config/banks.py), the same registry dataset_compiler.py and
# peer_signal.py already use for bank-specific vocabulary, plus a generic
# catch-all every bank shares.
_GENERIC_SUBSIDIARY_MARKER = r"\bdomestic\s+subsidiar\w*|\bconsolidated\s+level\b"


def _subsidiary_markers(bank_id: str = DEFAULT_BANK) -> re.Pattern:
    bank = BANKS.get(bank_id)
    keywords = bank.subsidiary_keywords if bank else []
    parts = [re.escape(k) for k in keywords] + [_GENERIC_SUBSIDIARY_MARKER]
    return re.compile(r"\b(?:" + "|".join(parts) + r")", re.IGNORECASE)


# PASS 1 -- byte-for-byte the pre-2026-09 pattern ("at"/"was(?: at)?"/"to" +
# a number, optionally "Rs."/"₹" + crores, or a %). Always runs first over
# the whole quarter (see extract_quarter_metrics) and, for any metric it
# finds a value for, that result is FINAL -- this is Axis's own proven
# extraction, unchanged, so it can never be corrupted by anything PASS 2
# below adds.
_VALUE_RE_NARROW = re.compile(
    r"(?:at|was(?:\s+at)?|to)\s+(?:Rs\.?\s*|₹\s*)?(?P<value>[\d,]+\.?\d*)\s*(?P<unit>%|crores?|cr\b|bps)?",
    re.IGNORECASE,
)
# PASS 2 -- broadened 2026-09: a value introduced by "is"/"of"/"stood"/
# "remained" as well ("NIM ... is 4.67%", "PAT of INR 3,400 crores", "ROE
# ... stood 11.92%"), see module docstring. Regression-tested standalone
# (single-pass, replacing _VALUE_RE outright) and found to corrupt 8 of
# Axis's own 21 quarters -- a generic trigger like "is"/"of" can let an
# EARLIER, unrelated number in Axis's denser narration win the re.search
# match ahead of the correct, later one (concrete case: q4fy22 PAT flipped
# from the correct ~13,025cr to ~4,118cr). Only ever tried via
# extract_quarter_metrics' second pass, restricted to metrics PASS 1 (above)
# found nothing for anywhere in the quarter -- so it can add a value, never
# overwrite one.
_VALUE_RE_BROAD = re.compile(
    r"(?:at|is|was(?:\s+at)?|to|of|stood(?:\s+at)?|remained(?:\s+at)?)\s+"
    r"(?:Rs\.?\s*|₹\s*|INR\s*)?(?P<value>[\d,]+\.?\d*)\s*"
    r"(?P<unit>%|crores?|cr\b|bps|basis\s+points)?",
    re.IGNORECASE,
)
_VALUE_SEARCH_WINDOW = 40
_DELTA_SEARCH_WINDOW = 90
_COMPARISON_SEARCH_WINDOW = 70

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

# Fallback for when no EXPLICIT delta number is stated at all -- Kotak's and
# IndusInd's dominant shape (see module docstring): two absolute values,
# "current versus/vs/as against/compared to prior" or "from prior ... to
# current". Tried only after both _DELTA_RE and _DELTA_RE_REV fail to match,
# so a transcript that DOES state an explicit delta keeps using that (more
# direct, less inference) rather than this computed fallback.
_VS_RE = re.compile(
    r"(?:versus|vs\.?|as\s+against|against|compared\s+to)\s+"
    r"(?:Rs\.?\s*|₹\s*|INR\s*)?(?P<value2>[\d,]+\.?\d*)",
    re.IGNORECASE,
)
_FROM_RE = re.compile(
    r"\bfrom\s+(?:Rs\.?\s*|₹\s*|INR\s*)?(?P<value2>[\d,]+\.?\d*)",
    re.IGNORECASE,
)
_YOY_MARKERS = re.compile(r"last\s+year|y-?o-?y|year[\s-]on[\s-]year", re.IGNORECASE)


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
    if u == "bps" or u.startswith("basis"):
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


def _guess_period(comparison_text: str) -> str:
    """Best-effort period label for a COMPUTED (not stated) delta -- looks
    for an explicit YoY marker in the comparison clause itself ("versus
    4.96% ... last year", "from 93 bps in Q1 last year"); falls back to QOQ
    otherwise. QOQ is the more common comparison in the Kotak/IndusInd
    sentences this fallback actually fires on ("versus last quarter" /
    "versus Q3" beats a same-quarter-last-year callout in what was read for
    this module) -- an explicit guess, not a silent default, and cheaper to
    correct later than to leave every computed delta unlabelled."""
    return "YOY" if _YOY_MARKERS.search(comparison_text) else "QOQ"


def _comparison_delta(metric: str, value: float, unit: str | None,
                      rest_before: str, rest_after: str) -> dict | None:
    """The fallback used when no explicit delta number is stated (see this
    module's docstring and _VS_RE/_FROM_RE above): finds a second, prior
    value via a comparison clause and COMPUTES the delta, rather than
    reading one. `value` is the metric's own already-extracted (current
    quarter) value; `unit` is its already-normalized unit ("%", "bps",
    "crore", or None). rest_after is searched first (the far more common
    "current versus/vs/against/compared to prior" order); rest_before (the
    text between the metric mention and the value match) is tried only if
    that fails, for the reverse "from prior ... to current" order.

    Delta unit follows the VALUE's OWN unit, not the metric's identity --
    Net_Credit_Cost is usually stated directly in bps ("46 basis points",
    never "0.46%"), unlike NIM/GNPA/ROE/etc which are stated as a
    percentage ("4.67%") -- an earlier version of this function assumed
    every ratio metric was a percentage and multiplied a bps value by 100
    again, inflating credit-cost deltas 100x (46 vs 93 bps became a 4700bps
    "delta"). So: value already in "%" -> delta is a percentage-POINT
    difference, expressed in bps (x100), matching how Axis's own explicit
    ratio deltas are already expressed. Value already in "bps" -> the two
    values are already directly comparable, no further scaling. PAT (and
    anything in "crore") is the one absolute rupee metric tracked here --
    its delta is a percentage CHANGE instead, matching how Axis's own
    explicit PAT deltas are already expressed ("PAT ... grew 23% Y-o-Y")."""
    m = _VS_RE.search(rest_after[:_COMPARISON_SEARCH_WINDOW])
    comparison_text = rest_after
    if not m:
        m = _FROM_RE.search(rest_before)
        comparison_text = rest_before
    if not m:
        return None
    try:
        value2 = float(m.group("value2").replace(",", ""))
    except ValueError:
        return None
    if value2 == 0:
        return None
    if metric == "PAT" or unit == "crore":
        delta = round((value - value2) / value2 * 100, 2)
        delta_unit = "%"
    elif unit == "%":
        delta = round((value - value2) * 100, 2)  # percentage points -> bps
        delta_unit = "bps"
    else:  # already "bps", or unit unknown -- treat as directly comparable
        delta = round(value - value2, 2)
        delta_unit = "bps"
    if delta == 0:
        direction = "flat"
    else:
        direction = "increase" if delta > 0 else "decrease"
    return {
        "delta": abs(delta),
        "delta_unit": delta_unit,
        "direction": direction,
        "period": _guess_period(comparison_text[:_COMPARISON_SEARCH_WINDOW]),
    }


def _split_sentences(text: str) -> list[str]:
    # Narration is one long run-on string per segment. Split on ". " AND on the
    # bullet/list markers used for subsidiary breakdowns (o, ▪, •) -- without
    # this, "Q1FY27 PAT grew 29%... o Strong asset quality... ▪ Axis AMC: PAT..."
    # stays one giant sentence and a metric mention bleeds into unrelated bullets.
    text = re.sub(r"\s+[▪•]\s+", ". ", text)
    text = re.sub(r"(?<=[a-z%\d])\s+o\s+(?=[A-Z])", ". ", text)
    return re.split(r"(?<=[a-z%\d])\.\s+(?=[A-Z])", text)


def _extract_pass1(sentence: str) -> list[dict]:
    """PASS 1 -- byte-for-byte the pre-2026-09 extraction logic: narrow
    trigger words (_VALUE_RE_NARROW), explicit-delta-only (no comparison
    fallback), delta searched over the same window used before this
    session's Kotak/IndusInd tuning work. Runs first over the WHOLE quarter
    (see extract_quarter_metrics); whatever it finds for a metric is FINAL.
    This is Axis's own proven extraction path, and the reason the broadened
    _extract_pass2 below can be added at all without regression risk -- do
    not add new trigger words, unit alternatives, or a comparison fallback
    here; that belongs in PASS 2."""
    records = []
    for metric, pattern in METRIC_PATTERNS_NARROW.items():
        m = re.search(pattern, sentence, re.IGNORECASE)
        if not m:
            continue
        rest = sentence[m.end(): m.end() + _DELTA_SEARCH_WINDOW]
        vm = _VALUE_RE_NARROW.search(rest[:_VALUE_SEARCH_WINDOW])
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


def _extract_pass2(sentence: str, only_metrics: set[str]) -> list[dict]:
    """PASS 2 -- broadened 2026-09 for Kotak/IndusInd (see module
    docstring): wider value triggers (_VALUE_RE_BROAD) and, when no
    explicit delta is stated, a COMPUTED comparison-delta fallback
    (_comparison_delta). only_metrics restricts this to metrics
    _extract_pass1 found nothing for anywhere in the quarter (set by
    extract_quarter_metrics) -- that restriction, not anything in this
    function's own logic, is what keeps PASS 2 unable to overwrite or
    corrupt a PASS-1 (Axis-proven) match, no matter how broad its patterns
    get."""
    records = []
    for metric, pattern in METRIC_PATTERNS.items():
        if metric not in only_metrics:
            continue
        m = re.search(pattern, sentence, re.IGNORECASE)
        if not m:
            continue
        rest = sentence[m.end(): m.end() + _DELTA_SEARCH_WINDOW]
        vm = _VALUE_RE_BROAD.search(rest[:_VALUE_SEARCH_WINDOW])
        if not vm:
            continue
        try:
            value = float(vm.group("value").replace(",", ""))
        except ValueError:
            continue
        value_unit = _normalize_unit(vm.group("unit"))
        rest_after_value = rest[vm.end():]
        dm = _DELTA_RE.search(rest_after_value) or _DELTA_RE_REV.search(rest_after_value)
        if dm:
            try:
                delta = float(dm.group("delta").replace(",", ""))
            except ValueError:
                continue
            delta_info = {
                "delta": delta,
                "delta_unit": "bps" if "bp" in dm.group("delta_unit").lower() else "%",
                "direction": _direction_from_word(dm.group("dirword")),
                "period": _normalize_period(dm.group("period")),
            }
        else:
            delta_info = _comparison_delta(metric, value, value_unit, rest[:vm.start()], rest_after_value)
            if delta_info is None:
                continue
        records.append({
            "metric": metric,
            "value": value,
            "value_unit": value_unit,
            **delta_info,
        })
    return records


def extract_quarter_metrics(narration_text: str, bank_id: str = DEFAULT_BANK) -> dict[str, dict]:
    """Returns {metric: {value, value_unit, delta, delta_unit, direction, period}}.
    Keeps the FIRST match per metric per quarter (the bank-level summary-line
    mention, which appears before the detailed walk-through in every transcript
    observed) -- skipping subsidiary-context sentences (this bank's own
    registered subsidiary names, see _subsidiary_markers), which report their
    own PAT/ROE and would otherwise be mistaken for the bank-level figure.

    Two FULL passes over every sentence, not one merged pass -- see
    _VALUE_RE_NARROW/_VALUE_RE_BROAD above for why a single broadened pass
    was rejected (it regressed 8 of Axis's own 21 quarters). PASS 1
    (_extract_pass1) runs over the whole quarter first and is byte-for-byte
    the pre-2026-09 logic; whatever it finds for a metric is FINAL. PASS 2
    (_extract_pass2) then runs, but ONLY for metrics PASS 1 found nothing
    for anywhere in the quarter -- this is what reaches Kotak/IndusInd's
    "is/of/stood/remained" phrasing and two-value comparisons without ever
    being able to touch a metric Axis's own narration already satisfied."""
    subsidiary_re = _subsidiary_markers(bank_id)
    sentences = [s for s in _split_sentences(narration_text) if not subsidiary_re.search(s)]

    result: dict[str, dict] = {}
    for sentence in sentences:
        for rec in _extract_pass1(sentence):
            if rec["metric"] not in result:
                result[rec["metric"]] = {k: v for k, v in rec.items() if k != "metric"}

    missing = set(METRIC_PATTERNS) - set(result)
    if missing:
        for sentence in sentences:
            for rec in _extract_pass2(sentence, missing):
                if rec["metric"] not in result:
                    result[rec["metric"]] = {k: v for k, v in rec.items() if k != "metric"}

    return result


def build_metrics_timeseries(path: str = METRICS_TIMESERIES_PATH, graph_path: str = None,
                             bank_id: str = DEFAULT_BANK) -> dict:
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
        result[quarter] = extract_quarter_metrics(full_text, bank_id=bank_id)

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
