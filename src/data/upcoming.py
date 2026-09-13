"""
upcoming.py — condition prediction on the UPCOMING quarter's disclosure.

Until now the pipeline predicted the next call from history alone, which is
only half the architecture: Section 2.1 lists "the draft disclosure
script/investor presentation close to call date" as an ingestion source
precisely because it is what makes Section 3.3's in-call arithmetic
follow-ups possible. You cannot predict "if NIM fell 16bps and assets grew
3-4%, NII should have declined — what am I missing" from history; you predict
it from the bridge numbers management is about to read out.

This module takes an uploaded PDF/TXT (draft script, investor presentation,
press release, or a real transcript) and turns it into prediction inputs:

  metrics          — structured (metric, value, delta, direction) tuples,
                     via the same regex extractor used on the archive
  anomaly_scores   — how unusual each disclosed move is against that metric's
                     own history, capped at the training cutoff so an upload
                     can be scored under held-out conditions too
  topic_salience   — which taxonomy topics the disclosure actually talks about
  drill_down_flags — Section 3.3's drill-down predictor: the specific
                     sentences whose SHAPE historically draws scrutiny
                     (multi-part bridges, deltas stated without a driver)

Nothing here calls an LLM. It is all deterministic parsing, so an upload
yields usable prediction inputs even in an air-gapped deployment.
"""

import io
import json
import os
import re

from src.config.settings import METRICS_TIMESERIES_PATH
from src.signals.metrics_extractor import extract_quarter_metrics, METRIC_TOPIC_MAP
from src.graphs.compiler import TOPICS as TOPIC_KEYWORDS

# Sentences that say a number CHANGED but never say why are the ones analysts
# reliably pick at; sentences that decompose a move into parts invite
# "does that add up" arithmetic checks.
_EXPLANATION_CUES = re.compile(
    r"\b(due to|driven by|on account of|primarily|reflect(?:s|ing)?|because|"
    r"led by|attributable to|as a result of|supported by|aided by)\b", re.I)
_BRIDGE_CUES = re.compile(
    r"\b(bridge|walk|basis points of|bps of|movement|decompos\w*|contributed|"
    r"split between|made up of|comprising|of which)\b", re.I)
_DELTA_CUES = re.compile(
    r"\b(increase[d]?|decrease[d]?|declin\w+|grew|growth|up|down|improv\w+|"
    r"compress\w+|expand\w+|fell|rose)\b", re.I)
_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|bps|basis points|crores?|bn|mn)?")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """PDF via pypdf, anything else as UTF-8 text."""
    if filename.lower().endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    return file_bytes.decode("utf-8", errors="ignore")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"[•▪▫]", ". ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]


def anomaly_scores_for_metrics(metrics: dict, history_quarters: list[str],
                               timeseries: dict | None = None) -> dict[str, float]:
    """Percentile rank of each disclosed |delta| against that metric's own
    history, restricted to history_quarters (so the caller controls the
    training cutoff). 1.0 = the largest move that metric has ever had."""
    if timeseries is None:
        with open(METRICS_TIMESERIES_PATH) as f:
            timeseries = json.load(f)

    history: dict[str, list[float]] = {}
    allowed = set(history_quarters)
    for quarter, ms in timeseries.items():
        if quarter not in allowed:
            continue
        for metric, rec in ms.items():
            if rec.get("delta") is not None:
                history.setdefault(metric, []).append(abs(rec["delta"]))

    scores: dict[str, float] = {}
    for metric, rec in metrics.items():
        topic = METRIC_TOPIC_MAP.get(metric)
        if not topic or rec.get("delta") is None:
            continue
        hist = history.get(metric, [])
        cur = abs(rec["delta"])
        score = 0.5 if not hist else sum(1 for h in hist if h <= cur) / len(hist)
        scores[topic] = max(scores.get(topic, 0.0), score)
    return scores


def topic_salience(text: str) -> dict[str, float]:
    """Share of taxonomy keyword hits per topic — what the disclosure spends
    its words on, normalised so the biggest theme is 1.0."""
    low = text.lower()
    counts = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", low)) for kw in keywords)
        if hits:
            counts[topic] = hits
    if not counts:
        return {}
    top = max(counts.values())
    return {t: round(c / top, 3) for t, c in sorted(counts.items(), key=lambda x: -x[1])}


def drill_down_flags(text: str, max_flags: int = 12) -> list[dict]:
    """Section 3.3's drill-down predictor. Flags disclosure sentences whose
    shape historically draws scrutiny, with the reason stated so an IR team
    can act on it before the call rather than get surprised on it."""
    flags = []
    for sentence in _split_sentences(text):
        numbers = _NUMBER.findall(sentence)
        numeric_tokens = [n for n in numbers if n.strip()]
        reasons = []

        if _BRIDGE_CUES.search(sentence) and len(numeric_tokens) >= 3:
            reasons.append("multi_part_bridge")
        if (_DELTA_CUES.search(sentence) and len(numeric_tokens) >= 1
                and not _EXPLANATION_CUES.search(sentence)):
            reasons.append("unexplained_delta")
        if len(numeric_tokens) >= 5:
            reasons.append("dense_numeric_claim")

        if reasons:
            topics = [t for t, kws in TOPIC_KEYWORDS.items()
                      if any(re.search(r"\b" + re.escape(kw) + r"\b", sentence.lower()) for kw in kws)]
            flags.append({
                "reasons": reasons,
                "sentence": sentence[:600],
                "numeric_components": len(numeric_tokens),
                "topics": topics[:3],
            })

    # Densest / most decomposed claims first — those are the ones that get picked at.
    flags.sort(key=lambda f: (-len(f["reasons"]), -f["numeric_components"]))
    return flags[:max_flags]


def parse_upcoming_document(file_bytes: bytes, filename: str,
                            history_quarters: list[str]) -> dict:
    """Everything the prediction pipeline needs from an uploaded disclosure."""
    text = extract_text(file_bytes, filename)
    if not text.strip():
        return {"error": "No extractable text found in that file (a scanned PDF needs OCR first)."}

    metrics = extract_quarter_metrics(text)
    anomalies = anomaly_scores_for_metrics(metrics, history_quarters)
    salience = topic_salience(text)
    flags = drill_down_flags(text)

    return {
        "filename": filename,
        "chars_extracted": len(text),
        "narration": text,
        "metrics": metrics,
        "anomaly_scores": anomalies,
        "topic_salience": salience,
        "drill_down_flags": flags,
        "summary": {
            "metrics_found": sorted(metrics.keys()),
            "topics_touched": list(salience.keys())[:8],
            "n_drill_down_flags": len(flags),
            "top_anomalies": sorted(anomalies.items(), key=lambda x: -x[1])[:5],
        },
    }
