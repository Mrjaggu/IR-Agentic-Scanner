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

# ── Novel-theme detection ────────────────────────────────────────────────
#
# topic_salience/drill_down_flags above are keyword-matched against the fixed
# 12-topic taxonomy (TOPIC_KEYWORDS), so a genuinely new theme the disclosure
# introduces -- an FCNR liquidity opportunity, a new subsidiary line, a
# one-off regulatory item -- never enters either signal: hits=0, topics=[].
# The pipeline was then structurally blind to it, not just imprecise about
# it (see RESULTS_V2.md's post-mortem on the Kunal Q1FY27 FCNR miss). This
# gives a flagged-but-untagged sentence (one already shaped like it invites
# scrutiny -- a bridge, an unexplained delta, a dense numeric claim) a
# deterministic ad-hoc label instead of silently dropping it.
_NOVEL_SKIP_ACRONYMS = {
    "YOY", "QOQ", "MOM", "GDP", "RBI", "SEBI", "FY", "INR", "USD", "CEO",
    "CFO", "COO", "CTO", "EPS", "PAT", "NII", "NIM", "ROE", "ROA", "CASA",
    "GNPA", "NNPA", "PCR", "RWA", "SME", "CBG", "Q1", "Q2", "Q3", "Q4",
}
_NOVEL_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
_NOVEL_PHRASE_ANCHOR = re.compile(
    r"\b([a-zA-Z][a-zA-Z\s]{2,30}?)\s+"
    r"(opportunity|book|integration|segment|portfolio|platform|initiative|"
    r"vertical|business line|charge|item|scheme)\b", re.I)


def _novel_theme_label(sentence: str) -> str | None:
    """Deterministic short label for a theme outside the fixed taxonomy --
    no LLM, so this still works air-gapped. Prefers a domain acronym (FCNR,
    ECB, ADR...) since that is how these one-off themes are actually named
    on calls; falls back to a short noun-phrase anchored on a business-shape
    word ("... book", "... opportunity")."""
    for m in _NOVEL_ACRONYM.finditer(sentence):
        if m.group(1) not in _NOVEL_SKIP_ACRONYMS:
            return m.group(1)
    m2 = _NOVEL_PHRASE_ANCHOR.search(sentence)
    if m2:
        words = [w for w in m2.group(1).strip().split() if w.lower() not in
                 ("the", "a", "an", "this", "that", "our", "their", "and", "of", "to", "in")]
        if words:
            label = " ".join(w.capitalize() for w in words[-3:]) + " " + m2.group(2).capitalize()
            return label
    return None


def attach_novel_themes(salience: dict[str, float], flags: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Labels flagged sentences that matched no taxonomy topic, folds each
    label into `salience` (so it scores through the same disclosure-weight
    channel as a known topic) and into that flag's own `topics`. Returns
    (salience, flags, novel_summary) -- novel_summary is surfaced to the UI/API
    so an emerging theme is visible, not just silently blended in."""
    novel_summary = []
    seen_labels = {}
    for flag in flags:
        if flag.get("topics"):
            continue
        label = _novel_theme_label(flag["sentence"])
        if not label:
            continue
        key = label.lower()
        canonical = seen_labels.setdefault(key, label)
        flag["topics"] = [canonical]
        flag["novel_theme"] = True
        # Strong, flat salience -- these sentences were already selected for
        # being shaped like scrutiny-inviting disclosure (bridge / unexplained
        # delta / dense numeric claim), which is itself the evidence.
        score = min(1.0, 0.75 + 0.05 * len(flag.get("reasons", [])))
        prev = salience.get(canonical, 0.0)
        if score > prev:
            salience[canonical] = score
        novel_summary.append({"topic": canonical, "sentence": flag["sentence"][:300],
                              "reasons": flag.get("reasons", [])})
    return salience, flags, novel_summary


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
    salience, flags, novel_themes = attach_novel_themes(salience, flags)

    return {
        "filename": filename,
        "chars_extracted": len(text),
        "narration": text,
        "metrics": metrics,
        "anomaly_scores": anomalies,
        "topic_salience": salience,
        "drill_down_flags": flags,
        "novel_themes": novel_themes,
        "summary": {
            "metrics_found": sorted(metrics.keys()),
            "topics_touched": list(salience.keys())[:8],
            "n_drill_down_flags": len(flags),
            "n_novel_themes": len(novel_themes),
            "top_anomalies": sorted(anomalies.items(), key=lambda x: -x[1])[:5],
        },
    }
