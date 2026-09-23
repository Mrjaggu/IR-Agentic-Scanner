"""
analyst_layer.py — Section 3.3: Analyst-Specific Layer.

Two distinct prediction problems, per the doc:
  1. Pre-call topic questions: reweight the Overall layer's ranked topics per
     analyst using their measured style (persona / history), same principle
     as the doc's example -- "a globally hot topic an analyst has never
     asked about in 5 years gets downweighted for them specifically."
  2. In-call arithmetic follow-ups: driven by the actual disclosure numbers,
     not history -- only activates for analysts whose style profile shows a
     granular/arithmetic-check pattern, and only when this quarter's metrics
     show a genuine bridge tension (e.g. NIM down while profitability is up).
     Bounded to a handful of LLM calls (quota discipline already established
     in this codebase -- see RESULTS_V2.md).
"""

import json

MAX_ARITHMETIC_CALLS = 3
ARITHMETIC_STYLE_KEYWORDS = ("granular", "mechanics", "guidance numbers", "detailed follow-ups")


# Blend weight for the upcoming disclosure against measured analyst preference.
#
# ADDITIVE, not multiplicative, and that is deliberate: RESULTS_V2.md documents
# three separate signals (narration-novelty, peer-salience, personalised
# W_NARR) that all died the same way -- the engine's multiplicative
# `p * (1 + w*signal)` form keeps amplifying whichever topic already had high
# base preference, so a sharper signal just reshuffles the already-favoured.
# Its own stated fix was "an additive/blended combination instead of a
# multiplicative amplifier". This is that.
DISCLOSURE_BLEND = 0.25

# Slot policy comes from settings, where it is set by a walk-forward sweep over
# TRAINING quarters only. The direction of travel (widen slots to buy recall)
# is also what RESULTS_V2.md found independently: its N+1 boost was
# cross-validated across all 15 training quarters (recall 45.4%->59.5%) and
# adopted as the deterministic engine's default.
from src.config.settings import SLOT_EXTRA, SLOT_CAP, TOPICS_LIST, EngineConfig


def _disclosure_signal(disclosure: dict | None) -> dict[str, float]:
    """Per-topic strength of the upcoming disclosure: how much it discusses the
    topic, and whether any of its sentences are shaped to invite scrutiny."""
    if not disclosure:
        return {}
    salience = disclosure.get("topic_salience", {}) or {}
    signal = dict(salience)
    for flag in disclosure.get("drill_down_flags", []) or []:
        for t in flag.get("topics", []) or []:
            signal[t] = max(signal.get(t, 0.0), 1.0)
    return signal


def sentiment_extra_slot(sentiment_score: float | None, anomaly_scores: dict | None,
                         pref: dict, already_in: set[str]) -> str | None:
    """The ONE candidate a trending-skeptical analyst's sentiment can add,
    never more -- same extra-slot shape as NOVELTY_SCORES/PEER_SALIENCE in
    the legacy deterministic engine (src/engine.py), ported to this
    pipeline's own additive-candidate mechanism (the disclosure-signal loop
    just above, which is what actually lets a topic enter the pool here,
    not a multiplicative reweight).

    sentiment_score must already be the analyst's LEAK-FREE running-average
    tone as of strictly before the target quarter (see
    src.signals.analyst_sentiment.sentiment_as_of) -- this function does not
    fetch it itself, so it stays pure and easy to unit-test/backtest.

    The hypothesis, stated plainly because it's the part that's actually
    being tested here: sentiment has no topic dimension of its own (a
    running average is one number, not a topic distribution), so it can't
    say WHICH topic an analyst will ask about. What it can plausibly do is
    say a more skeptical analyst is more likely to probe whatever looks
    weakest THIS quarter -- so the candidate it adds is simply the highest
    anomaly-scored topic not already selected, gated by the same anti-spray
    rule the disclosure signal uses (SENTIMENT_AFFINITY_FLOOR: the analyst
    needs SOME real history on it, not zero, or this would spray the same
    topic at every skeptical analyst regardless of whether they've ever
    asked about it).

    MEASURED RESULT (2026-09, via eval_harness.run_holdout_eval and a
    broader walk-forward sweep -- same method as cross_bank_persona.py's own
    honest write-up): on the 2 official TEST_QUARTERS, zero effect at all --
    every analyst whose sentiment cleared SENTIMENT_SKEPTICAL_THRESHOLD that
    quarter turned out not to be a SCORED analyst (they didn't ask a
    topic-tagged question that quarter, so ground_truth excludes them from
    the aggregate). Widening to a 17-quarter walk-forward sweep (144
    analyst-quarter observations, same window rank_position_calibration
    uses) found the gate firing on exactly ONE scored observation, adding
    one extra prediction that turned out to be a miss: mean_recall unchanged
    (0.8081 both ways), mean_precision -0.0002. The mechanism is sound and
    the anti-spray guard is doing its job, but the anomaly-topic-as-proxy
    hypothesis just doesn't fire often enough among analysts who are BOTH
    skeptical AND active that quarter to move any number. Left off by
    default for this reason -- not a bug, a null result -- same as
    cross_bank_persona.py's CROSS_BANK_MIX. Revisit if a richer proxy than
    "most anomalous topic" is found, or once more quarters of sentiment
    history exist to test against."""
    if sentiment_score is None or sentiment_score > EngineConfig.SENTIMENT_SKEPTICAL_THRESHOLD:
        return None
    if not anomaly_scores:
        return None
    ranked = sorted(anomaly_scores.items(), key=lambda kv: -kv[1])
    for topic, _ in ranked:
        if topic in already_in:
            continue
        if pref.get(topic, 0.0) >= EngineConfig.SENTIMENT_AFFINITY_FLOOR:
            return topic
    return None


def reweight_for_analyst(analyst: str, overall_ranked_topics: list[str], pref: dict, N: int,
                         disclosure: dict | None = None,
                         slot_extra: int | None = None, slot_cap: int | None = None,
                         sentiment_score: float | None = None,
                         anomaly_scores: dict | None = None,
                         use_sentiment_signal: bool = False) -> list[str]:
    """Reorders the overall (global) ranked topics for this analyst and
    truncates to their slot count. A topic the analyst has essentially never
    engaged with sinks even if it's globally hot (the doc's worked example);
    a topic the upcoming disclosure puts front and centre rises, but only to
    the extent this analyst plausibly cares about it.

    When a disclosure is supplied, topics it flags can ENTER this analyst's
    candidate pool even if they fell outside the global top-N -- that is the
    only path by which a theme with no historical base rate (a subsidiary
    result, a one-off charge) can reach the brief at all.

    use_sentiment_signal=True (default False everywhere -- opt-in only,
    same pattern as build_initial_state's use_cross_bank_signal, until a
    backtest actually promotes it) adds sentiment_extra_slot()'s one
    candidate, if any, as a genuine EXTRA slot appended after the normal
    N+extra cutoff -- it never displaces a topic that earned its place on
    pref/disclosure score, same "never displaces top-N picks" discipline as
    the legacy engine's novelty/peer signals."""
    signal = _disclosure_signal(disclosure)

    candidates = list(overall_ranked_topics)
    for t, v in signal.items():
        if t in candidates or v < 0.5:
            continue
        if t in TOPICS_LIST:
            # A known taxonomy topic this analyst has never engaged with is
            # the doc's exact "globally hot topic they've never asked about"
            # case -- only pull it in if they have SOME real history on it,
            # otherwise the disclosure would spray the same topic at every
            # analyst regardless of whether they've ever cared.
            if pref.get(t, 0.0) > 0.0:
                candidates.append(t)
        else:
            # A genuinely new theme (not part of the fixed taxonomy) has zero
            # history for EVERY analyst by construction -- pref==0 here means
            # "this didn't exist before", not "this analyst doesn't care", so
            # the anti-spray rule above doesn't apply. A strong disclosure
            # signal is itself the reason to surface it.
            candidates.append(t)

    max_pref = max((pref.get(t, 0.0) for t in candidates), default=0.0) or 1.0
    alpha = DISCLOSURE_BLEND if signal else 0.0

    def score(t: str) -> float:
        norm_pref = pref.get(t, 0.0) / max_pref
        return (1 - alpha) * norm_pref + alpha * signal.get(t, 0.0)

    extra = SLOT_EXTRA if slot_extra is None else slot_extra
    cap = SLOT_CAP if slot_cap is None else slot_cap
    slots = max(1, min(cap, N + extra))
    result = sorted(candidates, key=lambda t: -score(t))[:slots]

    if use_sentiment_signal:
        bonus = sentiment_extra_slot(sentiment_score, anomaly_scores, pref, set(result))
        if bonus:
            result = result + [bonus]  # genuine extra slot -- appended, never displacing the slots above

    return result


def _bridge_tension(val_quarter_metrics: dict) -> dict | None:
    """Detects the doc's exact pattern class: a margin/yield metric moving
    one way while profitability moves the other -- the textbook setup for
    "if X fell and Y grew, shouldn't Z have declined -- what am I missing?"."""
    nim = val_quarter_metrics.get("NIM")
    pat = val_quarter_metrics.get("PAT")
    if not nim or not pat:
        return None
    if nim.get("direction") == "decline" and pat.get("direction") in ("increase", "improve"):
        return {"nim": nim, "pat": pat}
    return None


def _build_arithmetic_prompt(analyst: str, style_note: str, tension: dict) -> str:
    nim, pat = tension["nim"], tension["pat"]
    return f"""Analyst: {analyst}
Style (measured): {style_note}

This quarter's ACTUAL disclosed numbers (do not use any other numbers):
- NIM: {nim['value']}{nim.get('value_unit','')}, {nim['direction']} of {nim['delta']}{nim.get('delta_unit','')} ({nim.get('period','')})
- PAT: {pat['value']}{pat.get('value_unit','')}, {pat['direction']} of {pat['delta']}{pat.get('delta_unit','')} ({pat.get('period','')})

Task: write ONE in-call arithmetic follow-up question in this analyst's voice,
in the style of a real example from this platform's design doc: "if NIM fell
16bps and assets grew 3-4%, NII should have declined -- what am I missing
here". Use ONLY the numbers given above -- do not invent any other figure.
If the numbers above don't actually support a genuine arithmetic tension,
return {{"question": null}}.

Return JSON only: {{"question": "<the question text, or null>"}}"""


def build_arithmetic_followups(active_analysts: list[str], persona_stats: dict,
                               val_quarter_metrics: dict, client, budget: int = MAX_ARITHMETIC_CALLS) -> dict:
    """Returns {analyst: question_text} for analysts where this fired.

    Eligibility (granular/arithmetic-check style) is read from the
    hand-curated web-derived persona file (src.memory.history.PERSONAS),
    since that's where style descriptions like "granular retail questions"
    or "margin mechanics" actually live -- the transcript-derived
    persona_stats style_note (persona_consistent/narration_triggered rates)
    is a different, complementary signal, used below only for the LLM
    prompt's style context, not the eligibility gate."""
    from src.memory.history import PERSONAS

    tension = _bridge_tension(val_quarter_metrics)
    if not tension:
        return {}

    eligible = []
    for a in active_analysts:
        hand_style = PERSONAS.get(a, {}).get("style", "")
        if any(k in hand_style.lower() for k in ARITHMETIC_STYLE_KEYWORDS):
            measured = persona_stats.get(a, {}).get("style_note", "")
            style = f"{hand_style} {measured}".strip()
            eligible.append((a, style))

    results = {}
    for analyst, style_note in eligible[:budget]:
        prompt = _build_arithmetic_prompt(analyst, style_note, tension)
        raw = client.call_llm(prompt, temperature=0.1, purpose="analyst_arithmetic_reweighting")
        if not raw:
            continue
        try:
            import re
            clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
            parsed = json.loads(clean)
        except Exception:
            continue
        q = parsed.get("question")
        if q:
            results[analyst] = {"question": q, "grounded_in": tension}
    return results
