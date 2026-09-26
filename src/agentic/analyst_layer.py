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


def peer_extra_slot(peer_salience: dict | None, pref: dict, already_in: set[str]) -> str | None:
    """The ONE candidate a peer-bank signal can add, never more -- faithful
    port of src/engine.py's PEER_SALIENCE extra-slot block (search "Peer-bank
    extra slot" there) into this pipeline's own additive-candidate mechanism.
    That block existed in the legacy deterministic engine but was never
    ported to this agentic pipeline until now -- "peer" appeared nowhere in
    run_agentic.py/analyst_layer.py/graph_app.py before this.

    Motivation (src.signals.peer_signal's module docstring): Axis reports
    mid-to-late in results season, so peer banks that report earlier give
    analysts a first look at sector-wide themes before Axis's own call --
    genuinely new information relative to Axis's own question history, not
    more arithmetic on it.

    Unlike sentiment, this signal IS already topic-shaped (peer_salience is
    a per-topic dict from src.signals.peer_signal.compute_peer_signal, not a
    scalar), so unlike sentiment_extra_slot it doesn't need an
    anomaly-scores proxy -- it picks straight from its own topics, highest
    salience first, tie-broken by this analyst's own affinity, the exact
    order src/engine.py used.

    Gated by EngineConfig.PEER_MIN_SALIENCE (ignore weakly-covered topics)
    and PEER_AFFINITY_FLOOR (anti-spray: this analyst needs SOME real
    history on the topic, or a peer-hot theme would get sprayed at every
    analyst regardless of whether they've ever cared about it).

    MEASURED RESULT (2026-09, via eval_harness.evaluate_quarter on both
    official TEST_QUARTERS, using real peer transcripts for q4fy26 AND
    q1fy27 -- src.signals.peer_signal ported to read kotak/indusind's own
    multi-quarter archives means this is the first time this signal has
    ever been measurable for both held-out quarters, not just q1fy27):
    ZERO effect, mean_recall/mean_precision byte-identical with the signal
    on or off (0.9375/0.3678). Root-caused precisely, not just observed: a
    per-analyst diagnostic across all 49 active-analyst-quarters (24 in
    q4fy26, 25 in q1fy27) found a qualifying peer-salient candidate for
    EVERY single one (never blocked by "no candidate clears the floors"),
    but in every single case that candidate was ALREADY inside the
    analyst's own top-N picks before the peer signal ran -- 49/49 blocked
    by "already in", 0/49 by "no candidate". This quarter's peer-salient
    topics (NIM & Yields, Credit Cost & Provisions, Deposits & CASA, Loan
    Growth, Profitability) are exactly the perennially-hot topics the base
    model already ranks highly for nearly every analyst from their own
    history, so peer salience is DIRECTIONALLY CORRECT (it flags real
    topics analysts do ask about) but never finds new ground to add -- same
    "mechanism sound, hypothesis doesn't fire often enough" shape as
    sentiment_extra_slot's own null result, and same reason to leave
    use_peer_signal off by default rather than call it a bug. Would plausibly
    fire on a quarter where peer banks surface something genuinely OFF an
    analyst's usual pattern (a fresh policy/news shock hitting a topic they
    don't already cover) -- worth re-testing once such a quarter exists,
    rather than concluding this mechanism can never help from one null
    result on ordinary quarters."""
    if not peer_salience:
        return None
    cand = [(t, sv) for t, sv in peer_salience.items()
            if sv >= EngineConfig.PEER_MIN_SALIENCE
            and t not in already_in
            and pref.get(t, 0.0) >= EngineConfig.PEER_AFFINITY_FLOOR]
    if not cand:
        return None
    cand.sort(key=lambda x: (-x[1], -pref.get(x[0], 0.0)))
    return cand[0][0]


def macro_event_extra_slot(macro_signal: dict | None, pref: dict, already_in: set[str]) -> str | None:
    """The ONE candidate a verified external macro/policy/news event can
    add, never more -- same extra-slot shape as peer_extra_slot/
    sentiment_extra_slot above. macro_signal is {topic: severity in [0,1]},
    already filtered and scored by the caller
    (src.signals.external_context.macro_topic_signal) from curated events --
    see that module's docstring for the event schema (real source_url,
    own-words summary, confirmed/plausible confidence) and verification
    discipline.

    Same anti-spray floor as peer/sentiment: this analyst needs SOME real
    history on the affected topic, or a single curated event would spray
    the same extra slot at every active analyst regardless of whether
    they've ever asked about that topic.

    VALIDATION CEILING, stated up front because it's the honest thing to do
    here (same discipline as src.signals.news_signal's module docstring):
    there is no historical archive of verified-macro-event ->
    next-quarter-question pairs, so this cannot be walk-forward backtested
    the way sentiment/adaptive-decay were -- only real curated events
    accumulate going forward, starting from whatever gets curated today.
    Permanently opt-in for that reason. The caller (run_agentic.
    build_initial_state) refuses this signal unconditionally in holdout
    mode, same reasoning news_signal already documents: using an event
    curated with hindsight to score a historical quarter would leak."""
    if not macro_signal:
        return None
    cand = [(t, sv) for t, sv in macro_signal.items()
            if t not in already_in
            and pref.get(t, 0.0) >= EngineConfig.MACRO_EVENT_AFFINITY_FLOOR]
    if not cand:
        return None
    cand.sort(key=lambda x: (-x[1], -pref.get(x[0], 0.0)))
    return cand[0][0]


def cross_bank_pullin_candidates(cross_bank_pref: dict | None, N: float,
                                 already_in: set[str]) -> list[str]:
    """Topics the RAW (unblended) cross-bank prior can pull into this
    analyst's candidate pool even though nothing in this bank's own
    anomaly/momentum signals put them there -- the missing half of
    cross_bank_persona.py's mechanism (see that module's docstring):
    blend_cross_bank_prior already lets cross-bank history reweight a topic
    that's ALREADY a candidate, but nothing previously let it ADD one. Per
    claude/cross-bank-candidate-pool-scope.md.

    Unlike sentiment/peer/macro's single-slot mechanism above, this can
    return MORE than one topic -- it mirrors the disclosure-signal shape a
    few lines into reweight_for_analyst below (entries ADDED to the
    candidate pool, still competing on score in that function's own
    ranking, not a guaranteed extra slot), per the scope doc's proposed
    code.

    Deliberately NOT gated on the blended `pref` dict being nonzero on a
    candidate topic, even though every other extra-slot mechanism in this
    file uses exactly that anti-spray check. Blended pref =
    (1-w)*own + w*cross (cross_bank_persona.blend_cross_bank_prior); once a
    topic has already cleared CROSS_BANK_PULLIN_THRESHOLD on the RAW cross
    value below, blended pref on it is then guaranteed >= w*THRESHOLD > 0 --
    checking blended pref again would be a vacuous anti-spray gate, not a
    real one (it would always pass whenever the eligibility check already
    passed). The anti-spray floor used instead is CROSS_BANK_PULLIN_MAX_N:
    this analyst's own-bank expected question count N (the same N that
    sizes their slot budget) must be at or below the floor -- i.e. pull-in
    only activates for genuinely thin-own-history analysts, the exact
    population the empirical case in cross_bank_persona.py's docstring is
    about (an overlapping analyst whose OWN-bank track record is too
    sparse to have built a reliable topic profile from it alone). N is an
    imperfect thinness proxy -- it is "expected questions this quarter",
    not literally "quarters of history on file" -- but it's already
    computed leak-free by the caller and needs no new plumbing; documented
    here as an approximation, not hidden.

    MEASURED RESULT (2026-09, via a walk-forward sweep over Kotak's and
    IndusInd's own TRAINING quarters -- 15 and 9 scored quarters
    respectively, TEST_QUARTERS never touched during the sweep): at the
    chosen (THRESHOLD=0.18, MAX_N=2.5), mean topic recall averaged
    +1.1pp over both the no-cross-bank baseline AND the already-committed
    reorder-only blend (CROSS_BANK_MIX alone), with precision essentially
    flat -- a real, if modest, improvement, and the first sign this
    mechanism does what the reorder-only blend structurally cannot (see
    this module's own docstring above). That uplift did NOT reproduce on
    the 2 official TEST_QUARTERS: byte-identical to reorder-only for both
    banks. Root-caused, not just observed: on q4fy26/q1fy27, the mechanism
    fires (adds a pulled-in candidate) for only 0-4 of 19-23 active
    analysts per quarter, and in every firing instance on those two
    quarters, either the analyst wasn't in the SCORED set that quarter
    (asked no topic-tagged question at all, so invisible to macro
    recall/precision regardless of what gets predicted for them) or the
    pulled-in topic wasn't their actual asked topic, so the swap changed
    their predicted list without changing their score. Same "mechanism
    sound, doesn't fire often enough on this particular small sample" shape
    as peer_extra_slot's own null result above -- left off by default
    (allow_cross_bank_pullin=False everywhere) for that reason, not a bug.
    Worth re-testing as more quarters accumulate in the official held-out
    set, or if this is ever run against a bank with a thinner own-history
    analyst roster than Kotak/IndusInd's current one."""
    if not cross_bank_pref or N > EngineConfig.CROSS_BANK_PULLIN_MAX_N:
        return []
    return [t for t, v in cross_bank_pref.items()
            if t not in already_in and v >= EngineConfig.CROSS_BANK_PULLIN_THRESHOLD]


def reweight_for_analyst(analyst: str, overall_ranked_topics: list[str], pref: dict, N: int,
                         disclosure: dict | None = None,
                         slot_extra: int | None = None, slot_cap: int | None = None,
                         sentiment_score: float | None = None,
                         anomaly_scores: dict | None = None,
                         use_sentiment_signal: bool = False,
                         news_signal: dict | None = None,
                         use_news_signal: bool = False,
                         peer_salience: dict | None = None,
                         use_peer_signal: bool = False,
                         macro_signal: dict | None = None,
                         use_macro_signal: bool = False,
                         cross_bank_pref: dict | None = None,
                         allow_cross_bank_pullin: bool = False) -> list[str]:
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
    the legacy engine's novelty/peer signals.

    use_news_signal=True (default False everywhere, and unlike
    use_sentiment_signal this one is NOT backtest-promotable from this
    module alone -- see src.signals.news_signal's module docstring for why:
    there is no historical news archive to backtest against, full stop)
    merges news_signal (src.signals.news_signal.news_topic_salience's
    output, computed by the caller -- this function stays pure) into the
    SAME disclosure-shaped signal dict below, so a strongly-covered topic
    can enter the candidate pool exactly the way a disclosure-flagged topic
    already can. Stays opt-in indefinitely, not "opt-in until backtested".

    use_peer_signal=True (default False everywhere -- opt-in until a
    backtest promotes it, same discipline as use_sentiment_signal/
    use_adaptive_decay -- UNLIKE use_news_signal this one CAN be
    backtested: peer_salience for a historical quarter comes from that same
    quarter's own real peer-bank transcripts, not "right now", so scoring a
    held-out quarter with it doesn't leak) adds peer_extra_slot()'s one
    candidate, if any, as a genuine extra slot -- same non-displacing
    mechanism as sentiment, faithfully ported from src/engine.py's
    PEER_SALIENCE block (the legacy engine's own peer signal, which this
    pipeline never had until now).

    use_macro_signal=True (default False everywhere, and like
    use_news_signal this one is NOT backtest-promotable from this module
    alone -- see src.signals.external_context.macro_topic_signal's
    docstring for why: curated events have no historical archive to test
    against either) adds macro_event_extra_slot()'s one candidate from a
    verified external macro/policy/news event tagged with topic+severity.
    Refused in holdout mode by the caller (build_initial_state), same
    reasoning as use_news_signal.

    allow_cross_bank_pullin=True (default False everywhere; only
    meaningful when use_cross_bank_signal is ALSO True, since
    cross_bank_pref is only ever populated in that case -- see
    build_initial_state's docstring) is the stronger, still-ungated
    mechanism claude/cross-bank-candidate-pool-scope.md scoped out from
    use_cross_bank_signal's original reorder-only blend: it lets
    cross_bank_pullin_candidates() add topics to `candidates` BEFORE the
    slot cutoff below, the same "enters the pool, still competes on score"
    shape as the disclosure-signal loop just above -- unlike
    sentiment/peer/macro's guaranteed, non-displacing extra slot. This is
    deliberately NOT exposed anywhere in run_agentic.main, graph_app's live
    call, or any FastAPI route/frontend toggle -- reachable only by calling
    build_initial_state / eval_harness.evaluate_quarter directly with the
    kwarg, same guardrail already documented on use_cross_bank_signal
    itself, because letting cross-bank data manufacture a prediction that
    wouldn't otherwise exist is the stronger use Section 8 flags as pending
    legal sign-off -- reordering an already-visible list (the committed
    CROSS_BANK_MIX path) is the milder one that shipped already."""
    signal = _disclosure_signal(disclosure)
    if use_news_signal and news_signal:
        for t, v in news_signal.items():
            signal[t] = max(signal.get(t, 0.0), v)

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

    if allow_cross_bank_pullin:
        for t in cross_bank_pullin_candidates(cross_bank_pref, N, set(candidates)):
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

    if use_peer_signal:
        bonus = peer_extra_slot(peer_salience, pref, set(result))
        if bonus:
            result = result + [bonus]  # same non-displacing shape, ported from src/engine.py

    if use_macro_signal:
        bonus = macro_event_extra_slot(macro_signal, pref, set(result))
        if bonus:
            result = result + [bonus]  # same non-displacing shape; permanently opt-in, see docstring

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
