"""
cross_bank_persona.py — opt-in cross-bank analyst topic-history signal.

Section 8 of this platform's original design doc flagged a "cross-bank
coverage graph" as future work "pending legal sign-off" -- using one bank's
analyst-behavior data to help predict another bank's questions is a
materially different thing than using a bank's own history, in a regulated
context. This module exists so that idea can be measured and evaluated
without being wired into any default prediction path.

The empirical case (2026-09 cross-bank analysis, run after Kotak/IndusInd
got real multi-quarter history in Phase 9): roughly a third to half of each
bank's analyst roster also covers at least one of the other two registered
banks. For those overlapping analysts, their Axis question history alone
out-predicts their own in-bank history on Kotak's and IndusInd's held-out
quarters in a standalone top-K-by-topic-preference backtest (75.0%/73.9%
recall vs 63.9%/66.7%), and combining both sources that way measured
81-89% recall, a +21-23pp uplift, in that same standalone script.

That uplift does NOT reproduce when this signal is wired through the real
scored pipeline (verified 2026-09 via eval_harness.run_holdout_eval with
use_cross_bank_signal=True): Kotak showed +0.0pp mean recall, IndusInd
showed -1.6pp. The reason is structural, not a bug -- analyst_layer.py's
reweight_for_analyst() only REORDERS topics already in the bank's global
candidate pool (built from that bank's own anomaly/momentum signals) and
caps each analyst's slots at their own-bank question count N plus a fixed
extra; it doesn't let a high cross-bank-preference topic enter the
candidate pool the way a strong disclosure signal can. So blending this
prior mostly reshuffles an already-adequate ranking rather than surfacing
topics that were missing. The standalone script measured something
different: it selected top-K topics directly from the blended preference,
which could surface topics outside that pool entirely. Expanding
reweight_for_analyst() to let this signal add candidates (not just reorder
them), the way disclosure signals already do, is the change that would be
needed to realize anything like the standalone uplift -- not done here,
both because it's out of this module's scope and because it would widen
the same pending-legal-sign-off surface this module was already built to
keep contained.

Nothing here is wired into any pipeline by default. build_initial_state()
only calls into this module when explicitly passed use_cross_bank_signal=True
(see run_agentic.py); every prediction and evaluation call site defaults that
flag to False, so the live product's behavior is completely unchanged until
someone opts in per-call, for testing -- and given the measured pipeline
result above, opting in currently buys little to nothing even then.
"""

import json
import os

from src.config.banks import BANKS
from src.config.settings import paths_for, TOPICS_LIST, EngineConfig


def _quarter_sort_key(qid: str) -> int:
    """Same formula as dataset_compiler.get_quarter_info(): fiscal quarters
    are labelled identically across every bank in the registry (all are
    April-March Indian fiscal years), so "q3fy26" means the same real-world
    quarter for Axis, Kotak and IndusInd alike -- this key is safe to compare
    across banks, not just within one bank's own quarter_order."""
    year = int(qid[4:6])
    qtr = int(qid[1])
    return year * 10 + qtr


def _load_bank_topic_history(bank_id: str) -> dict[str, dict[str, set[str]]] | None:
    """analyst -> {quarter_id: set(topics)} for one bank, or None if that
    bank has no compiled graph yet (a freshly-registered bank with no
    transcripts is not an error here, just has nothing to contribute)."""
    paths = paths_for(bank_id)
    if not os.path.exists(paths.graph_path):
        return None
    with open(paths.graph_path) as f:
        graph = json.load(f)
    out: dict[str, dict[str, set[str]]] = {}
    for n in graph.get("nodes", []):
        if n["type"] != "Question":
            continue
        p = n["properties"]
        analyst = p.get("analyst")
        if not analyst or analyst in ("Moderator", "Operator"):
            continue
        topics = {t for t in p.get("topics", []) if t != "General"}
        if not topics:
            continue
        out.setdefault(analyst, {}).setdefault(p["quarter"], set()).update(topics)
    return out


# Process-lifetime cache: graph.json files only change when someone recompiles
# a bank, which happens far less often than this is called per-analyst during
# a single prediction run. Call cross_bank_cache_clear() after a recompile
# (mirrors _load_live()'s cache invalidation pattern in fastapi_app.py).
_bank_history_cache: dict[str, dict | None] = {}


def cross_bank_cache_clear() -> None:
    _bank_history_cache.clear()


def _bank_history(bank_id: str) -> dict[str, dict[str, set[str]]] | None:
    if bank_id not in _bank_history_cache:
        _bank_history_cache[bank_id] = _load_bank_topic_history(bank_id)
    return _bank_history_cache[bank_id]


def cross_bank_topic_prior(analyst: str, exclude_bank_id: str,
                           as_of_quarter: str | None = None) -> dict[str, float] | None:
    """Normalized topic distribution for `analyst` from every OTHER
    registered bank's history, strictly before `as_of_quarter` (leak-free,
    same discipline as build_initial_state's own train/target split) --
    or all history if as_of_quarter is None.

    Returns None (not {}) when the analyst has no cross-bank history at all,
    so callers can tell "no signal" apart from "a signal that happens to be
    uniform" and skip blending entirely rather than diluting toward zeros.
    """
    cutoff_key = _quarter_sort_key(as_of_quarter) if as_of_quarter else None
    counts: dict[str, float] = {}
    total = 0.0
    for bank_id in BANKS:
        if bank_id == exclude_bank_id:
            continue
        hist = _bank_history(bank_id)
        if not hist or analyst not in hist:
            continue
        for qid, topics in hist[analyst].items():
            if cutoff_key is not None and _quarter_sort_key(qid) >= cutoff_key:
                continue
            for t in topics:
                counts[t] = counts.get(t, 0.0) + 1.0
                total += 1.0
    if total == 0.0:
        return None
    return {t: counts.get(t, 0.0) / total for t in TOPICS_LIST if t != "General"}


def blend_cross_bank_prior(analyst: str, bank_id: str, pref: dict[str, float],
                           as_of_quarter: str | None = None,
                           mix: float | None = None) -> dict[str, float]:
    """Blends `pref` (this analyst's own-bank topic preference, as returned
    by src.memory.history.get_analyst_profile) with their cross-bank topic
    history, additive-prior style -- the same pattern get_analyst_profile
    already uses to blend in the hand-curated PERSONAS topic_prior. Returns
    `pref` completely unchanged when the analyst has no cross-bank history.
    """
    cross = cross_bank_topic_prior(analyst, bank_id, as_of_quarter=as_of_quarter)
    if cross is None:
        return pref
    w = EngineConfig.CROSS_BANK_MIX if mix is None else mix
    all_topics = set(pref) | set(cross)
    return {t: (1 - w) * pref.get(t, 0.0) + w * cross.get(t, 0.0) for t in all_topics}
