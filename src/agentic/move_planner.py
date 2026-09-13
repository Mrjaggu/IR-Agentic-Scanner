"""
move_planner.py — assembles the evidence each analyst MOVE needs, then hands
moves.plan_topic_move_slots a set of (topic x move) slots it can actually
instantiate.

The design rule this enforces
-----------------------------
A move is only offered to the generator when the input that move requires is
present for that topic. No commitment on NIM as of the cutoff means no
guidance_callback on NIM — full stop. This matters for two reasons:

  1. It is the seen-set == accepted-set rule again. A guidance_callback with no
     commitment behind it is a fluent invention, which is exactly what the
     Verifier exists to reject; generating it burns retry budget to produce
     nothing.
  2. It makes the failure LEGIBLE. When a move-bearing question is missed, the
     harness can now say whether the generator lacked the input (a data-layer
     gap, no prompt change will ever fix it) or had the input and failed to use
     it (a reasoning gap, which is the only thing the improvement loop should
     act on). That distinction is the one the loop was missing: everything
     landed in 'reasoning_weakness' and nothing actionable came out.
"""

import re

from src.agentic.moves import MOVES, TARGET_MOVES, plan_topic_move_slots

# Language that promises something without a number attached — the setup for
# "are you able to quantify that".
_QUALITATIVE = re.compile(
    r"\b(?:improve\w*|improvement|stabili[sz]\w+|moderat\w+|elevated|healthy|robust"
    r"|strong|soft(?:ness)?|pick(?:ed|ing)? up|traction|momentum|normali[sz]\w+"
    r"|calibrat\w+|rebalanc\w+|meaningful|significant|marginal)\b", re.I)

# Language that marks a number as contaminated by something non-recurring.
_ONE_OFF = re.compile(
    r"\b(?:one[- ]?off|exceptional|reversal|release|write[- ]?back|prior[- ]period"
    r"|settlement|divestment|stake sale|technical impact|accounting|realign\w+"
    r"|interchange|true[- ]?up)\b", re.I)

_NUM = re.compile(r"\d")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or ""))
            if s.strip()]


def build_available_inputs(topics: list[str], *,
                           dataset: list[dict] | None = None,
                           cutoff_quarter: str | None = None,
                           disclosure: dict | None = None,
                           disclosed_metrics: dict | None = None,
                           narration_by_topic: dict | None = None,
                           anomaly_scores: dict | None = None,
                           anomaly_floor: float = 0.5) -> dict:
    """Map each move's `needs` key to {topic: evidence}.

    Everything is as-of the cutoff. Commitments are read through
    open_commitments_as_of, which excludes anything made in or after the target
    quarter, so this path cannot leak the future.
    """
    disclosed_metrics = disclosed_metrics or {}
    narration_by_topic = narration_by_topic or {}
    anomaly_scores = anomaly_scores or (disclosure or {}).get("anomaly_scores") or {}
    salience = (disclosure or {}).get("topic_salience") or {}
    flags = (disclosure or {}).get("drill_down_flags") or []

    # --- open_commitment -------------------------------------------------
    commitments: dict[str, list] = {}
    if dataset and cutoff_quarter:
        from src.data.commitments import open_commitments_as_of, by_topic
        commitments = by_topic(open_commitments_as_of(dataset, cutoff_quarter))

    # --- flags indexed by topic -----------------------------------------
    flags_by_topic: dict[str, list] = {}
    for f in flags:
        for t in f.get("topics") or []:
            flags_by_topic.setdefault(t, []).append(f)

    inputs: dict[str, dict] = {need: {} for need in
                               {MOVES[m]["needs"] for m in TARGET_MOVES}}

    for topic in topics:
        metrics = disclosed_metrics.get(topic) or []
        if isinstance(metrics, dict):
            metrics = [metrics]
        narr = narration_by_topic.get(topic) or []
        narr_text = " ".join(narr)
        anomaly = anomaly_scores.get(topic)

        # open_commitment — the promise, most pressable first
        if commitments.get(topic):
            inputs["open_commitment"][topic] = commitments[topic][:2]

        # disclosed_qualitative_claim — a claim with adjectives and no number
        qual = [s for s in _sentences(narr_text)
                if _QUALITATIVE.search(s) and not _NUM.search(s)]
        if qual:
            inputs["disclosed_qualitative_claim"][topic] = qual[:2]

        # anomalous_metric — this quarter's move is unusual for this series
        if anomaly is not None and anomaly >= anomaly_floor:
            inputs["anomalous_metric"][topic] = {
                "anomaly": anomaly, "metrics": metrics[:2]}

        # metric_with_one_offs — a number management has flagged as impure
        # Gated on the one-off LANGUAGE, not on having a mapped metric. The
        # metric->topic map is lossy, and "you are seeing some reversals" is
        # itself the whole setup for "so what's the clean run rate" — requiring
        # a mapped metric on top of it made this move unreachable on every
        # topic, which is how Mahrukh's opex run-rate ask stayed unpredictable.
        dirty = [s for s in _sentences(narr_text) if _ONE_OFF.search(s)]
        if dirty:
            inputs["metric_with_one_offs"][topic] = {
                "metrics": metrics[:2], "one_off_language": dirty[:2]}

        # two_related_metrics — a bridge that can be made not to tie
        bridge = [f for f in flags_by_topic.get(topic, [])
                  if "multi_part_bridge" in (f.get("reasons") or [])
                  or "dense_numeric_claim" in (f.get("reasons") or [])]
        if len(metrics) >= 2 or bridge:
            inputs["two_related_metrics"][topic] = {
                "metrics": metrics[:3], "bridge_sentences": [f["sentence"] for f in bridge[:2]]}

        # undisclosed_mover — the metric moved and management stayed quiet
        if metrics and salience and salience.get(topic, 0.0) < 0.2:
            inputs["undisclosed_mover"][topic] = {
                "metrics": metrics[:2], "salience": salience.get(topic, 0.0)}

        # disclosed_metric — the baseline input
        if metrics or narr:
            inputs["disclosed_metric"][topic] = {"metrics": metrics[:2], "narration": narr[:2]}

    return inputs


def plan_for_analyst(analyst: str, ranked_topics, move_profiles: dict,
                     available_inputs: dict, max_slots: int = 14,
                     per_topic_cap: int = 3) -> list[dict]:
    """Ranked (topic x move) slots for one analyst."""
    topics = [{"topic": t, "score": 1.0 - 0.02 * i} if isinstance(t, str) else t
              for i, t in enumerate(ranked_topics)]
    return plan_topic_move_slots(topics, analyst, move_profiles, available_inputs,
                                 max_slots=max_slots, per_topic_cap=per_topic_cap)


def coverage_report(available_inputs: dict, topics: list[str]) -> dict:
    """Which moves are reachable at all this quarter, and for how many topics.

    Read this before blaming the generator: a move with zero reachable topics
    was never on the table.
    """
    out = {}
    for m in TARGET_MOVES:
        need = MOVES[m]["needs"]
        have = [t for t in topics if (available_inputs.get(need) or {}).get(t)]
        out[m] = {"needs": need, "topics_with_input": have, "n": len(have)}
    return out
