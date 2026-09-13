"""
moves.py — the analyst MOVE taxonomy.

Why this module exists
----------------------
Topic recall hit 92.6% on the held-out calls and the improvement loop saw
nothing to fix. That was not the loop failing; it was the loop optimising a
target that had already saturated. Management already knows the topics — a
brief saying "NIM will come up" is worth nothing. What management cannot
prepare for is the ANGLE: that someone will press where the 3.8% structural
NIM target stands against the 12-to-15-month timeline given two quarters ago.

Measured on the two held-out calls (48 genuine questions after cleaning the
decomposer): 20 of them, 42%, carry one of the moves below. The other 58% are
plain level/driver questions that follow from the topic alone. So the target
set is the move-bearing subset, and the unit of prediction is not a topic but
a (topic x move) pair.

The load-bearing observation: the surprising question is surprising in its
TARGET, not in its FORM. Analysts reuse a small, stable set of rhetorical
moves. That is what makes this tractable rather than hopeless — we cannot
predict novel content, but we can enumerate the moves, learn which ones each
analyst reaches for, and aim them at this quarter's disclosures.

Everything here is deterministic (air-gapped constraint). Profiles are built
from TRAINING quarters only, honouring the rolling cutoff in
run_agentic.build_initial_state.
"""

import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# The taxonomy.
#
#   key        stable id, used in profiles / eval / attribution
#   label      human label for the brief
#   needs      the input a generator must have to instantiate this move.
#              A move whose input is missing cannot be generated — which is
#              exactly how guidance_callback was silently unreachable: the
#              commitments store is a stub, so the framer never had the
#              5-of-48 most dangerous move available to it.
#   pattern    detector, for profiling history and for scoring
# ---------------------------------------------------------------------------

_SPEC = [
    (
        "quantification_demand",
        "Quantify the qualitative claim",
        "disclosed_qualitative_claim",
        r"\b(?:quantif\w+|how much of|what (?:is|was|would be) the (?:quantum|amount|number|split)"
        r"|can you (?:size|break|split)|in absolute terms|rupee terms|in bps|give us a number"
        r"|able to quantify|what'?s the (?:quantum|number))",
    ),
    (
        "guidance_callback",
        "Press an outstanding commitment",
        "open_commitment",
        r"\b(?:you (?:had )?(?:said|guided|indicated|mentioned|committed)"
        r"|earlier guidance|soft guidance|previous guidance|target of|the \d[\d.]*% target"
        r"|stand on that|versus your (?:guidance|target)|against (?:your|the) (?:guidance|target)"
        r"|as (?:i|we) recall|last (?:time|quarter) (?:we|you) (?:said|indicated|guided)"
        r"|one of the conversations that we'?ve had previously"
        r"|when do we think|any timeframe)",
    ),
    (
        "one_off_vs_steady_state",
        "One-off or the new normal?",
        "anomalous_metric",
        r"\b(?:one[- ]?off|sustainab\w+|steady[- ]state|repeatab\w+|recurring"
        r"|is this the new|structural(?:ly)?\b|continue (?:at )?this|hold at this"
        r"|annuali[sz]|is that a fair|new normal)",
    ),
    (
        "normalized_run_rate",
        "What should we model?",
        "metric_with_one_offs",
        r"\b(?:normali[sz]ed|underlying|adjusted for|ex[- ](?:of )?(?:the )?one[- ]?off"
        r"|clean(?:ed)? (?:number|run[- ]?rate)|what should we (?:build|model|assume)"
        r"|run[- ]?rate (?:for|of|go)|steady[- ]state (?:opex|cost|number))",
    ),
    (
        "cross_line_bridge",
        "Reconcile two disclosures",
        "two_related_metrics",
        r"\b(?:reconcil\w+|discrepanc\w+|doesn'?t (?:add|tie)|does not (?:add|tie)"
        r"|bridge\b|period[- ]end|versus the average|implied|implies"
        r"|the math|maths|if i (?:do|back) (?:the )?math|back[- ]calculat\w+"
        r"|but (?:your|the) .{0,40}(?:shows|suggests|implies))",
    ),
    (
        "absence_probe",
        "Ask about what was not said",
        "undisclosed_mover",
        r"\b(?:you (?:haven'?t|didn'?t|have not|did not) (?:mention|talk|comment|disclose|give)"
        r"|no (?:mention|colour|color|disclosure) (?:of|on)|nothing on"
        r"|any (?:colour|color|comment|thoughts) on"
        r"|comment on that as well|if you could (?:also )?comment)",
    ),
    (
        "forward_trajectory",
        "Level to trajectory",
        "disclosed_metric",
        r"\b(?:go from here|from here on|outlook|trajector\w+|next (?:few )?quarters?"
        r"|fy2[6-9]|going forward|how do you see (?:this|it) (?:evolv|trend|play)"
        r"|by when|exit (?:rate|quarter))",
    ),
]

# Detected for completeness so propensities are rates over ALL questions, not
# only interesting ones. Never a prediction target: management prepares these
# from the topic list alone, so predicting them scores well and helps nobody.
_BASELINE_KEY = "level_driver"
_BASELINE_PATTERN = (
    r"\b(?:what (?:drove|led to|caused)|why (?:did|has|is)|what (?:is|was) behind"
    r"|drivers? of|walk us through|colour on the|break[- ]?up of|how much (?:is|was))"
)

MOVES = {
    key: {"key": key, "label": label, "needs": needs,
          "re": re.compile(pat, re.I), "target": True}
    for key, label, needs, pat in _SPEC
}
MOVES[_BASELINE_KEY] = {
    "key": _BASELINE_KEY, "label": "Level and drivers", "needs": "disclosed_metric",
    "re": re.compile(_BASELINE_PATTERN, re.I), "target": False,
}

TARGET_MOVES = [k for k, v in MOVES.items() if v["target"]]
ALL_MOVES = list(MOVES)


def detect_moves(text: str) -> list[str]:
    """Which moves this question text performs. A question can perform more
    than one (Mahrukh's NIM ask is guidance_callback + forward_trajectory)."""
    if not text:
        return []
    return [k for k, v in MOVES.items() if v["re"].search(text)]


def is_move_bearing(text: str) -> bool:
    """True when the question does something management would not have
    prepared for from the topic list alone."""
    return any(m in TARGET_MOVES for m in detect_moves(text))


# ---------------------------------------------------------------------------
# Per-analyst propensity, from training quarters only.
# ---------------------------------------------------------------------------

def build_move_profiles(dataset: list[dict], train_quarters: list[str],
                        alpha: float = 4.0) -> dict:
    """Laplace-smoothed move rates per analyst, backed off to the global rate.

    alpha is the strength of the backoff prior in pseudo-questions. It matters:
    most analysts appear on only a handful of the training calls, so an
    unsmoothed rate would read 0.00 or 1.00 off two observations.
    """
    from src.agentic.question_eval import decompose_concerns

    train = set(train_quarters)
    per_analyst_counts: dict[str, Counter] = defaultdict(Counter)
    per_analyst_total: Counter = Counter()
    per_analyst_bearing: Counter = Counter()
    global_counts: Counter = Counter()
    global_total = 0
    global_bearing = 0
    quarters_seen: dict[str, set] = defaultdict(set)

    for rec in dataset:
        qid = rec.get("quarter_id")
        if qid not in train:
            continue
        for turn in rec.get("qa", []):
            if "nalyst" not in str(turn.get("role", "")):
                continue
            name = turn.get("analyst_name") or "unknown"
            for concern in decompose_concerns(turn.get("text", "")):
                found = detect_moves(concern)
                per_analyst_total[name] += 1
                global_total += 1
                if any(m in TARGET_MOVES for m in found):
                    per_analyst_bearing[name] += 1
                    global_bearing += 1
                quarters_seen[name].add(qid)
                for m in found:
                    per_analyst_counts[name][m] += 1
                    global_counts[m] += 1

    global_rate = {m: (global_counts[m] / global_total if global_total else 0.0)
                   for m in ALL_MOVES}

    profiles = {}
    for name, total in per_analyst_total.items():
        rates = {}
        for m in ALL_MOVES:
            obs = per_analyst_counts[name][m]
            rates[m] = (obs + alpha * global_rate[m]) / (total + alpha)
        top = sorted(TARGET_MOVES, key=lambda m: -rates[m])
        profiles[name] = {
            "n_questions": total,
            "n_quarters": len(quarters_seen[name]),
            "move_rates": {m: round(rates[m], 4) for m in ALL_MOVES},
            "signature_moves": top[:3],
            "move_bearing_share": round(per_analyst_bearing[name] / total, 4),
            "lift": {m: round(rates[m] / global_rate[m], 2)
                     for m in TARGET_MOVES if global_rate[m] > 0},
        }

    profiles["_global"] = {
        "n_questions": global_total,
        "move_rates": {m: round(global_rate[m], 4) for m in ALL_MOVES},
        "move_bearing_share": round(global_bearing / global_total, 4) if global_total else 0.0,
        "train_quarters": list(train_quarters),
    }
    return profiles


# ---------------------------------------------------------------------------
# Generation contract: (topic x move) slots instead of one question per topic.
# ---------------------------------------------------------------------------

def plan_topic_move_slots(ranked_topics: list[dict], analyst: str, profiles: dict,
                          available_inputs: dict, max_slots: int = 14,
                          per_topic_cap: int = 3) -> list[dict]:
    """Expand ranked topics into ranked (topic, move) slots.

    available_inputs maps a move's `needs` key to the evidence that would let a
    generator instantiate it for a given topic, e.g.
        {"open_commitment": {"NIM & Yields": [<commitment>, ...]}, ...}
    A move with no evidence for that topic is NOT emitted — a guidance_callback
    with no commitment to call back to is exactly the fluent-but-hollow output
    the Verifier exists to reject, and it is cheaper to never generate it.

    Recall-first: max_slots is deliberately generous (you said 50 predictions
    for 25 actuals is a good trade). The cap that matters is per_topic_cap,
    which stops one loud topic eating the whole brief.
    """
    prof = profiles.get(analyst) or profiles.get("_global", {})
    rates = prof.get("move_rates", {})
    slots = []
    for rank, t in enumerate(ranked_topics):
        topic = t.get("topic") if isinstance(t, dict) else str(t)
        topic_score = float(t.get("score", 1.0)) if isinstance(t, dict) else 1.0
        cands = []
        for m in TARGET_MOVES:
            need = MOVES[m]["needs"]
            evidence = (available_inputs.get(need) or {}).get(topic)
            if not evidence:
                continue
            cands.append({
                "topic": topic,
                "topic_rank": rank,
                "move": m,
                "move_label": MOVES[m]["label"],
                "evidence": evidence,
                # additive, never multiplicative — RESULTS_V2.md documents three
                # signals that died from the p*(1+w*signal) form
                "score": round(topic_score + 0.6 * rates.get(m, 0.0), 6),
            })
        cands.sort(key=lambda c: -c["score"])
        slots.extend(cands[:per_topic_cap])
    slots.sort(key=lambda c: (c["topic_rank"], -c["score"]))
    return slots[:max_slots]
