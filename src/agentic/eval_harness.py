"""
eval_harness.py — Section 6: the numbers that decide whether any of this is
allowed near production.

Two families of metric, because they measure two different layers and mixing
them is how you end up reporting a flattering average:

  Topic-level ranking (the Overall layer, Section 3.2)
    Precision@5, Precision@10, Recall, MRR of the ranked topic list against
    the union of topics actually raised on the call.

  Per-analyst prediction (the Analyst-Specific layer, Section 3.3)
    Precision / Recall / F1 per analyst over their predicted slots, then
    macro-averaged -- the same shape RESULTS_V2.md reports for the
    deterministic engine, so the two are directly comparable.

  Question text (Section 3.4/3.5)
    Grounding rate: share of generated questions that cleared the Verifier
    rather than falling back to a bare topic.

Scored SEPARATELY for q4fy26 (one quarter past the training cutoff) and
q1fy27 (two quarters past), never averaged into one number -- the original
POC looked fine until exactly that distinction was drawn. The promotion gate
then requires both a mean floor and a bounded spread between them.

One assumption stated out loud: per-analyst scoring is conditioned on knowing
WHO attended the call, since ground truth is per analyst. That is attendance
information, not topic information, and it is the same framing RESULTS_V2.md
uses ("9 validated analysts"). Topic-level ranking metrics use no attendance
information at all.
"""

import statistics

from src.config.settings import (
    TEST_QUARTERS, TRAIN_CUTOFF, PromotionGate, F_BETA, SLOT_EXTRA, SLOT_CAP, paths_for,
)
from src.config.banks import DEFAULT_BANK
from src.data.loader import load_dataset
from src.agentic.run_agentic import build_initial_state
from src.agentic.planning_agent import run_planning_agent
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst
from src.agentic.question_framer import build_evidence_pool
from src.agentic.verifier import grounding_gate
from src.agentic.question_eval import evaluate_analyst_questions
from src.data.upcoming import topic_salience, drill_down_flags, attach_novel_themes


# ── Ground truth ────────────────────────────────────────────────────────────
def ground_truth(quarter: str, graph: dict) -> dict[str, set[str]]:
    """{analyst: set of topics they actually raised} for a quarter."""
    truth: dict[str, set[str]] = {}
    for n in graph["nodes"]:
        if n["type"] != "Question" or n["properties"]["quarter"] != quarter:
            continue
        p = n["properties"]
        topics = {t for t in p["topics"] if t != "General"}
        if topics:
            truth.setdefault(p["analyst"], set()).update(topics)
    return truth


def actual_question_text(quarter: str, graph: dict, analyst: str) -> str:
    """The analyst's own real question text for this quarter, verbatim from
    the transcript -- the ground-truth block question_eval.py decomposes
    into individual concerns and scores our predictions against. Usually one
    Question node per (analyst, quarter); joined just in case there is ever
    more than one."""
    return " ".join(
        n["properties"]["text"] for n in graph["nodes"]
        if n["type"] == "Question" and n["properties"]["quarter"] == quarter
        and n["properties"]["analyst"] == analyst
    )


# ── Metric primitives ──────────────────────────────────────────────────────
def precision_at_k(ranked: list[str], actual: set[str], k: int) -> float | None:
    if not actual:
        return None
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if t in actual) / len(top)


def recall_at_k(ranked: list[str], actual: set[str], k: int) -> float | None:
    if not actual:
        return None
    top = set(ranked[:k])
    return len(top & actual) / len(actual)


def mrr(ranked: list[str], actual: set[str]) -> float | None:
    """Mean reciprocal rank across every topic actually raised: a topic the
    ranking missed entirely contributes 0, so this rewards getting the real
    topics HIGH, not merely present."""
    if not actual:
        return None
    scores = []
    for t in actual:
        scores.append(1.0 / (ranked.index(t) + 1) if t in ranked else 0.0)
    return sum(scores) / len(scores)


def fbeta(p: float, r: float, beta: float = F_BETA) -> float:
    """Recall-weighted F. beta=2 makes recall twice as important as precision,
    which is the stated objective: a topic on the brief that never comes up
    costs prep minutes, a topic that comes up unprepared costs a bad answer."""
    if not (p or r):
        return 0.0
    b2 = beta * beta
    denom = (b2 * p) + r
    return ((1 + b2) * p * r / denom) if denom else 0.0


def prf(predicted: list[str], actual: set[str]) -> dict:
    pset = set(predicted)
    hits = pset & actual
    p = len(hits) / len(pset) if pset else 0.0
    r = len(hits) / len(actual) if actual else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "f2": round(fbeta(p, r), 4),
            "n_predicted": len(pset), "n_actual": len(actual), "n_hits": len(hits),
            "predicted": predicted, "actual": sorted(actual),
            "hits": sorted(hits), "missed": sorted(actual - pset)}


# ── Failure attribution / error taxonomy (Section 6.3) ─────────────────────
#
# Each category names ONE pipeline stage as the explanation, so "we predicted
# badly" turns into "which component caused it" (the Error Researcher's whole
# job -- see research_errors() below). Only reasoning_weakness and
# topic_ranked_low point at logic this codebase can actually change by
# reweighting or re-prompting; the rest point at data/candidate-generation
# gaps, which is exactly why lumping them into one "miss" count was hiding
# where effort should go.
from src.agentic.skills.registry import SKILL_FOR_ERROR_CATEGORY

ERROR_TAXONOMY = {
    "missing_context": "No narration, no disclosed metric, and no one has ever asked "
        "about this before -- the signal never entered the system at all. A data/ingestion "
        "gap, not a reasoning one.",
    "unpredictable": "Genuinely no precedent anywhere and no signal this quarter, for ANY "
        "analyst. A true cold case, not a system failure.",
    "topic_not_candidate": "Signal existed (narration, a disclosed metric, or history) but "
        "the topic never reached the ranked candidate pool at all -- a candidate-generation "
        "gap. This is the exact shape of the Kunal/FCNR miss: strong signal, structurally "
        "invisible to ranking because nothing outside the fixed taxonomy could become a "
        "candidate (see src/data/upcoming.py::attach_novel_themes for the fix targeting "
        "this category specifically).",
    "topic_ranked_low": "The topic WAS scored as a candidate (it appears in composite_scores) "
        "but the composite weights placed it outside the slots that made the final ranking. "
        "A weighting problem -- the one category the Framework Loop (backtest_and_promote_weights, "
        "below) exists to fix.",
    "lag": "The analyst asked about this same topic just last quarter and the system still "
        "failed to carry it forward -- a persistence/momentum gap.",
    "reasoning_weakness": "The topic was globally ranked and/or this analyst has real history "
        "on it, but the per-analyst reweighting in analyst_layer.py still dropped it out of "
        "their slots. The one category that points at the analyst-layer's own logic rather "
        "than upstream data.",
}


def attribute_miss(topic: str, analyst: str, quarter: str, state: dict,
                   overall_ranked: list[str], composite_scores: dict | None = None) -> str:
    """Walk the pipeline in the doc's order and return the FIRST stage that
    explains the miss. Only 'reasoning_weakness' and 'topic_ranked_low' justify
    a prompt/weight change -- the rest point at data or candidate-generation,
    which is the whole reason for classifying instead of just counting misses."""
    composite_scores = composite_scores or {}
    graph = state["graph"]
    anomaly = state["anomaly_scores"]
    prior = set(state["prior_quarters"])

    had_narration = any(
        n["type"] == "NarrationSegment"
        and n["properties"]["quarter"] == quarter
        and topic in n["properties"]["topics"]
        for n in graph["nodes"]
    )
    had_metric = topic in anomaly
    asked_before_by_anyone = any(
        n["type"] == "Question" and n["properties"]["quarter"] in prior
        and topic in n["properties"]["topics"]
        for n in graph["nodes"]
    )
    asked_before_by_them = any(
        n["type"] == "Question" and n["properties"]["quarter"] in prior
        and n["properties"]["analyst"] == analyst
        and topic in n["properties"]["topics"]
        for n in graph["nodes"]
    )

    # 5. No precedent anywhere and no signal this quarter -> genuinely unpredictable
    if not (had_narration or had_metric or asked_before_by_anyone):
        return "unpredictable"
    # 1. Signal never entered the system
    if not (had_narration or had_metric) and not asked_before_by_anyone:
        return "missing_context"
    # 2. Signal existed but the topic never made the candidate list
    if topic not in overall_ranked:
        # Distinguish "never even scored as a candidate" (candidate-generation
        # gap -- topic_not_candidate) from "scored, but the weights ranked it
        # too low to survive" (a weighting problem -- topic_ranked_low).
        return "topic_ranked_low" if topic in composite_scores else "topic_not_candidate"
    # 3. They asked it in the immediately prior quarter and we still dropped it
    prior_list = state["prior_quarters"]
    if prior_list:
        last_q = prior_list[-1]
        asked_last_quarter = any(
            n["type"] == "Question" and n["properties"]["quarter"] == last_q
            and n["properties"]["analyst"] == analyst
            and topic in n["properties"]["topics"]
            for n in graph["nodes"]
        )
        if asked_last_quarter:
            return "lag"
    # 4. Retrieved, current, still ranked out of their slots
    if asked_before_by_them or topic in overall_ranked:
        return "reasoning_weakness"
    return "topic_ranked_low" if topic in composite_scores else "topic_not_candidate"


# ── Per-quarter evaluation ─────────────────────────────────────────────────
def evaluate_quarter(quarter: str, holdout: bool = True,
                     with_questions: bool = False, upcoming: dict | None = None,
                     slot_extra: int | None = None, slot_cap: int | None = None,
                     score_question_recall: bool = False,
                     analysts: list[str] | None = None,
                     bank_id: str = DEFAULT_BANK,
                     use_cross_bank_signal: bool = False,
                     use_sentiment_signal: bool = False,
                     use_adaptive_decay: bool = True,
                     use_peer_signal: bool = False,
                     allow_cross_bank_pullin: bool = False) -> dict:
    """Run the agentic pipeline for one quarter under held-out conditions and
    score it. with_questions=True also runs the Question Framer + Verifier to
    measure grounding rate (costs LLM calls); default False keeps topic
    scoring cheap and deterministic.

    score_question_recall=True (requires with_questions=True) additionally
    judges each analyst's FRAMED QUESTION TEXT against their real question
    for the quarter via question_eval.evaluate_analyst_questions -- topic
    recall answers "was the bucket on the brief", this answers "did we
    anticipate what they actually asked". Costs one extra LLM judge call per
    decomposed concern, on top of the framing/grounding calls.

    analysts, if given, restricts the ENTIRE per-analyst loop (framing +
    grounding + question recall) to just those names -- a scoping knob added
    2026-09 after a full-held-out-set question-recall run got throttled to a
    crawl by OpenRouter's free-tier 20 RPM/50-per-day ceiling. Cuts LLM call
    volume roughly in proportion to len(analysts)/len(scored_analysts).
    Topic-ranking metrics (precision_at_k etc.) are unaffected -- they don't
    depend on the per-analyst loop -- but per-analyst macro/coverage numbers
    then describe only the filtered subset, not the full held-out set, so
    callers must not treat a scoped run's macro numbers as the headline.

    use_cross_bank_signal=True (default False) blends each active analyst's
    topic history on OTHER registered banks into their own-bank preference
    before scoring -- see src/signals/cross_bank_persona.py's module
    docstring for the empirical case and the pending-legal-sign-off caveat
    this stays opt-in for. Passed straight through to build_initial_state.

    use_sentiment_signal=True (default False) lets each active analyst's
    leak-free running-average sentiment add ONE extra, non-displacing
    candidate slot (the most anomalous topic they have real history on) --
    see analyst_layer.sentiment_extra_slot's docstring for the mechanism and
    this module's rank_position_calibration-adjacent backtest for whether it
    actually helps.

    use_adaptive_decay=True (DEFAULT, promoted 2026-09 -- see
    run_agentic.build_initial_state's docstring for the measured backtest
    numbers that promoted it) replaces the single global EngineConfig.DECAY
    with a per-analyst rate derived from that analyst's own topic-repeat
    propensity -- see src.memory.history.adaptive_decay_for_analyst's
    docstring for the mechanism. Pass False explicitly to reproduce the old
    global-decay behavior for comparison.

    use_peer_signal=True (default False -- opt-in until a backtest
    promotes it, see src.agentic.analyst_layer.peer_extra_slot's docstring)
    lets a peer-bank-salient topic add ONE extra, non-displacing candidate
    slot per analyst. UNLIKE the news/macro signals this one IS meaningful
    to score here: a historical quarter's peer file (if computed for it --
    see src.signals.peer_signal) reflects that same quarter's real,
    already-reported peer transcripts, not "right now".

    allow_cross_bank_pullin=True (default False, requires
    use_cross_bank_signal=True too -- see run_agentic.build_initial_state's
    docstring and claude/cross-bank-candidate-pool-scope.md) lets a
    cross-bank-salient topic enter the candidate pool even when nothing in
    this bank's own signals put it there, rather than only reordering an
    already-present topic. This is the ONE place in the whole codebase
    that can set it to True beyond an ad-hoc script -- no route, no
    frontend toggle, by design, until Section 8's sign-off lands.

    Passed straight through to build_initial_state."""
    state = build_initial_state(quarter, probe=with_questions, holdout=holdout, upcoming=upcoming,
                                bank_id=bank_id, use_cross_bank_signal=use_cross_bank_signal,
                                use_sentiment_signal=use_sentiment_signal,
                                use_adaptive_decay=use_adaptive_decay,
                                use_peer_signal=use_peer_signal,
                                allow_cross_bank_pullin=allow_cross_bank_pullin)
    bundles, tool_log = run_planning_agent(
        state["anomaly_scores"], state["graph"], state["prior_quarters"], state["global_rate"]
    )
    overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                   state["momentum"], state["client"], disclosure=upcoming,
                                   bank_name=state.get("bank_name", "Axis Bank"))
    ranked = overall["ranked_topics"]

    truth = ground_truth(quarter, state["graph"])
    scored_analysts = [a for a in state["active_analysts"] if a in truth]
    all_actual_topics = set().union(*truth.values()) if truth else set()

    # Topic ranking metrics above are computed from `ranked`/`truth` only, so
    # this filter (applied just before the expensive per-analyst loop) never
    # touches them -- only per_analyst/macro/coverage/question_recall narrow
    # to the requested subset.
    if analysts:
        scored_analysts = [a for a in scored_analysts if a in analysts]

    # Topic-level ranking quality (no attendance information used)
    topic_metrics = {
        "precision_at_5": precision_at_k(ranked, all_actual_topics, 5),
        "precision_at_10": precision_at_k(ranked, all_actual_topics, 10),
        "recall_at_5": recall_at_k(ranked, all_actual_topics, 5),
        "recall_at_10": recall_at_k(ranked, all_actual_topics, 10),
        "mrr": mrr(ranked, all_actual_topics),
        "ranked_topics": ranked,
        "actual_topics": sorted(all_actual_topics),
    }

    # Per-analyst prediction quality
    per_analyst, failures = {}, []
    grounded_slots = total_slots = 0
    question_recall_detail = {}
    for analyst in scored_analysts:
        pref, N = state["analyst_prefs"][analyst]
        predicted = reweight_for_analyst(analyst, ranked, pref, N, disclosure=upcoming,
                                         slot_extra=slot_extra, slot_cap=slot_cap,
                                         sentiment_score=state.get("sentiment_scores", {}).get(analyst),
                                         anomaly_scores=state["anomaly_scores"],
                                         use_sentiment_signal=use_sentiment_signal,
                                         peer_salience=state.get("peer_salience"),
                                         use_peer_signal=use_peer_signal,
                                         cross_bank_pref=state.get("cross_bank_prefs", {}).get(analyst),
                                         allow_cross_bank_pullin=state.get("cross_bank_pullin_enabled", False))
        score = prf(predicted, truth[analyst])
        per_analyst[analyst] = score

        for missed in score["missed"]:
            failures.append({
                "analyst": analyst, "topic": missed,
                "category": attribute_miss(missed, analyst, quarter, state, ranked,
                                           composite_scores=overall.get("composite_scores", {})),
            })

        if with_questions:
            style = state["persona_stats"].get(analyst, {}).get("style_note", "no measured style profile")
            pool = build_evidence_pool(analyst, predicted, state["graph"],
                                       set(state["prior_quarters"]),
                                       state["global_rate"], state["anomaly_scores"],
                                       disclosed_metrics=state.get("val_quarter_metrics"),
                                       target_quarter=quarter,
                                       disclosure_text=(upcoming or {}).get("narration", ""),
                                       bank_id=bank_id)
            results, _ = grounding_gate(analyst, style, predicted, pool, state["client"],
                                        target_quarter=quarter,
                                        bank_name=state.get("bank_name", "Axis Bank"))
            total_slots += len(results)
            grounded_slots += sum(1 for r in results if r["status"] == "grounded")

            if score_question_recall:
                actual_block = actual_question_text(quarter, state["graph"], analyst)
                predicted_qs = [r["question_text"] for r in results if r.get("question_text")]
                if actual_block and predicted_qs:
                    question_recall_detail[analyst] = evaluate_analyst_questions(
                        actual_block, predicted_qs, state["client"])

    def _macro(key):
        vals = [s[key] for s in per_analyst.values()]
        return round(sum(vals) / len(vals), 4) if vals else None

    # Micro/coverage: pooled over (analyst, topic) pairs rather than averaged
    # per analyst. This is the number to quote as "management walks in prepped
    # for X of Y topics that actually got raised".
    tot_pred = sum(s["n_predicted"] for s in per_analyst.values())
    tot_actual = sum(s["n_actual"] for s in per_analyst.values())
    tot_hits = sum(s["n_hits"] for s in per_analyst.values())
    micro_p = round(tot_hits / tot_pred, 4) if tot_pred else 0.0
    micro_r = round(tot_hits / tot_actual, 4) if tot_actual else 0.0

    fail_counts = {}
    for f in failures:
        fail_counts[f["category"]] = fail_counts.get(f["category"], 0) + 1

    question_recall_macro = None
    if question_recall_detail:
        judged = [d["question_recall"] for d in question_recall_detail.values()
                 if d["question_recall"] is not None]
        question_recall_macro = round(sum(judged) / len(judged), 4) if judged else None

    return {
        "quarter": quarter,
        "bank_id": bank_id,
        "holdout": holdout,
        "train_cutoff": state.get("train_cutoff"),
        "cutoff_mode": state.get("cutoff_mode"),
        "quarters_ahead_of_cutoff": (
            state["quarter_order"].index(quarter) - state["quarter_order"].index(state["train_cutoff"])
            if quarter in state["quarter_order"] and state.get("train_cutoff") in state["quarter_order"]
            else None
        ),
        "n_analysts_scored": len(scored_analysts),
        "topic_ranking": topic_metrics,
        "per_analyst": per_analyst,
        "macro": {
            "precision": _macro("precision"),
            "recall": _macro("recall"),
            "f1": _macro("f1"),
            "f2": _macro("f2"),
        },
        "coverage": {
            "topics_predicted": tot_pred,
            "topics_actually_raised": tot_actual,
            "topics_covered": tot_hits,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f2": round(fbeta(micro_p, micro_r), 4),
        },
        "slot_policy": {"extra": SLOT_EXTRA if slot_extra is None else slot_extra,
                        "cap": SLOT_CAP if slot_cap is None else slot_cap},
        "grounding_rate": (round(grounded_slots / total_slots, 4)
                           if with_questions and total_slots else None),
        "failure_attribution": {
            "counts": fail_counts,
            "total_misses": len(failures),
            "detail": failures,
        },
        "planning_tool_calls": len([t for t in tool_log if t["status"] == "called"]),
        "disclosure_conditioned": bool(upcoming),
        "composite_scores": overall.get("composite_scores", {}),
        "question_recall": ({"macro": question_recall_macro, "per_analyst": question_recall_detail}
                            if score_question_recall else None),
        "analyst_scope": sorted(scored_analysts) if analysts else None,
    }


# ── Held-out test set: both quarters, separately, plus the gate ────────────
def run_holdout_eval(with_questions: bool = False, score_question_recall: bool = False,
                     analysts: list[str] | None = None,
                     quarters: list[str] | None = None,
                     bank_id: str = DEFAULT_BANK,
                     use_cross_bank_signal: bool = False,
                     use_sentiment_signal: bool = False,
                     use_adaptive_decay: bool = True,
                     use_peer_signal: bool = False,
                     allow_cross_bank_pullin: bool = False) -> dict:
    """The headline result: q4fy26 and q1fy27 scored independently against a
    training cutoff of q3fy26, with the spread between them made explicit.

    score_question_recall=True (forces with_questions=True) additionally
    scores question-level recall -- see evaluate_quarter's docstring.

    analysts/quarters narrow the run for a cheap, targeted check (e.g. one
    analyst on one quarter) instead of the full held-out set -- see
    evaluate_quarter's docstring for why this exists and what it does NOT
    change (topic-ranking metrics stay full-set; only per-analyst numbers
    narrow). Only meant for ad-hoc verification; the promotion-gate checks
    below still run against whatever `results` this scoping produces, so a
    scoped run's gate verdict should not be read as the real gate result.

    use_cross_bank_signal/use_sentiment_signal/use_adaptive_decay/
    use_peer_signal are passed straight through to evaluate_quarter -- see
    its docstring for each. allow_cross_bank_pullin is documented on
    evaluate_quarter too -- this is the one entry point meant to set it."""
    with_questions = with_questions or score_question_recall
    results = [evaluate_quarter(q, holdout=True, with_questions=with_questions,
                                score_question_recall=score_question_recall,
                                analysts=analysts, bank_id=bank_id,
                                use_cross_bank_signal=use_cross_bank_signal,
                                use_sentiment_signal=use_sentiment_signal,
                                use_adaptive_decay=use_adaptive_decay,
                                use_peer_signal=use_peer_signal,
                                allow_cross_bank_pullin=allow_cross_bank_pullin)
               for q in (quarters or TEST_QUARTERS)]

    # None entries happen when a quarter has zero scored analysts (e.g. a
    # newly-registered, thin-history bank being run against the shared
    # TEST_QUARTERS default, which wasn't chosen with it in mind -- see
    # evaluate_quarter's _macro() helper). Filtered out here rather than
    # crashing the whole holdout run; a bank in that state should pass its
    # OWN quarters via the `quarters` param instead of relying on this
    # aggregate, but this keeps run_holdout_eval from raising on it.
    f1s = [v for r in results if (v := r["macro"]["f1"]) is not None]
    f2s = [v for r in results if (v := r["macro"]["f2"]) is not None]
    precisions = [v for r in results if (v := r["macro"]["precision"]) is not None]
    recalls = [v for r in results if (v := r["macro"]["recall"]) is not None]
    cov_r = [r["coverage"]["micro_recall"] for r in results]

    mean_p = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    mean_r = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    mean_f2 = round(sum(f2s) / len(f2s), 4) if f2s else 0.0
    spread = round(max(recalls) - min(recalls), 4) if len(recalls) > 1 else 0.0

    # Recall-first gate: recall is floored high, precision only guarded against
    # a degenerate "predict everything" config, stability measured on recall.
    checks = {
        "mean_recall": {"value": mean_r, "threshold": PromotionGate.MIN_MEAN_RECALL,
                        "pass": mean_r >= PromotionGate.MIN_MEAN_RECALL,
                        "note": "the objective — coverage of what actually got asked"},
        "mean_precision_floor": {"value": mean_p, "threshold": PromotionGate.MIN_MEAN_PRECISION,
                                 "pass": mean_p >= PromotionGate.MIN_MEAN_PRECISION,
                                 "note": "guard against predicting everything, not a target"},
        "recall_spread": {"value": spread, "threshold": PromotionGate.MAX_RECALL_SPREAD,
                          "pass": spread <= PromotionGate.MAX_RECALL_SPREAD,
                          "note": "stability one vs two quarters past the cutoff"},
    }
    all_fail_counts = {}
    for r in results:
        for cat, n in r["failure_attribution"]["counts"].items():
            all_fail_counts[cat] = all_fail_counts.get(cat, 0) + n

    qr_macros = [r["question_recall"]["macro"] for r in results
                if r.get("question_recall") and r["question_recall"]["macro"] is not None]
    mean_question_recall = round(sum(qr_macros) / len(qr_macros), 4) if qr_macros else None

    return {
        "bank_id": bank_id,
        "cross_bank_signal": use_cross_bank_signal,
        "cross_bank_pullin": allow_cross_bank_pullin,
        "sentiment_signal": use_sentiment_signal,
        "adaptive_decay": use_adaptive_decay,
        "train_cutoff": {r["quarter"]: r["train_cutoff"] for r in results},
        "cutoff_mode": results[0].get("cutoff_mode") if results else None,
        "test_quarters": quarters or TEST_QUARTERS,
        "per_quarter": results,
        "objective": {"headline": "recall", "fbeta": F_BETA,
                      "note": "Recall-first: over-prediction is cheap, an unprepared "
                              "topic on a recorded call is not."},
        "summary": {
            "mean_recall": mean_r,
            "mean_precision": mean_p,
            "mean_f2": mean_f2,
            "mean_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "recall_spread": spread,
            "recall_stdev": round(statistics.stdev(recalls), 4) if len(recalls) > 1 else 0.0,
            "mean_coverage_recall": round(sum(cov_r) / len(cov_r), 4) if cov_r else 0.0,
            "total_topics_covered": sum(r["coverage"]["topics_covered"] for r in results),
            "total_topics_raised": sum(r["coverage"]["topics_actually_raised"] for r in results),
            "mean_question_recall": mean_question_recall,
            "total_topics_predicted": sum(r["coverage"]["topics_predicted"] for r in results),
        },
        "promotion_gate": {
            "checks": checks,
            "verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
        },
        "failure_attribution_totals": all_fail_counts,
    }


# ── Error Researcher (Section 8/9 of the harness-engineering brief) ────────
#
# Its job is explicitly NOT to predict better questions -- it looks at
# everything the holdout eval already missed and asks "which component
# caused this", using the taxonomy above. No LLM call: it is pure
# aggregation over data evaluate_quarter() already computed, so it's free
# to run after every eval and never adds to the Groq free-tier budget.
def research_errors(holdout_result: dict | None = None, top_n: int = 5) -> dict:
    """Aggregates failure_attribution across the held-out quarters into a
    framework-level diagnosis: which error category dominates, and which
    specific (analyst, topic) pairs are driving it -- the concrete, actionable
    output the brief's 'Error Researcher' and error-taxonomy sections call for."""
    holdout_result = holdout_result if holdout_result is not None else run_holdout_eval()

    all_failures = []
    for r in holdout_result["per_quarter"]:
        for f in r["failure_attribution"]["detail"]:
            all_failures.append({**f, "quarter": r["quarter"]})

    total = len(all_failures)
    counts: dict[str, int] = {}
    by_category_examples: dict[str, list[dict]] = {}
    topic_counts_by_category: dict[str, dict[str, int]] = {}
    for f in all_failures:
        cat = f["category"]
        counts[cat] = counts.get(cat, 0) + 1
        by_category_examples.setdefault(cat, []).append(f)
        topic_counts_by_category.setdefault(cat, {})
        topic_counts_by_category[cat][f["topic"]] = topic_counts_by_category[cat].get(f["topic"], 0) + 1

    breakdown = []
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        top_topics = sorted(topic_counts_by_category[cat].items(), key=lambda kv: -kv[1])[:top_n]
        breakdown.append({
            "category": cat,
            "description": ERROR_TAXONOMY.get(cat, "(no description on file)"),
            "count": n,
            "share": round(n / total, 4) if total else 0.0,
            "actionable": cat in ("reasoning_weakness", "topic_ranked_low"),
            "skill": SKILL_FOR_ERROR_CATEGORY.get(cat),
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
            "examples": [{"analyst": e["analyst"], "topic": e["topic"], "quarter": e["quarter"]}
                        for e in by_category_examples[cat][:top_n]],
        })

    actionable_n = sum(b["count"] for b in breakdown if b["actionable"])
    diagnosis = None
    if breakdown:
        top = breakdown[0]
        diagnosis = (f"{top['count']} of {total} missed predictions ({top['share']:.0%}) are "
                    f"'{top['category']}': {top['description']}")
        if not top["actionable"]:
            diagnosis += (" This is NOT fixable by reweighting or re-prompting -- it points at "
                         "upstream data/candidate-generation, so a framework-weight change would "
                         "not move this number.")
        elif top["skill"]:
            diagnosis += f" The implicated skill is '{top['skill']}'."

    return {
        "total_misses": total,
        "actionable_misses": actionable_n,
        "actionable_share": round(actionable_n / total, 4) if total else 0.0,
        "breakdown": breakdown,
        "diagnosis": diagnosis,
        "taxonomy": ERROR_TAXONOMY,
        "scored_quarters": [r["quarter"] for r in holdout_result["per_quarter"]],
    }


# ── Framework Loop: backtest a candidate weight set before it can be promoted
#
# "The harness proposes changes; evaluation decides whether they deserve
# promotion." Never edits overall_layer.py's DEFAULT_WEIGHTS itself -- it runs
# the SAME held-out quarters twice (current defaults vs the candidate) and
# reuses the existing PromotionGate thresholds so a candidate is judged by
# the identical bar production already has to clear.
def backtest_and_promote_weights(candidate_weights: dict[str, float],
                                 with_questions: bool = False,
                                 bank_id: str = DEFAULT_BANK) -> dict:
    from src.agentic.overall_layer import DEFAULT_WEIGHTS
    from src.config.settings import PromotionGate

    def _run(weights):
        results = [evaluate_quarter_with_weights(q, weights, with_questions=with_questions,
                                                  bank_id=bank_id)
                  for q in TEST_QUARTERS]
        recalls = [r["macro"]["recall"] for r in results]
        precisions = [r["macro"]["precision"] for r in results]
        mean_r = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        mean_p = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
        spread = round(max(recalls) - min(recalls), 4) if len(recalls) > 1 else 0.0
        return {"per_quarter": results, "mean_recall": mean_r, "mean_precision": mean_p,
                "recall_spread": spread}

    baseline = _run(DEFAULT_WEIGHTS)
    candidate = _run({**DEFAULT_WEIGHTS, **candidate_weights})

    checks = {
        "recall_not_worse": {
            "pass": candidate["mean_recall"] >= baseline["mean_recall"],
            "baseline": baseline["mean_recall"], "candidate": candidate["mean_recall"],
        },
        "meets_recall_floor": {
            "pass": candidate["mean_recall"] >= PromotionGate.MIN_MEAN_RECALL,
            "threshold": PromotionGate.MIN_MEAN_RECALL, "candidate": candidate["mean_recall"],
        },
        "meets_precision_floor": {
            "pass": candidate["mean_precision"] >= PromotionGate.MIN_MEAN_PRECISION,
            "threshold": PromotionGate.MIN_MEAN_PRECISION, "candidate": candidate["mean_precision"],
        },
        "spread_not_worse": {
            "pass": candidate["recall_spread"] <= max(baseline["recall_spread"], PromotionGate.MAX_RECALL_SPREAD),
            "baseline": baseline["recall_spread"], "candidate": candidate["recall_spread"],
        },
    }
    verdict = "PROMOTE" if all(c["pass"] for c in checks.values()) else "REJECT"

    return {
        "candidate_weights": {**DEFAULT_WEIGHTS, **candidate_weights},
        "baseline_weights": DEFAULT_WEIGHTS,
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "verdict": verdict,
        "note": "Scored on the SAME held-out quarters as production (TEST_QUARTERS) using the "
                "existing PromotionGate thresholds -- this never edits production weights itself, "
                "it only tells you whether the candidate would clear the bar production already has to.",
    }


def synthetic_disclosure_for_quarter(quarter: str, graph: dict) -> dict:
    """Builds a disclosure dict (topic_salience/drill_down_flags/novel_themes)
    from a held-out quarter's OWN actual prepared-remarks narration, using the
    exact same deterministic parsing an uploaded document would get. This is
    not leakage -- it is that quarter's real script, the same text
    question_framer.py already reads for phrasing even when no disclosure was
    uploaded (src/agentic/question_framer.py::_narration_for_topic) -- it is
    only NEW here in that it also feeds the Overall layer's ranking signals
    (topic_salience/drill_down_flags), not just question phrasing.

    Exists so backtest_and_promote_weights() can actually exercise the
    disclosure/drill_flag composite weights: plain run_holdout_eval() never
    passes an `upcoming` disclosure at all, so those two weights multiply by
    zero and a candidate that only touches them would score identically to
    the baseline -- silently, not because the candidate has no effect."""
    text = "\n".join(
        n["properties"]["text"] for n in graph["nodes"]
        if n["type"] == "NarrationSegment" and n["properties"]["quarter"] == quarter
    )
    if not text.strip():
        return {"topic_salience": {}, "drill_down_flags": [], "narration": ""}
    salience = topic_salience(text)
    flags = drill_down_flags(text)
    salience, flags, _ = attach_novel_themes(salience, flags)
    return {"topic_salience": salience, "drill_down_flags": flags, "narration": text}


def evaluate_quarter_with_weights(quarter: str, weights: dict[str, float],
                                  with_questions: bool = False,
                                  use_synthetic_disclosure: bool = True,
                                  bank_id: str = DEFAULT_BANK) -> dict:
    """Same as evaluate_quarter(), but injecting an alternate composite-weight
    set into the Overall layer instead of the module defaults -- the one extra
    hook backtest_and_promote_weights needs that evaluate_quarter() doesn't
    expose, since production call sites should never need to pass weights.

    use_synthetic_disclosure=True (the default here, unlike production) feeds
    that quarter's own real narration through synthetic_disclosure_for_quarter
    so a weight change on disclosure/drill_flag is actually exercised by the
    backtest instead of scoring identically to baseline by construction."""
    state = build_initial_state(quarter, probe=with_questions, holdout=True, bank_id=bank_id)
    disclosure = synthetic_disclosure_for_quarter(quarter, state["graph"]) if use_synthetic_disclosure else None
    bundles, _ = run_planning_agent(
        state["anomaly_scores"], state["graph"], state["prior_quarters"], state["global_rate"]
    )
    overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                   state["momentum"], state["client"], disclosure=disclosure,
                                   weights=weights, bank_name=state.get("bank_name", "Axis Bank"))
    ranked = overall["ranked_topics"]
    truth = ground_truth(quarter, state["graph"])
    scored_analysts = [a for a in state["active_analysts"] if a in truth]

    per_analyst = {}
    for analyst in scored_analysts:
        pref, N = state["analyst_prefs"][analyst]
        predicted = reweight_for_analyst(analyst, ranked, pref, N, disclosure=disclosure)
        per_analyst[analyst] = prf(predicted, truth[analyst])

    def _macro(key):
        vals = [s[key] for s in per_analyst.values()]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {"quarter": quarter, "macro": {"precision": _macro("precision"), "recall": _macro("recall"),
                                          "f1": _macro("f1"), "f2": _macro("f2")}}


if __name__ == "__main__":
    import json
    print(json.dumps(run_holdout_eval(), indent=2, default=str))


# ── Choosing the slot policy honestly ──────────────────────────────────────
def sweep_slot_policy(configs: list[tuple[int, int]] | None = None,
                      score_quarters: list[str] | None = None) -> dict:
    """Walk-forward sweep over TRAINING quarters only, so the operating point
    is chosen without ever looking at q4fy26/q1fy27.

    Each scored quarter is predicted using only the quarters before it, which
    is the same as-of discipline the held-out evaluation uses -- just applied
    inside the training range.
    """
    from src.agentic.run_agentic import build_initial_state as _bis
    configs = configs or [(0, 5), (1, 5), (2, 6), (3, 7), (4, 8)]
    order = None
    results: dict[str, dict] = {}

    # Score the last stretch of training quarters -- enough prior history for a
    # profile to exist, and never a test quarter.
    if score_quarters is None:
        from src.data.loader import load_dataset
        order = [q["quarter_id"] for q in load_dataset()]
        cutoff = order.index(TRAIN_CUTOFF)
        score_quarters = order[cutoff - 7:cutoff + 1]

    # Build each quarter's state and ranking ONCE, then vary only the slots.
    prepared = []
    for q in score_quarters:
        state = _bis(q, probe=False, holdout=False)
        bundles, _ = run_planning_agent(state["anomaly_scores"], state["graph"],
                                        state["prior_quarters"], state["global_rate"])
        overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                       state["momentum"], state["client"],
                                       bank_name=state.get("bank_name", "Axis Bank"))
        truth = ground_truth(q, state["graph"])
        prepared.append((q, state, overall["ranked_topics"], truth))

    for extra, cap in configs:
        per_q = []
        tot_pred = tot_actual = tot_hits = 0
        for q, state, ranked, truth in prepared:
            scored = [a for a in state["active_analysts"] if a in truth]
            if not scored:
                continue
            rs, ps = [], []
            for a in scored:
                pref, N = state["analyst_prefs"][a]
                pred = reweight_for_analyst(a, ranked, pref, N, slot_extra=extra, slot_cap=cap)
                sc = prf(pred, truth[a])
                rs.append(sc["recall"]); ps.append(sc["precision"])
                tot_pred += sc["n_predicted"]; tot_actual += sc["n_actual"]; tot_hits += sc["n_hits"]
            per_q.append({"quarter": q,
                          "recall": round(sum(rs) / len(rs), 4),
                          "precision": round(sum(ps) / len(ps), 4)})
        mr = round(sum(x["recall"] for x in per_q) / len(per_q), 4) if per_q else 0.0
        mp = round(sum(x["precision"] for x in per_q) / len(per_q), 4) if per_q else 0.0
        results[f"extra={extra},cap={cap}"] = {
            "slot_extra": extra, "slot_cap": cap,
            "mean_recall": mr, "mean_precision": mp, "mean_f2": round(fbeta(mp, mr), 4),
            "coverage_recall": round(tot_hits / tot_actual, 4) if tot_actual else 0.0,
            "topics_predicted": tot_pred, "topics_raised": tot_actual, "topics_covered": tot_hits,
            "per_quarter": per_q,
        }

    best = max(results.items(), key=lambda kv: kv[1]["mean_f2"])
    return {"scored_quarters": score_quarters, "train_cutoff": TRAIN_CUTOFF,
            "configs": results,
            "chosen_by_f2": {"config": best[0], **best[1]},
            "note": "Chosen on training quarters only; the held-out quarters were not "
                    "consulted in this sweep."}


def rank_position_calibration(bank_id: str = DEFAULT_BANK, warmup_quarters: int = 4,
                              max_rank: int = 8) -> dict:
    """Historical hit-rate by rank position -- the calibration number the
    Prepare-next-call view's predicted-question list needs to show a real
    "how often is the #1 predicted topic actually right" figure instead of
    a fabricated confidence score (see boot()'s comment in the frontend on
    why a made-up confidence number was deliberately avoided).

    Deliberately walk-forward across EVERY quarter this bank has enough
    training history for, not just the two official TEST_QUARTERS -- both
    use the same leak-free rolling cutoff (build_initial_state's default
    cutoff_mode="rolling": training on quarters strictly before the target,
    same as every other holdout number this platform reports), but
    TEST_QUARTERS alone is only 2 quarters x a handful of analysts, too few
    analyst-quarter observations for a believable per-rank breakdown.
    warmup_quarters skips the earliest quarters, where "training history"
    is too thin to have produced a meaningful analyst preference yet, before
    starting the walk-forward window.

    Free and deterministic (with_questions=False, no LLM calls) -- same cost
    profile as the topic-ranking metrics already computed for every
    Evaluation-tab load, just run across more quarters.

    Returns {"by_rank": [{"rank": 1, "hit_rate": 0.72, "n": 34}, ...],
             "quarters_used": [...], "warmup_skipped": [...],
             "note": "..."}. hit_rate is null (not 0.0) for a rank with n=0
    (e.g. slot_cap trims most analysts' lists before that rank is reached)
    -- a rank that was never predicted has no observed hit rate, and that is
    a different fact from "predicted and always wrong"."""
    paths = paths_for(bank_id)
    dataset = load_dataset(dataset_path=paths.dataset_path)
    quarter_order = [q["quarter_id"] for q in dataset]

    warmup = quarter_order[:warmup_quarters]
    eligible = quarter_order[warmup_quarters:]

    hit_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    quarters_used = []

    for q in eligible:
        result = evaluate_quarter(q, holdout=True, with_questions=False, bank_id=bank_id)
        per_analyst = result.get("per_analyst") or {}
        if not per_analyst:
            continue
        quarters_used.append(q)
        for score in per_analyst.values():
            predicted = score.get("predicted") or []
            actual = set(score.get("actual") or [])
            for i, topic in enumerate(predicted[:max_rank]):
                rank = i + 1
                total_counts[rank] = total_counts.get(rank, 0) + 1
                if topic in actual:
                    hit_counts[rank] = hit_counts.get(rank, 0) + 1

    by_rank = []
    for rank in range(1, max_rank + 1):
        n = total_counts.get(rank, 0)
        hit_rate = round(hit_counts.get(rank, 0) / n, 4) if n else None
        by_rank.append({"rank": rank, "hit_rate": hit_rate, "n": n})

    return {
        "by_rank": by_rank,
        "quarters_used": quarters_used,
        "warmup_skipped": warmup,
        "note": "Walk-forward, leak-free (same rolling cutoff as every other holdout "
                "number here): each quarter's hit/miss only used training data strictly "
                "before it. hit_rate is the share of analyst-quarter observations where "
                "the topic predicted AT THIS RANK was actually raised by that analyst "
                "that quarter -- not a per-topic accuracy, a per-POSITION one.",
    }
