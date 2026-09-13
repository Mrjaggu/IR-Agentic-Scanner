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
    TEST_QUARTERS, TRAIN_CUTOFF, PromotionGate, F_BETA, SLOT_EXTRA, SLOT_CAP,
)
from src.agentic.run_agentic import build_initial_state
from src.agentic.planning_agent import run_planning_agent
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst
from src.agentic.question_framer import build_evidence_pool
from src.agentic.verifier import grounding_gate


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


# ── Failure attribution (Section 6.3) ──────────────────────────────────────
def attribute_miss(topic: str, analyst: str, quarter: str, state: dict,
                   overall_ranked: list[str]) -> str:
    """Walk the pipeline in the doc's order and return the FIRST stage that
    explains the miss. Only 'reasoning_weakness' justifies a prompt change --
    the rest point at data or retrieval, which is the whole reason for
    classifying instead of just counting misses."""
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
        return "retrieval_gap"
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
    return "retrieval_gap"


# ── Per-quarter evaluation ─────────────────────────────────────────────────
def evaluate_quarter(quarter: str, holdout: bool = True,
                     with_questions: bool = False, upcoming: dict | None = None,
                     slot_extra: int | None = None, slot_cap: int | None = None) -> dict:
    """Run the agentic pipeline for one quarter under held-out conditions and
    score it. with_questions=True also runs the Question Framer + Verifier to
    measure grounding rate (costs LLM calls); default False keeps topic
    scoring cheap and deterministic."""
    state = build_initial_state(quarter, probe=with_questions, holdout=holdout, upcoming=upcoming)
    bundles, tool_log = run_planning_agent(
        state["anomaly_scores"], state["graph"], state["prior_quarters"], state["global_rate"]
    )
    overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                   state["momentum"], state["client"], disclosure=upcoming)
    ranked = overall["ranked_topics"]

    truth = ground_truth(quarter, state["graph"])
    scored_analysts = [a for a in state["active_analysts"] if a in truth]
    all_actual_topics = set().union(*truth.values()) if truth else set()

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
    for analyst in scored_analysts:
        pref, N = state["analyst_prefs"][analyst]
        predicted = reweight_for_analyst(analyst, ranked, pref, N, disclosure=upcoming,
                                         slot_extra=slot_extra, slot_cap=slot_cap)
        score = prf(predicted, truth[analyst])
        per_analyst[analyst] = score

        for missed in score["missed"]:
            failures.append({
                "analyst": analyst, "topic": missed,
                "category": attribute_miss(missed, analyst, quarter, state, ranked),
            })

        if with_questions:
            style = state["persona_stats"].get(analyst, {}).get("style_note", "no measured style profile")
            pool = build_evidence_pool(analyst, predicted, state["graph"],
                                       set(state["prior_quarters"]),
                                       state["global_rate"], state["anomaly_scores"],
                                       disclosed_metrics=state.get("val_quarter_metrics"),
                                       target_quarter=quarter,
                                       disclosure_text=(upcoming or {}).get("narration", ""))
            results, _ = grounding_gate(analyst, style, predicted, pool, state["client"],
                                        target_quarter=quarter)
            total_slots += len(results)
            grounded_slots += sum(1 for r in results if r["status"] == "grounded")

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

    return {
        "quarter": quarter,
        "holdout": holdout,
        "train_cutoff": state.get("train_cutoff"),
        "cutoff_mode": state.get("cutoff_mode"),
        "quarters_ahead_of_cutoff": (
            state["quarter_order"].index(quarter) - state["quarter_order"].index(state["train_cutoff"])
            if state.get("train_cutoff") in state["quarter_order"] else None
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
    }


# ── Held-out test set: both quarters, separately, plus the gate ────────────
def run_holdout_eval(with_questions: bool = False) -> dict:
    """The headline result: q4fy26 and q1fy27 scored independently against a
    training cutoff of q3fy26, with the spread between them made explicit."""
    results = [evaluate_quarter(q, holdout=True, with_questions=with_questions)
               for q in TEST_QUARTERS]

    f1s = [r["macro"]["f1"] for r in results]
    f2s = [r["macro"]["f2"] for r in results]
    precisions = [r["macro"]["precision"] for r in results]
    recalls = [r["macro"]["recall"] for r in results]
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

    return {
        "train_cutoff": {r["quarter"]: r["train_cutoff"] for r in results},
        "cutoff_mode": results[0].get("cutoff_mode") if results else None,
        "test_quarters": TEST_QUARTERS,
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
            "total_topics_predicted": sum(r["coverage"]["topics_predicted"] for r in results),
        },
        "promotion_gate": {
            "checks": checks,
            "verdict": "PASS" if all(c["pass"] for c in checks.values()) else "FAIL",
        },
        "failure_attribution_totals": all_fail_counts,
    }


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
                                       state["momentum"], state["client"])
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
