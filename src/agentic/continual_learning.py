"""
continual_learning.py -- the missing "propose" and "human review gate" steps
of the continual-learning loop (see claude/continual-learning-loop-scope.md
for the full design, what already existed before this file, and exactly
what this first slice does and does not do).

Everything downstream of "propose" already existed and is reused unchanged:
- eval_harness.research_errors() / attribute_miss() for failure attribution
- eval_harness.backtest_and_promote_weights() for the backtest-and-gate step
  (itself extended, additively, with a bootstrap confidence signal -- see
  eval_harness.bootstrap_recall_delta())

This module adds the two pieces that did not exist anywhere:
  1. propose_weight_adjustment() -- turns a held-out run's topic_ranked_low
     misses into ONE concrete, explainable weight-delta candidate, instead
     of a human inventing five numbers by hand in the Framework Loop panel.
  2. A durable, reviewable proposal log (list_proposals/review_proposal) --
     today, a backtest result vanishes the moment the response is read; this
     persists it with a status a human can act on.

Deliberately NOT built here (see the scope doc's "what this deliberately
does not do"): auto-apply on approval (approving a proposal records a
decision, it never writes to overall_layer.py's DEFAULT_WEIGHTS), and
proposals for the LLM-prompt skills (question_framing, verification,
composite_scoring, question_recall) -- only composite-weight changes have a
deterministic, free backtest path today.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

from src.config.banks import DEFAULT_BANK
from src.config.settings import writable_data_dir
from src.agentic.eval_harness import run_holdout_eval, backtest_and_promote_weights
from src.agentic.run_agentic import build_initial_state
from src.agentic.overall_layer import DEFAULT_WEIGHTS

# Only these three composite-score components can be non-zero on a plain
# held-out run (no uploaded disclosure document) -- disclosure/drill_flag
# are structurally zero without one, so proposing a delta on either from a
# bare holdout run would be proposing a fix for a signal that was never
# actually exercised this run. See propose_weight_adjustment() below and the
# scope doc's "bootstrap-replay" section for the same disclosure/drill_flag
# caveat made elsewhere in this codebase.
_HOLDOUT_ACTIONABLE_WEIGHTS = ("anomaly", "momentum", "base_rate")

# Safety cap on any single proposed weight increase -- see the scope doc's
# "what this deliberately does not do": this proposes a modest, inspectable
# nudge for a human to evaluate, never a large, untested swing.
_MAX_WEIGHT_DELTA = 0.15

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _path(bank_id: str) -> str:
    return os.path.join(writable_data_dir("continual_learning"), f"proposals_{bank_id}.json")


def _load(bank_id: str) -> list[dict]:
    path = _path(bank_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _save(bank_id: str, proposals: list[dict]) -> None:
    path = _path(bank_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(proposals, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic on POSIX -- a crash mid-write never corrupts the store


def list_proposals(bank_id: str = DEFAULT_BANK) -> list[dict]:
    """Most recent first -- what the review-gate panel lists."""
    return sorted(_load(bank_id), key=lambda p: p["created_at"], reverse=True)


def review_proposal(bank_id: str, proposal_id: str, decision: str, reviewer: str,
                    note: str | None = None) -> dict:
    """The human review gate. Sets status and records who decided and when --
    does NOT touch overall_layer.py's DEFAULT_WEIGHTS even on approval (see
    module docstring: applying an approved change stays a deliberate,
    separate step an engineer takes after seeing this record). Raises
    KeyError if proposal_id doesn't exist for this bank, ValueError if
    decision isn't a recognized action."""
    if decision not in ("approved", "rejected"):
        raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
    with _lock:
        proposals = _load(bank_id)
        for p in proposals:
            if p["id"] == proposal_id:
                p["status"] = decision
                p["reviewer"] = (reviewer or "").strip() or "unknown"
                p["review_note"] = note
                p["decided_at"] = _now()
                _save(bank_id, proposals)
                return p
    raise KeyError(f"no proposal {proposal_id!r} on file for bank {bank_id!r}")


def _closeable_misses(holdout: dict, bank_id: str) -> list[dict]:
    """Every topic_ranked_low miss in the held-out run, decomposed into which
    of the three holdout-actionable weights could close the gap between its
    actual composite score and the score it needed to survive the Overall
    layer's final cut, and by how much. One build_initial_state() call per
    DISTINCT quarter that has such a miss (cheap, deterministic, no LLM --
    the same construction evaluate_quarter() itself does internally, so this
    is re-derivation of the same state, not a parallel implementation)."""
    by_quarter: dict[str, dict] = {}
    for r in holdout["per_quarter"]:
        misses = [f for f in r["failure_attribution"]["detail"] if f["category"] == "topic_ranked_low"]
        if misses:
            by_quarter[r["quarter"]] = {"misses": misses, "composite_scores": r["composite_scores"],
                                        "ranked_topics": r["topic_ranking"]["ranked_topics"]}

    out = []
    for quarter, info in by_quarter.items():
        state = build_initial_state(quarter, holdout=True, bank_id=bank_id)
        anomaly_scores = state["anomaly_scores"]
        momentum = state["momentum"]
        global_rate = state["global_rate"]
        max_momentum = max(momentum.values()) if momentum else 1.0
        max_rate = max(global_rate.values()) if global_rate else 1.0
        comp = info["composite_scores"]
        ranked = info["ranked_topics"]
        # The lowest score among topics that actually survived to the final
        # ranked list -- the real cut line, whatever produced it, rather than
        # assuming a fixed TOP_K and risking drift from overall_layer.py's own
        # constant.
        threshold = min((comp.get(t, 0.0) for t in ranked), default=0.0)

        for f in info["misses"]:
            topic = f["topic"]
            gap = threshold - comp.get(topic, 0.0)
            if gap <= 0:
                continue  # scored at/above the cutoff despite the category -- not a weight gap
            raw = {
                "anomaly": anomaly_scores.get(topic, 0.0),
                "momentum": (momentum.get(topic, 0.0) / max_momentum) if max_momentum else 0.0,
                "base_rate": (global_rate.get(topic, 0.0) / max_rate) if max_rate else 0.0,
            }
            for weight_name in _HOLDOUT_ACTIONABLE_WEIGHTS:
                raw_val = raw[weight_name]
                if raw_val > 1e-9:
                    out.append({
                        "quarter": quarter, "analyst": f["analyst"], "topic": topic,
                        "weight": weight_name, "gap": round(gap, 4),
                        "required_increase": round(gap / raw_val, 4),
                    })
    return out


def propose_weight_adjustment(bank_id: str = DEFAULT_BANK, with_questions: bool = False) -> dict:
    """The missing 'propose' step: turns a held-out run's topic_ranked_low
    misses into ONE concrete composite-weight delta, backtests it with the
    EXISTING backtest_and_promote_weights() (bootstrap confidence included),
    and logs the full proposal with status=pending_review for a human to
    act on via review_proposal().

    Selection rule (see claude/continual-learning-loop-scope.md): for each of
    the three holdout-actionable weights, find every miss it could close
    within _MAX_WEIGHT_DELTA, and pick the weight that closes the most misses
    -- ties broken by the smaller required increase. This is a real,
    inspectable computation from this run's own evidence, not a five-way
    guess or an LLM invention.

    Returns a dict with a top-level "result" key (separate from the
    persisted record's own "status", which tracks review-gate state and
    would otherwise collide with this one -- see below) set to one of:
      "no_actionable_misses"  -- no topic_ranked_low misses this run (the
                                 honest null result -- see attribute_miss's
                                 docstring for why the other five categories
                                 are not weight-fixable at all)
      "no_closeable_misses"   -- topic_ranked_low misses exist, but none are
                                 closeable by any single weight within the cap
      "proposed"              -- a candidate was generated, backtested, and
                                 logged with status=pending_review
    """
    holdout = run_holdout_eval(with_questions=with_questions, bank_id=bank_id)
    misses = _closeable_misses(holdout, bank_id)

    total_topic_ranked_low = sum(
        1 for r in holdout["per_quarter"]
        for f in r["failure_attribution"]["detail"] if f["category"] == "topic_ranked_low"
    )
    if total_topic_ranked_low == 0:
        return {"result": "no_actionable_misses", "bank_id": bank_id,
                "message": "No topic_ranked_low misses in the current held-out run -- "
                           "nothing for a weight change to fix. See research_errors() for "
                           "what IS driving misses, if any."}

    by_weight: dict[str, list[dict]] = {}
    for m in misses:
        if m["required_increase"] <= _MAX_WEIGHT_DELTA:
            by_weight.setdefault(m["weight"], []).append(m)

    if not by_weight:
        return {"result": "no_closeable_misses", "bank_id": bank_id,
                "total_topic_ranked_low_misses": total_topic_ranked_low,
                "message": f"{total_topic_ranked_low} topic_ranked_low miss(es) this run, but none "
                           f"closeable by a single weight within the +{_MAX_WEIGHT_DELTA} safety cap -- "
                           "proposing nothing rather than a change too large to trust unverified."}

    # Pick the weight helping the most misses; tie-break on the smaller max
    # required increase (the cheaper fix).
    def _rank(item):
        weight_name, helped = item
        needed = max(m["required_increase"] for m in helped)
        return (-len(helped), needed)

    weight_name, helped = sorted(by_weight.items(), key=_rank)[0]
    needed = max(m["required_increase"] for m in helped)
    delta = round(min(needed * 1.1, _MAX_WEIGHT_DELTA), 4)
    candidate_weights = {weight_name: round(DEFAULT_WEIGHTS[weight_name] + delta, 4)}

    backtest = backtest_and_promote_weights(candidate_weights, with_questions=with_questions,
                                            bank_id=bank_id)

    record = {
        "id": uuid.uuid4().hex[:12],
        "created_at": _now(),
        "bank_id": bank_id,
        "status": "pending_review",
        "reviewer": None, "review_note": None, "decided_at": None,
        "diagnosis": {
            "category": "topic_ranked_low",
            "total_misses_this_run": total_topic_ranked_low,
            "misses_addressed": [
                {"quarter": m["quarter"], "analyst": m["analyst"], "topic": m["topic"],
                 "required_increase": m["required_increase"]}
                for m in helped
            ],
        },
        "proposed_weight": weight_name,
        "proposed_delta": delta,
        "baseline_weight_value": DEFAULT_WEIGHTS[weight_name],
        "candidate_weights": candidate_weights,
        "backtest": backtest,
        "note": "Proposed by continual_learning.propose_weight_adjustment(), not a human or an "
                "LLM -- a deterministic, inspectable computation from this run's own "
                "topic_ranked_low misses. Approval here does NOT edit overall_layer.py's "
                "DEFAULT_WEIGHTS; that stays a deliberate, separate step. See "
                "claude/continual-learning-loop-scope.md.",
    }

    with _lock:
        proposals = _load(bank_id)
        proposals.append(record)
        _save(bank_id, proposals)

    # NOTE: record["status"] is "pending_review" (the review-gate state) --
    # this wrapper key is deliberately named "result", not "status", so it
    # is not silently overwritten by the **record spread.
    return {"result": "proposed", **record}
