"""
run_move_eval.py — walk-forward move-level evaluation on the held-out calls.

Rolling cutoff throughout: predicting q1fy27 trains through q4fy26, predicting
q4fy26 trains through q3fy26. The target quarter's prepared remarks stand in
for the uploaded disclosure (that is what the live pipeline receives near the
call date); the Q&A of the target quarter is never read except as ground truth.

  python3 -m src.agentic.run_move_eval
"""

import json
import re
import sys
from pathlib import Path

from src.agentic.move_eval import evaluate_moves
from src.agentic.move_planner import (build_available_inputs, coverage_report,
                                      plan_for_analyst, _ONE_OFF)
from src.agentic.moves import build_move_profiles
from src.data.commitments import _TOPIC_TERMS
from src.data.upcoming import parse_upcoming_document

TEST_QUARTERS = ["q4fy26", "q1fy27"]
MAX_SLOTS = 14
PER_TOPIC_CAP = 3


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))
            if len(s.split()) > 6]


def narration_by_topic(text, per_topic=6):
    """Relevance-ranked disclosure sentences per topic.

    Ranking, not document order — and a sentence carrying one-off language is
    promoted, because that is the evidence the normalized_run_rate move needs
    and a plain top-6 slice was dropping it.
    """
    sents = _sentences(text)
    out = {}
    for topic, terms in _TOPIC_TERMS.items():
        scored = []
        for s in sents:
            low = s.lower()
            hits = sum(1 for k in terms if k in low)
            if not hits:
                continue
            scored.append((hits + (2 if _ONE_OFF.search(s) else 0)
                           + (1 if re.search(r"\d", s) else 0), s))
        if scored:
            scored.sort(key=lambda x: -x[0])
            out[topic] = [s for _, s in scored[:per_topic]]
    return out


def metrics_by_topic(disclosure_metrics, topics):
    out = {}
    for topic in topics:
        terms = " ".join(_TOPIC_TERMS.get(topic, []))
        rows = []
        for name, m in (disclosure_metrics or {}).items():
            if name.lower().replace("_", " ") in terms or name.lower() in terms:
                rows.append({"metric": name, **{k: m.get(k) for k in
                             ("value", "value_unit", "delta", "delta_unit",
                              "direction", "period")}})
        if rows:
            out[topic] = rows
    return out


def active_analysts(record, min_questions=1):
    seen = {}
    for t in record.get("qa", []):
        if "nalyst" in str(t.get("role", "")):
            n = t.get("analyst_name") or "unknown"
            seen[n] = seen.get(n, 0) + 1
    return [a for a, c in seen.items() if c >= min_questions]


def run(dataset_path="data/db/dataset.json", quarters=None, verbose=True):
    data = sorted(json.loads(Path(dataset_path).read_text()),
                  key=lambda r: r.get("sort_key", 0))
    order = [r["quarter_id"] for r in data]
    byq = {r["quarter_id"]: r for r in data}
    topics = list(_TOPIC_TERMS)
    results = {}

    for tq in (quarters or TEST_QUARTERS):
        i = order.index(tq)
        cutoff = order[i - 1]                       # rolling
        train = order[:i]                           # target quarter excluded

        text = " ".join(n["text"] for n in byq[tq].get("narration", [])
                        if isinstance(n, dict))
        disclosure = parse_upcoming_document(text.encode(), f"{tq}_disclosure.txt", train)
        nbt = narration_by_topic(text)
        mbt = metrics_by_topic(disclosure.get("metrics"), topics)

        inputs = build_available_inputs(topics, dataset=data, cutoff_quarter=cutoff,
                                        disclosure=disclosure, disclosed_metrics=mbt,
                                        narration_by_topic=nbt)
        profiles = build_move_profiles(data, train)

        # Topic ranking stands in for the Overall layer here: base rate over
        # training quarters only. The point of this harness is the MOVE axis;
        # topic recall is already known to be 90%+.
        from collections import Counter
        rate = Counter()
        for q in train:
            for t in byq[q].get("qa", []):
                if "nalyst" not in str(t.get("role", "")):
                    continue
                from src.data.commitments import _score_topic
                tp = _score_topic(t.get("text", ""))
                if tp:
                    rate[tp] += 1
        ranked = [t for t, _ in rate.most_common()] or topics

        slots = {a: plan_for_analyst(a, ranked, profiles, inputs,
                                     max_slots=MAX_SLOTS, per_topic_cap=PER_TOPIC_CAP)
                 for a in active_analysts(byq[tq])}

        ev = evaluate_moves(byq[tq], slots, inputs)
        ev["quarter"] = tq
        ev["train_cutoff"] = cutoff
        ev["n_train_quarters"] = len(train)
        ev["reachability"] = {m: v["n"] for m, v in coverage_report(inputs, topics).items()}
        results[tq] = ev

        if verbose:
            print(f"\n{'='*72}\n{tq.upper()}  (trained through {cutoff}, "
                  f"{len(train)} quarters)\n{'='*72}")
            print(f"move-bearing questions actually asked : {ev['n_move_bearing_questions']}")
            print(f"move instances (a question can do two): {ev['n_move_instances']}")
            print(f"(topic x move) slots predicted        : {ev['n_predicted_slots']}")
            print(f"MOVE RECALL                          : {ev['move_recall']}")
            print("\nper move:")
            for m, v in ev["per_move"].items():
                print(f"   {m:<24} {v['hit']:>2}/{v['n']:<2} recall {v['recall']:.2f}"
                      f"   (reachable on {ev['reachability'][m]} topics)")
            print("\nwhere the misses belong:")
            for row in ev["owner_of_next_fix"]:
                print(f"   {row['n']:>2}  {row['verdict']:<20} {row['owner']}")

    if verbose and len(results) > 1:
        rs = [r["move_recall"] for r in results.values() if r["move_recall"] is not None]
        print(f"\n{'='*72}\nmean move recall across held-out quarters: "
              f"{sum(rs)/len(rs):.3f}   spread: {max(rs)-min(rs):.3f}")
    return results


if __name__ == "__main__":
    out = run()
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    Path("data/outputs/move_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote data/outputs/move_eval.json")
