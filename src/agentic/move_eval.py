"""
move_eval.py — the metric the improvement loop should have been optimising.

The diagnosis this module encodes
---------------------------------
The loop was not broken. It was converged on the wrong objective. Topic recall
on the held-out calls was 92.6% with 4 misses out of 40, so the gradient was
flat and the failure attribution had almost nothing to sort. Meanwhile the
questions management actually could not have prepared for were being missed at
roughly three times that rate, invisibly, because topic buckets cannot see an
angle.

So: score (topic x move) pairs on the MOVE-BEARING subset of actual questions —
20-of-48 on the held-out calls — and attribute every miss to the layer that
owns the fix:

  input_missing        the move's required input did not exist for that topic as
                       of the cutoff. A DATA-LAYER gap. No prompt change can
                       ever fix it, and the loop must never propose one.
  slot_not_allocated   the input existed and the planner did not rank the pair
                       into this analyst's slate. A RANKING gap: slot budget,
                       propensity weights, per-topic cap.
  wrong_analyst        the pair was predicted, for someone else. An ANALYST-
                       ROUTING gap — cheap to fix by widening, and the reason
                       Subsidiaries' Performance reached Piran Engineer instead
                       of Rikin Shah.
  generation_failure   the slot was allocated and the generated text does not
                       perform the move. The ONLY bucket a prompt/skill change
                       should respond to.
  off_taxonomy         the concern maps to no topic in the taxonomy. A
                       TAXONOMY gap, tracked separately so it neither flatters
                       nor punishes the generator.

That last distinction is the whole point. Before this, every miss landed in
'reasoning_weakness' and the loop dutifully rewrote prompts against gaps that
were structural — guidance_callback was unreachable because the commitments
store was a stub, and no wording would have conjured a commitment.
"""

from collections import Counter, defaultdict

from src.agentic.moves import MOVES, TARGET_MOVES, detect_moves
from src.agentic.question_eval import decompose_concerns


def _attribute_topic(text: str) -> str | None:
    from src.data.commitments import _score_topic
    return _score_topic(text)


def collect_actual_moves(quarter_record: dict) -> list[dict]:
    """Every move-bearing question actually asked on one call."""
    out = []
    for turn in quarter_record.get("qa", []):
        if "nalyst" not in str(turn.get("role", "")):
            continue
        analyst = turn.get("analyst_name") or "unknown"
        for concern in decompose_concerns(turn.get("text", "")):
            moves = [m for m in detect_moves(concern) if m in TARGET_MOVES]
            if not moves:
                continue
            out.append({
                "analyst": analyst,
                "text": concern,
                "moves": moves,
                "topic": _attribute_topic(concern),
            })
    return out


def _slot_index(slots_by_analyst: dict) -> tuple[set, set]:
    """(analyst, topic, move) pairs and (topic, move) pairs that were predicted."""
    exact, anywhere = set(), set()
    for analyst, slots in (slots_by_analyst or {}).items():
        for s in slots:
            exact.add((analyst, s["topic"], s["move"]))
            anywhere.add((s["topic"], s["move"]))
    return exact, anywhere


def attribute_move_miss(actual: dict, move: str, exact: set, anywhere: set,
                        available_inputs: dict,
                        generated_text_by_slot: dict | None = None) -> str:
    topic = actual.get("topic")
    if not topic:
        return "off_taxonomy"
    key = (actual["analyst"], topic, move)
    if key in exact:
        # The slot was allocated. Did the text actually perform the move?
        if generated_text_by_slot is not None:
            text = generated_text_by_slot.get(key)
            if text and move in detect_moves(text):
                return "hit"
            return "generation_failure"
        return "hit"
    if (topic, move) in anywhere:
        return "wrong_analyst"
    need = MOVES[move]["needs"]
    if not (available_inputs.get(need) or {}).get(topic):
        return "input_missing"
    return "slot_not_allocated"


def evaluate_moves(quarter_record: dict, slots_by_analyst: dict,
                   available_inputs: dict,
                   generated_text_by_slot: dict | None = None) -> dict:
    """Move-level recall for one call, with every miss attributed to an owner.

    Recall-first, so this reports recall and coverage prominently and precision
    only as a degeneracy guard: the denominator is what analysts actually asked,
    and predicting generously is the intended trade.
    """
    actuals = collect_actual_moves(quarter_record)
    exact, anywhere = _slot_index(slots_by_analyst)

    rows = []
    for a in actuals:
        for m in a["moves"]:
            verdict = attribute_move_miss(a, m, exact, anywhere, available_inputs,
                                          generated_text_by_slot)
            rows.append({**a, "move": m, "verdict": verdict})

    n = len(rows)
    hits = sum(1 for r in rows if r["verdict"] == "hit")
    scorable = [r for r in rows if r["verdict"] != "off_taxonomy"]
    n_scorable = len(scorable)

    per_move = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        per_move[r["move"]]["n"] += 1
        per_move[r["move"]]["hit"] += r["verdict"] == "hit"

    n_predicted = sum(len(v) for v in (slots_by_analyst or {}).values())

    return {
        "n_move_bearing_questions": len({(r["analyst"], r["text"]) for r in rows}),
        "n_move_instances": n,
        "n_predicted_slots": n_predicted,
        # HEADLINE: hits over everything analysts actually asked. An
        # off-taxonomy concern is a question we failed to predict, so it counts
        # as a miss. Excluding it was how the first draft of this metric
        # reported 1.00 on Q4FY26 while missing 6 of 11 questions — the same
        # self-flattery that made topic recall useless, reproduced one level up.
        "move_recall": round(hits / n, 3) if n else None,
        # Diagnostic only: recall restricted to concerns the taxonomy can see.
        # Use it to separate a taxonomy gap from a prediction gap, never as the
        # number reported.
        "move_recall_on_taxonomy": round(hits / n_scorable, 3) if n_scorable else None,
        "per_move": {m: {**v, "recall": round(v["hit"] / v["n"], 3)}
                     for m, v in sorted(per_move.items())},
        "attribution": dict(Counter(r["verdict"] for r in rows)),
        "owner_of_next_fix": _owner(Counter(r["verdict"] for r in rows)),
        "misses": [{"analyst": r["analyst"], "topic": r["topic"], "move": r["move"],
                    "verdict": r["verdict"], "text": r["text"][:180]}
                   for r in rows if r["verdict"] != "hit"],
    }


_OWNER = {
    "input_missing": "data layer — add the missing input; do NOT touch prompts",
    "slot_not_allocated": "ranking — slot budget / move propensity / per-topic cap",
    "wrong_analyst": "analyst routing — widen pull-in beyond prior-topic history",
    "generation_failure": "prompt or skill — the only bucket the loop should act on",
    "off_taxonomy": "taxonomy — the bucket list itself is short",
}


def _owner(counts: Counter) -> list[dict]:
    """Ranked list of what to fix next, by how many misses each layer owns.

    The promotion gate reads this: a loop iteration that proposes a prompt
    change while `input_missing` dominates is optimising the wrong layer, and
    should be blocked rather than promoted.
    """
    return [{"verdict": v, "n": n, "owner": _OWNER[v]}
            for v, n in counts.most_common() if v != "hit"]
