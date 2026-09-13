"""
verifier.py — Section 3.5: Verifier / Grounding Gate.

"Fact-checks every generated question's cited numbers/claims against source
data before it reaches output. Hard gate, not a soft check: reject and
regenerate (capped at 2 retries) rather than pass through with a caveat. If
regeneration fails twice, output the bare topic without fabricated phrasing
rather than block the pipeline."

Implemented as an actual LangGraph cyclic state graph (frame -> check ->
[conditional: back to frame with only the failing subset, or done]) rather
than a plain while-loop — the concrete instance of Section 12's stated reason
for picking LangGraph: "loops back to earlier nodes with conditional routing
are first-class, not a workaround."

Two things are checked, and the second exists because of a real failure:

1. Numbers. Every figure in a question must come from a fact that exists in
   the world — the analyst's own prior wording, or a metric management is
   disclosing. Crucially the ranker's OWN features (base rate, anomaly
   percentile) do NOT count as grounding. They used to, which is how
   "given the historical base rate of 17% … and this quarter's anomaly score
   of 1.00" passed the gate: those numbers were real, they were just real
   *internal scores*, not facts about the bank.

2. Vocabulary. A question naming base rates, anomaly scores, percentiles or
   confidence is rejected outright regardless of its numbers. No analyst
   speaks that way, so its presence means the model is narrating the
   pipeline instead of asking a question.

Both checks are deterministic — no second LLM call to grade the first, which
is cheaper and far more auditable.
"""

import re
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.agentic.question_framer import frame_questions

MAX_RETRIES = 2

# Vocabulary that betrays the pipeline talking about itself.
_SCAFFOLDING = re.compile(
    r"\b(base[- ]rate|base rates|anomaly|percentile|confidence (?:score|level)|"
    r"probability|probabilit|likelihood score|ranking score|score of|"
    r"predicted probability|model (?:score|output)|historical base)\b", re.I)


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def _legit_numbers(evidence: dict) -> set[str]:
    """Only facts about the bank: the analyst's own prior words, and the
    figures management is disclosing. Base rate and anomaly percentile are
    deliberately excluded — they are how we chose the topic, not something
    anyone said."""
    legit = set()
    if evidence.get("precedent"):
        legit |= _numbers_in(evidence["precedent"]["text"])
        legit |= set(re.findall(r"\d", evidence["precedent"]["quarter"]))
    for m in evidence.get("metrics", []) or []:
        for key in ("value", "delta"):
            v = m.get(key)
            if v is None:
                continue
            legit.add(str(v))
            legit.add(str(int(v)) if float(v) == int(v) else str(v))
            legit.add(f"{float(v):.2f}")
            legit.add(f"{float(v):.1f}")
    # Anything management itself puts on the record for this call. See
    # question_framer.disclosure_numbers for why this boundary sits here.
    legit |= evidence.get("disclosure_numbers") or set()
    return legit


def check_question(question_text: str, evidence: dict) -> tuple[bool, str]:
    """(ok, reason). The reason is fed back to the framer on retry."""
    if not question_text:
        return False, "no question was produced"

    if _SCAFFOLDING.search(question_text):
        hit = _SCAFFOLDING.search(question_text).group(0)
        return False, (f'it said "{hit}" — internal scoring vocabulary must never appear '
                       f"in a question an analyst would speak")

    cited = _numbers_in(question_text)
    if not cited:
        return True, ""          # no numeric claim, nothing to fabricate

    legit = _legit_numbers(evidence)
    ungrounded = cited - legit
    if ungrounded:
        return False, (f"it used figure(s) {sorted(ungrounded)[:4]} that appear neither in the "
                       f"analyst's own prior question nor in this quarter's disclosed metrics")
    return True, ""


class GateState(TypedDict, total=False):
    analyst: str
    style_note: str
    pool: dict
    client: object
    target_quarter: str
    remaining: list[str]
    drafts: dict
    accepted: dict
    attempt: int
    log: list[dict]
    feedback: str


def _frame_node(state: GateState) -> GateState:
    drafts = frame_questions(state["analyst"], state["style_note"], state["remaining"],
                             state["pool"], state["client"],
                             target_quarter=state.get("target_quarter", ""),
                             feedback=state.get("feedback", ""))
    return {"drafts": drafts}


def _check_node(state: GateState) -> GateState:
    accepted = dict(state.get("accepted", {}))
    log = list(state.get("log", []))
    still_failing, reasons = [], []
    for t in state["remaining"]:
        qtext = state["drafts"].get(t)
        ok, reason = check_question(qtext, state["pool"][t])
        if ok:
            accepted[t] = qtext
            log.append({"topic": t, "attempt": state["attempt"], "status": "accepted"})
        else:
            still_failing.append(t)
            reasons.append(reason)
            log.append({"topic": t, "attempt": state["attempt"],
                        "status": "rejected", "reason": reason, "draft": qtext})
    return {"accepted": accepted, "log": log, "remaining": still_failing,
            "attempt": state["attempt"] + 1,
            "feedback": reasons[0] if reasons else ""}


def _route_after_check(state: GateState) -> str:
    if state["remaining"] and state["attempt"] <= MAX_RETRIES:
        return "retry"
    return "done"


_graph = StateGraph(GateState)
_graph.add_node("frame", _frame_node)
_graph.add_node("check", _check_node)
_graph.set_entry_point("frame")
_graph.add_edge("frame", "check")
_graph.add_conditional_edges("check", _route_after_check, {"retry": "frame", "done": END})
_compiled_gate = _graph.compile()


def grounding_gate(analyst: str, style_note: str, topics: list[str], pool: dict, client,
                   target_quarter: str = "") -> tuple[list[dict], list[dict]]:
    """Returns ([{topic, question_text, status}], log)."""
    if not topics:
        return [], []
    final = _compiled_gate.invoke({
        "analyst": analyst, "style_note": style_note, "pool": pool, "client": client,
        "target_quarter": target_quarter,
        "remaining": list(topics), "accepted": {}, "attempt": 0, "log": [], "feedback": "",
    })
    accepted = final.get("accepted", {})
    results = []
    for t in topics:
        if t in accepted:
            results.append({"topic": t, "question_text": accepted[t], "status": "grounded"})
        else:
            results.append({"topic": t, "question_text": None, "status": "bare_topic_after_retries"})
    return results, final.get("log", [])
