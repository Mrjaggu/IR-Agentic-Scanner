"""
tools.py — the retrieval tools the Planning Agent (Section 3.1) chooses among.

history tool and commitments tool are real, built from data already in
graph.json / metrics_timeseries.json. The policy tool is honestly a stub:
no RBI/policy-circular ingestion exists in this build (Section 2.1 lists it
as a source; it was out of scope for this session per the "agentic core on
top of existing data" decision). It is still wired into the planning agent
so the tool-selection mechanism is real and auditable — it just returns
"no data source ingested" rather than fabricating a signal, which is the
honest thing to do and costs nothing (no LLM call).
"""

import re
from src.config.settings import TOPICS_LIST

# Topics where a discrete commitment (a plan, a raise, a stated target) is a
# plausible, checkable thing to look for in narration -- kept narrow on purpose
# so the commitments tool doesn't just re-run keyword search over everything.
COMMITMENT_PRONE_TOPICS = {
    "Capital Adequacy",
    "Subsidiaries' Performance",
    "Strategy & Competitive Positioning",
    "Citibank Integration",
}

_COMMITMENT_VERBS = re.compile(
    r"\b(plan to|will raise|approved|committed to|target of|guidance of|"
    r"expect to|intend to|on track to|by (?:the end of |)(?:fy|q[1-4]))\b",
    re.IGNORECASE,
)

# Macro/regulatory-sensitive topics -- exactly where an RBI circular or policy
# change would plausibly matter, which is why the planning agent reaches for
# the (currently empty) policy tool on these specifically, not everything.
POLICY_SENSITIVE_TOPICS = {"NIM & Yields", "Deposits & CASA", "Capital Adequacy"}


def history_tool(topic: str, graph: dict, train_quarters: list[str],
                 global_rate: dict, analyst: str | None = None) -> dict:
    """Deterministic retrieval over the graph layer. No LLM call."""
    qs = [n for n in graph["nodes"]
          if n["type"] == "Question" and n["properties"]["quarter"] in train_quarters]
    topic_count = sum(1 for q in qs if topic in q["properties"]["topics"])
    result = {
        "topic": topic,
        "global_base_rate": round(global_rate.get(topic, 0.0), 4),
        "times_raised_in_history": topic_count,
    }
    if analyst:
        analyst_count = sum(
            1 for q in qs
            if q["properties"]["analyst"] == analyst and topic in q["properties"]["topics"]
        )
        result["analyst_times_raised"] = analyst_count
    return result


def commitments_tool(topic: str, graph: dict, train_quarters: list[str]) -> list[dict]:
    """Heuristic scan of narration for commitment-language sentences tagged
    with this topic. Approximates the doc's graph-layer `commitments`
    store (which would need structured extraction this build didn't do) —
    documented approximation, not the real thing."""
    if topic not in COMMITMENT_PRONE_TOPICS:
        return []
    hits = []
    for n in graph["nodes"]:
        if n["type"] != "NarrationSegment" or n["properties"]["quarter"] not in train_quarters:
            continue
        if topic not in n["properties"]["topics"]:
            continue
        text = n["properties"]["text"]
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if _COMMITMENT_VERBS.search(sentence):
                hits.append({"quarter": n["properties"]["quarter"], "text": sentence.strip()[:300]})
    # Most recent first, capped
    hits.sort(key=lambda h: h["quarter"], reverse=True)
    return hits[:3]


def policy_tool(topic: str) -> dict | None:
    """Stub. Returns a transparent 'no data' marker rather than fabricating
    a policy signal — see module docstring."""
    if topic not in POLICY_SENSITIVE_TOPICS:
        return None
    return {"topic": topic, "note": "no RBI/policy-circular data source ingested for this build"}
