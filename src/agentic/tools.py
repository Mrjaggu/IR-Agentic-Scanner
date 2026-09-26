"""
tools.py — the retrieval tools the Planning Agent (Section 3.1) chooses among.

history tool and commitments tool are real, built from data already in
graph.json / metrics_timeseries.json. commitments_tool (2026-09) is grounded
in src.data.commitments's graph-native structured extraction rather than a
bare keyword scan -- see that module for the extraction rules and the
open/due/stale status semantics. The policy tool is honestly a
stub: no RBI/policy-circular ingestion exists in this build (Section 2.1
lists it as a source; it was out of scope for this session per the
"agentic core on top of existing data" decision). It is still wired into
the planning agent so the tool-selection mechanism is real and auditable —
it just returns "no data source ingested" rather than fabricating a signal,
which is the honest thing to do and costs nothing (no LLM call).
"""

from src.config.settings import TOPICS_LIST

# Topics where a discrete commitment (a plan, a raise, a stated target) is a
# plausible, checkable thing to look for in narration -- kept narrow on purpose
# so the commitments tool doesn't just re-run extraction over every topic.
# NIM & Yields added 2026-09: it's the module's own motivating example
# (src.data.commitments's docstring — the 3.8% NIM target pressed across two
# calls) and was, before this, only reachable through decide_tools' separate
# anomaly_score >= 0.85 fallback, not unconditionally like the other four.
COMMITMENT_PRONE_TOPICS = {
    "Capital Adequacy",
    "Subsidiaries' Performance",
    "Strategy & Competitive Positioning",
    "Citibank Integration",
    "NIM & Yields",
}

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
    """Real structured extraction, not a keyword scan: for each outstanding
    commitment on this topic, the quarter it was made in, its stated
    target/number (if any), its horizon (if any), and its status as of
    this call's own training cutoff (open/due/stale). Delegates to
    src.data.commitments's graph-native path, which reads the same graph
    already passed in here -- no dataset.json load, so this stays correct
    per-bank automatically (graph is already whichever bank's own graph the
    caller loaded).

    cutoff is train_quarters' own most recent quarter. Every current caller
    already passes train_quarters in chronological order (see
    run_agentic.py's prior_quarters), but this re-derives "most recent" from
    the graph's own Quarter.sort_key rather than trusting list order --
    a wrong cutoff here would silently misdate every status, and re-deriving
    it costs one extra dict lookup."""
    if topic not in COMMITMENT_PRONE_TOPICS or not train_quarters:
        return []
    from src.data.commitments import open_commitments_as_of_graph, _quarter_index_from_graph
    q_index = _quarter_index_from_graph(graph)
    known = [q for q in train_quarters if q in q_index]
    if not known:
        return []
    cutoff_quarter = max(known, key=lambda q: q_index[q])
    rows = open_commitments_as_of_graph(graph, cutoff_quarter)
    return [r for r in rows if r.get("topic") == topic][:3]


def policy_tool(topic: str) -> dict | None:
    """Stub. Returns a transparent 'no data' marker rather than fabricating
    a policy signal — see module docstring."""
    if topic not in POLICY_SENSITIVE_TOPICS:
        return None
    return {"topic": topic, "note": "no RBI/policy-circular data source ingested for this build"}
