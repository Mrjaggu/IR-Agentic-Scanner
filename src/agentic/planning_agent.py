"""
planning_agent.py — Section 3.1: autonomous per-anomaly tool selection.

Replaces the "pull every signal type every time" pattern with a bounded
decision per flagged anomaly: which of {history, commitments, policy} is
worth calling for THIS topic. Caps: ~3-4 tool calls per anomaly (structurally
at most 3 here since there are only 3 tools), ~10 per quarter total — once
the quarter cap is hit, remaining calls are logged as skipped rather than
made, so the bound is real and auditable rather than aspirational.
"""

from src.agentic.tools import (
    history_tool, commitments_tool, policy_tool,
    COMMITMENT_PRONE_TOPICS, POLICY_SENSITIVE_TOPICS,
)

MAX_CALLS_PER_QUARTER = 10


def decide_tools(topic: str, anomaly_score: float) -> list[str]:
    """Returns the ordered list of tool names to call for this anomaly."""
    tools = ["history"]  # always cheap + always relevant
    if topic in COMMITMENT_PRONE_TOPICS or anomaly_score >= 0.85:
        tools.append("commitments")
    if topic in POLICY_SENSITIVE_TOPICS:
        tools.append("policy")
    return tools


def run_planning_agent(anomaly_scores: dict, graph: dict, train_quarters: list[str],
                       global_rate: dict) -> tuple[dict, list[dict]]:
    """anomaly_scores: {topic: score} from metrics_extractor.compute_topic_anomaly_scores.

    Returns (evidence_bundles, call_log) where evidence_bundles is
    {topic: {history, commitments, policy}} and call_log records every
    decision made (including skips once the quarter cap is hit) for audit."""
    evidence_bundles = {}
    call_log = []
    calls_made = 0

    # Highest-anomaly topics get priority if the cap is going to bind.
    ordered = sorted(anomaly_scores.items(), key=lambda x: -x[1])

    for topic, score in ordered:
        planned = decide_tools(topic, score)
        bundle = {}
        for tool_name in planned:
            if calls_made >= MAX_CALLS_PER_QUARTER:
                call_log.append({"topic": topic, "tool": tool_name, "anomaly_score": score,
                                 "status": "skipped_cap"})
                continue
            if tool_name == "history":
                bundle["history"] = history_tool(topic, graph, train_quarters, global_rate)
            elif tool_name == "commitments":
                bundle["commitments"] = commitments_tool(topic, graph, train_quarters)
            elif tool_name == "policy":
                bundle["policy"] = policy_tool(topic)
            calls_made += 1
            call_log.append({"topic": topic, "tool": tool_name, "anomaly_score": score,
                             "status": "called"})
        evidence_bundles[topic] = bundle

    return evidence_bundles, call_log
