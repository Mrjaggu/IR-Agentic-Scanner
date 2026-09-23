"""
graph_app.py — Section 4 (multi-agent coordination) + Section 12
(orchestration recommendation): wires the pipeline together as a LangGraph
state graph.

Sequenced by cadence per the doc's Orchestrator: the Overall (Topic) layer
runs first (can run off financial metrics alone, weeks ahead of the call),
then the Analyst-Specific layer consumes its ranked topics and reweights per
analyst (Section 4's "inter-layer dependency"). The Verifier's own
reject-and-regenerate cycle lives inside verifier.py as a nested LangGraph
subgraph (see that module) — this top graph is intentionally one-directional
(Overall -> Analyst-specific only), per the doc's Section 4 recommendation
to "start one-directional for the pilot."
"""

from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.agentic.planning_agent import run_planning_agent
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst, build_arithmetic_followups
from src.agentic.verifier import grounding_gate
from src.agentic.question_framer import build_evidence_pool


class PipelineState(TypedDict, total=False):
    graph: dict
    prior_quarters: list[str]
    bank_id: str
    bank_name: str
    val_quarter: str
    global_rate: dict
    momentum: dict
    anomaly_scores: dict
    active_analysts: list[str]
    analyst_prefs: dict          # {analyst: (pref_dict, N)}
    persona_stats: dict
    val_quarter_metrics: dict
    client: object

    evidence_bundles: dict
    tool_log: list
    overall: dict
    analyst_outputs: dict
    arithmetic_followups: dict


def _planning_node(state: PipelineState) -> PipelineState:
    bundles, log = run_planning_agent(
        state["anomaly_scores"], state["graph"], state["prior_quarters"], state["global_rate"]
    )
    return {"evidence_bundles": bundles, "tool_log": log}


def _overall_node(state: PipelineState) -> PipelineState:
    overall = build_overall_topics(
        state["evidence_bundles"], state["anomaly_scores"], state["global_rate"],
        state["momentum"], state["client"], disclosure=state.get("upcoming"),
        bank_name=state.get("bank_name", "Axis Bank"),
    )
    return {"overall": overall}


def _analyst_specific_node(state: PipelineState) -> PipelineState:
    prior_quarters_set = set(state["prior_quarters"])
    outputs = {}
    for analyst in state["active_analysts"]:
        pref, N = state["analyst_prefs"][analyst]
        ranked = reweight_for_analyst(analyst, state["overall"]["ranked_topics"], pref, N,
                                      disclosure=state.get("upcoming"),
                                      sentiment_score=state.get("sentiment_scores", {}).get(analyst),
                                      anomaly_scores=state.get("anomaly_scores"),
                                      use_sentiment_signal=state.get("sentiment_signal", False),
                                      news_signal=state.get("news_signal"),
                                      use_news_signal=state.get("news_signal_enabled", False))
        style_note = state["persona_stats"].get(analyst, {}).get(
            "style_note", "no measured style profile"
        )
        pool = build_evidence_pool(
            analyst, ranked, state["graph"], prior_quarters_set,
            state["global_rate"], state["anomaly_scores"],
            disclosed_metrics=state.get("val_quarter_metrics"),
            target_quarter=state["val_quarter"],
            disclosure_text=(state.get("upcoming") or {}).get("narration", ""),
        )
        results, gate_log = grounding_gate(analyst, style_note, ranked, pool, state["client"],
                                           target_quarter=state["val_quarter"],
                                           bank_name=state.get("bank_name", "Axis Bank"))
        outputs[analyst] = {"topics": results, "verifier_log": gate_log}
    return {"analyst_outputs": outputs}


def _arithmetic_node(state: PipelineState) -> PipelineState:
    followups = build_arithmetic_followups(
        state["active_analysts"], state["persona_stats"],
        state["val_quarter_metrics"], state["client"],
    )
    return {"arithmetic_followups": followups}


_graph = StateGraph(PipelineState)
_graph.add_node("planning", _planning_node)
_graph.add_node("overall", _overall_node)
_graph.add_node("analyst_specific", _analyst_specific_node)
_graph.add_node("arithmetic_followups", _arithmetic_node)
_graph.set_entry_point("planning")
_graph.add_edge("planning", "overall")
_graph.add_edge("overall", "analyst_specific")
_graph.add_edge("analyst_specific", "arithmetic_followups")
_graph.add_edge("arithmetic_followups", END)
compiled_app = _graph.compile()


def run_pipeline(initial_state: PipelineState) -> PipelineState:
    return compiled_app.invoke(initial_state)
