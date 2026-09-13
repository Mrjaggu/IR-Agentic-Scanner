"""
Agentic prediction core — Section 3 of the architecture doc
(ir-intelligence-platform-architecture.pdf).

Built ON TOP of the existing v2/v3 foundation layers (graph.json, dataset.json,
analyst_personas*.json, metrics_timeseries.json) rather than re-ingesting from
scratch: those already implement the doc's Section 2 foundation layers well
enough to ground an agentic prediction layer. This package is additive — it
does not modify engine.py's validated deterministic pipeline or predictions.json.

Modules:
    tools.py           — history / commitments / policy retrieval tools
    planning_agent.py  — Section 3.1: per-anomaly autonomous tool selection
    overall_layer.py   — Section 3.2: Researcher -> Analyst -> Verifier crew
    analyst_layer.py   — Section 3.3: per-analyst reweighting + arithmetic follow-ups
    question_framer.py — Section 3.4: phrased, attributed question text
    verifier.py         — Section 3.5: grounding gate (reject-and-regenerate, capped)
    graph_app.py        — Section 4/12: LangGraph state-graph wiring it all together
    run_agentic.py       — CLI entry point (`python3 run.py agentic`)
"""
