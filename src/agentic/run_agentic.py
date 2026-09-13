"""
run_agentic.py — CLI entry point for the agentic prediction core
(`python3 run.py agentic`).

Builds the same four foundation-layer inputs the deterministic engine.py
uses (graph.json, dataset.json, persona stats, metrics_timeseries.json),
then runs the Section 3 agentic pipeline (graph_app.py) on top of them for
VAL_QUARTER. Writes results to data/outputs/agentic_predictions.json and a
human-readable brief to data/outputs/agentic_brief.md. Does NOT touch
predictions.json or engine.py's validated deterministic output.
"""

import json
import os

from src.config.settings import (
    VAL_QUARTER, BASE_DIR, METRICS_TIMESERIES_PATH, TEST_QUARTERS, TRAIN_CUTOFF, CUTOFF_MODE,
)
from src.data.loader import load_dataset, load_graph, get_active_analysts
from src.memory.history import get_global_rate, get_analyst_profile
from src.signals.metrics_extractor import compute_topic_anomaly_scores
from src.signals.persona_synthesis import synthesize_personas
from src.model_provider.llm_client import client
from src.agentic.graph_app import run_pipeline

OUT_JSON = os.path.join(BASE_DIR, "data", "outputs", "agentic_predictions.json")
OUT_MD = os.path.join(BASE_DIR, "data", "outputs", "agentic_brief.md")


def _compute_momentum(graph: dict, quarter_order: list[str], val_ord: int) -> dict:
    """Cross-analyst topic momentum over the prior two quarters (1.0 / 0.5
    weighted) -- same convention engine.py uses."""
    momentum = {}
    for q_idx, weight in ((val_ord - 1, 1.0), (val_ord - 2, 0.5)):
        if q_idx < 0:
            continue
        qname = quarter_order[q_idx]
        for n in graph["nodes"]:
            if n["type"] == "Question" and n["properties"]["quarter"] == qname:
                for t in n["properties"]["topics"]:
                    if t != "General":
                        momentum[t] = momentum.get(t, 0.0) + weight
    return momentum


def build_initial_state(val_quarter: str = VAL_QUARTER, probe: bool = True,
                        holdout: bool = False, upcoming: dict | None = None,
                        cutoff_mode: str | None = None) -> dict:
    """Everything the pipeline needs, built once. Exposed separately from
    main() so the API layer can run just the Overall layer, or just one
    analyst's Analyst-Specific layer, without re-deriving all of this.

    The training split is always leak-free: only quarters strictly before the
    target feed memory, and the target plus every later quarter is excluded
    from derived stats too (predicting q4fy26 must not see q1fy27).

    cutoff_mode="rolling" (default) trains through the quarter immediately
    before the target -- the production setting. cutoff_mode="fixed" freezes
    training at TRAIN_CUTOFF for every target, which is the doc's Section 6.1
    distance test (one vs two quarters ahead).

    upcoming={"narration": str, "metrics": {...}, ...} conditions the run on the
    UPCOMING quarter's draft disclosure instead of predicting from history
    alone (Section 2.1's draft-script source / Section 3.3's in-call
    arithmetic follow-ups)."""
    cutoff_mode = cutoff_mode or CUTOFF_MODE
    dataset = load_dataset()
    graph = load_graph()
    quarter_order = [q["quarter_id"] for q in dataset]
    q_ord = {q: i for i, q in enumerate(quarter_order)}
    # Recency distances always use the TRUE calendar order, so holdout mode
    # doesn't silently make q3fy26 look adjacent to q1fy27.
    val_ord = q_ord.get(val_quarter, len(quarter_order))

    # Rolling (walk-forward) cutoff: train on everything strictly BEFORE the
    # target quarter. Predicting q1fy27 trains through q4fy26; predicting
    # q4fy26 trains through q3fy26. `holdout` no longer changes the split --
    # the split is always leak-free -- it only selects the reporting mode.
    #
    # Excluding the target alone is not enough: when the target is q4fy26 the
    # archive still holds q1fy27, which is the FUTURE relative to that call.
    # Everything from the target onwards is excluded from derived memory too.
    if holdout and cutoff_mode == "fixed":
        cutoff_ord = q_ord[TRAIN_CUTOFF]
        prior_quarters = quarter_order[:cutoff_ord + 1]
        persona_exclude = set(TEST_QUARTERS)
    else:
        prior_quarters = quarter_order[:val_ord]
        persona_exclude = set(quarter_order[val_ord:])   # target + everything after it

    train_cutoff = prior_quarters[-1] if prior_quarters else None

    global_rate = get_global_rate(graph, prior_quarters)
    momentum = _compute_momentum(graph, quarter_order, min(val_ord, len(quarter_order)))
    # Anomaly percentiles compare only against the training pool.
    anomaly_order = prior_quarters + [val_quarter]
    anomaly_scores = compute_topic_anomaly_scores(val_quarter, anomaly_order)

    with open(METRICS_TIMESERIES_PATH) as f:
        timeseries = json.load(f)
    val_quarter_metrics = timeseries.get(val_quarter, {})

    # An uploaded upcoming-quarter disclosure overrides the archive: this is
    # the whole point of the upload -- predict against what management is
    # ABOUT to say, not against a quarter we already have on file.
    if upcoming:
        if upcoming.get("metrics"):
            val_quarter_metrics = upcoming["metrics"]
        if upcoming.get("anomaly_scores"):
            anomaly_scores = upcoming["anomaly_scores"]

    active_analysts = get_active_analysts(dataset, val_quarter)
    analyst_prefs = {}
    for a in active_analysts:
        pref, N = get_analyst_profile(a, graph, prior_quarters, val_ord, q_ord, global_rate)
        analyst_prefs[a] = (pref, N)

    persona_stats = synthesize_personas(exclude_quarters=persona_exclude, out_path=None)

    if probe:
        api_mode = client.probe_llm()
        print(f"[agentic] LLM mode: {api_mode or 'none available -- deterministic fallbacks only'}")

    return {
        "graph": graph,
        "dataset": dataset,
        "quarter_order": quarter_order,
        "holdout": holdout,
        "cutoff_mode": cutoff_mode,
        "train_cutoff": train_cutoff,
        "upcoming": upcoming,
        "prior_quarters": prior_quarters,
        "val_quarter": val_quarter,
        "global_rate": global_rate,
        "momentum": momentum,
        "anomaly_scores": anomaly_scores,
        "active_analysts": active_analysts,
        "analyst_prefs": analyst_prefs,
        "persona_stats": persona_stats,
        "val_quarter_metrics": val_quarter_metrics,
        "client": client,
    }


def main(val_quarter: str = VAL_QUARTER) -> dict:
    initial_state = build_initial_state(val_quarter)
    print(f"[agentic] Quarter: {val_quarter}  Active analysts: {len(initial_state['active_analysts'])}  "
          f"Anomalous topics: {list(initial_state['anomaly_scores'].keys())}")

    result = run_pipeline(initial_state)

    _write_outputs(val_quarter, result)
    return result


def _write_outputs(val_quarter: str, result: dict) -> None:
    serializable = {
        "quarter": val_quarter,
        "planning_agent_tool_log": result["tool_log"],
        "overall_topics": {
            "ranked": result["overall"]["ranked_topics"],
            "rationale": result["overall"]["rationale"],
            "verifier_log": result["overall"]["verifier_log"],
        },
        "arithmetic_followups": result["arithmetic_followups"],
        "analyst_predictions": {
            a: {
                "topics": out["topics"],
                "verifier_log": out["verifier_log"],
            }
            for a, out in result["analyst_outputs"].items()
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[agentic] Wrote {OUT_JSON}")

    lines = [f"# Agentic IR prep brief — {val_quarter}\n"]
    lines.append("## Overall topic ranking (Researcher -> Analyst -> Verifier)\n")
    for t in result["overall"]["ranked_topics"]:
        rationale = result["overall"]["rationale"].get(t)
        lines.append(f"- **{t}**" + (f" — {rationale}" if rationale else " *(bare topic — no grounded rationale)*"))
    if result["arithmetic_followups"]:
        lines.append("\n## In-call arithmetic follow-ups flagged\n")
        for a, item in result["arithmetic_followups"].items():
            lines.append(f"- **{a}**: {item['question']}")
    lines.append("\n## Per-analyst question sheet\n")
    for a, out in result["analyst_outputs"].items():
        lines.append(f"### {a}")
        for item in out["topics"]:
            if item["question_text"]:
                lines.append(f"- **{item['topic']}**: {item['question_text']}")
            else:
                lines.append(f"- **{item['topic']}** *(bare topic — grounding gate rejected phrased text after retries)*")
        lines.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"[agentic] Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
