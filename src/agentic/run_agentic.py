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
    paths_for,
)
from src.config.banks import DEFAULT_BANK, get_bank
from src.data.loader import load_dataset, load_graph, get_active_analysts
from src.memory.history import get_global_rate, get_analyst_profile, adaptive_decay_for_analyst
from src.signals.metrics_extractor import compute_topic_anomaly_scores
from src.signals.persona_synthesis import synthesize_personas, apply_ask_patterns
from src.signals.analyst_sentiment import sentiment_as_of
from src.signals.news_signal import news_topic_salience
from src.model_provider.llm_client import client
from src.agentic.graph_app import run_pipeline


def _out_paths(bank_id: str) -> tuple[str, str]:
    d = os.path.join(BASE_DIR, "data", "outputs", bank_id)
    return os.path.join(d, "agentic_predictions.json"), os.path.join(d, "agentic_brief.md")


# Kept for back-compat introspection -- not read internally, every call site
# below resolves its own paths via _out_paths(bank_id).
OUT_JSON, OUT_MD = _out_paths(DEFAULT_BANK)


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
                        cutoff_mode: str | None = None, bank_id: str = DEFAULT_BANK,
                        use_cross_bank_signal: bool = False,
                        use_sentiment_signal: bool = False,
                        use_news_signal: bool = False,
                        use_adaptive_decay: bool = True) -> dict:
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
    arithmetic follow-ups).

    use_cross_bank_signal=True (default False everywhere -- opt-in only, see
    src/signals/cross_bank_persona.py's module docstring for why) blends each
    active analyst's topic history on OTHER registered banks into their
    own-bank preference, the same additive-prior pattern already used for the
    hand-curated PERSONAS topic_prior. Leak-free: only cross-bank history
    strictly before this run's own train_cutoff is used.

    use_sentiment_signal=True (default False everywhere -- opt-in only, same
    backtest-before-promote discipline as use_cross_bank_signal) computes
    each active analyst's leak-free running-average sentiment as of strictly
    before val_quarter (src.signals.analyst_sentiment.sentiment_as_of) and
    hands it to analyst_layer.reweight_for_analyst's sentiment_extra_slot
    mechanism. Axis-only for now -- analyst_sentiment.py isn't bank-scoped
    yet -- so this is a no-op (sentiment_scores all None) for Kotak/IndusInd.

    use_news_signal=True (default False everywhere, and permanently opt-in
    -- see src.signals.news_signal's module docstring: there is no
    historical news archive, so this can never be backtested the way
    use_sentiment_signal was) computes this bank's CURRENT news-derived
    topic salience via src.signals.news_signal.news_topic_salience and hands
    it to analyst_layer.reweight_for_analyst. Refused whenever holdout=True,
    unconditionally, no matter what the caller passes -- news is always
    "right now", so using it to score a HISTORICAL quarter would leak
    today's headlines into a prediction about the past. A holdout call
    therefore always gets news_signal={}, silently, rather than an error,
    the same "no signal" convention every leak-free signal here uses when
    it has nothing to contribute.

    use_adaptive_decay=True (DEFAULT since 2026-09, promoted -- was opt-in
    while being backtested, same discipline as use_sentiment_signal, and
    the riskiest of the three signals built this session since it changes
    the WEIGHTING of every existing prediction rather than adding a
    candidate) replaces the single global EngineConfig.DECAY with a
    per-analyst rate derived from that analyst's own leak-free
    quarter-over-quarter topic-repeat propensity (see
    src.memory.history.adaptive_decay_for_analyst). An analyst with fewer
    than 3 qualifying consecutive-quarter appearances keeps the global
    default (no override) rather than getting a guessed rate.

    MEASURED RESULT that promoted it (2026-09, axis): official 2-quarter
    TEST_QUARTERS -- mean_recall 0.9264->0.9375, mean_precision
    0.3599->0.3678, promotion gate still PASS (recall_spread 0.0694, well
    under the 0.18 threshold). 17-quarter walk-forward sweep (144
    analyst-quarter observations, same window rank_position_calibration
    uses) -- mean_recall 0.8081->0.8095, mean_precision 0.3022->0.3032:
    smaller but the SAME direction, so not a fluke of the 2-quarter sample.
    Kotak/IndusInd: byte-identical with the signal on or off -- neither bank
    has enough per-analyst history yet to clear the den>=3 floor for anyone,
    so this is a safe no-op there, not a risk. Pass use_adaptive_decay=False
    explicitly to get the old global-decay behavior back for comparison."""
    cutoff_mode = cutoff_mode or CUTOFF_MODE
    bank = get_bank(bank_id)
    paths = paths_for(bank_id)
    dataset = load_dataset(dataset_path=paths.dataset_path)
    graph = load_graph(graph_path=paths.graph_path)
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
    anomaly_scores = compute_topic_anomaly_scores(val_quarter, anomaly_order,
                                                  timeseries_path=paths.metrics_timeseries_path)

    val_quarter_metrics = {}
    if os.path.exists(paths.metrics_timeseries_path):
        with open(paths.metrics_timeseries_path) as f:
            timeseries = json.load(f)
        val_quarter_metrics = timeseries.get(val_quarter, {})
    # else: newly-registered bank with no metrics_timeseries.json yet (see
    # compute_topic_anomaly_scores' matching guard below) -- val_quarter_metrics
    # stays {} rather than crashing build_initial_state.

    # An uploaded upcoming-quarter disclosure overrides the archive: this is
    # the whole point of the upload -- predict against what management is
    # ABOUT to say, not against a quarter we already have on file.
    if upcoming:
        if upcoming.get("metrics"):
            val_quarter_metrics = upcoming["metrics"]
        if upcoming.get("anomaly_scores"):
            anomaly_scores = upcoming["anomaly_scores"]

    active_analysts = get_active_analysts(dataset, val_quarter)
    analyst_decay = {}
    if use_adaptive_decay:
        analyst_decay = {a: adaptive_decay_for_analyst(a, graph, prior_quarters, q_ord)
                         for a in active_analysts}
    analyst_prefs = {}
    for a in active_analysts:
        pref, N = get_analyst_profile(a, graph, prior_quarters, val_ord, q_ord, global_rate,
                                      decay_override=analyst_decay.get(a))
        analyst_prefs[a] = (pref, N)

    if use_cross_bank_signal:
        from src.signals.cross_bank_persona import blend_cross_bank_prior
        analyst_prefs = {
            a: (blend_cross_bank_prior(a, bank_id, pref, as_of_quarter=train_cutoff), N)
            for a, (pref, N) in analyst_prefs.items()
        }

    sentiment_scores = {}
    if use_sentiment_signal:
        sentiment_scores = {a: sentiment_as_of(a, val_quarter) for a in active_analysts}

    news_signal = {}
    if use_news_signal and not holdout:  # never for a historical/backtest run -- see docstring above
        news_signal = news_topic_salience(bank_id, bank.display_name)

    persona_stats = synthesize_personas(intent_path=paths.question_intent_path,
                                        exclude_quarters=persona_exclude, out_path=None)
    # Layer the hand-curated, cross-checked ask-pattern taxonomy on top -- see
    # persona_synthesis.apply_ask_patterns' docstring. Additive: analysts not in
    # that file are unaffected. Currently axis-only content (data/inputs/axis/
    # analyst_ask_patterns.json); paths.ask_patterns_path resolves to a file
    # that doesn't exist yet for other banks, which apply_ask_patterns already
    # handles gracefully (returns persona_stats unchanged).
    persona_stats = apply_ask_patterns(persona_stats, path=paths.ask_patterns_path)

    if probe:
        api_mode = client.probe_llm()
        print(f"[agentic] LLM mode: {api_mode or 'none available -- deterministic fallbacks only'}")

    return {
        "bank_id": bank_id,
        "bank_name": bank.display_name,
        "cross_bank_signal": use_cross_bank_signal,
        "sentiment_signal": use_sentiment_signal,
        "sentiment_scores": sentiment_scores,
        "news_signal_enabled": use_news_signal and not holdout,
        "news_signal": news_signal,
        "adaptive_decay_enabled": use_adaptive_decay,
        "analyst_decay": analyst_decay,
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


def main(val_quarter: str = VAL_QUARTER, bank_id: str = DEFAULT_BANK,
        use_cross_bank_signal: bool = False) -> dict:
    initial_state = build_initial_state(val_quarter, bank_id=bank_id,
                                        use_cross_bank_signal=use_cross_bank_signal)
    print(f"[agentic] Bank: {initial_state['bank_name']}  Quarter: {val_quarter}  "
          f"Active analysts: {len(initial_state['active_analysts'])}  "
          f"Anomalous topics: {list(initial_state['anomaly_scores'].keys())}")

    result = run_pipeline(initial_state)

    _write_outputs(val_quarter, result, bank_id=bank_id)
    return result


def _write_outputs(val_quarter: str, result: dict, bank_id: str = DEFAULT_BANK) -> None:
    out_json, out_md = _out_paths(bank_id)
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
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[agentic] Wrote {out_json}")

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
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[agentic] Wrote {out_md}")


if __name__ == "__main__":
    main()
