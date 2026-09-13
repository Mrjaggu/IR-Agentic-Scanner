"""
agent_predict.py — retrieval-grounded prediction agent (Stage 3).

Doesn't feed through the engine's multiplicative score formula at all (three
signals already hit that wall this session — see RESULTS_V2.md). Instead:
deterministic retrieval (no LLM) gathers per-analyst style + this quarter's
metric anomalies + the analyst's own historical precedent questions on those
anomalous topics, then a bounded Groq call may propose AT MOST ONE swap into
the formula's own top-N picks, grounded in that retrieved evidence.

This is the evolution of engine.py's existing `_llm_select_topics` (tried
once before with a weaker model and thinner context, made things worse) --
extends that pattern rather than duplicating it, now with metric-anomaly
scores and measured per-analyst style to ground the LLM's choice instead of
letting it freely re-rank.

Gated behind AGENT_PREDICT=1, off by default. Only applies to analysts with
a reliable persona-style sample (n >= MIN_N_FOR_STYLE_NOTE) -- same
"don't touch what we can't measure" floor used elsewhere this session.
"""

import json
import re

from src.signals.persona_synthesis import MIN_N_FOR_STYLE_NOTE

ANOMALY_THRESHOLD = 0.7   # only surface topics with a genuinely unusual move
MAX_PRECEDENTS = 2


def retrieve_context(analyst: str, val_quarter: str, quarter_order: list[str],
                     graph: dict, persona_stats: dict, anomaly_scores: dict) -> dict | None:
    """Returns None if the analyst doesn't qualify (thin persona history) or
    nothing is anomalous enough this quarter to be worth surfacing."""
    stat = persona_stats.get(analyst)
    if not stat or stat["n"] < MIN_N_FOR_STYLE_NOTE:
        return None

    anomalous = sorted(
        [(t, s) for t, s in anomaly_scores.items() if s >= ANOMALY_THRESHOLD],
        key=lambda x: -x[1],
    )
    if not anomalous:
        return None

    vi = quarter_order.index(val_quarter)
    prior_quarters = set(quarter_order[:vi])
    anomalous_topic_set = {t for t, _ in anomalous}

    precedents = []
    for n in graph["nodes"]:
        if n["type"] != "Question":
            continue
        p = n["properties"]
        if p["analyst"] != analyst or p["quarter"] not in prior_quarters:
            continue
        overlap = anomalous_topic_set & {t for t in p["topics"] if t != "General"}
        if overlap:
            precedents.append({
                "quarter": p["quarter"],
                "topics": sorted(overlap),
                "text": p["text"][:350],
                "_qi": quarter_order.index(p["quarter"]),
            })
    precedents.sort(key=lambda x: -x["_qi"])
    precedents = [{k: v for k, v in p.items() if k != "_qi"} for p in precedents[:MAX_PRECEDENTS]]
    if not precedents:
        return None  # no evidence this analyst engages with these topics -- nothing to ground a swap in

    return {
        "style_note": stat["style_note"],
        "anomalous_topics": anomalous,
        "precedents": precedents,
    }


def _build_swap_prompt(analyst: str, target_topics: list[str], context: dict) -> str:
    anomalous_fmt = "\n".join(f"  - {t} (anomaly score {s:.2f})" for t, s in context["anomalous_topics"])
    precedents_fmt = "\n".join(
        f"  - [{p['quarter']}, topics: {', '.join(p['topics'])}] \"{p['text']}\""
        for p in context["precedents"]
    )
    return f"""Analyst: {analyst}
Style (measured from historical transcripts): {context['style_note']}

Topics with unusually large metric moves THIS quarter (0-1 score, 1.0 = the
biggest move that metric has ever had for Axis Bank):
{anomalous_fmt}

This analyst's OWN historical questions on these same topics (evidence they
personally engage with this kind of theme, not just that it's a hot topic
generally):
{precedents_fmt}

Current predicted topics for this analyst (ranked, weakest/least confident last):
{json.dumps(target_topics)}

Task: you may propose AT MOST ONE swap -- replace the LAST (weakest) topic in
the current list with ONE topic from the anomalous list above. Only do this
if you can cite BOTH a specific anomalous metric AND a specific historical
precedent question supporting it. If the evidence isn't solid for both, do
not propose a swap -- it is correct and expected to say no swap most of the time.

Return JSON only:
{{"swap": true|false, "remove_topic": "<topic currently in the list, or null>",
  "add_topic": "<topic from the anomalous list, or null>",
  "evidence": "<cite the specific metric AND precedent, or reason for no swap>"}}"""


def _parse_swap_response(raw: str, target_topics: list[str], candidate_topics: set[str]) -> dict | None:
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
    except Exception:
        return None
    if not parsed.get("swap"):
        return None
    remove_t = parsed.get("remove_topic")
    add_t = parsed.get("add_topic")
    if remove_t not in target_topics:
        return None
    if add_t not in candidate_topics or add_t in target_topics:
        return None
    return {"remove_topic": remove_t, "add_topic": add_t, "evidence": parsed.get("evidence", "")}


def propose_swap(client, analyst: str, target_topics: list[str], context: dict) -> dict | None:
    """Returns {remove_topic, add_topic, evidence} or None (no swap / failure)."""
    prompt = _build_swap_prompt(analyst, target_topics, context)
    raw = client.call_llm(prompt, temperature=0.0)
    candidate_topics = {t for t, _ in context["anomalous_topics"]}
    return _parse_swap_response(raw, target_topics, candidate_topics)
