"""
analyst_sentiment.py -- LLM-based per-analyst, per-quarter sentiment scoring.

Motivation: the "on the UI also we are showing now sentiment trend of
analyst from all the quarter" request -- analysts' TONE toward the bank
(skeptical/probing vs. constructive vs. neutral) shifts quarter to quarter
as their view of the story evolves, and that shift is currently invisible
anywhere in this platform. This module scores it.

Batched one LLM call per quarter (same pattern as question_intent.py /
narration_novelty.py): ONE call classifies every analyst's question blocks
for that quarter at once, not one call per analyst -- keeps cost at
O(quarters), not O(quarters x analysts).

Each analyst's block for a quarter gets a single sentiment_score in
[-1.0, +1.0] (-1 = openly skeptical/critical, 0 = neutral/matter-of-fact,
+1 = constructive/complimentary) plus a short evidence citation, based
ONLY on the wording and framing of their questions that quarter (not on
whether the topic itself is "bad news" -- asking a sharp, skeptical
question about a routine topic scores differently than asking a plain
clarifying question about a genuinely bad number).

IMPORTANT -- this is display-only. It is deliberately NOT wired into
src/memory/history.py's get_analyst_profile() prediction blend: sentiment
tone is noisier signal than topic-recurrence history, and this project's
overriding priority is protecting question recall (see eval_history.py's
docstring). Wiring this into predictions would require running it through
eval_harness.backtest_and_promote_weights() first, per this codebase's
established backtest-before-enable discipline (see engine.py's NOVELTY_/
PEER_ extra-slot comments, RESULTS_V2.md) -- not done here.

Usage:
    python3 run.py analyst-sentiment --quarter q1fy27
    python3 run.py analyst-sentiment            # loops over the last 8 quarters

Output: data/outputs/<bank>/analyst_sentiment.json
    [{"quarter": "q1fy27", "analyst": "...", "sentiment_score": 0.3,
      "label": "constructive", "evidence": "..."}]
"""

import json
import os
import re

from src.config.settings import GRAPH_PATH, ANALYST_SENTIMENT_PATH
from src.model_provider.llm_client import LLMClient

LAST_N_QUARTERS = 8
QUESTION_CHAR_BUDGET = 300

_LABELS = ("skeptical", "neutral", "constructive")


def _label_for(score: float) -> str:
    if score <= -0.34:
        return "skeptical"
    if score >= 0.34:
        return "constructive"
    return "neutral"


def _quarter_sort_key(quarter: str) -> tuple[int, int]:
    """(fiscal_year, quarter_number) so q4fy26 sorts before q1fy27 -- a plain
    string sort gets this backwards ("q1fy27" < "q4fy26" alphabetically)
    across a fiscal-year boundary, which would silently scramble the
    running-average trend for any analyst whose history crosses one."""
    m = re.match(r"q(\d)fy(\d+)", quarter)
    if not m:
        return (0, 0)
    return (int(m.group(2)), int(m.group(1)))


def _load_graph():
    with open(GRAPH_PATH) as f:
        return json.load(f)


def _quarter_order(graph: dict) -> list[str]:
    q_nodes = sorted([n for n in graph["nodes"] if n["type"] == "Quarter"],
                     key=lambda n: n["properties"]["sort_key"])
    return [n["id"] for n in q_nodes]


def _group_quarter_blocks(graph: dict, quarter: str,
                          analysts: set[str] | None = None) -> list[dict]:
    """One block per analyst per quarter, all of that analyst's question
    text for the quarter concatenated -- tone is read across the whole
    exchange, not question-by-question."""
    by_analyst: dict[str, list[str]] = {}
    for n in graph["nodes"]:
        if n["type"] != "Question" or n["properties"]["quarter"] != quarter:
            continue
        a = n["properties"]["analyst"]
        if analysts is not None and a not in analysts:
            continue
        by_analyst.setdefault(a, []).append(n["properties"]["text"][:QUESTION_CHAR_BUDGET])

    return [{"analyst": a, "question_text": " || ".join(texts)}
            for a, texts in sorted(by_analyst.items())]


def _build_prompt(quarter: str, blocks: list[dict]) -> str:
    blocks_fmt = json.dumps([
        {"index": i, "analyst": b["analyst"], "question_text": b["question_text"]}
        for i, b in enumerate(blocks)
    ], indent=2)

    return f"""You are analyzing the TONE of each sell-side analyst's questions on an
Axis Bank earnings call for {quarter} -- not the topic, the tone.

### Question blocks this quarter (index, analyst, that analyst's question text
### for the quarter, concatenated with " || " between separate questions):
{blocks_fmt}

For EACH block, score that analyst's tone toward the bank this quarter as a
sentiment_score from -1.0 to +1.0:
  near -1.0  -- openly skeptical, pushing back, challenging management's framing,
                repeated "why should we believe..." / "that doesn't reconcile..." tone
  near  0.0  -- neutral, matter-of-fact, plain clarifying questions
  near +1.0  -- constructive, complimentary, framing questions around good execution

Judge the WORDING AND FRAMING of the questions, not whether the underlying topic
is good or bad news -- a sharply skeptical question about a routine, unremarkable
topic still scores negative; a plain clarifying question about a genuinely weak
number still scores near neutral.

Return JSON only:
{{
  "scores": [
    {{"index": 0, "sentiment_score": 0.3, "evidence": "<one sentence citing the actual wording>"}}
  ]
}}
One entry per block index, same order."""


def compute_analyst_sentiment(quarter: str, path: str = ANALYST_SENTIMENT_PATH,
                              analysts: set[str] | None = None) -> list[dict]:
    graph = _load_graph()
    quarter_order = _quarter_order(graph)
    if quarter not in quarter_order:
        raise SystemExit(f"Unknown quarter '{quarter}'. Known: {quarter_order}")

    blocks = _group_quarter_blocks(graph, quarter, analysts=analysts)
    if not blocks:
        print(f"[analyst-sentiment] {quarter}: no question blocks found for the given filter, skipping.")
        return []

    client = LLMClient()
    active = client.probe_llm()
    if not active:
        raise SystemExit("No working LLM API (set GROQ_API_KEY / GEMINI_API_KEY in .env).")

    raw = client.call_llm(_build_prompt(quarter, blocks), temperature=0.0,
                          purpose="analyst_sentiment")
    if not raw:
        raise SystemExit(f"[analyst-sentiment] {quarter}: LLM call failed.")

    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    parsed = json.loads(clean)
    scores = {s["index"]: s for s in parsed.get("scores", [])}

    results = []
    for i, b in enumerate(blocks):
        s = scores.get(i, {})
        try:
            score = max(-1.0, min(1.0, float(s.get("sentiment_score", 0.0))))
        except (TypeError, ValueError):
            score = 0.0
        results.append({
            "quarter": quarter,
            "analyst": b["analyst"],
            "sentiment_score": round(score, 3),
            "label": _label_for(score),
            "evidence": str(s.get("evidence", ""))[:300],
        })

    # merge into existing file, replacing this quarter's entries (idempotent recompute)
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history = [h for h in history if h["quarter"] != quarter]
    history.extend(results)
    history.sort(key=lambda h: (_quarter_sort_key(h["quarter"]), h["analyst"]))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"[analyst-sentiment] {quarter}: {len(results)} analysts scored -- {counts}")
    print(f"Saved -> {path}")
    return results


def compute_analyst_sentiment_recent(n: int = LAST_N_QUARTERS, path: str = ANALYST_SENTIMENT_PATH,
                                     analysts: set[str] | None = None):
    graph = _load_graph()
    quarter_order = _quarter_order(graph)
    quarters = quarter_order[-n:]
    if analysts is not None:
        relevant = set()
        for node in graph["nodes"]:
            if (node["type"] == "Question" and node["properties"]["quarter"] in quarters
                    and node["properties"]["analyst"] in analysts):
                relevant.add(node["properties"]["quarter"])
        quarters = [q for q in quarters if q in relevant]
    for q in quarters:
        compute_analyst_sentiment(q, path=path, analysts=analysts)


def sentiment_trend(analyst: str, path: str = ANALYST_SENTIMENT_PATH) -> dict:
    """Running average up to each quarter, for one analyst -- the definition
    the user picked explicitly over cumulative sum or unweighted overall
    average (AskUserQuestion: "Running average up to each quarter").

    Returns {"analyst", "points": [{"quarter", "sentiment_score",
    "running_avg", "label"}]}, oldest quarter first. Never raises -- an
    analyst with no scored quarters yet just gets an empty points list."""
    points = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
        rows = sorted((h for h in history if h["analyst"] == analyst),
                      key=lambda h: _quarter_sort_key(h["quarter"]))
        running_sum = 0.0
        for i, h in enumerate(rows, start=1):
            running_sum += h["sentiment_score"]
            running_avg = round(running_sum / i, 3)
            points.append({
                "quarter": h["quarter"],
                "sentiment_score": h["sentiment_score"],
                "running_avg": running_avg,
                "label": _label_for(running_avg),
            })
    return {"analyst": analyst, "points": points}


if __name__ == "__main__":
    compute_analyst_sentiment_recent()
