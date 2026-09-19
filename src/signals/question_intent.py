"""
question_intent.py — LLM-based "why did this analyst ask this" classification.

Motivation: the engine currently reduces every question to a keyword topic
label and never asks *why* it was asked. Confirmed on real examples (MB
Mahesh's Q4FY26 RAROC question directly cites "that RAROC argument that you
presented" — a narration trigger, not a recurring personal interest). This
pass classifies each historical question block into exactly one of:

  persona_consistent — matches this analyst's own recurring topic pattern
                        (cite the prior-quarter evidence)
  narration_triggered — reacts to something specific in THIS quarter's
                         opening remarks (cite the excerpt)
  unexplained         — doesn't clearly map to either; the honest bucket the
                         far-miss-ranked 25% of questions likely live in

Batched one LLM call per quarter (same pattern as narration_novelty.py),
covering the last 8 quarters (q2fy25-q1fy27) per the project's scoping
decision. External-news context (a 4th category) was descoped mid-build —
see RESULTS_V2.md / the plan file for why.

Usage:
    python3 run.py intent --quarter q1fy27
    python3 run.py intent            # loops over the last 8 quarters

Output: data/db/question_intent.json
    [{"quarter": "q1fy27", "analyst": "...", "topics": [...],
      "intent": "persona_consistent", "evidence": "..."}]
"""

import json
import re

from src.config.settings import GRAPH_PATH, QUESTION_INTENT_PATH
from src.model_provider.llm_client import LLMClient

_VALID_INTENTS = {"persona_consistent", "narration_triggered", "unexplained"}

LAST_N_QUARTERS = 8
NARR_CHAR_BUDGET = 3500   # our Groq tier caps at 8000 TPM per request — stay well under
QUESTION_CHAR_BUDGET = 300


def _load_graph():
    with open(GRAPH_PATH) as f:
        return json.load(f)


def _quarter_order(graph: dict) -> list[str]:
    q_nodes = sorted([n for n in graph["nodes"] if n["type"] == "Quarter"],
                     key=lambda n: n["properties"]["sort_key"])
    return [n["id"] for n in q_nodes]


def _build_prompt(quarter: str, narration_text: str, blocks: list[dict]) -> str:
    blocks_fmt = json.dumps([
        {"index": i, "analyst": b["analyst"], "topics": b["topics"],
         "question_text": b["question_text"][:QUESTION_CHAR_BUDGET],
         "analyst_prior_topics": b["prior_topics"]}
        for i, b in enumerate(blocks)
    ], indent=2)

    return f"""You are analyzing an Axis Bank earnings call to understand WHY each analyst
asked what they asked — not just what topic it falls under.

### {quarter} management opening remarks (narration):
{narration_text[:NARR_CHAR_BUDGET]}

### Question blocks this quarter (index, analyst, topics already tagged, question text,
### and that analyst's own topics from PRIOR quarters for persona context):
{blocks_fmt}

For EACH question block, classify into exactly one:
  "persona_consistent" — ONLY if at least one of this block's own "topics" values also
                          appears in that analyst's "analyst_prior_topics" list. Your
                          evidence MUST name that exact shared topic string — do not cite
                          a different prior topic that isn't in this block's own topics.
  "narration_triggered" — the question clearly reacts to something specific said in
                           THIS quarter's narration above (a number, a new slide, a
                           policy). Quote the narration snippet it reacts to.
  "unexplained" — doesn't clearly map to either. Be honest here — don't force a fit.

Return JSON only:
{{
  "classifications": [
    {{"index": 0, "intent": "persona_consistent|narration_triggered|unexplained",
      "evidence": "<one sentence citing the specific prior topic or narration excerpt>"}}
  ]
}}
One entry per question block index, same order."""


def _group_quarter_blocks(graph: dict, quarter: str, quarter_order: list[str],
                          analysts: set[str] | None = None) -> tuple[str, list[dict]]:
    qi = quarter_order.index(quarter)
    prior_quarters = set(quarter_order[:qi])

    narr_segs = [n["properties"] for n in graph["nodes"]
                 if n["type"] == "NarrationSegment" and n["properties"]["quarter"] == quarter]
    parts, used = [], 0
    for s in narr_segs:
        if not s.get("topics"):  # untagged segments are usually boilerplate
            continue
        txt = s["text"].strip()
        take = txt[: max(0, NARR_CHAR_BUDGET - used)]
        if not take:
            break
        parts.append(take)
        used += len(take)
    narration_text = "\n---\n".join(parts)

    prior_topics_by_analyst: dict[str, set] = {}
    for n in graph["nodes"]:
        if n["type"] != "Question" or n["properties"]["quarter"] not in prior_quarters:
            continue
        a = n["properties"]["analyst"]
        prior_topics_by_analyst.setdefault(a, set()).update(
            t for t in n["properties"]["topics"] if t != "General")

    blocks = []
    for n in graph["nodes"]:
        if n["type"] != "Question" or n["properties"]["quarter"] != quarter:
            continue
        a = n["properties"]["analyst"]
        if analysts is not None and a not in analysts:
            continue
        blocks.append({
            "analyst": a,
            "topics": [t for t in n["properties"]["topics"] if t != "General"],
            "question_text": n["properties"]["text"],
            "prior_topics": sorted(prior_topics_by_analyst.get(a, set())),
        })
    return narration_text, blocks


def compute_question_intent(quarter: str, path: str = QUESTION_INTENT_PATH,
                            analysts: set[str] | None = None) -> list[dict]:
    graph = _load_graph()
    quarter_order = _quarter_order(graph)
    if quarter not in quarter_order:
        raise SystemExit(f"Unknown quarter '{quarter}'. Known: {quarter_order}")

    narration_text, blocks = _group_quarter_blocks(graph, quarter, quarter_order, analysts=analysts)
    if not blocks:
        print(f"[intent] {quarter}: no question blocks found for the given filter, skipping.")
        return []

    client = LLMClient()
    active = client.probe_llm()
    if not active:
        raise SystemExit("No working LLM API (set GROQ_API_KEY / GEMINI_API_KEY in .env).")

    raw = client.call_llm(_build_prompt(quarter, narration_text, blocks), temperature=0.0,
                          purpose="question_intent")
    if not raw:
        raise SystemExit(f"[intent] {quarter}: LLM call failed.")

    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    parsed = json.loads(clean)
    classifications = {c["index"]: c for c in parsed.get("classifications", [])}

    results = []
    for i, b in enumerate(blocks):
        c = classifications.get(i, {})
        intent = c.get("intent", "unexplained")
        if intent not in _VALID_INTENTS:
            intent = "unexplained"
        results.append({
            "quarter": quarter,
            "analyst": b["analyst"],
            "topics": b["topics"],
            "intent": intent,
            "evidence": str(c.get("evidence", ""))[:300],
        })

    # merge into existing file, replacing this quarter's entries
    history = []
    import os
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history = [h for h in history if h["quarter"] != quarter]
    history.extend(results)
    history.sort(key=lambda h: (h["quarter"], h["analyst"]))

    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    counts = {}
    for r in results:
        counts[r["intent"]] = counts.get(r["intent"], 0) + 1
    print(f"[intent] {quarter}: {len(results)} question blocks classified — {counts}")
    print(f"Saved -> {path}")
    return results


def compute_question_intent_recent(n: int = LAST_N_QUARTERS, path: str = QUESTION_INTENT_PATH,
                                   analysts: set[str] | None = None):
    graph = _load_graph()
    quarter_order = _quarter_order(graph)
    quarters = quarter_order[-n:]
    if analysts is not None:
        # only touch quarters where at least one target analyst actually asked something
        relevant = set()
        for node in graph["nodes"]:
            if (node["type"] == "Question" and node["properties"]["quarter"] in quarters
                    and node["properties"]["analyst"] in analysts):
                relevant.add(node["properties"]["quarter"])
        quarters = [q for q in quarters if q in relevant]
    for q in quarters:
        compute_question_intent(q, path=path, analysts=analysts)


if __name__ == "__main__":
    compute_question_intent_recent()
