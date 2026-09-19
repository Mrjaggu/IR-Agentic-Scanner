"""
narration_novelty.py — LLM-based semantic narration novelty signal.

Motivation: analysts probe what management NEWLY emphasizes (e.g. the Q4FY26
RAROC slide drew Profitability questions from Chintan Joshi and MB Mahesh).
Keyword counting cannot see this — profitability boilerplate appears in every
quarter's narration — so an LLM compares this quarter's presentation against
the prior quarters' and flags genuinely new/escalated themes.

Usage:
    python3 run.py novelty                 # target = VAL_QUARTER (q4fy26)
    python3 run.py novelty --quarter q3fy26

Output: data/inputs/narration_novelty.json
    {"quarter": "q4fy26",
     "novel_topics": [{"topic": "...", "novelty_score": 0.9, "evidence": "..."}]}

The engine picks this file up automatically at predict time (only if the
file's quarter matches VAL_QUARTER) and boosts scores by
W_NOVELTY * novelty_score. Delete/rename the file to disable.

VALIDATION DISCIPLINE: this signal is unproven. Measure before trusting:
    python3 run.py predict                       # with the file  -> note blind F1
    mv data/inputs/narration_novelty.json /tmp/  # remove it
    python3 run.py predict                       # without        -> compare
Keep it only if the blind block improves (or at minimum doesn't regress).
"""

import json
import re

from src.config.settings import (
    GRAPH_PATH, TOPICS_LIST, VAL_QUARTER, NOVELTY_PATH,
)
from src.model_provider.llm_client import LLMClient

NOV_LOOKBACK = 3          # prior quarters to compare against
MAX_CUR_CHARS = 7000      # current-quarter narration budget for the prompt
MAX_PREV_CHARS = 2000     # per prior quarter


def _load_narration(graph: dict, quarter: str) -> list[dict]:
    return [n["properties"] for n in graph["nodes"]
            if n["type"] == "NarrationSegment"
            and n["properties"]["quarter"] == quarter]


def _narr_text(segs: list[dict], budget: int) -> str:
    """Concatenate substantive narration segments (skip operator boilerplate)."""
    parts, used = [], 0
    for s in segs:
        txt = s["text"].strip()
        if not s.get("topics"):          # untagged segments are usually boilerplate
            continue
        take = txt[: max(0, budget - used)]
        if not take:
            break
        parts.append(take)
        used += len(take)
    return "\n---\n".join(parts)


def _build_prompt(quarter: str, cur_text: str, prev_blocks: list[tuple[str, str]]) -> str:
    topics_fmt = "\n".join(f"  - {t}" for t in TOPICS_LIST)
    prev_fmt = "\n\n".join(f"### {q} narration (prior quarter):\n{txt}"
                           for q, txt in prev_blocks)
    return f"""You are analyzing Axis Bank earnings-call opening narrations to find what management
NEWLY emphasizes this quarter — themes analysts are likely to probe in the Q&A.

### {quarter} narration (CURRENT quarter, the call being prepared for):
{cur_text}

{prev_fmt}

Task: Compare the CURRENT narration against the prior quarters. Identify topics where
management introduced a genuinely NEW theme, framework, metric, slide, or a sharply
escalated emphasis — NOT routine recurring boilerplate (quarterly RoE/NIM/growth stats
recited every quarter do NOT count as novel).

Good example of novelty: management presents a new RAROC-based segmental framework, a
new provisioning policy, a first-time guidance number, an acquisition update with new
specifics, or a strategy pivot.

Allowed topic labels (use EXACTLY these strings):
{topics_fmt}

Return JSON only:
{{
  "novel_topics": [
    {{"topic": "<one of the allowed labels>",
      "novelty_score": <0.0-1.0, how new/emphasized vs prior quarters>,
      "evidence": "<one sentence: what is new, quoting the narration>"}}
  ]
}}
Include ONLY topics with real novelty (typically 0-4 items). Empty list is a valid answer."""


def compute_novelty(quarter: str | None = None) -> dict:
    quarter = quarter or VAL_QUARTER
    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    # ordered quarter list from the graph's Quarter nodes
    q_nodes = sorted([n for n in graph["nodes"] if n["type"] == "Quarter"],
                     key=lambda n: n["properties"]["sort_key"])
    order = [n["id"] for n in q_nodes]
    if quarter not in order:
        raise SystemExit(f"Unknown quarter '{quarter}'. Known: {order}")
    qi = order.index(quarter)
    prev_qs = order[max(0, qi - NOV_LOOKBACK):qi]
    if not prev_qs:
        raise SystemExit(f"No prior quarters before {quarter} to compare against.")

    cur_text = _narr_text(_load_narration(graph, quarter), MAX_CUR_CHARS)
    prev_blocks = [(pq, _narr_text(_load_narration(graph, pq), MAX_PREV_CHARS))
                   for pq in prev_qs]

    client = LLMClient()
    active = client.probe_llm()
    if not active:
        raise SystemExit("No working LLM API (set GROQ_API_KEY / GEMINI_API_KEY in .env).")
    print(f"LLM: {active} | comparing {quarter} narration vs {prev_qs}")

    raw = client.call_llm(_build_prompt(quarter, cur_text, prev_blocks), temperature=0.0,
                          purpose="narration_novelty")
    if not raw:
        raise SystemExit("LLM call failed.")

    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    parsed = json.loads(clean)
    items = parsed.get("novel_topics", [])

    # validate against taxonomy, clamp scores
    valid = []
    for it in items:
        t = it.get("topic")
        if t in TOPICS_LIST:
            valid.append({
                "topic": t,
                "novelty_score": max(0.0, min(1.0, float(it.get("novelty_score", 0)))),
                "evidence": str(it.get("evidence", ""))[:300],
            })
        else:
            print(f"  [skip] LLM returned unknown topic label: {t!r}")

    out = {"quarter": quarter, "compared_against": prev_qs, "novel_topics": valid}
    with open(NOVELTY_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nNovel themes found: {len(valid)}")
    for v in valid:
        print(f"  {v['topic']:38s} score={v['novelty_score']:.2f}  {v['evidence'][:90]}")
    print(f"Saved -> {NOVELTY_PATH}")
    return out


if __name__ == "__main__":
    compute_novelty()
