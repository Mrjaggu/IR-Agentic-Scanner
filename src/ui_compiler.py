"""
ui_compiler.py — Section 9 (Application Layer / UI).

Compiles a single self-contained frontend/ir_platform_ui.html with three
tabs, matching the existing project convention (dashboard_gen.py) of a
static HTML file with embedded JSON rather than a live backend -- chosen
deliberately for Tab 2 as well: the whole corpus here is small (~580 text
nodes), so hybrid retrieval (BM25 + TF-IDF cosine + graph traversal + RRF)
runs client-side in the browser at query time, with no server and no
dependency on reachable LLM APIs (this session's network couldn't reach
Groq/Gemini -- see AGENTIC_ARCHITECTURE.md). Synthesis is extractive
(quoted, cited snippets), not generative, for the same reason: it always
works, and it satisfies the doc's citation requirement directly rather than
through an LLM's paraphrase.

Tab 1 — Analyst Profiles: direct structured browsing over the graph +
  persona data. No agent, no ranking model -- exactly what the doc specifies
  ("precise retrieval, not reasoning").
Tab 2 — Advanced Search: query understanding (filter extraction via the
  entity registry) + hybrid retrieval (BM25 + semantic proxy + graph
  traversal, fused via reciprocal rank fusion) + citation-attributed
  synthesis + table-vs-prose response formatting.
Tab 3 — Prediction: surfaces data/outputs/agentic_predictions.json (the
  Section 3 agentic core's output) as a topic brief + per-analyst question
  sheet + full audit trail (tool calls, verifier decisions).

Cross-bank coverage (Section 8) is NOT merged into this UI's graph -- the
two peer transcripts in earnings_transcript/peers/ are used only by
signals/peer_signal.py's separate salience computation, not ingested as
graph nodes. Tab 1's "bank" dimension is Axis Bank only; this is an honest
gap, not a hidden one (Section 8 itself flags cross-bank ingestion as the
most likely piece to slip).
"""

import json
import os

from src.config.settings import (
    BASE_DIR, GRAPH_PATH, DATASET_PATH, TOPICS_LIST, PERSONAS_PATH,
)
from src.data.loader import load_dataset, load_graph
from src.memory.history import PERSONAS
from src.signals.persona_synthesis import synthesize_personas

OUT_HTML = os.path.join(BASE_DIR, "frontend", "ir_platform_ui.html")
AGENTIC_JSON = os.path.join(BASE_DIR, "data", "outputs", "agentic_predictions.json")
AGENTIC_MD = os.path.join(BASE_DIR, "data", "outputs", "agentic_brief.md")


def _build_answer_context(graph: dict) -> dict:
    """ans_id -> {analyst, quarter, topics, speakers} of the question it answers."""
    q_by_id = {n["id"]: n["properties"] for n in graph["nodes"] if n["type"] == "Question"}
    ctx = {}
    speakers_by_ans = {}
    for e in graph["edges"]:
        if e["type"] == "ANSWERED_BY":
            # One answer can carry several ANSWERED_BY edges to the same person;
            # dedupe while preserving speaking order so citations read
            # "Puneet Sharma" rather than "Puneet Sharma, Puneet Sharma, ...".
            lst = speakers_by_ans.setdefault(e["source"], [])
            if e["target"] not in lst:
                lst.append(e["target"])
    for e in graph["edges"]:
        if e["type"] == "RESPONDED_TO" and e["target"] in q_by_id:
            qp = q_by_id[e["target"]]
            ctx[e["source"]] = {
                "analyst": qp["analyst"], "quarter": qp["quarter"], "topics": qp["topics"],
                "speakers": speakers_by_ans.get(e["source"], []),
            }
    return ctx


def compile_search_corpus(graph: dict) -> list[dict]:
    ans_ctx = _build_answer_context(graph)
    corpus = []
    for n in graph["nodes"]:
        if n["type"] == "Question":
            p = n["properties"]
            corpus.append({"id": n["id"], "type": "Question", "quarter": p["quarter"],
                           "analyst": p["analyst"], "topics": p["topics"], "text": p["text"]})
        elif n["type"] == "Answer":
            p = n["properties"]
            ctx = ans_ctx.get(n["id"], {})
            corpus.append({"id": n["id"], "type": "Answer", "quarter": p["quarter"],
                           "analyst": ctx.get("analyst"), "topics": p["topics"], "text": p["text"],
                           "speakers": ctx.get("speakers", [])})
        elif n["type"] == "NarrationSegment":
            p = n["properties"]
            corpus.append({"id": n["id"], "type": "NarrationSegment", "quarter": p["quarter"],
                           "analyst": None, "topics": p["topics"], "text": p["text"],
                           "speaker": p.get("speaker")})
    return corpus


def compile_analyst_profiles(dataset: list, graph: dict) -> list[dict]:
    persona_stats_full = synthesize_personas(exclude_quarters=None, out_path=None)
    all_analysts = {}
    for n in graph["nodes"]:
        if n["type"] != "Question":
            continue
        p = n["properties"]
        a = p["analyst"]
        if a in ("Moderator", "Operator"):
            continue
        rec = all_analysts.setdefault(a, {
            "analyst": a, "bank": "Axis Bank", "quarters": set(),
            "topic_counts": {}, "questions": [],
        })
        rec["quarters"].add(p["quarter"])
        for t in p["topics"]:
            if t != "General":
                rec["topic_counts"][t] = rec["topic_counts"].get(t, 0) + 1
        rec["questions"].append({"quarter": p["quarter"], "text": p["text"], "topics": p["topics"]})

    profiles = []
    for a, rec in all_analysts.items():
        rec["quarters"] = sorted(rec["quarters"])
        rec["questions"].sort(key=lambda q: q["quarter"])
        rec["hand_curated_persona"] = PERSONAS.get(a)
        rec["measured_style"] = persona_stats_full.get(a)
        profiles.append(rec)
    profiles.sort(key=lambda r: -len(r["questions"]))
    return profiles


def _load_agentic_output() -> dict | None:
    if not os.path.exists(AGENTIC_JSON):
        return None
    with open(AGENTIC_JSON) as f:
        return json.load(f)


def main():
    dataset = load_dataset()
    graph = load_graph()
    quarter_order = [q["quarter_id"] for q in dataset]

    profiles = compile_analyst_profiles(dataset, graph)
    corpus = compile_search_corpus(graph)
    agentic = _load_agentic_output()

    ui_data = {
        "quarters": quarter_order,
        "topics": TOPICS_LIST,
        "analyst_profiles": profiles,
        "search_corpus": corpus,
        "agentic": agentic,
    }

    with open(os.path.join(os.path.dirname(__file__), "..", "frontend", "_ui_template.html")) as f:
        template = f.read()

    html = template.replace("__UI_DATA_JSON__", json.dumps(ui_data))

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"[ui] {len(profiles)} analyst profiles, {len(corpus)} search-corpus docs, "
          f"agentic data: {'present' if agentic else 'MISSING -- run `python3 run.py agentic` first'}")
    print(f"[ui] Wrote {OUT_HTML} ({os.path.getsize(OUT_HTML)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
