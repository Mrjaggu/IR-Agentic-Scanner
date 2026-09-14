"""
fastapi_app.py — the live backend behind the dashboard.

Serves one page and the endpoints it needs: analyst dossiers (full Q&A threads
with named responders and evidence-based rationale), hybrid search, a
streaming chat, the three prediction run modes (Overall layer / one analyst /
full pipeline), upcoming-quarter disclosure upload, the held-out evaluation,
and ingestion preview/commit.

Run with: `python3 run.py app` then open http://localhost:8000/

Design notes that matter for a regulated deployment:
  - No CDN, no external calls from the page. Everything is served locally,
    so this works air-gapped.
  - Every expensive computation is cached in-process and invalidated
    explicitly (an ingest commit rebuilds the data cache).
  - Chat streams over SSE. When no LLM is reachable it streams the grounded
    source passages instead of fabricating an answer.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.config.settings import (
    BASE_DIR, VAL_QUARTER, TOPICS_LIST, TEST_QUARTERS, TRAIN_CUTOFF, CUTOFF_MODE,
)
from src.data.loader import load_dataset, load_graph
from src.data.ingest_test import preview_ingest, commit_ingest
from src.data.dossier import build_analyst_dossier
from src.data.upcoming import parse_upcoming_document
from src.ui_compiler import compile_analyst_profiles, compile_search_corpus
from src.search.hybrid_search import SearchIndex
from src.agentic.run_agentic import build_initial_state, main as run_agentic_full, OUT_JSON as AGENTIC_JSON
from src.agentic.planning_agent import run_planning_agent
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst, build_arithmetic_followups
from src.agentic.verifier import grounding_gate
from src.agentic.question_framer import build_evidence_pool
from src.agentic.eval_harness import (
    run_holdout_eval, evaluate_quarter, research_errors, backtest_and_promote_weights,
)
from src.model_provider.llm_client import client, stats as llm_stats, reset_stats as llm_reset, RATE_LIMITS

APP_HTML_PATH = os.path.join(BASE_DIR, "frontend", "ir_platform_app.html")

app = FastAPI(title="IR Question-Intelligence Platform")

# ── Caches ──────────────────────────────────────────────────────────────────
_cache: dict = {}
_dossier_cache: dict[str, dict] = {}
_overall_cache: dict[str, dict] = {}
_eval_cache: dict[str, dict] = {}
_disclosures: dict[str, dict] = {}


def _load_live():
    dataset = load_dataset()
    graph = load_graph()
    _cache.clear()
    _cache.update({
        "dataset": dataset,
        "graph": graph,
        "profiles": compile_analyst_profiles(dataset, graph),
        "corpus": compile_search_corpus(graph),
        "quarters": [q["quarter_id"] for q in dataset],
    })
    _cache["index"] = SearchIndex(_cache["corpus"])
    _dossier_cache.clear()
    _overall_cache.clear()
    _eval_cache.clear()
    return _cache


def _live():
    if not _cache:
        _load_live()
    return _cache


@app.on_event("startup")
def _startup():
    _load_live()
    client.probe_llm()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(APP_HTML_PATH) as f:
        return f.read()


def _topic_recall_pct() -> float | None:
    try:
        mean_recall = _cached_holdout()["summary"]["mean_recall"]
        return round(mean_recall * 100, 1)
    except Exception:
        return None


# ── Meta / data ─────────────────────────────────────────────────────────────
@app.get("/api/meta")
def get_meta():
    live = _live()
    from src.data.loader import get_active_analysts
    try:
        n_active = len(get_active_analysts(live["dataset"], VAL_QUARTER))
    except Exception:
        n_active = 0

    # Real, unfabricated freshness signal for the workspace context bar: when
    # the most recent call actually happened (from the transcript itself) and
    # when the archive file on disk last changed. No calendar integration
    # exists, so there is no live "next call in N days" countdown here — that
    # would be invented rather than measured.
    latest = max(live["dataset"], key=lambda r: r.get("sort_key", 0)) if live["dataset"] else None
    from src.config.settings import DATASET_PATH
    try:
        archive_refreshed = datetime.fromtimestamp(os.path.getmtime(DATASET_PATH)).isoformat()
    except OSError:
        archive_refreshed = None

    return {
        "quarters": live["quarters"],
        "topics": TOPICS_LIST,
        "test_quarters": TEST_QUARTERS,
        "train_cutoff": TRAIN_CUTOFF,
        "cutoff_mode": CUTOFF_MODE,
        "default_quarter": VAL_QUARTER,
        "active_analysts": n_active,
        "llm": {"active": client.active_llm, "available": bool(client.active_llm),
                "limits": RATE_LIMITS},
        "counts": {
            "analysts": len(live["profiles"]),
            "quarters": len(live["quarters"]),
            "corpus_docs": len(live["corpus"]),
        },
        "latest_reported_quarter": latest.get("quarter_id") if latest else None,
        "latest_call_date": latest.get("call_date") if latest else None,
        "archive_last_refreshed": archive_refreshed,
        "topic_recall_pct": _topic_recall_pct(),
    }


# ── Saved work: prep brief + recent activity ────────────────────────────────
# The one part of the review's "saved work" gap that is actually
# infrastructure, not UI: pinned items and activity persist to a JSON file on
# disk (src/data/workspace_state.py) so a browser reload or a server restart
# doesn't lose them. This is a single-workspace, single-user tool, so a flat
# file with a process lock is the right amount of engineering here.
from src.data.workspace_state import (
    get_state, pin_item, unpin_item, clear_brief, log_activity, brief_as_markdown,
)


class PinRequest(BaseModel):
    kind: str
    title: str
    body: str
    meta: dict | None = None


class ActivityRequest(BaseModel):
    kind: str
    label: str
    meta: dict | None = None


@app.get("/api/brief")
def api_get_brief():
    return get_state()


@app.post("/api/brief/pin")
def api_pin(req: PinRequest):
    return pin_item(req.kind, req.title, req.body, req.meta)


@app.delete("/api/brief/{item_id}")
def api_unpin(item_id: str):
    return unpin_item(item_id)


@app.post("/api/brief/clear")
def api_clear_brief():
    return clear_brief()


@app.get("/api/brief/export")
def api_export_brief(quarter: str = ""):
    from fastapi.responses import PlainTextResponse
    md = brief_as_markdown(quarter)
    return PlainTextResponse(md, headers={
        "Content-Disposition": f'attachment; filename="prep-brief-{quarter or "axis"}.md"'
    })


@app.post("/api/activity")
def api_log_activity(req: ActivityRequest):
    return log_activity(req.kind, req.label, req.meta)


@app.get("/api/activity")
def api_get_activity():
    return get_state()["activity"][::-1]   # newest first


@app.get("/api/analysts")
def get_analysts():
    """Roster for the profile list: enough to render rows without the full
    question payload for all 43 analysts."""
    live = _live()
    out = []
    for p in live["profiles"]:
        top = sorted(p["topic_counts"].items(), key=lambda x: -x[1])[:3]
        out.append({
            "analyst": p["analyst"],
            "bank": p["bank"],
            "n_questions": len(p["questions"]),
            "quarters": p["quarters"],
            "top_topics": [t for t, _ in top],
            "firm": (p.get("hand_curated_persona") or {}).get("firm"),
            "style": (p.get("hand_curated_persona") or {}).get("style"),
            "measured_style": (p.get("measured_style") or {}).get("style_note"),
        })
    return out


@app.get("/api/dossier")
def get_dossier(analyst: str):
    """Full Q&A threads: the analyst's complete question, each named management
    responder, their own follow-up reaction, and why they asked."""
    if analyst in _dossier_cache:
        return _dossier_cache[analyst]
    live = _live()
    d = build_analyst_dossier(analyst, live["dataset"], live["graph"])
    profile = next((p for p in live["profiles"] if p["analyst"] == analyst), None)
    if profile:
        d["topic_counts"] = profile["topic_counts"]
        d["hand_curated_persona"] = profile.get("hand_curated_persona")
        d["measured_style"] = profile.get("measured_style")
    _dossier_cache[analyst] = d
    return d


# ── Search ──────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str


def _extract_filters(query: str, analyst_names: list[str]) -> dict:
    import re
    ql = query.lower()
    f = {"analyst": None, "quarter": None, "topic": None, "intent": "lookup"}
    for name in analyst_names:
        if name.lower() in ql:
            f["analyst"] = name
            break
        last = name.split(" ")[-1].lower()
        if len(last) > 3 and last in ql:
            f["analyst"] = name
    qm = re.search(r"q([1-4])\s*fy\s*(\d{2})", ql)
    if qm:
        f["quarter"] = f"q{qm.group(1)}fy{qm.group(2)}"
    if re.search(r"who (else )?(covers|asks|raised|ask)|who else", ql):
        f["intent"] = "relational"
    if re.search(r"compare|across quarters|trend|over time", ql):
        f["intent"] = "comparative"
    for t in TOPICS_LIST:
        if t.lower() in ql:
            f["topic"] = t
    return f


def _who(doc: dict) -> str:
    if doc["type"] == "Question":
        return doc.get("analyst") or "Unknown analyst"
    if doc["type"] == "Answer":
        # Defensive dedupe: one answer can carry several ANSWERED_BY edges to
        # the same person, and a citation reading the same name five times is
        # worse than no attribution.
        seen = list(dict.fromkeys(doc.get("speakers", []) or []))
        return ", ".join(seen) or "Management"
    return doc.get("speaker") or "Management"


def _retrieve(query: str, filters: dict, top_k: int = 12) -> list[dict]:
    live = _live()
    index: SearchIndex = live["index"]
    pool = list(range(len(index.docs)))
    if filters["analyst"]:
        pool = [i for i in pool if index.docs[i].get("analyst") == filters["analyst"]]
    if filters["quarter"]:
        pool = [i for i in pool if index.docs[i]["quarter"] == filters["quarter"]]
    hits = index.search(query, candidate_idx=pool, top_k=top_k)
    return [{
        "quarter": h["doc"]["quarter"], "type": h["doc"]["type"], "who": _who(h["doc"]),
        "text": h["doc"]["text"], "topics": [t for t in h["doc"]["topics"] if t != "General"],
        "score": round(h["score"], 4),
    } for h in hits]


@app.post("/api/search")
def api_search(req: SearchRequest):
    live = _live()
    names = [p["analyst"] for p in live["profiles"]]
    filters = _extract_filters(req.query, names)

    if filters["intent"] == "relational" and (filters["topic"] or filters["analyst"]):
        topic = filters["topic"]
        if not topic and filters["analyst"]:
            p = next((p for p in live["profiles"] if p["analyst"] == filters["analyst"]), None)
            if p and p["topic_counts"]:
                topic = max(p["topic_counts"], key=p["topic_counts"].get)
        if topic:
            rows = sorted(
                [[p["analyst"], p["topic_counts"].get(topic, 0)] for p in live["profiles"]
                 if p["topic_counts"].get(topic, 0) > 0], key=lambda r: -r[1])
            return {"filters": filters, "mode": "relational_table",
                    "table": {"title": f'Analysts who have raised "{topic}"',
                              "columns": ["Analyst", "Times raised"], "rows": rows},
                    "passages": _retrieve(req.query, filters)}

    if filters["intent"] == "comparative" and filters["topic"]:
        by_q = {q: 0 for q in live["quarters"]}
        for d in live["corpus"]:
            if d["type"] == "Question" and filters["topic"] in d["topics"]:
                by_q[d["quarter"]] = by_q.get(d["quarter"], 0) + 1
        return {"filters": filters, "mode": "comparative_table",
                "table": {"title": f'"{filters["topic"]}" questions by quarter',
                          "columns": ["Quarter", "Questions"], "rows": [[q, n] for q, n in by_q.items()]},
                "passages": _retrieve(req.query, filters)}

    return {"filters": filters, "mode": "prose", "table": None,
            "passages": _retrieve(req.query, filters)}


# ── Chat (SSE streaming) ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


def _build_chat_prompt(message: str, passages: list[dict], history: list[dict] | None) -> str:
    evidence = "\n\n".join(
        f'[{i + 1}] ({p["quarter"]} · {p["type"]} · {p["who"]}) "{p["text"][:900]}"'
        for i, p in enumerate(passages))
    hist = ""
    if history:
        hist = "\n".join(f'{h["role"]}: {h["content"]}' for h in history[-6:]) + "\n\n"
    return (
        "You are an analyst-relations assistant for Axis Bank's IR team, answering "
        "questions about the earnings-call transcript archive.\n"
        "Use ONLY the evidence passages below. Do not use outside knowledge and do not "
        "invent numbers. Cite passages inline as [n]. Use markdown: short paragraphs, "
        "bullet lists where it helps, and a table when comparing several rows.\n\n"
        f"{hist}Evidence:\n{evidence}\n\nQuestion: {message}\n\n"
        'Return JSON only: {"answer": "<your grounded markdown answer with [n] citations>"}'
    )


def _extractive_answer(message: str, passages: list[dict]) -> str:
    """No LLM reachable: say so plainly and hand over the grounded passages
    rather than fabricating an answer."""
    lines = [
        "_No language model is reachable from this environment, so this is a "
        "retrieval-only answer — the passages below are the actual matches from "
        "the transcript archive, unsummarised._\n",
    ]
    for i, p in enumerate(passages[:5], 1):
        snippet = p["text"][:420].strip()
        lines.append(f"**[{i}] {p['quarter']} · {p['who']}**\n\n> {snippet}…\n")
    return "\n".join(lines)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    live = _live()
    names = [p["analyst"] for p in live["profiles"]]
    filters = _extract_filters(req.message, names)
    passages = _retrieve(req.message, filters, top_k=8)

    async def gen():
        def sse(event: str, data) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("meta", {"filters": filters, "n_passages": len(passages)})

        if not passages:
            yield sse("delta", {"text": "I couldn't find anything in the transcript archive "
                                        "relevant to that. Try naming an analyst, a quarter, "
                                        "or one of the taxonomy topics."})
            yield sse("done", {"mode": "empty"})
            return

        mode = "llm" if (client.active_llm or client.probe_llm()) else "extractive"
        answer = None
        if mode == "llm":
            raw = await asyncio.to_thread(
                client.call_llm, _build_chat_prompt(req.message, passages, req.history), 0.1)
            if raw:
                import re
                clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
                try:
                    answer = json.loads(clean).get("answer")
                except Exception:
                    answer = raw
            if not answer:
                mode = "extractive"
        if mode == "extractive":
            answer = _extractive_answer(req.message, passages)

        yield sse("mode", {"mode": mode})

        # Stream in word groups so the surface feels live rather than blocking
        # on one big payload.
        words = answer.split(" ")
        chunk = 6
        for i in range(0, len(words), chunk):
            yield sse("delta", {"text": " ".join(words[i:i + chunk]) + " "})
            await asyncio.sleep(0.012)

        yield sse("citations", passages)
        yield sse("done", {"mode": mode})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Disclosure upload (upcoming quarter) ───────────────────────────────────
@app.post("/api/disclosure/parse")
async def disclosure_parse(file: UploadFile = File(...), holdout: str = Form("true")):
    """Parse an upcoming-quarter draft script / investor presentation /
    transcript into prediction inputs. Nothing is written to the archive."""
    live = _live()
    order = live["quarters"]
    use_holdout = holdout.lower() in ("1", "true", "yes")
    cutoff = order.index(TRAIN_CUTOFF) if TRAIN_CUTOFF in order else len(order) - 1
    history = order[:cutoff + 1] if use_holdout else order

    data = await file.read()
    parsed = parse_upcoming_document(data, file.filename, history)
    if parsed.get("error"):
        return JSONResponse(status_code=400, content=parsed)

    did = uuid.uuid4().hex[:12]
    _disclosures[did] = parsed
    return {
        "disclosure_id": did,
        "filename": parsed["filename"],
        "chars_extracted": parsed["chars_extracted"],
        "metrics": parsed["metrics"],
        "anomaly_scores": parsed["anomaly_scores"],
        "topic_salience": parsed["topic_salience"],
        "drill_down_flags": parsed["drill_down_flags"],
        "novel_themes": parsed.get("novel_themes", []),
        "summary": parsed["summary"],
        "history_pool": {"holdout": use_holdout, "n_quarters": len(history), "through": history[-1]},
    }


# ── Prediction runs ─────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    quarter: str | None = None
    disclosure_id: str | None = None
    holdout: bool = False


class AnalystRunRequest(RunRequest):
    analyst: str


def _disclosure(req: RunRequest) -> dict | None:
    return _disclosures.get(req.disclosure_id) if req.disclosure_id else None


@app.get("/api/llm/stats")
def get_llm_stats():
    """Polled by the dashboard while a prediction runs, so a long wait reads as
    'throttled by the free-tier limit' rather than as a hang."""
    return llm_stats()


@app.get("/api/agentic")
def get_agentic():
    if not os.path.exists(AGENTIC_JSON):
        return {"present": False}
    with open(AGENTIC_JSON) as f:
        return {"present": True, "data": json.load(f)}


@app.post("/api/run/overall")
def run_overall(req: RunRequest):
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    disclosure = _disclosure(req)
    state = build_initial_state(quarter, holdout=req.holdout, upcoming=disclosure)
    bundles, tool_log = run_planning_agent(state["anomaly_scores"], state["graph"],
                                           state["prior_quarters"], state["global_rate"])
    overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                   state["momentum"], state["client"], disclosure=disclosure)
    key = f"{quarter}:{req.disclosure_id or 'none'}:{req.holdout}"
    _overall_cache[key] = {"state": state, "overall": overall, "tool_log": tool_log}
    return {"quarter": quarter, "cache_key": key, "tool_log": tool_log, "overall": overall,
            "train_cutoff": state.get("train_cutoff"),
            "disclosure_conditioned": bool(disclosure), "holdout": req.holdout,
            "llm_stats": llm_stats()}


@app.post("/api/run/analyst")
def run_analyst(req: AnalystRunRequest):
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    key = f"{quarter}:{req.disclosure_id or 'none'}:{req.holdout}"
    if key not in _overall_cache:
        run_overall(RunRequest(quarter=quarter, disclosure_id=req.disclosure_id, holdout=req.holdout))
    cached = _overall_cache[key]
    state, overall = cached["state"], cached["overall"]
    disclosure = _disclosure(req)

    if req.analyst not in state["analyst_prefs"]:
        return JSONResponse(status_code=404,
                            content={"error": f"{req.analyst} is not an active analyst for {quarter}"})

    pref, N = state["analyst_prefs"][req.analyst]
    ranked = reweight_for_analyst(req.analyst, overall["ranked_topics"], pref, N, disclosure=disclosure)
    style = state["persona_stats"].get(req.analyst, {}).get("style_note", "no measured style profile")
    pool = build_evidence_pool(req.analyst, ranked, state["graph"], set(state["prior_quarters"]),
                               state["global_rate"], state["anomaly_scores"],
                               disclosed_metrics=state.get("val_quarter_metrics"),
                               target_quarter=quarter,
                               disclosure_text=(disclosure or {}).get("narration", ""))
    results, gate_log = grounding_gate(req.analyst, style, ranked, pool, state["client"],
                                       target_quarter=quarter)
    return {"quarter": quarter, "analyst": req.analyst, "style_note": style,
            "train_cutoff": state.get("train_cutoff"),
            "topics": results, "verifier_log": gate_log,
            "disclosure_conditioned": bool(disclosure), "llm_stats": llm_stats()}


@app.post("/api/run/full")
def run_full(req: RunRequest):
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    disclosure = _disclosure(req)
    if not disclosure and not req.holdout:
        run_agentic_full(quarter)
        log_activity("prepare_call", f"Prepared {quarter}", {"quarter": quarter, "mode": "full"})
        with open(AGENTIC_JSON) as f:
            return {"quarter": quarter, "data": json.load(f), "disclosure_conditioned": False,
                    "llm_stats": llm_stats()}

    # Disclosure-conditioned or held-out runs stay in memory rather than
    # overwriting the archive's committed prediction file.
    from src.agentic.graph_app import run_pipeline
    state = build_initial_state(quarter, holdout=req.holdout, upcoming=disclosure)
    result = run_pipeline(state)
    data = {
        "quarter": quarter,
        "planning_agent_tool_log": result["tool_log"],
        "overall_topics": {"ranked": result["overall"]["ranked_topics"],
                           "rationale": result["overall"]["rationale"],
                           "verifier_log": result["overall"]["verifier_log"],
                           "composite_scores": result["overall"].get("composite_scores", {})},
        "arithmetic_followups": result["arithmetic_followups"],
        "analyst_predictions": {a: {"topics": o["topics"], "verifier_log": o["verifier_log"]}
                                for a, o in result["analyst_outputs"].items()},
    }
    log_activity("prepare_call", f"Prepared {quarter}",
                {"quarter": quarter, "mode": "full", "disclosure_conditioned": bool(disclosure),
                 "holdout": req.holdout})
    return {"quarter": quarter, "data": data, "disclosure_conditioned": bool(disclosure),
            "holdout": req.holdout, "persisted": False, "llm_stats": llm_stats()}


# ── Evaluation ──────────────────────────────────────────────────────────────
def _cached_holdout(with_questions: bool = False, refresh: bool = False) -> dict:
    key = f"holdout:{with_questions}"
    if refresh or key not in _eval_cache:
        _eval_cache[key] = run_holdout_eval(with_questions=with_questions)
    return _eval_cache[key]


@app.get("/api/eval/holdout")
def eval_holdout(with_questions: bool = False, refresh: bool = False):
    """The headline numbers: q4fy26 and q1fy27 scored separately against a
    q3fy26 training cutoff, plus the promotion gate."""
    return _cached_holdout(with_questions=with_questions, refresh=refresh)


class EvalCompareRequest(BaseModel):
    disclosure_ids: dict[str, str] | None = None


@app.post("/api/eval/compare")
def eval_compare(req: EvalCompareRequest):
    """History-only vs disclosure-conditioned on the held-out quarters, so the
    effect of the upload is a measured number rather than a claim."""
    out = {}
    for q in TEST_QUARTERS:
        base = evaluate_quarter(q, holdout=True)
        row = {"history_only": {"macro": base["macro"],
                                "misses": base["failure_attribution"]["counts"],
                                "ranked": base["topic_ranking"]["ranked_topics"],
                                "actual": base["topic_ranking"]["actual_topics"]}}
        did = (req.disclosure_ids or {}).get(q)
        if did and did in _disclosures:
            cond = evaluate_quarter(q, holdout=True, upcoming=_disclosures[did])
            row["disclosure_conditioned"] = {"macro": cond["macro"],
                                             "misses": cond["failure_attribution"]["counts"],
                                             "ranked": cond["topic_ranking"]["ranked_topics"]}
        out[q] = row
    return {"train_cutoff": TRAIN_CUTOFF, "per_quarter": out}


@app.get("/api/eval/error-report")
def eval_error_report(with_questions: bool = False, refresh: bool = False):
    """The 'Error Researcher': aggregates the held-out eval's per-miss failure
    attribution into a framework-level diagnosis -- which error category
    dominates, whether it's even fixable by reweighting, and which specific
    (analyst, topic) pairs are driving it. Reuses the same cached holdout run
    as /api/eval/holdout; no extra LLM calls."""
    holdout = _cached_holdout(with_questions=with_questions, refresh=refresh)
    return research_errors(holdout)


class WeightBacktestRequest(BaseModel):
    weights: dict[str, float]
    with_questions: bool = False


@app.post("/api/eval/backtest-weights")
def eval_backtest_weights(req: WeightBacktestRequest):
    """The 'Framework Loop': backtests a candidate composite-weight override
    (anomaly/disclosure/momentum/base_rate/drill_flag) against the SAME
    held-out quarters and PromotionGate thresholds production uses, and
    returns PROMOTE/REJECT -- it never writes to overall_layer.py's own
    defaults. Uses each held-out quarter's own real narration as a synthetic
    disclosure so disclosure/drill_flag weights are actually exercised (a
    plain holdout run passes no disclosure at all, so those two weights would
    otherwise always score identically to baseline)."""
    valid_keys = {"anomaly", "disclosure", "momentum", "base_rate", "drill_flag"}
    unknown = set(req.weights) - valid_keys
    if unknown:
        return JSONResponse(status_code=400,
                            content={"error": f"unknown weight key(s): {sorted(unknown)}",
                                     "valid_keys": sorted(valid_keys)})
    return backtest_and_promote_weights(req.weights, with_questions=req.with_questions)


# ── Ingestion ───────────────────────────────────────────────────────────────
@app.post("/api/ingest/preview")
async def ingest_preview(file: UploadFile = File(...)):
    return preview_ingest(await file.read(), file.filename)


@app.post("/api/ingest/commit")
async def ingest_commit(file: UploadFile = File(...), overwrite: str = Form("false")):
    was_overwrite = overwrite.lower() in ("1", "true", "yes")
    result = commit_ingest(await file.read(), file.filename, overwrite=was_overwrite)
    if result.get("status") == "ok":
        _load_live()
    # Real audit entry for a real write to the archive — this is the one
    # action in the app that isn't a read, so it is the one that most needs a
    # record of who/when/what, per the review's governance gap.
    log_activity(
        "ingest_commit" if result.get("status") == "ok" else "ingest_commit_failed",
        f"Committed {file.filename}" if result.get("status") == "ok"
        else f"Commit failed: {file.filename}",
        {"quarter": result.get("detected_quarter_id"), "overwrite": was_overwrite,
         "status": result.get("status")},
    )
    return result


def run_app(port: int = 8000):
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
