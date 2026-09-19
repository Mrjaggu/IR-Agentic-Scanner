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
import threading
import uuid
from collections import OrderedDict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from pydantic import BaseModel

from src.config.settings import (
    BASE_DIR, VAL_QUARTER, TOPICS_LIST, TEST_QUARTERS, TRAIN_CUTOFF, CUTOFF_MODE, paths_for,
)
from src.config.banks import BANKS, DEFAULT_BANK
from src.data.loader import load_dataset, load_graph
from src.data.ingest_test import preview_ingest, commit_ingest
from src.data.dossier import build_analyst_dossier
from src.data.upcoming import parse_upcoming_document
from src.ui_compiler import compile_analyst_profiles, compile_search_corpus
from src.search.hybrid_search import SearchIndex
from src.search import embeddings as search_embeddings
from src.agentic.run_agentic import (
    build_initial_state, main as run_agentic_full, _out_paths as _agentic_out_paths,
)
from src.agentic.planning_agent import run_planning_agent
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst, build_arithmetic_followups
from src.agentic.verifier import grounding_gate
from src.agentic import pattern_retrieval
from src.tts import piper_tts
from src.news import news_api
from src.signals.metrics_extractor import METRIC_TOPIC_MAP
from src.agentic.question_framer import build_evidence_pool
from src.agentic.eval_harness import (
    run_holdout_eval, evaluate_quarter, research_errors, backtest_and_promote_weights,
)
from src.agentic.skills import definitions as _skills_definitions  # noqa: F401 -- populates the skill registry
from src.agentic.skills.registry import list_skills

from src.model_provider.llm_client import client, stats as llm_stats, reset_stats as llm_reset, RATE_LIMITS

APP_HTML_PATH = os.path.join(BASE_DIR, "frontend", "ir_platform_app.html")

app = FastAPI(title="IR Question-Intelligence Platform")

# ── Caches ──────────────────────────────────────────────────────────────────
# 2026-09: every cache below is now keyed by bank_id (a plain dict-of-dicts,
# or a "<bank_id>:..." string-prefixed key for the flatter eval/overall
# caches) instead of holding exactly one bank's data -- so switching banks in
# the UI doesn't require a server restart, and one bank's cached run never
# leaks into another's. _disclosures stays global: a disclosure_id is already
# a random uuid, unique regardless of which bank it was parsed for.
_cache: dict[str, dict] = {}
_dossier_cache: dict[str, dict[str, dict]] = {}
_overall_cache: dict[str, dict] = {}
_eval_cache: dict[str, dict] = {}
_disclosures: dict[str, dict] = {}
# TTS "Listen" audio, keyed by a uuid handed to the client in the chat
# SSE stream's "audio" event. None means synthesis is still running
# (populated the instant the background task is scheduled, so the /audio
# endpoint can tell "still working" apart from "no such id" -- see
# chat_stream()/get_tts_audio() below). Bounded LRU, not a TTL cache: a long
# chat session can generate plenty of these, and nothing here is sensitive
# enough to need eviction on any faster schedule than "least recently made."
_tts_cache: "OrderedDict[str, bytes | None]" = OrderedDict()
_TTS_CACHE_MAX = 40


def _bank(bank_id: str) -> str:
    """Validates a bank query/body param, raising a clean 400 instead of the
    500 get_bank()'s ValueError would give. Call at the top of every route
    that takes a `bank` param, before touching any bank-scoped cache."""
    if bank_id not in BANKS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown bank {bank_id!r}. Valid: {sorted(BANKS)}")
    return bank_id


def _load_live(bank_id: str = DEFAULT_BANK) -> dict:
    paths = paths_for(bank_id)
    dataset = load_dataset(dataset_path=paths.dataset_path)
    graph = load_graph(graph_path=paths.graph_path)
    data = {
        "dataset": dataset,
        "graph": graph,
        "profiles": compile_analyst_profiles(dataset, graph, bank_name=BANKS[bank_id].display_name),
        "corpus": compile_search_corpus(graph),
        "quarters": [q["quarter_id"] for q in dataset],
    }
    # None (not a list of zero vectors) when the embedding model isn't
    # installed on this machine -- SearchIndex then just runs BM25+TF-IDF,
    # same as before this signal existed. See src/search/embeddings.py.
    embed_vecs = search_embeddings.embed_many([d["text"] for d in data["corpus"]])
    data["index"] = SearchIndex(data["corpus"], embed_vecs=embed_vecs)
    _cache[bank_id] = data
    _dossier_cache.pop(bank_id, None)
    for k in [k for k in _overall_cache if k.startswith(f"{bank_id}:")]:
        del _overall_cache[k]
    for k in [k for k in _eval_cache if k.startswith(f"{bank_id}:")]:
        del _eval_cache[k]
    return data


def _live(bank_id: str = DEFAULT_BANK) -> dict:
    if bank_id not in _cache:
        _load_live(bank_id)
    return _cache[bank_id]


def _warm_bank(bank_id: str) -> None:
    """Background warmup for a non-default bank -- see _startup(). Swallows
    everything: a thin/new bank with no compiled dataset yet (or one whose
    holdout eval genuinely can't run on too little data, per the eval_harness
    docstring) should fail quietly here and just fall back to lazy-loading on
    its first real request, exactly as it did before this warmup existed."""
    try:
        _load_live(bank_id)
        _cached_holdout(bank_id=bank_id)
    except Exception:
        pass


@app.on_event("startup")
def _startup():
    # The default bank loads eagerly and blocks startup -- it's the one
    # everyone hits first. Every OTHER registered bank is warmed in a
    # background thread instead of being left to load lazily on whichever
    # request first switches to it: that lazy path is what made switching
    # workspaces feel laggy (reported: Axis -> Kotak -> IndusInd, ~6-8s) --
    # _load_live() alone (dataset+graph load, profile compile, corpus
    # embedding) measured ~5s for Kotak and ~2.4s for IndusInd cold, all of
    # it spent inside the /api/meta call the frontend's switchBank() awaits
    # before it will even show the workspace. A thin/new bank (e.g. one
    # added to the registry with no transcripts compiled yet) still can't
    # crash startup for everyone -- _warm_bank() swallows its own failures
    # and that bank just falls back to lazy-loading on first request, same
    # as before this warmup existed.
    _load_live(DEFAULT_BANK)
    client.probe_llm()
    other_banks = [b for b in BANKS if b != DEFAULT_BANK]
    if other_banks:
        threading.Thread(
            target=lambda: [_warm_bank(b) for b in other_banks],
            daemon=True, name="bank-warmup",
        ).start()


@app.get("/api/banks")
def get_banks():
    """Registry introspection for the frontend's bank selector: which banks
    exist, and how much data each one actually has, so a thin/new bank (e.g.
    freshly registered with no transcripts compiled yet) can be shown as such
    rather than silently 500ing when selected."""
    out = []
    for bank_id, cfg in BANKS.items():
        paths = paths_for(bank_id)
        out.append({
            "bank_id": bank_id,
            "display_name": cfg.display_name,
            "has_dataset": os.path.exists(paths.dataset_path),
            "has_graph": os.path.exists(paths.graph_path),
            "has_personas": os.path.exists(paths.personas_path),
        })
    return {"banks": out, "default_bank": DEFAULT_BANK}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(APP_HTML_PATH) as f:
        return f.read()


def _topic_recall_pct(bank_id: str = DEFAULT_BANK) -> float | None:
    try:
        mean_recall = _cached_holdout(bank_id=bank_id)["summary"]["mean_recall"]
        return round(mean_recall * 100, 1)
    except Exception:
        return None


# ── Meta / data ─────────────────────────────────────────────────────────────
@app.get("/api/meta")
def get_meta(bank: str = DEFAULT_BANK):
    bank_id = _bank(bank)
    live = _live(bank_id)
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
    try:
        archive_refreshed = datetime.fromtimestamp(
            os.path.getmtime(paths_for(bank_id).dataset_path)).isoformat()
    except OSError:
        archive_refreshed = None

    return {
        "bank": bank_id,
        "display_name": BANKS[bank_id].display_name,
        "quarters": live["quarters"],
        "topics": TOPICS_LIST,
        "test_quarters": TEST_QUARTERS,
        "train_cutoff": TRAIN_CUTOFF,
        "cutoff_mode": CUTOFF_MODE,
        "default_quarter": VAL_QUARTER,
        "active_analysts": n_active,
        "llm": {"active": client.active_llm, "available": bool(client.active_llm),
                "limits": RATE_LIMITS},
        # Same honesty pattern as "llm" above: whether this run actually has
        # a semantic signal in search/chat retrieval, or is BM25+TF-IDF only
        # because the embedding model isn't installed on this machine.
        "retrieval": {"semantic": live["index"].has_semantic,
                      "semantic_model": "en_core_web_md (spaCy, 300d GloVe)" if live["index"].has_semantic else None,
                      "cognitive_patterns": pattern_retrieval.is_available(bank_id)},
        # Whether the chat "Listen" button has a voice to speak with -- see
        # src/tts/piper_tts.py's docstring for why this can be False (voice
        # model not installed yet) and how honestly that degrades: no error,
        # the frontend just doesn't render the button.
        "tts": {"available": piper_tts.is_available(), "engine": "piper" if piper_tts.is_available() else None},
        "counts": {
            "analysts": len(live["profiles"]),
            "quarters": len(live["quarters"]),
            "corpus_docs": len(live["corpus"]),
        },
        "latest_reported_quarter": latest.get("quarter_id") if latest else None,
        "latest_call_date": latest.get("call_date") if latest else None,
        "archive_last_refreshed": archive_refreshed,
        "topic_recall_pct": _topic_recall_pct(bank_id),
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
    bank: str = DEFAULT_BANK


class ActivityRequest(BaseModel):
    kind: str
    label: str
    meta: dict | None = None
    bank: str = DEFAULT_BANK


@app.get("/api/brief")
def api_get_brief(bank: str = DEFAULT_BANK):
    return get_state(bank_id=_bank(bank))


@app.post("/api/brief/pin")
def api_pin(req: PinRequest):
    return pin_item(req.kind, req.title, req.body, req.meta, bank_id=_bank(req.bank))


@app.delete("/api/brief/{item_id}")
def api_unpin(item_id: str, bank: str = DEFAULT_BANK):
    return unpin_item(item_id, bank_id=_bank(bank))


@app.post("/api/brief/clear")
def api_clear_brief(bank: str = DEFAULT_BANK):
    return clear_brief(bank_id=_bank(bank))


@app.get("/api/brief/export")
def api_export_brief(quarter: str = "", bank: str = DEFAULT_BANK):
    from fastapi.responses import PlainTextResponse
    bank_id = _bank(bank)
    md = brief_as_markdown(quarter, bank_id=bank_id)
    return PlainTextResponse(md, headers={
        "Content-Disposition": f'attachment; filename="prep-brief-{quarter or bank_id}.md"'
    })


@app.post("/api/activity")
def api_log_activity(req: ActivityRequest):
    return log_activity(req.kind, req.label, req.meta, bank_id=_bank(req.bank))


@app.get("/api/activity")
def api_get_activity(bank: str = DEFAULT_BANK):
    return get_state(bank_id=_bank(bank))["activity"][::-1]   # newest first


@app.get("/api/news/external")
def get_news_external(bank: str = DEFAULT_BANK, refresh: bool = False):
    """External headlines for the News tab (Search & chat), via TheNewsAPI.
    Degrades honestly per src.news.news_api's own discipline -- no token
    configured, a network failure, or the daily request budget being spent
    all come back as a normal 200 with `available`/`error`/`note` fields for
    the frontend to render, never a 5xx. See that module for the per-bank
    cache TTL and the hard daily call cap that keep this well under
    TheNewsAPI's free-tier 100 requests/day."""
    bank_id = _bank(bank)
    return news_api.get_news(bank_id, BANKS[bank_id].display_name, force_refresh=refresh)


@app.get("/api/metrics/timeseries")
def get_metrics_timeseries(bank: str = DEFAULT_BANK):
    """Quarter view's data source (Search & chat): the regex-extracted
    per-quarter metrics (NIM, GNPA, PAT, ROE, CET1, ...) already computed by
    src.signals.metrics_extractor -- this route just reads and returns the
    bank-scoped file, no new computation. A quarter with nothing extracted
    (some early quarters have none) comes back as an empty object for that
    quarter's key, not an error -- the frontend chart treats a missing value
    as a gap, not a zero."""
    bank_id = _bank(bank)
    paths = paths_for(bank_id)
    if not os.path.exists(paths.metrics_timeseries_path):
        return {"bank": bank_id, "metrics_by_quarter": {}, "metric_topics": METRIC_TOPIC_MAP}
    with open(paths.metrics_timeseries_path) as f:
        data = json.load(f)
    return {"bank": bank_id, "metrics_by_quarter": data, "metric_topics": METRIC_TOPIC_MAP}


@app.get("/api/analysts")
def get_analysts(bank: str = DEFAULT_BANK):
    """Roster for the profile list: enough to render rows without the full
    question payload for all 43 analysts."""
    live = _live(_bank(bank))
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
def get_dossier(analyst: str, bank: str = DEFAULT_BANK):
    """Full Q&A threads: the analyst's complete question, each named management
    responder, their own follow-up reaction, and why they asked."""
    bank_id = _bank(bank)
    bank_dossiers = _dossier_cache.setdefault(bank_id, {})
    if analyst in bank_dossiers:
        return bank_dossiers[analyst]
    live = _live(bank_id)
    d = build_analyst_dossier(analyst, live["dataset"], live["graph"])
    profile = next((p for p in live["profiles"] if p["analyst"] == analyst), None)
    if profile:
        d["topic_counts"] = profile["topic_counts"]
        d["hand_curated_persona"] = profile.get("hand_curated_persona")
        d["measured_style"] = profile.get("measured_style")
    bank_dossiers[analyst] = d
    return d


# ── Search ──────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    bank: str = DEFAULT_BANK


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


def _retrieve(query: str, filters: dict, top_k: int = 12, bank_id: str = DEFAULT_BANK) -> list[dict]:
    live = _live(bank_id)
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
    live_bank_id = _bank(req.bank)
    live = _live(live_bank_id)
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
                    "passages": _retrieve(req.query, filters, bank_id=live_bank_id)}

    if filters["intent"] == "comparative" and filters["topic"]:
        by_q = {q: 0 for q in live["quarters"]}
        for d in live["corpus"]:
            if d["type"] == "Question" and filters["topic"] in d["topics"]:
                by_q[d["quarter"]] = by_q.get(d["quarter"], 0) + 1
        return {"filters": filters, "mode": "comparative_table",
                "table": {"title": f'"{filters["topic"]}" questions by quarter',
                          "columns": ["Quarter", "Questions"], "rows": [[q, n] for q, n in by_q.items()]},
                "passages": _retrieve(req.query, filters, bank_id=live_bank_id)}

    return {"filters": filters, "mode": "prose", "table": None,
            "passages": _retrieve(req.query, filters, bank_id=live_bank_id)}


# ── Chat (SSE streaming) ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None
    bank: str = DEFAULT_BANK


def _build_chat_prompt(message: str, passages: list[dict], history: list[dict] | None,
                       bank_name: str = "Axis Bank") -> str:
    # bank_name default only -- chat/search still runs off a single global
    # corpus (_cache), not yet bank-scoped (that is a further phase: the
    # search index itself, not just this prompt, needs to become per-bank).
    # This param exists so the prompt text stops hardcoding "Axis Bank" the
    # moment the corpus is scoped; it is not wired to anything bank-specific
    # yet.
    evidence = "\n\n".join(
        f'[{i + 1}] ({p["quarter"]} · {p["type"]} · {p["who"]}) "{p["text"][:900]}"'
        for i, p in enumerate(passages))
    hist = ""
    if history:
        hist = "\n".join(f'{h["role"]}: {h["content"]}' for h in history[-6:]) + "\n\n"
    return (
        f"You are an analyst-relations assistant for {bank_name}'s IR team, answering "
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
    bank_id = _bank(req.bank)
    live = _live(bank_id)
    names = [p["analyst"] for p in live["profiles"]]
    filters = _extract_filters(req.message, names)
    passages = _retrieve(req.message, filters, top_k=8, bank_id=bank_id)

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
                client.call_llm,
                _build_chat_prompt(req.message, passages, req.history,
                                   bank_name=BANKS[bank_id].display_name), 0.1)
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

        # Kick TTS off the INSTANT the answer text exists -- not after the
        # word-by-word drip below, and not on the client's first click of
        # the Listen button. That drip is already the whole reason the text
        # feels live; synthesis is real CPU work (real seconds for a long
        # answer, unlike the drip's fake 12ms/chunk pacing) and must never
        # block it. So this is a genuinely DETACHED asyncio task: the
        # audio_id is handed to the client in the very next SSE event, the
        # generator moves straight on to the drip below without awaiting
        # the task, and playback fetches /api/tts/audio/<id> later --
        # ready by then for most answers, "still working" if not (see
        # get_tts_audio()). A slow Listen button would be a worse UX
        # regression than no Listen button.
        audio_id = None
        if answer and piper_tts.is_available():
            audio_id = uuid.uuid4().hex
            _tts_cache[audio_id] = None  # placeholder: "pending", not "unknown id"
            asyncio.create_task(_synthesize_tts(audio_id, answer))
            yield sse("audio", {"audio_id": audio_id})

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


async def _synthesize_tts(audio_id: str, text: str) -> None:
    """Runs detached from any request (scheduled via asyncio.create_task in
    chat_stream, never awaited there) -- piper_tts.synthesize_wav is blocking
    CPU work, so it goes through asyncio.to_thread rather than tying up the
    event loop that the rest of the app's requests share."""
    try:
        wav_bytes = await asyncio.to_thread(piper_tts.synthesize_wav, text)
    except Exception:
        wav_bytes = None
    if wav_bytes:
        _tts_cache[audio_id] = wav_bytes
        _tts_cache.move_to_end(audio_id, last=True)
        while len(_tts_cache) > _TTS_CACHE_MAX:
            _tts_cache.popitem(last=False)
    else:
        _tts_cache.pop(audio_id, None)  # synthesis failed -- back to "unknown id", not "stuck pending"


@app.get("/api/tts/audio/{audio_id}")
def get_tts_audio(audio_id: str):
    """Polled by the Listen button after the chat SSE stream's "audio" event
    hands it an id. Three states, all honest: id never issued (or already
    evicted from the bounded cache) -> 404; issued but the background
    synthesis in _synthesize_tts hasn't finished (or failed) -> 202 so the
    client knows to retry rather than treat it as an error; done -> the
    actual audio/wav bytes."""
    if audio_id not in _tts_cache:
        raise HTTPException(status_code=404, detail="unknown or expired audio_id")
    wav_bytes = _tts_cache[audio_id]
    if wav_bytes is None:
        return JSONResponse(status_code=202, content={"status": "pending"})
    return Response(content=wav_bytes, media_type="audio/wav")


# ── Disclosure upload (upcoming quarter) ───────────────────────────────────
@app.post("/api/disclosure/parse")
async def disclosure_parse(file: UploadFile = File(...), holdout: str = Form("true"),
                           bank: str = Form(DEFAULT_BANK)):
    """Parse an upcoming-quarter draft script / investor presentation /
    transcript into prediction inputs. Nothing is written to the archive."""
    live = _live(_bank(bank))
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
    bank: str = DEFAULT_BANK


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
def get_agentic(bank: str = DEFAULT_BANK):
    bank_id = _bank(bank)
    agentic_json, _ = _agentic_out_paths(bank_id)
    if not os.path.exists(agentic_json):
        return {"present": False}
    with open(agentic_json) as f:
        return {"present": True, "data": json.load(f)}


@app.post("/api/run/overall")
def run_overall(req: RunRequest):
    bank_id = _bank(req.bank)
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    disclosure = _disclosure(req)
    state = build_initial_state(quarter, holdout=req.holdout, upcoming=disclosure, bank_id=bank_id)
    bundles, tool_log = run_planning_agent(state["anomaly_scores"], state["graph"],
                                           state["prior_quarters"], state["global_rate"])
    overall = build_overall_topics(bundles, state["anomaly_scores"], state["global_rate"],
                                   state["momentum"], state["client"], disclosure=disclosure,
                                   bank_name=state.get("bank_name", "Axis Bank"))
    key = f"{bank_id}:{quarter}:{req.disclosure_id or 'none'}:{req.holdout}"
    _overall_cache[key] = {"state": state, "overall": overall, "tool_log": tool_log}
    return {"quarter": quarter, "bank": bank_id, "cache_key": key, "tool_log": tool_log,
            "overall": overall, "train_cutoff": state.get("train_cutoff"),
            "disclosure_conditioned": bool(disclosure), "holdout": req.holdout,
            "llm_stats": llm_stats()}


@app.post("/api/run/analyst")
def run_analyst(req: AnalystRunRequest):
    bank_id = _bank(req.bank)
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    key = f"{bank_id}:{quarter}:{req.disclosure_id or 'none'}:{req.holdout}"
    if key not in _overall_cache:
        run_overall(RunRequest(quarter=quarter, disclosure_id=req.disclosure_id,
                               holdout=req.holdout, bank=bank_id))
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
                               disclosure_text=(disclosure or {}).get("narration", ""),
                               bank_id=bank_id)
    results, gate_log = grounding_gate(req.analyst, style, ranked, pool, state["client"],
                                       target_quarter=quarter,
                                       bank_name=state.get("bank_name", "Axis Bank"))
    return {"quarter": quarter, "analyst": req.analyst, "style_note": style,
            "train_cutoff": state.get("train_cutoff"),
            "topics": results, "verifier_log": gate_log,
            "disclosure_conditioned": bool(disclosure), "llm_stats": llm_stats()}


@app.post("/api/run/full")
def run_full(req: RunRequest):
    bank_id = _bank(req.bank)
    quarter = req.quarter or VAL_QUARTER
    llm_reset()
    disclosure = _disclosure(req)
    if not disclosure and not req.holdout:
        run_agentic_full(quarter, bank_id=bank_id)
        log_activity("prepare_call", f"Prepared {quarter}",
                     {"quarter": quarter, "mode": "full"}, bank_id=bank_id)
        agentic_json, _ = _agentic_out_paths(bank_id)
        with open(agentic_json) as f:
            return {"quarter": quarter, "bank": bank_id, "data": json.load(f),
                    "disclosure_conditioned": False, "llm_stats": llm_stats()}

    # Disclosure-conditioned or held-out runs stay in memory rather than
    # overwriting the archive's committed prediction file.
    from src.agentic.graph_app import run_pipeline
    state = build_initial_state(quarter, holdout=req.holdout, upcoming=disclosure, bank_id=bank_id)
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
                 "holdout": req.holdout}, bank_id=bank_id)
    return {"quarter": quarter, "bank": bank_id, "data": data,
            "disclosure_conditioned": bool(disclosure),
            "holdout": req.holdout, "persisted": False, "llm_stats": llm_stats()}


# ── Evaluation ──────────────────────────────────────────────────────────────
def _cached_holdout(with_questions: bool = False, refresh: bool = False,
                    bank_id: str = DEFAULT_BANK) -> dict:
    key = f"{bank_id}:holdout:{with_questions}"
    if refresh or key not in _eval_cache:
        _eval_cache[key] = run_holdout_eval(with_questions=with_questions, bank_id=bank_id)
    return _eval_cache[key]


@app.get("/api/eval/holdout")
def eval_holdout(with_questions: bool = False, refresh: bool = False, bank: str = DEFAULT_BANK):
    """The headline numbers: q4fy26 and q1fy27 scored separately against a
    q3fy26 training cutoff, plus the promotion gate."""
    return _cached_holdout(with_questions=with_questions, refresh=refresh, bank_id=_bank(bank))


def _cached_question_recall(refresh: bool = False, bank_id: str = DEFAULT_BANK) -> dict:
    key = f"{bank_id}:question_recall"
    if refresh or key not in _eval_cache:
        _eval_cache[key] = run_holdout_eval(score_question_recall=True, bank_id=bank_id)
    return _eval_cache[key]


@app.get("/api/eval/question-recall")
def eval_question_recall(refresh: bool = False, analysts: str = "", quarter: str = "",
                         bank: str = DEFAULT_BANK):
    """Question-level recall: did the FRAMED QUESTION TEXT anticipate what
    the analyst actually asked, not just whether the topic bucket matched.
    Topic recall (see /api/eval/holdout) answers "was the bucket on the
    brief" -- this answers "did we anticipate what they actually asked",
    scored via question_eval.evaluate_analyst_questions's 0/0.5/1.0 rubric
    with an LLM judge. Costs real LLM calls (question framing + grounding +
    one judge call per decomposed concern) across every held-out analyst in
    both test quarters -- cached like the other eval endpoints, so this is
    only expensive on first call or an explicit refresh.

    `analysts` (comma-separated names) and `quarter` (a single TEST_QUARTERS
    value, e.g. "q1fy27") scope the run to a cheap, targeted check instead of
    the full held-out set -- added 2026-09 after a full run got throttled to
    a crawl by OpenRouter's free-tier rate limit. A scoped call bypasses the
    cache entirely (it's not the headline result and shouldn't overwrite or
    be served as it), so it always makes fresh LLM calls -- use sparingly."""
    bank_id = _bank(bank)
    if analysts or quarter:
        analyst_list = [a.strip() for a in analysts.split(",") if a.strip()] or None
        quarter_list = [quarter.strip()] if quarter.strip() else None
        return run_holdout_eval(score_question_recall=True,
                                analysts=analyst_list, quarters=quarter_list, bank_id=bank_id)
    return _cached_question_recall(refresh=refresh, bank_id=bank_id)


class EvalCompareRequest(BaseModel):
    disclosure_ids: dict[str, str] | None = None
    bank: str = DEFAULT_BANK


@app.post("/api/eval/compare")
def eval_compare(req: EvalCompareRequest):
    """History-only vs disclosure-conditioned on the held-out quarters, so the
    effect of the upload is a measured number rather than a claim."""
    bank_id = _bank(req.bank)
    out = {}
    for q in TEST_QUARTERS:
        base = evaluate_quarter(q, holdout=True, bank_id=bank_id)
        row = {"history_only": {"macro": base["macro"],
                                "misses": base["failure_attribution"]["counts"],
                                "ranked": base["topic_ranking"]["ranked_topics"],
                                "actual": base["topic_ranking"]["actual_topics"]}}
        did = (req.disclosure_ids or {}).get(q)
        if did and did in _disclosures:
            cond = evaluate_quarter(q, holdout=True, upcoming=_disclosures[did], bank_id=bank_id)
            row["disclosure_conditioned"] = {"macro": cond["macro"],
                                             "misses": cond["failure_attribution"]["counts"],
                                             "ranked": cond["topic_ranking"]["ranked_topics"]}
        out[q] = row
    return {"bank": bank_id, "train_cutoff": TRAIN_CUTOFF, "per_quarter": out}


@app.get("/api/eval/error-report")
def eval_error_report(with_questions: bool = False, refresh: bool = False, bank: str = DEFAULT_BANK):
    """The 'Error Researcher': aggregates the held-out eval's per-miss failure
    attribution into a framework-level diagnosis -- which error category
    dominates, whether it's even fixable by reweighting, and which specific
    (analyst, topic) pairs are driving it. Reuses the same cached holdout run
    as /api/eval/holdout; no extra LLM calls."""
    holdout = _cached_holdout(with_questions=with_questions, refresh=refresh, bank_id=_bank(bank))
    return research_errors(holdout)


class WeightBacktestRequest(BaseModel):
    weights: dict[str, float]
    with_questions: bool = False
    bank: str = DEFAULT_BANK


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
    return backtest_and_promote_weights(req.weights, with_questions=req.with_questions,
                                        bank_id=_bank(req.bank))


@app.get("/api/skills")
def get_skills():
    """Lists the registered agentic-pipeline skills (agents/skills/harness
    separation): what each one does, which pipeline stage it belongs to, and
    whether calling it can make an LLM request. Pure discovery/introspection
    -- does not run anything, and the pipeline's own routes above still call
    these functions directly rather than through this registry."""
    return {"skills": list_skills()}


# ── Ingestion ───────────────────────────────────────────────────────────────
@app.post("/api/ingest/preview")
async def ingest_preview(file: UploadFile = File(...)):
    return preview_ingest(await file.read(), file.filename)


@app.post("/api/ingest/commit")
async def ingest_commit(file: UploadFile = File(...), overwrite: str = Form("false")):
    # Ingestion is still axis-only as of Phase 6 -- src.data.ingest_test's
    # commit_ingest() writes through settings.py's flat DATASET_PATH/etc.
    # shim, not paths_for(bank_id), so it always lands in data/db/axis/
    # regardless of which bank is selected in the UI. Making ingestion itself
    # bank-aware (accepting a target bank, writing to that bank's own
    # dataset/graph) is out of scope for this phase; flagged here rather than
    # silently letting a Kotak/IndusInd "upload" land in Axis's archive.
    was_overwrite = overwrite.lower() in ("1", "true", "yes")
    result = commit_ingest(await file.read(), file.filename, overwrite=was_overwrite)
    if result.get("status") == "ok":
        _load_live(DEFAULT_BANK)
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
