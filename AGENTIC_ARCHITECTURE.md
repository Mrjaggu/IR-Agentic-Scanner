# Agentic Prediction Core — build notes

Implements Section 3 of `ir-intelligence-platform-architecture.pdf` (the
"agentic core" scope, chosen over rebuilding ingestion/entity-registry from
scratch since that foundation data already exists in `data/db/graph.json`,
`data/db/dataset.json`, `data/inputs/analyst_personas*.json`, and
`data/db/metrics_timeseries.json`).

Additive only — does not modify `engine.py` or `predictions.json`. Run with:

```
python3 run.py agentic [--quarter q1fy27]
```

Outputs: `data/outputs/agentic_predictions.json` (full structured trace) and
`data/outputs/agentic_brief.md` (human-readable topic brief + question sheet).

## What's implemented, mapped to the doc

| Doc section | File | Notes |
|---|---|---|
| 3.1 Planning Agent | `src/agentic/planning_agent.py` + `tools.py` | Per-anomaly tool selection among history/commitments/policy tools, capped at 10 calls/quarter (logged, not just asserted). `policy_tool` is an honest stub — no RBI/policy-circular ingestion in this build. `commitments_tool` is a heuristic narration scan, not a real structured commitments store. |
| 3.2 Overall (Topic) Layer | `src/agentic/overall_layer.py` | Researcher (deterministic: anomaly scores, base rates, cross-analyst momentum, commitments) -> Analyst (1 LLM call, must cite evidence) -> Verifier (deterministic grounding check, reject-and-regenerate capped at 2 retries, then bare topic). |
| 3.3 Analyst-Specific Layer | `src/agentic/analyst_layer.py` | Reweights the Overall layer's ranked topics per analyst by measured preference (a globally-hot topic an analyst never asks about sinks for them specifically — the doc's own example). Also implements the in-call arithmetic-follow-up drilldown: gated on (a) a real bridge-tension in this quarter's disclosed metrics (NIM down + PAT up, same pattern class as the doc's NIM/NII example) and (b) the analyst's hand-curated style profile containing granular/mechanics-style language (Kunal Shah, Abhishek Murarka, Rikin Shah, Param Subramanian, Piran Engineer qualify for Q1FY27), bounded to 3 LLM calls. |
| 3.4 Question Framer | `src/agentic/question_framer.py` | One LLM call per analyst, batching their topics, grounded in their own precedent question + base rate + anomaly score. |
| 3.5 Verifier / Grounding Gate | `src/agentic/verifier.py` | Implemented as an actual **LangGraph cyclic state graph** (frame -> check -> conditional edge back to frame or END) — the concrete instance of Section 12's stated reason for choosing LangGraph, not just a while-loop with the same shape. Deterministic regex+set-membership check, no second LLM call needed to grade the first. |
| Section 4 coordination | `src/agentic/graph_app.py` | Top-level LangGraph: planning -> overall -> analyst_specific -> arithmetic_followups. One-directional (overall never revised by analyst-specific findings) per the doc's own pilot recommendation. |
| Section 12 | `langgraph` (pip-installed, v1.2.11) | Used for both the top-level pipeline and the nested Verifier subgraph. LangSmith was NOT wired in this session (memory already notes this as deferred) — tracing/eval-harness integration is future work. |

## Not built this session (by explicit scope choice)

- Section 2 foundation layers were reused as-is, not rebuilt with a real
  entity registry / fuzzy-matching monitor / structured `open_threads` and
  `commitments` store. `commitments_tool` and `policy_tool` are honest
  approximations/stubs, not the real thing.
- Section 5 (continual learning loop + human review gate) and Section 6
  (eval harness / walk-forward validation for the agentic pipeline
  specifically) — the existing `evaluation.py` cross-validation harness
  covers the deterministic engine, not this new pipeline.
- Section 9 (three-tab UI) — no dashboard wiring for agentic output yet;
  `frontend/ir_dashboard.html` still reflects the deterministic engine's
  `predictions.json` only.

## A real limitation hit during this build, not a code bug

This session's network (both the cloud sandbox and the bridge to your Mac)
could not reach `api.groq.com` or `generativelanguage.googleapis.com` —
blocked at the proxy level (403 on CONNECT), independent of the API keys in
`.env` being valid. Every LLM-dependent step in the pipeline (Overall-layer
ranking rationale, per-analyst question phrasing, arithmetic follow-up text)
therefore ran through its documented graceful-degradation path: the
Verifier's reject-and-regenerate loop correctly ran all 3 attempts (initial +
2 retries) and then correctly fell back to bare topics rather than emit
anything fabricated — this is the pipeline behaving exactly as designed when
the LLM is unreachable, not silent failure.

**To see it with real phrased output**, run `python3 run.py agentic` directly
in a terminal on your Mac (outside this bridged session) where your Groq/
Gemini keys have reached those hosts successfully before (per `RESULTS_V2.md`'s existing
LLM-backed runs). The planning-agent tool log, overall topic ranking, and
per-analyst topic reweighting are all already fully working now (visible in
`data/outputs/agentic_predictions.json`) since those steps are deterministic.

## Live FastAPI dashboard (added after the static UI)

The static UI above (`frontend/ir_platform_ui.html`, compiled via `python3 run.py ui`)
still works standalone with no server. But the user wanted one live dashboard to
trigger ingestion, run Overall vs Analyst-Specific prediction independently, and
search/chat the data interactively — that needed a real backend, so:

```
python3 run.py app [--port 8000]
```

starts a FastAPI app (`src/api/fastapi_app.py`) serving `frontend/ir_platform_app.html`
at `http://localhost:8000/`. It replaces the old `python3 run.py server` for
interactive use (that command and `src/api/server.py` are untouched and still work
for the older `ir_dashboard.html` / ad-hoc `/api/predict` flow).

Endpoints:
- `GET /api/data` — analyst profiles + topics + quarters (Tab 1)
- `POST /api/search` — query understanding + hybrid retrieval (BM25 + TF-IDF cosine
  + RRF, server-side port of the static UI's client-side JS, in `src/search/hybrid_search.py`)
- `POST /api/chat` — same retrieval, then an LLM-synthesized grounded answer if
  Groq/Gemini is reachable, else the raw cited passages (never fabricates an answer
  when the LLM is unreachable)
- `GET /api/agentic`, `POST /api/run/overall`, `POST /api/run/analyst`,
  `POST /api/run/full` — the three ways to trigger prediction from Tab 3:
  Overall layer alone, one analyst's Analyst-Specific layer alone (reweights the
  last cached Overall run, computing it first if needed), or the whole pipeline
  (equivalent to `python3 run.py agentic`, writes the same files)
- `POST /api/ingest/preview`, `POST /api/ingest/commit` — same as before, now also
  invalidating the live data cache on a successful commit so Tab 1/2/3 immediately
  reflect the new quarter without restarting the server

All five endpoint groups were exercised end-to-end against real data in this
session (a real Q1FY27 overall run, a real per-analyst run, a real full-pipeline
run, a real ingest preview, and real search/chat queries including relational
"who else covers X" and comparative "compare X across quarters" — both correctly
routed to table output). Chat correctly falls back to extractive/cited passages
rather than fabricating an answer, since Groq/Gemini remain unreachable from this
environment (see the note above) — same honest degradation as everywhere else in
this pipeline.

---

# Rebuild pass — recall-first objective, dossiers, disclosure input, new UI

## 1. Held-out test discipline is now enforced in code

`TEST_QUARTERS = [q4fy26, q1fy27]`, `TRAIN_CUTOFF = q3fy26` (`src/config/settings.py`).
`build_initial_state(..., holdout=True)` caps training memory at the cutoff regardless of
which quarter is being predicted, so predicting q1fy27 cannot see q4fy26. Recency
distances still use the true calendar order, and anomaly percentiles compare only
against the training pool. The two quarters are always scored **separately** —
one quarter past the cutoff vs two — never averaged into one headline.

## 2. The objective is recall, not F1

An IR brief listing a topic that never comes up costs a few minutes of prep. A topic
that comes up unprepared costs a bad answer on a recorded call. So:

- headline metric is **coverage / recall**; the reported F is **F2** (recall weighted 2x)
- the promotion gate floors recall (0.75) and only *guards* precision (0.22) against a
  degenerate "predict everything" config, with stability measured on recall spread
- slot policy: each analyst gets `measured topic count + 4`, capped at 8 of 12 topics

The slot policy was chosen by `sweep_slot_policy()` over **8 training quarters**
(q4fy24–q3fy26), never on the held-out set. Measured there: extra=4/cap=8 gave the
highest F2 (0.619) and sits at the knee of the recall curve (recall 0.825, precision
0.310). Wider settings were measured and rejected — cap=12 buys only +3.5pp recall
while predicting every topic for every analyst, which is not a brief.

### Held-out result at that policy

| | Q4FY26 (1Q past) | Q1FY27 (2Q past) |
|---|---|---|
| Recall | 90.3% | 95.0% |
| Precision | 37.9% | 34.1% |
| F2 | 0.702 | 0.659 |
| Coverage | 16/18 raised, from 42 predicted | 20/22 raised, from 60 predicted |

**36 of the 40 topics actually raised across both calls were on the brief (90%), from
102 predictions.** Recall spread 4.7%. Promotion gate: PASS. Total misses fell from 19
to 4 (2 reasoning-weakness, 2 retrieval-gap).

## 3. Upcoming-quarter disclosure as a prediction input

`src/data/upcoming.py` parses a draft disclosure script / investor presentation /
press release / transcript (PDF or TXT) into: structured metrics, anomaly percentiles
against the training pool, topic salience, and **drill-down flags** — the specific
sentences whose shape historically draws scrutiny (multi-part bridges, a stated change
with no stated driver). All deterministic, no LLM, so it works air-gapped.

Wired into the ranking through a transparent composite score (metric anomaly 0.35,
disclosure salience 0.25, cross-analyst momentum 0.15, base rate 0.15, drill-down flag
0.10 — fixed a priori, **not** tuned on the held-out quarters), and into the analyst
layer as an **additive** blend. Additive is deliberate: RESULTS_V2.md documents three
signals that died because the engine's multiplicative `p * (1 + w*signal)` form keeps
amplifying whatever already had high base preference; its own stated fix was an
additive combination.

### Honest read on what the disclosure input does and does not do

- It is what makes the system usable for a **genuinely upcoming** call — there is no
  archive row to predict from, so the history-only path simply does not apply.
- The drill-down brief is a deliverable in its own right: for Q1FY27 it flags the
  ₹15,608cr non-NPA provisions breakdown (13 numeric components) as a multi-part
  bridge, before anyone asks about it.
- On the held-out quarters it was **metric-neutral** at topic level. Reported as
  measured; not dressed up as a win.
- Known limitation: an off-list flagged topic is only pulled into an analyst's slots
  if that analyst has prior history on it. So Q1FY27's Subsidiaries' Performance
  reached Piran Engineer (whose published style covers subsidiaries) but **not** Rikin
  Shah, who actually asked it — it stays a `retrieval_gap`. The fix is to allow
  pull-in on style match rather than prior-topic history only; deliberately left
  unimplemented rather than tuned against the test set in the same session.

## 4. Analyst profiles: full dossiers

`src/data/dossier.py` builds threads from `dataset.json`'s `qa` turns rather than the
graph's merged Question blocks, which recovers the actual conversation:

**full question (untruncated) -> each NAMED management responder and what they said ->
the analyst's own follow-up reaction -> further responses.**

Each exchange also carries an evidence-based `why`, assembled from data already in the
archive rather than guessed: the intent classification and its evidence sentence
(`question_intent.json`), their own prior question this follows up on (`FOLLOWS_UP_ON` /
`CROSS_FOLLOWS_PREV_CALL` edges), what management had just disclosed on that topic, the
metric move behind it with its anomaly percentile, and how much of a recurring theme it
is for them vs every other analyst.

## 5. UI rebuilt

`frontend/ir_platform_app.html`, served by the FastAPI app. Sidebar navigation, a real
design system (validated categorical/status palette, light + dark both selected rather
than auto-flipped, bar specs capped at 24px with 4px rounded data-ends and hairline
gridlines, hover tooltips, table views alongside every chart), prose measures held to
~74ch for transcript reading, and no CDN anywhere so it runs air-gapped.

Chat is a proper surface: SSE streaming, inline markdown (headings, lists, tables,
code, blockquotes), clickable `[n]` citation chips that expand and scroll to the source
passage, suggested prompts, stop/copy, and an explicit mode badge that says when no
model is reachable and the answer is retrieval-only rather than pretending otherwise.

Verified by rendering with Playwright and reading the screenshots, which caught two
real bugs that static review missed: a `??`/`||` precedence SyntaxError that killed the
whole script, and an infinite loop in the markdown renderer on a **partially streamed
table** (header row present, separator row not yet arrived) that would have hung the
chat mid-answer for a real user.

## Still open

- Section 5 continual-learning loop and the human review gate
- Real entity registry / structured `commitments` store (still heuristic)
- Cross-bank coverage graph (Section 8, pending legal sign-off)
- Grounding rate is measurable (`evaluate_quarter(with_questions=True)`) but unmeasured
  here: every LLM-dependent path degrades to bare topics in this environment, because
  Groq/Gemini are blocked at the proxy. Run from a terminal with network to populate it.
