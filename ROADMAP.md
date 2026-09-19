# Roadmap

A living scope of what's built, what's next, and what's still an open question.
This isn't a promise of dates -- it's here so contributors (and future us) can see
where a change fits before starting on it. Propose additions or reprioritization
via a GitHub issue; see [CONTRIBUTING.md](CONTRIBUTING.md) for how.

---

## Shipped

**Core prediction & agentic pipeline**
- LangGraph multi-agent execution graph (Planning Agent, Overall Topic Layer,
  Analyst-Specific Reweighter, Verifier Grounding Gate)
- Per-analyst question forecasting from historical Q&A trajectories, narration
  novelty, and persona metrics
- Held-out semantic evaluation (LLM-as-a-judge) and leave-one-quarter-out
  cross-validation backtesting
- Error taxonomy + Error Researcher and Framework Loop weight backtesting for
  tuning the prediction weights against real historical misses

**Multi-bank archive**
- Bank registry (`src/config/banks.py`) with per-bank data paths, covering
  Axis Bank, Kotak Mahindra Bank, and IndusInd Bank
- Cross-bank peer signal (`src/signals/peer_signal.py`) as an auxiliary input
  to prep-sheet generation

**Search, chat & workspace**
- Hybrid search (BM25 + TF-IDF + optional spaCy semantic signal) over the
  full transcript archive, with citation attribution
- Chat interface grounded in the same retrieval layer
- Analyst profiles, evaluation dashboard, transcript ingestion, and a
  server-persisted prep brief with pinning and an activity log

**Quarter view & News feeds** *(most recent addition)*
- Multi-quarter metric comparison charts (NIM, GNPA, NNPA, PCR, PAT, ROA, ROE,
  CET1, Cost-to-Income, Cost-to-Assets, Net Credit Cost), grouped by topic,
  restricted to same-unit comparisons
- Bank-specific external headlines (TheNewsAPI) with a Last-7-days /
  Last-30-days / All-time filter
- Analyst-mention coverage: a best-effort filter over the same headline set
  for mentions of the bank's own covering analysts (documented limitation:
  this matches general news mentions, not paywalled sell-side research notes
  -- see "Near-term" below)

**Usage & Audit**
- Durable, append-only audit trail for every LLM call across every provider in
  the fallback chain (`src/audit/llm_audit.py`, `data/logs/llm_calls_*.jsonl`)
  -- provider, model, purpose, token usage, latency, estimated cost, and
  success/failure, captured at the call site rather than reconstructed after
  the fact
- Usage & Audit dashboard: daily call volume, token/cost totals, a
  per-provider breakdown, and a filterable recent-calls log, backed by
  `/api/audit/summary` and `/api/audit/log`

**Other**
- Offline, CPU-only text-to-speech for chat answers (Piper)
- `.env`-driven configuration, loaded reliably at startup

---

## Near-term

Natural next steps that build directly on what's shipped, roughly in the order
they'd unblock the most value:

- **More banks in the registry.** The pipeline is bank-agnostic already
  (Phase 0-2 of the original multi-bank plan); adding a bank is mostly
  sourcing and confirming transcript PDFs, then running the existing compile
  pipeline. HDFC Bank and ICICI Bank are the obvious next two peers.
- **Held-out eval for Kotak/IndusInd.** Both currently have too few quarters
  for a meaningful cross-validation backtest -- this closes once enough
  historical transcripts are ingested for them.
- **News feeds: real research-note awareness.** The current "Analyst
  coverage" panel is an honest proxy (general news mentioning an analyst's
  name), not actual research-note ingestion. A real version would need either
  a paywalled research-note data source or analysts self-reporting their own
  published notes -- worth scoping as its own feature, not a quiet upgrade to
  the current filter.
- **Quarter view: cross-bank overlay.** Compare a metric (say NIM) across
  Axis vs. Kotak vs. IndusInd on one chart, reusing the peer signal data
  that already exists for prep-sheet generation.
- **Export a prep brief.** One-click PDF/slide export of the pinned prep
  brief for sharing outside the app (a document people take into the actual
  call prep meeting).

## Mid-term

Bigger pieces that need their own design pass before starting:

- **Multi-user / permissions.** Right now the app has no concept of separate
  users -- prep briefs, pins, and activity are workspace-wide, not
  per-person. Needed before this is genuinely multi-analyst-team usable.
- **Notifications.** Alert a user when a new disclosure lands for a bank
  they're tracking, or when a metric moves outside its historical range.
- **Chat/Slack integration.** Push prep-brief summaries or notable metric
  moves into a channel instead of requiring someone to open the dashboard.
- **On-prem / air-gapped model serving.** The provider fallback chain
  (Groq/Gemini/OpenAI/OpenRouter/...) assumes internet access to hosted APIs;
  a fully local model path would matter for any regulated-environment
  deployment of this pattern.

## Exploratory

Ideas worth keeping on the radar, not yet scoped:

- Live transcript ingestion during an active earnings call (rather than
  after-the-fact PDF parsing)
- A mobile-optimized view of the workspace (current UI is responsive but
  desktop-first)
- Fine-tuning a smaller model on this platform's own historical
  prediction/verification data, instead of relying on general-purpose hosted
  LLMs for every layer

---

## Won't do (for now)

Explicitly out of scope, to save contributors from re-proposing:

- Real-time stock trading signals or investment advice -- this is a call-prep
  and research tool, not a trading platform.
- Any feature that requires storing analysts' or executives' personal data
  beyond what's already public in earnings-call transcripts.
