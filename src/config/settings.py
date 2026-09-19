import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment keys from .env if present
_env_path = os.path.join(_BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip("'\"")

# Directory settings
BASE_DIR = _BASE_DIR
DASHBOARD_HTML_PATH = os.path.join(_BASE_DIR, "frontend", "ir_dashboard.html")
PLATFORM_HTML_PATH = os.path.join(_BASE_DIR, "frontend", "ir_platform_ui.html")

# ── Multi-bank path resolution ───────────────────────────────────────────────
# 2026-09: this platform is becoming multi-bank (see src/config/banks.py).
# Every per-bank data file now lives under data/db/<bank_id>/,
# data/inputs/<bank_id>/, data/outputs/<bank_id>/, earnings_transcript/<bank_id>/
# instead of the old flat data/db/, data/inputs/, earnings_transcript/ layout.
#
# paths_for(bank_id) is the one place that knows this layout. The flat
# constants below (DATASET_PATH, GRAPH_PATH, ...) are kept as a COMPATIBILITY
# SHIM -- they resolve to paths_for(DEFAULT_BANK)'s paths, so the ~20+ existing
# call sites that still `from src.config.settings import DATASET_PATH` keep
# working unchanged and unaware anything moved. Those call sites are migrated
# to take a bank_id and call paths_for(bank_id) directly one module at a time
# (dataset_compiler.py and graphs/compiler.py first); this shim is deleted
# once no call site imports the flat names anymore.
from dataclasses import dataclass
from src.config.banks import DEFAULT_BANK


@dataclass(frozen=True)
class BankPaths:
    bank_id: str
    dataset_path: str
    graph_path: str
    predictions_path: str
    personas_path: str
    analyst_enrichment_path: str
    novelty_path: str
    peer_signal_path: str
    external_context_path: str
    question_intent_path: str
    persona_derived_path: str
    ask_patterns_path: str
    metrics_timeseries_path: str
    prep_sheet_path: str
    crossval_path: str
    semantic_eval_path: str
    earnings_transcript_dir: str


def paths_for(bank_id: str) -> BankPaths:
    db = os.path.join(_BASE_DIR, "data", "db", bank_id)
    inputs = os.path.join(_BASE_DIR, "data", "inputs", bank_id)
    outputs = os.path.join(_BASE_DIR, "data", "outputs", bank_id)
    return BankPaths(
        bank_id=bank_id,
        dataset_path=os.path.join(db, "dataset.json"),
        graph_path=os.path.join(db, "graph.json"),
        predictions_path=os.path.join(outputs, "predictions.json"),
        personas_path=os.path.join(inputs, "analyst_personas.json"),
        analyst_enrichment_path=os.path.join(inputs, "analyst_profile_enrichment.json"),
        novelty_path=os.path.join(inputs, "narration_novelty.json"),
        peer_signal_path=os.path.join(inputs, "peer_signal.json"),
        external_context_path=os.path.join(inputs, "external_context_history.json"),
        question_intent_path=os.path.join(db, "question_intent.json"),
        persona_derived_path=os.path.join(inputs, "analyst_personas_transcript_derived.json"),
        ask_patterns_path=os.path.join(inputs, "analyst_ask_patterns.json"),
        metrics_timeseries_path=os.path.join(db, "metrics_timeseries.json"),
        prep_sheet_path=os.path.join(outputs, "ir_prep_sheet.json"),
        crossval_path=os.path.join(outputs, "cross_validation.json"),
        semantic_eval_path=os.path.join(outputs, "semantic_eval_full.json"),
        earnings_transcript_dir=os.path.join(_BASE_DIR, "earnings_transcript", bank_id),
    )


_DEFAULT_PATHS = paths_for(DEFAULT_BANK)

# ── Compatibility shim (see note above) -- remove once every call site is
# migrated to paths_for(bank_id) directly. ───────────────────────────────────
DATASET_PATH = _DEFAULT_PATHS.dataset_path
GRAPH_PATH = _DEFAULT_PATHS.graph_path
PREDICTIONS_PATH = _DEFAULT_PATHS.predictions_path
PERSONAS_PATH = _DEFAULT_PATHS.personas_path
ANALYST_ENRICHMENT_PATH = _DEFAULT_PATHS.analyst_enrichment_path
NOVELTY_PATH = _DEFAULT_PATHS.novelty_path
PEER_SIGNAL_PATH = _DEFAULT_PATHS.peer_signal_path
EXTERNAL_CONTEXT_PATH = _DEFAULT_PATHS.external_context_path
QUESTION_INTENT_PATH = _DEFAULT_PATHS.question_intent_path
PERSONA_DERIVED_PATH = _DEFAULT_PATHS.persona_derived_path
ASK_PATTERNS_PATH = _DEFAULT_PATHS.ask_patterns_path
METRICS_TIMESERIES_PATH = _DEFAULT_PATHS.metrics_timeseries_path
PREP_SHEET_PATH = _DEFAULT_PATHS.prep_sheet_path
CROSSVAL_PATH = _DEFAULT_PATHS.crossval_path
SEMANTIC_EVAL_PATH = _DEFAULT_PATHS.semantic_eval_path
EARNINGS_TRANSCRIPT_DIR = _DEFAULT_PATHS.earnings_transcript_dir
# PEER_TRANSCRIPT_DIR is NOT bank-scoped -- src/signals/peer_signal.py's whole
# purpose is reading OTHER banks' transcripts as a side signal for the active
# bank's prep sheet, so it deliberately stays pointed at the shared peers/
# folder rather than any one bank's own transcript dir. Untouched by the
# migration in scripts/migrate_to_bank_layout.py (that script COPIES, not
# moves, kotak/indusind PDFs into their new per-bank dirs, leaving the
# originals here so peer_signal.py keeps working unmodified).
PEER_TRANSCRIPT_DIR = os.path.join(_BASE_DIR, "earnings_transcript", "peers")

# ── Writable runtime-state directory (Vercel and similar read-only deploys) ──
# BASE_DIR is the repo root, which is fine for reading committed data but not
# for writing it: on Vercel the whole deployed tree (/var/task) is read-only
# except /tmp, so anything that persists RUNTIME state under BASE_DIR (the
# per-bank workspace_state.json in src/data/workspace_state.py, the LLM audit
# trail in src/audit/llm_audit.py) needs somewhere that's actually writable
# in production, not just in local/on-prem dev where BASE_DIR itself is fine.
#
# writable_data_dir(*parts) tries BASE_DIR/data/<parts> first (so a real,
# persistent-across-restarts location is still preferred wherever the
# filesystem allows it -- local dev, a real VM, an on-prem box) and falls
# back to a tempdir-rooted mirror of the same relative path only if that
# fails. The probe result is cached per relative path so this is a real
# filesystem write attempt exactly once, not once per call.
#
# Important honesty note: the /tmp fallback is NOT durable on Vercel the way
# the rest of this codebase's "plain JSON files on disk" persistence assumes
# elsewhere -- Vercel's /tmp is scoped to one function instance and can be
# wiped on the next cold start, so state written there survives repeated
# requests to the SAME warm instance but not a redeploy or a scale-to-zero.
# That's a real limitation of running local-disk persistence on serverless,
# not something this helper can paper over -- it only prevents a crash and
# keeps things working within a warm instance, which is strictly better than
# the unhandled OSError this replaces.
import tempfile

_writable_dir_cache: dict[str, str] = {}


def writable_data_dir(*parts: str) -> str:
    rel = os.path.join(*parts) if parts else ""
    if rel in _writable_dir_cache:
        return _writable_dir_cache[rel]

    primary = os.path.join(_BASE_DIR, "data", rel) if rel else os.path.join(_BASE_DIR, "data")
    try:
        os.makedirs(primary, exist_ok=True)
        probe = os.path.join(primary, ".write_probe")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        _writable_dir_cache[rel] = primary
        return primary
    except OSError:
        pass

    fallback = os.path.join(tempfile.gettempdir(), "ir_platform_data", rel) if rel else \
        os.path.join(tempfile.gettempdir(), "ir_platform_data")
    os.makedirs(fallback, exist_ok=True)
    _writable_dir_cache[rel] = fallback
    return fallback


# Validation parameters
VAL_QUARTER = "q1fy27"

# ── Held-out test discipline (architecture doc Section 6.1) ──────────────────
# Q4FY26 and Q1FY27 are the held-out test set and are NEVER used to build
# training memory in holdout mode. Scored separately so we can see the
# one-quarter-ahead (q4fy26) result next to the two-quarters-ahead (q1fy27)
# result, which is exactly the degradation the original POC missed.
TEST_QUARTERS = ["q4fy26", "q1fy27"]

# Cutoff mode:
#   "rolling" (default) — train on everything strictly before the target, so
#      predicting q1fy27 trains through q4fy26 and predicting q4fy26 trains
#      through q3fy26. This is the production setting: before any real call you
#      genuinely do have every prior call.
#   "fixed" — freeze training at TRAIN_CUTOFF for every target, which makes
#      q4fy26 a one-quarter-ahead test and q1fy27 a two-quarters-ahead test.
#      Kept because that distance comparison is the doc's Section 6.1
#      degradation check, and it is the thing the original POC hid.
CUTOFF_MODE = "rolling"
TRAIN_CUTOFF = "q3fy26"      # only used when CUTOFF_MODE == "fixed"

# ── Objective: RECALL-FIRST ──────────────────────────────────────────────────
# This is an IR prep tool. A topic on the brief that never comes up costs a few
# minutes of management's prep time; a topic that comes up unprepared costs a
# bad answer on a recorded call. So the objective is coverage of what actually
# gets asked, and over-prediction is cheap by design -- predicting 50 topics to
# catch 20 of 25 actuals is a good trade here, not a failure.
#
# Headline metric is therefore F-beta with beta=2 (recall weighted 2x
# precision), never balanced F1, and the gate floors recall high while letting
# precision run low.
F_BETA = 2.0

class PromotionGate:
    MIN_MEAN_RECALL = 0.75      # the metric that actually matters
    MIN_MEAN_PRECISION = 0.22   # a floor against "just predict everything"
    MAX_RECALL_SPREAD = 0.18    # cross-quarter stability, on recall

# ── Slot policy ──────────────────────────────────────────────────────────────
# Slots per analyst = their measured topic count + SLOT_EXTRA, capped at
# SLOT_CAP. Chosen by a walk-forward sweep over TRAINING quarters only
# (src/agentic/eval_harness.py::sweep_slot_policy) -- never by picking whichever
# value flattered the held-out quarters.
# Chosen by sweep_slot_policy() over 8 training quarters (q4fy24..q3fy26):
# highest F2 (0.619), the knee of the recall curve (recall 0.825 at precision
# 0.310), and still a ranked brief rather than the whole taxonomy. Wider
# settings were measured and rejected: cap=12 buys only +3.5pp recall while
# predicting every topic for every analyst, which is not a brief.
SLOT_EXTRA = 4
SLOT_CAP = 8

# Canonical analyst names (fixes transcript-parsing splits)
ANALYST_ALIASES = {
    "Ma hrukh Adajania": "Mahrukh Adajania",
    "Sam eer Bhise": "Sameer Bhise",
    "Sum eet Kariwala": "Sumeet Kariwala",
    "Pra khar Sharma": "Prakhar Sharma",
    "Harsh Modi": "Harsh Wardhan Modi",
    "Krishnan": "Krishnan ASV",
    "Nilanjan": "Nilanjan Karfa",
}

# The 12-topic taxonomy
TOPICS_LIST = [
    "NIM & Yields",
    "Deposits & CASA",
    "Loan & Advances Growth",
    "Slippages & Asset Quality",
    "Credit Cost & Provisions",
    "Opex & Cost-to-Income",
    "Capital Adequacy",
    "Credit Cards & Spends",
    "Citibank Integration",
    "Subsidiaries' Performance",
    "Profitability & Returns",
    "Strategy & Competitive Positioning",
]

# Baseline Production Hyperparameters
class EngineConfig:
    DECAY = 0.40        # Recency decay rate (per quarter)
    W_NARR = 3.5        # Narration salience boost
    W_Q3_CROSS = 1.5    # Cross-analyst momentum weight
    W_Q3_SELF = 1.2     # Base self-bonus weight
    BASE_PROB = 0.04     # Probability floor for rare/new topics
    PRIOR_MIX = 0.35    # Global base-rate blend weight
    PERSONA_MIX = 0.45  # Web persona prior blend weight
    # 2026-09: opt-in only (see src/signals/cross_bank_persona.py) -- an
    # analyst's topic history on OTHER registered banks, blended in the same
    # additive-prior style as PERSONA_MIX above. A standalone top-K backtest
    # showed +21-23pp mean topic recall on Kotak/IndusInd's held-out
    # quarters, but that did NOT reproduce when run through the real scored
    # pipeline (eval_harness.run_holdout_eval, verified 2026-09): +0.0pp
    # Kotak, -1.6pp IndusInd -- see cross_bank_persona.py's module docstring
    # for why the two measurements diverge. Left off by default both for
    # that reason and because it's pending confirmation that using one
    # bank's analyst-behavior data to inform another bank's predictions has
    # the sign-off this project's build notes flag it as needing (Section 8,
    # "cross-bank coverage graph, pending legal sign-off").
    CROSS_BANK_MIX = 0.25
    RECENT_WIN = 6.0    # Recent history window (quarters)
    # LLM narration-novelty (active only when narration_novelty.json exists for VAL_QUARTER).
    # Applied as ONE extra prep slot per analyst — never displaces top-N picks.
    NOVELTY_MIN_SCORE = 0.5       # ignore weak novelty signals
    NOVELTY_AFFINITY_FLOOR = 0.04 # analyst's blended pref must clear this to get the slot
    # Peer-bank signal (active only when peer_signal.json exists for VAL_QUARTER).
    # Same extra-slot mechanism as novelty — never displaces top-N picks.
    PEER_MIN_SALIENCE = 0.5       # ignore topics peer analysts barely touched
    PEER_AFFINITY_FLOOR = 0.04    # analyst's blended pref must clear this to get the slot
