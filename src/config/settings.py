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
DATASET_PATH = os.path.join(_BASE_DIR, "data", "db", "dataset.json")
GRAPH_PATH = os.path.join(_BASE_DIR, "data", "db", "graph.json")
PREDICTIONS_PATH = os.path.join(_BASE_DIR, "data", "outputs", "predictions.json")
PERSONAS_PATH = os.path.join(_BASE_DIR, "data", "inputs", "analyst_personas.json")
NOVELTY_PATH = os.path.join(_BASE_DIR, "data", "inputs", "narration_novelty.json")
PEER_SIGNAL_PATH = os.path.join(_BASE_DIR, "data", "inputs", "peer_signal.json")
EXTERNAL_CONTEXT_PATH = os.path.join(_BASE_DIR, "data", "inputs", "external_context_history.json")
QUESTION_INTENT_PATH = os.path.join(_BASE_DIR, "data", "db", "question_intent.json")
PERSONA_DERIVED_PATH = os.path.join(_BASE_DIR, "data", "inputs", "analyst_personas_transcript_derived.json")
METRICS_TIMESERIES_PATH = os.path.join(_BASE_DIR, "data", "db", "metrics_timeseries.json")
PREP_SHEET_PATH = os.path.join(_BASE_DIR, "data", "outputs", "ir_prep_sheet.json")
CROSSVAL_PATH = os.path.join(_BASE_DIR, "data", "outputs", "cross_validation.json")
SEMANTIC_EVAL_PATH = os.path.join(_BASE_DIR, "data", "outputs", "semantic_eval_full.json")
EARNINGS_TRANSCRIPT_DIR = os.path.join(_BASE_DIR, "earnings_transcript")
PEER_TRANSCRIPT_DIR = os.path.join(_BASE_DIR, "earnings_transcript", "peers")
DASHBOARD_HTML_PATH = os.path.join(_BASE_DIR, "frontend", "ir_dashboard.html")
PLATFORM_HTML_PATH = os.path.join(_BASE_DIR, "frontend", "ir_platform_ui.html")

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
    RECENT_WIN = 6.0    # Recent history window (quarters)
    # LLM narration-novelty (active only when narration_novelty.json exists for VAL_QUARTER).
    # Applied as ONE extra prep slot per analyst — never displaces top-N picks.
    NOVELTY_MIN_SCORE = 0.5       # ignore weak novelty signals
    NOVELTY_AFFINITY_FLOOR = 0.04 # analyst's blended pref must clear this to get the slot
    # Peer-bank signal (active only when peer_signal.json exists for VAL_QUARTER).
    # Same extra-slot mechanism as novelty — never displaces top-N picks.
    PEER_MIN_SALIENCE = 0.5       # ignore topics peer analysts barely touched
    PEER_AFFINITY_FLOOR = 0.04    # analyst's blended pref must clear this to get the slot
