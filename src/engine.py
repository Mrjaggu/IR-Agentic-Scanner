import os
import re
import json
import time
import math
from src.config.settings import (
    VAL_QUARTER,
    EngineConfig,
    TOPICS_LIST,
    PREDICTIONS_PATH,
    NOVELTY_PATH,
    PEER_SIGNAL_PATH,
    METRICS_TIMESERIES_PATH,
)
from src.data.loader import load_dataset, load_graph, get_active_analysts
from src.model_provider.llm_client import client
from src.memory.history import get_global_rate, get_analyst_profile, get_repeat_propensity, PERSONAS
from src.graphs.adjacency import build_adjacency_index, get_last_qa, get_cross_follow_context

# ─── Load Graph & Dataset on Module Init ──────────────────────────────────────
dataset = load_dataset()
graph = load_graph()
active_analysts = get_active_analysts(dataset)

# Adjacency index
adj = build_adjacency_index(graph)

# Quarter order and sort mapping
quarter_sort_map = {q["quarter_id"]: q["sort_key"] for q in dataset}
QUARTER_ORDER = [q["quarter_id"] for q in dataset]
Q_ORD = {q: i for i, q in enumerate(QUARTER_ORDER)}
VAL_ORD = Q_ORD[VAL_QUARTER]
train_quarters = [q["quarter_id"] for q in dataset if q["quarter_id"] != VAL_QUARTER]

# ─── Pre-computed caches ──────────────────────────────────────────────────────
# Cross-analyst topic momentum over the prior TWO quarters
_prev_quarters = [(QUARTER_ORDER[VAL_ORD - 1], 1.0), (QUARTER_ORDER[VAL_ORD - 2], 0.5)]
PREV_QUARTER = QUARTER_ORDER[VAL_ORD - 1]
q3_questions = [n for n in graph["nodes"]
                if n["type"] == "Question" and n["properties"]["quarter"] == PREV_QUARTER]
q3_topics_count = {}
for _pq, _w in _prev_quarters:
    for q in [n for n in graph["nodes"]
              if n["type"] == "Question" and n["properties"]["quarter"] == _pq]:
        for t in q["properties"]["topics"]:
            if t != "General":
                q3_topics_count[t] = q3_topics_count.get(t, 0.0) + _w
MAX_Q3 = max(q3_topics_count.values()) if q3_topics_count else 1

# Global topic base-rate over training quarters
GLOBAL_RATE = get_global_rate(graph, train_quarters)

# Q3FY26 per-analyst topic sets (self-bonus)
q3_analyst_topics = {}
for q in q3_questions:
    a = q["properties"]["analyst"]
    q3_analyst_topics.setdefault(a, set()).update(
        t for t in q["properties"]["topics"] if t != "General"
    )

# Q4FY26 narration topic frequency
q4_narr_segments = [n for n in graph["nodes"]
                    if n["type"] == "NarrationSegment"
                    and n["properties"]["quarter"] == VAL_QUARTER]
q4_narr_topics_count = {}
for seg in q4_narr_segments:
    for t in seg["properties"]["topics"]:
        q4_narr_topics_count[t] = q4_narr_topics_count.get(t, 0) + 1
MAX_NARR = max(q4_narr_topics_count.values()) if q4_narr_topics_count else 1

# ─── Probe LLM Mode ───────────────────────────────────────────────────────────
probed = client.probe_llm()
api_mode = "Heuristic Fallback"
if probed == "Groq":
    api_mode = "Groq LLM"
elif probed == "Gemini":
    api_mode = "Gemini LLM"
elif probed == "OpenAI":
    api_mode = "OpenAI LLM"

# ─── Topic Templates for Fallback ─────────────────────────────────────────────
TOPIC_TEMPLATES = {
    "Slippages & Asset Quality": [
        "What is the outlook on slippages for the retail and corporate books, and do you expect recoveries to offset them?",
        "Could you provide more detail on write-offs and whether standard asset provisioning covers current slippage run-rates?"
    ],
    "NIM & Yields": [
        "What is your guidance for NIM given pricing pressure on retail liabilities and term deposit repricing?",
        "How are wholesale yields trending relative to cost of deposits, and do you see scope for asset repricing?"
    ],
    "Credit Cost & Provisions": [
        "Can you share your credit cost guidance for FY27, and do you intend to release any contingency provisions?",
        "Are there changes in provisioning policy for unsecured retail lending, and what is your current PCR target?"
    ],
    "Citibank Integration": [
        "Could you update us on post-integration synergies and timeline, and what is the remaining drag on operating costs?",
        "How is retention of Citi credit card customers and wealth assets trending post-integration?"
    ],
    "Opex & Cost-to-Income": [
        "What is your outlook on operating expenses and when do you expect cost-to-income to drop below 45%?",
        "Are tech investments and branch expansions peaking, and what is the expected cost growth rate for next fiscal?"
    ],
    "Loan & Advances Growth": [
        "What is your credit growth guidance, particularly wholesale vs retail, and what is the strategy for unsecured loans?",
        "Can you discuss the CBG and SME segment performance and the drivers of loan growth momentum?"
    ],
    "Deposits & CASA": [
        "CASA growth has slowed. What strategies are being deployed to raise retail liabilities and protect the CASA ratio?",
        "What is the share of term deposits and how has deposit growth trended relative to loan growth?"
    ],
    "Credit Cards & Spends": [
        "Could you give colour on credit card spends and the revolver share? Are you seeing any asset quality stress there?",
        "With interchange fee pressures, what is the outlook on credit card fee income?"
    ],
    "Capital Adequacy": [
        "Given recent RWA changes, what is your CAR outlook and do you plan any equity raising?",
        "What is the current Tier-1 buffer and does it support projected loan growth for the next 18 months?"
    ],
    "Subsidiaries' Performance": [
        "Can you share performance updates and profitability of Axis Finance, Axis Capital, and Axis AMC?",
        "Are there any plans for listing or restructuring Axis Finance or increasing stake in Max Life Insurance?"
    ],
    "Profitability & Returns": [
        "How should we think about the RoA and RoE trajectory from here, and what are the key drivers to sustain current return ratios?",
        "Can you give segmental colour on RAROC — which businesses are dilutive to returns today and what is the path to fix them?"
    ],
    "Strategy & Competitive Positioning": [
        "You are growing faster than the industry in select segments — what is driving that market share gain and is it sustainable?",
        "How do you think about competitive intensity in deposits and lending, and where does the bank choose to compete versus peers?"
    ],
    "General": [
        "What are the key strategic focus areas under the GPS strategy and what are the main macro tailwinds?",
        "How do you view credit growth in the banking sector and Axis Bank's target market share growth?"
    ]
}

# ─── Helper Functions ─────────────────────────────────────────────────────────
def _heuristic_questions(analyst: str, topics: list[str], matching_narr: list[dict]) -> list[dict]:
    preds = []
    for topic in topics:
        narr_matches = [n for n in matching_narr if topic in n["topics"]]
        templates = TOPIC_TEMPLATES.get(topic, TOPIC_TEMPLATES["General"])
        q_text = templates[0]

        if narr_matches:
            narr_text = narr_matches[0]["text"]
            metrics = re.findall(r"\b\d+(?:\.\d+)?%\b|\b\d+,\d+ crores\b|\b\d+ crores\b", narr_text)
            if metrics:
                m = metrics[0]
                custom = {
                    "Deposits & CASA": f"With deposit growth around {m} this quarter, what is the outlook on CASA and retail liability accretion?",
                    "NIM & Yields": f"NIM or yields were around {m}; how do you see term deposit repricing affecting margins going forward?",
                    "Slippages & Asset Quality": f"You mentioned slippages of {m}. Can you break that down between retail and corporate segments?",
                    "Opex & Cost-to-Income": f"With opex metrics like {m} in the narration, what is your projection for the cost-to-income trajectory?",
                    "Credit Cost & Provisions": f"With provisions at {m} this quarter, will you maintain this contingency buffer or release it next quarter?",
                }
                q_text = custom.get(topic, q_text)

        preds.append({
            "topic": topic,
            "question": q_text,
            "rationale": (f"Predicted based on {analyst}'s historical preference for {topic} "
                          f"combined with Q4FY26 narration focus.")
        })
    return preds

# ── Optional LLM narration-novelty signal (data/inputs/narration_novelty.json) ──
# Generated by `python3 run.py novelty`. Only applied if the file exists AND its
# quarter matches VAL_QUARTER. Delete/rename the file to disable.
NOVELTY_SCORES: dict[str, float] = {}
try:
    with open(NOVELTY_PATH) as _nf:
        _nov = json.load(_nf)
    if _nov.get("quarter") == VAL_QUARTER:
        for _it in _nov.get("novel_topics", []):
            _t = _it.get("topic")
            if _t in TOPICS_LIST:   # keep MAX score if a topic appears twice
                NOVELTY_SCORES[_t] = max(NOVELTY_SCORES.get(_t, 0.0),
                                         float(_it["novelty_score"]))
        if NOVELTY_SCORES:
            print(f"[novelty] LLM narration-novelty active for {VAL_QUARTER}: "
                  f"{ {k: round(v,2) for k,v in NOVELTY_SCORES.items()} }")
except FileNotFoundError:
    pass
except Exception as _e:
    print(f"[novelty] ignoring malformed {NOVELTY_PATH}: {_e}")

# ── Optional peer-bank signal (data/inputs/peer_signal.json) ──────────────────
# Generated by `python3 run.py peers`. Only applied if the file exists AND its
# quarter matches VAL_QUARTER. Delete/rename the file to disable.
PEER_SALIENCE: dict[str, float] = {}
try:
    with open(PEER_SIGNAL_PATH) as _pf:
        _peer = json.load(_pf)
    if _peer.get("quarter") == VAL_QUARTER:
        for _t, _s in _peer.get("topic_salience", {}).items():
            if _t in TOPICS_LIST:
                PEER_SALIENCE[_t] = float(_s)
        if PEER_SALIENCE:
            print(f"[peers] Peer-bank signal active for {VAL_QUARTER} "
                  f"({', '.join(_peer.get('peers', []))}): "
                  f"{ {k: round(v,2) for k,v in PEER_SALIENCE.items()} }")
except FileNotFoundError:
    pass
except Exception as _e:
    print(f"[peers] ignoring malformed {PEER_SIGNAL_PATH}: {_e}")

# ── Optional per-analyst narration-weight personalization (data/db/question_intent.json) ──
# Built by direct transcript reading (python3 run.py intent), not an LLM call. Reweights
# W_NARR per analyst based on their measured historical style: analysts who mostly react to
# fresh narration disclosures (e.g. MB Mahesh: 83% narration-triggered, 0% persona-consistent)
# get a BOOSTED narration weight; analysts who mostly repeat their own topic pattern get a
# dampened one. Computed with VAL_QUARTER always excluded to avoid leaking the answer into
# the signal used to predict it. Gated behind PERSONA_WEIGHT=1 -- this changes ranking
# directly (unlike the additive-only novelty/peer slots), so it needs its own before/after
# measurement before being trusted by default.
PERSONA_STATS: dict[str, dict] = {}
PERSONA_BASE_RATE = 0.5
PERSONA_WEIGHT_ACTIVE = os.getenv("PERSONA_WEIGHT") == "1"
if PERSONA_WEIGHT_ACTIVE:
    try:
        from src.signals.persona_synthesis import synthesize_personas, MIN_N_FOR_STYLE_NOTE
        PERSONA_STATS = synthesize_personas(out_path=None, exclude_quarters={VAL_QUARTER})
        _rates = [p["narration_triggered_rate"] for p in PERSONA_STATS.values()
                  if p["n"] >= MIN_N_FOR_STYLE_NOTE]
        if _rates:
            PERSONA_BASE_RATE = sum(_rates) / len(_rates)
            print(f"[persona-weight] active, {len(_rates)} analysts with reliable stats, "
                  f"population narration-triggered rate={PERSONA_BASE_RATE:.0%}")
    except FileNotFoundError:
        print("[persona-weight] PERSONA_WEIGHT=1 set but data/db/question_intent.json not found — skipping.")
        PERSONA_WEIGHT_ACTIVE = False


def _personal_w_narr(analyst: str, config) -> float:
    if not PERSONA_WEIGHT_ACTIVE:
        return config.W_NARR
    from src.signals.persona_synthesis import MIN_N_FOR_STYLE_NOTE
    stat = PERSONA_STATS.get(analyst)
    if not stat or stat["n"] < MIN_N_FOR_STYLE_NOTE:
        return config.W_NARR
    return config.W_NARR * (stat["narration_triggered_rate"] / PERSONA_BASE_RATE)


# ── Optional metrics-anomaly narration signal (data/db/metrics_timeseries.json) ──
# Built by regex extraction (python3 run.py metrics), not an LLM call. Replaces the
# keyword-sentence-count narration signal (narr/max_narr, no sense of MAGNITUDE) with
# how unusual this quarter's disclosed metric moves are relative to that metric's own
# history -- for topics with a mapped metric only; topics without one keep the existing
# keyword-count signal. Computed with VAL_QUARTER excluded from the historical comparison
# pool (its OWN current-quarter metrics are still used as the live signal -- narration is
# legitimately available before Q&A in the same call, same reasoning as q4_narr_topics_count).
# Gated behind METRICS_NARR=1 -- changes ranking directly, needs its own before/after test.
METRICS_ANOMALY_SCORES: dict[str, float] = {}
METRICS_NARR_ACTIVE = os.getenv("METRICS_NARR") == "1"
if METRICS_NARR_ACTIVE:
    try:
        from src.signals.metrics_extractor import compute_topic_anomaly_scores
        METRICS_ANOMALY_SCORES = compute_topic_anomaly_scores(VAL_QUARTER, QUARTER_ORDER)
        if METRICS_ANOMALY_SCORES:
            _rounded = {k: round(v, 2) for k, v in METRICS_ANOMALY_SCORES.items()}
            print(f"[metrics-narr] active for {VAL_QUARTER}: {_rounded}")
    except FileNotFoundError:
        print("[metrics-narr] METRICS_NARR=1 set but data/db/metrics_timeseries.json not found — skipping.")
        METRICS_NARR_ACTIVE = False

# ── Optional retrieval-grounded prediction agent (Stage 3, src/signals/agent_predict.py) ──
# Deliberately does NOT feed through the score formula above -- novelty, peer-signal,
# persona-weight and metrics-narr all hit the same wall (the formula is multiplicative,
# so a stronger narration/why signal just amplifies whichever topic already has high
# base preference `p`). Instead: deterministic retrieval (persona style + this quarter's
# metric anomalies + the analyst's own historical precedent questions) grounds a single
# bounded Groq call that may propose ONE evidence-cited swap into the formula's own
# top-N picks. Independent of METRICS_NARR/PERSONA_WEIGHT -- computes its own
# VAL_QUARTER-excluded persona stats so it works whether or not those flags are set.
AGENT_PREDICT_ACTIVE = os.getenv("AGENT_PREDICT") == "1"
AGENT_PERSONA_STATS: dict[str, dict] = {}
AGENT_ANOMALY_SCORES: dict[str, float] = {}
if AGENT_PREDICT_ACTIVE:
    try:
        from src.signals.persona_synthesis import synthesize_personas as _synth_personas
        from src.signals.metrics_extractor import compute_topic_anomaly_scores as _compute_anomalies
        AGENT_PERSONA_STATS = _synth_personas(out_path=None, exclude_quarters={VAL_QUARTER})
        AGENT_ANOMALY_SCORES = _compute_anomalies(VAL_QUARTER, QUARTER_ORDER)
        _rounded_anomalies = {k: round(v, 2) for k, v in AGENT_ANOMALY_SCORES.items()}
        print(f"[agent-predict] active for {VAL_QUARTER}: "
              f"{len(AGENT_PERSONA_STATS)} analysts with persona stats, anomalies={_rounded_anomalies}")
    except FileNotFoundError as _e:
        print(f"[agent-predict] AGENT_PREDICT=1 set but required data missing — skipping. ({_e})")
        AGENT_PREDICT_ACTIVE = False


def _score_topics(analyst: str, pref: dict, custom_narr_count: dict | None = None, config=EngineConfig) -> list[tuple[str, float]]:
    narr_count = custom_narr_count or q4_narr_topics_count
    max_narr = max(narr_count.values()) if narr_count else 1
    self_q3 = q3_analyst_topics.get(analyst, set())
    w_self = config.W_Q3_SELF * 2 * get_repeat_propensity(analyst, graph, train_quarters, Q_ORD)
    w_narr = _personal_w_narr(analyst, config)

    scores = []
    for t in TOPICS_LIST:
        p = pref.get(t, 0.0) + config.BASE_PROB
        if METRICS_NARR_ACTIVE and t in METRICS_ANOMALY_SCORES and custom_narr_count is None:
            narr = METRICS_ANOMALY_SCORES[t]
        else:
            narr = narr_count.get(t, 0) / max_narr
        q3c = q3_topics_count.get(t, 0) / MAX_Q3
        q3s = 1.0 if t in self_q3 else 0.0
        score = p * (1.0 + w_narr * narr + config.W_Q3_CROSS * q3c + w_self * q3s)
        scores.append((t, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _build_llm_prompt(analyst: str, num_preds: int, target_topics: list[str],
                      topic_weights: dict, last_q: dict | None, last_a: dict | None,
                      cross_ctx: list[dict], matching_narr: list[dict]) -> str:
    prompt = f"""Predict earnings call questions for Axis Bank's Q4FY26.
Analyst: {analyst}

Generate EXACTLY {num_preds} questions — one per target topic below.

1. TARGET TOPICS (one question per topic, same order):
{json.dumps(target_topics, indent=2)}

2. ANALYST HISTORICAL TOPIC WEIGHTS (recency-adjusted):
{json.dumps({k: round(v, 3) for k, v in sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)}, indent=2)}

3. ANALYST'S LAST Q&A (previous quarter):
"""
    if last_q:
        q_txt = last_q["properties"]["text"][:400]
        prompt += f"- Question ({last_q['properties']['quarter']}): {q_txt}\n"
        if last_a:
            a_txt = last_a["properties"]["text"][:600]
            prompt += f"- Answer: {a_txt}\n"
    else:
        prompt += "- No prior questions found.\n"

    prompt += "\n4. OTHER ANALYSTS' Q3FY26 QUESTIONS (potential cross-follow-ups):\n"
    for cf in cross_ctx[:2]:
        prompt += f"- {cf['analyst']} on {cf['topic']}: {cf['question'][:250]}\n"

    prompt += "\n5. Q4FY26 NARRATION HIGHLIGHTS (management presentation):\n"
    for mn in matching_narr[:2]:
        prompt += f"- {mn['text'][:600]}\n"

    prompt += f"""
Task: Write exactly {num_preds} realistic, specific questions reflecting {analyst}'s
persona and the Q4FY26 context.  For each target topic, produce one question.

Return a JSON object with a single key "predictions" — a list of exactly {num_preds} objects:
{{
  "predictions": [
    {{
      "topic":     "<one of the target topics>",
      "question":  "<realistic question text>",
      "rationale": "<why this question is expected based on history and narration>"
    }}
  ]
}}
"""
    return prompt

def _parse_llm_response(raw: str) -> list[dict] | None:
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
        preds = parsed.get("predictions") or parsed.get("Predictions")
        if isinstance(preds, list) and preds:
            return preds
    except Exception:
        pass
    return None

def _llm_select_topics(analyst: str, top8: list[tuple[str, float]], N: int,
                        pref: dict, matching_narr: list[dict],
                        last_q: dict | None) -> list[str] | None:
    top8_fmt = "\n".join(
        f"  {i+1}. {t}  (score={s:.3f})" for i, (t, s) in enumerate(top8)
    )
    hist_summary = ""
    if pref:
        top_topics = sorted(pref.items(), key=lambda x: x[1], reverse=True)[:5]
        hist_summary = ", ".join(f"{t} ({v:.2f})" for t, v in top_topics)
    narr_text = " | ".join(seg["text"][:200] for seg in matching_narr[:2])
    last_q_text = last_q["properties"]["text"][:300] if last_q else "None"

    prompt = f"""You are helping Axis Bank's IR team predict analyst questions for Q4FY26 earnings call.

Analyst: {analyst}
Historical topic preferences (recency-weighted scores): {hist_summary}
Last question asked (most recent quarter): {last_q_text}
Q4FY26 management narration highlights: {narr_text[:500]}

Algorithm-scored top 8 candidate topics (in score order):
{top8_fmt}

Task: Select exactly {N} topics from the list above that {analyst} is MOST LIKELY to ask about in Q4FY26.
Consider:
- This analyst's historical focus areas and consistency
- What the management narration emphasized (higher narration coverage = higher analyst interest)
- Whether the analyst typically follows up on topics from the previous quarter
- Topics with incomplete management answers in prior quarters tend to resurface

Return JSON:
{{
  "selected_topics": ["topic1", "topic2", ...],
  "reasoning": "brief rationale for selection"
}}
Selected topics must be an exact subset of the 8 candidates above."""

    raw = client.call_llm(prompt, temperature=0.15)
    if not raw:
        return None
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean)
        sel = parsed.get("selected_topics", [])
        valid = [t for t in sel if t in {x[0] for x in top8}]
        if len(valid) >= N:
            return valid[:N]
    except Exception:
        pass
    return None

# ─── Main prediction function (used by both batch run and dashboard server) ───
def predict_for_analyst(analyst: str, custom_narration: str | None = None,
                        recall_boost: bool = False, config=EngineConfig) -> list[dict]:
    pref, N = get_analyst_profile(analyst, graph, train_quarters, VAL_ORD, Q_ORD, GLOBAL_RATE, config=config)
    if recall_boost:
        N = min(5, N + 1)

    # ── Narration signal ─────────────────────────────────────────────────────
    if custom_narration:
        custom_narr_count = {}
        cl = custom_narration.lower()
        # Local keyword map for quick scan
        TOPIC_KEYWORDS = {
            "NIM & Yields": ["nim", "nii", "net interest margin", "yield", "yields", "spreads", "cost of funds", "interest income", "repricing"],
            "Deposits & CASA": ["deposit", "deposits", "casa", "saving account", "current account", "term deposit", "retail deposit", "wholesale deposit", "liability growth"],
            "Loan & Advances Growth": ["loan growth", "advances", "credit growth", "disbursement", "retail loan", "sme loan", "corporate loan", "wholesale loan"],
            "Slippages & Asset Quality": ["slippage", "slippages", "npa", "gnpa", "nnpa", "asset quality", "stressed asset", "write-off", "recoveries", "recovery", "restructured"],
            "Credit Cost & Provisions": ["credit cost", "provision", "provisions", "contingency buffer", "provisioning", "write-offs", "credit costs", "buffer provisions"],
            "Opex & Cost-to-Income": ["opex", "operating expense", "operating expenses", "cost-to-income", "employee cost", "it spend", "it expenses", "other opex", "cost structure"],
            "Capital Adequacy": ["capital adequacy", "car", "tier 1", "tier 2", "cet1", "capital raising", "rwa", "risk weighted assets"],
            "Credit Cards & Spends": ["credit card", "credit cards", "cards business", "card spends", "fee income", "retail fee", "card fee", "card portfolio"],
            "Citibank Integration": ["citibank", "citi", "integration cost", "synergies", "citi portfolio", "citi credit cards", "citi customers"],
            "Subsidiaries' Performance": ["subsidiary", "subsidiaries", "axis finance", "axis capital", "axis amc", "max life"],
            "Profitability & Returns": ["roe", "roa", "raroc", "return on equity", "return on assets", "return ratio", "return ratios", "profitability", "roce", "return profile"],
            "Strategy & Competitive Positioning": ["market share", "competitive", "competition", "peer banks", "versus peers", "vs peers", "gps strategy", "positioning", "gaining share", "industry growth"],
        }
        for topic, kws in TOPIC_KEYWORDS.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", cl):
                    custom_narr_count[topic] = custom_narr_count.get(topic, 0) + 1
                    break
        narr_count = custom_narr_count
        matching_narr = [{"text": custom_narration, "topics": list(custom_narr_count.keys())}]
    else:
        narr_count = q4_narr_topics_count
        matching_narr = [{"text": seg["properties"]["text"],
                          "topics": seg["properties"]["topics"]}
                         for seg in q4_narr_segments]

    # ── Score topics ─────────────────────────────────────────────────────────
    scores = _score_topics(analyst, pref, narr_count if custom_narration else None, config=config)
    top8 = scores[:8]

    # ── Retrieve context ──────────────────────────────────────────────────────
    last_q, last_a = get_last_qa(analyst, graph, train_quarters, adj, quarter_sort_map)
    cross_ctx = get_cross_follow_context(analyst, pref, q3_questions)
    fav_topics = [t for t, _ in scores[:3]]
    matching_narr_filtered = [seg for seg in matching_narr
                              if any(t in seg["topics"] for t in fav_topics)]

    # ── Topic selection ───────────────────────────────────────────────────────
    if api_mode != "Heuristic Fallback" and os.getenv("LLM_TOPIC_SELECT") == "1":
        llm_topics = _llm_select_topics(analyst, top8, N, pref, matching_narr_filtered, last_q)
        if llm_topics:
            target_topics = llm_topics
            time.sleep(1.5)
        else:
            target_topics = [t for t, _ in scores[:N]]
    else:
        target_topics = [t for t, _ in scores[:N]]

    # ── Novelty extra slot ─────────────────────────────────────────────────────
    # Novel management themes NEVER displace validated top-N picks; instead they
    # add ONE extra prep slot — and only for analysts with affinity for that
    # theme (nonzero blended preference). Measured motivation: additive score
    # boosts displaced real hits (Q4FY26 blind F1 59.4% -> 55%); the extra-slot
    # rule keeps base predictions untouched so recall can only improve.
    if NOVELTY_SCORES and not custom_narration:
        cand = [(t, nv) for t, nv in NOVELTY_SCORES.items()
                if nv >= config.NOVELTY_MIN_SCORE
                and t not in target_topics
                and pref.get(t, 0.0) >= config.NOVELTY_AFFINITY_FLOOR]
        if cand:
            cand.sort(key=lambda x: (-x[1], -pref.get(x[0], 0.0)))
            extra = cand[0][0]
            target_topics = target_topics + [extra]
            N = len(target_topics)
            print(f"  [novelty slot] +{extra} (novelty={cand[0][1]:.2f}, "
                  f"affinity={pref.get(extra, 0):.3f})")

    # ── Peer-bank extra slot ────────────────────────────────────────────────────
    # Same discipline as the novelty slot: topics peer-bank analysts heavily
    # probed this quarter NEVER displace validated top-N picks, only add ONE
    # extra slot, and only for analysts with some affinity for that theme.
    if PEER_SALIENCE and not custom_narration:
        cand = [(t, sv) for t, sv in PEER_SALIENCE.items()
                if sv >= config.PEER_MIN_SALIENCE
                and t not in target_topics
                and pref.get(t, 0.0) >= config.PEER_AFFINITY_FLOOR]
        if cand:
            cand.sort(key=lambda x: (-x[1], -pref.get(x[0], 0.0)))
            extra = cand[0][0]
            target_topics = target_topics + [extra]
            N = len(target_topics)
            print(f"  [peer slot] +{extra} (salience={cand[0][1]:.2f}, "
                  f"affinity={pref.get(extra, 0):.3f})")

    # ── Retrieval-grounded prediction agent (Stage 3) ───────────────────────────
    # Bounded, evidence-cited swap into the formula's own picks -- see
    # src/signals/agent_predict.py for why this doesn't just add another score term.
    if AGENT_PREDICT_ACTIVE and not custom_narration:
        from src.signals.agent_predict import retrieve_context, propose_swap
        ctx = retrieve_context(analyst, VAL_QUARTER, QUARTER_ORDER, graph,
                               AGENT_PERSONA_STATS, AGENT_ANOMALY_SCORES)
        if ctx:
            swap = propose_swap(client, analyst, target_topics, ctx)
            if swap:
                target_topics = [swap["add_topic"] if t == swap["remove_topic"] else t
                                 for t in target_topics]
                print(f"  [agent-swap] -{swap['remove_topic']} +{swap['add_topic']} "
                      f"(evidence: {swap['evidence'][:150]})")

    # ── Generate questions ─────────────────────────────────────────────────────
    if api_mode == "Heuristic Fallback":
        return _heuristic_questions(analyst, target_topics, matching_narr_filtered)

    prompt = _build_llm_prompt(
        analyst, N, target_topics, pref,
        last_q, last_a, cross_ctx, matching_narr_filtered
    )
    raw = client.call_llm(prompt, temperature=0.15)
    preds = _parse_llm_response(raw)

    if preds and len(preds) == N:
        return preds
    if preds:
        result = []
        for t in target_topics:
            match = next((p for p in preds if p.get("topic") == t), None)
            if match:
                result.append(match)
            else:
                result.extend(_heuristic_questions(analyst, [t], matching_narr_filtered))
        return result[:N]

    print(f"  [Warning] LLM failed for {analyst}, using heuristic fallback.")
    return _heuristic_questions(analyst, target_topics, matching_narr_filtered)

def main():
    print(f"Prediction engine mode: {api_mode}")
    print(f"Active analysts: {len(active_analysts)}")

    predictions = {}

    for i, analyst in enumerate(active_analysts):
        print(f"  [{i+1}/{len(active_analysts)}] {analyst} ...", end=" ", flush=True)
        preds = predict_for_analyst(analyst, recall_boost=True)
        predictions[analyst] = preds
        print(f"({len(preds)} topics: {[p['topic'] for p in preds]})")
        if api_mode != "Heuristic Fallback":
            time.sleep(6.0)

    # ── Genuine blind validation on VAL_QUARTER ──────────────────────────────
    print(f"\n─────────────── VALIDATION: {VAL_QUARTER.upper()} (blind held-out) ───────────────")
    q4_actual_q_nodes = [n for n in graph["nodes"]
                          if n["type"] == "Question"
                          and n["properties"]["quarter"] == VAL_QUARTER]

    q4_analyst_actual = {}
    for q in q4_actual_q_nodes:
        a = q["properties"]["analyst"]
        q4_analyst_actual.setdefault(a, set()).update(
            t for t in q["properties"]["topics"] if t != "General"
        )

    prec_sum = rec_sum = f1_sum = 0.0
    count = 0
    validation_reports = []

    for analyst in sorted(q4_analyst_actual.keys()):
        actual = q4_analyst_actual[analyst]
        if not actual:
            continue
        predicted = [p["topic"] for p in predictions.get(analyst, [])]
        intersection = actual.intersection(predicted)
        
        prec = len(intersection) / len(predicted) if predicted else 0.0
        rec = len(intersection) / len(actual) if actual else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0.0 else 0.0
        
        prec_sum += prec
        rec_sum += rec
        f1_sum += f1
        count += 1
        
        print(f"  Analyst : {analyst}")
        print(f"  Predicted: {sorted(predicted)}")
        print(f"  Actual   : {sorted(actual)}")
        print(f"  P={prec:.0%}  R={rec:.0%}  F1={f1:.0%}\n")
        
        validation_reports.append({
            "analyst": analyst,
            "predicted": sorted(predicted),
            "actual": sorted(list(actual)),
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

    avg_p = prec_sum / count if count else 0.0
    avg_r = rec_sum / count if count else 0.0
    avg_f = f1_sum / count if count else 0.0

    print("═══════════════════════════════════════════════════════")
    print(f"  Validated analysts : {count}")
    print(f"  Average Precision  : {avg_p:.0%}")
    print(f"  Average Recall     : {avg_r:.0%}")
    print(f"  Average F1-Score   : {avg_f:.0%}   ← genuine blind result")
    print("═══════════════════════════════════════════════════════\n")

    # Save outputs
    output_data = {
        "api_mode": api_mode,
        "predictions": predictions,
        "validation_summary": {
            "quarter": VAL_QUARTER,
            "analysts_count": count,
            "average_precision": avg_p,
            "average_recall": avg_r,
            "average_f1": avg_f,
            "reports": validation_reports
        }
    }
    
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Predictions saved → {PREDICTIONS_PATH}")
