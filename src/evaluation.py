import os
import re
import json
import math
import time
import urllib.request
import urllib.error
from src.config.settings import (
    VAL_QUARTER,
    ANALYST_ALIASES,
    TOPICS_LIST,
    PREDICTIONS_PATH,
    CROSSVAL_PATH,
    SEMANTIC_EVAL_PATH,
)
from src.data.loader import load_dataset, load_graph, get_active_analysts
from src.model_provider.llm_client import client
from src.memory.history import PERSONAS

# ─── Semantic Evaluator Logic ──────────────────────────────────────────────────
def _build_judge_prompt(analyst: str, pred_questions: list[str],
                        actual_questions: list[str]) -> str:
    # Truncate to allow full text to be seen by judge
    preds_text   = "\n".join(f"  P{i+1}: {q[:500]}" for i, q in enumerate(pred_questions))
    actuals_text = "\n".join(f"  A{i+1}: {q[:2000]}" for i, q in enumerate(actual_questions))
    return f"""Evaluate predicted vs actual analyst questions for Axis Bank Q4FY26 (analyst: {analyst}).

PREDICTED:
{preds_text}

ACTUAL:
{actuals_text}

Score each (P_i, A_j) pair:
  1.0 = same financial concern (even if worded differently)
  0.5 = same broad topic, different angle
  0.0 = different topic

Then compute:
  semantic_precision = average of max(scores across all A_j) for each P_i
  semantic_recall    = average of max(scores across all P_i) for each A_j

Return JSON only:
{{"analyst":"{analyst}","pred_vs_actual_scores":[{{"pred_idx":1,"actual_idx":1,"score":0.0,"reason":"short reason"}}],"semantic_precision":0.0,"semantic_recall":0.0}}"""

def _safe_parse(raw: str) -> dict | None:
    if not raw: return None
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(clean)
    except Exception:
        return None

def _kw_score(pred_q: str, actual_q: str, pred_topic: str, actual_topic: str) -> float:
    """Keyword-based semantic similarity score (offline fallback)."""
    TOPIC_KWS = {
        "NIM & Yields":             {"nim","nii","net interest","margin","yield","cost of fund","repricing","interest income","spread"},
        "Deposits & CASA":          {"deposit","deposits","casa","current account","saving","term deposit","liability","rtd"},
        "Loan & Advances Growth":   {"loan","advance","advances","credit growth","disbursement","retail lending","wholesale","sme","cbg","unsecured"},
        "Slippages & Asset Quality":{"slippage","slippages","npa","gnpa","nnpa","asset quality","stressed","write-off","recovery","restructured","delinquency"},
        "Credit Cost & Provisions": {"credit cost","provision","provisions","provisioning","pcr","contingency","buffer","write-back"},
        "Opex & Cost-to-Income":    {"opex","operating expense","operating cost","cost-to-income","employee","technology spend","it spend"},
        "Capital Adequacy":         {"capital","car","tier1","tier 1","cet1","rwa","risk weighted","equity raising"},
        "Credit Cards & Spends":    {"credit card","cards","card spend","spends","revolver","interchange","card portfolio","card fee"},
        "Citibank Integration":     {"citi","citibank","integration","synergy","synergies","goodwill","acquisition"},
        "Subsidiaries' Performance":{"subsidiary","subsidiaries","axis finance","axis amc","axis capital","max life"},
        "Profitability & Returns":  {"roe","roa","raroc","return on equity","return on assets","return ratio","profitability","roce"},
        "Strategy & Competitive Positioning": {"market share","competitive","competition","peer banks","versus peers","positioning","gps strategy"},
    }
    pred_lower   = pred_q.lower()
    actual_lower = actual_q.lower()
    topic_bonus = 0.5 if pred_topic == actual_topic else 0.0
    kws = TOPIC_KWS.get(pred_topic, set())
    hits = sum(1 for kw in kws if kw in actual_lower)
    kw_score = min(0.5, hits / max(len(kws), 1) * 2.0) if kws else 0.0
    return min(1.0, topic_bonus + kw_score)

def run_semantic_evaluator():
    with open(PREDICTIONS_PATH) as f: pred_data  = json.load(f)
    graph      = load_graph()
    with open(CROSSVAL_PATH) as f: cv_data = json.load(f)

    # ── Startup probe
    probed = client.probe_llm()
    use_llm = bool(probed)
    active_llm_name = probed

    actual_by_analyst = {}
    for n in graph["nodes"]:
        if n["type"] == "Question" and n["properties"]["quarter"] == VAL_QUARTER:
            a = n["properties"]["analyst"]
            actual_by_analyst.setdefault(a, []).append({
                "text":   n["properties"]["text"],
                "topics": [t for t in n["properties"]["topics"] if t != "General"]
            })

    print()
    print("=" * 72)
    print("  IR COCKPIT — SEMANTIC EVALUATION REPORT (Q4FY26 blind test)")
    print("=" * 72)
    print()

    all_sem_p  = []
    all_sem_r  = []
    all_sem_f1 = []
    all_top_f1 = []
    detailed_results = []

    validated = [a for a in pred_data["predictions"] if a in actual_by_analyst]
    print(f"Analysts validated: {len(validated)}\n")

    for analyst in sorted(validated):
        preds      = pred_data["predictions"][analyst]
        actual_qs  = actual_by_analyst[analyst]
        pred_qs    = [p["question"] for p in preds]
        pred_topics = [p["topic"] for p in preds]

        actual_topics_all = set(t for q in actual_qs for t in q["topics"])
        pred_topics_set   = set(pred_topics)
        inter             = pred_topics_set & actual_topics_all
        tp_p              = len(inter) / len(pred_topics_set)   if pred_topics_set   else 0.0
        tp_r              = len(inter) / len(actual_topics_all) if actual_topics_all else 0.0
        tp_f1             = 2 * tp_p * tp_r / (tp_p + tp_r) if (tp_p + tp_r) > 0 else 0.0

        sem_p = sem_r = sem_f1 = None
        pair_scores = []

        if use_llm:
            prompt = _build_judge_prompt(analyst, pred_qs, [q["text"] for q in actual_qs])
            raw    = client.call_llm(prompt, temperature=0.0)
            parsed = _safe_parse(raw) if raw else None
            if parsed and "semantic_precision" in parsed:
                sem_p  = parsed["semantic_precision"]
                sem_r  = parsed["semantic_recall"]
                sem_f1 = 2 * sem_p * sem_r / (sem_p + sem_r) if (sem_p + sem_r) > 0 else 0.0
                pair_scores = parsed.get("pred_vs_actual_scores", [])
                print(f"  ✓ LLM judge scored ({sem_p:.0%} P, {sem_r:.0%} R)")
            else:
                print(f"  ⚠  LLM parse failed for {analyst}, using keyword fallback.")
            if analyst != sorted(validated)[-1]:
                print(f"  [Rate limit pause 20s ...]")
                time.sleep(20)

        if sem_p is None:
            # Offline keyword fallback
            p_scores = []
            for pred_q, pred_topic in zip(pred_qs, pred_topics):
                best = max(
                    (_kw_score(pred_q, aq["text"], pred_topic, t)
                     for aq in actual_qs for t in (aq["topics"] or ["General"])),
                    default=0.0
                )
                p_scores.append(best)
            r_scores = []
            for aq in actual_qs:
                for t in (aq["topics"] or ["General"]):
                    best = max(
                        (_kw_score(pq, aq["text"], pt, t)
                         for pq, pt in zip(pred_qs, pred_topics)),
                        default=0.0
                    )
                    r_scores.append(best)
            sem_p  = sum(p_scores) / len(p_scores) if p_scores else 0.0
            sem_r  = sum(r_scores) / len(r_scores) if r_scores else 0.0
            sem_f1 = 2 * sem_p * sem_r / (sem_p + sem_r) if (sem_p + sem_r) > 0 else 0.0

        all_top_f1.append(tp_f1)
        all_sem_p.append(sem_p)
        all_sem_r.append(sem_r)
        all_sem_f1.append(sem_f1)

        print(f"  Analyst : {analyst}")
        print(f"  Predicted topics : {sorted(pred_topics_set)}")
        print(f"  Actual topics    : {sorted(actual_topics_all)}")
        print(f"  Matched topics   : {sorted(inter)}")
        print(f"  Topic-F1={tp_f1:.0%}   Semantic-P={sem_p:.0%}  Semantic-R={sem_r:.0%}  Semantic-F1={sem_f1:.0%}")
        if pair_scores:
            print("  Pair scores from LLM judge:")
            for ps in pair_scores[:6]:
                print(f"    P{ps['pred_idx']} vs A{ps['actual_idx']}: {ps['score']} — {ps.get('reason','')[:80]}")
        print()

        detailed_results.append({
            "analyst":       analyst,
            "topic_f1":      tp_f1,
            "semantic_p":    sem_p,
            "semantic_r":    sem_r,
            "semantic_f1":   sem_f1,
            "pred_topics":   sorted(pred_topics_set),
            "actual_topics": sorted(actual_topics_all),
            "matched_topics":sorted(inter),
            "pair_scores":   pair_scores
        })

    avg_top  = sum(all_top_f1) / len(all_top_f1)  if all_top_f1  else 0.0
    avg_sp   = sum(all_sem_p)  / len(all_sem_p)   if all_sem_p   else 0.0
    avg_sr   = sum(all_sem_r)  / len(all_sem_r)   if all_sem_r   else 0.0
    avg_sf   = sum(all_sem_f1) / len(all_sem_f1)  if all_sem_f1  else 0.0

    print("=" * 72)
    print(f"  Q4FY26 OVERALL ({len(validated)} analysts):")
    print(f"    Topic-F1 (exact label)   : {avg_top:.1%}")
    print(f"    Semantic Precision       : {avg_sp:.1%}")
    print(f"    Semantic Recall          : {avg_sr:.1%}")
    print(f"    Semantic-F1 (LLM judge)  : {avg_sf:.1%}")
    print()
    print(f"  Cross-val Topic-F1 (92 calls, 14 quarters) : {cv_data['overall_f1_base']:.1%}")
    print(f"  Cross-val with recall-boost               : {cv_data['overall_f1_boost']:.1%}")
    print("=" * 72)

    output_summary = {
        "evaluation_mode": f"{active_llm_name} LLM" if active_llm_name else "Keyword Offline",
        "q4fy26": {
            "analysts": len(validated),
            "avg_topic_f1": avg_top,
            "avg_semantic_p": avg_sp,
            "avg_semantic_r": avg_sr,
            "avg_semantic_f1": avg_sf,
            "details": detailed_results
        },
        "crossval": {
            "quarters": cv_data["quarters"],
            "total_calls": cv_data["total_calls"],
            "avg_topic_f1_base": cv_data["overall_f1_base"],
            "avg_topic_f1_boost": cv_data["overall_f1_boost"]
        }
    }

    with open(SEMANTIC_EVAL_PATH, "w") as f:
        json.dump(output_summary, f, indent=2)
    print(f"Results saved to {SEMANTIC_EVAL_PATH}")


# ─── Cross Validation Logic ─────────────────────────────────────────────────────
class Config:
    canonical = False
    ordinal   = False
    prior     = False
    momentum  = False
    repeat    = False
    personas  = False
    boost     = False
    novelty   = False   # narration-novelty boost (new/rising themes in mgmt presentation)
    cooc      = False   # topic co-occurrence boost (topics that bundle with picked ones)
    metrics_narr = False  # replace keyword-count narration signal with metric-anomaly score
    DECAY      = 0.40
    W_NARR     = 3.5
    W_CROSS    = 1.5
    W_SELF     = 2.0
    BASE_PROB  = 0.04
    PRIOR_MIX  = 0.25
    PERSONA_MIX= 0.30
    RECENT_WIN = 2.0
    W_NOV      = 1.5    # novelty boost weight
    W_COOC     = 0.8    # co-occurrence boost weight
    NOV_LOOKBACK = 3    # prior quarters of narration to compare against

def canon(name: str, enabled: bool) -> str:
    aliases = {
        "Ma hrukh Adajania": "Mahrukh Adajania",
        "Sam eer Bhise": "Sameer Bhise",
        "Sum eet Kariwala": "Sumeet Kariwala",
        "Pra khar Sharma": "Prakhar Sharma",
        "Harsh Modi": "Harsh Wardhan Modi",
        "Krishnan": "Krishnan ASV",
        "Nilanjan": "Nilanjan Karfa",
    }
    return aliases.get(name, name) if enabled else name

def run_cv(cfg: Config, ALL_Q, NARR, quarters, q_index, q_sortkey, topics_list, start_quarter="q3fy23", end_quarter=None, verbose=False):
    per_quarter = {}
    all_f1, all_p, all_r = [], [], []
    lifetime = {}

    start_i = q_index[start_quarter]
    end_i = q_index[end_quarter] if end_quarter else len(quarters) - 1

    for vi in range(start_i, end_i + 1):
        vq = quarters[vi]
        train_qs = set(quarters[:vi])
        prev_qs = quarters[max(0, vi - (2 if cfg.momentum else 1)):vi]

        train_pool = [q for q in ALL_Q if q["properties"]["quarter"] in train_qs]

        gcount = {}
        for q in train_pool:
            for t in q["properties"]["topics"]:
                if t != "General":
                    gcount[t] = gcount.get(t, 0) + 1
        gtot = sum(gcount.values()) or 1
        grate = {t: gcount.get(t, 0) / gtot for t in topics_list}

        cross_count = {}
        self_topics = {}
        for pq_i, pq in enumerate(reversed(prev_qs)):
            w = 1.0 if pq_i == 0 else 0.5
            for q in train_pool:
                if q["properties"]["quarter"] != pq:
                    continue
                a = canon(q["properties"]["analyst"], cfg.canonical)
                for t in q["properties"]["topics"]:
                    if t == "General":
                        continue
                    cross_count[t] = cross_count.get(t, 0.0) + w
                    if pq_i == 0:
                        self_topics.setdefault(a, set()).add(t)
        max_cross = max(cross_count.values()) if cross_count else 1.0

        narr_count = {}
        for seg in NARR:
            if seg["properties"]["quarter"] == vq:
                for t in seg["properties"]["topics"]:
                    if t != "General":
                        narr_count[t] = narr_count.get(t, 0) + 1
        max_narr = max(narr_count.values()) if narr_count else 1

        # ── Metrics-anomaly narration signal: replaces narr/max_narr for topics with a
        # mapped metric, using how unusual this quarter's disclosed move is vs. that
        # metric's own history (quarters strictly before vq) -- same discipline as engine.py.
        metrics_topic_scores = {}
        if cfg.metrics_narr:
            from src.signals.metrics_extractor import compute_topic_anomaly_scores
            try:
                metrics_topic_scores = compute_topic_anomaly_scores(vq, quarters)
            except FileNotFoundError:
                pass

        # ── Narration novelty: current narration salience minus prior-quarters avg ──
        novelty = {}
        if cfg.novelty:
            look_qs = quarters[max(0, vi - cfg.NOV_LOOKBACK):vi]
            prev_norm = {}
            for lq in look_qs:
                lc = {}
                for seg in NARR:
                    if seg["properties"]["quarter"] == lq:
                        for t in seg["properties"]["topics"]:
                            if t != "General":
                                lc[t] = lc.get(t, 0) + 1
                lmax = max(lc.values()) if lc else 1
                for t, c in lc.items():
                    prev_norm[t] = prev_norm.get(t, 0.0) + (c / lmax) / max(len(look_qs), 1)
            for t in topics_list:
                now = narr_count.get(t, 0) / max_narr
                novelty[t] = max(0.0, now - prev_norm.get(t, 0.0))

        # ── Topic co-occurrence: P(t in block | s in block) from training pool ──
        cooc = {}
        if cfg.cooc:
            single = {}
            for q in train_pool:
                ts = [t for t in q["properties"]["topics"] if t != "General"]
                for s in ts:
                    single[s] = single.get(s, 0) + 1
                    for t in ts:
                        if t != s:
                            cooc.setdefault(s, {})[t] = cooc.get(s, {}).get(t, 0) + 1
            cooc = {s: {t: c / single[s] for t, c in d.items()}
                    for s, d in cooc.items()}

        hist = {}
        for q in train_pool:
            a = canon(q["properties"]["analyst"], cfg.canonical)
            hist.setdefault(a, []).append(q)

        repeat_rate = {}
        if cfg.repeat:
            for a, qs in hist.items():
                by_q = {}
                for q in qs:
                    by_q.setdefault(q["properties"]["quarter"], set()).update(
                        t for t in q["properties"]["topics"] if t != "General")
                keys = sorted(by_q, key=lambda x: q_index[x])
                num = den = 0
                for k1, k2 in zip(keys, keys[1:]):
                    if q_index[k2] - q_index[k1] == 1:
                        den += len(by_q[k1])
                        num += len(by_q[k1] & by_q[k2])
                repeat_rate[a] = num / den if den >= 3 else 0.5

        actual = {}
        for q in ALL_Q:
            if q["properties"]["quarter"] != vq:
                continue
            a = canon(q["properties"]["analyst"], cfg.canonical)
            ts = {t for t in q["properties"]["topics"] if t != "General"}
            if ts:
                actual.setdefault(a, set()).update(ts)

        q_f1, q_p, q_r, n_val = [], [], [], 0
        max_key = max(q_sortkey[q] for q in train_qs)

        for a, actual_set in actual.items():
            qs = hist.get(a, [])
            if not qs:
                continue
            tw = {}
            total_w = 0.0
            all_counts, recent_counts = [], []
            for q in qs:
                qq = q["properties"]["quarter"]
                qa = (vi - 1 - q_index[qq]) if cfg.ordinal else (max_key - q_sortkey[qq]) / 10.0
                w = math.exp(-cfg.DECAY * qa)
                total_w += w
                tc = [t for t in q["properties"]["topics"] if t != "General"]
                all_counts.append(len(tc))
                if qa <= cfg.RECENT_WIN:
                    recent_counts.append(len(tc))
                for t in tc:
                    tw[t] = tw.get(t, 0.0) + w
            pref = {t: v / total_w for t, v in tw.items()} if total_w else {}

            n_hist_quarters = len({q["properties"]["quarter"] for q in qs})
            if cfg.prior:
                mix = cfg.PRIOR_MIX if n_hist_quarters >= 3 else min(0.6, cfg.PRIOR_MIX * 2)
                pref = {t: (1 - mix) * pref.get(t, 0.0) + mix * grate.get(t, 0.0)
                        for t in topics_list}
            if cfg.personas and a in PERSONAS:
                pp = PERSONAS[a].get("topic_prior", {})
                if pp:
                    s = sum(pp.values()) or 1
                    pp = {t: v / s for t, v in pp.items()}
                    pw = cfg.PERSONA_MIX / (1 + 0.35 * max(0, n_hist_quarters - 2))
                    pref = {t: (1 - pw) * pref.get(t, 0.0) + pw * pp.get(t, 0.0)
                            for t in topics_list}

            avg_all = sum(all_counts) / len(all_counts) if all_counts else 2.0
            max_recent = max(recent_counts) if recent_counts else round(avg_all)
            N = max(1, min(5, round((max_recent + avg_all) / 2.0)))
            if cfg.boost:
                N = min(5, N + 1)

            self_set = self_topics.get(a, set())
            w_self_a = cfg.W_SELF
            if cfg.repeat:
                w_self_a = cfg.W_SELF * 2 * repeat_rate.get(a, 0.5)

            scores = []
            for t in topics_list:
                p = pref.get(t, 0.0) + cfg.BASE_PROB
                if cfg.metrics_narr and t in metrics_topic_scores:
                    narr_term = metrics_topic_scores[t]
                else:
                    narr_term = narr_count.get(t, 0) / max_narr
                s = p * (1.0
                         + cfg.W_NARR * narr_term
                         + cfg.W_CROSS * (cross_count.get(t, 0.0) / max_cross)
                         + w_self_a * (1.0 if t in self_set else 0.0)
                         + (cfg.W_NOV * novelty.get(t, 0.0) if cfg.novelty else 0.0))
                scores.append((t, s))
            scores.sort(key=lambda x: x[1], reverse=True)

            # ── Co-occurrence re-rank: boost topics that bundle with the seeds ──
            if cfg.cooc and N >= 2:
                seeds = [t for t, _ in scores[:max(1, N - 1)]]
                rescored = []
                for t, s in scores:
                    if t in seeds:
                        rescored.append((t, s))
                    else:
                        cf = max((cooc.get(sd, {}).get(t, 0.0) for sd in seeds), default=0.0)
                        rescored.append((t, s * (1.0 + cfg.W_COOC * cf)))
                rescored.sort(key=lambda x: x[1], reverse=True)
                scores = rescored

            pred_set = {t for t, _ in scores[:N]}

            inter = pred_set & actual_set
            p_ = len(inter) / len(pred_set) if pred_set else 0.0
            r_ = len(inter) / len(actual_set) if actual_set else 0.0
            f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
            q_f1.append(f1); q_p.append(p_); q_r.append(r_); n_val += 1
            lifetime.setdefault(a, []).append(f1)

        if n_val:
            per_quarter[vq] = {"n": n_val,
                               "f1": sum(q_f1) / n_val,
                               "precision": sum(q_p) / n_val,
                               "recall": sum(q_r) / n_val}
            all_f1 += q_f1; all_p += q_p; all_r += q_r
            if verbose:
                print(f"  {vq}: n={n_val}  P={per_quarter[vq]['precision']:.0%} "
                      f"R={per_quarter[vq]['recall']:.0%}  F1={per_quarter[vq]['f1']:.0%}")

    n = len(all_f1)
    mean_f1 = sum(all_f1) / n
    std = math.sqrt(sum((x - mean_f1) ** 2 for x in all_f1) / n)
    return {
        "overall_f1": mean_f1,
        "overall_precision": sum(all_p) / n,
        "overall_recall": sum(all_r) / n,
        "std": std,
        "ci95": 1.96 * std / math.sqrt(n),
        "total_calls": n,
        "per_quarter": per_quarter,
        "lifetime": {a: {"calls": len(v), "avg_f1": sum(v) / len(v)}
                     for a, v in lifetime.items()},
    }

def run_cross_validation():
    # Load dataset & graph directly
    dataset = load_dataset(canonicalize=False)
    graph = load_graph(canonicalize=False)

    print("Running cross-validation backtest over 14 historical quarters...")
    
    # ── Prepare Index ──
    quarters = [q["quarter_id"] for q in dataset]
    q_index = {q: i for i, q in enumerate(quarters)}
    q_sortkey = {q["quarter_id"]: q["sort_key"] for q in dataset}

    ALL_Q = [n for n in graph["nodes"] if n["type"] == "Question"]
    NARR  = [n for n in graph["nodes"] if n["type"] == "NarrationSegment"]
    topics_list = [n["id"] for n in graph["nodes"] if n["type"] == "Topic" and n["id"] != "General"]

    # 1. Base config (no momentum, no prior blend, etc.)
    base_cfg = Config()
    
    # 2. V2 config (production engine parameters)
    v2_cfg = Config()
    v2_cfg.canonical = True
    v2_cfg.ordinal   = True
    v2_cfg.prior     = True
    v2_cfg.momentum  = True
    v2_cfg.repeat    = True
    v2_cfg.personas  = True
    v2_cfg.boost     = False
    
    # Production values
    v2_cfg.DECAY      = 0.40
    v2_cfg.W_NARR     = 3.5
    v2_cfg.W_CROSS    = 1.5
    v2_cfg.W_SELF     = 1.2
    v2_cfg.BASE_PROB  = 0.04
    v2_cfg.PRIOR_MIX  = 0.35
    v2_cfg.PERSONA_MIX= 0.45
    v2_cfg.RECENT_WIN = 6.0

    res_base = run_cv(v2_cfg, ALL_Q, NARR, quarters, q_index, q_sortkey, topics_list, verbose=True)
    
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION RESULTS (V2 PRODUCTION ENGINE)")
    print("=" * 50)
    print(f"Overall F1 Score: {res_base['overall_f1']:.1%}")
    print(f"Overall Precision: {res_base['overall_precision']:.1%}")
    print(f"Overall Recall: {res_base['overall_recall']:.1%}")
    print(f"Total Call Turns: {res_base['total_calls']}")
    
    # Boost config
    boost_cfg = Config()
    boost_cfg.canonical = True
    boost_cfg.ordinal   = True
    boost_cfg.prior     = True
    boost_cfg.momentum  = True
    boost_cfg.repeat    = True
    boost_cfg.personas  = True
    boost_cfg.boost     = True
    boost_cfg.DECAY      = 0.40
    boost_cfg.W_NARR     = 3.5
    boost_cfg.W_CROSS    = 1.5
    boost_cfg.W_SELF     = 1.2
    boost_cfg.BASE_PROB  = 0.04
    boost_cfg.PRIOR_MIX  = 0.35
    boost_cfg.PERSONA_MIX= 0.45
    boost_cfg.RECENT_WIN = 6.0

    res_boost = run_cv(boost_cfg, ALL_Q, NARR, quarters, q_index, q_sortkey, topics_list, verbose=False)
    print(f"Overall F1 Score (with recall boost): {res_boost['overall_f1']:.1%}")
    print(f"Overall Recall (with recall boost): {res_boost['overall_recall']:.1%}")
    print("=" * 50)

    # Save to cross_validation.json
    output = {
        "quarters": len(res_base["per_quarter"]),
        "total_calls": res_base["total_calls"],
        "overall_f1_base": res_base["overall_f1"],
        "overall_precision_base": res_base["overall_precision"],
        "overall_recall_base": res_base["overall_recall"],
        "overall_f1_boost": res_boost["overall_f1"],
        "overall_recall_boost": res_boost["overall_recall"],
        "per_quarter_details": res_base["per_quarter"]
    }
    
    with open(CROSSVAL_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Cross-validation results saved to {CROSSVAL_PATH}")
