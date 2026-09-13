import math
import os
import json
from src.config.settings import (
    PERSONAS_PATH,
    TOPICS_LIST,
    EngineConfig,
)

# Load personas
def load_personas(personas_path=PERSONAS_PATH):
    if os.path.exists(personas_path):
        with open(personas_path) as f:
            return {p["analyst"]: p for p in json.load(f)["personas"]}
    return {}

# Cache personas globally for easy import
PERSONAS = load_personas()

def get_global_rate(graph, train_quarters, topics_list=TOPICS_LIST):
    """Calculates global base rates of topics across training quarters."""
    global_count = {}
    for q in [n for n in graph["nodes"]
              if n["type"] == "Question" and n["properties"]["quarter"] in train_quarters]:
        for t in q["properties"]["topics"]:
            if t != "General":
                global_count[t] = global_count.get(t, 0) + 1
    gtot = sum(global_count.values()) or 1
    return {t: global_count.get(t, 0) / gtot for t in topics_list if t != "General"}

def get_analyst_profile(analyst, graph, train_quarters, val_ord, q_ord_map, global_rate, config=EngineConfig, personas=PERSONAS, topics_list=TOPICS_LIST):
    """
    Returns (recency_weighted_topic_pref, predicted_N).
    
    N is computed as round((max_topics_in_recent_call + avg_topics_per_call) / 2)
    clamped to [1, 5].
    """
    qs = [n for n in graph["nodes"]
          if n["type"] == "Question"
          and n["properties"]["analyst"] == analyst
          and n["properties"]["quarter"] in train_quarters]

    if not qs:
        # zero history → persona prior if available, else global base-rate
        if analyst in personas and personas[analyst].get("topic_prior"):
            pp = personas[analyst]["topic_prior"]
            s = sum(pp.values()) or 1
            return {t: pp.get(t, 0.0) / s for t in topics_list if t != "General"}, 2
        return dict(global_rate), 2

    tw = {}
    total_w = 0.0
    all_topic_counts = []
    recent_topic_counts = []   # calls within RECENT_WIN quarters

    for q in qs:
        qa = val_ord - 1 - q_ord_map[q["properties"]["quarter"]]   # ordinal quarters ago
        w  = math.exp(-config.DECAY * qa)
        total_w += w
        tc = [t for t in q["properties"]["topics"] if t != "General"]
        all_topic_counts.append(len(tc))
        if qa <= config.RECENT_WIN:
            recent_topic_counts.append(len(tc))
        for t in tc:
            tw[t] = tw.get(t, 0.0) + w

    pref = {t: v / total_w for t, v in tw.items()} if total_w > 0 else {}

    # ── Smoothing: blend with global base-rate (stronger for sparse analysts) ─
    n_hist_quarters = len({q["properties"]["quarter"] for q in qs})
    mix = config.PRIOR_MIX if n_hist_quarters >= 3 else min(0.6, config.PRIOR_MIX * 2)
    pref = {t: (1 - mix) * pref.get(t, 0.0) + mix * global_rate.get(t, 0.0)
            for t in topics_list if t != "General"}

    # ── Persona prior blend (web-derived; weight shrinks as history grows) ────
    if analyst in personas and personas[analyst].get("topic_prior"):
        pp = personas[analyst]["topic_prior"]
        s = sum(pp.values()) or 1
        pw = config.PERSONA_MIX / (1 + 0.35 * max(0, n_hist_quarters - 2))
        pref = {t: (1 - pw) * pref.get(t, 0.0) + pw * (pp.get(t, 0.0) / s)
                for t in pref}

    avg_all = sum(all_topic_counts) / len(all_topic_counts) if all_topic_counts else 2.0
    max_recent = max(recent_topic_counts) if recent_topic_counts else round(avg_all)

    N = max(1, min(5, round((max_recent + avg_all) / 2.0)))
    return pref, N

def get_repeat_propensity(analyst, graph, train_quarters, q_ord_map):
    """P(topic asked in quarter Q | analyst asked it in Q-1), from history."""
    qs = [n for n in graph["nodes"]
          if n["type"] == "Question"
          and n["properties"]["analyst"] == analyst
          and n["properties"]["quarter"] in train_quarters]
    by_q = {}
    for q in qs:
        by_q.setdefault(q["properties"]["quarter"], set()).update(
            t for t in q["properties"]["topics"] if t != "General")
    keys = sorted(by_q, key=lambda x: q_ord_map[x])
    num = den = 0
    for k1, k2 in zip(keys, keys[1:]):
        if q_ord_map[k2] - q_ord_map[k1] == 1:
            den += len(by_q[k1])
            num += len(by_q[k1] & by_q[k2])
    return num / den if den >= 3 else 0.5
