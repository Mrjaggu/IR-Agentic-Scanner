"""
dossier.py — full question dossiers for the Analyst Profile view.

The graph's Question nodes concatenate each analyst's turns into a single
block, which loses the actual conversation. dataset.json's `qa` array keeps
the real turn-by-turn thread (speaker, role, analyst attribution), so
dossiers are built from there instead: the analyst's full question, each
NAMED management responder and what they actually said, and the analyst's own
follow-up reaction to that answer.

Every dossier also carries a `why` block — an evidence-based rationale for
why this analyst asked this, assembled from data already in the archive
rather than guessed by a model:
  intent      — the persona_consistent / narration_triggered / unexplained
                classification and its evidence sentence (question_intent.json)
  follows_up  — their own prior question this one follows up on
                (FOLLOWS_UP_ON / CROSS_FOLLOWS_PREV_CALL graph edges)
  narration   — what management disclosed in prepared remarks that quarter on
                the same topic (i.e. the thing they were reacting to)
  metrics     — the disclosed metric move behind the topic, plus how unusual
                that move was against the metric's own history
  recurrence  — how often this analyst raises this topic vs. everyone else

Nothing here is truncated: the UI gets full text and decides how to present it.
"""

import json
import os
from functools import lru_cache

from src.config.settings import METRICS_TIMESERIES_PATH, QUESTION_INTENT_PATH
from src.signals.metrics_extractor import METRIC_TOPIC_MAP, compute_topic_anomaly_scores

_TOPIC_TO_METRICS: dict[str, list[str]] = {}
for _m, _t in METRIC_TOPIC_MAP.items():
    _TOPIC_TO_METRICS.setdefault(_t, []).append(_m)


def _load_intent_records() -> dict[tuple[str, str], dict]:
    """(quarter, analyst) -> intent record. These carry a human-written
    `evidence` sentence, which is the single best 'why' signal we have."""
    if not os.path.exists(QUESTION_INTENT_PATH):
        return {}
    with open(QUESTION_INTENT_PATH) as f:
        records = json.load(f)
    return {(r["quarter"], r["analyst"]): r for r in records}


def _load_metrics_timeseries() -> dict:
    if not os.path.exists(METRICS_TIMESERIES_PATH):
        return {}
    with open(METRICS_TIMESERIES_PATH) as f:
        return json.load(f)


def _build_precedent_index(graph: dict) -> dict[str, list[dict]]:
    """question_id -> the prior questions it follows up on, resolved to
    {quarter, analyst, text}. Uses the FOLLOWS_UP_ON and
    CROSS_FOLLOWS_PREV_CALL edges the graph compiler already computes."""
    q_by_id = {n["id"]: n["properties"] for n in graph["nodes"] if n["type"] == "Question"}
    index: dict[str, list[dict]] = {}
    for e in graph["edges"]:
        if e["type"] not in ("FOLLOWS_UP_ON", "CROSS_FOLLOWS_PREV_CALL"):
            continue
        target = q_by_id.get(e["target"])
        if not target:
            continue
        index.setdefault(e["source"], []).append({
            "quarter": target["quarter"],
            "analyst": target["analyst"],
            "text": target["text"],
            "topics": [t for t in target["topics"] if t != "General"],
            "relation": e["type"],
        })
    return index


def _questions_for(graph: dict, quarter: str, analyst: str) -> list[dict]:
    return [
        {"id": n["id"], **n["properties"]}
        for n in graph["nodes"]
        if n["type"] == "Question"
        and n["properties"]["quarter"] == quarter
        and n["properties"]["analyst"] == analyst
    ]


def _narration_on_topics(dataset_quarter: dict, graph: dict, quarter: str,
                         topics: list[str]) -> list[dict]:
    """The prepared-remarks segments that quarter tagged with the same topics —
    what management said that the analyst may have been reacting to."""
    out = []
    for n in graph["nodes"]:
        if n["type"] != "NarrationSegment" or n["properties"]["quarter"] != quarter:
            continue
        seg_topics = [t for t in n["properties"]["topics"] if t != "General"]
        overlap = set(seg_topics) & set(topics)
        if not overlap:
            continue
        out.append({
            "speaker": n["properties"].get("speaker") or "Management",
            "text": n["properties"]["text"],
            "topics": sorted(overlap),
        })
    return out


def _metric_context(quarter: str, topics: list[str], quarter_order: list[str],
                    timeseries: dict) -> list[dict]:
    """The disclosed metric moves behind these topics, with how unusual each
    move was relative to that metric's own history (percentile, 1.0 = biggest
    move ever). Grounds 'why this came up' in an actual number."""
    q_metrics = timeseries.get(quarter, {})
    try:
        anomaly = _anomaly_for_quarter(quarter, tuple(quarter_order))
    except Exception:
        anomaly = {}
    out = []
    for topic in topics:
        for metric in _TOPIC_TO_METRICS.get(topic, []):
            rec = q_metrics.get(metric)
            if not rec:
                continue
            out.append({
                "topic": topic,
                "metric": metric,
                "value": rec.get("value"),
                "value_unit": rec.get("value_unit"),
                "delta": rec.get("delta"),
                "delta_unit": rec.get("delta_unit"),
                "direction": rec.get("direction"),
                "period": rec.get("period"),
                "anomaly_percentile": round(anomaly.get(topic), 3) if anomaly.get(topic) is not None else None,
            })
    return out


@lru_cache(maxsize=64)
def _anomaly_for_quarter(quarter: str, quarter_order: tuple) -> dict:
    return compute_topic_anomaly_scores(quarter, list(quarter_order))


def _recurrence(graph: dict, analyst: str, topics: list[str],
                up_to_quarter: str, quarter_order: list[str]) -> list[dict]:
    """How much of a recurring theme each topic is for THIS analyst, against
    how often it comes up for everyone — the persona signal, stated plainly."""
    idx = quarter_order.index(up_to_quarter) if up_to_quarter in quarter_order else len(quarter_order)
    window = set(quarter_order[:idx + 1])

    analyst_quarters, analyst_topic_quarters = set(), {}
    all_quarters, all_topic_hits, all_hits = set(), {}, 0
    for n in graph["nodes"]:
        if n["type"] != "Question":
            continue
        p = n["properties"]
        if p["quarter"] not in window:
            continue
        tl = [t for t in p["topics"] if t != "General"]
        all_quarters.add(p["quarter"])
        all_hits += len(tl)
        for t in tl:
            all_topic_hits[t] = all_topic_hits.get(t, 0) + 1
        if p["analyst"] == analyst:
            analyst_quarters.add(p["quarter"])
            for t in tl:
                analyst_topic_quarters.setdefault(t, set()).add(p["quarter"])

    n_calls = len(analyst_quarters) or 1
    out = []
    for topic in topics:
        asked_in = len(analyst_topic_quarters.get(topic, set()))
        own_rate = asked_in / n_calls
        global_rate = (all_topic_hits.get(topic, 0) / all_hits) if all_hits else 0.0
        out.append({
            "topic": topic,
            "asked_in_calls": asked_in,
            "of_calls_attended": n_calls,
            "own_rate": round(own_rate, 3),
            "global_rate": round(global_rate, 4),
            "vs_global": round(own_rate / global_rate, 2) if global_rate else None,
        })
    return out


def _thread_for_exchange(turns: list[dict]) -> list[dict]:
    """Turn the raw qa turns of one exchange into a clean thread. Moderator
    turns are dropped (they're just 'next question from the line of X')."""
    thread = []
    for t in turns:
        role = (t.get("role") or "").lower()
        if role == "moderator":
            continue
        thread.append({
            "role": "analyst" if role == "analyst" else "management",
            "speaker": t.get("speaker") or ("Analyst" if role == "analyst" else "Management"),
            "text": t.get("text") or "",
        })
    return thread


def _exchanges_for_analyst(dataset_quarter: dict, analyst: str) -> list[list[dict]]:
    """Contiguous runs of qa turns attributed to this analyst. Each run is one
    exchange: their question, management's reply, their reaction, and so on."""
    runs, current = [], []
    for turn in dataset_quarter.get("qa", []):
        if turn.get("analyst_name") == analyst:
            current.append(turn)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def build_analyst_dossier(analyst: str, dataset: list, graph: dict) -> dict:
    """Full dossier for one analyst: every exchange they had, threaded, with
    management responders named and an evidence-based rationale attached."""
    quarter_order = [q["quarter_id"] for q in dataset]
    intents = _load_intent_records()
    timeseries = _load_metrics_timeseries()
    precedents = _build_precedent_index(graph)

    exchanges = []
    for dq in dataset:
        quarter = dq["quarter_id"]
        runs = _exchanges_for_analyst(dq, analyst)
        if not runs:
            continue

        q_nodes = _questions_for(graph, quarter, analyst)
        topics = sorted({t for qn in q_nodes for t in qn["topics"] if t != "General"})
        node_ids = [qn["id"] for qn in q_nodes]

        follows_up = []
        for nid in node_ids:
            follows_up.extend(precedents.get(nid, []))
        # their OWN prior questions are the interesting precedent; keep others
        # separately so the UI can distinguish "they came back to this" from
        # "this was going around the street".
        own_follow_ups = [f for f in follows_up if f["analyst"] == analyst]
        peer_follow_ups = [f for f in follows_up if f["analyst"] != analyst]

        intent = intents.get((quarter, analyst))
        why = {
            "intent": intent["intent"] if intent else None,
            "intent_evidence": intent.get("evidence") if intent else None,
            "follows_up_on": own_follow_ups[:3],
            "peers_asked_before": peer_follow_ups[:3],
            "narration_trigger": _narration_on_topics(dq, graph, quarter, topics),
            "metric_context": _metric_context(quarter, topics, quarter_order, timeseries),
            "recurrence": _recurrence(graph, analyst, topics, quarter, quarter_order),
        }

        for run in runs:
            thread = _thread_for_exchange(run)
            if not thread:
                continue
            responders = []
            for msg in thread:
                if msg["role"] == "management" and msg["speaker"] not in responders:
                    responders.append(msg["speaker"])
            analyst_turns = [m for m in thread if m["role"] == "analyst"]
            exchanges.append({
                "quarter": quarter,
                "call_date": dq.get("call_date"),
                "topics": topics,
                "thread": thread,
                "responders": responders,
                "n_analyst_turns": len(analyst_turns),
                "had_followup_reaction": len(analyst_turns) > 1,
                "why": why,
            })

    exchanges.sort(key=lambda e: quarter_order.index(e["quarter"]), reverse=True)
    return {
        "analyst": analyst,
        "n_exchanges": len(exchanges),
        "quarters_covered": sorted({e["quarter"] for e in exchanges},
                                   key=lambda q: quarter_order.index(q)),
        "exchanges": exchanges,
    }
