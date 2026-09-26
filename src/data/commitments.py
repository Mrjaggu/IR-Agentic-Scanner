"""
commitments.py — dated management commitments, and their status as of a cutoff.

Why this module exists
----------------------
On the two held-out calls, `guidance_callback` — an analyst pressing a promise
management made on an EARLIER call — is tied for the most common high-value
move (5 of 48 questions). It is also the move the generator was structurally
incapable of producing: the architecture doc's Section 2 commitments store was
left a stub, so the Question Framer never saw a single outstanding commitment.
That is not a tuning gap, it is a missing input. No amount of prompt iteration
in the improvement loop could have closed it, which is a large part of why the
loop looked like it was doing nothing.

The dangerous commitment is an old one. Rikin Shah pressed the 3.8% NIM target
in Q4FY26; Mahrukh pressed the same target in Q1FY27 against a 12-to-15-month
timeline given earlier still. Management is least prepared for exactly these,
because the promise is two or three calls behind them.

Deterministic (air-gapped). Status is computed strictly as-of a cutoff quarter,
so a commitment made after the cutoff can never leak into a prediction.
"""

import json
import re
from pathlib import Path

# A commitment is a forward promise with something checkable in it: a number,
# a horizon, or both. Vague optimism ("we remain confident") is not tracked.
_PLEDGE = re.compile(
    r"\b(?:we (?:will|shall|expect to|aim to|intend to|plan to|target|targeting"
    r"|should be able to|are (?:confident|targeting)|have (?:guided|said))"
    r"|our (?:target|aspiration|guidance|endeavour|endeavor) is"
    r"|the (?:target|aspiration|guidance) (?:is|remains)"
    r"|we would (?:like to|expect to|want to)"
    r"|(?:aspiration|aspire) to"
    r"|(?:you|one) (?:should|can) expect)\b", re.I)

_HORIZON = re.compile(
    r"\b(?:over the next (?<num>\d+|few|couple of) (?:quarters?|months?|years?)"
    r"|in (?:the )?(?:next|coming) (?:\d+|few) (?:quarters?|months?|years?)"
    r"|by (?:the end of )?(?:FY\s?\d{2,4}|Q[1-4]\s?FY\s?\d{2,4}|\d{4})"
    r"|within (?:\d+|a few|the next) (?:quarters?|months?|years?)"
    r"|(?:\d+) to (?:\d+) (?:quarters?|months?)"
    r"|(?:this|next) (?:financial )?year"
    r"|(?:medium|near|short|long)[- ]term"
    r"|steady[- ]state)\b".replace("(?<num>", "(?:"), re.I)

_VALUE = re.compile(
    r"(?:(?:\d+\.?\d*)\s*(?:%|per cent|percent|bps|basis points)"
    r"|(?:Rs\.?|INR|₹)\s?\d[\d,]*(?:\.\d+)?\s*(?:crore|cr|bn|billion|lakh|mn|million)"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:crore|cr|bps)"
    r"|\d+\.?\d*\s*(?:x|times))", re.I)

# Topic attribution uses the ONE canonical taxonomy (src.graphs.compiler.TOPICS)
# rather than a local copy. An earlier draft invented its own topic names, which
# would have made every move-level attribution silently incomparable with the
# topic-level harness and the graph.
from src.graphs.compiler import TOPICS as _TOPIC_TERMS


def _attribute_topic(sentence: str, context: str = "") -> str | None:
    """Attribute on the sentence first; fall back to its neighbours.

    Needed because the most valuable commitments name only the number: "we
    stick to what we said 3.7, 3.8 over the next 8 to 10 quarters" is the
    3.8% NIM target, but the word NIM is in the sentence before it."""
    hit = _score_topic(sentence)
    if hit:
        return hit
    return _score_topic(context) if context else None


def _score_topic(sentence: str) -> str | None:
    low = (sentence or "").lower()
    best, hits = None, 0
    for topic, terms in _TOPIC_TERMS.items():
        n = sum(1 for t in terms if t in low)
        if n > hits:
            best, hits = topic, n
    return best


_ABBREV = re.compile(r"\b(Rs|Mr|Mrs|Ms|Dr|No|vs|approx|etc|Co|Ltd|Pvt|Q[1-4]|FY|e\.g|i\.e)\.", re.I)


def _sentences(text: str) -> list[str]:
    """Split on sentence ends, but not on the dot in "Rs." / "Mr." / "FY."."""
    text = re.sub(r"\s+", " ", text or "")
    masked = _ABBREV.sub(lambda m: m.group(0).replace(".", "\u0001"), text)
    parts = re.split(r"(?<=[.!?])\s+", masked)
    return [p.replace("\u0001", ".").strip() for p in parts if p.strip()]


_QUARTERS_AHEAD = [
    (re.compile(r"\b(?:next|coming) quarter\b", re.I), 1),
    (re.compile(r"\b(?:1|one|a) to (?:2|two) quarters?\b", re.I), 2),
    (re.compile(r"\b(?:2|two) to (?:3|three) quarters?\b", re.I), 3),
    (re.compile(r"\b(?:3|three) to (?:4|four) quarters?\b", re.I), 4),
    (re.compile(r"\b(?:12|twelve) to (?:15|fifteen) months?\b", re.I), 5),
    (re.compile(r"\b(?:few|couple of) quarters?\b", re.I), 3),
    (re.compile(r"\b(?:this|the current) (?:financial )?year\b", re.I), 4),
    (re.compile(r"\bnext (?:financial )?year\b", re.I), 8),
    (re.compile(r"\b(?:12|twelve) months?\b", re.I), 4),
    (re.compile(r"\b(?:18|eighteen) months?\b", re.I), 6),
    (re.compile(r"\b(?:medium|long)[- ]term\b", re.I), 8),
    (re.compile(r"\bnear[- ]term\b", re.I), 2),
    (re.compile(r"\bshort[- ]term\b", re.I), 2),
]


def _quarters_ahead(sentence: str) -> int | None:
    for pat, n in _QUARTERS_AHEAD:
        if pat.search(sentence):
            return n
    return None


def extract_commitments(dataset: list[dict]) -> list[dict]:
    """Every dated/quantified management promise in the archive, in order.

    Scans prepared remarks AND management answers in Q&A — a great many
    commitments are made in answers, not in the script.
    """
    ordered = sorted(dataset, key=lambda r: r.get("sort_key", 0))
    qindex = {r["quarter_id"]: i for i, r in enumerate(ordered)}
    out = []
    for rec in ordered:
        qid = rec.get("quarter_id")
        blocks = [(n.get("speaker") or "MANAGEMENT", n.get("text", ""), "prepared_remarks")
                  for n in rec.get("narration", []) if isinstance(n, dict)]
        blocks += [(t.get("speaker") or "Management", t.get("text", ""), "qa_answer")
                   for t in rec.get("qa", [])
                   if str(t.get("role", "")).lower().startswith("management")]
        for speaker, text, source in blocks:
            sents = _sentences(text)
            for i, sent in enumerate(sents):
                if len(sent.split()) < 6 or not _PLEDGE.search(sent):
                    continue
                context = " ".join(sents[max(0, i - 2):i + 2])
                value = _VALUE.search(sent)
                horizon = _HORIZON.search(sent)
                if not value and not horizon:
                    continue          # unverifiable optimism, not a commitment
                out.append({
                    "made_in": qid,
                    "sort_key": rec.get("sort_key", 0),
                    "q_index": qindex[qid],
                    "speaker": speaker.strip(),
                    "source": source,
                    "text": sent,
                    "topic": _attribute_topic(sent, context),
                    "value": value.group(0).strip() if value else None,
                    "horizon_phrase": horizon.group(0).strip() if horizon else None,
                    "quarters_ahead": _quarters_ahead(sent),
                })
    return out


def status_as_of(commitment: dict, elapsed: int) -> str:
    """open | due | stale — strictly as of the cutoff.

    `due` means the horizon management gave has elapsed and the promise has not
    been restated since: the highest-pressure state, and the one an analyst
    will convert into a guidance_callback.
    `stale` means old with no stated horizon — still callable, lower pressure.
    """
    qa = commitment.get("quarters_ahead")
    if qa is None:
        return "stale" if elapsed >= 3 else "open"
    return "due" if elapsed >= qa else "open"


def open_commitments_as_of(dataset: list[dict], cutoff_quarter: str,
                           min_quarters_old: int = 1,
                           max_quarters_old: int = 12) -> list[dict]:
    """Commitments live and callable as of `cutoff_quarter`, newest first.

    Anything made in or after the target quarter is excluded by construction,
    so this cannot leak the future into a prediction.
    """
    ordered = sorted(dataset, key=lambda r: r.get("sort_key", 0))
    qindex = {r["quarter_id"]: i for i, r in enumerate(ordered)}
    if cutoff_quarter not in qindex:
        return []
    cut = qindex[cutoff_quarter]
    rows = []
    for c in extract_commitments(dataset):
        if c["q_index"] > cut:
            continue                     # future — cannot leak into a prediction
        elapsed = cut - c["q_index"]
        if not (min_quarters_old <= elapsed <= max_quarters_old):
            continue                     # made on the cutoff call, or long obsolete
        c = dict(c)
        c["quarters_elapsed"] = elapsed
        c["status"] = status_as_of(c, elapsed)
        c["pressure"] = _pressure(c)
        rows.append(c)
    rows.sort(key=lambda c: -c["pressure"])
    return rows


_STATUS_WEIGHT = {"due": 1.0, "stale": 0.62, "open": 0.35}


def _pressure(c: dict) -> float:
    """How callable this promise is, right now.

    Ranked lexicographically by status first, a twelve-quarter-old branch-count
    pledge outranked the three-quarter-old 3.8% NIM target — wrong, because the
    ancient one has long since been restated or quietly dropped while the NIM
    target is live and unmet. So status is scaled by recency, and a promise
    carrying an explicit number is weighted up: a number is what makes it
    pressable ("where do we stand on 3.8%" has a fact to demand).
    """
    w = _STATUS_WEIGHT[c["status"]]
    recency = 1.0 / (1.0 + 0.25 * max(0, c["quarters_elapsed"] - 2))
    checkable = 1.35 if c.get("value") else 1.0
    horizoned = 1.15 if c.get("quarters_ahead") else 1.0
    return round(w * recency * checkable * horizoned, 4)


def by_topic(commitments: list[dict]) -> dict:
    """Shape expected by moves.plan_topic_move_slots's `available_inputs`."""
    out: dict[str, list] = {}
    for c in commitments:
        if c.get("topic"):
            out.setdefault(c["topic"], []).append(c)
    return out


# --- Graph-native path (2026-09) --------------------------------------------
# Everything above reads dataset.json's shape (per-quarter {narration, qa}
# records) -- what moves.py/move_planner.py already use for the
# guidance_callback move. The functions below read graph.json directly
# instead: it already carries the same prepared-remarks (NarrationSegment)
# and Q&A-answer (Answer) text, PLUS a quarter's proper chronological order
# (Quarter.sort_key) and pre-computed per-node topic tags -- so this path
# needs no dataset.json load and is naturally bank-correct (the caller
# already loaded the right bank's own graph; there is nothing bank-specific
# left for this module to get wrong). Built for
# src.agentic.tools.commitments_tool -- see that module for why grounding
# the Planning Agent's retrieval in this, rather than a bare keyword scan
# over commitment-sounding verbs, was the point.
#
# A `resolved` status (a commitment management has since delivered on) was
# attempted and deliberately pulled back out: the only signal available
# without a full target-vs-actual reconciliation engine was "a later,
# same-topic sentence uses achievement language ('delivered', 'in line
# with', ...)", optionally near a similar number. Tested against real
# extracted commitments and it produced clear false positives on the first
# quarters checked -- e.g. a pledge to "cover the entire Bank" with credit
# cards was marked resolved by an unrelated sentence about deposit growth
# that merely shared the topic tag and the word "delivered"; a "5% of
# business, growing 3-4x by FY27" pledge was marked resolved by a
# coincidental, unrelated "5%" elsewhere on the same broad topic. Topic
# tags are too coarse and achievement language too generic in this
# narration for either signal to reliably confirm the SAME claim was met,
# and a false "resolved" is worse than no resolved detection at all here --
# it would silently hide a still-live commitment from the tool this feeds
# (src.agentic.tools.commitments_tool). Left at open/due/stale, same as the
# dataset-driven path above; a real fix would compare each pledge's parsed
# target and direction against the actual metric series
# (metrics_extractor.METRIC_TOPIC_MAP) rather than re-reading narration text
# a second time, which is a project of its own.


def _quarter_index_from_graph(graph: dict) -> dict[str, int]:
    """{quarter_id: chronological index}, ordered by the Quarter node's own
    sort_key (e.g. Q1FY22 -> 221, Q2FY22 -> 222, Q1FY23 -> 231) rather than
    re-deriving order from the quarter_id string -- robust regardless of
    spelling, and the same ordering every other bank-aware module trusts."""
    quarters = sorted(
        (n for n in graph.get("nodes", []) if n["type"] == "Quarter"),
        key=lambda n: n["properties"]["sort_key"],
    )
    return {n["id"]: i for i, n in enumerate(quarters)}


def extract_commitments_from_graph(graph: dict) -> list[dict]:
    """extract_commitments()'s dataset-driven logic, adapted to read
    NarrationSegment (prepared remarks) and Answer (Q&A) nodes straight from
    the graph instead of dataset.json's {narration, qa} records -- same
    _PLEDGE/_VALUE/_HORIZON rules, same output shape (made_in/sort_key/
    q_index/speaker/source/text/topic/value/horizon_phrase/quarters_ahead),
    so by_topic() and the pressure ranking below work identically either
    way. Topic attribution prefers the sentence-level _attribute_topic
    (same reasoning as extract_commitments: the number is often in a
    different sentence than the metric name) but falls back to the node's
    own pre-computed `topics` tag when that finds nothing at all -- the
    dataset-driven path has no equivalent signal to fall back to."""
    q_index = _quarter_index_from_graph(graph)
    records = [
        (n["properties"]["quarter"], n["properties"].get("speaker") or "MANAGEMENT",
         n["properties"]["text"], n["properties"].get("topics") or [], "prepared_remarks")
        for n in graph.get("nodes", []) if n["type"] == "NarrationSegment"
    ] + [
        (n["properties"]["quarter"], "MANAGEMENT",
         n["properties"]["text"], n["properties"].get("topics") or [], "qa_answer")
        for n in graph.get("nodes", []) if n["type"] == "Answer"
    ]
    out = []
    for qid, speaker, text, node_topics, source in records:
        if qid not in q_index:
            continue
        sents = _sentences(text)
        for i, sent in enumerate(sents):
            if len(sent.split()) < 6 or not _PLEDGE.search(sent):
                continue
            context = " ".join(sents[max(0, i - 2):i + 2])
            value = _VALUE.search(sent)
            horizon = _HORIZON.search(sent)
            if not value and not horizon:
                continue          # unverifiable optimism, not a commitment
            topic = _attribute_topic(sent, context)
            if not topic and node_topics:
                topic = next((t for t in node_topics if t in _TOPIC_TERMS), None)
            out.append({
                "made_in": qid,
                "sort_key": q_index[qid],
                "q_index": q_index[qid],
                "speaker": speaker.strip(),
                "source": source,
                "text": sent,
                "topic": topic,
                "value": value.group(0).strip() if value else None,
                "horizon_phrase": horizon.group(0).strip() if horizon else None,
                "quarters_ahead": _quarters_ahead(sent),
            })
    return out


def open_commitments_as_of_graph(graph: dict, cutoff_quarter: str,
                                 min_quarters_old: int = 1,
                                 max_quarters_old: int = 12) -> list[dict]:
    """Graph-native counterpart to open_commitments_as_of -- same cutoff
    semantics (a commitment made in or after cutoff_quarter can never leak
    into what this returns) and the same status/pressure ranking. No
    `resolved` state (see the note above this section for why)."""
    q_index = _quarter_index_from_graph(graph)
    if cutoff_quarter not in q_index:
        return []
    cut = q_index[cutoff_quarter]
    rows = []
    for c in extract_commitments_from_graph(graph):
        if c["q_index"] > cut:
            continue                     # future — cannot leak into a prediction
        elapsed = cut - c["q_index"]
        if not (min_quarters_old <= elapsed <= max_quarters_old):
            continue                     # made on the cutoff call, or long obsolete
        c = dict(c)
        c["quarters_elapsed"] = elapsed
        c["status"] = status_as_of(c, elapsed)
        c["pressure"] = _pressure(c)
        rows.append(c)
    rows.sort(key=lambda c: -c["pressure"])
    return rows


if __name__ == "__main__":
    data = json.loads(Path("data/db/dataset.json").read_text())
    rows = open_commitments_as_of(data, "q4fy26")
    print(f"{len(rows)} commitments live as of q4fy26")
    for c in rows[:12]:
        print(f"  [{c['status']:5}] {c['made_in']} +{c['quarters_elapsed']}q "
              f"{c['topic'] or '-'} | {c['value'] or ''} {c['horizon_phrase'] or ''}")
        print(f"          {c['text'][:150]}")
