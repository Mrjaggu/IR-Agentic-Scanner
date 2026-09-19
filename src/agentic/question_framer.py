"""
question_framer.py — Section 3.4.

Converts (ranked topic, analyst style profile) into phrased, attributed
question text. "Never invents; every generated question must ground to a real
number or historical pattern."

What the model is allowed to see, and what it must never say, are two
different things — an earlier version conflated them and it showed. It was
handed the ranker's own features (base rate, anomaly percentile) and told to
cite a number, so it dutifully produced lines like "given the historical base
rate of 17% for deposits & CASA … and this quarter's anomaly score of 1.00".
No analyst has ever said that on a call. Those features decide WHICH topic
gets asked about; they have no business in the sentence.

So the prompt now carries only facts that exist in the world — the analyst's
own prior wording, and the figures management is about to disclose — and the
Verifier rejects any draft that leaks internal scoring vocabulary.
"""

import json
import re

from src.signals.metrics_extractor import METRIC_TOPIC_MAP
from src.agentic import pattern_retrieval

_TOPIC_METRICS: dict[str, list[str]] = {}
for _m, _t in METRIC_TOPIC_MAP.items():
    _TOPIC_METRICS.setdefault(_t, []).append(_m)


def period_label(quarter: str) -> str:
    """q1fy27 -> 'Q1 FY27' — how an analyst would actually say it."""
    m = re.match(r"q([1-4])fy(\d{2})", (quarter or "").lower())
    return f"Q{m.group(1)} FY{m.group(2)}" if m else (quarter or "").upper()


def _find_precedent(analyst: str, topic: str, graph: dict, prior_quarters: set[str]) -> dict | None:
    candidates = [
        n["properties"] for n in graph["nodes"]
        if n["type"] == "Question"
        and n["properties"]["analyst"] == analyst
        and n["properties"]["quarter"] in prior_quarters
        and topic in n["properties"]["topics"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p["quarter"], reverse=True)
    top = candidates[0]
    return {"quarter": top["quarter"], "text": top["text"][:280]}


def _metrics_for_topic(topic: str, disclosed: dict | None) -> list[dict]:
    """The figures management is disclosing this quarter for this topic. These
    are real, checkable facts, so they are the only numbers a question may
    quote besides the analyst's own prior words."""
    if not disclosed:
        return []
    out = []
    for metric in _TOPIC_METRICS.get(topic, []):
        rec = disclosed.get(metric)
        if rec:
            out.append({"metric": metric, **{k: rec.get(k) for k in
                        ("value", "value_unit", "delta", "delta_unit", "direction", "period")}})
    return out


def _narration_for_topic(graph: dict, target_quarter: str, topic: str,
                         disclosure_text: str = "", max_snips: int = 3) -> list[str]:
    """What management actually SAYS about this topic on the call being
    prepared for — from an uploaded draft script if there is one, else the
    prepared remarks for that quarter.

    The extracted metric tuples cover eleven canonical metrics; management's
    own sentences cover everything else (a structural NIM target, a guidance
    range, a one-off charge). Showing them is what lets a question sound like
    it was written by someone who read the script, and it keeps the set of
    figures the model SEES identical to the set the Verifier will ACCEPT.
    Section 2.1 lists the draft disclosure script as a legitimate pre-call
    source; analyst QUESTIONS from the target quarter are never used and
    remain the leak boundary.

    A qualitative sentence with no number yet (e.g. "FCNR deposits are
    attracting strong interest, a meaningful opportunity") is exactly the
    kind of disclosure that provokes a real analyst's "how much?" follow-up
    -- so it is no longer hard-excluded for lacking a digit (2026-09 fix,
    found via a live cross-check against Kunal Shah's real Q1FY27 FCNR
    question: the sentence was IN the narration but a digit requirement was
    silently dropping it from the evidence pool, so the framed question
    talked about the repo-cut instead). Number-bearing sentences are still
    ranked ahead of qualitative ones when both are on-topic, since the
    Verifier needs concrete figures to ground a claim against -- this just
    stops qualitative-but-relevant sentences from being invisible."""
    from src.graphs.compiler import TOPICS as TOPIC_KEYWORDS
    kws = [k.lower() for k in TOPIC_KEYWORDS.get(topic, [])]
    kws += [w for w in re.split(r"[^a-z]+", topic.lower()) if len(w) > 3]

    if disclosure_text:
        source = [disclosure_text]
    else:
        source = [n["properties"]["text"] for n in graph["nodes"]
                  if n["type"] == "NarrationSegment"
                  and n["properties"]["quarter"] == target_quarter
                  and topic in n["properties"]["topics"]]

    # Rank by how much the sentence is actually ABOUT this topic, not by where
    # it happens to sit — the opening boilerplate is tagged onto everything by
    # the keyword tagger and would otherwise win by being first.
    scored = []
    for seg in source:
        for sent in re.split(r"(?<=[.!?])\s+", seg):
            sent = sent.strip()
            if len(sent) < 40:
                continue
            low = sent.lower()
            hits = sum(1 for k in kws if k in low)
            if hits:
                has_digit = bool(re.search(r"\d", sent))
                # Rank by topic relevance first, then prefer number-bearing
                # sentences (the Verifier needs a figure to check a claim
                # against) -- but a qualitative, on-topic sentence with no
                # digit yet still gets a slot instead of being dropped.
                scored.append((hits, 0 if has_digit else 1, len(sent), sent))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [s for _, _, _, s in scored[:max_snips]]


def build_evidence_pool(analyst: str, topics: list[str], graph: dict, prior_quarters: set[str],
                        global_rate: dict, anomaly_scores: dict,
                        disclosed_metrics: dict | None = None,
                        target_quarter: str = "", disclosure_text: str = "",
                        bank_id: str = "axis") -> dict:
    """base_rate and anomaly_score stay in the pool for ranking and audit, but
    they are deliberately NOT shown to the model — see the module docstring."""
    pool = {}
    for t in topics:
        narration = (_narration_for_topic(graph, target_quarter, t, disclosure_text)
                     if target_quarter else [])
        cognitive_pattern = pattern_retrieval.retrieve_for_analyst(
            analyst, t, bank_id=bank_id, exclude_quarter=target_quarter)
        pool[t] = {
            "precedent": _find_precedent(analyst, t, graph, prior_quarters),
            "cognitive_pattern": cognitive_pattern,
            "cognitive_pattern_overall": ([] if cognitive_pattern else
                pattern_retrieval.retrieve_overall(t, exclude_analyst=analyst, bank_id=bank_id,
                                                    exclude_quarter=target_quarter)),
            "base_rate": global_rate.get(t, 0.0),
            "anomaly_score": anomaly_scores.get(t),
            "metrics": _metrics_for_topic(t, disclosed_metrics),
            "narration": narration,
            # Which of the narration snippets above mention something without
            # putting a number on it -- e.g. "FCNR deposits are attracting
            # strong interest" with no quantum yet. This is the evidence
            # shape behind a real, confirmed miss (Kunal Shah, Q1FY27: topic
            # recall scored it a hit, question recall scored it 0.5, because
            # the framed question reacted generally instead of asking for the
            # figure). Surfacing it lets _build_frame_prompt nudge toward the
            # quantify-it follow-up a real analyst tends to ask, without any
            # per-analyst customization -- this fires for whoever gets this
            # evidence, not a name-specific rule. (2026-09)
            "qualitative_gaps": [snip for snip in narration if not re.search(r"\d", snip)],
            # Exactly the figures shown for THIS topic — so what the model may
            # say and what the Verifier will accept are the same set.
            "disclosure_numbers": set(re.findall(r"\d+(?:\.\d+)?", " ".join(narration))),
        }
    return pool


def _fmt_metric(m: dict) -> str:
    val = f"{m['value']}{m.get('value_unit') or ''}" if m.get("value") is not None else "—"
    if m.get("delta") is not None:
        return (f"{m['metric']} at {val}, {m.get('direction') or 'change'} of "
                f"{m['delta']}{m.get('delta_unit') or ''} {m.get('period') or ''}".strip())
    return f"{m['metric']} at {val}"


def _build_frame_prompt(analyst: str, style_note: str, topics: list[str], pool: dict,
                        target_quarter: str, bank_name: str = "Axis Bank") -> str:
    period = period_label(target_quarter)
    blocks = []
    for t in topics:
        ev = pool[t]
        lines = [f"### {t}"]
        if ev["precedent"]:
            lines.append(f'- How they put it in {period_label(ev["precedent"]["quarter"])}, in their own words: '
                         f'"{ev["precedent"]["text"]}"')
        cp = ev.get("cognitive_pattern")
        if cp:
            follow_up = (cp.get("typical_follow_up") or "none noted").rstrip(". ")
            lines.append(f'- This analyst has a recurring reasoning pattern here: {cp["trigger"]} '
                         f'-> {cp["reasoning_pattern"]} They tend to phrase it like: '
                         f'"{cp["question_style"]}" Typical follow-up: {follow_up}. '
                         f'Apply this reasoning shape to the ACTUAL figures below -- do not just '
                         f'describe the topic.')
        elif ev.get("cognitive_pattern_overall"):
            other = ev["cognitive_pattern_overall"][0]
            lines.append(f'- No pattern on record for this analyst on this topic, but other analysts '
                         f'covering this bank tend to reason this way here ({other["analyst"]}): '
                         f'{other["trigger"]} -> {other["reasoning_pattern"]} Treat this as a weak hint '
                         f'only, not this analyst\'s own established behavior.')
        for m in ev["metrics"]:
            lines.append(f"- Management is disclosing this quarter: {_fmt_metric(m)}")
        for snip in ev.get("narration", []) or []:
            lines.append(f'- Management says on this call: "{snip[:300]}"')
        if ev.get("qualitative_gaps"):
            lines.append("- The line(s) above with no number attached are exactly the kind of "
                         "thing a sharp analyst presses on next -- don't just react to the "
                         "general subject; ask management to put a figure on it (how much, "
                         "what proportion, what run-rate) rather than settling for the "
                         "qualitative description they gave.")
        if not ev["precedent"] and not ev["metrics"] and not ev.get("narration"):
            lines.append("- No prior question from them on this and no disclosed figure — "
                         "keep it a plain, open question with no numbers at all.")
        blocks.append("\n".join(lines))

    return f"""You are drafting the questions {analyst} is most likely to ask on {bank_name}'s
{period} earnings call. Write them the way they would actually be spoken on the call.

Analyst: {analyst}
How they ask: {style_note or "no measured style profile — keep the phrasing plain and cautious"}

For each topic you get their own most recent question on it, and the figures
management is disclosing this quarter.

{chr(10).join(blocks)}

Write ONE question per topic, in this analyst's voice.

Rules:
- The reported period is {period}. Refer to periods the way an analyst on that call would.
- Use only the figures listed above, or the analyst's own prior wording. Invent no numbers.
- Cite AT MOST one or two figures — the one that actually provokes the question.
  Reading the whole disclosure back to management is not how analysts speak; they
  pick the number that bothers them and press on it.
- Lead with the concern, not a recitation. One to three sentences.
- No greetings or pleasantries. An analyst says good evening once per call, not once
  per topic, and these are separate predicted questions rather than a script.
- NEVER mention base rates, anomaly scores, percentiles, probabilities, confidence,
  rankings, or anything about how the question was selected or generated. An analyst
  on a live call would never utter those words — a question containing them is rejected.
- Sound like a person asking a question, not like a system describing its evidence.

Return JSON only:
{{"questions": [{{"topic": "<exact topic>", "question_text": "<the question>"}}, ...]}}
Include every topic listed above, in the same order."""


_PLEASANTRY_WORDS = re.compile(
    r"\b(hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks?|thank\s+you|"
    r"congratulations|congrats|good\s+set\s+of\s+numbers|nice\s+set\s+of\s+numbers|"
    r"am\s+i\s+audible|can\s+you\s+hear\s+me)\b", re.I)

# Standalone acknowledgements ("Yes.", "Sure.", "Okay.") that open a turn on a
# live call and carry nothing. Only dropped when the whole sentence is one.
_FILLER_ONLY = re.compile(r"^(yes|yeah|yep|sure|okay|ok|right|alright)[\s,.!—-]*$", re.I)


def _strip_pleasantries(text: str | None) -> str | None:
    """Drop leading greeting sentences. The prompt already forbids them, but
    models reach for call-transcript cadence anyway, and seven questions each
    opening "Hi, good evening" reads as a template rather than an analyst.

    Works sentence by sentence rather than on a leading token run, so
    "Thanks, and congratulations on the numbers." goes as a unit instead of
    leaving a dangling "And congratulations…". A sentence is only dropped when
    it is short, carries no figure, and is pleasantry all the way through —
    so "Congratulations. My first question is on LDR." loses the greeting and
    keeps the question. Stripping beats rejecting here: it costs no retry
    budget and never demotes an otherwise good question to a bare topic."""
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    i = 0
    while i < len(parts):
        p = parts[i].strip()
        words = p.split()
        if (p and len(words) <= 12 and not re.search(r"\d", p) and "?" not in p
                and (_PLEASANTRY_WORDS.search(p) or _FILLER_ONLY.match(p))):
            i += 1
            continue
        break
    cleaned = " ".join(parts[i:]).strip()
    if not cleaned:
        return text
    return cleaned[0].upper() + cleaned[1:]


def frame_questions(analyst: str, style_note: str, topics: list[str], pool: dict, client,
                    target_quarter: str = "", feedback: str = "",
                    bank_name: str = "Axis Bank") -> dict:
    """Returns {topic: question_text or None}. Retries live in
    verifier.grounding_gate, which calls this again with only the failing
    subset plus a `feedback` line saying what was wrong."""
    if not topics:
        return {}
    prompt = _build_frame_prompt(analyst, style_note, topics, pool, target_quarter, bank_name=bank_name)
    if feedback:
        prompt += f"\n\nThe previous attempt was rejected: {feedback}\nFix exactly that."
    raw = client.call_llm(prompt, temperature=0.15, purpose="question_framer", bank_id=bank_name)
    if not raw:
        return {t: None for t in topics}
    clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean).get("questions", [])
    except Exception:
        return {t: None for t in topics}
    out = {t: None for t in topics}
    for item in parsed:
        t = item.get("topic")
        if t in out:
            out[t] = _strip_pleasantries(item.get("question_text"))
    return out


# ===========================================================================
# Move-aware framing.
#
# One question per topic was the shape that capped question-level recall: an
# analyst asking three things about NIM got one shot, and the one shot was
# almost always the level/driver question management had already prepared.
# Slots are now (topic x move) pairs from move_planner, and each move carries
# its own instruction plus the specific evidence that licenses it.
#
# The evidence is not decoration. A guidance_callback is only generated when a
# real outstanding commitment is in hand, and the commitment's own words go
# into the prompt — so the figure the model may use is a figure it was shown,
# and the Verifier's accept-set still equals the seen-set.
# ===========================================================================

MOVE_BRIEF = {
    "guidance_callback": (
        "Press management on a promise they made on an EARLIER call and have not "
        "closed out. Name what they said and ask where it stands now. This is the "
        "question management is least ready for, because the promise is several "
        "calls behind them."),
    "quantification_demand": (
        "Management has made a claim in words with no number attached. Ask them to "
        "put a number on it. Do not soften it into a general question about the topic."),
    "one_off_vs_steady_state": (
        "A figure moved unusually this quarter. Ask whether it repeats — is this the "
        "new run level or does it reverse next quarter."),
    "normalized_run_rate": (
        "Management has flagged something non-recurring inside a number. Ask what the "
        "clean, modellable figure is once that is stripped out."),
    "cross_line_bridge": (
        "Two disclosed figures sit awkwardly together. Ask them to reconcile the two — "
        "state the tension plainly rather than asking for general colour."),
    "absence_probe": (
        "Something moved and management did not address it in the remarks. Ask about the "
        "thing they left out."),
    "forward_trajectory": (
        "Management has given the level. Ask where it goes from here — the trajectory, "
        "not the print."),
}


def _fmt_evidence(move: str, evidence) -> list[str]:
    """Render only what this move is licensed to use."""
    lines = []
    if move == "guidance_callback":
        for c in (evidence or [])[:2]:
            lines.append(f'- They committed in {period_label(c["made_in"])} '
                         f'({c["quarters_elapsed"]} quarters ago, status {c["status"]}): '
                         f'"{c["text"][:280]}"')
    elif move == "quantification_demand":
        for s in (evidence or [])[:2]:
            lines.append(f'- Management says, with no number attached: "{s[:260]}"')
    elif move == "one_off_vs_steady_state":
        for m in (evidence or {}).get("metrics", [])[:2]:
            lines.append(f"- Unusual move this quarter: {_fmt_metric(m)}")
    elif move == "normalized_run_rate":
        for m in (evidence or {}).get("metrics", [])[:2]:
            lines.append(f"- Disclosed: {_fmt_metric(m)}")
        for s in (evidence or {}).get("one_off_language", [])[:2]:
            lines.append(f'- Management flags something non-recurring: "{s[:260]}"')
    elif move == "cross_line_bridge":
        for m in (evidence or {}).get("metrics", [])[:3]:
            lines.append(f"- Disclosed: {_fmt_metric(m)}")
        for s in (evidence or {}).get("bridge_sentences", [])[:2]:
            lines.append(f'- Management\'s own bridge: "{s[:260]}"')
    elif move == "absence_probe":
        for m in (evidence or {}).get("metrics", [])[:2]:
            lines.append(f"- Moved, but barely addressed in the remarks: {_fmt_metric(m)}")
    elif move == "forward_trajectory":
        for m in (evidence or {}).get("metrics", [])[:2]:
            lines.append(f"- Disclosed level: {_fmt_metric(m)}")
        for s in (evidence or {}).get("narration", [])[:1]:
            lines.append(f'- Management says on this call: "{s[:260]}"')
    return lines


def _build_move_prompt(analyst: str, style_note: str, slots: list[dict],
                       pool: dict, target_quarter: str, bank_name: str = "Axis Bank") -> str:
    period = period_label(target_quarter)
    blocks = []
    for n, s in enumerate(slots):
        ev = _fmt_evidence(s["move"], s.get("evidence"))
        prec = (pool.get(s["topic"]) or {}).get("precedent")
        lines = [f"### [{n}] {s['topic']} — {s['move_label']}",
                 f"- What to do: {MOVE_BRIEF[s['move']]}"]
        if prec:
            lines.append(f'- How they put a question on this topic in '
                         f'{period_label(prec["quarter"])}: "{prec["text"][:240]}"')
        lines.extend(ev or ["- No specific figure available — ask it plainly, with no numbers."])
        blocks.append("\n".join(lines))

    return f"""You are drafting the questions {analyst} is most likely to ask on {bank_name}'s
{period} earnings call. Write them the way they would actually be spoken.

Analyst: {analyst}
How they ask: {style_note or "no measured style profile — keep the phrasing plain and cautious"}

Each item below is a topic PLUS the specific kind of pressure this analyst tends
to apply. Write the question that applies that pressure. A generic question about
the topic is a failure even if it is well written — management has already
prepared for those.

{chr(10).join(blocks)}

Rules:
- The reported period is {period}. Refer to periods as an analyst on that call would.
- Use ONLY the figures and quotations given under that item. Invent no numbers.
- Cite at most one or two figures — the one that provokes the question.
- Lead with the concern. One to three sentences.
- No greetings or pleasantries.
- NEVER mention base rates, anomaly scores, percentiles, probabilities, confidence,
  rankings, or anything about how the question was selected. A question containing
  them is rejected.
- Do the move that item asks for. If the evidence given does not actually support
  that move, return null for that item rather than writing something vague.

Return JSON only:
{{"questions": [{{"index": <int>, "topic": "<exact topic>", "move": "<exact move>",
                 "question_text": "<the question, or null>"}}, ...]}}
Include every item above, in the same order."""


def frame_move_questions(analyst: str, style_note: str, slots: list[dict], pool: dict,
                         client, target_quarter: str = "",
                         bank_name: str = "Axis Bank") -> list[dict]:
    """Generate one question per (topic x move) slot.

    Falls back to the bare topic+move label when no LLM is reachable, so the
    air-gapped path still produces a usable brief — the same graceful
    degradation the rest of the pipeline uses.
    """
    if not slots:
        return []

    def _bare():
        return [{"topic": s["topic"], "move": s["move"], "move_label": s["move_label"],
                 "question_text": None, "status": "topic_move_only"} for s in slots]

    if client is None or not (getattr(client, "active_llm", None) or client.probe_llm()):
        return _bare()

    raw = client.call_llm(_build_move_prompt(analyst, style_note, slots, pool, target_quarter,
                                             bank_name=bank_name),
                          temperature=0.3, purpose="question_framer_move_classification", bank_id=bank_name)
    if not raw:
        return _bare()
    try:
        clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean)
    except Exception:
        return _bare()

    by_index = {q.get("index"): q for q in parsed.get("questions", []) if isinstance(q, dict)}
    out = []
    for n, s in enumerate(slots):
        q = by_index.get(n) or {}
        text = _strip_pleasantries(q.get("question_text"))
        out.append({"topic": s["topic"], "move": s["move"], "move_label": s["move_label"],
                    "question_text": text,
                    "evidence": s.get("evidence"),
                    "status": "drafted" if text else "topic_move_only"})
    return out
