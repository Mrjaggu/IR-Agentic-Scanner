"""
external_context.py — verified external news/macro context per quarter.

Unlike narration_novelty.py / peer_signal.py (which call Groq/Gemini directly
over HTTP), web search is an agent-level tool, not something this codebase can
invoke on its own — there is no scriptable "web search API key" in .env. This
module is therefore a validation/persistence helper, not a standalone runner:
an agent performs the actual search, curates verified results (real source
URL, own-words summary, confidence flag), and calls save_external_context().

VERIFICATION DISCIPLINE (non-negotiable — see RESULTS_V2.md's Jindal Saw
near-miss from the peer-bank sourcing today): every event must carry a real,
checkable source_url actually returned by a search — never fabricate or guess
one. Summarize in your own words, don't reproduce scraped article text
(copyright). Mark confidence "confirmed" only if the source clearly concerns
Axis Bank / the named sector event; use "plausible" for anything inferred or
uncertain, so downstream reasoning (question_intent.py) can weight it.

Output: data/inputs/external_context_history.json
    [{"quarter": "q1fy27", "call_date": "July 18, 2026",
      "events": [{"date": "...", "headline": "...", "summary": "...",
                  "source_url": "...", "confidence": "confirmed|plausible",
                  "topics": ["NIM & Yields", "..."], "severity": "low|medium|high",
                  "direction": "negative|positive|neutral"}]}]

`topics`/`severity`/`direction` are optional on write (older/partial
records stay valid) but required for an event to feed
src.agentic.analyst_layer.macro_event_extra_slot via macro_topic_signal()
below -- an event missing them is stored fine, just contributes no signal.
"""

import json
import os

from src.config.settings import EXTERNAL_CONTEXT_PATH

_VALID_CONFIDENCE = {"confirmed", "plausible"}


def _validate_event(ev: dict) -> None:
    required = {"date", "headline", "summary", "source_url", "confidence"}
    missing = required - ev.keys()
    if missing:
        raise ValueError(f"external context event missing fields: {missing} — {ev}")
    if ev["confidence"] not in _VALID_CONFIDENCE:
        raise ValueError(f"invalid confidence {ev['confidence']!r}, must be one of {_VALID_CONFIDENCE}")
    if not ev["source_url"].startswith("http"):
        raise ValueError(f"source_url must be a real URL, got: {ev['source_url']!r}")
    # topics/severity/direction are OPTIONAL (see macro_topic_signal below) --
    # an event without them is still a valid stored record, it just
    # contributes no macro-event signal. But if given, they must be
    # well-formed, since a typo here would silently mis-score a topic.
    if "severity" in ev and ev["severity"] not in ("low", "medium", "high"):
        raise ValueError(f"invalid severity {ev['severity']!r}, must be one of low/medium/high")
    if "topics" in ev and not isinstance(ev["topics"], list):
        raise ValueError(f"topics must be a list, got: {ev['topics']!r}")
    if "direction" in ev and ev["direction"] not in ("negative", "positive", "neutral"):
        raise ValueError(f"invalid direction {ev['direction']!r}, must be one of negative/positive/neutral")


def save_external_context(quarter: str, call_date: str, events: list[dict],
                          path: str = EXTERNAL_CONTEXT_PATH) -> dict:
    for ev in events:
        _validate_event(ev)

    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history = [h for h in history if h["quarter"] != quarter]  # replace existing entry

    entry = {"quarter": quarter, "call_date": call_date, "events": events}
    history.append(entry)
    history.sort(key=lambda h: h["quarter"])

    os.makedirs(os.path.dirname(path), exist_ok=True)  # newly-registered banks (kotak/indusind)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[external] {quarter}: saved {len(events)} verified event(s) -> {path}")
    return entry


def load_external_context(quarter: str, path: str = EXTERNAL_CONTEXT_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        history = json.load(f)
    for h in history:
        if h["quarter"] == quarter:
            return h
    return None


_SEVERITY_SCORE = {"low": 0.34, "medium": 0.67, "high": 1.0}


def macro_topic_signal(quarter: str, min_severity: str = "medium",
                       path: str = EXTERNAL_CONTEXT_PATH) -> dict[str, float]:
    """{topic: severity_score in [0,1]} from this quarter's verified
    external events, filtered to >= min_severity, or {} if nothing is
    curated for this quarter yet (or nothing curated carries topics/
    severity -- see below). Mirrors src.signals.news_signal.news_topic_salience's
    contract: an empty dict means "no signal", never "zero salience for
    every topic", so callers must not treat {} as a confident negative.

    Reads events via save_external_context's schema, extended with two
    fields this function actually needs: `topics` (list[str], taxonomy
    names from src.config.settings.TOPICS_LIST) and `severity`
    ("low"|"medium"|"high"). Neither is required by _validate_event --
    an event curated without them is still a valid, storable external-
    context record (real source_url, own-words summary, confidence), it
    just contributes nothing HERE, the same graceful "no signal" the rest
    of this codebase uses (e.g. src.agentic.tools.policy_tool) rather than
    crashing on an older or partial record.

    `direction` ("negative"|"positive"|"neutral", if present) is
    informational only for this mechanism -- severity alone decides
    whether a topic gets flagged, because the extra-slot mechanism this
    feeds (src.agentic.analyst_layer.macro_event_extra_slot) is about
    whether an analyst is more likely to PROBE the topic at all, not which
    way the event cuts: a positive surprise invites "is this sustainable"
    scrutiny the same way a negative one invites "how bad is the damage"."""
    entry = load_external_context(quarter, path=path)
    if not entry:
        return {}
    min_score = _SEVERITY_SCORE.get(min_severity, 0.67)
    signal: dict[str, float] = {}
    for ev in entry.get("events", []):
        sev = _SEVERITY_SCORE.get(ev.get("severity"), 0.0)
        if sev < min_score:
            continue
        for t in ev.get("topics", []) or []:
            signal[t] = max(signal.get(t, 0.0), sev)
    return signal
