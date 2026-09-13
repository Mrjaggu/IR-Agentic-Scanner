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
                  "source_url": "...", "confidence": "confirmed|plausible"}]}]
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
