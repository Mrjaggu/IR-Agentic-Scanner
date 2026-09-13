"""
workspace_state.py — the "saved work" the UX review flagged as missing:
pinned prep-brief items and a recent-activity log, persisted to disk so they
survive a server restart and a browser reload — not just in-memory or
localStorage, which would lose everything the moment uvicorn restarts.

Single JSON file, single workspace (Axis Bank, one user). No concurrency
control beyond a process-local lock, which is the right amount of
engineering for what this is: nobody else is writing to this file.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

from src.config.settings import BASE_DIR

STATE_PATH = os.path.join(BASE_DIR, "data", "outputs", "workspace_state.json")
_lock = threading.Lock()

MAX_ACTIVITY = 50
MAX_BRIEF_ITEMS = 200   # a sanity cap, not a real-world limit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict:
    return {"brief_items": [], "activity": []}


def _read() -> dict:
    if not os.path.exists(STATE_PATH):
        return _empty()
    try:
        with open(STATE_PATH, "r") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty()
    d.setdefault("brief_items", [])
    d.setdefault("activity", [])
    return d


def _write(d: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, STATE_PATH)   # atomic on the same filesystem


def get_state() -> dict:
    with _lock:
        return _read()


def pin_item(kind: str, title: str, body: str, meta: dict | None = None) -> dict:
    """Add one item to the prep brief. Returns the whole updated state."""
    with _lock:
        d = _read()
        item = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,               # "chat_answer" | "passage" | "predicted_question" | "search_result"
            "title": title,
            "body": body,
            "meta": meta or {},
            "pinned_at": _now(),
        }
        d["brief_items"].append(item)
        d["brief_items"] = d["brief_items"][-MAX_BRIEF_ITEMS:]
        _write(d)
        return d


def unpin_item(item_id: str) -> dict:
    with _lock:
        d = _read()
        d["brief_items"] = [i for i in d["brief_items"] if i["id"] != item_id]
        _write(d)
        return d


def clear_brief() -> dict:
    with _lock:
        d = _read()
        d["brief_items"] = []
        _write(d)
        return d


def log_activity(kind: str, label: str, meta: dict | None = None) -> dict:
    """Append one line to the recent-activity feed. Best-effort — a failure
    here should never break the action that triggered it."""
    with _lock:
        d = _read()
        d["activity"].append({
            "kind": kind, "label": label, "meta": meta or {}, "at": _now(),
        })
        d["activity"] = d["activity"][-MAX_ACTIVITY:]
        _write(d)
        return d


def brief_as_markdown(quarter_label: str = "") -> str:
    """Renders the current prep brief as a shareable markdown document —
    the review's "export/share prep brief" step."""
    d = get_state()
    items = d["brief_items"]
    lines = [f"# Prep brief{' — ' + quarter_label if quarter_label else ''}", ""]
    if not items:
        lines.append("_Nothing pinned yet._")
        return "\n".join(lines)
    by_kind = {}
    for it in items:
        by_kind.setdefault(it["kind"], []).append(it)
    KIND_LABEL = {
        "chat_answer": "Answers", "passage": "Source passages",
        "predicted_question": "Predicted questions", "search_result": "Search results",
    }
    for kind, its in by_kind.items():
        lines.append(f"## {KIND_LABEL.get(kind, kind)}")
        lines.append("")
        for it in its:
            lines.append(f"**{it['title']}**")
            lines.append("")
            lines.append(it["body"])
            m = it.get("meta") or {}
            tag_bits = [f"{k}: {v}" for k, v in m.items() if v]
            if tag_bits:
                lines.append("")
                lines.append(f"_{' · '.join(tag_bits)}_")
            lines.append("")
    return "\n".join(lines)
