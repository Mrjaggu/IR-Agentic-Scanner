"""
workspace_state.py — the "saved work" the UX review flagged as missing:
pinned prep-brief items and a recent-activity log, persisted to disk so they
survive a server restart and a browser reload — not just in-memory or
localStorage, which would lose everything the moment uvicorn restarts.

One JSON file PER BANK (data/outputs/<bank_id>/workspace_state.json), single
user. No concurrency control beyond a process-local lock, which is the right
amount of engineering for what this is: nobody else is writing these files.

2026-09: every function below gained a `bank_id` param, default
DEFAULT_BANK ("axis") -- so every existing caller (fastapi_app.py's brief
routes, which don't pass bank_id yet -- that's a later phase) keeps working
completely unchanged, pinned to axis's own workspace file, same as before
this file had any notion of "bank" at all.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

from src.config.settings import BASE_DIR, writable_data_dir
from src.config.banks import DEFAULT_BANK

_lock = threading.Lock()

MAX_ACTIVITY = 50
MAX_BRIEF_ITEMS = 200   # a sanity cap, not a real-world limit


def _state_path(bank_id: str = DEFAULT_BANK) -> str:
    # writable_data_dir falls back off BASE_DIR only if BASE_DIR/data/outputs
    # itself isn't writable (e.g. Vercel's read-only /var/task) -- see its
    # docstring in src/config/settings.py for what that fallback does and
    # does not guarantee.
    return os.path.join(writable_data_dir("outputs", bank_id), "workspace_state.json")


# Kept for any external code that still imports the flat constant directly
# (introspection/back-compat only -- every function below resolves its own
# path via _state_path(bank_id), this is not read internally).
STATE_PATH = _state_path(DEFAULT_BANK)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> dict:
    return {"brief_items": [], "activity": []}


def _read(bank_id: str = DEFAULT_BANK) -> dict:
    path = _state_path(bank_id)
    if not os.path.exists(path):
        return _empty()
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty()
    d.setdefault("brief_items", [])
    d.setdefault("activity", [])
    return d


def _write(d: dict, bank_id: str = DEFAULT_BANK) -> None:
    """Best-effort, same discipline as log_activity's own docstring already
    promises: a failure to persist must never break the pin/unpin/log action
    that triggered it. _state_path already prefers a writable directory over
    BASE_DIR (see writable_data_dir), but this still can't raise even if
    THAT somehow fails too (disk full, permissions) -- the in-memory dict the
    caller already mutated is what the response is built from either way."""
    path = _state_path(bank_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)   # atomic on the same filesystem
    except OSError as e:
        print(f"  [workspace_state] failed to persist state for bank={bank_id!r} "
              f"(non-fatal, in-memory result still returned): {e}")


def get_state(bank_id: str = DEFAULT_BANK) -> dict:
    with _lock:
        return _read(bank_id)


def pin_item(kind: str, title: str, body: str, meta: dict | None = None,
            bank_id: str = DEFAULT_BANK) -> dict:
    """Add one item to the prep brief. Returns the whole updated state."""
    with _lock:
        d = _read(bank_id)
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
        _write(d, bank_id)
        return d


def unpin_item(item_id: str, bank_id: str = DEFAULT_BANK) -> dict:
    with _lock:
        d = _read(bank_id)
        d["brief_items"] = [i for i in d["brief_items"] if i["id"] != item_id]
        _write(d, bank_id)
        return d


def clear_brief(bank_id: str = DEFAULT_BANK) -> dict:
    with _lock:
        d = _read(bank_id)
        d["brief_items"] = []
        _write(d, bank_id)
        return d


def log_activity(kind: str, label: str, meta: dict | None = None,
                 bank_id: str = DEFAULT_BANK) -> dict:
    """Append one line to the recent-activity feed. Best-effort — a failure
    here should never break the action that triggered it."""
    with _lock:
        d = _read(bank_id)
        d["activity"].append({
            "kind": kind, "label": label, "meta": meta or {}, "at": _now(),
        })
        d["activity"] = d["activity"][-MAX_ACTIVITY:]
        _write(d, bank_id)
        return d


KIND_LABEL = {
    "chat_answer": "Answers", "passage": "Source passages",
    "predicted_question": "Predicted questions", "search_result": "Search results",
}


def grouped_brief_items(bank_id: str = DEFAULT_BANK) -> list[tuple[str, list[dict]]]:
    """The brief's items grouped by kind, in KIND_LABEL's own display order --
    the one grouping brief_as_markdown() built inline, now shared so
    src/data/brief_export.py's docx/pdf renderers don't duplicate it (and so
    all three export formats stay in sync by construction, not by hand)."""
    items = get_state(bank_id)["brief_items"]
    by_kind: dict[str, list[dict]] = {}
    for it in items:
        by_kind.setdefault(it["kind"], []).append(it)
    ordered = [k for k in KIND_LABEL if k in by_kind] + [k for k in by_kind if k not in KIND_LABEL]
    return [(k, by_kind[k]) for k in ordered]


def brief_as_markdown(quarter_label: str = "", bank_id: str = DEFAULT_BANK) -> str:
    """Renders the current prep brief as a shareable markdown document —
    the review's "export/share prep brief" step."""
    items = get_state(bank_id)["brief_items"]
    lines = [f"# Prep brief{' — ' + quarter_label if quarter_label else ''}", ""]
    if not items:
        lines.append("_Nothing pinned yet._")
        return "\n".join(lines)
    for kind, its in grouped_brief_items(bank_id):
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
