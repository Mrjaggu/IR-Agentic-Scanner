"""
eval_history.py -- Section: continuous eval trend (Usage & Audit's sibling
for prediction quality, not LLM spend).

Everything this module logs already gets COMPUTED by eval_harness.py every
time someone opens the Evaluation tab or an ingest lands a new quarter --
it just previously vanished the moment the in-memory _eval_cache entry got
evicted or the server restarted. This is the same "durable JSONL, not just
an in-memory counter" upgrade that src.audit.llm_audit already did for LLM
calls, applied to eval RESULTS instead: one row per real
eval_harness.run_holdout_eval() computation (never for a cache hit, and
never for the scoped/ad-hoc analyst-or-quarter checks eval_harness's own
docstring says shouldn't be read as the real result), so a recall trend can
be plotted over time without re-running anything.

QUESTION RECALL IS THE HEADLINE METRIC HERE, NOT TOPIC RECALL. That's a
deliberate product decision, not an oversight: topic recall (mean_recall)
answers "was the right bucket on the brief" and is free to compute on every
run, so it's logged every time. Question recall (mean_question_recall)
answers "did the framed QUESTION TEXT anticipate what was actually asked" --
the real objective per this project's own Question Recall panel -- but it
costs real LLM judge calls, so it's only logged when a user deliberately
runs that check (see /api/eval/question-recall's own docstring on cost).
Every reader in this module treats mean_question_recall as the number that
matters whenever it's available, and topic recall as the always-on context
metric that fills the gaps in between -- not the other way around.
"""

import json
import os
import threading
from datetime import datetime, timezone

from src.config.settings import writable_data_dir

_lock = threading.Lock()


def _path(bank_id: str) -> str:
    return os.path.join(writable_data_dir("eval_history"), f"eval_runs_{bank_id}.jsonl")


def log_run(result: dict, bank_id: str, kind: str) -> None:
    """Append one row for a REAL eval_harness.run_holdout_eval() computation.

    kind: "topic" for a score_question_recall=False run (free, deterministic,
    logged on every real computation -- see _cached_holdout in fastapi_app.py)
    or "question" for score_question_recall=True (real LLM judge calls,
    logged only from the one cache-populating call site that represents the
    full held-out set -- see _cached_question_recall).

    Never raises -- a logging failure here must not break the eval request
    that triggered it, same discipline as src.audit.llm_audit.log_call."""
    try:
        summary = result.get("summary", {}) or {}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "bank_id": bank_id,
            "kind": kind,
            "test_quarters": result.get("test_quarters"),
            "mean_recall": summary.get("mean_recall"),
            "mean_question_recall": summary.get("mean_question_recall"),
            "mean_precision": summary.get("mean_precision"),
            "mean_f2": summary.get("mean_f2"),
            "recall_spread": summary.get("recall_spread"),
            "promotion_gate_verdict": (result.get("promotion_gate") or {}).get("verdict"),
        }
        path = _path(bank_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"  [eval_history] failed to log eval run for bank={bank_id!r} (non-fatal): {e}")


def _read_all(bank_id: str) -> list[dict]:
    path = _path(bank_id)
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a partially-written line (crash mid-write) -- skip, don't crash the reader
    except OSError:
        return []
    return out


def trend(bank_id: str, limit: int = 60) -> dict:
    """Chronological (oldest-first, chart-ready) trend for the Evaluation
    tab. `runs` is capped to the most recent `limit` real computations.
    latest_topic_recall / latest_question_recall surface the two most
    recently KNOWN values independently, since question recall is logged
    far more sparsely than topic recall by design (see module docstring) --
    the most recent row often has a topic number but no question number,
    and reading straight off the last row would make question recall look
    like it regressed to "no data" every time only a cheap run happened."""
    rows = _read_all(bank_id)
    rows = rows[-limit:] if limit else rows
    latest_topic = next((r["mean_recall"] for r in reversed(rows) if r.get("mean_recall") is not None), None)
    latest_question = next((r["mean_question_recall"] for r in reversed(rows)
                            if r.get("mean_question_recall") is not None), None)
    question_runs = sum(1 for r in rows if r.get("mean_question_recall") is not None)
    return {
        "bank_id": bank_id,
        "runs": rows,
        "latest_topic_recall": latest_topic,
        "latest_question_recall": latest_question,
        "question_recall_runs_logged": question_runs,
        "total_runs_logged": len(rows),
    }
