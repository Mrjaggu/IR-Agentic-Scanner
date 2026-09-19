"""
llm_audit.py -- Section: LLM call audit trail (token usage + logging).

Every LLM provider call this platform makes -- across Groq, Cerebras,
OpenRouter, NVIDIA NIM, Gemini, and OpenAI -- gets logged here: who called
it (`purpose`), which provider/model answered, how many tokens it used,
how long it took, and whether it succeeded. This is the "can we track any
time" audit trail: a durable, append-only record that survives a server
restart, not just the in-memory STATS counters in
src.model_provider.llm_client (which reset every run and were never meant
to answer "what did we spend on Tuesday").

Storage: one JSONL file per UTC day under data/logs/ (llm_calls_YYYY-MM-DD.jsonl),
one JSON object per line, append-only. This mirrors the rest of this
codebase's persistence convention (plain JSON files, no database) while
still being genuinely durable and trivially greppable/tailable -- an
operator can `tail -f data/logs/llm_calls_$(date +%F).jsonl` and watch
calls happen live. Daily rotation keeps any single file from growing
unbounded; nothing here needs cross-day joins beyond what summary()/recent()
already do by reading the last few days' files.

Every write is append-only and wrapped in a lock -- concurrent requests
(FastAPI serving several chat/predict calls at once) must not interleave
or corrupt a line. A write failure here never breaks the LLM call itself:
log_call() catches and swallows its own I/O errors (degrade honestly, same
as src.news.news_api and src.tts.piper_tts) -- an audit log that can crash
the feature it's auditing would be worse than no audit log.
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from src.config.settings import BASE_DIR

LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
_lock = threading.Lock()

# Rough $/1K-token pricing for cost estimation. Only providers/models with a
# real, published per-token price are listed -- everything else in this
# platform's fallback chain (Groq, Gemini free tier, OpenRouter free models,
# NVIDIA NIM) is genuinely $0 on the tier this app uses, not "unknown", so
# they're listed explicitly as free rather than omitted (omitting them would
# make a summary's cost total silently wrong, undercounting nothing but
# looking incomplete). A model not listed here at all gets `cost_usd: null`
# in its log line -- estimate-honestly, not a guessed number.
PRICING_PER_1K_TOKENS = {
    # provider -> {model_prefix: {"prompt": $/1K prompt tokens, "completion": $/1K completion tokens}}
    "OpenAI": {
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    },
    "Groq": {"_free": True},
    "Gemini": {"_free": True},   # free tier, per PROVIDER_LIMITS in llm_client.py
    "OpenRouter": {"_free": True},  # ":free"-suffixed models only, per this app's own config
    "NVIDIA NIM": {"_free": True},
    "Cerebras": {"_free": True},
}


def _estimate_cost_usd(provider: str, model: str, prompt_tokens: int | None,
                        completion_tokens: int | None) -> float | None:
    table = PRICING_PER_1K_TOKENS.get(provider)
    if table is None:
        return None
    if table.get("_free"):
        return 0.0
    if prompt_tokens is None or completion_tokens is None:
        return None
    # Match by prefix since a model string can carry a date/version suffix
    # (e.g. "gpt-4o-mini-2026-07-18") that the pricing table doesn't track.
    for prefix, rates in table.items():
        if prefix.startswith("_"):
            continue
        if model and model.startswith(prefix):
            return round(prompt_tokens / 1000 * rates["prompt"]
                         + completion_tokens / 1000 * rates["completion"], 6)
    return None


def _log_path(day: str | None = None) -> str:
    day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"llm_calls_{day}.jsonl")


def log_call(*, provider: str, model: str | None, purpose: str | None,
             success: bool, latency_ms: float,
             prompt_tokens: int | None = None, completion_tokens: int | None = None,
             total_tokens: int | None = None, tokens_estimated: bool = False,
             error: str | None = None, bank_id: str | None = None) -> None:
    """Append one audit record. Never raises -- a logging failure must not
    take down the LLM call it's auditing (see module docstring)."""
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "provider": provider,
            "model": model,
            "purpose": purpose or "unspecified",
            "bank_id": bank_id,
            "success": success,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_estimated": tokens_estimated,
            "cost_usd": _estimate_cost_usd(provider, model, prompt_tokens, completion_tokens),
            "error": error[:300] if error else None,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _lock:
            with open(_log_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        print(f"  [llm_audit] failed to write audit log entry (non-fatal): {e}")


def _iter_recent_days(days: int):
    """Yields (day_str, file_path) for the last `days` UTC days that actually
    have a log file, newest first -- most summaries only need a handful of
    days, so this avoids scanning the whole data/logs/ directory."""
    today = datetime.now(timezone.utc).date()
    for offset in range(days):
        day = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        path = _log_path(day)
        if os.path.exists(path):
            yield day, path


def _read_lines(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partially-written line (crash mid-write) -- skip, don't crash the reader
    except FileNotFoundError:
        return


def summary(days: int = 7) -> dict:
    """Aggregated totals over the last `days` UTC days: call counts, token
    totals, estimated cost, a per-provider breakdown, and a per-day series
    (for the Usage & Audit chart). Reads whatever daily files exist --
    empty/missing days just don't contribute, no error."""
    totals = {"calls": 0, "success": 0, "failed": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
              "cost_usd": 0.0, "cost_has_unknown": False}
    by_provider: dict[str, dict] = {}
    by_day: dict[str, dict] = {}

    for day, path in _iter_recent_days(days):
        day_bucket = by_day.setdefault(day, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0})
        for rec in _read_lines(path):
            totals["calls"] += 1
            totals["success" if rec.get("success") else "failed"] += 1
            pt, ct, tt = rec.get("prompt_tokens"), rec.get("completion_tokens"), rec.get("total_tokens")
            if isinstance(pt, (int, float)):
                totals["prompt_tokens"] += pt
            if isinstance(ct, (int, float)):
                totals["completion_tokens"] += ct
            if isinstance(tt, (int, float)):
                totals["total_tokens"] += tt
                day_bucket["total_tokens"] += tt
            cost = rec.get("cost_usd")
            if cost is None:
                totals["cost_has_unknown"] = True
            else:
                totals["cost_usd"] += cost
                day_bucket["cost_usd"] += cost
            day_bucket["calls"] += 1

            prov = rec.get("provider") or "unknown"
            pb = by_provider.setdefault(prov, {"calls": 0, "success": 0, "failed": 0,
                                                "total_tokens": 0, "cost_usd": 0.0})
            pb["calls"] += 1
            pb["success" if rec.get("success") else "failed"] += 1
            if isinstance(tt, (int, float)):
                pb["total_tokens"] += tt
            if cost is not None:
                pb["cost_usd"] += cost

    totals["cost_usd"] = round(totals["cost_usd"], 4)
    for pb in by_provider.values():
        pb["cost_usd"] = round(pb["cost_usd"], 4)
    for db in by_day.values():
        db["cost_usd"] = round(db["cost_usd"], 4)

    return {
        "days": days,
        "totals": totals,
        "by_provider": by_provider,
        "by_day": [{"day": d, **by_day[d]} for d in sorted(by_day.keys())],
    }


def recent(limit: int = 50, provider: str | None = None, purpose: str | None = None,
           days_to_scan: int = 14) -> list[dict]:
    """Most recent `limit` calls, newest first, optionally filtered by
    provider/purpose. Scans back up to `days_to_scan` daily files -- enough
    for a low-traffic app like this one to reliably fill `limit`, without
    reading unbounded history for what's meant to be a recent-activity view
    (see summary() for longer-range aggregates)."""
    out: list[dict] = []
    for _day, path in _iter_recent_days(days_to_scan):
        day_records = list(_read_lines(path))
        for rec in reversed(day_records):  # file is append-ordered oldest-first; reverse for newest-first
            if provider and rec.get("provider") != provider:
                continue
            if purpose and rec.get("purpose") != purpose:
                continue
            out.append(rec)
            if len(out) >= limit:
                return out
    return out
