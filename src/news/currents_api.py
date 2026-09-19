"""
currents_api.py -- Section: News tab, external headlines via CurrentsAPI.

Same "degrade honestly" discipline as everywhere else in this codebase that
depends on something outside its control (src.tts.piper_tts's voice model,
src.search.embeddings' spaCy model, src.model_provider.llm_client's
providers): is_available() is a plain boolean callers check before relying
on this, and every failure path returns a normal-looking result rather than
raising, so a missing key or a network hiccup never breaks the News tab --
it just shows nothing external, or shows the last thing that DID work.

Budget discipline is the other half of this module's whole reason to exist.
CurrentsAPI's free tier is 200 requests/day for the WHOLE account, not per
bank -- so naive "fetch on every page load" across three bank workspaces and
however many people have the News tab open would blow through that in
minutes. Two independent guards, both conservative on purpose:

  1. A per-bank cache with a multi-hour TTL -- news doesn't change fast
     enough to need fetching more often than that, so almost all real
     traffic is served from cache, not the API.
  2. A hard daily call cap (MAX_CALLS_PER_DAY), well under 200, so even a
     burst of manual "Refresh" clicks across every bank cannot exhaust the
     account's real daily budget. Once the cap is hit for the day, this
     serves the last good cache (with a note saying why) instead of the API.

No caching library, no background scheduler -- a plain in-memory dict is
enough here: this is headlines, not anything that needs to survive a
restart, and every other short-lived cache in this codebase (_tts_cache,
_eval_cache, _cache in fastapi_app.py) follows the same in-memory convention.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request

API_KEY = os.environ.get("CURRENT_NEWS_API_KEY")
SEARCH_URL = "https://api.currentsapi.services/v1/search"

# Free tier is 200/day for the account. This cap leaves real headroom under
# that even with several banks and manual refreshes in the same day -- see
# the module docstring for why this exists alongside the per-bank TTL below.
MAX_CALLS_PER_DAY = 150
CACHE_TTL_SECONDS = 3 * 60 * 60  # 3 hours -- news doesn't move fast enough to need more.
REQUEST_TIMEOUT_SECONDS = 10

_lock = threading.Lock()
_cache: dict[str, dict] = {}          # bank_id -> {"articles": [...], "fetched_at": float, "error": str|None}
_daily_calls = {"date": None, "count": 0}


def is_available() -> bool:
    return bool(API_KEY)


def _budget_ok() -> bool:
    """Resets the counter at UTC-day rollover. Not persisted across a
    process restart -- an occasional extra day of full budget after a
    restart is a fine trade for not needing a database just to count
    requests."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _daily_calls["date"] != today:
        _daily_calls["date"] = today
        _daily_calls["count"] = 0
    return _daily_calls["count"] < MAX_CALLS_PER_DAY


def _fetch(bank_name: str) -> list[dict]:
    """Phrase-matches the bank's own display name via CurrentsAPI's `query`
    param (boolean/quoted syntax), not `keywords` (plain term search) --
    "Kotak Mahindra Bank" as a phrase is materially more precise than the
    three words matched independently, which would surface a lot of
    unrelated "bank" or "Kotak" noise for a multi-word name like this."""
    params = urllib.parse.urlencode({
        "query": f'"{bank_name}"',
        "language": "en",
        "page_size": 10,
    })
    url = f"{SEARCH_URL}?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("status") != "ok":
        raise RuntimeError(data.get("message") or "CurrentsAPI returned a non-ok status")
    articles = []
    for a in data.get("news", []) or []:
        articles.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "author": a.get("author"),
            "published": a.get("published"),
            "image": a.get("image"),
        })
    return articles


def get_news(bank_id: str, bank_name: str, force_refresh: bool = False) -> dict:
    """Cached per bank, refreshed at most every CACHE_TTL_SECONDS and hard
    capped at MAX_CALLS_PER_DAY across ALL banks combined. On a cache miss
    that fails (budget exhausted, network error, bad response), the last
    good cache is returned with a `note` explaining why it's stale, rather
    than an error -- headlines that are a few hours old beat a broken tab."""
    if not is_available():
        return {"available": False, "articles": [], "fetched_at": None,
                "error": "no CurrentsAPI key configured (CURRENT_NEWS_API_KEY)"}

    with _lock:
        cached = _cache.get(bank_id)
        fresh_enough = bool(cached) and (time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS)
        if cached and fresh_enough and not force_refresh:
            return {"available": True, **cached}

        if not _budget_ok():
            if cached:
                return {"available": True, **cached,
                        "note": "today's news-request budget is used up -- showing the last cached result"}
            return {"available": True, "articles": [], "fetched_at": None,
                    "error": "today's news-request budget is used up and nothing is cached yet"}

        _daily_calls["count"] += 1
        try:
            articles = _fetch(bank_name)
            _cache[bank_id] = {"articles": articles, "fetched_at": time.time(), "error": None}
        except Exception as e:
            if cached:
                return {"available": True, **cached, "note": f"refresh failed ({e}) -- showing the last cached result"}
            _cache[bank_id] = {"articles": [], "fetched_at": time.time(), "error": str(e)}

        return {"available": True, **_cache[bank_id]}
