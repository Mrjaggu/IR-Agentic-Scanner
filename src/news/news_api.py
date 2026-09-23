"""
news_api.py -- Section: News tab, external headlines via TheNewsAPI.

This module replaces src/news/currents_api.py (CurrentsAPI), which was
switched away from after that provider's own auth backend started
returning a persistent 503 ("Authentication service temporarily
unavailable") on every request -- confirmed via scripts/test_news_api.py
across urllib and requests, with and without a custom User-Agent, so it
wasn't anything on our end. TheNewsAPI (thenewsapi.com) is what replaced
it. The public interface (is_available(), get_news(bank_id, bank_name,
force_refresh)) is unchanged, so src/api/fastapi_app.py's /api/news/external
route and the News feeds tab's frontend code don't need to know a provider
switch happened at all.

Same "degrade honestly" discipline as everywhere else in this codebase that
depends on something outside its control (src.tts.piper_tts's voice model,
src.search.embeddings' spaCy model, src.model_provider.llm_client's
providers): is_available() is a plain boolean callers check before relying
on this, and every failure path returns a normal-looking result rather than
raising, so a missing token or a provider hiccup never breaks the News tab
-- it just shows nothing external, or shows the last thing that DID work.

Budget discipline is the other half of this module's whole reason to exist.
TheNewsAPI's free tier is 100 requests/day for the WHOLE account (and only
3 articles per request on that tier) -- so naive "fetch on every page load"
across three bank workspaces and however many people have the News tab open
would blow through that fast. Two independent guards, both conservative on
purpose:

  1. A per-bank cache with a multi-hour TTL -- news doesn't change fast
     enough to need fetching more often than that, so almost all real
     traffic is served from cache, not the API.
  2. A hard daily call cap (MAX_CALLS_PER_DAY), well under 100, so even a
     burst of manual "Refresh" clicks across every bank cannot exhaust the
     account's real daily budget. Once the cap is hit for the day, this
     serves the last good cache (with a note saying why) instead of the API.

No caching library, no background scheduler -- a plain in-memory dict is
enough here: this is headlines, not anything that needs to survive a
restart, and every other short-lived cache in this codebase (_tts_cache,
_eval_cache, _cache in fastapi_app.py) follows the same in-memory convention.

The one exception: a cold start (a fresh process, or Vercel's usual case --
a fresh /tmp-only filesystem per invocation, see settings.writable_data_dir's
docstring) means an empty _cache and nothing to show until the first live
call lands, which costs part of the daily budget just to paint the tab.
data/news_seed.json -- a small, committed, real set of headlines captured
once by hand -- primes an empty cache on first use per bank so the News tab
never opens blank. It's a starting point, not a live feed: the very next
budget-permitting request still replaces it with a real fetch, same as any
other stale cache entry.
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

API_TOKEN = os.environ.get("THE_NEWS_API_TOKEN")
ALL_NEWS_URL = "https://api.thenewsapi.com/v1/news/all"

# Free tier is 100/day for the account and caps `limit` at 3 articles per
# request. This call cap leaves real headroom under the daily 100 even with
# several banks and manual refreshes in the same day -- see the module
# docstring for why this exists alongside the per-bank TTL below.
MAX_CALLS_PER_DAY = 70
CACHE_TTL_SECONDS = 3 * 60 * 60  # 3 hours -- news doesn't move fast enough to need more.
REQUEST_TIMEOUT_SECONDS = 10
ARTICLES_PER_CALL = 3  # the free plan's own per-request cap; asking for more just 400s.

_lock = threading.Lock()
_cache: dict[tuple, dict] = {}        # (bank_id, days) -> {"articles": [...], "fetched_at": float, "error": str|None}
_daily_calls = {"date": None, "count": 0}

SEED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "news_seed.json")
_seed_data: dict | None = None   # loaded once, lazily -- None until first access, {} if the file is missing/bad


def _load_seed(bank_id: str) -> dict | None:
    """A one-time, real (not synthetic) headline capture per bank, used only
    to prime an empty cache -- see the module docstring's "one exception"
    paragraph. Never raises: a missing or malformed seed file just means no
    priming, not a broken News tab."""
    global _seed_data
    if _seed_data is None:
        try:
            with open(SEED_PATH) as f:
                _seed_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            _seed_data = {}
    entry = _seed_data.get(bank_id)
    if not entry:
        return None
    return {"articles": entry.get("articles", []), "fetched_at": entry.get("fetched_at"),
            "error": entry.get("error")}


ANALYST_MENTIONS_SEED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "analyst_mentions_seed.json")
_analyst_mentions_data: dict | None = None


def get_analyst_mentions(bank_id: str) -> list[dict]:
    """Real, individually-attributed sell-side rating actions (analyst name,
    firm, rating, price target, date, source URL) for this bank's covering
    analysts -- hand-gathered via search, not from TheNewsAPI.

    Why this exists as its OWN seed rather than a live feed: TheNewsAPI's
    general-news search (see get_news() above) almost never names an
    individual sell-side analyst -- confirmed by directly searching for
    each of this bank's covering analysts by name. Public news attributes
    price-target moves to the BROKER ("Jefferies raises target"), not the
    analyst who wrote the note; the note itself is typically paywalled or
    distributed only to the broker's own clients. A small number of
    individual-analyst-attributed items DO exist via third-party analyst-
    rating trackers (e.g. TipRanks' per-analyst pages) -- this file is a
    point-in-time, manually verified capture of those, not a live feed.
    It will be thin and will go stale; that's the honest state of what's
    publicly attributable, not a bug in the matching logic below."""
    global _analyst_mentions_data
    if _analyst_mentions_data is None:
        try:
            with open(ANALYST_MENTIONS_SEED_PATH) as f:
                _analyst_mentions_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            _analyst_mentions_data = {}
    return _analyst_mentions_data.get(bank_id, [])


def is_available() -> bool:
    return bool(API_TOKEN)


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


def _fetch(bank_name: str, days: int | None) -> list[dict]:
    """Phrase-matches the bank's own display name via TheNewsAPI's `search`
    param, which supports quoted-phrase syntax -- "Kotak Mahindra Bank" as a
    phrase is materially more precise than the three words matched
    independently, which would surface a lot of unrelated "bank" or "Kotak"
    noise for a multi-word name like this. Restricted to title+description
    via `search_fields` so a bank name buried in an article's full body text
    (an unrelated listicle that happens to mention it once) doesn't count.

    `days` (7, 30, or None for TheNewsAPI's full archive) becomes
    `published_after` -- a Y-m-d cutoff computed in UTC, since TheNewsAPI's
    docs specify UTC timestamps throughout and a bare ISO date is the one
    format they document accepting for this parameter."""
    query = {
        "api_token": API_TOKEN,
        "search": f'"{bank_name}"',
        "search_fields": "title,description",
        "language": "en",
        "limit": ARTICLES_PER_CALL,
    }
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        query["published_after"] = cutoff
    params = urllib.parse.urlencode(query)
    url = f"{ALL_NEWS_URL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # TheNewsAPI's error body is JSON ({"error": {"code": ..., "message": ...}})
        # -- surface that message instead of the bare status line when we can.
        try:
            body = json.loads(e.read().decode("utf-8"))
            msg = body.get("error", {}).get("message") or body.get("error", {}).get("code")
        except Exception:
            msg = None
        raise RuntimeError(msg or f"HTTP {e.code}: {e.reason}")

    articles = []
    for a in data.get("data", []) or []:
        articles.append({
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
            "author": a.get("source"),  # TheNewsAPI gives a source domain, not a byline
            "published": a.get("published_at"),
            "image": a.get("image_url"),
        })
    return articles


def get_news(bank_id: str, bank_name: str, force_refresh: bool = False, days: int | None = None) -> dict:
    """Cached per (bank, days) -- "last 7 days" and "last 30 days" are
    genuinely different result sets, not a client-side slice of one fetch,
    so each range gets its own cache entry and its own call against the
    daily budget below. Refreshed at most every CACHE_TTL_SECONDS per
    entry, hard capped at MAX_CALLS_PER_DAY across ALL banks and ranges
    combined. On a cache miss that fails (budget exhausted, network error,
    bad response), the last good cache for that same range is returned
    with a `note` explaining why it's stale, rather than an error --
    headlines that are a few hours old beat a broken tab.

    The seed (see _load_seed / the module docstring) is checked even when no
    token is configured at all -- a deploy that hasn't been given
    THE_NEWS_API_TOKEN yet still gets real pre-loaded headlines instead of
    a bare "not configured" error, which is the whole point of shipping one."""
    cache_key = (bank_id, days)
    with _lock:
        cached = _cache.get(cache_key)
        if cached is None and not force_refresh:
            seed = _load_seed(bank_id)
            if seed:
                _cache[cache_key] = seed
                cached = seed

    if not is_available():
        if cached:
            return {"available": True, **cached,
                    "note": "no TheNewsAPI token configured -- showing pre-loaded headlines"}
        return {"available": False, "articles": [], "fetched_at": None,
                "error": "no TheNewsAPI token configured (THE_NEWS_API_TOKEN)"}

    with _lock:
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
            articles = _fetch(bank_name, days)
            _cache[cache_key] = {"articles": articles, "fetched_at": time.time(), "error": None}
        except Exception as e:
            if cached:
                return {"available": True, **cached, "note": f"refresh failed ({e}) -- showing the last cached result"}
            _cache[cache_key] = {"articles": [], "fetched_at": time.time(), "error": str(e)}

        return {"available": True, **_cache[cache_key]}
