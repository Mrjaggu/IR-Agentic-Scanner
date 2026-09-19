"""
test_news_api.py -- Section: News tab, TheNewsAPI diagnostic script.

Run this directly on a machine with real internet access (NOT through any
sandboxed dev shell -- see src/news/news_api.py's module docstring for why
that matters here):

    python3 scripts/test_news_api.py

It isolates each moving part one at a time so a failure tells you WHICH
piece is wrong, instead of one opaque "it doesn't work":

  1. Is the token even loaded from .env?
  2. Bare urllib call to /v1/news/all with the phrase-search shape this
     app actually uses.
  3. requests, same shape -- a second, independent way to reach the same
     conclusion (rules out anything urllib-specific).
  4. The app's actual news_api.get_news(), exactly as the News feeds tab
     calls it.

Each step prints PASS/FAIL and the raw error so the failure mode is visible
rather than swallowed. This replaces the old CurrentsAPI version of this
script -- CurrentsAPI's own auth backend started returning a persistent
503 ("Authentication service temporarily unavailable") that never cleared
across repeated tries, confirmed with this same isolate-each-layer approach,
so the provider was switched to TheNewsAPI instead of chasing that further.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Match run.py's own startup order (see run.py's comment): load .env into
# the real process environment before importing anything that reads a
# token at import time. Without this, step 4 below fails with "no token
# configured" even though steps 1-3 (which parse .env by hand) find it fine.
from dotenv import load_dotenv
load_dotenv(override=False)

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def load_dotenv_token():
    """Minimal .env parse -- avoids depending on load_dotenv() above having
    actually found the file, so step 1 still tells the truth if it didn't."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return os.environ.get("THE_NEWS_API_TOKEN")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("THE_NEWS_API_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("THE_NEWS_API_TOKEN")


def step(n, title):
    print(f"\n--- Step {n}: {title} " + "-" * max(0, 50 - len(title)))


def main():
    token = load_dotenv_token()

    step(1, "token loaded from .env")
    if not token:
        print("FAIL -- no THE_NEWS_API_TOKEN found in .env or the environment. Stopping here.")
        print("Add a line to .env: THE_NEWS_API_TOKEN=<your token from thenewsapi.com>")
        return
    print(f"PASS -- token found, length {len(token)}, starts with {token[:4]}...")

    params = urllib.parse.urlencode({
        "api_token": token,
        "search": '"Axis Bank"',
        "search_fields": "title,description",
        "language": "en",
        "limit": 3,
    })
    url = f"https://api.thenewsapi.com/v1/news/all?{params}"

    step(2, "bare urllib call to /v1/news/all")
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        found = data.get("meta", {}).get("found")
        print(f"PASS -- {len(data.get('data', []))} articles returned, {found} total matches")
        if data.get("data"):
            print(f"  sample title: {data['data'][0].get('title')!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"FAIL -- HTTP Error {e.code}: {e.reason}\n  body: {body}")
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")

    step(3, "requests, same shape (rules out anything urllib-specific)")
    try:
        import requests
    except ImportError:
        print("SKIP -- 'requests' isn't installed (pip install requests to run this step)")
    else:
        try:
            res = requests.get(
                "https://api.thenewsapi.com/v1/news/all",
                params={
                    "api_token": token,
                    "search": '"Axis Bank"',
                    "search_fields": "title,description",
                    "language": "en",
                    "limit": 3,
                },
                timeout=15,
            )
            print(f"HTTP status: {res.status_code}")
            if res.ok:
                data = res.json()
                print(f"PASS -- {len(data.get('data', []))} articles returned")
                if data.get("data"):
                    print(f"  sample title: {data['data'][0].get('title')!r}")
            else:
                print(f"FAIL -- body: {res.text[:300]}")
        except Exception as e:
            print(f"FAIL -- {type(e).__name__}: {e}")

    step(4, "the app's own news_api.get_news(), exactly as News feeds calls it")
    try:
        from src.news import news_api
        result = news_api.get_news("axis", "Axis Bank", force_refresh=True)
        print(json.dumps(result, indent=2, default=str)[:1000])
        if result.get("error"):
            print(f"\nFAIL -- {result['error']}")
        else:
            print(f"\nPASS -- {len(result.get('articles', []))} articles, fetched_at={result.get('fetched_at')}")
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Done. If steps 2-4 all PASS, News feeds is working end to end.")
    print("If any FAIL, paste this whole output back -- the error body")
    print("usually says exactly what TheNewsAPI didn't like.")


if __name__ == "__main__":
    main()
