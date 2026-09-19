"""
test_news_api.py -- Section: News tab, CurrentsAPI diagnostic script.

Run this directly on a machine with real internet access (NOT through any
sandboxed dev shell -- see src/news/currents_api.py's module docstring for
why that matters here):

    python3 scripts/test_news_api.py

It isolates each moving part one at a time so a failure tells you WHICH
piece is wrong, instead of one opaque "it doesn't work":

  1. Is the key even loaded from .env?
  2. Bare urllib, no custom User-Agent (this is what broke -- CurrentsAPI
     sits behind Cloudflare, which blocks urllib's default
     "Python-urllib/x.y" UA as bot traffic and returns a 403 before your
     key is ever checked).
  3. urllib + a browser User-Agent (the fix now in src/news/currents_api.py).
  4. requests + keywords param (CurrentsAPI's own docs quick-start shape --
     a second, independent way to reach the same conclusion).
  5. The app's actual currents_api.get_news(), exactly as the News feeds
     tab calls it.

Each step prints PASS/FAIL and the raw error so the failure mode is visible
rather than swallowed.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def load_dotenv_key():
    """Minimal .env parse -- avoids requiring python-dotenv just for this script."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return os.environ.get("CURRENT_NEWS_API_KEY")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("CURRENT_NEWS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("CURRENT_NEWS_API_KEY")


def step(n, title):
    print(f"\n--- Step {n}: {title} " + "-" * max(0, 50 - len(title)))


def main():
    key = load_dotenv_key()

    step(1, "key loaded from .env")
    if not key:
        print("FAIL -- no CURRENT_NEWS_API_KEY found in .env or the environment. Stopping here.")
        return
    print(f"PASS -- key found, length {len(key)}, starts with {key[:4]}...")

    step(2, "bare urllib, no custom User-Agent (reproduces the original bug)")
    params = urllib.parse.urlencode({"query": '"Axis Bank"', "language": "en", "page_size": 3})
    url = f"https://api.currentsapi.services/v1/search?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"PASS (unexpected) -- got {len(data.get('news', []))} articles even without a custom UA")
    except urllib.error.HTTPError as e:
        print(f"FAIL -- HTTP Error {e.code}: {e.reason}  (this is the bug we're chasing if it's 403)")
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")

    step(3, "urllib + browser User-Agent (the fix now in currents_api.py)")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "User-Agent": BROWSER_UA,
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print(f"PASS -- got {len(data.get('news', []))} articles")
        if data.get("news"):
            print(f"  sample title: {data['news'][0].get('title')!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"FAIL -- HTTP Error {e.code}: {e.reason}\n  body: {body}")
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")

    step(4, "requests + keywords param (CurrentsAPI's own docs quick-start shape)")
    try:
        import requests
    except ImportError:
        print("SKIP -- 'requests' isn't installed (pip install requests to run this step)")
    else:
        try:
            res = requests.get(
                "https://api.currentsapi.services/v1/search",
                params={"keywords": "Axis Bank", "language": "en", "page_number": 1, "page_size": 3},
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            print(f"HTTP status: {res.status_code}")
            if res.ok:
                data = res.json()
                print(f"PASS -- got {len(data.get('news', []))} articles")
                if data.get("news"):
                    print(f"  sample title: {data['news'][0].get('title')!r}")
            else:
                print(f"FAIL -- body: {res.text[:300]}")
        except Exception as e:
            print(f"FAIL -- {type(e).__name__}: {e}")

    step(5, "the app's own currents_api.get_news(), exactly as News feeds calls it")
    try:
        from src.news import currents_api
        result = currents_api.get_news("axis", "Axis Bank", force_refresh=True)
        print(json.dumps(result, indent=2, default=str)[:1000])
        if result.get("error"):
            print(f"\nFAIL -- {result['error']}")
        else:
            print(f"\nPASS -- {len(result.get('articles', []))} articles, fetched_at={result.get('fetched_at')}")
    except Exception as e:
        print(f"FAIL -- {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Done. If step 2 failed with a 403 and step 3 passed, the")
    print("User-Agent fix is confirmed and nothing further is needed.")
    print("If step 3 ALSO failed, paste this whole output back -- the")
    print("error body usually says exactly what CurrentsAPI didn't like.")


if __name__ == "__main__":
    main()
