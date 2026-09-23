"""
news_signal.py -- opt-in, NEVER-BACKTESTED news-derived topic signal.

Built after wiring analyst sentiment into predictions (see
analyst_layer.sentiment_extra_slot's docstring for that backtest) and
hitting a structural blocker doing the same for news: src.news.news_api is
a live, in-memory cache with a 3-hour TTL plus one manual seed snapshot
(data/news_seed.json, captured 2026-09) -- there is no historical,
per-quarter news archive anywhere in this codebase. eval_harness's holdout
quarters go back to Q1FY22; TheNewsAPI (and the seed) only ever reflect
"right now". There is therefore no way to ask "would this signal have
predicted q2fy24's questions" -- the data a backtest would need for any
past quarter simply doesn't exist and can't be reconstructed after the
fact (news articles are not archived retroactively by the API).

This module exists anyway, at the user's explicit request, as INFRASTRUCTURE
FOR A SIGNAL THAT HAS NEVER BEEN MEASURED TO HELP. It is not promoted to a
default the way a backtested signal would be, and it never will be from
this module alone -- promoting it would require either (a) a genuine
historical news archive, built forward from whenever someone starts logging
it (see the news-history idea considered and declined this session in
favor of building this instead), or (b) some other way to validate it that
doesn't need one. Until then this stays live-only, opt-in, off by default,
called from nowhere in the default prediction path.

Mechanism: reuses the SAME keyword taxonomy analyst_layer.py's disclosure
signal already uses to let a topic enter an analyst's candidate pool (Section
3.3's "when a disclosure is supplied, topics it flags can ENTER this
analyst's candidate pool even if they fell outside the global top-N") --
not the extra-slot mechanism sentiment uses, because news headlines, like a
disclosure, are naturally topic-shaped (unlike a single sentiment scalar),
so they can plausibly point at WHICH topic, not just how sharply. A news
article's title+description is matched against src.graphs.compiler.TOPICS'
keyword lists; each topic's salience is the fraction of this bank's fetched
articles that mention any of its keywords, capped at 1.0. An empty or
unavailable news fetch (no token, budget exhausted, no cache) returns {},
which callers must treat as "no signal", not zero salience for every topic.
"""

from src.graphs.compiler import TOPICS
from src.news import news_api


def news_topic_salience(bank_id: str, bank_name: str, days: int = 7) -> dict[str, float]:
    """{topic: salience in [0, 1]} from this bank's currently-cached/fetched
    news, or {} if nothing is available right now. Always CURRENT -- there
    is no as_of_quarter parameter, unlike every leak-free signal elsewhere
    in this codebase, because there is nothing to be leak-free FROM: this
    reflects whatever news exists at the moment it's called, which is only
    ever "now". Callers must not use this for a holdout/backtest run (that
    would leak today's news into a historical prediction) -- see
    run_agentic.build_initial_state's use_news_signal guard, which refuses
    to compute this when holdout=True."""
    result = news_api.get_news(bank_id, bank_name, days=days)
    articles = result.get("articles") or []
    if not articles:
        return {}

    counts: dict[str, int] = {t: 0 for t in TOPICS}
    for a in articles:
        hay = f"{a.get('title') or ''} {a.get('description') or ''}".lower()
        for topic, keywords in TOPICS.items():
            if any(kw in hay for kw in keywords):
                counts[topic] += 1

    total = len(articles)
    return {t: round(min(c / total, 1.0), 4) for t, c in counts.items() if c > 0}
