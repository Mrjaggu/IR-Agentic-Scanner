# Contributing

Thanks for wanting to work on this. This is a small, actively-changing project --
this guide is meant to make it easy to contribute without a lot of ceremony, while
keeping `main` in a state that can deploy at any time (it does -- pushes to `main`
trigger the Vercel deploy in `.github/workflows/`).

## Getting set up

Follow the [Quick Start](README.md#-quick-start) in the README to get the app
running locally first. Come back here once you can open `http://localhost:8000`
and see the dashboard.

## Branching model

We use a simple **one permanent branch, many short-lived ones** model (this is
sometimes called "GitHub Flow"):

- `main` is always the deployable branch. Don't push directly to it for
  anything beyond a trivial fix -- open a PR instead, even for small changes.
- Branch off `main` for every change, named by what it is:
  - `feature/<short-description>` -- a new capability (`feature/cross-bank-quarter-overlay`)
  - `fix/<short-description>` -- a bug fix (`fix/news-refresh-double-fetch`)
  - `docs/<short-description>` -- documentation only
  - `chore/<short-description>` -- dependency bumps, cleanup, no behavior change
- Keep branches short-lived. Open the PR as soon as the change is coherent,
  even in draft -- long-lived branches drift and get painful to merge.
- Delete the branch once its PR is merged.

We don't maintain `develop`, `release`, or per-environment branches. If a
change needs to be tested somewhere before it reaches production, use a PR
preview deployment (Vercel generates one automatically for each PR) rather
than a long-lived staging branch.

## Commit messages

Look at `git log` in this repo for the pattern we use: a short summary line,
then a body that explains **why**, not just what changed. A commit message
that says "Fixed the news feed" is much less useful six months from now than
one that says why it broke and what the fix actually does. If you found the
root cause through some investigation (log output, a failing test, a
reproduction script), it's worth a line or two -- that context is exactly
what the next person debugging a related issue needs.

## Before opening a PR

There's no CI test suite yet (see [ROADMAP.md](ROADMAP.md) if you want to
help change that), so verification is manual. At minimum:

- **Python changes**: run `python3 -m py_compile <changed files>` and
  exercise the actual code path (a route, a CLI command, a script) rather
  than relying on syntax-checking alone.
- **Frontend changes** (`frontend/ir_platform_app.html`): the whole app is
  one file with an inline `<script>` block. After editing, extract and
  syntax-check it:
  ```bash
  python3 -c "
  import re
  html = open('frontend/ir_platform_app.html').read()
  open('/tmp/_check.js', 'w').write(re.search(r'<script>(.*)</script>', html, re.DOTALL).group(1))
  "
  node --check /tmp/_check.js
  ```
  Then actually click through the change in a browser -- a syntax check
  catches typos, not broken behavior.
- **New external dependency (API, package)**: note its free-tier limits (if
  any) in a code comment near where it's called, and make sure a missing
  key/token degrades gracefully instead of crashing the app (see "Degrading
  honestly" below).
- Update `README.md` and/or `ROADMAP.md` if the change adds a feature, a new
  `.env` variable, or changes setup steps. A feature without a README
  mention is easy to lose track of.

## Code conventions worth knowing

A few patterns this codebase leans on consistently -- following them keeps a
PR easy to review:

- **Degrade honestly.** Anything that depends on something outside this
  app's control (an external API, an optional model download, a missing
  key) should never crash the app. Return a normal-looking result with an
  `available` / `error` / `note` field instead of raising, and let the
  frontend render a clear message. See `src/news/news_api.py` or
  `src/tts/piper_tts.py` for the pattern.
- **In-memory caching for short-lived data.** Headlines, TTS output, and
  similar don't need Redis or a database -- a plain in-memory dict with a
  TTL is the existing convention (`_cache` in a few modules). Don't
  introduce a new caching dependency without a real reason.
- **Respect provider budgets.** If you're adding a call to a rate-limited
  external API, add both a cache TTL and a hard daily call cap well under
  the provider's real limit -- see `src/news/news_api.py` for the exact
  shape.
- **`.env` is the single source of truth for secrets/config.** It's loaded
  once, in `run.py`, before anything else imports. Don't read environment
  variables as a substitute for a real shell export elsewhere -- if a module
  needs a new key, add it to `.env.example` too.

## Reporting a bug or proposing a feature

Open a GitHub issue. For a bug, include: what you expected, what happened
instead, and how to reproduce it (a specific bank/quarter/view if relevant).
For a feature, a short description of the problem it solves is more useful
than a fully-specified design -- that's what the PR discussion is for.

If you're not sure whether something fits the project's direction, check
[ROADMAP.md](ROADMAP.md) first, or just open an issue and ask before writing
code.
