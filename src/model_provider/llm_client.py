import os
import json
import urllib.request
import urllib.error
import time

from src.config.settings import BASE_DIR

_MAX_RETRIES = 5
_BASE_WAIT   = 10   # seconds for first retry (doubles each attempt)

# Gemini free tier (per user, 2026-09): 15 requests/min. Rather than only
# reacting to 429s after the fact (which still burns a request and, per the
# 6-analyst run, an attempt's worth of latency even with the retry loop
# above), self-pace calls to stay under a buffered ceiling -- 12/min, not 15
# -- so a burst of framing/grounding calls for one analyst doesn't trip the
# limit in the first place. Sliding 60s window of call timestamps.
_GEMINI_SAFE_RPM = 12
_gemini_call_times: list[float] = []


def _gemini_pace() -> None:
    """Block, if needed, so this call stays within _GEMINI_SAFE_RPM over a
    trailing 60s window. Called once per attempt, right before each Gemini
    HTTP request (including retries, so a retry doesn't itself re-trip the
    limit)."""
    now = time.time()
    while _gemini_call_times and now - _gemini_call_times[0] > 60:
        _gemini_call_times.pop(0)
    if len(_gemini_call_times) >= _GEMINI_SAFE_RPM:
        wait = 60 - (now - _gemini_call_times[0]) + 0.5
        if wait > 0:
            print(f"  [Gemini pacing] {_GEMINI_SAFE_RPM}/min buffer reached -- "
                  f"waiting {wait:.1f}s before next call.")
            time.sleep(wait)
        now = time.time()
        while _gemini_call_times and now - _gemini_call_times[0] > 60:
            _gemini_call_times.pop(0)
    _gemini_call_times.append(time.time())

# ── Published free-tier limits per provider ─────────────────────────────────
# Groq free tier, openai/gpt-oss-120b: 8,000 tokens/min, 30 req/min, 1,000
# req/day. TPM was the binding constraint there -- a Question-Framer prompt
# carries evidence passages and runs ~1.5-2k tokens a call, so only ~4-5 calls
# fit in a minute before throttling, which is why Cerebras/OpenRouter were
# investigated as higher-TPM alternatives (2026-09).
#
# Cerebras: DROPPED from the active rotation (2026-09) -- despite "free trial"
# framing in its own docs, actually using it required a payment method on
# file, which defeats the point of a free-tier fallback here. The
# call_cerebras() code path is left in (harmless, no-ops without a key) in
# case that changes, but it is now last in the fallback order, after Groq.
#
# OpenRouter free tier (":free"-suffixed model slugs): 20 RPM, 50 requests/day
# (1,000/day once $10+ in lifetime credit has been purchased on the account,
# even if none of it is spent). No published TPM ceiling -- it is governed by
# whichever free model you route to, which OpenRouter can also silently
# deprecate/replace, so OPENROUTER_MODEL should be checked against
# https://openrouter.ai/models?max_price=0 periodically rather than assumed
# stable. Source: OpenRouter docs, Sept 2026.
#
# NVIDIA NIM (build.nvidia.com hosted inference API, added 2026-09): free API
# key, OpenAI-compatible /v1/chat/completions endpoint. NVIDIA does not
# publish a fixed TPM/RPM number for the free tier the way Groq/Cerebras do --
# treat it as "free but unpublished/variable limits" rather than a known
# ceiling, and let the existing 429 backoff handle whatever throttling shows
# up in practice.
PROVIDER_LIMITS = {
    "Groq": {"provider": "groq", "tier": "free", "tpm": 8000, "rpm": 30, "rpd": 1000,
             "approx_tokens_per_call": 1800},
    "Cerebras": {"provider": "cerebras", "tier": "requires_payment_method", "tpm": 30000,
                 "rpm": 5, "rpd": None, "tph_tpd": 1_000_000, "approx_tokens_per_call": 1800,
                 "note": "Despite 'free trial' framing in Cerebras's own docs, actually "
                         "using it required a payment method on file (confirmed 2026-09) "
                         "-- not a no-payment free tier. Dropped from the active fallback "
                         "rotation; kept here only in case that changes."},
    "OpenRouter": {"provider": "openrouter", "tier": "free", "tpm": None, "rpm": 20,
                   "rpd": 50, "rpd_with_credit": 1000, "approx_tokens_per_call": 1800,
                   "note": "TPM not published -- governed by whichever :free model is "
                           "configured, and that model can be deprecated/replaced by "
                           "OpenRouter without notice."},
    "NVIDIA NIM": {"provider": "nvidia_nim", "tier": "free", "tpm": None, "rpm": None,
                   "rpd": None, "approx_tokens_per_call": 1800,
                   "note": "NVIDIA does not publish fixed free-tier TPM/RPM numbers -- "
                           "limits are unconfirmed/variable, handled via 429 backoff."},
    "Gemini": {"provider": "gemini", "tier": "free", "tpm": None, "rpm": 15, "rpd": None,
               "note": "15 req/min published free-tier ceiling (per user, 2026-09) -- "
                       "call_gemini self-paces to 12/min via _gemini_pace() rather than "
                       "only reacting to 429s after the fact."},
    "OpenAI": {"provider": "openai", "tier": "paid", "tpm": None, "rpm": None, "rpd": None},
}

# Backward-compatible alias -- some call sites imported this name directly
# when Groq was the only provider. Kept pointing at Groq's numbers so nothing
# that formats "the free tier" in an error string breaks; prefer current_limits()
# for anything that should reflect whichever provider is actually active.
RATE_LIMITS = PROVIDER_LIMITS["Groq"]

# Lightweight in-process telemetry so the dashboard can show what a run is
# actually doing while it waits. Not persisted; reset at the start of each run.
STATS = {
    "calls": 0,            # attempts that reached a provider
    "ok": 0,
    "failed": 0,
    "rate_limited": 0,     # 429s observed
    "wait_seconds": 0.0,   # cumulative time spent backing off
    "last_error": None,
    "started_at": None,
}


def current_limits(client_obj: "LLMClient | None" = None) -> dict:
    """Limits for whichever provider is actually active (falls back to Groq's
    numbers, matching RATE_LIMITS, if nothing has been probed yet)."""
    c = client_obj if client_obj is not None else client
    base = dict(PROVIDER_LIMITS.get(c.active_llm, PROVIDER_LIMITS["Groq"]))
    model = {"Groq": c.groq_model, "Cerebras": c.cerebras_model, "OpenRouter": c.openrouter_model,
             "NVIDIA NIM": c.nim_model, "Gemini": c.gemini_model, "OpenAI": c.openai_model}.get(c.active_llm)
    if model:
        base["model"] = model
    return base


def stats() -> dict:
    """Snapshot of LLM activity, plus a derived 'throttled' flag the UI uses to
    explain a slow run instead of leaving the user staring at a spinner."""
    out = dict(STATS)
    out["limits"] = current_limits()
    out["throttled"] = STATS["rate_limited"] > 0
    if STATS["started_at"]:
        out["elapsed_seconds"] = round(time.time() - STATS["started_at"], 1)
    return out


def reset_stats():
    STATS.update({"calls": 0, "ok": 0, "failed": 0, "rate_limited": 0,
                  "wait_seconds": 0.0, "last_error": None, "started_at": time.time()})


# 2026-09: genericized off "Axis Bank" -- this platform now serves multiple
# banks (see src.config.banks). Each individual prompt that needs to name a
# specific bank does so itself (build_overall_topics/question_framer.py both
# take a bank_name param); this system-level prompt only needs to state the
# assistant's general role and output-format contract, which is bank-agnostic.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a financial IR assistant helping a bank's investor relations team "
    "prepare for analyst earnings calls. Return ONLY valid JSON with the exact "
    "schema requested. Do NOT include markdown code fences."
)


class LLMClient:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.cerebras_key = os.getenv("CEREBRAS_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.nim_key = os.getenv("NVIDIA_NIM_API_KEY")
        self.preferred_provider = os.getenv("MODEL_PROVIDER", "").upper()

        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        # Cerebras uses the bare model name (no "openai/" prefix) for the same
        # gpt-oss-120b family Groq serves -- confirmed against Cerebras' own
        # models-overview docs, Sept 2026.
        self.cerebras_model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
        # OpenRouter's free-model roster rotates; there is no single slug
        # that has stayed free and available for long, so this only has a
        # placeholder default -- set OPENROUTER_MODEL explicitly to a current
        # ":free" slug from https://openrouter.ai/models?max_price=0.
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.nim_model = os.getenv("NVIDIA_NIM_MODEL", "")

        self.active_llm = None

    # ── Probing ──────────────────────────────────────────────────────────────
    def probe_llm(self):
        """Probes the APIs to find which key actually works. Tries the
        explicit MODEL_PROVIDER preference first (if its key is present), then
        falls back through providers in an order biased toward higher-TPM free
        options before Groq's tighter 8,000 TPM ceiling."""
        order = ["OPENROUTER", "GROQ", "NVIDIA_NIM", "CEREBRAS", "GEMINI", "OPENAI"]
        callers = {
            "GROQ": (self.groq_key, self.call_groq, "Groq"),
            "CEREBRAS": (self.cerebras_key, self.call_cerebras, "Cerebras"),
            "OPENROUTER": (self.openrouter_key, self.call_openrouter, "OpenRouter"),
            "NVIDIA_NIM": (self.nim_key, self.call_nvidia_nim, "NVIDIA NIM"),
            "GEMINI": (self.gemini_key, self.call_gemini, "Gemini"),
            "OPENAI": (self.openai_key, self.call_openai, "OpenAI"),
        }

        # 1. Respect an explicit MODEL_PROVIDER preference if its key is present.
        if self.preferred_provider in callers:
            key, fn, label = callers[self.preferred_provider]
            if key:
                res = fn('Return valid JSON: {"ok": true}', temperature=0.0, no_retry=True)
                if res:
                    self.active_llm = label
                    return label
                print(f"⚠  {label} key present but preferred-provider probe failed -- "
                      f"falling back to the default order.")

        # 2. Default fallback order: higher-TPM free options before Groq,
        #    Cerebras/OpenRouter before the un-rate-limited-but-paid OpenAI.
        for name in order:
            key, fn, label = callers[name]
            if not key:
                continue
            res = fn('Return valid JSON: {"ok": true}', temperature=0.0, no_retry=True)
            if res:
                self.active_llm = label
                return label
            print(f"⚠  {label} key present but API call failed "
                  f"(403 = blocked/bad key; 429 = rate limit; check the model name too).")

        return None

    # ── Shared HTTP + retry/backoff for OpenAI-compatible chat endpoints ────
    def _chat_completions(self, url: str, api_key: str, model: str, prompt: str,
                          temperature: float, system_prompt: str | None,
                          provider_label: str, tpm_hint: int | None,
                          extra_headers: dict | None = None,
                          json_mode: bool = True,
                          no_retry: bool = False) -> str | None:
        """POSTs an OpenAI-style /chat/completions request with 429 backoff and
        a one-shot fallback that drops response_format if the model rejects
        strict JSON mode (some OpenRouter free models don't support it).

        no_retry=True skips the sleep-and-retry loop entirely: a 429 fails
        immediately instead of working through the full exponential backoff
        (up to 10+20+40+80+160=310s). This exists for probe_llm() -- probing
        is choosing WHICH provider to use, so a rate-limited provider should
        just lose that race and let the next provider in the order get tried,
        not block server startup for minutes on a provider it may not even
        end up using. Real question-framing/grounding calls still retry
        normally (no_retry defaults to False)."""
        sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT

        def _body(with_json_mode: bool) -> bytes:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if with_json_mode:
                payload["response_format"] = {"type": "json_object"}
            return json.dumps(payload).encode()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "python-requests/2.31.0",
        }
        headers.update(extra_headers or {})

        STATS["calls"] += 1
        if STATS["started_at"] is None:
            STATS["started_at"] = time.time()

        use_json_mode = json_mode
        for attempt in range(_MAX_RETRIES):
            req = urllib.request.Request(url, data=_body(use_json_mode), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    STATS["ok"] += 1
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    STATS["rate_limited"] += 1
                    if no_retry:
                        STATS["last_error"] = f"{provider_label}: rate limited (429) -- skipped (no_retry probe)"
                        print(f"  [{provider_label} rate limit] Skipping (probe, no retry).")
                        return None
                    wait = None
                    try:
                        retry_after = e.headers.get("retry-after") or e.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                    except Exception:
                        pass
                    if wait is None:
                        wait = _BASE_WAIT * (2 ** attempt)
                    STATS["wait_seconds"] += wait
                    tpm_note = f" [{tpm_hint} TPM free tier]" if tpm_hint else ""
                    STATS["last_error"] = f"{provider_label}: rate limited (429) -- waiting {wait:.0f}s{tpm_note}"
                    print(f"  [{provider_label} rate limit] Waiting {wait:.0f}s before retry "
                          f"(attempt {attempt+1}/{_MAX_RETRIES}) ...")
                    time.sleep(wait)
                    continue
                if e.code == 400 and use_json_mode:
                    # Some free models 400 on response_format -- retry once
                    # without it rather than giving up on the whole provider.
                    use_json_mode = False
                    print(f"  [{provider_label}] 400 with response_format=json_object -- "
                          f"retrying once without strict JSON mode.")
                    continue
                if e.code == 403:
                    try:
                        err_body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        err_body = str(e)
                    STATS["failed"] += 1
                    STATS["last_error"] = f"{provider_label}: 403 (blocked or bad key): {err_body[:120]}"
                    print(f"  [{provider_label} 403 -- blocked or bad key, skipping] {err_body[:120]}")
                    return None
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = str(e)
                STATS["failed"] += 1
                STATS["last_error"] = f"{provider_label}: HTTP {e.code}: {err_body[:160]}"
                print(f"  [{provider_label} error {e.code}] {err_body[:200]}")
                return None
            except Exception as e:
                STATS["failed"] += 1
                STATS["last_error"] = f"{provider_label}: {str(e)[:160]}"
                print(f"  [{provider_label} error] {e}")
                return None

        STATS["failed"] += 1
        STATS["last_error"] = f"{provider_label}: gave up after {_MAX_RETRIES} retries"
        print(f"  [{provider_label}] Gave up after {_MAX_RETRIES} retries.")
        return None

    # ── Providers ────────────────────────────────────────────────────────────
    def call_groq(self, prompt: str, temperature: float = 0.15, system_prompt: str = None,
                  no_retry: bool = False) -> str | None:
        if not self.groq_key:
            return None
        return self._chat_completions(
            "https://api.groq.com/openai/v1/chat/completions", self.groq_key, self.groq_model,
            prompt, temperature, system_prompt, "Groq", PROVIDER_LIMITS["Groq"]["tpm"],
            no_retry=no_retry,
        )

    def call_cerebras(self, prompt: str, temperature: float = 0.15, system_prompt: str = None,
                      no_retry: bool = False) -> str | None:
        if not self.cerebras_key:
            return None
        return self._chat_completions(
            "https://api.cerebras.ai/v1/chat/completions", self.cerebras_key, self.cerebras_model,
            prompt, temperature, system_prompt, "Cerebras", PROVIDER_LIMITS["Cerebras"]["tpm"],
            no_retry=no_retry,
        )

    def call_openrouter(self, prompt: str, temperature: float = 0.15, system_prompt: str = None,
                        no_retry: bool = False) -> str | None:
        if not self.openrouter_key or not self.openrouter_model:
            if self.openrouter_key and not self.openrouter_model:
                print("  [OpenRouter] OPENROUTER_API_KEY is set but OPENROUTER_MODEL is not -- "
                      "pick a current ':free' slug from https://openrouter.ai/models?max_price=0 "
                      "and set it in .env.")
            return None
        return self._chat_completions(
            "https://openrouter.ai/api/v1/chat/completions", self.openrouter_key, self.openrouter_model,
            prompt, temperature, system_prompt, "OpenRouter", PROVIDER_LIMITS["OpenRouter"]["tpm"],
            # OpenRouter asks for these but works without them; harmless to include.
            extra_headers={"HTTP-Referer": "https://axisbank-ir-platform.local",
                           "X-Title": "IR Question Intelligence"},
            no_retry=no_retry,
        )

    def call_nvidia_nim(self, prompt: str, temperature: float = 0.15, system_prompt: str = None,
                        no_retry: bool = False) -> str | None:
        if not self.nim_key or not self.nim_model:
            if self.nim_key and not self.nim_model:
                print("  [NVIDIA NIM] NVIDIA_NIM_API_KEY is set but NVIDIA_NIM_MODEL is not -- "
                      "set it in .env to a model slug from https://build.nvidia.com/models.")
            return None
        return self._chat_completions(
            "https://integrate.api.nvidia.com/v1/chat/completions", self.nim_key, self.nim_model,
            prompt, temperature, system_prompt, "NVIDIA NIM", PROVIDER_LIMITS["NVIDIA NIM"]["tpm"],
            no_retry=no_retry,
        )

    def call_gemini(self, prompt: str, temperature: float = 0.15, system_prompt: str = None,
                    no_retry: bool = False) -> str | None:
        # no_retry accepted for signature parity with the other providers'
        # call_* methods (see probe_llm) -- this method never retries anyway.
        #
        # 2026-09 fix: a live 6-analyst run against Gemini produced grounding_rate
        # 0.0 -- every framed question failed the Verifier's JSON parse, while a
        # bare {"ok": true} probe had worked fine. Two real gaps caused it:
        #   1. system_prompt was silently dropped -- every OTHER provider gets
        #      _DEFAULT_SYSTEM_PROMPT (or an override) via _chat_completions, but
        #      this method didn't even accept the parameter, so Gemini never saw
        #      the "return ONLY valid JSON, no markdown fences" instruction that
        #      the other providers' system role carries. Now passed via Gemini's
        #      own systemInstruction field (a sibling of `contents`, not another
        #      message in it -- that's the v1beta shape, distinct from the
        #      OpenAI-style chat array the other providers use).
        #   2. No maxOutputTokens was set. question_framer's prompt asks for
        #      several questions back in one JSON object; on Gemini's default
        #      output budget for a "flash-lite" tier model that can truncate
        #      before the closing brace, producing unparseable JSON with no
        #      error raised at the HTTP layer (200 OK, just cut off) -- silent
        #      truncation, not a Gemini bug, but this call site needs to ask for
        #      enough headroom explicitly. 4096 is generous for a handful of
        #      short question_text fields.
        # Also: this call path previously never touched STATS, so a Gemini run
        # was invisible to /api/llm/stats and the dashboard's throttled banner --
        # now bookkept the same way _chat_completions does, and a truncated or
        # safety-blocked response is now a distinguishable, logged failure
        # instead of a bare KeyError caught by the generic except clause.
        if not self.gemini_key:
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.gemini_model}:generateContent?key={self.gemini_key}")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": temperature,
                                 "maxOutputTokens": 4096},
        }
        sys_msg = system_prompt or _DEFAULT_SYSTEM_PROMPT
        payload["systemInstruction"] = {"parts": [{"text": sys_msg}]}
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        STATS["calls"] += 1
        if STATS["started_at"] is None:
            STATS["started_at"] = time.time()

        # 2026-09 fix #2: the 6-analyst wider run exposed that this method
        # never retried a 429 at all (unlike _chat_completions, which the
        # other four providers share) -- Gemini's free tier throttles well
        # before 6 analysts' worth of framing calls finish, so grounding_rate
        # collapsed to 0.146 (17/44 calls rate-limited, all lost for good).
        # Now mirrors _chat_completions' backoff: real calls (no_retry=False,
        # the default for question-framing/grounding) retry up to
        # _MAX_RETRIES times with exponential backoff (or the server's
        # Retry-After header if present); no_retry=True (probe_llm only)
        # still fails a 429 instantly, same as before.
        body = None
        for attempt in range(_MAX_RETRIES):
            _gemini_pace()
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    STATS["rate_limited"] += 1
                    if no_retry:
                        STATS["last_error"] = "Gemini: rate limited (429) -- skipped (no_retry probe)"
                        print("  [Gemini rate limit] 429 -- skipped (no_retry probe).")
                        return None
                    wait = None
                    try:
                        retry_after = e.headers.get("retry-after") or e.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                    except Exception:
                        pass
                    if wait is None:
                        wait = _BASE_WAIT * (2 ** attempt)
                    STATS["wait_seconds"] += wait
                    STATS["last_error"] = f"Gemini: rate limited (429) -- waiting {wait:.0f}s"
                    print(f"  [Gemini rate limit] Waiting {wait:.0f}s before retry "
                          f"(attempt {attempt+1}/{_MAX_RETRIES}) ...")
                    time.sleep(wait)
                    continue
                else:
                    try:
                        err_body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        err_body = str(e)
                    STATS["failed"] += 1
                    STATS["last_error"] = f"Gemini: HTTP {e.code}: {err_body[:160]}"
                    print(f"  [Gemini error {e.code}] {err_body[:200]}")
                    return None
            except Exception as e:
                STATS["failed"] += 1
                STATS["last_error"] = f"Gemini: {str(e)[:160]}"
                print(f"  [Gemini error] {e}")
                return None
        if body is None:
            STATS["failed"] += 1
            STATS["last_error"] = f"Gemini: gave up after {_MAX_RETRIES} retries"
            print(f"  [Gemini] Gave up after {_MAX_RETRIES} retries.")
            return None

        candidates = body.get("candidates") or []
        if not candidates:
            block_reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            STATS["failed"] += 1
            STATS["last_error"] = f"Gemini: {block_reason}"
            print(f"  [Gemini error] {block_reason} -- full response: {json.dumps(body)[:300]}")
            return None
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        if not parts:
            finish_reason = cand.get("finishReason", "unknown")
            STATS["failed"] += 1
            STATS["last_error"] = f"Gemini: empty response (finishReason={finish_reason})"
            print(f"  [Gemini error] empty content, finishReason={finish_reason} "
                 f"-- likely truncated (raise maxOutputTokens) or safety-blocked.")
            return None
        if cand.get("finishReason") == "MAX_TOKENS":
            print("  [Gemini warning] response hit maxOutputTokens and may be truncated/unparseable.")
        STATS["ok"] += 1
        return parts[0].get("text")

    def call_openai(self, prompt: str, temperature: float = 0.15, no_retry: bool = False) -> str | None:
        # no_retry accepted for signature parity with the other providers'
        # call_* methods (see probe_llm) -- this method never retries anyway.
        if not self.openai_key:
            return None
        url = "https://api.openai.com/v1/chat/completions"
        data = json.dumps({
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": "You are a financial IR assistant. Return JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Authorization": f"Bearer {self.openai_key}",
                     "Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [OpenAI error] {e}")
            return None

    def call_llm(self, prompt: str, temperature: float = 0.15, system_prompt: str = None) -> str | None:
        dispatch = {
            "Groq": lambda: self.call_groq(prompt, temperature=temperature, system_prompt=system_prompt),
            "Cerebras": lambda: self.call_cerebras(prompt, temperature=temperature, system_prompt=system_prompt),
            "OpenRouter": lambda: self.call_openrouter(prompt, temperature=temperature, system_prompt=system_prompt),
            "NVIDIA NIM": lambda: self.call_nvidia_nim(prompt, temperature=temperature, system_prompt=system_prompt),
            "Gemini": lambda: self.call_gemini(prompt, temperature=temperature),
            "OpenAI": lambda: self.call_openai(prompt, temperature=temperature),
        }
        if self.active_llm in dispatch:
            return dispatch[self.active_llm]()

        # Fallback order if not probed/set -- same higher-TPM-first bias as probe_llm.
        for label in ("OpenRouter", "Groq", "NVIDIA NIM", "Cerebras", "Gemini", "OpenAI"):
            key = {"Groq": self.groq_key, "Cerebras": self.cerebras_key,
                   "OpenRouter": self.openrouter_key, "NVIDIA NIM": self.nim_key,
                   "Gemini": self.gemini_key, "OpenAI": self.openai_key}[label]
            if key:
                return dispatch[label]()
        return None

# Singleton instance
client = LLMClient()
