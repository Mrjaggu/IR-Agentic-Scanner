import os
import json
import urllib.request
import urllib.error
import time

from src.config.settings import BASE_DIR

_GROQ_MAX_RETRIES = 5
_GROQ_BASE_WAIT   = 10   # seconds for first retry (doubles each attempt)

# ── Published free-tier limits for the model we call ────────────────────────
# Groq free tier, openai/gpt-oss-120b: 8,000 tokens/min, 30 req/min, 1,000
# req/day. TPM is the binding constraint here, not RPM: a Question-Framer
# prompt carries evidence passages, so it runs ~1.5-2k tokens a call, which
# means roughly 4-5 calls a minute before throttling. A full pipeline run over
# ~25 analysts therefore takes minutes, not seconds, and that needs to be
# visible in the UI rather than looking like a hang.
RATE_LIMITS = {
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "tier": "free",
    "tpm": 8000,
    "rpm": 30,
    "rpd": 1000,
    "approx_tokens_per_call": 1800,
}

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


def stats() -> dict:
    """Snapshot of LLM activity, plus a derived 'throttled' flag the UI uses to
    explain a slow run instead of leaving the user staring at a spinner."""
    out = dict(STATS)
    out["limits"] = RATE_LIMITS
    out["throttled"] = STATS["rate_limited"] > 0
    if STATS["started_at"]:
        out["elapsed_seconds"] = round(time.time() - STATS["started_at"], 1)
    return out


def reset_stats():
    STATS.update({"calls": 0, "ok": 0, "failed": 0, "rate_limited": 0,
                  "wait_seconds": 0.0, "last_error": None, "started_at": time.time()})

class LLMClient:
    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.preferred_provider = os.getenv("MODEL_PROVIDER", "").upper()
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.active_llm = None
        
    def probe_llm(self):
        """Probes the APIs to find which key actually works."""
        # 1. Respect explicit MODEL_PROVIDER preference if valid key present
        if self.preferred_provider == "GROQ" and self.groq_key:
            res = self.call_groq('Return valid JSON: {"ok": true}', temperature=0.0)
            if res:
                self.active_llm = "Groq"
                return "Groq"
        elif self.preferred_provider == "GEMINI" and self.gemini_key:
            res = self.call_gemini('Return valid JSON: {"ok": true}', temperature=0.0)
            if res:
                self.active_llm = "Gemini"
                return "Gemini"
        elif self.preferred_provider == "OPENAI" and self.openai_key:
            res = self.call_openai('Return valid JSON: {"ok": true}', temperature=0.0)
            if res:
                self.active_llm = "OpenAI"
                return "OpenAI"

        # 2. Default fallback order if preferred provider not set or failed probe
        if self.groq_key:
            res = self.call_groq('Return valid JSON: {"ok": true}', temperature=0.0)
            if res:
                self.active_llm = "Groq"
                return "Groq"
            else:
                print("⚠  Groq key present but API call failed (403 = Cloudflare block; 429 = rate limit).")
                
        if self.gemini_key:
            res = self.call_gemini('Return valid JSON: {"ok": true}', temperature=0.0)
            if res:
                self.active_llm = "Gemini"
                return "Gemini"
            else:
                print("⚠  Gemini key present but API call failed — check key validity.")
                
        if self.openai_key:
            self.active_llm = "OpenAI"
            return "OpenAI"
            
        return None

    def call_groq(self, prompt: str, temperature: float = 0.15, system_prompt: str = None) -> str | None:
        if not self.groq_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        sys_msg = system_prompt or (
            "You are a financial IR assistant helping Axis Bank prepare for analyst "
            "earnings calls. Return ONLY valid JSON with the exact schema requested. "
            "Do NOT include markdown code fences."
        )
        
        body = json.dumps({
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }).encode()

        STATS["calls"] += 1
        if STATS["started_at"] is None:
            STATS["started_at"] = time.time()

        for attempt in range(_GROQ_MAX_RETRIES):
            req = urllib.request.Request(
                url, data=body,
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "python-requests/2.31.0",
                },
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    STATS["ok"] += 1
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    STATS["rate_limited"] += 1
                    wait = None
                    try:
                        retry_after = e.headers.get("retry-after") or e.headers.get("Retry-After")
                        if retry_after:
                            wait = float(retry_after)
                    except Exception:
                        pass
                    if wait is None:
                        wait = _GROQ_BASE_WAIT * (2 ** attempt)
                    STATS["wait_seconds"] += wait
                    STATS["last_error"] = (f"rate limited (429) — waiting {wait:.0f}s "
                                           f"[{RATE_LIMITS['tpm']} TPM free tier]")
                    print(f"  [Groq TPM limit] Waiting {wait:.0f}s before retry "
                          f"(attempt {attempt+1}/{_GROQ_MAX_RETRIES}) ...")
                    time.sleep(wait)
                    continue
                if e.code == 403:
                    try:
                        err_body = e.read().decode("utf-8", errors="ignore")
                    except Exception:
                        err_body = str(e)
                    STATS["failed"] += 1
                    STATS["last_error"] = f"403 (blocked or bad key): {err_body[:120]}"
                    print(f"  [Groq 403 — Cloudflare block or bad key, skipping Groq] {err_body[:120]}")
                    return None
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = str(e)
                STATS["failed"] += 1
                STATS["last_error"] = f"HTTP {e.code}: {err_body[:160]}"
                print(f"  [Groq error {e.code}] {err_body[:200]}")
                return None
            except Exception as e:
                STATS["failed"] += 1
                STATS["last_error"] = str(e)[:160]
                print(f"  [Groq error] {e}")
                return None

        STATS["failed"] += 1
        STATS["last_error"] = (f"gave up after {_GROQ_MAX_RETRIES} retries against the "
                               f"{RATE_LIMITS['tpm']} TPM free-tier limit")
        print(f"  [Groq] Gave up after {_GROQ_MAX_RETRIES} retries.")
        return None

    def call_gemini(self, prompt: str, temperature: float = 0.15) -> str | None:
        if not self.gemini_key:
            return None
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.gemini_model}:generateContent?key={self.gemini_key}")
        data = json.dumps({
            "contents": [{"parts": [{"text": prompt + "\n\nReturn ONLY valid JSON."}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": temperature}
        }).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"  [Gemini error] {e}")
            return None

    def call_openai(self, prompt: str, temperature: float = 0.15) -> str | None:
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
        if self.active_llm == "Gemini":
            return self.call_gemini(prompt, temperature=temperature)
        if self.active_llm == "Groq":
            return self.call_groq(prompt, temperature=temperature, system_prompt=system_prompt)
        if self.active_llm == "OpenAI":
            return self.call_openai(prompt, temperature=temperature)
            
        # Fallback order if not probed/set
        if self.groq_key:
            return self.call_groq(prompt, temperature=temperature, system_prompt=system_prompt)
        if self.gemini_key:
            return self.call_gemini(prompt, temperature=temperature)
        if self.openai_key:
            return self.call_openai(prompt, temperature=temperature)
        return None

# Singleton instance
client = LLMClient()
