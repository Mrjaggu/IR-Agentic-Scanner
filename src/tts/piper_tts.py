"""
piper_tts.py -- Section: chat "Listen" button, TTS synthesis.

Same reasoning as everywhere else in this platform that touches models
(src.model_provider.llm_client, src.search.embeddings): prefer a small,
fully offline, CPU-only open model over an API, because this is built for a
regulated, potentially air-gapped deployment. Piper (github.com/rhasspy/piper)
is a neural TTS engine whose English voices run 20-65MB as ONNX models,
synthesize on CPU near real-time, and need no network once installed.

The voice model is NOT checked into this repo -- binary model weights don't
belong in git. Two ways it can end up on disk before is_available() is ever
asked:

  1. Manual, for local/on-prem/air-gapped use (unchanged from before):

         pip install piper-tts
         python -m piper.download_voices en_US-lessac-medium
         # or a smaller one: en_US-amy-low (~25MB) / en_US-ryan-low (~25MB)

     That downloads <voice>.onnx + <voice>.onnx.json into PIPER_VOICE_DIR
     (defaults to data/tts_voices/ in this repo).

  2. Auto-download, for a deployment like Vercel where nobody can run step 1
     against the deployed filesystem and the repo's own git history is the
     only thing that ships. Compute is not the constraint there (Piper is a
     CPU, sub-100ms-per-call model) -- the model FILES just never arrive. So
     if the manual path's files aren't found, is_available() makes one
     best-effort attempt to fetch the same public files from the same
     Hugging Face-hosted source `piper.download_voices` itself uses, into a
     writable temp directory (Vercel's function filesystem is read-only
     outside /tmp), and caches the outcome for this process's lifetime --
     one real download on a cold start, instant on every request after
     that. Opt out with PIPER_AUTO_DOWNLOAD=0 for a strict "never touch the
     network for this" deployment.

Either way, until real model files exist somewhere reachable, is_available()
is False and every caller degrades honestly -- same discipline as
pattern_retrieval.is_available() / embeddings.is_available(): the "Listen"
button simply doesn't render rather than erroring.
"""

import io
import os
import re
import shutil
import tempfile
import threading
import urllib.request
import wave

from src.config.settings import BASE_DIR

VOICE_DIR = os.environ.get("PIPER_VOICE_DIR", os.path.join(BASE_DIR, "data", "tts_voices"))
VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")

# Auto-download destination -- deliberately NOT VOICE_DIR. A deployment
# filesystem (Vercel's included) is commonly read-only outside a temp dir,
# and even where VOICE_DIR itself is writable, keeping the auto-fetched copy
# separate means a manual install a user does later is never silently
# shadowed by (or clobbered by) something this module fetched on its own.
_AUTO_VOICE_DIR = os.path.join(tempfile.gettempdir(), "piper_voices_auto")

# The same source + path scheme piper.download_voices (this app's own
# piper-tts dependency) uses -- see its source for the reference
# implementation this mirrors.
_HF_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{voice_name}/{voice_quality}/"
    "{lang_code}-{voice_name}-{voice_quality}{extension}?download=true"
)
_VOICE_NAME_RE = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)

_voice = None
_load_failed = False
_lock = threading.Lock()          # guards the PiperVoice singleton load only
_download_lock = threading.Lock()  # guards the auto-download attempt only --
                                    # kept separate from _lock because
                                    # _get_voice() holds _lock while calling
                                    # is_available(), which can reach the
                                    # download path; re-entering the same
                                    # non-reentrant Lock would deadlock.
_download_attempted = False
_download_ok = False


def _model_paths(base_dir: str, voice_name: str = VOICE_NAME) -> tuple[str, str]:
    onnx = os.path.join(base_dir, f"{voice_name}.onnx")
    cfg = os.path.join(base_dir, f"{voice_name}.onnx.json")
    return onnx, cfg


def _dir_has_voice(base_dir: str) -> bool:
    onnx, cfg = _model_paths(base_dir)
    return os.path.exists(onnx) and os.path.exists(cfg)


def _existing_voice_dir() -> str | None:
    """The manually-installed dir first (real local/on-prem installs should
    never be shadowed by an auto-fetched copy), then wherever a previous
    auto-download in this process already landed."""
    for d in (VOICE_DIR, _AUTO_VOICE_DIR):
        if _dir_has_voice(d):
            return d
    return None


def _download_voice_to(base_dir: str) -> bool:
    """One real attempt to fetch VOICE_NAME's two files from the public
    Hugging Face source into `base_dir`. Best-effort: any failure (network,
    timeout, disk) is swallowed and reported False, same as everywhere else
    in this module -- an auto-download that can crash the app would be
    worse than no Listen button."""
    match = _VOICE_NAME_RE.match(VOICE_NAME)
    if not match:
        print(f"  [piper_tts] PIPER_VOICE={VOICE_NAME!r} doesn't match "
              f"<lang>_<REGION>-<name>-<quality> -- can't auto-download it")
        return False
    fmt = {
        "lang_family": match.group("lang_family"),
        "lang_code": f"{match.group('lang_family')}_{match.group('lang_region')}",
        "voice_name": match.group("voice_name"),
        "voice_quality": match.group("voice_quality"),
    }
    try:
        os.makedirs(base_dir, exist_ok=True)
        onnx, cfg = _model_paths(base_dir)
        for extension, dest in ((".onnx", onnx), (".onnx.json", cfg)):
            url = _HF_VOICE_URL.format(extension=extension, **fmt)
            tmp_dest = dest + ".part"
            with urllib.request.urlopen(url, timeout=20) as resp, open(tmp_dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            os.replace(tmp_dest, dest)
        return True
    except Exception as e:
        print(f"  [piper_tts] voice auto-download failed (non-fatal, Listen "
              f"button just won't show): {e}")
        return False


def _ensure_auto_downloaded() -> bool:
    """Attempts the auto-download at most once per process -- is_available()
    is on the hot path of every chat response (it decides whether the
    button renders), so a real network call there is only acceptable
    because it happens exactly once per cold start, cached after."""
    global _download_attempted, _download_ok
    if _download_attempted:
        return _download_ok
    with _download_lock:
        if _download_attempted:
            return _download_ok
        _download_attempted = True
        if os.environ.get("PIPER_AUTO_DOWNLOAD", "1") == "0":
            _download_ok = False
        else:
            _download_ok = _download_voice_to(_AUTO_VOICE_DIR)
        return _download_ok


def is_available() -> bool:
    if _existing_voice_dir() is not None:
        return True
    return _ensure_auto_downloaded()


def _get_voice():
    """Lazy singleton load -- the ONNX session is the expensive part
    (~tens of ms), so it happens once per process, not once per request."""
    global _voice, _load_failed
    if _voice is not None or _load_failed:
        return _voice
    with _lock:
        if _voice is not None or _load_failed:
            return _voice
        if not is_available():
            _load_failed = True
            return None
        try:
            from piper import PiperVoice
            base_dir = _existing_voice_dir()
            onnx, cfg = _model_paths(base_dir)
            _voice = PiperVoice.load(onnx, cfg)
        except Exception:
            _load_failed = True
            return None
        return _voice


_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADER_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_BOLD_RE = re.compile(r"\*\*([^*]+?)\*\*|__([^_]+?)__")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)|(?<!_)_([^_\n]+?)_(?!_)")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_STRIKE_RE = re.compile(r"~~([^~]+?)~~")
_MD_BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
_MD_NUMLIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_MD_BLOCKQUOTE_RE = re.compile(r"(?m)^\s*>+\s*")
_MD_HR_OR_SEP_RE = re.compile(r"(?m)^[\s\-|:*_=]+$")
_TABLE_PIPE_RE = re.compile(r"\|")
_LEFTOVER_MD_RE = re.compile(r"[#*_`~]")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{2,}")
# A colon is spoken as the literal word "colon" by this TTS voice rather than
# read as a natural pause -- common in bullet-point labels like "NIM: management
# noted...". Swapped for a comma (a real pause, never spoken aloud) everywhere
# except digit:digit, so times/ratios ("3:30", "12:1") are left alone.
_COLON_RE = re.compile(r"(?<!\d):(?!\d)")


def _clean_for_speech(text: str) -> str:
    """Strip markdown formatting and inline [n]/[n,m] source-citation markers
    before handing text to the TTS engine -- Piper has no notion of markdown
    or citations, so left alone it reads the literal characters aloud
    ("hash hash", "bracket one comma two bracket", etc). Order matters: links
    are unwrapped before the bare-citation pattern runs (a link's own
    brackets would otherwise look like a citation), and structural markers
    (headers/bullets/quotes/table pipes) are stripped line-by-line before the
    final whitespace collapse."""
    if not text:
        return text
    cleaned = _MD_LINK_RE.sub(r"\1", text)
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _MD_HEADER_RE.sub("", cleaned)
    cleaned = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), cleaned)
    cleaned = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), cleaned)
    cleaned = _MD_CODE_RE.sub(r"\1", cleaned)
    cleaned = _MD_STRIKE_RE.sub(r"\1", cleaned)
    cleaned = _MD_BLOCKQUOTE_RE.sub("", cleaned)
    cleaned = _MD_BULLET_RE.sub("", cleaned)
    cleaned = _MD_NUMLIST_RE.sub("", cleaned)
    cleaned = _MD_HR_OR_SEP_RE.sub("", cleaned)
    cleaned = _TABLE_PIPE_RE.sub(" ", cleaned)
    cleaned = _LEFTOVER_MD_RE.sub("", cleaned)
    cleaned = _COLON_RE.sub(",", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned)
    cleaned = _BLANKLINES_RE.sub("\n", cleaned)
    return cleaned.strip()


def synthesize_wav(text: str) -> bytes | None:
    """Blocking -- call via asyncio.to_thread from request handlers, never
    directly in an async def. Piper's own voice.synthesize() already yields
    one AudioChunk per SENTENCE (its own sentence splitter, not a hand-rolled
    regex) rather than one chunk for the whole text -- that's the actual
    latency win: the first chunk of a long answer is ready long before the
    last one. Every chunk gets concatenated as raw PCM under ONE wave header;
    naively concatenating several complete .wav files back-to-back produces a
    file most players reject (each has its own header/size fields), so this
    always goes through the `wave` module instead. Text is run through
    _clean_for_speech() first so markdown syntax (#, **, bullets, tables) and
    inline [n] source citations aren't read aloud literally. Returns None if
    no voice is installed, or the text is empty."""
    voice = _get_voice()
    text = _clean_for_speech(text) if text else text
    if voice is None or not text or not text.strip():
        return None
    buf = io.BytesIO()
    wav_out = None
    try:
        for chunk in voice.synthesize(text):
            if wav_out is None:
                wav_out = wave.open(buf, "wb")
                wav_out.setnchannels(chunk.sample_channels)
                wav_out.setsampwidth(chunk.sample_width)
                wav_out.setframerate(chunk.sample_rate)
            wav_out.writeframes(chunk.audio_int16_bytes)
    finally:
        if wav_out is not None:
            wav_out.close()
    return buf.getvalue() if wav_out is not None else None
