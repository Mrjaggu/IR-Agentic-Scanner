"""
piper_tts.py -- Section: chat "Listen" button, TTS synthesis.

Same reasoning as everywhere else in this platform that touches models
(src.model_provider.llm_client, src.search.embeddings): prefer a small,
fully offline, CPU-only open model over an API, because this is built for a
regulated, potentially air-gapped deployment. Piper (github.com/rhasspy/piper)
is a neural TTS engine whose English voices run 20-65MB as ONNX models,
synthesize on CPU near real-time, and need no network once installed.

The voice model is NOT bundled in this repo -- binary model weights don't
belong in git, and huggingface.co (where Piper voices are hosted) isn't on
this environment's network allowlist during development. Install it once,
from a real terminal with normal internet access:

    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium
    # or a smaller one: en_US-amy-low (~25MB) / en_US-ryan-low (~25MB)

That downloads <voice>.onnx + <voice>.onnx.json. Point PIPER_VOICE_DIR at
wherever they land (defaults to data/tts_voices/ in this repo) and
PIPER_VOICE at the voice name (defaults to en_US-lessac-medium). Until those
files exist, is_available() is False and every caller degrades honestly --
same discipline as pattern_retrieval.is_available() / embeddings.is_available():
the "Listen" button simply doesn't render rather than erroring.
"""

import io
import os
import re
import threading
import wave

from src.config.settings import BASE_DIR

VOICE_DIR = os.environ.get("PIPER_VOICE_DIR", os.path.join(BASE_DIR, "data", "tts_voices"))
VOICE_NAME = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")

_voice = None
_load_failed = False
_lock = threading.Lock()


def _model_paths(voice_name: str = VOICE_NAME) -> tuple[str, str]:
    onnx = os.path.join(VOICE_DIR, f"{voice_name}.onnx")
    cfg = os.path.join(VOICE_DIR, f"{voice_name}.onnx.json")
    return onnx, cfg


def is_available() -> bool:
    onnx, cfg = _model_paths()
    return os.path.exists(onnx) and os.path.exists(cfg)


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
            onnx, cfg = _model_paths()
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
