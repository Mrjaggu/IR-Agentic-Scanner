"""
embeddings.py — CPU-lightweight semantic retrieval signal, additive to the
existing BM25 + TF-IDF-cosine hybrid search (src/search/hybrid_search.py).

Why this exists: TF-IDF cosine, despite the old docstring calling it a
"semantic proxy", is still literal token overlap -- it has no notion of
synonymy or paraphrase. Verified against this platform's own Axis corpus
(2026-09): the query "low cost deposits" (a plain-English paraphrase of
CASA) surfaced an unrelated Credit Cost passage as its #1 result, while the
literal query "CASA" correctly hit CASA-specific passages. TF-IDF cosine
gave zero help there because "low cost deposits" and "CASA" share no
tokens at all.

Why spaCy's en_core_web_md rather than an HF sentence-transformer: this
environment (both the cloud sandbox and the bridge to the user's Mac)
cannot reach huggingface.co (confirmed via curl -- 403/timeout on
huggingface.co, cdn-lfs.huggingface.co, hf-mirror.com). github.com and
files.pythonhosted.org ARE reachable, and spaCy's model wheels are
published as GitHub release assets, not HF-hosted -- see the model install
command in requirements.txt / README. en_core_web_md ships 300-dim GloVe
word vectors (20k-word vocab), mean-pooled per document by spaCy's own
Doc.vector -- no torch, no ONNX runtime, ~45MB, embeds this platform's
~600-document per-bank corpus in ~4s at load time and a query in <1ms.

Known gap this module works around: GloVe vectors are trained on generic
web/Wikipedia text, so finance ACRONYMS used bare in these transcripts
(CASA, NIM, PCR, GNPA...) either collide with an unrelated generic meaning
or have a noisy vector -- verified CASA-vs-"low cost deposits" cosine
similarity was -0.09 (i.e. anti-correlated) using the raw model, indistin-
guishable from a random unrelated phrase. ACRONYM_EXPANSIONS spells these
out before embedding ONLY (never before BM25/TF-IDF, so exact-acronym
literal matching is untouched) -- after expansion, "CASA" and "low cost
deposits" both embed close to "current account savings account", closing
the gap this module exists to close. Every other pairing tested (no
acronym involved) already scored well on the raw model: "cost efficiency"
vs "cost to income" 0.71, "bad loans" vs "asset quality" 0.59, "margin
compression" vs "net interest margin" 0.67.

Degrades honestly: if spacy or the en_core_web_md model isn't installed,
every function here returns None / a no-op rather than raising, and
hybrid_search.py falls back to BM25 + TF-IDF only (today's behavior) --
same "don't fabricate, say so" discipline this codebase already applies to
LLM availability (see meta.llm.available in fastapi_app.py).
"""

import re

# Grounded in what's actually in these transcripts (verified 2026-09: every
# one of these is used bare, never spelled out inline, across Axis's
# earnings calls) plus this platform's own TOPICS_LIST vocabulary
# (src/config/settings.py). Not exhaustive -- add to this as a real query
# surfaces a real miss, rather than guessing ahead of evidence.
ACRONYM_EXPANSIONS = {
    "casa": "current account savings account low cost deposits",
    "nim": "net interest margin",
    "nims": "net interest margins",
    "nii": "net interest income",
    "pat": "profit after tax",
    "pcr": "provision coverage ratio",
    "gnpa": "gross non performing assets bad loans",
    "nnpa": "net non performing assets bad loans",
    "npa": "non performing asset bad loan",
    "npas": "non performing assets bad loans",
    "sma": "special mention account overdue stressed loan",
    "roe": "return on equity",
    "roa": "return on assets",
    "yoy": "year over year",
    "qoq": "quarter over quarter",
    "opex": "operating expense",
    "alm": "asset liability management",
    "rwa": "risk weighted assets",
    "car": "capital adequacy ratio",
}

_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ACRONYM_EXPANSIONS) + r")\b",
    re.IGNORECASE,
)


def _expand_acronyms(text: str) -> str:
    """Appends (not replaces) each acronym's expansion right after it, so the
    embedding sees both forms -- a query or passage that already spells the
    term out is unaffected (re.sub only fires on the bare acronym), and one
    that uses only the acronym still gets the spelled-out signal alongside
    it rather than losing the literal token."""
    def _sub(m: "re.Match") -> str:
        acro = m.group(1)
        return f"{acro} {ACRONYM_EXPANSIONS[acro.lower()]}"
    return _ACRONYM_RE.sub(_sub, text)


_nlp = None
_load_attempted = False


def _get_nlp():
    global _nlp, _load_attempted
    if _load_attempted:
        return _nlp
    _load_attempted = True
    try:
        import spacy
        _nlp = spacy.load(
            "en_core_web_md",
            disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"],
        )
    except Exception:
        # ImportError (spacy not installed) or OSError (model not
        # downloaded) both mean the same thing here: no semantic signal
        # available this run. Never crash retrieval over it.
        _nlp = None
    return _nlp


def is_available() -> bool:
    return _get_nlp() is not None


def embed_many(texts: list[str]) -> list[list[float]] | None:
    """Batch-embeds; returns None (not a list of Nones) if the model isn't
    available, so callers can cleanly skip the semantic signal entirely."""
    nlp = _get_nlp()
    if nlp is None:
        return None
    expanded = [_expand_acronyms(t or "") for t in texts]
    return [doc.vector.tolist() for doc in nlp.pipe(expanded, batch_size=64)]


def embed_one(text: str) -> list[float] | None:
    nlp = _get_nlp()
    if nlp is None:
        return None
    return nlp(_expand_acronyms(text or "")).vector.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)
