"""
hybrid_search.py — server-side port of the same BM25 + TF-IDF-cosine +
reciprocal-rank-fusion retrieval used client-side in the static UI
(frontend/_ui_template.html), so /api/chat and /api/search can ground
answers the same way whether or not an LLM is reachable.

2026-09 update: TF-IDF cosine was previously called the "semantic proxy"
here, but it isn't one -- it's still literal token overlap, just weighted
differently than BM25. Verified against this platform's own Axis corpus
that a plain-English paraphrase of CASA ("low cost deposits") surfaced an
unrelated passage as its #1 result under BM25+TF-IDF alone, because the
query and the right passage share zero tokens. A real (if CPU-light)
semantic signal is now layered in via src/search/embeddings.py -- spaCy
word vectors, chosen specifically because huggingface.co (where sentence-
transformer weights live) is unreachable from this environment but
github.com, where spaCy's model wheels are hosted, is. See that module's
docstring for the full reachability check and the before/after numbers.

That signal is optional at the SearchIndex level (embed_vecs=None skips it
entirely) and degrades honestly if the model isn't installed on a given
machine -- BM25 + TF-IDF alone (today's behavior) is always the floor, not
a placeholder pretending to be semantic search.
"""

import math
import re
from collections import Counter

from src.search import embeddings

STOPWORDS = {
    'the','a','an','is','are','was','were','of','on','in','to','for','and','or','what','did',
    'do','does','about','that','this','with','as','at','by','be','it','their','his','her','which',
    'who','how','much','many','has','have','had','can','could','would','should','will',
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9%&']+", (text or "").lower())
            if t not in STOPWORDS and len(t) > 1]


class SearchIndex:
    def __init__(self, corpus: list[dict], embed_vecs: list[list[float]] | None = None):
        self.docs = corpus
        self.tokens = [tokenize(d["text"]) for d in corpus]
        self.n = len(corpus)
        self.df = Counter()
        for toks in self.tokens:
            self.df.update(set(toks))
        self.avgdl = sum(len(t) for t in self.tokens) / max(self.n, 1)
        self.tfidf_vecs = [self._tfidf_vec(t) for t in self.tokens]
        # Precomputed at load time (see fastapi_app.py's _load_live), one
        # vector per corpus doc in the same order -- None on a machine
        # where the embedding model isn't installed, in which case search()
        # below simply doesn't add the semantic ranked list to the fusion.
        self.embed_vecs = embed_vecs
        self.has_semantic = embed_vecs is not None

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _tfidf_vec(self, tokens: list[str]) -> dict:
        tf = Counter(tokens)
        return {t: c * self.idf(t) for t, c in tf.items()}

    def bm25(self, q_tokens: list[str], idx: int, k1=1.5, b=0.75) -> float:
        doc_tokens = self.tokens[idx]
        tf = Counter(doc_tokens)
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            num = tf[t] * (k1 + 1)
            den = tf[t] + k1 * (1 - b + b * len(doc_tokens) / max(self.avgdl, 1e-9))
            score += self.idf(t) * num / den
        return score

    def cosine(self, q_vec: dict, idx: int) -> float:
        d_vec = self.tfidf_vecs[idx]
        dot = sum(v * d_vec.get(k, 0.0) for k, v in q_vec.items())
        n1 = math.sqrt(sum(v * v for v in q_vec.values()))
        n2 = math.sqrt(sum(v * v for v in d_vec.values()))
        if not n1 or not n2:
            return 0.0
        return dot / (n1 * n2)

    def search(self, query: str, candidate_idx: list[int] | None = None, top_k: int = 10) -> list[dict]:
        q_tokens = tokenize(query)
        q_vec = Counter(q_tokens)
        q_vec = {t: c * self.idf(t) for t, c in q_vec.items()}
        pool = candidate_idx if candidate_idx is not None else list(range(self.n))

        bm25_ranked = sorted(pool, key=lambda i: -self.bm25(q_tokens, i))
        bm25_ranked = [i for i in bm25_ranked if self.bm25(q_tokens, i) > 0]
        sem_ranked = sorted(pool, key=lambda i: -self.cosine(q_vec, i))
        sem_ranked = [i for i in sem_ranked if self.cosine(q_vec, i) > 0]

        # Real semantic signal (word-vector cosine), additive to the two
        # lexical rankers above -- see this module's docstring. Skipped
        # entirely (embed_ranked stays []) if no embedding vectors were
        # precomputed for this corpus, or the query itself fails to embed.
        embed_ranked = []
        if self.embed_vecs is not None:
            qv = embeddings.embed_one(query)
            if qv is not None:
                sims = {i: embeddings.cosine(qv, self.embed_vecs[i]) for i in pool}
                # A higher floor than the lexical rankers' ">0": GloVe cosine
                # is noisy near zero (unrelated pairs land anywhere from
                # -0.1 to 0.15), so a low bar would let noise vote in the
                # fusion below rather than genuinely related passages.
                embed_ranked = sorted((i for i in pool if sims[i] > 0.2), key=lambda i: -sims[i])

        k = 60
        scores = {}
        for rank_list in (bm25_ranked, sem_ranked, embed_ranked):
            for rank, i in enumerate(rank_list):
                scores[i] = scores.get(i, 0.0) + 1 / (k + rank + 1)

        fused = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [{"doc": self.docs[i], "score": s} for i, s in fused]
