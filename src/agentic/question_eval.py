"""
question_eval.py — Section 6.2's OTHER eval: did we anticipate the actual
QUESTION, not merely the topic bucket?

Topic recall answers "was 'NIM & Yields' on the brief". Question recall answers
"did we anticipate that she would press on where the 3.8% structural NIM target
stands". Those are very different bars, and the first flatters the second:
with a 12-topic taxonomy and 8 slots per analyst, hitting the bucket is nearly
free. The doc says so directly — "a different eval than topic labels, since
hallucinated-but-fluent questions are worse than missed topics" — and
RESULTS_V2.md found the same thing empirically, with semantic F1 (55.3%)
landing BELOW topic F1 rather than above it.

Two further traps this module exists to expose:

  Under-counted ground truth. A question block often carries several distinct
  concerns; the topic tagger collapses them to one or two labels. Mahrukh's
  Q1FY27 block raises NIM, an opex run-rate, AND foreign-loan growth, but is
  tagged with two topics. Scored on topics she looks fully covered; scored on
  concerns, one third of what she asked was never anticipated.

  Off-taxonomy concerns. "Foreign loans have grown sharply" has no clean
  bucket, so topic scoring cannot even see the miss.

Scoring uses the 0 / 0.5 / 1.0 rubric RESULTS_V2.md established:
  1.0  the predicted question asks substantially the same thing
  0.5  right subject and direction, misses the specific hook being pressed
  0.0  not anticipated
An LLM judge is used when one is reachable; otherwise a lexical-overlap proxy
runs and is labelled as a proxy, never as the judged score.
"""

import json
import re

from src.search.hybrid_search import tokenize

# ---------------------------------------------------------------------------
# Decomposition, v2.
#
# v1 split ON the transition marker, which cut sentences in half: "My first
# question is again around NIM" became "is again around NIM". Boundaries now
# fall between SENTENCES, and a sentence that announces a new ask opens a new
# group while staying intact. Acknowledgement-only sentences ("Got it, thank
# you", "That's helpful") are dropped, so they stop inflating the denominator.
# ---------------------------------------------------------------------------

# A new ask announces itself at the START of a sentence.
_NEW_ASK = re.compile(
    r"^(?:and\s+|also[,\s]+|so\s+|but\s+)*(?:"
    r"(?:my\s+|the\s+)?(?:first|second|third|fourth|next|last|other|another)\s+question"
    r"|secondly|thirdly|lastly|finally"
    r"|(?:just\s+)?one\s+(?:last|more|final)\s+(?:thing|question)"
    r"|moving\s+on|coming\s+(?:to|back\s+to)"
    r"|the\s+(?:second|other|next|last)\s+(?:one|thing|question)"
    r")", re.I)

# Mid-sentence sign-off: whatever follows starts a new ask.
_HANDOFF = re.compile(
    r"that'?s\s+(?:my|the)\s+(?:first|second|third|next)\s+question"
    r"|that'?s\s+(?:it|all)\s+from\s+me", re.I)

# Pure courtesy / acknowledgement — carries no ask at all.
_ACK_ONLY = re.compile(
    r"^(?:(?:okay|ok|yes|yeah|yep|sure|right|alright|fine|great|perfect"
    r"|got\s+it|got\s+the\s+answer|understood|noted|fair\s+enough"
    r"|thanks?|thank\s+you|thank\s+you\s+so\s+much"
    r"|hi|hello|good\s+(?:morning|afternoon|evening)|congratulations|congrats"
    r"|that\s+(?:is|'s)\s+(?:very\s+)?(?:helpful|clear|useful|great|fine)"
    r"|i\s+got\s+the\s+answer|appreciate\s+it|all\s+the\s+best"
    r"|(?:thanks?|thank\s+you)\s+for\s+taking\s+my\s+question[s]?)"
    r"[\s,.!?—-]*)+$", re.I)

# A group only counts as a concern if it actually poses something. Deliberately
# stricter than "contains a verb": bare declaratives are context, not asks.
_ASKS = re.compile(
    r"\?|\b(?:how|what|why|where|when|which"
    r"|could\s+you|can\s+you|would\s+you|will\s+you|are\s+you\s+able"
    r"|any\s+(?:sense|colour|color|guidance|comment|thoughts|update|number)"
    r"|if\s+you\s+could|wanted\s+to\s+(?:know|understand|check|clarify)"
    r"|walk\s+us|help\s+us\s+understand|give\s+us|tell\s+us"
    r"|clarify|quantify|elaborate|comment\s+on)\b", re.I)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


def decompose_concerns(question_text: str, min_words: int = 6) -> list[str]:
    """Split one analyst turn into the distinct things they actually asked.

    Ground truth at topic level under-counts these, which is precisely why
    question-level scoring needs its own decomposition rather than reusing the
    topic labels: Mahrukh's Q1FY27 turn is tagged with two topics and asks
    three separate things."""
    if not question_text:
        return []

    sents = [s for s in _sentences(question_text) if not _ACK_ONLY.match(s)]
    groups: list[list[str]] = []
    cur: list[str] = []
    force_new = False
    for s in sents:
        if cur and (force_new or _NEW_ASK.match(s)):
            groups.append(cur)
            cur = []
        force_new = bool(_HANDOFF.search(s))
        cur.append(s)
    if cur:
        groups.append(cur)

    out = []
    for g in groups:
        txt = " ".join(g).strip()
        if len(txt.split()) < min_words:
            continue
        if not _ASKS.search(txt):
            continue
        out.append(txt)
    return out


def _overlap(a: str, b: str) -> float:
    """Lexical proxy — content-word overlap, weighted toward the concern's
    own vocabulary. A proxy for triage only; never reported as a judged score."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


_JUDGE_RUBRIC = """You are scoring whether an IR team's PREDICTED question anticipated what an
analyst ACTUALLY asked on an earnings call.

Score strictly:
  1.0 — the prediction asks substantially the same thing; management preparing
        for it would have been ready for the actual question
  0.5 — right subject and direction, but misses the specific hook the analyst
        pressed on (e.g. predicts "how will NIM evolve" when they asked
        "where does the 3.8% structural target stand")
  0.0 — not anticipated

Return JSON only:
{"score": 1.0|0.5|0.0, "best_prediction_index": <int or null>, "why": "<one sentence>"}"""


def score_concern(concern: str, predictions: list[str], client=None) -> dict:
    """Best-matching prediction for one concern, plus a score."""
    if not predictions:
        return {"concern": concern, "score": 0.0, "best": None,
                "why": "nothing was predicted for this analyst", "method": "none"}

    ranked = sorted(range(len(predictions)),
                    key=lambda i: -_overlap(concern, predictions[i]))
    top = ranked[:4]

    if client is not None and (getattr(client, "active_llm", None) or client.probe_llm()):
        listed = "\n".join(f"[{n}] {predictions[i]}" for n, i in enumerate(top))
        prompt = (f"{_JUDGE_RUBRIC}\n\nACTUAL question the analyst asked:\n\"{concern}\"\n\n"
                  f"PREDICTED questions to choose from:\n{listed}\n")
        raw = client.call_llm(prompt, temperature=0.0)
        if raw:
            clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
            try:
                d = json.loads(clean)
                idx = d.get("best_prediction_index")
                best = predictions[top[idx]] if isinstance(idx, int) and 0 <= idx < len(top) else None
                return {"concern": concern, "score": float(d.get("score", 0.0)),
                        "best": best, "why": d.get("why", ""), "method": "llm_judge"}
            except Exception:
                pass

    # Proxy path, explicitly labelled.
    best_i = ranked[0]
    ov = _overlap(concern, predictions[best_i])
    return {"concern": concern, "score": None, "proxy_overlap": round(ov, 3),
            "best": predictions[best_i], "method": "lexical_proxy",
            "why": "no LLM judge reachable — overlap is a triage signal, not a score"}


def evaluate_analyst_questions(actual_block: str, predicted_questions: list[str],
                               client=None) -> dict:
    """Question-level recall for one analyst on one call."""
    concerns = decompose_concerns(actual_block)
    scored = [score_concern(c, predicted_questions, client) for c in concerns]
    judged = [s for s in scored if s.get("score") is not None]
    return {
        "n_concerns": len(concerns),
        "n_predicted": len(predicted_questions),
        "per_concern": scored,
        "question_recall": (round(sum(s["score"] for s in judged) / len(judged), 3)
                            if judged else None),
        "fully_anticipated": sum(1 for s in judged if s["score"] >= 1.0),
        "partially_anticipated": sum(1 for s in judged if s["score"] == 0.5),
        "missed": sum(1 for s in judged if s["score"] == 0.0),
        "method": "llm_judge" if judged else "lexical_proxy",
    }
