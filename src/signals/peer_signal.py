"""
peer_signal.py — cross-bank topic salience signal.

Motivation: Axis reports mid-to-late in results season. Peer banks that report
earlier in the same quarter (Kotak, IndusInd, HDFC, ICICI, ...) give analysts a
first look at sector-wide themes — margin compression, a specific asset-quality
pocket, a capital raise — before they get on Axis's call. That's genuinely new
information relative to Axis's own question history, not more arithmetic on it.

Usage:
    python3 run.py peers                  # target = VAL_QUARTER
    python3 run.py peers --quarter q1fy27

Reads every PDF in earnings_transcript/peers/ whose filename contains the
target quarter (e.g. "kotak-q1fy27-earnings-call-transcript.pdf"), parses each
with the same transcript parser used for Axis, tags each peer analyst question
block against the shared 12-topic taxonomy, and counts how many DISTINCT peer
analysts raised each topic this quarter (not raw question count, so one chatty
analyst can't dominate).

Output: data/inputs/peer_signal.json
    {"quarter": "q1fy27", "peers": ["kotak", "indusind"],
     "topic_salience": {"NIM & Yields": 1.0, ...},
     "peer_analyst_counts": {"NIM & Yields": 5, ...}}

The engine picks this file up automatically at predict time (only if the
file's quarter matches VAL_QUARTER) and adds AT MOST ONE extra prep slot per
analyst for a peer-salient topic — same discipline as narration_novelty.py,
never displacing a validated top-N pick.

VALIDATION DISCIPLINE: this signal is unproven. Measure before trusting:
    python3 run.py predict                    # with the file  -> note blind F1
    mv data/inputs/peer_signal.json /tmp/      # remove it
    python3 run.py predict                    # without        -> compare
Keep it only if the blind block improves (or at minimum doesn't regress).
"""

import json
import os
import re

from src.config.settings import PEER_TRANSCRIPT_DIR, PEER_SIGNAL_PATH, VAL_QUARTER
from src.data.dataset_compiler import parse_transcript
from src.graphs.compiler import classify_topics


def _quarter_of(filename: str) -> str | None:
    m = re.search(r'(q[1-4]fy\d{2})', filename, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _peer_name_of(filename: str) -> str:
    return filename.split("-")[0]


def _group_qa_blocks(qa_turns: list[dict]) -> list[dict]:
    """Group flat Q&A turns into per-analyst blocks (mirrors graphs/compiler.py)."""
    blocks = []
    current_analyst = None
    current_questions = []

    for turn in qa_turns:
        role = turn["role"]
        text = turn["text"]
        a_name = turn["analyst_name"]

        if role == "Moderator":
            if current_analyst and current_questions:
                blocks.append({"analyst": current_analyst, "questions": current_questions})
            current_analyst = a_name
            current_questions = []
        elif role == "Analyst":
            current_questions.append(text)

    if current_analyst and current_questions:
        blocks.append({"analyst": current_analyst, "questions": current_questions})
    return blocks


def compute_peer_signal(quarter: str | None = None) -> dict:
    quarter = quarter or VAL_QUARTER

    if not os.path.exists(PEER_TRANSCRIPT_DIR):
        raise SystemExit(f"No peer transcript directory at {PEER_TRANSCRIPT_DIR}")

    files = [f for f in os.listdir(PEER_TRANSCRIPT_DIR)
             if f.endswith(".pdf") and _quarter_of(f) == quarter]
    if not files:
        raise SystemExit(f"No peer transcripts found for {quarter} in {PEER_TRANSCRIPT_DIR}")

    peers_used = []
    topic_analyst_sets: dict[str, set] = {}

    for fname in sorted(files):
        peer = _peer_name_of(fname)
        peers_used.append(peer)
        path = os.path.join(PEER_TRANSCRIPT_DIR, fname)
        _, qa_turns, _ = parse_transcript(path)
        blocks = _group_qa_blocks(qa_turns)

        for block in blocks:
            analyst = block["analyst"]
            if not analyst:
                continue
            q_text = " ".join(block["questions"])
            topics = [t for t in classify_topics(q_text)]
            for t in topics:
                topic_analyst_sets.setdefault(t, set()).add(f"{peer}:{analyst}")

    peer_analyst_counts = {t: len(s) for t, s in topic_analyst_sets.items()}
    max_count = max(peer_analyst_counts.values()) if peer_analyst_counts else 1
    topic_salience = {t: round(c / max_count, 3) for t, c in peer_analyst_counts.items()}

    out = {
        "quarter": quarter,
        "peers": peers_used,
        "topic_salience": topic_salience,
        "peer_analyst_counts": peer_analyst_counts,
    }
    with open(PEER_SIGNAL_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Peer transcripts used: {peers_used}")
    print("\nTopic salience (peer analysts asking, normalized):")
    for t, s in sorted(topic_salience.items(), key=lambda x: -x[1]):
        print(f"  {t:38s} salience={s:.2f}  ({peer_analyst_counts[t]} peer analysts)")
    print(f"\nSaved -> {PEER_SIGNAL_PATH}")
    return out


if __name__ == "__main__":
    compute_peer_signal()
