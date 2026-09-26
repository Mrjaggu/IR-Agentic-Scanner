"""
peer_signal.py — cross-bank topic salience signal.

Motivation: Axis reports mid-to-late in results season. Peer banks that report
earlier in the same quarter (Kotak, IndusInd, HDFC, ICICI, ...) give analysts a
first look at sector-wide themes — margin compression, a specific asset-quality
pocket, a capital raise — before they get on Axis's call. That's genuinely new
information relative to Axis's own question history, not more arithmetic on it.

Usage:
    python3 run.py peers                       # target = VAL_QUARTER, bank = axis
    python3 run.py peers --quarter q1fy27
    python3 run.py peers --quarter q2fy27 --bank axis

Transcript sources, in order, both scanned and merged (2026-09, ported into
the agentic path -- see src/agentic/analyst_layer.py's peer_extra_slot):

  1. Every OTHER bank already registered in src.config.banks.BANKS, read
     straight from ITS OWN earnings_transcript/<bank_id>/ archive (the same
     one that bank's own dataset/graph pipeline compiles from) via
     settings.paths_for(bank_id). Kotak/IndusInd both already have real,
     many-quarter archives there -- no separately-maintained copy needed
     for these anymore.
  2. The legacy shared earnings_transcript/peers/ folder, kept for any peer
     that ISN'T a fully-onboarded registered bank (e.g. HDFC, ICICI -- named
     above but never added as full BankConfig entries). Deduped against
     (1) by peer id so an already-registered bank's leftover copy there
     isn't double-counted.

Each matching PDF is parsed with the same transcript parser Axis's own
pipeline uses -- now with that peer's own legal_name for the header-scrub
and its own subsidiary_keywords for topic classification (previously this
called parse_transcript()/classify_topics() with neither, so a peer's own
subsidiary mentions were classified with no bank-specific vocabulary at
all) -- tags each peer analyst's question block against the shared 12-topic
taxonomy, and counts how many DISTINCT peer analysts raised each topic this
quarter (not raw question count, so one chatty analyst can't dominate).

Output: data/inputs/<target_bank_id>/peer_signal.json
    {"quarter": "q1fy27", "target_bank_id": "axis", "peers": ["kotak", "indusind"],
     "topic_salience": {"NIM & Yields": 1.0, ...},
     "peer_analyst_counts": {"NIM & Yields": 5, ...}}

Consumed automatically by BOTH: the legacy deterministic engine (engine.py,
via the flat PEER_SIGNAL_PATH shim -- axis only, unchanged) and, as of
2026-09, the agentic pipeline (src/agentic/run_agentic.py's
use_peer_signal, via src.agentic.analyst_layer.peer_extra_slot) -- each
adds AT MOST ONE extra prep slot per analyst for a peer-salient topic, same
discipline as narration_novelty.py, never displacing a validated top-N pick.

UNLIKE the news/macro signals, this one IS walk-forward backtestable: a
historical quarter's peer signal comes from that same quarter's real,
already-reported peer transcripts, not "right now" -- see
run_agentic.build_initial_state's docstring. Validate with:
    python3 -c "from src.agentic.eval_harness import run_holdout_eval as r; \
                print(r(use_peer_signal=True)['promotion_gate'])"
compared against the same call with use_peer_signal=False (the default),
same method as every other opt-in signal in this codebase.
"""

import json
import os
import re

from src.config.settings import PEER_TRANSCRIPT_DIR, VAL_QUARTER, paths_for
from src.config.banks import BANKS, DEFAULT_BANK
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


def _peer_files_for_quarter(quarter: str, target_bank_id: str) -> list[tuple[str, str | None, str]]:
    """(peer_id, legal_name_or_None, full_pdf_path) for every peer transcript
    matching `quarter`, from both sources described in the module docstring.
    peer_id is a real BANKS key for source (1), or whatever the legacy
    folder's filename implies for source (2) -- BANKS.get(peer_id) may be
    None for a peer that isn't a registered bank, which is fine: legal_name
    and subsidiary_keywords just aren't available for it, same as before
    this change for every peer."""
    found = []
    seen_peers = set()

    for bank_id, bank in BANKS.items():
        if bank_id == target_bank_id:
            continue
        tdir = paths_for(bank_id).earnings_transcript_dir
        if not os.path.isdir(tdir):
            continue
        for fname in sorted(os.listdir(tdir)):
            if fname.lower().endswith(".pdf") and _quarter_of(fname) == quarter:
                found.append((bank_id, bank.legal_name, os.path.join(tdir, fname)))
                seen_peers.add(bank_id)

    if os.path.isdir(PEER_TRANSCRIPT_DIR):
        for fname in sorted(os.listdir(PEER_TRANSCRIPT_DIR)):
            if not fname.lower().endswith(".pdf") or _quarter_of(fname) != quarter:
                continue
            peer_id = _peer_name_of(fname)
            if peer_id in seen_peers or peer_id == target_bank_id:
                continue  # already covered via its own registered archive above
            legal_name = BANKS[peer_id].legal_name if peer_id in BANKS else None
            found.append((peer_id, legal_name, os.path.join(PEER_TRANSCRIPT_DIR, fname)))

    return found


def compute_peer_signal(quarter: str | None = None, target_bank_id: str = DEFAULT_BANK) -> dict:
    quarter = quarter or VAL_QUARTER

    files = _peer_files_for_quarter(quarter, target_bank_id)
    if not files:
        raise SystemExit(
            f"No peer transcripts found for {quarter} (checked every OTHER registered "
            f"bank's own earnings_transcript/<bank>/ archive, plus {PEER_TRANSCRIPT_DIR})"
        )

    peers_used = []
    topic_analyst_sets: dict[str, set] = {}

    for peer_id, legal_name, path in files:
        peers_used.append(peer_id)
        if legal_name:
            _, qa_turns, _ = parse_transcript(path, legal_name=legal_name)
        else:
            _, qa_turns, _ = parse_transcript(path)
        blocks = _group_qa_blocks(qa_turns)
        subsidiary_kws = BANKS[peer_id].subsidiary_keywords if peer_id in BANKS else None

        for block in blocks:
            analyst = block["analyst"]
            if not analyst:
                continue
            q_text = " ".join(block["questions"])
            topics = classify_topics(q_text, subsidiary_keywords=subsidiary_kws)
            for t in topics:
                topic_analyst_sets.setdefault(t, set()).add(f"{peer_id}:{analyst}")

    peer_analyst_counts = {t: len(s) for t, s in topic_analyst_sets.items()}
    max_count = max(peer_analyst_counts.values()) if peer_analyst_counts else 1
    topic_salience = {t: round(c / max_count, 3) for t, c in peer_analyst_counts.items()}

    out_path = paths_for(target_bank_id).peer_signal_path
    out = {
        "quarter": quarter,
        "target_bank_id": target_bank_id,
        "peers": sorted(set(peers_used)),
        "topic_salience": topic_salience,
        "peer_analyst_counts": peer_analyst_counts,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Peer transcripts used (for {target_bank_id}'s prep sheet): {sorted(set(peers_used))}")
    print("\nTopic salience (peer analysts asking, normalized):")
    for t, s in sorted(topic_salience.items(), key=lambda x: -x[1]):
        print(f"  {t:38s} salience={s:.2f}  ({peer_analyst_counts[t]} peer analysts)")
    print(f"\nSaved -> {out_path}")
    return out


if __name__ == "__main__":
    compute_peer_signal()
