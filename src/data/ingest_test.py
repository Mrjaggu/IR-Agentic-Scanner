"""
ingest_test.py — Section 2.1 extraction-quality monitor, made testable.

The architecture doc's ingestion layer calls for a monitor that fuzzy-matches
every extracted entity (analyst names, firms) against a canonical registry,
flagging things like "Piran Engineer" mis-OCR'd as "Kiran Engineer" -- real
examples the doc cites from actual transcripts. This wasn't built in the
first agentic-core session (Section 2 was out of scope); it's built now
specifically to answer "how do I test that ingesting a new quarter's PDF
actually works" -- a preview (dry run, nothing written) and a commit
(actually adds the quarter and recompiles dataset.json + graph.json).
"""

import difflib
import os
import tempfile

from src.config.settings import EARNINGS_TRANSCRIPT_DIR, ANALYST_ALIASES
from src.data.dataset_compiler import parse_transcript, get_quarter_info
from src.data.loader import load_graph


def _canonical_registry(graph: dict) -> set[str]:
    """Every analyst name already known good: seen in the graph, or a
    canonical alias target. This is the 'canonical registry' the doc's
    extraction-quality monitor fuzzy-matches new extractions against."""
    names = {n["properties"]["analyst"] for n in graph["nodes"] if n["type"] == "Question"}
    names |= set(ANALYST_ALIASES.values())
    names.discard(None)
    return names


def _fuzzy_flag(name: str, registry: set[str]) -> dict | None:
    """None if the name is already canonical or has no close match worth
    flagging. Otherwise a flag describing the closest canonical name --
    exactly the "repeats across calls" case the doc says should surface
    (a one-off OCR glitch is noise; a name that keeps almost-matching a
    known analyst is a real split/typo worth a human look)."""
    if name in registry:
        return None
    close = difflib.get_close_matches(name, registry, n=1, cutoff=0.72)
    if close and close[0] != name:
        return {"extracted": name, "closest_canonical": close[0],
               "similarity": round(difflib.SequenceMatcher(None, name, close[0]).ratio(), 3)}
    return None


def preview_ingest(pdf_bytes: bytes, filename: str) -> dict:
    """Dry run: parses the PDF, returns diagnostics. Writes nothing."""
    qid, skey = get_quarter_info(filename)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        narration, qa, call_date = parse_transcript(tmp_path)
    finally:
        os.unlink(tmp_path)

    analysts = {}
    for turn in qa:
        name = turn.get("analyst_name")
        if not name or name in ("Moderator", "Operator"):
            continue
        rec = analysts.setdefault(name, {"name": name, "firm": turn.get("analyst_firm"), "turns": 0})
        rec["turns"] += 1

    graph = load_graph()
    registry = _canonical_registry(graph)
    flags = [f for a in analysts.values() if (f := _fuzzy_flag(a["name"], registry))]

    already_exists = any(
        get_quarter_info(f)[0] == qid
        for f in os.listdir(EARNINGS_TRANSCRIPT_DIR) if f.endswith(".pdf")
    )

    return {
        "detected_quarter_id": qid,
        "sort_key": skey,
        "call_date": call_date,
        "narration_length_chars": len(narration or ""),
        "qa_turn_count": len(qa),
        "analysts": sorted(analysts.values(), key=lambda a: -a["turns"]),
        "extraction_quality_flags": flags,
        "quarter_already_in_dataset": already_exists,
    }


def commit_ingest(pdf_bytes: bytes, filename: str, overwrite: bool = False) -> dict:
    """Writes the PDF into earnings_transcript/ and recompiles dataset.json
    + graph.json (full batch recompile -- the existing compilers don't
    support incremental append, so this reuses the same recompile path
    `python3 run.py compile --dataset/--graph` already uses)."""
    qid, _ = get_quarter_info(filename)
    os.makedirs(EARNINGS_TRANSCRIPT_DIR, exist_ok=True)

    existing = [f for f in os.listdir(EARNINGS_TRANSCRIPT_DIR)
               if f.endswith(".pdf") and get_quarter_info(f)[0] == qid]
    if existing and not overwrite:
        return {"status": "conflict",
               "message": f"Quarter {qid} already has a transcript ({existing[0]}). "
                          f"Pass overwrite=true to replace it."}

    for f in existing:
        os.remove(os.path.join(EARNINGS_TRANSCRIPT_DIR, f))

    dest = os.path.join(EARNINGS_TRANSCRIPT_DIR, filename)
    with open(dest, "wb") as f:
        f.write(pdf_bytes)

    from src.data import dataset_compiler
    from src.graphs import compiler as graph_compiler
    from src.data.loader import load_dataset

    before = len(load_dataset())
    try:
        dataset_compiler.main()
        graph_compiler.main()
    except Exception as e:
        return {"status": "error", "message": f"Recompile failed: {e}"}

    after_dataset = load_dataset()
    after = len(after_dataset)
    present = any(q["quarter_id"] == qid for q in after_dataset)

    return {
        "status": "ok" if present else "error",
        "quarter_id": qid,
        "quarters_before": before,
        "quarters_after": after,
        "quarter_present_after_recompile": present,
        "next_steps": "Re-run `python3 run.py agentic --quarter " + qid +
                      "` and `python3 run.py ui` to refresh the prediction/UI outputs with this new quarter.",
    }
