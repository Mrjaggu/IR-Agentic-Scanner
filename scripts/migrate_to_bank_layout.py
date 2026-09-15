"""
migrate_to_bank_layout.py — one-time move to the per-bank directory layout.

Run once, from the repo root: `python3 scripts/migrate_to_bank_layout.py`

Scope: moves ONLY the files that src/config/settings.py's paths_for(bank_id)
resolves (i.e. the files the compatibility shim points at) into
data/db/axis/, data/inputs/axis/, data/outputs/axis/, earnings_transcript/axis/.
Everything else in those directories -- backup files (graph_v1_backup.json,
predictions_groq_backup.json, cross_validation_v2.json, semantic_eval.json,
semantic_eval_claude.json), orphan/unused files (llm_judge_prompts.json --
grepped, nothing in src/ imports it), fixtures (ui_fixture.json, move_eval.json),
and files with their OWN hardcoded path constants outside settings.py's shim
(workspace_state.json via src/data/workspace_state.py, agentic_predictions.json
/ agentic_brief.md via src/agentic/run_agentic.py + src/ui_compiler.py) --
is left exactly where it is. Those get their own bank-scoping in a later phase
(Phase 3 of the plan), not this migration.

COPIES (not moves) the two existing peer transcripts into their own
first-class bank directories, leaving the originals in earnings_transcript/
peers/ untouched so src/signals/peer_signal.py keeps working exactly as
before (see the note in settings.py next to PEER_TRANSCRIPT_DIR):
    earnings_transcript/peers/kotak-*.pdf    -> earnings_transcript/kotak/ (copy)
    earnings_transcript/peers/indusind-*.pdf -> earnings_transcript/indusind/ (copy)

Prints a before/after sanity check. Exits non-zero (nothing left half-done
to reconcile by hand) if any expected source file is missing.
"""

import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exactly the filenames paths_for("axis") resolves, relative to their current
# (pre-migration) top-level location.
DB_FILES = ["dataset.json", "graph.json", "metrics_timeseries.json", "question_intent.json"]
INPUT_FILES = [
    "analyst_personas.json", "analyst_profile_enrichment.json", "narration_novelty.json",
    "peer_signal.json", "external_context_history.json",
    "analyst_personas_transcript_derived.json", "analyst_ask_patterns.json",
]
OUTPUT_FILES = ["predictions.json", "ir_prep_sheet.json", "cross_validation.json", "semantic_eval_full.json"]


def git_mv(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["git", "mv", src, dst], cwd=BASE, check=True)
    return True


def main() -> None:
    data_db = os.path.join(BASE, "data", "db")
    data_inputs = os.path.join(BASE, "data", "inputs")
    data_outputs = os.path.join(BASE, "data", "outputs")
    transcripts = os.path.join(BASE, "earnings_transcript")
    peers = os.path.join(transcripts, "peers")

    moved, missing = [], []

    for fname in DB_FILES:
        (moved if git_mv(os.path.join(data_db, fname), os.path.join(data_db, "axis", fname))
         else missing).append(f"data/db/{fname}")

    for fname in INPUT_FILES:
        (moved if git_mv(os.path.join(data_inputs, fname), os.path.join(data_inputs, "axis", fname))
         else missing).append(f"data/inputs/{fname}")

    for fname in OUTPUT_FILES:
        (moved if git_mv(os.path.join(data_outputs, fname), os.path.join(data_outputs, "axis", fname))
         else missing).append(f"data/outputs/{fname}")

    axis_pdfs = [f for f in os.listdir(transcripts) if f.endswith(".pdf")] if os.path.isdir(transcripts) else []
    for fname in axis_pdfs:
        git_mv(os.path.join(transcripts, fname), os.path.join(transcripts, "axis", fname))
        moved.append(f"earnings_transcript/{fname}")

    copied = {"kotak": 0, "indusind": 0}
    if os.path.isdir(peers):
        for fname in os.listdir(peers):
            if not fname.endswith(".pdf"):
                continue
            bank = fname.split("-")[0].lower()
            if bank not in ("kotak", "indusind"):
                print(f"  [skip] {fname}: unrecognized peer bank prefix {bank!r}")
                continue
            dst_dir = os.path.join(transcripts, bank)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(os.path.join(peers, fname), dst)
            subprocess.run(["git", "add", dst], cwd=BASE, check=True)
            copied[bank] += 1

    print(f"Moved {len(moved)} files:")
    for m in moved:
        print(f"  {m}")
    if missing:
        print(f"\n{len(missing)} expected files were not found (left as a note, not fatal -- "
              f"e.g. peer_signal.json/external_context_history.json/personas files may not "
              f"exist yet on a fresh checkout):")
        for m in missing:
            print(f"  {m}")
    print(f"\nCopied into new peer bank dirs (originals kept in earnings_transcript/peers/): {copied}")

    problems = []
    missing_names = [m.rsplit("/", 1)[-1] for m in missing]
    for fname in DB_FILES:
        if fname not in missing_names and not os.path.exists(os.path.join(data_db, "axis", fname)):
            problems.append(f"data/db/axis/{fname} missing after migration")
    for fname in INPUT_FILES:
        if fname not in missing_names and not os.path.exists(os.path.join(data_inputs, "axis", fname)):
            problems.append(f"data/inputs/axis/{fname} missing after migration")
    for fname in OUTPUT_FILES:
        if fname not in missing_names and not os.path.exists(os.path.join(data_outputs, "axis", fname)):
            problems.append(f"data/outputs/axis/{fname} missing after migration")
    axis_dir = os.path.join(transcripts, "axis")
    after_axis_pdfs = len([f for f in os.listdir(axis_dir) if f.endswith(".pdf")]) if os.path.isdir(axis_dir) else 0
    if len(axis_pdfs) != after_axis_pdfs:
        problems.append("earnings_transcript/axis/ pdf count doesn't match pre-migration count")

    if problems:
        print("\n!! Problems found:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)
    print("\nAll expected files reconciled.")


if __name__ == "__main__":
    main()
