"""
persona_synthesis.py — aggregate question_intent.json into per-analyst style profiles.

Pure aggregation, no LLM calls (question_intent.json was built by direct
transcript reading this session, not Groq/Gemini, per project decision to
reserve LLM quota for prediction). Turns the raw why-classifications into a
per-analyst rate profile: how much of what they ask is persona-consistent
(recurring interest) vs narration-triggered (reacts to fresh disclosures) vs
unexplained.

Key finding from the first run (8 quarters, 78 blocks): 73% of ALL questions
are narration-triggered, only 18% persona-consistent. Some analysts are
almost entirely narration-driven (MB Mahesh: 5/6, 0 persona-consistent) —
that's not noise, it's a stable style that a pure topic-frequency profile
can't see.

Kept SEPARATE from the existing hand-curated data/inputs/analyst_personas.json
(web-derived, already validated in production) — compared, not merged, until
this is proven at prediction time.

Usage:
    python3 run.py personas-derived

Output: data/inputs/analyst_personas_transcript_derived.json
    {"analyst": {"n": 8, "persona_consistent_rate": 0.375,
                 "narration_triggered_rate": 0.625, "unexplained_rate": 0.0,
                 "style_note": "..."}}
"""

import json

from src.config.settings import QUESTION_INTENT_PATH, PERSONA_DERIVED_PATH

MIN_N_FOR_STYLE_NOTE = 3


def _style_note(n: int, pc: float, nt: float, ux: float) -> str:
    if n < MIN_N_FOR_STYLE_NOTE:
        return f"Only {n} tracked question block(s) — too little history for a reliable style read."
    if nt >= 0.7:
        return (f"Strongly narration-driven ({nt:.0%} of {n} tracked questions reacted to "
                f"something specific management just disclosed, not a recurring topic interest). "
                f"Predict by watching what's NEW in the narration for this analyst, not just their history.")
    if pc >= 0.6:
        return (f"Strongly persona-consistent ({pc:.0%} of {n} tracked questions matched their own "
                f"recurring topic pattern). Historical topic frequency is a reliable predictor here.")
    if ux >= 0.3:
        return (f"High unexplained rate ({ux:.0%} of {n}) — meaningful share of this analyst's "
                f"questions don't map to their own history or this quarter's narration. Treat "
                f"predictions for this analyst with lower confidence.")
    return (f"Mixed style: {pc:.0%} persona-consistent, {nt:.0%} narration-triggered, "
            f"{ux:.0%} unexplained across {n} tracked questions.")


def synthesize_personas(intent_path: str = QUESTION_INTENT_PATH,
                        out_path: str | None = PERSONA_DERIVED_PATH,
                        exclude_quarters: set[str] | None = None) -> dict:
    """exclude_quarters: drop these quarters before aggregating -- use this to
    exclude VAL_QUARTER when the result will be used to predict that same
    quarter, otherwise the persona stat leaks the answer it's predicting."""
    with open(intent_path) as f:
        records = json.load(f)
    if exclude_quarters:
        records = [r for r in records if r["quarter"] not in exclude_quarters]

    by_analyst: dict[str, list[str]] = {}
    for r in records:
        by_analyst.setdefault(r["analyst"], []).append(r["intent"])

    result = {}
    for analyst, intents in by_analyst.items():
        n = len(intents)
        pc = intents.count("persona_consistent") / n
        nt = intents.count("narration_triggered") / n
        ux = intents.count("unexplained") / n
        result[analyst] = {
            "n": n,
            "persona_consistent_rate": round(pc, 3),
            "narration_triggered_rate": round(nt, 3),
            "unexplained_rate": round(ux, 3),
            "style_note": _style_note(n, pc, nt, ux),
        }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Synthesized personas for {len(result)} analysts -> {out_path}")
        for a, p in sorted(result.items(), key=lambda x: -x[1]["n"]):
            print(f"  {a:22s} n={p['n']:2d}  PC={p['persona_consistent_rate']:.0%}  "
                  f"NT={p['narration_triggered_rate']:.0%}  UX={p['unexplained_rate']:.0%}")
    return result


if __name__ == "__main__":
    synthesize_personas()
