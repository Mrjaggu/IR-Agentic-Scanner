#!/usr/bin/env python3
"""
scripts/extract_analyst_patterns.py

Mines this bank's own transcript history to build per-analyst COGNITIVE
patterns -- not "Kunal asks about NIM" but "Kunal's trigger is a stated
trade-off between two management objectives; his question tests whether
pursuing one hurts the other."

Why this exists: measured question-recall (src/agentic/eval_harness.py,
score_question_recall) came out to ~0.47 on Q1FY27 even though topic recall
is ~93% -- the framer reliably guesses WHICH topic an analyst will raise but
not the specific numeric angle or follow-up logic they use once there. This
script is the evidence-extraction step behind fixing that: for every analyst
with enough question history, feed their full chronological Q&A -- paired
with the deterministic metric deltas (metrics_extractor.py) and the on-topic
management narration available at the time -- to an LLM and ask it to name
the recurring trigger -> reasoning -> question-style -> follow-up pattern,
grounded in at least two actual quarters per pattern (cited).

Scope note: this only covers what's actually in the pipeline. Axis has full
history (21 quarters); Kotak/IndusInd currently have one quarter each
(Q1FY27) in earnings_transcript/peers/ -- not enough to fingerprint anyone
cross-bank. Cross-bank validation against HDFC/ICICI/SBI/Federal (as sketched
in chat) would require ingesting THOSE banks' own transcript archives through
this same per-bank pipeline (src/config/banks.py's BANKS registry) first --
that data isn't in this corpus today, so this run is Axis-only.

Usage:
    python3 scripts/extract_analyst_patterns.py [--bank axis] [--min-questions 4]

Output:
    data/analysis/<bank>_analyst_patterns.json   -- per-analyst LLM-synthesized patterns
    data/analysis/<bank>_metric_correlations.json -- deterministic pairwise metric correlation (no LLM)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config.settings import paths_for
from src.model_provider.llm_client import LLMClient
from src.agentic.question_framer import _narration_for_topic


def load_bank_data(bank_id: str):
    paths = paths_for(bank_id)
    graph = json.load(open(paths.graph_path))
    metrics_path = os.path.join(os.path.dirname(paths.graph_path), "metrics_timeseries.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return graph, metrics


def build_analyst_history(graph: dict, min_questions: int) -> dict[str, list[dict]]:
    qnodes = [n["properties"] for n in graph["nodes"] if n["type"] == "Question"]
    by_analyst = defaultdict(list)
    for q in qnodes:
        by_analyst[q["analyst"]].append(q)
    out = {}
    for analyst, qs in by_analyst.items():
        if len(qs) < min_questions:
            continue
        qs.sort(key=lambda x: x["quarter"])
        out[analyst] = qs
    return out


def _fmt_metrics(metrics_for_q: dict) -> str:
    if not metrics_for_q:
        return "(no structured metrics extracted this quarter)"
    lines = []
    for metric, rec in metrics_for_q.items():
        lines.append(f"{metric}: {rec.get('value')}{rec.get('value_unit') or ''} "
                     f"({rec.get('direction')} {rec.get('delta')}{rec.get('delta_unit') or ''} {rec.get('period') or ''})")
    return "; ".join(lines)


def build_prompt(analyst: str, history: list[dict], graph: dict, metrics: dict) -> str:
    blocks = []
    for q in history:
        quarter = q["quarter"]
        m = _fmt_metrics(metrics.get(quarter, {}))
        narr_bits = []
        for t in q.get("topics", []):
            narr_bits += _narration_for_topic(graph, quarter, t, max_snips=2)
        narr = " | ".join(dict.fromkeys(narr_bits))[:600] or "(no matching narration snippet)"
        blocks.append(
            f"### {quarter}\n"
            f"Metrics disclosed: {m}\n"
            f"Management narration touching this question's topics: {narr}\n"
            f"Question asked: \"{q['text'][:900]}\""
        )

    evidence = "\n\n".join(blocks)
    return f"""You are analyzing one equity analyst's REAL question history on a bank's earnings
calls, across {len(history)} quarters, to find recurring COGNITIVE patterns -- not
which topics they ask about, but the reasoning shape: what data or management
statement provokes them, how they connect it to something else, and what kind
of question results.

Analyst: {analyst}

Chronological evidence (quarter, the metric deltas disclosed that quarter, the
on-topic management narration available, and their actual question):

{evidence}

Identify 3 to 5 recurring patterns. Each pattern must be grounded in AT LEAST
TWO of the quarters above -- cite them. Do not invent behavior not evidenced
in the text given. If you only find one or two solid patterns, return fewer --
do not pad to reach 5.

Return strict JSON only, this shape:
{{
  "analyst": "{analyst}",
  "patterns": [
    {{
      "trigger": "what kind of data/statement sets this off",
      "reasoning_pattern": "how they connect trigger to conclusion, one sentence",
      "question_style": "the sentence shape/opener they tend to use, e.g. 'If X, does that mean Y?'",
      "typical_follow_up": "what they usually ask next once the first question is answered, or null",
      "evidence_quarters": ["q1fy27", "q3fy24"]
    }}
  ]
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="axis")
    ap.add_argument("--min-questions", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    graph, metrics = load_bank_data(args.bank)
    history = build_analyst_history(graph, args.min_questions)
    print(f"{len(history)} analysts with >= {args.min_questions} questions in {args.bank} history:")
    for a, qs in sorted(history.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a}: {len(qs)} questions across {qs[0]['quarter']}..{qs[-1]['quarter']}")

    client = LLMClient()
    active = client.probe_llm()
    if not active:
        print("\nNo LLM provider reachable from this environment -- run this on a machine "
              "with working API access (same requirement as the live server). Aborting.")
        sys.exit(1)
    print(f"\nUsing {active} for pattern synthesis.\n")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "data", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.bank}_analyst_patterns.json")

    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
        except Exception:
            results = {}

    for analyst, qs in history.items():
        if analyst in results:
            print(f"skip {analyst} (already extracted)")
            continue
        print(f"extracting patterns for {analyst} ({len(qs)} questions)...")
        prompt = build_prompt(analyst, qs, graph, metrics)
        raw = client.call_llm(prompt, temperature=0.1)
        if not raw:
            print(f"  !! no response for {analyst}, skipping")
            continue
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  !! JSON parse failed for {analyst}: {e}")
            parsed = {"analyst": analyst, "patterns": [], "_raw": raw[:2000]}
        results[analyst] = parsed
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"  -> {len(parsed.get('patterns', []))} patterns saved")

    print(f"\nSaved analyst patterns to {out_path}")

    # ── Deterministic metric-pair correlation (no LLM) ──────────────────────
    corr_path = os.path.join(out_dir, f"{args.bank}_metric_correlations.json")
    series = defaultdict(dict)
    for quarter, recs in metrics.items():
        for metric, rec in recs.items():
            sign = 1.0 if rec.get("direction") in ("increase", "improve") else -1.0
            delta = rec.get("delta")
            if delta is not None:
                series[metric][quarter] = sign * delta

    metric_names = sorted(series.keys())
    correlations = []
    for i, m1 in enumerate(metric_names):
        for m2 in metric_names[i + 1:]:
            common = sorted(set(series[m1]) & set(series[m2]))
            if len(common) < 8:
                continue
            xs = [series[m1][q] for q in common]
            ys = [series[m2][q] for q in common]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
            sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
            if sx == 0 or sy == 0:
                continue
            r = round(cov / (sx * sy), 3)
            correlations.append({"metric_a": m1, "metric_b": m2, "correlation": r, "n_quarters": n})

    correlations.sort(key=lambda c: -abs(c["correlation"]))
    json.dump({"bank": args.bank, "correlations": correlations}, open(corr_path, "w"), indent=2)
    print(f"Saved metric correlations to {corr_path}")
    print("\nTop correlated metric pairs:")
    for c in correlations[:10]:
        print(f"  {c['metric_a']} <-> {c['metric_b']}: r={c['correlation']} (n={c['n_quarters']})")


if __name__ == "__main__":
    main()
