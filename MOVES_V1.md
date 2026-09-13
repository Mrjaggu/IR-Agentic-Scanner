# Move-level prediction — v1 results

## Why this exists

Topic recall on the held-out calls was 92.6% and the improvement loop proposed
nothing. That was not the loop failing. It was the loop converged on a target
that had saturated: with a 12-topic taxonomy and 8 slots per analyst, hitting
the bucket is nearly free, and management already knows the topics. A brief
saying "NIM will come up" is worth nothing.

What management cannot prepare for is the **angle**. Measured on the two
held-out calls, after cleaning the concern decomposer:

| | Q4FY26 | Q1FY27 | both |
|---|---|---|---|
| genuine questions asked | 18 | 30 | 48 |
| of those, move-bearing | 8 | 14 | 22 |

So 22 of 48 questions do something management would not have prepared for from
the topic list alone. Those are the target set.

## The taxonomy

The load-bearing observation: **the surprising question is surprising in its
target, not in its form.** Analysts reuse a small, stable set of moves. That is
what makes this tractable.

| move | what it does | input it requires |
|---|---|---|
| `guidance_callback` | presses a promise made on an earlier call | an open commitment |
| `quantification_demand` | management said it in words; give me the number | a qualitative claim |
| `one_off_vs_steady_state` | does this repeat? | an anomalous metric |
| `normalized_run_rate` | what should we model, stripped of one-offs | one-off language |
| `cross_line_bridge` | these two disclosures don't tie | two related metrics |
| `absence_probe` | you didn't mention this | an undisclosed mover |
| `forward_trajectory` | you gave the level; give the trajectory | a disclosed metric |

Per-analyst propensities are learned from **training quarters only** (513
questions, 19–20 quarters, Laplace-smoothed against the global rate because
most analysts appear on only a handful of calls). Real, interpretable lift:

    MB Mahesh            quantification_demand   x3.4
    Abhishek Murarka     quantification_demand   x2.9
    Rikin Shah           normalized_run_rate     x3.1
    Jai Mundhra          one_off_vs_steady_state x2.6
    Mahrukh Adajania     one_off_vs_steady_state x2.0
    Kunal Shah           guidance_callback       x1.9

## The missing data layer

`guidance_callback` was tied for the most common high-value move **and was
structurally impossible to generate**: the Section 2 commitments store was a
stub, so the framer never saw a single outstanding promise. No amount of prompt
iteration could have fixed that, which is most of why the loop looked inert.

`src/data/commitments.py` now extracts dated, quantified management promises
from prepared remarks *and* Q&A answers, and scores how callable each is as of a
cutoff (`due` / `stale` / `open`, scaled by recency, weighted up when it carries
a number). As of Q4FY26 it surfaces 36 live commitments across 8 topics, ranked
#1 being:

> Q1 FY26, 3 quarters old, stale — "Our stated position is we are confident
> that we can deliver a 3.8% margin on a through cycle basis."

That is the exact promise Rikin Shah pressed in Q4FY26 and Mahrukh Adajania
pressed in Q1FY27 — both of which the old pipeline missed. Of the 6
`guidance_callback` questions across both held-out calls, 5 had their underlying
commitment present in the store as of the correct rolling cutoff.

## Generation

Slots are now **(topic × move)** pairs, not topics. A move is only offered when
its required input exists for that topic — a `guidance_callback` with no
commitment behind it is a fluent invention, and the seen-set == accepted-set
rule still holds because the commitment's own words go into the prompt.

Mahrukh's Q1FY27 slate went from 8 topic slots to 14 (topic × move) slots, and
now includes `NIM & Yields × guidance_callback` carrying the 3.8% commitment —
her actual first question.

## Results (rolling cutoff, walk-forward)

    Q4FY26  trained through q3fy26   move recall 0.455
    Q1FY27  trained through q4fy26   move recall 0.471
    mean 0.463   spread 0.016

Against topic recall of 92.6% on the same calls. **That gap is the honest state
of the system.**

Per move, both calls:

    one_off_vs_steady_state   3/3   1.00
    forward_trajectory        6/7   0.86
    guidance_callback         4/6   0.67
    absence_probe             0/2   0.00
    cross_line_bridge         0/2   0.00
    normalized_run_rate       0/3   0.00
    quantification_demand     0/5   0.00

## What the metric refuses to do

An earlier draft excluded off-taxonomy concerns from the denominator and
reported **1.00** on Q4FY26 while missing 6 of 11 questions — the same
self-flattery that made topic recall useless, reproduced one level up. Headline
recall now counts every concern analysts actually raised. `move_recall_on_taxonomy`
is kept as a diagnostic only.

Two further honesty notes:

- A "hit" means the (analyst, topic, move) slot was allocated, not that the
  generated text nails it. **0.463 is an upper bound on end-to-end move recall**;
  the generation axis needs an LLM run to measure (Groq is 403'd at the proxy
  in this environment).
- 9 of 15 misses are `off_taxonomy`, and some of those are the question-side
  topic attributor being weak rather than a genuine gap. The denominator needs
  work before 0.463 is treated as final.

## Every miss now has an owner

This is the part the loop was missing. Before, every miss landed in
`reasoning_weakness` and the loop dutifully rewrote prompts against gaps that
were structural.

| verdict | n | who owns the fix |
|---|---|---|
| `off_taxonomy` | 9 | taxonomy — no Regulatory/PSL bucket, nothing for balance-sheet composition |
| `input_missing` | 5 | **data layer — do not touch prompts** |
| `slot_not_allocated` | 1 | ranking — slot budget / propensity / per-topic cap |
| `generation_failure` | 0 | prompt or skill — the only bucket the loop should act on |

A loop iteration that proposes a prompt change while `input_missing` dominates
is optimising the wrong layer and should be blocked, not promoted.

## Next, in order of measured leverage

1. **Taxonomy** (9 misses). Add Regulatory & Policy (PSL, ECLGS, RBI drafts) and
   a balance-sheet-composition bucket (foreign loans, period-end vs average).
   Strengthen question-side topic attribution.
2. **`quantification_demand` input** (0/5, 5 misses). Needs the qualitative-claim
   detector to run over management's *answers*, not only prepared remarks — the
   claim analysts pounce on is usually made live.
3. **`normalized_run_rate` input** (0/3). One-off language must be picked up from
   the numbers themselves (a delta that reverses trend), not only from
   management volunteering the word "reversal".
4. **Ranking** (1 miss). Lowest leverage; leave it.

## Reproducing

    python3 -m src.agentic.run_move_eval     # writes data/outputs/move_eval.json

## Files

    src/agentic/moves.py           taxonomy, detection, per-analyst profiles
    src/agentic/move_planner.py    per-move evidence assembly, slot planning
    src/agentic/move_eval.py       move recall + layer-owning miss attribution
    src/agentic/run_move_eval.py   walk-forward harness over the held-out calls
    src/data/commitments.py        dated commitments, status as of a cutoff
    src/agentic/question_framer.py move-aware prompts (frame_move_questions)
    src/agentic/question_eval.py   concern decomposer (rewritten, v2)
