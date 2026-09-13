# IR Prediction Engine v2/v3 — Results & Honest Ceiling

## Q1FY27 update (August 8, 2026)

Added the Q1FY27 transcript (21 quarters total) and ran it as the new blind holdout (`VAL_QUARTER = "q1fy27"`).

**Bug fix:** `src/engine.py` hardcoded the self-bonus/repeat-topic "previous quarter" as the literal string `"q3fy26"` instead of deriving it from `VAL_QUARTER`. Harmless while q4fy26 was the holdout (q3fy26 genuinely was the prior quarter), silently wrong once the holdout moved. Now computed as `QUARTER_ORDER[VAL_ORD - 1]`.

**Recall-boost mode (N+1) is now the default in `engine.main()`.** It was already implemented and separately reported in this doc, but never wired into the actual production batch run. Cross-validated on all 15 quarters (115 calls, `cross_validation.json`): base F1 40.4%/recall 45.4% → boost F1 45.2%/recall 59.5%. This is a broad, non-overfit win, not tuned on q1fy27.

**Q1FY27 blind result (9 analysts):**

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no boost, no novelty) | 40% | 51% | 40% |
| Boost + novelty signal | 34% | 65% | 42% |
| **Boost only (adopted)** | **40%** | **65%** | **46%** |

**Novelty signal tested and rejected for this quarter.** `python3 run.py novelty --quarter q1fy27` correctly identified 3 real new narration themes (FCNR NRI deposits, AI-transformation strategy update, AT1/senior-debt capital raise), and the engine's extra-slot mechanism added one of them for 6 of the 9 validated analysts — but **0 of those 6 slots matched an actual question topic** this quarter, so it was pure precision drag (F1 46%→42%) with no recall payoff. Per the signal's own documented validation discipline, it's disabled (`data/inputs/narration_novelty.json` removed, not deleted — backed up for reference) rather than kept on faith. Re-test quarter by quarter; don't assume it's dead.

Structural finding worth carrying forward: a signal-coverage check across Q3FY26/Q4FY26/Q1FY27 showed **100% of actual (analyst, topic) instances had at least one supporting signal already in the feature set** (self-history, cross-analyst momentum, or narration). The gap to a higher ceiling is in *ranking and slot-sizing*, not missing raw signal — reinforcing the v2 finding that the recoverable misses need new information (peer-bank transcripts, live-call context) rather than more arithmetic on existing signals.

## Peer-bank signal (August 8, 2026) — tested and rejected for now

Built `src/signals/peer_signal.py` + `run.py peers`: parses peer-bank transcripts (`earnings_transcript/peers/`), tags peer analyst questions against the same 12-topic taxonomy, and computes topic salience (fraction of distinct peer analysts who raised each topic this quarter). Wired into the engine with the *same* extra-slot discipline as narration-novelty — never displaces a validated top-N pick, only adds one slot per analyst, gated by a salience floor and the analyst's own affinity.

Sourced Kotak Mahindra and IndusInd Bank's official Q1FY27 transcripts (verified against the actual company name before use — a BSE-search candidate that looked like an ICICI transcript turned out to be Jindal Saw Ltd's, caught before download). HDFC and ICICI's official transcripts weren't reliably locatable via search; only Kotak + IndusInd are in `earnings_transcript/peers/` for now.

**Bug found and fixed along the way:** `dataset_compiler.py`'s analyst-name-extraction regex only recognized `"line of X from Y"` / `"...of Y"` / `"...,  Y"` separators. Kotak's transcript uses `"line of X **with** Y"`, which silently dropped 8 of 9 analyst blocks (parsed as 1 analyst, 9 QA turns instead of 7 analysts, 96 turns). Added a `with`-separator pattern as a fallback — additive only, confirmed byte-identical re-compile of the existing 21-quarter Axis dataset before relying on it for peer parsing.

Peer salience for Q1FY27: NIM & Yields (1.00, 9 peer analysts), Deposits & CASA (0.67, 6), Loan & Advances Growth / Profitability & Returns (0.44). Result: **F1 46%→45%** (boost-only baseline vs. boost+peer). The slot fired for 8 analysts but only 1 (MB Mahesh) was in the validated set that actually asked this quarter, and it missed (`+Deposits & CASA`, actual was `Loan & Advances Growth`/`NIM & Yields`). Same failure mode as the novelty signal: NIM/CASA are so close to universal analyst interest this quarter that the extra-slot mechanism only fires for the minority of analysts where the base model's own ranking had *already* correctly excluded the topic — adverse selection, not missing signal. Disabled (`data/inputs/peer_signal.json` removed, not deleted) per the same validation discipline. The idea may still be worth it with more peer banks (HDFC/ICICI) or a salience measure that isn't dominated by sector-wide macro themes (e.g. weighting by topics peers discussed that Axis specifically diverges on, not just raw peer volume) — re-test, don't assume dead.

**Adopted config as of this update:** boost mode only (novelty and peer signal both off). Q1FY27 blind: P=40%, R=65%, F1=46%.

## "Why" reasoning layer (August 8, 2026) — question_intent + persona-weight, tested and rejected

Built without Groq (per decision to reserve LLM quota for `predict`): `src/signals/question_intent.py` classifies each historical question as `persona_consistent` / `narration_triggered` / `unexplained` by direct transcript reading (no LLM calls — the classification was done manually, question by question, cross-referencing narration text). Ran across the last 8 quarters (q2fy25–q1fy27), 78 question blocks total. `src/signals/persona_synthesis.py` aggregates this into per-analyst style rates (`data/inputs/analyst_personas_transcript_derived.json`), pure arithmetic, no LLM.

**Real finding, independent of what came next:** 73% of all 78 questions were `narration_triggered` (a direct reaction to something specific management disclosed that quarter), only 18% `persona_consistent`. Some analysts are almost entirely narration-driven — MB Mahesh: 5/6 narration-triggered, 0/6 persona-consistent, which explains why his lifetime F1 has always been hard to improve via pure topic-frequency modeling. This independently validates that `W_NARR` (already the largest weight in `EngineConfig`) was correctly prioritized.

**Attempted signal:** personalize `W_NARR` per analyst — scale it up for narration-driven analysts (like MB Mahesh), down for persona-driven ones, computed with `VAL_QUARTER` always excluded from the persona stats to avoid leaking the answer into the signal predicting it (`PERSONA_WEIGHT=1` env flag, off by default).

**Result: F1 46%→42%, regression.** Root cause: the score formula is multiplicative — `p * (1 + w_narr*narr + ...)` — so boosting `w_narr` amplifies proportional to each topic's *base preference* `p`, not narration alone. For MB Mahesh specifically, `Credit Cost & Provisions` has high `p` (his real historical persona) so the boost helped it pull further ahead of `NIM & Yields` (his actual answer, but a topic with low `p` for him) — the opposite of the intended effect. Disabled (flag defaults off, no file cleanup needed). If revisited: the fix is likely an additive/blended combination of persona-signal and narration-signal instead of a multiplicative amplifier, so a personalized narration weight doesn't get diluted or inverted by an unrelated topic's higher base preference.

**Adopted config unchanged:** boost mode only. Q1FY27 blind: P=40%, R=65%, F1=46%.

## Metrics-anomaly narration signal (August 8, 2026) — tested, neutral, kept as opt-in

Built per the "agentic pipeline" proposal, reframed as deterministic (no LLM) once it became clear Axis's CFO narration is heavily templated: `src/signals/metrics_extractor.py` regex-extracts structured `(metric, value, delta, direction, period)` tuples for 11 canonical metrics (NIM, GNPA, NNPA, PCR, PAT, ROA, ROE, CET-1, Cost-to-Income/Assets, Net Credit Cost) across all 21 quarters (`python3 run.py metrics` → `data/db/metrics_timeseries.json`), then scores how anomalous each metric's move is relative to its own historical `|delta|` distribution (percentile rank, `VAL_QUARTER` excluded from the comparison pool). Wired into `engine.py` as `METRICS_NARR=1`: for topics with a mapped metric, replaces the keyword-sentence-count narration signal (`narr/max_narr`, no sense of magnitude) with the anomaly score; topics without a mapped metric keep the existing signal unchanged.

**Five real extraction bugs found and fixed during spot-checking** (two full quarters verified line-by-line against raw transcript text): `₹` currency symbol not recognized (only `"Rs."` was — this caused the correct bank-level PAT sentence to fail matching entirely and fall through to an unrelated later clause); unconstrained search window let the value-matcher skip past the intended number into a much later, unrelated one; subsidiary bullet points (Axis Finance/AMC/Capital PAT) got picked up as if they were the bank-level figure since sentence-splitting didn't treat `▪`/`o` bullet markers as boundaries; delta phrasing order flips between quarters (`"declined 6 bps QOQ"` vs `"QoQ growth of 9%"`) and only the first form was originally handled; and `_extract_from_sentence` returned on the first metric matched in dict-definition order rather than extracting every metric actually present in a merged sentence, silently dropping later ones (e.g. `Cost_to_Assets` vanishing whenever `GNPA` appeared later in the same run-on sentence). All confirmed fixed against source text after the fix.

**Result: F1 tied at 46% (P=40%, R=65%) on the Q1FY27 blind holdout** — the same as boost-mode-only. Individual predictions did change (e.g. Abhishek Murarka's `Slippages & Asset Quality` pick was replaced by `Loan & Advances Growth`; Kunal Shah's `Slippages` replaced by `Credit Cards & Spends`), so the signal is doing something to the ranking — it just didn't happen to flip a hit into a miss or vice versa on this specific 9-analyst sample. That's a genuinely neutral result, different from novelty/peer-signal's clear regressions — not proven to help, not proven to hurt.

**Update — cross-validated across all 15 historical quarters, and the single-quarter tie was misleading.** Wired `metrics_narr` into `evaluation.py`'s CV harness (`run_cv`, same per-topic-override pattern as `engine.py`) and re-ran boost-mode with vs. without it:

| | F1 | Precision | Recall |
|---|---|---|---|
| Baseline (boost mode) | 45.2% | 40.6% | 59.5% |
| + METRICS_NARR | 43.1% | 38.6% | 57.2% |

**Real regression: -2.1pp overall.** Per-quarter: 10 of 15 quarters got worse (worst: q3fy23 -8.3%, q3fy24 -8.2%, q4fy24 -8.0%), only one meaningfully improved (q2fy24 +11.4%), the rest flat. Q1FY27's earlier "tied at 46%" result was a coincidence of that one quarter's specific mix, not evidence the signal is safe — this is exactly why the project's own discipline requires cross-validation, not a single blind quarter, before trusting a result.

**Decision: rejected.** `METRICS_NARR` stays off by default (already was). The regex extraction pipeline itself (`metrics_extractor.py`) is solid and reusable — the five bugs found were real and are fixed — but replacing the keyword-count narration signal with metric-anomaly scores, in this multiplicative formula, net-hurts more than it helps. Likely cause, consistent with the earlier persona-weight rejection: the score formula's multiplicative structure (`p * (1 + w_narr*narr + ...)`) keeps rewarding whichever topic already has high base preference `p`, and a sharper narration signal doesn't fix that — it just changes which already-favored topic wins.

## Stage 3 — retrieval-grounded prediction agent (August 8, 2026) — promising, NOT cross-validated

Three signals in a row (novelty, peer-signal, persona-weight, metrics-narr) hit the same wall: the multiplicative score formula amplifies whichever topic already has high base preference, so a sharper narration/why signal just changes which already-favored topic wins. Stage 3 was built to sidestep that formula entirely rather than add another term to it: `src/signals/agent_predict.py` does deterministic retrieval (per-analyst style from `persona_synthesis.py`, this quarter's metric anomalies from `metrics_extractor.py`, and the analyst's own historical questions on those anomalous topics — all computed with `VAL_QUARTER` excluded to avoid leakage), then a single bounded Groq call may propose **at most one swap** into the formula's own top-N picks, only if it can cite both a specific anomalous metric and a specific historical precedent. This is the evolution of `engine.py`'s existing `_llm_select_topics` (tried once before with a weaker model and thinner context, made things worse) — same idea, stronger model, much richer grounding.

Gated behind `AGENT_PREDICT=1`. Only applies to analysts with `n >= 3` in their (VAL_QUARTER-excluded) persona stats and at least one historical precedent on a currently-anomalous topic — most analysts get no swap proposed at all, by design.

**Result on Q1FY27 (9 validated analysts): F1 46%→50%, Precision 40%→43%, Recall 65%→70% — improved across all three metrics**, not just recall. Three swaps fired total; one landed directly in the validated set: Mahrukh Adajania's swap (`-Loan & Advances Growth +Opex & Cost-to-Income`, citing her Q3FY26 precedent question on Opex plus this quarter's 0.92 anomaly score) converted a partial miss into an exact match — her actual questions were precisely `NIM & Yields` + `Opex & Cost-to-Income`.

**Explicit caveat — this is NOT the same confidence level as the other results in this document.** Per an explicit cost/risk tradeoff decision with the user, this was tested on Q1FY27 only, not cross-validated across historical quarters — the exact same single-quarter-testing pattern that made `METRICS_NARR` look neutral/safe before cross-validation revealed a real -2.1pp regression. One good quarter is encouraging, not proof. **Do not promote this to default without cross-validating first** — the next session should budget for that (~11 analysts × N quarters of Groq calls) before trusting this result at the same level as the boost-mode fix.

**Adopted config unchanged pending cross-validation:** boost mode only (`AGENT_PREDICT` opt-in, promising but unproven). Q1FY27 blind, cross-validated config: P=40%, R=65%, F1=46%. Q1FY27 blind, with AGENT_PREDICT=1 (not cross-validated): P=43%, R=70%, F1=50%.

**Second data point — Q4FY25, requested as a partial cross-validation check.** Baseline (boost-mode only): 11 validated analysts, P=48%, R=77%, F1=58% — a notably stronger quarter for the base model than Q1FY27. With `AGENT_PREDICT=1`: **identical result, F1=58%/P=48%/R=77%, because zero swaps fired.** This quarter's metric anomaly scores topped out at 0.67 (`Credit Cost & Provisions`) — every topic stayed under the 0.7 threshold, so `retrieve_context` correctly returned no candidates for any analyst and the LLM was never even called. This confirms the gating is working as designed (no evidence → no forced swap, not a spurious swap that happens to wash out) but it means Q4FY25 doesn't add new evidence either way about whether the swap mechanism helps — it simply never activated. Q1FY27 remains the only quarter where the mechanism has actually fired and been checked against ground truth. A real test of a second *active* case still requires a quarter with at least one metric anomaly ≥0.7, which isn't guaranteed by picking a quarter at random.

**Adopted config unchanged:** boost mode only. Q1FY27 blind: P=40%, R=65%, F1=46%.

## Negative results log (July 2, 2026) — tested and rejected

Kept for the record so these aren't re-tried blindly. Both are implemented in `src/evaluation.py` behind `Config.novelty` / `Config.cooc` flags, **off by default**:

1. **LLM topic re-rank** (llama-3.1-8b selecting from top-8 scored candidates): Q4FY26 blind F1 59.4% → 50%. It swapped scored hits for misses. Disabled by default in the engine (`LLM_TOPIC_SELECT=1` to re-test with a stronger model).
2. **Narration-novelty boost** (topics new/rising in this quarter's narration vs prior 3): no weight setting helped; tuning-window F1 fell at every W_NOV. Root cause: keyword tagging can't distinguish a genuinely new theme (the Q4 RAROC slide) from routine boilerplate — Profitability keywords appear in narration every single quarter. The idea needs *semantic* narration comparison (LLM/embedding pass), not keyword counts.
3. **Topic co-occurrence boost** (boost topics that historically bundle with the top-scored picks): small tuning gain at W_COOC=0.4 but holdout fell 46.3% → 44.5% and larger weights collapsed Q4 (59% → 46%). The bundling signal is already largely captured by the analyst profile itself.

Working hypothesis after these: the remaining recoverable misses need *new information* (peer-bank transcripts, results-surprise deltas, semantic narration analysis), not new arithmetic on existing signals.

## v3 update: 12-topic taxonomy (July 2, 2026)

Added two topics the old taxonomy couldn't represent: **Profitability & Returns** (RoE/RoA/RAROC — 21 historical questions) and **Strategy & Competitive Positioning** (market share, vs-peers — 29 questions). All 190 questions and narration segments re-tagged via `graph_compiler.py` (recompiled `graph.json`; old graph backed up to `graph_v1_backup.json`). General-only questions fell 23 → 20. Name canonicalization now happens at compile time too.

**Read the v3 numbers carefully — the metric got stricter, not the engine worse.** With more labels per question, misses that were previously invisible now count against recall. The 12-topic numbers are *not comparable* to the 10-topic v1/v2 numbers; they converge toward the honest concern-level semantic score (55.3%), which was the point.

| Metric (12-topic taxonomy) | v3 |
|---|---|
| Cross-val topic F1 (re-tuned: DECAY 0.40, W_NARR 3.5) | 40.5% |
| Cross-val F1 / recall, boost mode | 45.1% / 59.0% |
| Q4FY26 blind F1 / recall | 59.4% / 61.1% |
| Q4FY26 boost-mode recall | 65.3% |

What the new taxonomy caught immediately: Abhishek Murarka's Q4FY26 RAROC/growth-mix concern is now **predicted and hit** (persona prior + re-tagged history). What it exposed: Chintan Joshi's and MB Mahesh's Q4 profitability questions were reactions to a RAROC slide management presented *on the call itself* — the textbook irreducible case. Expect new-topic prediction to improve over the next 2–3 quarters as tagged history accumulates for these topics.

---

# v2 results (10-topic taxonomy) — for reference

*July 2, 2026 — improvements to the Graph RAG analyst-question prediction engine, with all claims re-verified against actual outputs.*

## Headline numbers (before → after)

| Metric | v1 | v2 | Delta |
|---|---|---|---|
| Cross-val topic F1 (14 quarters, leave-one-out) | 40.4% | **43.0%** | +2.6 |
| Cross-val F1, recall-boost mode (N+1) | 43.6% | **46.9%** | +3.3 |
| Cross-val recall, boost mode | — | **64.7%** | — |
| Q4FY26 blind topic F1 | 57.1% | **62.7%** | +5.6 |
| Q4FY26 blind recall | 72.2% | **77.8%** (81.9% boosted) | +5.6 |
| MB Mahesh lifetime F1 | 15.2% | **29.5%** | ~2x |
| Kunal Shah lifetime F1 | 45.2% | **58.1%** | +12.9 |
| Mahrukh Adajania lifetime F1 | 48.1% | **58.2%** | +10.1 |

Hyperparameters were tuned only on q3fy23–q2fy26; q3fy26 and q4fy26 were never touched during tuning. On that untouched holdout, F1 went 50.0% → 52.1%.

## What changed

1. **Canonical analyst names.** Transcript parsing had split analysts into fragments ("Ma hrukh Adajania", "Sam eer Bhise", "Sum eet Kariwala", "Harsh Modi"/"Harsh Wardhan Modi", "Krishnan"/"Krishnan ASV", "Nilanjan"/"Nilanjan Karfa"). Merging them repairs the affected profiles. This alone was worth +0.8 F1.
2. **Ordinal recency.** The sort-key gap between fiscal years (224 → 231) made recency decay uneven; quarters now age uniformly, decay retuned (λ 0.40 → 0.25 per quarter).
3. **Global base-rate smoothing.** Analyst profiles blend 35% with the market-wide topic distribution (doubled for analysts with <3 quarters of history). This is what fixed the data-sparsity problem — sparse analysts now get a sensible prior instead of near-random guesses.
4. **Web-derived personas** (`analyst_personas.json`). 15 recurring analysts profiled from public research coverage (firm, style, coarse topic priors). Used as a weak prior that shrinks as transcript history accumulates, so it mostly helps new/sparse analysts and never overrides observed behaviour.
5. **Two-quarter cross-analyst momentum** (1.0/0.5 weighted) instead of one.
6. **Per-analyst repeat propensity.** The self-bonus is now scaled by each analyst's measured probability of repeating a topic from the prior quarter — rotators like MB Mahesh no longer get a misleading repeat bonus. This (plus smoothing) is why his lifetime F1 doubled.
7. **Honest tuning protocol.** Grid search runs only on q3fy23–q2fy26 inside the new `cross_validation.py` harness; the last two quarters stay as a pure holdout.

## The LLM judge — a correction

The v1 write-up estimated a real LLM judge would add +15–20 points over keyword-based semantic scoring. Groq/Gemini were unreachable, so Claude judged all pred-vs-actual pairs directly (same 0/0.5/1.0 rubric), decomposing each actual question block into its distinct concerns. Result: **semantic F1 = 55.3%** (P 54.9%, R 56.9%) — *below* topic F1, not above it.

The estimate was wrong for an instructive reason: the keyword scorer awarded a 0.5 bonus just for a matching topic label, inflating its baseline; and a real judge is stricter on recall because it sees concerns the 10-topic taxonomy cannot represent. MB Mahesh's RAROC/ROE question and Chintan Joshi's "corporate growth at 34% — what do you see that peers don't?" question have no bucket, so topic-F1 silently ignores them while the judge correctly counts them as misses. Full pair scores and judge notes are in `semantic_eval_claude.json`.

Two taxonomy additions would likely capture most of these: **Profitability / RoE / RAROC** and **Strategy & Competitive Positioning**.

## The honest ceiling, restated

The structural analysis in v1 stands, with sharper numbers. Roughly 30% of questions remain genuinely unpredictable (novel concerns, live follow-ups on management remarks, same-morning macro events). What changed is where v2 sits against that ceiling: at 77.8% blind recall (81.9% in boost mode), management now walks in prepped for about 4 out of 5 actual question-topics — up from 3 out of 4. The remaining gap is concentrated in exactly the places the ceiling predicts: off-taxonomy concerns and single-question rotators.

## Files

| File | Status |
|---|---|
| `predictive_graph_rag.py` | Updated: v2 scoring, canonical names, personas, portable paths, current Gemini model |
| `cross_validation.py` | New: leave-one-quarter-out harness with feature flags and holdout-safe grid search |
| `analyst_personas.json` | New: 15 web-derived analyst personas |
| `semantic_eval_claude.json` | New: concern-level judge scores for Q4FY26 |
| `semantic_evaluator.py` | Updated: current Gemini model, portable paths (for re-runs on your machine) |
| `predictions.json` | Regenerated with v2 engine (v1 backed up to `predictions_groq_backup.json`) |
| `cross_validation.json`, `semantic_eval.json` | Regenerated with v2 numbers (dashboard-compatible schema) |
| `ir_dashboard.html`, `ir_prep_sheet.json` | Regenerated from v2 predictions |

Note: `predictions.json` question *text* is currently from the heuristic templates because Groq/Gemini aren't reachable from this environment — re-run `python3 predictive_graph_rag.py` on your machine to regenerate LLM-written question text on top of the improved topic selection.
