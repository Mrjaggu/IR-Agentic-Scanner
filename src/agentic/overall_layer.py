"""
overall_layer.py — Section 3.2: Researcher -> Analyst -> Verifier crew.

Produces topics ranked as likely to come up "regardless of who asks" — driven
by KPI anomalies, graph history, and commitments. Runs independent of any
specific analyst.

Researcher: deterministic gathering (evidence bundles from the planning
agent + cross-analyst momentum + global base rates). No LLM call.

Analyst: one bounded LLM call that ranks topics and must cite specific
evidence (a number, a quarter, a base-rate) per topic.

Verifier: deterministic grounding check on the Analyst's citations (Section
3.5's discipline applied here too, not just to per-analyst questions) — a
citation that doesn't match anything in the evidence bundle is dropped and
the topic keeps only its bare label. Reject-and-regenerate capped at 2
retries for the batch as a whole.
"""

import json
import re

from src.config.settings import TOPICS_LIST

MAX_RETRIES = 2
TOP_K = 8

# Deterministic composite weights used BOTH to order candidates before the LLM
# sees them and as the ranking when no LLM is reachable (an air-gapped
# deployment is a supported mode here, not a degraded one).
#
# These weights are set a priori from what each signal means and are NOT tuned
# on q4fy26/q1fy27 -- tuning them on the held-out quarters would recreate
# exactly the single-quarter overfitting failure this redesign exists to fix.
W_ANOMALY = 0.35       # how unusual this quarter's disclosed metric move is
W_DISCLOSURE = 0.25    # how much the upcoming disclosure actually talks about it
W_MOMENTUM = 0.15      # what the street asked the last two quarters
W_BASE_RATE = 0.15     # long-run frequency of the topic
W_DRILL_FLAG = 0.10    # disclosure sentence shaped like it invites scrutiny


# Named so a candidate weight set (Framework Loop / backtest_and_promote_weights
# in eval_harness.py) can be described and logged the same way the production
# defaults are, rather than as a positional tuple.
DEFAULT_WEIGHTS = {
    "anomaly": W_ANOMALY, "disclosure": W_DISCLOSURE, "momentum": W_MOMENTUM,
    "base_rate": W_BASE_RATE, "drill_flag": W_DRILL_FLAG,
}


def composite_scores(candidate_topics: list[str], anomaly_scores: dict, global_rate: dict,
                     momentum: dict, disclosure: dict | None,
                     weights: dict[str, float] | None = None) -> dict[str, float]:
    """One transparent score per topic, from named signals with fixed weights
    (or an override set, so the Framework Loop can propose and backtest a
    different weighting WITHOUT touching the module defaults production
    actually runs on -- see eval_harness.py::backtest_and_promote_weights)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    max_momentum = max(momentum.values()) if momentum else 1.0
    max_rate = max(global_rate.values()) if global_rate else 1.0
    salience = (disclosure or {}).get("topic_salience", {}) or {}
    flagged = set()
    for flag in (disclosure or {}).get("drill_down_flags", []) or []:
        flagged.update(flag.get("topics", []))

    scores = {}
    for t in candidate_topics:
        scores[t] = (
            w["anomaly"] * anomaly_scores.get(t, 0.0)
            + w["disclosure"] * salience.get(t, 0.0)
            + w["momentum"] * (momentum.get(t, 0.0) / max_momentum if max_momentum else 0.0)
            + w["base_rate"] * (global_rate.get(t, 0.0) / max_rate if max_rate else 0.0)
            + w["drill_flag"] * (1.0 if t in flagged else 0.0)
        )
    return scores


def _format_evidence(topic: str, evidence_bundles: dict, global_rate: dict,
                     momentum: dict, max_momentum: float) -> str:
    bundle = evidence_bundles.get(topic, {})
    lines = [f"- global historical base rate: {global_rate.get(topic, 0.0):.1%}"]
    if topic in momentum:
        lines.append(f"- raised by peers in the last 2 quarters (momentum score {momentum[topic] / max(max_momentum, 1):.2f})")
    if "history" in bundle:
        lines.append(f"- raised {bundle['history']['times_raised_in_history']} times across historical transcripts")
    if bundle.get("commitments"):
        for c in bundle["commitments"][:2]:
            lines.append(f"- commitment/plan language in {c['quarter']}: \"{c['text']}\"")
    if bundle.get("policy"):
        lines.append(f"- policy signal: {bundle['policy']['note']}")
    if bundle.get("disclosure"):
        d = bundle["disclosure"]
        if d.get("salience") is not None:
            lines.append(f"- the UPCOMING disclosure discusses this topic (salience {d['salience']:.2f} of the most-discussed topic)")
        for flag in d.get("flags", [])[:2]:
            lines.append(f"- disclosure sentence flagged {'/'.join(flag['reasons'])} "
                         f"({flag['numeric_components']} numeric components): \"{flag['sentence'][:220]}\"")
    return "\n".join(lines)


def _build_rank_prompt(candidate_topics: list[str], evidence_bundles: dict,
                       global_rate: dict, momentum: dict, max_momentum: float,
                       anomaly_scores: dict, bank_name: str = "Axis Bank") -> str:
    blocks = []
    for t in candidate_topics:
        anomaly = anomaly_scores.get(t)
        anomaly_line = f"\n- THIS QUARTER'S metric anomaly score: {anomaly:.2f} (1.0 = biggest move ever)" if anomaly is not None else ""
        blocks.append(f"### {t}{anomaly_line}\n{_format_evidence(t, evidence_bundles, global_rate, momentum, max_momentum)}")
    evidence_text = "\n\n".join(blocks)
    return f"""You are ranking topics likely to come up on {bank_name}'s upcoming earnings
call, regardless of which specific analyst asks. Base this ONLY on the evidence
given per topic below — do not invent numbers or facts not listed.

{evidence_text}

Task: rank these topics from most to least likely to be raised this quarter.
For each, give a ONE-SENTENCE rationale that explicitly cites one piece of the
evidence above (a percentage, a count, a quarter, or the anomaly score) — do
not use vague language like "concerns about" without a specific number.

Return JSON only: {{"ranked": [{{"topic": "<exact topic name>", "rationale": "<one sentence citing evidence>"}}, ...]}}
Order the list from most to least likely. Include every topic given above."""


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _evidence_numbers_for_topic(topic: str, evidence_bundles: dict, global_rate: dict,
                                momentum: dict, anomaly_scores: dict) -> set[str]:
    """All numbers that legitimately appear in this topic's evidence, so the
    Verifier can check the Analyst didn't cite a number from nowhere."""
    nums = set()
    gr = global_rate.get(topic, 0.0)
    nums.add(f"{gr * 100:.0f}")
    nums.add(f"{gr * 100:.1f}")
    bundle = evidence_bundles.get(topic, {})
    if "history" in bundle:
        nums.add(str(bundle["history"]["times_raised_in_history"]))
    for c in bundle.get("commitments", []):
        nums.update(_extract_numbers(c["text"]))
    if topic in anomaly_scores:
        a = anomaly_scores[topic]
        nums.add(f"{a:.2f}"); nums.add(f"{a:.1f}"); nums.add(f"{a * 100:.0f}")
    if topic in momentum:
        nums.update(_extract_numbers(str(momentum[topic])))
    return nums


def _verify_rationale(topic: str, rationale: str, evidence_bundles: dict, global_rate: dict,
                      momentum: dict, anomaly_scores: dict) -> bool:
    cited = set(_extract_numbers(rationale))
    if not cited:
        # No number cited at all -- doesn't meet "cite specific evidence".
        return False
    legit = _evidence_numbers_for_topic(topic, evidence_bundles, global_rate, momentum, anomaly_scores)
    return bool(cited & legit)


def build_overall_topics(evidence_bundles: dict, anomaly_scores: dict, global_rate: dict,
                         momentum: dict, client, disclosure: dict | None = None,
                         weights: dict[str, float] | None = None,
                         bank_name: str = "Axis Bank") -> dict:
    """disclosure: the parsed upcoming-quarter document (src/data/upcoming.py).
    When present, topics the disclosure actually talks about or flags become
    candidates in their own right -- that is how a theme with no historical
    base rate (a subsidiary result, a one-off charge) gets onto the brief at
    all, which pure history-based ranking structurally cannot do.

    weights: optional override of the five composite weights (see
    DEFAULT_WEIGHTS above) -- production always omits this; only the
    Framework Loop's backtest passes a candidate set, and only on a copy of
    the pipeline it runs for comparison, never on the live path."""
    max_momentum = max(momentum.values()) if momentum else 1.0

    salience = (disclosure or {}).get("topic_salience", {}) or {}
    flagged_topics = []
    for flag in (disclosure or {}).get("drill_down_flags", []) or []:
        flagged_topics.extend(flag.get("topics", []))

    # Candidates: anomalous topics + disclosure-salient/flagged topics + top
    # base-rate topics. Attach disclosure evidence to each bundle so the
    # Analyst node and the Verifier both see it.
    # Flagged/salient topics are NOT restricted to the fixed 12-topic taxonomy
    # here -- a genuinely new theme the disclosure introduces (an FCNR
    # liquidity opportunity, a one-off charge) only ever reaches drill-down
    # flags / salience with a synthetic ad-hoc label (see
    # src/data/upcoming.py::attach_novel_themes), never as a bare taxonomy
    # name, so gating on `t in TOPICS_LIST` here is exactly what made those
    # themes structurally unrepresentable regardless of how strongly the
    # disclosure signalled them.
    candidates = list(dict.fromkeys(
        list(anomaly_scores.keys())
        + flagged_topics
        + [t for t, v in sorted(salience.items(), key=lambda x: -x[1]) if v >= 0.25]
        + sorted(TOPICS_LIST, key=lambda t: -global_rate.get(t, 0.0))
    ))

    if disclosure:
        for t in candidates:
            t_flags = [f for f in (disclosure.get("drill_down_flags") or []) if t in (f.get("topics") or [])]
            if t in salience or t_flags:
                evidence_bundles.setdefault(t, {})["disclosure"] = {
                    "salience": salience.get(t),
                    "flags": t_flags,
                }

    # Order by the transparent composite score, then cap.
    comp = composite_scores(candidates, anomaly_scores, global_rate, momentum, disclosure, weights=weights)
    candidates = sorted(candidates, key=lambda t: -comp[t])[:TOP_K]

    verifier_log = []
    to_rank = list(candidates)
    final_rationale = {}
    ranked_order = []
    attempt = 0

    while to_rank and attempt <= MAX_RETRIES:
        prompt = _build_rank_prompt(to_rank, evidence_bundles, global_rate, momentum,
                                    max_momentum, anomaly_scores, bank_name=bank_name)
        raw = client.call_llm(prompt, temperature=0.1)
        parsed = None
        if raw:
            clean = re.sub(r"^```(?:json)?\s*|```\s*$", "", raw.strip(), flags=re.MULTILINE)
            try:
                parsed = json.loads(clean).get("ranked")
            except Exception:
                parsed = None

        if not parsed:
            # LLM unavailable/unparseable -- fall back to the composite ranking.
            # This is a real ranking from named signals, not an arbitrary order,
            # so an air-gapped run still produces a defensible brief.
            to_rank = sorted(to_rank, key=lambda t: -comp.get(t, 0.0))
            for t in to_rank:
                if t not in ranked_order:
                    ranked_order.append(t)
                final_rationale.setdefault(t, None)
            verifier_log.append({"attempt": attempt, "status": "llm_failed_fallback_to_bare", "topics": to_rank})
            break

        still_failing = []
        for item in parsed:
            t = item.get("topic")
            rationale = item.get("rationale", "")
            if t not in to_rank:
                continue
            if _verify_rationale(t, rationale, evidence_bundles, global_rate, momentum, anomaly_scores):
                final_rationale[t] = rationale
                if t not in ranked_order:
                    ranked_order.append(t)
                verifier_log.append({"attempt": attempt, "topic": t, "status": "accepted", "rationale": rationale})
            else:
                still_failing.append(t)
                verifier_log.append({"attempt": attempt, "topic": t, "status": "rejected_ungrounded", "rationale": rationale})

        to_rank = still_failing
        attempt += 1

    # Anything still failing after MAX_RETRIES: bare topic, no fabricated rationale (Section 3.5).
    for t in to_rank:
        if t not in ranked_order:
            ranked_order.append(t)
        final_rationale.setdefault(t, None)
        verifier_log.append({"attempt": attempt, "topic": t, "status": "bare_topic_after_retries"})

    return {
        "ranked_topics": ranked_order,
        "rationale": final_rationale,
        "verifier_log": verifier_log,
        "composite_scores": {t: round(v, 4) for t, v in sorted(comp.items(), key=lambda x: -x[1])},
        "disclosure_conditioned": bool(disclosure),
    }
