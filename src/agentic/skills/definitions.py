"""Registers the existing agentic-pipeline functions as named skills.

Importing this module populates the registry in src.agentic.skills.registry
as a side effect -- it only registers references to functions defined
elsewhere; it does not redefine, wrap, or change their behavior. Imported
once at API startup (see src/api/fastapi_app.py).
"""

from src.agentic.skills.registry import register
from src.agentic.overall_layer import build_overall_topics
from src.agentic.analyst_layer import reweight_for_analyst
from src.agentic.question_framer import frame_questions
from src.agentic.verifier import grounding_gate
from src.data.upcoming import attach_novel_themes
from src.agentic.eval_harness import research_errors, backtest_and_promote_weights
from src.agentic.question_eval import evaluate_analyst_questions

register(
    name="composite_scoring",
    description="Score and rank candidate topics for the whole call (anomaly, disclosure, "
                "momentum, base-rate, drill-flag signals) into the Overall layer's ranked list.",
    module="src.agentic.overall_layer",
    fn=build_overall_topics,
    category="candidate_generation",
    llm=True,
)

register(
    name="analyst_reweight",
    description="Re-rank the Overall layer's topics against one analyst's personal "
                "question-history preferences, and admit genuinely novel (off-taxonomy) "
                "themes on disclosure-signal strength alone.",
    module="src.agentic.analyst_layer",
    fn=reweight_for_analyst,
    category="ranking",
    llm=False,
)

register(
    name="question_framing",
    description="Turn an analyst's ranked topic slots plus retrieved evidence into the "
                "actual predicted question text.",
    module="src.agentic.question_framer",
    fn=frame_questions,
    category="question_generation",
    llm=True,
)

register(
    name="verification",
    description="Grounding gate: checks a framed question's claimed numbers against the "
                "evidence it cites and reframes it once if unsupported.",
    module="src.agentic.verifier",
    fn=grounding_gate,
    category="verification",
    llm=True,
)

register(
    name="novel_theme_detection",
    description="Labels disclosure sentences whose topic doesn't fit the fixed 12-topic "
                "taxonomy, so a genuinely new theme (e.g. FCNR) can still surface as a "
                "candidate instead of being structurally invisible to ranking.",
    module="src.data.upcoming",
    fn=attach_novel_themes,
    category="candidate_generation",
    llm=False,
)

register(
    name="error_research",
    description="Aggregates every held-out miss into the error taxonomy, ranks categories "
                "by share, and names which are actionable vs. not fixable by reweighting "
                "or re-prompting.",
    module="src.agentic.eval_harness",
    fn=research_errors,
    category="diagnostics",
    llm=False,
)

register(
    name="weight_backtest",
    description="Freeze-and-experiment: runs a candidate set of composite-score weights "
                "against the held-out quarters and applies the promotion gate to return "
                "PROMOTE/REJECT.",
    module="src.agentic.eval_harness",
    fn=backtest_and_promote_weights,
    category="backtesting",
    llm=False,
)

register(
    name="question_recall",
    description="Judges the FRAMED QUESTION TEXT against what the analyst actually asked "
                "(0/0.5/1.0 rubric via LLM judge), not just whether the topic bucket "
                "matched -- topic recall can look fine while the actual question misses "
                "the real hook entirely.",
    module="src.agentic.question_eval",
    fn=evaluate_analyst_questions,
    category="diagnostics",
    llm=True,
)
