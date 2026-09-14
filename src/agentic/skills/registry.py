"""Skill registry -- agents/skills/harness separation (2026-09).

An "agent" in this codebase (overall_layer, analyst_layer, question_framer,
verifier, eval_harness) orchestrates a loop or a pipeline stage. A "skill" is
one named, reusable unit of work inside that orchestration -- a function
with a stable name, a one-line description, and metadata about which
pipeline stage it belongs to and whether calling it can make an LLM request.

This registry does NOT change how any of these functions are called today --
every existing call site keeps importing and calling the underlying function
directly, unchanged. It is a discovery/introspection layer, built so that:

  - the Error Researcher (eval_harness.research_errors) can name which skill
    is implicated by a failure category, not just describe the category in
    prose (see SKILL_FOR_ERROR_CATEGORY below and its use there);
  - a future skill-evolution loop has one place to look up "what skills
    exist, what do they do, which one would a v2 replace" instead of
    re-discovering the pipeline by reading five modules;
  - GET /api/skills gives a plain listing instead of grepping the codebase.

Each entry's `fn` is the actual function, not a wrapper -- run_skill() below
calls it with whatever positional/keyword arguments the caller supplies, so
existing call sites keep using their real signature directly rather than
going through a generic reshaping layer.

Deliberately NOT built in this pass:
  - a uniform call interface across skills (their signatures differ too much
    to normalize safely without live-testing every call site against a
    reachable LLM, which this environment can't do);
  - rewiring the existing pipeline (fastapi_app.py's routes, run_agentic.py)
    to call through this registry instead of its current direct imports --
    those call sites are tested and working; adding an indirection layer
    under them is a separate, riskier change than this pass scopes to.
"""

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Skill:
    name: str
    description: str
    module: str
    fn: Callable
    category: str   # pipeline stage this skill belongs to
    llm: bool       # True if calling it can make an LLM request
    version: str = "v1"


_REGISTRY: dict[str, Skill] = {}


def register(name: str, description: str, module: str, fn: Callable,
             category: str, llm: bool, version: str = "v1") -> None:
    if name in _REGISTRY:
        raise ValueError(f"skill '{name}' already registered")
    _REGISTRY[name] = Skill(name=name, description=description, module=module,
                             fn=fn, category=category, llm=llm, version=version)


def get_skill(name: str) -> Skill:
    if name not in _REGISTRY:
        raise KeyError(f"no skill registered as '{name}' -- see list_skills()")
    return _REGISTRY[name]


def run_skill(name: str, *args, **kwargs) -> Any:
    """Calls the named skill's real function with the given args/kwargs.
    No argument reshaping happens here -- callers still need to know that
    skill's actual signature (see get_skill(name).fn for the live function,
    or its module's source)."""
    return get_skill(name).fn(*args, **kwargs)


def list_skills() -> list[dict]:
    return [
        {"name": s.name, "description": s.description, "module": s.module,
         "category": s.category, "llm": s.llm, "version": s.version}
        for s in _REGISTRY.values()
    ]


# Which registered skill (if any) is implicated by each error-taxonomy
# category in eval_harness.ERROR_TAXONOMY -- lets the Error Researcher name
# a concrete lever instead of only describing the failure in prose.
# None means the category points at upstream data/ingestion or is a genuine
# cold case, not at any one skill's logic.
SKILL_FOR_ERROR_CATEGORY = {
    "missing_context": None,
    "unpredictable": None,
    "topic_not_candidate": "composite_scoring",
    "topic_ranked_low": "weight_backtest",
    "lag": "composite_scoring",
    "reasoning_weakness": "analyst_reweight",
}
