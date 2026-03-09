"""Bayesian decision-theoretic agent.

Implements the full decision loop from SPEC §3.2, using the inference layer's
Beta posteriors, VOI calculations, and EU-based action selection. Category
inference via keyword heuristics (SPEC §3.6 Option A).
"""

from __future__ import annotations

import re

import numpy as np

from src.agents.common import AgentResult, DecisionStep
from src.environment.categories import CATEGORIES, NUM_CATEGORIES
from src.inference.beta_posterior import (
    CategoryPosterior,
    ReliabilityTable,
    make_reliability_table,
)
from src.inference.decision import (
    Action,
    ActionType,
    QuestionState,
    apply_reliability_updates,
    apply_tool_response,
    compute_reliability_updates,
    initial_question_state,
    select_action,
)
from src.inference.voi import ToolConfig, compute_voi, eu_abstain, eu_submit


# --- Category inference (keyword heuristics, SPEC §3.6 Option A) ---

_NUMERICAL_PATTERN = re.compile(
    r"(?:\bcalcul|\bcomput|\bhow many\b|\bhow much\b|\bwhat is \d|"
    r"\bsquare root\b|\bsum of\b|\barea\b|\bradius\b|"
    r"\binvest|\btip on\b|\d+\s*[\+\-\*\/\%\^]\s*\d+|\d+%)",
    re.IGNORECASE,
)
_RECENT_PATTERN = re.compile(
    r"\b(202[0-9]|recent|latest|current|this year|last year|"
    r"who won the 20|hosted the 20|released .* in 20)\b",
    re.IGNORECASE,
)
_MISCONCEPTION_PATTERN = re.compile(
    r"(?:\btrue or false\b|\bcommon belief\b|\bmyth\b|"
    r"\bdo .* really\b|\bis it true\b|\bdoes .* have a\b|"
    r"\bvisible from space\b|percent of the brain|"
    r"\bmemory span\b|\bmother reject|\bonly use\b)",
    re.IGNORECASE,
)
_REASONING_PATTERN = re.compile(
    r"\b(if .* then|therefore|conclude|logic|implies|"
    r"can we conclude|probability|minimum number|"
    r"how long does it take .* machines|"
    r"all but|overtake|missing dollar|"
    r"counterfeit)\b",
    re.IGNORECASE,
)


def infer_category_prior(question_text: str) -> CategoryPosterior:
    """Keyword-based category prior. Returns a probability distribution, not a point estimate."""
    weights = np.ones(NUM_CATEGORIES, dtype=np.float64)  # base weight 1 each

    if _NUMERICAL_PATTERN.search(question_text):
        weights[CATEGORIES.index("numerical")] += 9.0
    if _RECENT_PATTERN.search(question_text):
        weights[CATEGORIES.index("recent_events")] += 9.0
    if _MISCONCEPTION_PATTERN.search(question_text):
        weights[CATEGORIES.index("misconceptions")] += 9.0
    if _REASONING_PATTERN.search(question_text):
        weights[CATEGORIES.index("reasoning")] += 9.0

    # If nothing matched strongly, factual gets a mild boost (most common category)
    if weights.max() == 1.0:
        weights[CATEGORIES.index("factual")] += 1.0

    return weights / weights.sum()


# --- Bayesian Agent ---

class BayesianAgent:
    """EU-maximising agent with Beta-Bernoulli reliability tracking.

    Conforms to the benchmark's Agent protocol (on_question_start / choose_action /
    on_tool_response / on_question_end) AND provides solve_question for direct use.
    """

    def __init__(
        self,
        tool_configs: list[ToolConfig],
        forgetting: float = 1.0,
        name: str = "bayesian",
    ):
        self.name = name
        self.tool_configs = tool_configs
        self.forgetting = forgetting
        self.num_tools = len(tool_configs)
        self.reliability_table: ReliabilityTable = make_reliability_table(self.num_tools)

        # Per-question state (set in on_question_start)
        self._state: QuestionState | None = None
        self._trace: list[DecisionStep] = []
        self._step: int = 0
        self._question_text: str = ""

    # --- Benchmark Agent protocol ---

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        cat_prior = infer_category_prior(question_text) if question_text else _uniform_category()
        self._state = initial_question_state(cat_prior)
        self._trace = []
        self._step = 0
        self._question_text = question_text

    def choose_action(self) -> Action:
        assert self._state is not None
        self._step += 1

        # Log decision landscape before choosing
        eu_sub = eu_submit(self._state.answer_posterior)
        eu_abs = eu_abstain()
        eu_queries: dict[int, float] = {}
        for t_idx in range(self.num_tools):
            if t_idx in self._state.used_tools:
                continue
            voi = compute_voi(
                self._state.answer_posterior,
                self.reliability_table,
                t_idx,
                self._state.category_posterior,
                self.tool_configs[t_idx],
            )
            eu_queries[t_idx] = voi - self.tool_configs[t_idx].cost

        action = select_action(self._state, self.reliability_table, self.tool_configs)

        if action.action_type == ActionType.SUBMIT:
            chosen = f"submit({action.answer_idx})"
        elif action.action_type == ActionType.ABSTAIN:
            chosen = "abstain"
        else:
            chosen = f"query({action.tool_idx})"

        self._trace.append(DecisionStep(
            step=self._step,
            eu_submit=eu_sub,
            eu_abstain=eu_abs,
            eu_query=eu_queries,
            chosen_action=chosen,
        ))

        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        assert self._state is not None
        self._state = apply_tool_response(
            self._state, tool_idx, response,
            self.reliability_table, self.tool_configs[tool_idx],
        )

    def on_question_end(self, was_correct: bool | None) -> None:
        assert self._state is not None
        # Determine what we submitted
        submitted = None
        for step in reversed(self._trace):
            if step.chosen_action.startswith("submit("):
                submitted = int(step.chosen_action[7:-1])
                break

        updates = compute_reliability_updates(
            submitted, was_correct, self._state.tool_responses,
        )
        self.reliability_table = apply_reliability_updates(
            self.reliability_table, updates,
            self._state.category_posterior, self.forgetting,
        )

    def get_belief_snapshot(self) -> dict | None:
        if self._state is None:
            return None
        return {
            "answer_posterior": self._state.answer_posterior.tolist(),
            "category_posterior": self._state.category_posterior.tolist(),
            "used_tools": list(self._state.used_tools),
        }

    # --- High-level solve_question interface ---

    def solve_question(
        self,
        question_text: str,
        candidates: tuple[str, ...],
        category_hint: str | None,
        tool_query_fn,
    ) -> AgentResult:
        """Solve a single question using EU maximisation."""
        if category_hint:
            cat_prior = _category_prior_from_hint(category_hint)
        else:
            cat_prior = infer_category_prior(question_text)

        self._state = initial_question_state(cat_prior)
        self._trace = []
        self._step = 0
        tools_used: list[int] = []

        while True:
            action = self.choose_action()

            if action.action_type == ActionType.QUERY:
                assert action.tool_idx is not None
                response = tool_query_fn(action.tool_idx)
                self.on_tool_response(action.tool_idx, response)
                tools_used.append(action.tool_idx)

            elif action.action_type == ActionType.SUBMIT:
                confidence = float(np.max(self._state.answer_posterior))
                return AgentResult(
                    answer=action.answer_idx,
                    tools_queried=tuple(tools_used),
                    confidence=confidence,
                    decision_trace=tuple(self._trace),
                )

            elif action.action_type == ActionType.ABSTAIN:
                confidence = float(np.max(self._state.answer_posterior))
                return AgentResult(
                    answer=None,
                    tools_queried=tuple(tools_used),
                    confidence=confidence,
                    decision_trace=tuple(self._trace),
                )


def _uniform_category() -> CategoryPosterior:
    return np.full(NUM_CATEGORIES, 1.0 / NUM_CATEGORIES)


def _category_prior_from_hint(hint: str) -> CategoryPosterior:
    """Strong prior on the hinted category."""
    prior = np.ones(NUM_CATEGORIES, dtype=np.float64)
    if hint in CATEGORIES:
        prior[CATEGORIES.index(hint)] += 9.0
    return prior / prior.sum()
