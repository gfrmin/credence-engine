"""Bayesian decision-theoretic agent.

Implements the full decision loop from SPEC §3.2, using the inference layer's
Beta posteriors, VOI calculations, and EU-based action selection. Fully
domain-agnostic: categories and category inference are injected at construction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from credence.agents.common import AgentResult, DecisionStep
from credence.inference.beta_posterior import (
    CategoryPosterior,
    ReliabilityTable,
    make_reliability_table,
)
from credence.inference.decision import (
    Action,
    ActionType,
    QuestionState,
    apply_reliability_updates,
    apply_tool_response,
    compute_reliability_updates,
    initial_question_state,
    select_action,
)
from credence.inference.voi import ScoringRule, ToolConfig, compute_voi, eu_abstain, eu_submit

_DEFAULT_SCORING = ScoringRule()


# --- Bayesian Agent ---

class BayesianAgent:
    """EU-maximising agent with Beta-Bernoulli reliability tracking.

    Conforms to the benchmark's Agent protocol (on_question_start / choose_action /
    on_tool_response / on_question_end) AND provides solve_question for direct use.

    Categories and category inference are injected at construction, making this
    agent fully domain-agnostic. When ``categories`` is None, the agent uses
    ``num_categories`` anonymous categories with a uniform prior.
    """

    def __init__(
        self,
        tool_configs: list[ToolConfig],
        categories: tuple[str, ...] | None = None,
        num_categories: int = 5,
        category_infer_fn: Callable[[str], NDArray] | None = None,
        forgetting: float = 1.0,
        name: str = "bayesian",
        scoring: ScoringRule = _DEFAULT_SCORING,
    ):
        self.name = name
        self.tool_configs = tool_configs
        self.forgetting = forgetting
        self.scoring = scoring
        self.num_tools = len(tool_configs)
        self._num_categories = len(categories) if categories else num_categories
        self._categories = categories
        self._category_infer_fn = category_infer_fn
        self.reliability_table: ReliabilityTable = make_reliability_table(
            self.num_tools, self._num_categories,
        )

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
        if question_text and self._category_infer_fn is not None:
            cat_prior = self._category_infer_fn(question_text)
        else:
            cat_prior = self._uniform_category()
        self._state = initial_question_state(cat_prior, n_candidates=len(candidates))
        self._trace = []
        self._step = 0
        self._question_text = question_text

    def choose_action(self) -> Action:
        assert self._state is not None
        self._step += 1

        # Log decision landscape before choosing
        eu_sub = eu_submit(self._state.answer_posterior, self.scoring)
        eu_abs = eu_abstain(self.scoring)
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
                self.scoring,
            )
            eu_queries[t_idx] = voi - self.tool_configs[t_idx].cost

        action = select_action(self._state, self.reliability_table, self.tool_configs, self.scoring)

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
            cat_prior = self._category_prior_from_hint(category_hint)
        elif self._category_infer_fn is not None:
            cat_prior = self._category_infer_fn(question_text)
        else:
            cat_prior = self._uniform_category()

        self._state = initial_question_state(cat_prior, n_candidates=len(candidates))
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

    # --- Private helpers ---

    def _uniform_category(self) -> CategoryPosterior:
        return np.full(self._num_categories, 1.0 / self._num_categories)

    def _category_prior_from_hint(self, hint: str) -> CategoryPosterior:
        """Strong prior on the hinted category."""
        prior = np.ones(self._num_categories, dtype=np.float64)
        if self._categories and hint in self._categories:
            prior[self._categories.index(hint)] += 9.0
        return prior / prior.sum()
