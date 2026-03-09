"""Baseline agents: Random, AllTools, Oracle, SingleBestTool.

These provide reference points for the benchmark. The OracleAgent uses the
same inference layer as BayesianAgent but with true reliabilities pre-loaded,
giving an upper bound on what EU maximisation can achieve.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.agents.common import DecisionStep
from src.environment.categories import CATEGORIES
from src.inference.beta_posterior import make_reliability_table
from src.inference.decision import Action, ActionType, select_action
from src.inference.voi import ToolConfig, eu_abstain, eu_submit, compute_voi
from src.environment.tools import SimulatedTool, tool_config_for

# Reuse BayesianAgent's full decision machinery for the oracle
from src.agents.bayesian_agent import BayesianAgent, infer_category_prior


class RandomAgent:
    """Picks a random available tool, submits its answer. Lower bound."""

    def __init__(self, num_tools: int, seed: int = 0, name: str = "random"):
        self.name = name
        self._num_tools = num_tools
        self._rng = np.random.default_rng(seed)
        self._candidates: tuple[str, ...] = ()
        self._tool_responses: dict[int, int | None] = {}
        self._queried: bool = False
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._tool_responses = {}
        self._queried = False
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if not self._queried:
            self._queried = True
            tool_idx = int(self._rng.integers(self._num_tools))
            action = Action(ActionType.QUERY, tool_idx=tool_idx)
        else:
            # Submit: use tool response if available, else random candidate
            answer = None
            for resp in self._tool_responses.values():
                if resp is not None:
                    answer = resp
                    break
            if answer is None:
                answer = int(self._rng.integers(len(self._candidates)))
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._tool_responses[tool_idx] = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"tool_responses": dict(self._tool_responses)}


class AllToolsAgent:
    """Queries all tools, majority-votes on the answers, always submits."""

    def __init__(self, num_tools: int, name: str = "all_tools"):
        self.name = name
        self._num_tools = num_tools
        self._candidates: tuple[str, ...] = ()
        self._tool_responses: dict[int, int | None] = {}
        self._next_tool: int = 0
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._tool_responses = {}
        self._next_tool = 0
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if self._next_tool < self._num_tools:
            action = Action(ActionType.QUERY, tool_idx=self._next_tool)
            self._next_tool += 1
        else:
            answer = self._majority_vote()
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def _majority_vote(self) -> int:
        """Return the most common non-None response, or 0 as fallback."""
        votes = [r for r in self._tool_responses.values() if r is not None]
        if not votes:
            return 0
        counter = Counter(votes)
        return counter.most_common(1)[0][0]

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._tool_responses[tool_idx] = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"tool_responses": dict(self._tool_responses)}


class OracleAgent(BayesianAgent):
    """Knows true tool reliabilities. Upper bound on EU maximisation.

    Uses exactly the same inference layer as BayesianAgent, but pre-loads
    the reliability table with the true values (as if it had observed
    infinitely many questions). No learning occurs.
    """

    def __init__(
        self,
        tools: list[SimulatedTool],
        tool_configs: list[ToolConfig],
        name: str = "oracle",
    ):
        super().__init__(tool_configs=tool_configs, name=name)
        # Set reliability table to true values with strong pseudo-counts
        # Beta(100*r, 100*(1-r)) has mean r and tight concentration
        n_pseudo = 100.0
        for t_idx, tool in enumerate(tools):
            for c_idx, cat in enumerate(CATEGORIES):
                r = tool.reliability_by_category.get(cat, 0.0)
                self.reliability_table[t_idx, c_idx, 0] = max(0.01, n_pseudo * r)
                self.reliability_table[t_idx, c_idx, 1] = max(0.01, n_pseudo * (1.0 - r))

    def on_question_end(self, was_correct: bool | None) -> None:
        # Oracle does NOT update — it already knows the true reliabilities
        pass


class SingleBestToolAgent:
    """Always queries Tool A (cheapest reliable tool), always submits its answer."""

    def __init__(self, tool_idx: int = 0, name: str = "single_best_tool"):
        self.name = name
        self._tool_idx = tool_idx
        self._candidates: tuple[str, ...] = ()
        self._response: int | None = None
        self._queried: bool = False
        self._trace: list[DecisionStep] = []
        self._step: int = 0

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
        question_text: str = "",
    ) -> None:
        self._candidates = candidates
        self._response = None
        self._queried = False
        self._trace = []
        self._step = 0

    def choose_action(self) -> Action:
        self._step += 1
        if not self._queried:
            self._queried = True
            action = Action(ActionType.QUERY, tool_idx=self._tool_idx)
        else:
            answer = self._response if self._response is not None else 0
            action = Action(ActionType.SUBMIT, answer_idx=answer)

        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._response = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {"response": self._response}


def _action_str(action: Action) -> str:
    if action.action_type == ActionType.SUBMIT:
        return f"submit({action.answer_idx})"
    elif action.action_type == ActionType.ABSTAIN:
        return "abstain"
    return f"query({action.tool_idx})"
