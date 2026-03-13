"""EU-based action selection and question state management.

The decision loop: given current beliefs, select the action (submit, abstain,
or query a tool) that maximises expected utility. Manages immutable question
state through the two-stage Bayesian update, and computes reliability updates
from post-question feedback.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from credence.inference.beta_posterior import (
    CategoryPosterior,
    ReliabilityTable,
    effective_reliability,
    uniform_answer_prior,
    update_answer_posterior,
    update_category_posterior_on_response,
    update_reliability_table,
)
from credence.inference.voi import (
    ScoringRule,
    ToolConfig,
    compute_voi,
    eu_abstain,
    eu_submit,
)

_DEFAULT_SCORING = ScoringRule()


class ActionType(Enum):
    SUBMIT = auto()
    ABSTAIN = auto()
    QUERY = auto()


class Action(NamedTuple):
    action_type: ActionType
    tool_idx: int | None = None      # for QUERY
    answer_idx: int | None = None    # for SUBMIT
    eu: float = 0.0


class QuestionState(NamedTuple):
    answer_posterior: np.ndarray           # shape (4,)
    category_posterior: CategoryPosterior  # shape (5,)
    used_tools: frozenset[int]
    tool_responses: dict[int, int | None]  # tool_idx -> response (int or None)


def initial_question_state(
    category_posterior: CategoryPosterior,
    n_candidates: int = 4,
) -> QuestionState:
    """Fresh state for a new question: uniform answer prior, no tools used."""
    return QuestionState(
        answer_posterior=uniform_answer_prior(n_candidates),
        category_posterior=category_posterior.copy(),
        used_tools=frozenset(),
        tool_responses={},
    )


def select_action(
    state: QuestionState,
    reliability_table: ReliabilityTable,
    tool_configs: list[ToolConfig],
    scoring: ScoringRule = _DEFAULT_SCORING,
) -> Action:
    """Choose the action that maximises expected utility.

    Computes EU for submit, abstain, and each unused tool's (VOI - cost).
    Tie-breaking: submit > abstain > query (no exploration bonus).
    """
    best_answer_idx = int(np.argmax(state.answer_posterior))
    eu_sub = eu_submit(state.answer_posterior, scoring)
    eu_abs = eu_abstain(scoring)

    best_action = Action(ActionType.SUBMIT, answer_idx=best_answer_idx, eu=eu_sub)
    # Tie-break: submit > abstain (use tolerance for floating-point ties)
    if eu_abs > eu_sub + 1e-12:
        best_action = Action(ActionType.ABSTAIN, eu=eu_abs)

    # EU of the best non-query action — used to convert VOI (a delta) to absolute EU
    eu_current = best_action.eu

    for tool_idx, config in enumerate(tool_configs):
        if tool_idx in state.used_tools:
            continue
        voi = compute_voi(
            state.answer_posterior,
            reliability_table,
            tool_idx,
            state.category_posterior,
            config,
            scoring,
        )
        net_eu = eu_current + voi - config.cost
        if net_eu > best_action.eu:
            best_action = Action(ActionType.QUERY, tool_idx=tool_idx, eu=net_eu)

    return best_action


def apply_tool_response(
    state: QuestionState,
    tool_idx: int,
    response: int | None,
    reliability_table: ReliabilityTable,
    tool_config: ToolConfig,
) -> QuestionState:
    """Apply a tool response via the two-stage Bayesian update. Returns new state.

    Stage 1: Update category posterior on response type (answer vs no-answer).
    Stage 2 (only if response is an int): Compute r_effective from the UPDATED
             category posterior, then update the answer posterior.

    The original state is never mutated.
    """
    got_answer = response is not None
    coverage = tool_config.coverage_by_category

    # Stage 1: category update
    new_cat_post = update_category_posterior_on_response(
        state.category_posterior, coverage, got_answer,
    )

    # Stage 2: answer update (only if we got an answer)
    if got_answer:
        r_eff = effective_reliability(reliability_table, tool_idx, new_cat_post)
        new_ans_post = update_answer_posterior(state.answer_posterior, response, r_eff)
    else:
        new_ans_post = state.answer_posterior.copy()

    return QuestionState(
        answer_posterior=new_ans_post,
        category_posterior=new_cat_post,
        used_tools=state.used_tools | {tool_idx},
        tool_responses={**state.tool_responses, tool_idx: response},
    )


def compute_reliability_updates(
    submitted_answer_idx: int | None,
    was_correct: bool | None,
    tool_responses: dict[int, int | None],
) -> dict[int, bool | None]:
    """Determine per-tool correctness from post-question feedback.

    Returns: {tool_idx: True (correct), False (wrong), or None (unknown)}.

    Logic:
    - Abstained (submitted_answer_idx is None): no ground truth -> all None
    - Correct submission: tool agreed -> True, disagreed -> False
    - Wrong submission: tool agreed with our wrong answer -> False,
                        tool disagreed -> None (might have given different wrong answer)
    - Tools that returned None: always None (no reliability signal)
    """
    if submitted_answer_idx is None or was_correct is None:
        return {t: None for t in tool_responses}

    updates: dict[int, bool | None] = {}
    for tool_idx, resp in tool_responses.items():
        if resp is None:
            updates[tool_idx] = None
        elif was_correct:
            updates[tool_idx] = (resp == submitted_answer_idx)
        else:
            # Wrong submission: tools that agreed with us were wrong.
            # Tools that disagreed: we don't know if they were right.
            updates[tool_idx] = False if resp == submitted_answer_idx else None

    return updates


def compute_binary_reliability_updates(
    tool_responses: dict[int, int | None],
    was_correct: bool | None,
) -> dict[int, bool | None]:
    """Determine per-tool correctness from binary (correct/wrong) feedback.

    Simpler than compute_reliability_updates: no candidate-agreement logic.
    Each tool that returned a response inherits the overall outcome; tools
    that returned None get None (no reliability signal).
    """
    if was_correct is None:
        return {t: None for t in tool_responses}
    return {
        t: (was_correct if resp is not None else None)
        for t, resp in tool_responses.items()
    }


def apply_reliability_updates(
    table: ReliabilityTable,
    updates: dict[int, bool | None],
    category_posterior: CategoryPosterior,
    forgetting: float = 1.0,
) -> ReliabilityTable:
    """Apply all tool correctness updates to the reliability table. Returns new table."""
    result = table
    for tool_idx, was_correct in updates.items():
        result = update_reliability_table(
            result, tool_idx, category_posterior, was_correct, forgetting,
        )
    return result
