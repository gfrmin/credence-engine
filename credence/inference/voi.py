"""Value of Information calculations and expected utility functions.

EU-based scoring for submit/abstain actions and exact VOI computation for
querying a tool, using the two-stage Bayesian update (category posterior
refined on response type, then answer posterior updated with effective reliability).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from credence.inference.beta_posterior import (
    AnswerPosterior,
    CategoryPosterior,
    ReliabilityTable,
    effective_reliability,
    update_answer_posterior,
    update_category_posterior_on_response,
)

# --- Scoring ---


class ScoringRule(NamedTuple):
    reward_correct: float = 10.0
    penalty_wrong: float = -5.0
    reward_abstain: float = 0.0


_DEFAULT_SCORING = ScoringRule()

# Backward-compatible module-level constants
REWARD_CORRECT = _DEFAULT_SCORING.reward_correct
PENALTY_WRONG = _DEFAULT_SCORING.penalty_wrong
REWARD_ABSTAIN = _DEFAULT_SCORING.reward_abstain


class ToolConfig(NamedTuple):
    cost: float
    coverage_by_category: NDArray[np.float64]  # shape (num_categories,)


def eu_submit(
    answer_posterior: AnswerPosterior,
    scoring: ScoringRule = _DEFAULT_SCORING,
) -> float:
    """EU of submitting the best answer.

    EU = p_best * reward_correct + (1 - p_best) * penalty_wrong
    """
    p_best = float(np.max(answer_posterior))
    return p_best * scoring.reward_correct + (1.0 - p_best) * scoring.penalty_wrong


def eu_abstain(scoring: ScoringRule = _DEFAULT_SCORING) -> float:
    """EU of abstaining."""
    return scoring.reward_abstain


def eu_star(
    answer_posterior: AnswerPosterior,
    scoring: ScoringRule = _DEFAULT_SCORING,
) -> float:
    """Best achievable EU given current beliefs: max(eu_submit, eu_abstain)."""
    return max(eu_submit(answer_posterior, scoring), eu_abstain(scoring))


def compute_voi(
    answer_posterior: AnswerPosterior,
    reliability_table: ReliabilityTable,
    tool_idx: int,
    category_posterior: CategoryPosterior,
    tool_config: ToolConfig,
    scoring: ScoringRule = _DEFAULT_SCORING,
) -> float:
    """Exact Value of Information for querying a tool.

    Uses the two-stage Bayesian update internally:
      Stage 1: Update category posterior on response type (answer vs no-answer).
      Stage 2: Compute r_effective from updated category posterior, then
               update answer posterior via Bayes' rule.

    Returns raw VOI (non-negative). Caller computes net = voi - cost.

    The VOI handles all tool types uniformly:
      - Tools A/D (coverage=1.0): no-answer branch contributes 0.
      - Tool B (partial coverage): no-answer branch preserves current EU.
      - Tool C (binary coverage): answer branch only fires for P(numerical).
    """
    eu_current = eu_star(answer_posterior, scoring)
    coverage = tool_config.coverage_by_category

    # P(tool returns an answer) = sum_c P(c) * coverage[c]
    p_covered = float(np.dot(category_posterior, coverage))

    n = len(answer_posterior)
    expected_eu = 0.0

    if p_covered > 0:
        # --- Answer branch ---
        # Stage 1: category posterior conditioned on getting an answer
        cat_given_answer = update_category_posterior_on_response(
            category_posterior, coverage, got_answer=True,
        )
        r_eff = effective_reliability(reliability_table, tool_idx, cat_given_answer)

        # Enumerate possible responses
        wrong_lik = (1.0 - r_eff) / (n - 1)
        for j in range(n):
            # P(response=j | answer) = sum_i P(x_i) * L(j | x_i, r_eff)
            lik_j = np.where(
                np.arange(n) == j, r_eff, wrong_lik,
            )
            p_resp_j = float(np.dot(answer_posterior, lik_j))
            if p_resp_j > 1e-15:
                post_j = update_answer_posterior(answer_posterior, j, r_eff)
                expected_eu += p_covered * p_resp_j * eu_star(post_j, scoring)

    # --- No-answer branch: answer posterior unchanged ---
    expected_eu += (1.0 - p_covered) * eu_current

    # Clamp for numerical safety (VOI is theoretically non-negative)
    return max(expected_eu - eu_current, 0.0)
