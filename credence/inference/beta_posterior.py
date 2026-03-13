"""Beta-Bernoulli posteriors and Bayesian updates for tool reliability tracking.

The core probabilistic building blocks: Beta-distributed reliability estimates
per tool-category pair, Bayesian answer posterior updates on tool responses,
and a two-stage update that first refines category beliefs from response type
(answer vs no-answer) before computing effective reliability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Type aliases ---

AnswerPosterior = NDArray[np.float64]     # shape (n,), sums to 1
CategoryPosterior = NDArray[np.float64]   # shape (num_categories,), sums to 1
ReliabilityTable = NDArray[np.float64]    # shape (num_tools, num_categories, 2); last dim = (alpha, beta)
CoverageTable = NDArray[np.float64]      # shape (num_tools, num_categories, 2); last dim = (alpha, beta)

_EPSILON = 1e-10


def uniform_answer_prior(n: int = 4) -> AnswerPosterior:
    return np.full(n, 1.0 / n)


def uniform_category_prior(n: int = 5) -> CategoryPosterior:
    return np.full(n, 1.0 / n)


def make_reliability_table(num_tools: int, num_categories: int = 5) -> ReliabilityTable:
    """All Beta(1,1) — uniform, genuinely uninformative."""
    return np.ones((num_tools, num_categories, 2), dtype=np.float64)


def expected_reliability(alpha: float, beta: float) -> float:
    """E[Beta(alpha, beta)] = alpha / (alpha + beta)."""
    return alpha / (alpha + beta)


def effective_reliability(
    table: ReliabilityTable,
    tool_idx: int,
    category_posterior: CategoryPosterior,
) -> float:
    """r_effective = sum_c P(c) * E[r[t][c]].

    Marginalises tool reliability over category uncertainty.
    """
    params = table[tool_idx]  # shape (num_categories, 2)
    per_cat_reliability = params[:, 0] / (params[:, 0] + params[:, 1])
    return float(np.dot(category_posterior, per_cat_reliability))


def update_category_posterior_on_response(
    category_posterior: CategoryPosterior,
    coverage: NDArray[np.float64],
    got_answer: bool,
) -> CategoryPosterior:
    """Two-stage update, Stage 1: update category beliefs from response type.

    got_answer=True:  P(c) *= coverage[c],         then normalise.
    got_answer=False: P(c) *= (1 - coverage[c]),    then normalise.

    This handles no_result (Tool B) and not_applicable (Tool C) uniformly —
    both are "did not get an answer" with tool-specific coverage vectors.
    Tools with all-1.0 coverage produce a no-op (no category information).
    """
    weights = coverage if got_answer else (1.0 - coverage)
    updated = category_posterior * weights
    total = updated.sum()
    if total < _EPSILON:
        return category_posterior.copy()
    return updated / total


def update_answer_posterior(
    prior: AnswerPosterior,
    response_idx: int,
    r_effective: float,
) -> AnswerPosterior:
    """Bayes' rule update on answer posterior given a tool response.

    Likelihood: P(tool says x_j | true answer is x_i, reliability r)
        = r           if i == j   (tool correct)
        = (1 - r) / 3 if i != j   (uniform random error)

    Returns normalised posterior.
    """
    n = len(prior)
    wrong_likelihood = (1.0 - r_effective) / (n - 1)
    likelihood = np.full(n, wrong_likelihood)
    likelihood[response_idx] = r_effective
    updated = prior * likelihood
    total = updated.sum()
    if total < _EPSILON:
        return prior.copy()
    return updated / total


def make_coverage_table(
    num_tools: int,
    num_categories: int,
    priors: NDArray[np.float64] | None = None,
    prior_strength: float = 10.0,
) -> CoverageTable:
    """Initialise a Beta-parameterised coverage table.

    Each entry tracks P(tool returns an answer | category) as a Beta distribution.

    Args:
        priors: shape (num_tools, num_categories) with values in (0, 1).
                If None, defaults to 0.5 (uninformative).
        prior_strength: total pseudo-count per cell (alpha + beta = prior_strength).
    """
    if priors is None:
        half = prior_strength * 0.5
        alpha = np.full((num_tools, num_categories), half, dtype=np.float64)
        beta = np.full((num_tools, num_categories), half, dtype=np.float64)
    else:
        priors = np.asarray(priors, dtype=np.float64)
        alpha = priors * prior_strength
        beta = (1.0 - priors) * prior_strength
    return np.stack([alpha, beta], axis=-1)


def expected_coverage(table: CoverageTable, tool_idx: int) -> NDArray[np.float64]:
    """E[Beta(alpha, beta)] per category for one tool, clipped to [0.1, 1.0]."""
    params = table[tool_idx]  # shape (num_categories, 2)
    return np.clip(params[:, 0] / (params[:, 0] + params[:, 1]), 0.1, 1.0)


def update_coverage_table(
    table: CoverageTable,
    tool_idx: int,
    category_posterior: CategoryPosterior,
    was_useful: bool | None,
) -> CoverageTable:
    """Soft Beta update: alpha += P(c) on success, beta += P(c) on failure.

    Returns a NEW table (original is not mutated).
    """
    new_table = table.copy()
    if was_useful is None:
        return new_table
    if was_useful:
        new_table[tool_idx, :, 0] += category_posterior
    else:
        new_table[tool_idx, :, 1] += category_posterior
    return new_table


def update_reliability_table(
    table: ReliabilityTable,
    tool_idx: int,
    category_posterior: CategoryPosterior,
    tool_was_correct: bool | None,
    forgetting: float = 1.0,
) -> ReliabilityTable:
    """Fractional Beta update weighted by category posterior. Returns NEW table.

    tool_was_correct=True:  alpha += w_c per category
    tool_was_correct=False: beta  += w_c per category
    tool_was_correct=None:  no update (return copy)

    forgetting < 1.0: decay existing counts before adding
        alpha_new = max(epsilon, lambda * alpha_old) + update
        beta_new  = max(epsilon, lambda * beta_old)  + update
    """
    new_table = table.copy()
    if tool_was_correct is None:
        return new_table

    params = new_table[tool_idx]  # shape (num_categories, 2)

    if forgetting < 1.0:
        params[:, 0] = np.maximum(_EPSILON, forgetting * params[:, 0])
        params[:, 1] = np.maximum(_EPSILON, forgetting * params[:, 1])

    if tool_was_correct:
        params[:, 0] += category_posterior  # alpha += w_c
    else:
        params[:, 1] += category_posterior  # beta += w_c

    return new_table
