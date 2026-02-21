"""Tests for Beta-Bernoulli posteriors and Bayesian updates."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.inference.beta_posterior import (
    NUM_CANDIDATES,
    NUM_CATEGORIES,
    effective_reliability,
    expected_reliability,
    make_reliability_table,
    uniform_answer_prior,
    uniform_category_prior,
    update_answer_posterior,
    update_category_posterior_on_response,
    update_reliability_table,
)


# --- Expected reliability ---


def test_expected_reliability_uniform():
    assert expected_reliability(1.0, 1.0) == pytest.approx(0.5)


def test_expected_reliability_high():
    assert expected_reliability(9.0, 1.0) == pytest.approx(0.9)


def test_expected_reliability_low():
    assert expected_reliability(1.0, 9.0) == pytest.approx(0.1)


# --- Effective reliability ---


def test_effective_reliability_point_mass_category():
    """Point-mass category posterior should give single-category reliability."""
    table = make_reliability_table(2)
    # Set tool 0, category 1 to Beta(9, 1) -> E = 0.9
    table[0, 1] = [9.0, 1.0]
    cat_post = np.zeros(NUM_CATEGORIES)
    cat_post[1] = 1.0
    assert effective_reliability(table, 0, cat_post) == pytest.approx(0.9)


def test_effective_reliability_uniform_category():
    """Uniform category posterior averages across all categories."""
    table = make_reliability_table(1)
    # Set varied reliabilities: E = 0.9, 0.5, 0.5, 0.5, 0.5
    table[0, 0] = [9.0, 1.0]
    cat_post = uniform_category_prior()
    # mean = (0.9 + 0.5*4) / 5 = 2.9/5 = 0.58
    assert effective_reliability(table, 0, cat_post) == pytest.approx(0.58)


# --- Category posterior update ---


def test_category_update_all_ones_coverage_is_noop():
    """All-1.0 coverage (Tools A, D) provides no category information."""
    cat_post = np.array([0.3, 0.2, 0.1, 0.1, 0.3])
    coverage = np.ones(NUM_CATEGORIES)
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    assert_allclose(updated, cat_post)


def test_category_update_tool_c_answer_collapses_to_numerical():
    """Tool C returning an answer collapses category to numerical."""
    cat_post = uniform_category_prior()
    # Tool C: only numerical has coverage
    coverage = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    assert_allclose(updated, [0.0, 1.0, 0.0, 0.0, 0.0])


def test_category_update_tool_c_no_answer_eliminates_numerical():
    """Tool C returning not_applicable eliminates numerical category."""
    cat_post = uniform_category_prior()
    coverage = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=False)
    # Numerical gets weight 0, others get 1.0 each -> uniform over non-numerical
    assert updated[1] == pytest.approx(0.0)
    assert_allclose(updated[[0, 2, 3, 4]], 0.25)


def test_category_update_tool_b_no_answer_upweights_low_coverage():
    """Tool B no_result upweights categories with low coverage."""
    cat_post = uniform_category_prior()
    # Tool B coverage from the spec
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=False)
    # Weights: 1-coverage = [0.3, 0.7, 0.6, 0.4, 0.8]
    # Category 4 (reasoning, coverage=0.2) should be highest
    assert updated[4] > updated[0]
    assert updated.sum() == pytest.approx(1.0)


def test_category_update_tool_b_answer_upweights_high_coverage():
    """Tool B returning an answer upweights high-coverage categories."""
    cat_post = uniform_category_prior()
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    # Category 0 (factual, coverage=0.7) should be highest
    assert updated[0] > updated[4]
    assert updated.sum() == pytest.approx(1.0)


def test_category_update_already_zero_stays_zero():
    """Zero-probability category stays zero after update."""
    cat_post = np.array([0.5, 0.0, 0.2, 0.3, 0.0])
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    assert updated[1] == pytest.approx(0.0)
    assert updated[4] == pytest.approx(0.0)
    assert updated.sum() == pytest.approx(1.0)


def test_category_update_degenerate_returns_prior():
    """If update would zero out all categories, return prior unchanged."""
    cat_post = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    coverage = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    # got_answer=True with zero coverage everywhere -> all weights zero
    updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    assert_allclose(updated, cat_post)


# --- Likelihood properties ---


def test_likelihood_correct_response():
    """L(j, j, r) = r for any j, r."""
    prior = uniform_answer_prior()
    r = 0.9
    post = update_answer_posterior(prior, 0, r)
    # Likelihood for candidate 0: r = 0.9
    # Likelihood for others: (1-r)/3 = 0.1/3
    # Prior is uniform, so posterior ∝ likelihood
    expected_0 = r / (r + (1 - r))  # = 0.9
    assert post[0] == pytest.approx(expected_0)


def test_likelihood_wrong_response():
    """L(j, i, r) = (1-r)/3 for j != i."""
    prior = uniform_answer_prior()
    r = 0.9
    post = update_answer_posterior(prior, 0, r)
    # All non-response candidates should have equal posterior
    assert_allclose(post[1:], post[1])


def test_likelihood_sums_to_one():
    """Sum of likelihoods over responses = 1.0 for each hypothesis."""
    r = 0.7
    n = NUM_CANDIDATES
    total = r + (n - 1) * (1 - r) / (n - 1)
    assert total == pytest.approx(1.0)


# --- Answer posterior update ---


def test_answer_update_high_reliability_concentrates():
    """High-reliability tool concentrates posterior on the response."""
    prior = uniform_answer_prior()
    post = update_answer_posterior(prior, 2, 0.9)
    assert post[2] > 0.85
    assert post.sum() == pytest.approx(1.0)


def test_answer_update_uninformative_tool():
    """r = 1/4 means tool is pure noise — posterior should stay uniform."""
    prior = uniform_answer_prior()
    post = update_answer_posterior(prior, 1, 0.25)
    assert_allclose(post, prior, atol=1e-10)


def test_answer_update_sums_to_one():
    """Posterior always sums to 1."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        prior = rng.dirichlet(np.ones(NUM_CANDIDATES))
        r = rng.uniform(0.0, 1.0)
        j = rng.integers(NUM_CANDIDATES)
        post = update_answer_posterior(prior, j, r)
        assert post.sum() == pytest.approx(1.0)


def test_answer_update_hand_computed():
    """Hand-computed example with non-uniform prior."""
    # Prior: [0.4, 0.3, 0.2, 0.1], response=0, r=0.8
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    r = 0.8
    # Likelihoods: L(0|i=0) = 0.8, L(0|i!=0) = 0.2/3
    lik = np.array([0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3])
    unnorm = prior * lik
    expected = unnorm / unnorm.sum()
    post = update_answer_posterior(prior, 0, r)
    assert_allclose(post, expected)


# --- Two-stage update integration ---


def test_two_stage_tool_c_answer():
    """Tool C answer: category collapses to numerical, r_effective = E[r[t][numerical]]."""
    table = make_reliability_table(4)
    # Set tool 2 (calculator), category 1 (numerical) to Beta(10, 0.001) ~ 1.0
    table[2, 1] = [10.0, 0.001]
    cat_post = uniform_category_prior()
    coverage = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    # Stage 1: category collapses to numerical
    cat_updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    assert_allclose(cat_updated, [0.0, 1.0, 0.0, 0.0, 0.0])

    # Stage 2: r_effective with point-mass on numerical
    r_eff = effective_reliability(table, 2, cat_updated)
    assert r_eff == pytest.approx(10.0 / 10.001, abs=1e-4)


def test_two_stage_tool_b_answer_shifts_category():
    """Tool B answer shifts category toward high-coverage categories."""
    table = make_reliability_table(4)
    # Tool 1 (knowledge base): high reliability for factual
    table[1, 0] = [19.0, 1.0]  # E = 0.95 for factual

    cat_post = uniform_category_prior()
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])

    # Stage 1: upweight high-coverage
    cat_updated = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    # Factual has highest coverage -> should be upweighted
    assert cat_updated[0] > cat_post[0]

    # Stage 2: r_effective should be pulled toward factual's high reliability
    r_eff = effective_reliability(table, 1, cat_updated)
    r_eff_before = effective_reliability(table, 1, cat_post)
    assert r_eff > r_eff_before


# --- Reliability table update ---


def test_reliability_update_correct():
    """Correct tool updates alpha."""
    table = make_reliability_table(2)
    cat_post = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    new_table = update_reliability_table(table, 0, cat_post, tool_was_correct=True)
    assert new_table[0, 0, 0] == pytest.approx(1.5)  # alpha += 0.5
    assert new_table[0, 1, 0] == pytest.approx(1.5)
    assert new_table[0, 0, 1] == pytest.approx(1.0)  # beta unchanged


def test_reliability_update_wrong():
    """Wrong tool updates beta."""
    table = make_reliability_table(2)
    cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    new_table = update_reliability_table(table, 0, cat_post, tool_was_correct=False)
    assert new_table[0, 0, 0] == pytest.approx(1.0)  # alpha unchanged
    assert new_table[0, 0, 1] == pytest.approx(2.0)  # beta += 1.0


def test_reliability_update_none():
    """None (unknown) produces no change."""
    table = make_reliability_table(2)
    cat_post = uniform_category_prior()
    new_table = update_reliability_table(table, 0, cat_post, tool_was_correct=None)
    assert_allclose(new_table, table)


def test_reliability_update_immutability():
    """Original table is unchanged after update."""
    table = make_reliability_table(2)
    original = table.copy()
    cat_post = uniform_category_prior()
    update_reliability_table(table, 0, cat_post, tool_was_correct=True)
    assert_allclose(table, original)


def test_reliability_update_forgetting():
    """Forgetting factor decays before adding."""
    table = make_reliability_table(1)
    table[0, 0] = [10.0, 10.0]
    cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    new_table = update_reliability_table(
        table, 0, cat_post, tool_was_correct=True, forgetting=0.9,
    )
    # alpha_new = max(eps, 0.9 * 10.0) + 1.0 = 9.0 + 1.0 = 10.0
    assert new_table[0, 0, 0] == pytest.approx(10.0)
    # beta_new = max(eps, 0.9 * 10.0) + 0.0 = 9.0
    assert new_table[0, 0, 1] == pytest.approx(9.0)
