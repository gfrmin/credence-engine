"""Tests for Value of Information calculations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.inference.beta_posterior import (
    NUM_CANDIDATES,
    NUM_CATEGORIES,
    make_reliability_table,
    uniform_answer_prior,
    uniform_category_prior,
)
from src.inference.voi import (
    PENALTY_WRONG,
    REWARD_ABSTAIN,
    REWARD_CORRECT,
    ToolConfig,
    compute_voi,
    eu_abstain,
    eu_star,
    eu_submit,
)


# --- EU functions ---


def test_eu_submit_certain():
    """Certain posterior gives full reward."""
    post = np.array([1.0, 0.0, 0.0, 0.0])
    assert eu_submit(post) == pytest.approx(REWARD_CORRECT)


def test_eu_submit_uniform():
    """Uniform posterior: EU = 0.25 * 10 + 0.75 * (-5) = -1.25."""
    post = uniform_answer_prior()
    assert eu_submit(post) == pytest.approx(0.25 * 10.0 + 0.75 * (-5.0))


def test_eu_submit_breakeven():
    """At p = 1/3: EU = 1/3 * 10 + 2/3 * (-5) = 0."""
    post = np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6])
    assert eu_submit(post) == pytest.approx(0.0, abs=1e-10)


def test_eu_abstain_is_zero():
    assert eu_abstain() == REWARD_ABSTAIN == 0.0


def test_eu_star_picks_max():
    """eu_star = max(eu_submit, eu_abstain)."""
    # Uniform: eu_submit = -1.25 < 0 = eu_abstain
    assert eu_star(uniform_answer_prior()) == pytest.approx(0.0)
    # Certain: eu_submit = 10 > 0
    assert eu_star(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(10.0)


def test_eu_star_at_threshold():
    """At p = 1/3, eu_submit = 0 = eu_abstain, eu_star = 0."""
    post = np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6])
    assert eu_star(post) == pytest.approx(0.0, abs=1e-10)


# --- VOI properties ---


def test_voi_nonnegative_random():
    """VOI >= 0 for 100 random posteriors and tool configs."""
    rng = np.random.default_rng(42)
    table = make_reliability_table(1)
    for _ in range(100):
        ans_post = rng.dirichlet(np.ones(NUM_CANDIDATES))
        cat_post = rng.dirichlet(np.ones(NUM_CATEGORIES))
        coverage = rng.uniform(0.0, 1.0, NUM_CATEGORIES)
        # Set some random reliability
        table[0, :, 0] = rng.uniform(1.0, 10.0, NUM_CATEGORIES)
        table[0, :, 1] = rng.uniform(1.0, 10.0, NUM_CATEGORIES)
        config = ToolConfig(cost=1.0, coverage_by_category=coverage)
        voi = compute_voi(ans_post, table, 0, cat_post, config)
        assert voi >= -1e-10, f"VOI was negative: {voi}"


def test_voi_zero_when_point_mass():
    """VOI = 0 when posterior is already a point mass (nothing to learn)."""
    post = np.array([1.0, 0.0, 0.0, 0.0])
    table = make_reliability_table(1)
    table[0, :, 0] = 9.0  # reliable tool
    cat_post = uniform_category_prior()
    config = ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES))
    voi = compute_voi(post, table, 0, cat_post, config)
    assert voi == pytest.approx(0.0, abs=1e-10)


def test_voi_zero_when_uninformative():
    """VOI = 0 when tool is pure noise (r = 0.25 for 4 candidates)."""
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    # Beta(1, 3) -> E = 0.25 = 1/NUM_CANDIDATES -> uninformative
    table[0, :, 0] = 1.0
    table[0, :, 1] = 3.0
    cat_post = uniform_category_prior()
    config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
    voi = compute_voi(post, table, 0, cat_post, config)
    assert voi == pytest.approx(0.0, abs=1e-6)


def test_voi_positive_for_informative_tool():
    """VOI > 0 for an informative tool with uncertain posterior."""
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    table[0, :, 0] = 9.0  # E = 0.9
    table[0, :, 1] = 1.0
    cat_post = uniform_category_prior()
    config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
    voi = compute_voi(post, table, 0, cat_post, config)
    assert voi > 1.0


# --- Specific VOI calculations ---


def test_voi_perfect_tool_uniform_prior():
    """Perfect tool (r=1.0), uniform prior: VOI = 10.0.

    With r=1.0, after any response j, posterior collapses to point mass on j.
    eu_star goes from 0 (uniform -> abstain) to 10 (certain -> submit).
    VOI = 10 - 0 = 10.
    """
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    table[0, :, 0] = 1000.0  # E ≈ 1.0
    table[0, :, 1] = 1.0
    cat_post = uniform_category_prior()
    config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
    voi = compute_voi(post, table, 0, cat_post, config)
    assert voi == pytest.approx(10.0, abs=0.05)


def test_voi_tool_c_with_half_numerical():
    """Tool C with P(numerical)=0.5: VOI ≈ 0.5 * (10 - 0) = 5.0.

    Tool C: coverage = [0,1,0,0,0], perfect reliability for numerical.
    P(covered) = P(numerical) = 0.5.
    If answer: category collapses to numerical, r_eff ≈ 1.0, posterior → point mass, EU* = 10.
    If no answer: posterior unchanged, EU* = 0 (uniform → abstain).
    VOI = 0.5 * 10 + 0.5 * 0 - 0 = 5.0.
    """
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    table[0, 1, :] = [1000.0, 1.0]  # perfect for numerical
    cat_post = np.array([0.1, 0.5, 0.1, 0.1, 0.2])
    coverage = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    config = ToolConfig(cost=0.0, coverage_by_category=coverage)
    voi = compute_voi(post, table, 0, cat_post, config)
    assert voi == pytest.approx(5.0, abs=0.1)


def test_voi_tool_b_partial_coverage():
    """Tool B with partial coverage: VOI comes only from answer responses."""
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    table[0, :, 0] = 9.0  # E = 0.9
    table[0, :, 1] = 1.0
    cat_post = uniform_category_prior()
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])
    config_partial = ToolConfig(cost=0.0, coverage_by_category=coverage)
    config_full = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))

    voi_partial = compute_voi(post, table, 0, cat_post, config_partial)
    voi_full = compute_voi(post, table, 0, cat_post, config_full)
    # Partial coverage should have lower VOI than full coverage
    assert voi_partial < voi_full
    assert voi_partial > 0


# --- Response probability consistency ---


def test_response_probabilities_sum_to_one_standard():
    """For standard tools (coverage=1), response probabilities sum to 1."""
    post = np.array([0.4, 0.3, 0.2, 0.1])
    r_eff = 0.8
    wrong_lik = (1.0 - r_eff) / (NUM_CANDIDATES - 1)
    total = 0.0
    for j in range(NUM_CANDIDATES):
        lik_j = np.where(np.arange(NUM_CANDIDATES) == j, r_eff, wrong_lik)
        total += float(np.dot(post, lik_j))
    assert total == pytest.approx(1.0)


def test_response_probabilities_sum_with_coverage():
    """P(answer responses) + P(no_answer) = 1."""
    post = np.array([0.4, 0.3, 0.2, 0.1])
    cat_post = uniform_category_prior()
    coverage = np.array([0.7, 0.3, 0.4, 0.6, 0.2])
    p_covered = float(np.dot(cat_post, coverage))

    table = make_reliability_table(1)
    table[0, :, 0] = 4.0
    table[0, :, 1] = 1.0
    from src.inference.beta_posterior import (
        effective_reliability as eff_r,
        update_category_posterior_on_response,
    )
    cat_given_answer = update_category_posterior_on_response(cat_post, coverage, got_answer=True)
    r_eff = eff_r(table, 0, cat_given_answer)

    wrong_lik = (1.0 - r_eff) / (NUM_CANDIDATES - 1)
    p_answers = 0.0
    for j in range(NUM_CANDIDATES):
        lik_j = np.where(np.arange(NUM_CANDIDATES) == j, r_eff, wrong_lik)
        p_answers += p_covered * float(np.dot(post, lik_j))

    # p_answers + p_no_answer = p_covered * 1.0 + (1 - p_covered) = 1.0
    assert (p_answers + (1.0 - p_covered)) == pytest.approx(1.0)


# --- Two-candidate simplified case ---


def test_voi_two_candidate_hand_computed():
    """Two-candidate simplified case for hand verification.

    With 2 candidates, prior = [0.5, 0.5], r = 0.8:
    - P(resp=0) = 0.5*0.8 + 0.5*0.2 = 0.5
    - P(resp=1) = 0.5*0.2 + 0.5*0.8 = 0.5
    - post_0 = [0.8, 0.2], eu_star = max(0.8*10 + 0.2*(-5), 0) = max(7, 0) = 7
    - post_1 = [0.2, 0.8], eu_star = max(0.8*10 + 0.2*(-5), 0) = 7
    - VOI = 0.5*7 + 0.5*7 - 0 = 7

    We simulate this with 4 candidates but uniform prior and specific structure.
    """
    # Use 4-candidate with r = 0.8
    post = uniform_answer_prior()
    table = make_reliability_table(1)
    table[0, :, 0] = 4.0  # E = 0.8
    table[0, :, 1] = 1.0
    cat_post = uniform_category_prior()
    config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
    voi = compute_voi(post, table, 0, cat_post, config)

    # With uniform prior over 4, eu_current = max(-1.25, 0) = 0
    # After seeing response j with r=0.8:
    #   post_j[j] = 0.25*0.8 / (0.25*0.8 + 3*0.25*(0.2/3)) = 0.8
    #   eu_submit = 0.8*10 + 0.2*(-5) = 7.0
    #   eu_star = 7.0
    # All 4 responses equally likely (symmetric), so:
    # VOI = 4 * 0.25 * 7.0 - 0 = 7.0
    assert voi == pytest.approx(7.0, abs=0.01)
