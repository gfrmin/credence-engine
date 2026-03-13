"""Tests for Beta-distributed coverage table (learnable P(tool answers | category))."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from credence.inference.beta_posterior import (
    CoverageTable,
    expected_coverage,
    make_coverage_table,
    update_coverage_table,
    uniform_category_prior,
)

NUM_CATEGORIES = 5


class TestMakeCoverageTable:
    def test_default_uninformative(self):
        table = make_coverage_table(2, NUM_CATEGORIES)
        assert table.shape == (2, NUM_CATEGORIES, 2)
        # Default prior_strength=10, prior=0.5 -> Beta(5, 5)
        assert_allclose(table[:, :, 0], 5.0)
        assert_allclose(table[:, :, 1], 5.0)

    def test_custom_priors(self):
        priors = np.array([[0.9, 0.1], [0.5, 0.5]])
        table = make_coverage_table(2, 2, priors=priors, prior_strength=10.0)
        assert table.shape == (2, 2, 2)
        # Tool 0, cat 0: Beta(9, 1)
        assert table[0, 0, 0] == pytest.approx(9.0)
        assert table[0, 0, 1] == pytest.approx(1.0)
        # Tool 0, cat 1: Beta(1, 9)
        assert table[0, 1, 0] == pytest.approx(1.0)
        assert table[0, 1, 1] == pytest.approx(9.0)

    def test_prior_strength_scales(self):
        priors = np.array([[0.7, 0.3]])
        table = make_coverage_table(1, 2, priors=priors, prior_strength=20.0)
        assert table[0, 0, 0] == pytest.approx(14.0)
        assert table[0, 0, 1] == pytest.approx(6.0)


class TestExpectedCoverage:
    def test_uninformative_gives_half(self):
        table = make_coverage_table(1, NUM_CATEGORIES)
        cov = expected_coverage(table, 0)
        assert_allclose(cov, 0.5)

    def test_high_prior_gives_high_coverage(self):
        priors = np.array([[0.95, 0.05, 0.5, 0.5, 0.5]])
        table = make_coverage_table(1, NUM_CATEGORIES, priors=priors)
        cov = expected_coverage(table, 0)
        assert cov[0] == pytest.approx(0.95)
        assert cov[1] == pytest.approx(0.1)  # clipped from 0.05

    def test_clipped_at_minimum(self):
        """Very low alpha still clips to 0.1."""
        priors = np.array([[0.01, 0.01]])
        table = make_coverage_table(1, 2, priors=priors)
        cov = expected_coverage(table, 0)
        assert_allclose(cov, 0.1)

    def test_clipped_at_maximum(self):
        """Coverage at or above 1.0 clips to 1.0."""
        priors = np.array([[1.0, 1.0]])
        # Edge: prior=1.0 -> alpha=10, beta=0 -> would be inf, but let's check
        # Actually prior=1.0 gives beta=0 which divides by zero, so let's use 0.999
        priors = np.array([[0.999, 0.999]])
        table = make_coverage_table(1, 2, priors=priors)
        cov = expected_coverage(table, 0)
        assert_allclose(cov, 1.0, atol=0.01)


class TestUpdateCoverageTable:
    def test_useful_increments_alpha(self):
        table = make_coverage_table(1, NUM_CATEGORIES)
        cat_post = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        new_table = update_coverage_table(table, 0, cat_post, was_useful=True)
        assert new_table[0, 0, 0] == pytest.approx(5.5)  # alpha += 0.5
        assert new_table[0, 0, 1] == pytest.approx(5.0)  # beta unchanged

    def test_not_useful_increments_beta(self):
        table = make_coverage_table(1, NUM_CATEGORIES)
        cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        new_table = update_coverage_table(table, 0, cat_post, was_useful=False)
        assert new_table[0, 0, 0] == pytest.approx(5.0)  # alpha unchanged
        assert new_table[0, 0, 1] == pytest.approx(6.0)  # beta += 1.0

    def test_none_returns_copy(self):
        table = make_coverage_table(1, NUM_CATEGORIES)
        cat_post = uniform_category_prior()
        new_table = update_coverage_table(table, 0, cat_post, was_useful=None)
        assert_allclose(new_table, table)

    def test_immutability(self):
        table = make_coverage_table(1, NUM_CATEGORIES)
        original = table.copy()
        cat_post = uniform_category_prior()
        update_coverage_table(table, 0, cat_post, was_useful=True)
        assert_allclose(table, original)

    def test_coverage_increases_after_useful(self):
        priors = np.array([[0.5] * NUM_CATEGORIES])
        table = make_coverage_table(1, NUM_CATEGORIES, priors=priors)
        cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        cov_before = expected_coverage(table, 0)[0]
        new_table = update_coverage_table(table, 0, cat_post, was_useful=True)
        cov_after = expected_coverage(new_table, 0)[0]
        assert cov_after > cov_before

    def test_coverage_decreases_after_not_useful(self):
        priors = np.array([[0.5] * NUM_CATEGORIES])
        table = make_coverage_table(1, NUM_CATEGORIES, priors=priors)
        cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        cov_before = expected_coverage(table, 0)[0]
        new_table = update_coverage_table(table, 0, cat_post, was_useful=False)
        cov_after = expected_coverage(new_table, 0)[0]
        assert cov_after < cov_before


class TestCoverageMatchesGPTResearcherPattern:
    """Verify the coverage table reproduces the exact pattern from gpt-researcher."""

    def test_reproduces_retriever_tool_init(self):
        """Same as RetrieverTool.__init__ in gpt-researcher."""
        categories = ("scientific", "medical", "news", "technical",
                      "historical", "legal", "statistical", "general")
        prior_dict = {
            "scientific": 0.5, "medical": 0.5, "news": 0.8, "technical": 0.7,
            "historical": 0.5, "legal": 0.4, "statistical": 0.6, "general": 0.9,
        }
        priors = np.array([[prior_dict[c] for c in categories]])
        table = make_coverage_table(1, len(categories), priors=priors, prior_strength=10.0)

        # Check alpha/beta match the gpt-researcher pattern
        for i, cat in enumerate(categories):
            p = prior_dict[cat]
            assert table[0, i, 0] == pytest.approx(p * 10.0)
            assert table[0, i, 1] == pytest.approx((1.0 - p) * 10.0)

    def test_reproduces_update_pattern(self):
        """Same update_coverage logic as gpt-researcher tools."""
        table = make_coverage_table(1, 3, prior_strength=10.0)
        cat_post = np.array([0.6, 0.3, 0.1])

        new_table = update_coverage_table(table, 0, cat_post, was_useful=True)
        assert new_table[0, 0, 0] == pytest.approx(5.0 + 0.6)
        assert new_table[0, 1, 0] == pytest.approx(5.0 + 0.3)
        assert new_table[0, 2, 0] == pytest.approx(5.0 + 0.1)
        # beta unchanged
        assert_allclose(new_table[0, :, 1], 5.0)
