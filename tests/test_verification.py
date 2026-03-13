"""Hand-worked numerical verification of the inference layer.

Each test prints expected vs actual values for pen-and-paper cross-checking.
Run with: uv run pytest tests/test_verification.py -v -s
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from credence.inference.beta_posterior import (
    effective_reliability,
    expected_reliability,
    make_reliability_table,
    uniform_answer_prior,
    uniform_category_prior,
    update_answer_posterior,
    update_category_posterior_on_response,
    update_reliability_table,
)

NUM_CANDIDATES = 4
NUM_CATEGORIES = 5
from credence.inference.decision import (
    ActionType,
    QuestionState,
    select_action,
)
from credence.inference.voi import (
    ToolConfig,
    compute_voi,
    eu_abstain,
    eu_star,
    eu_submit,
)


# =============================================================================
# Group 1: Beta Posterior Basics
# =============================================================================


class TestBetaPosteriorBasics:
    """Hand-verified Beta distribution arithmetic."""

    def test_uniform_prior_mean_and_variance(self):
        """Beta(1,1): mean = 0.5, variance = 1*1/(2^2 * 3) = 1/12."""
        a, b = 1.0, 1.0
        mean = expected_reliability(a, b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1))

        print(f"Beta({a},{b}): mean={mean}, variance={variance:.6f}")
        print(f"  Expected: mean=0.5, variance={1/12:.6f}")

        assert mean == pytest.approx(0.5)
        assert variance == pytest.approx(1.0 / 12.0)

    def test_eight_successes_two_failures(self):
        """Beta(1,1) + 8 successes + 2 failures -> Beta(9,3), mean = 9/12 = 0.75."""
        table = make_reliability_table(1)
        cat_post = np.zeros(NUM_CATEGORIES)
        cat_post[0] = 1.0

        for _ in range(8):
            table = update_reliability_table(table, 0, cat_post, True)
        for _ in range(2):
            table = update_reliability_table(table, 0, cat_post, False)

        alpha = table[0, 0, 0]
        beta_ = table[0, 0, 1]
        mean = expected_reliability(alpha, beta_)
        variance = alpha * beta_ / ((alpha + beta_) ** 2 * (alpha + beta_ + 1))

        print(f"After 8 successes, 2 failures: Beta({alpha},{beta_})")
        print(f"  mean={mean}, variance={variance:.6f}")
        print(f"  Expected: Beta(9,3), mean=0.75, variance={9*3/(12**2*13):.6f}")

        assert alpha == pytest.approx(9.0)
        assert beta_ == pytest.approx(3.0)
        assert mean == pytest.approx(0.75)
        assert variance == pytest.approx(9.0 * 3.0 / (144.0 * 13.0))

    def test_forgetting_increases_variance(self):
        """Forgetting lambda=0.5 on Beta(9,3) -> Beta(4.5,1.5): same mean, higher variance."""
        table = make_reliability_table(1)
        cat_post = np.zeros(NUM_CATEGORIES)
        cat_post[0] = 1.0

        for _ in range(8):
            table = update_reliability_table(table, 0, cat_post, True)
        for _ in range(2):
            table = update_reliability_table(table, 0, cat_post, False)

        orig_var = 9.0 * 3.0 / (12.0**2 * 13.0)

        # Trigger decay without adding data: zero-weight category posterior
        zero_cat = np.zeros(NUM_CATEGORIES)
        decayed = update_reliability_table(table, 0, zero_cat, True, forgetting=0.5)

        alpha_d = decayed[0, 0, 0]
        beta_d = decayed[0, 0, 1]
        mean_d = expected_reliability(alpha_d, beta_d)
        var_d = alpha_d * beta_d / ((alpha_d + beta_d) ** 2 * (alpha_d + beta_d + 1))

        print(f"After forgetting: Beta({alpha_d},{beta_d})")
        print(f"  mean={mean_d:.6f}, variance={var_d:.6f}")
        print(f"  Original: mean=0.75, variance={orig_var:.6f}")
        print(f"  Variance increased: {var_d:.6f} > {orig_var:.6f} = {var_d > orig_var}")

        assert alpha_d == pytest.approx(4.5)
        assert beta_d == pytest.approx(1.5)
        assert mean_d == pytest.approx(0.75)
        assert var_d == pytest.approx(4.5 * 1.5 / (36.0 * 7.0))
        assert var_d > orig_var


# =============================================================================
# Group 2: VOI Invariants
# =============================================================================


class TestVOIInvariants:
    """Value-of-information structural properties."""

    @staticmethod
    def _table_with_reliability(r: float) -> np.ndarray:
        """Reliability table where all categories have E[r] = r.

        Uses Beta(r*1000, (1-r)*1000) for precise mean.
        Near-perfect (r>0.999) uses Beta(10000,1).
        """
        table = make_reliability_table(1)
        if r > 0.999:
            table[0, :, 0] = 10000.0
            table[0, :, 1] = 1.0
        else:
            table[0, :, 0] = r * 1000
            table[0, :, 1] = (1.0 - r) * 1000
        return table

    def test_voi_non_negative_random(self):
        """VOI >= 0 across 100 random configurations."""
        rng = np.random.default_rng(42)
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))

        for i in range(100):
            post = rng.dirichlet(np.ones(NUM_CANDIDATES))
            cat_post = rng.dirichlet(np.ones(NUM_CATEGORIES))
            table = make_reliability_table(1)
            table[0, :, 0] = rng.uniform(1, 10, NUM_CATEGORIES)
            table[0, :, 1] = rng.uniform(1, 10, NUM_CATEGORIES)

            voi = compute_voi(post, table, 0, cat_post, config)
            assert voi >= -1e-12, f"Trial {i}: VOI = {voi} < 0"

    def test_perfect_tool_uniform_prior(self):
        """Perfect tool (r~1) on uniform prior: VOI ~ 10.

        EU*(current) = max(0.25*10 + 0.75*(-5), 0) = max(-1.25, 0) = 0.
        After response: posterior ~ point mass, EU* ~ 10.
        Expected EU after = 4 * 0.25 * 10 = 10.  VOI = 10 - 0 = 10.
        """
        table = self._table_with_reliability(0.9999)
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))

        voi = compute_voi(uniform_answer_prior(), table, 0, uniform_category_prior(), config)

        print(f"Perfect tool VOI: {voi:.4f} (expected ~ 10.0)")
        assert voi == pytest.approx(10.0, abs=0.05)

    def test_useless_tool_uniform_prior(self):
        """Useless tool (r=0.25, 4 candidates): all likelihoods equal, VOI = 0.

        L(j|x_i) = 0.25 for all i,j since (1-0.25)/3 = 0.25 = r.
        Response is pure noise -> posterior unchanged.
        """
        table = self._table_with_reliability(0.25)
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))

        voi = compute_voi(uniform_answer_prior(), table, 0, uniform_category_prior(), config)

        print(f"Useless tool VOI: {voi:.8f} (expected = 0.0)")
        assert voi == pytest.approx(0.0, abs=1e-6)

    def test_certain_posterior_voi_zero(self):
        """Already-certain posterior [1,0,0,0]: no information gain, VOI = 0.

        Regardless of response, posterior remains [1,0,0,0].
        """
        table = self._table_with_reliability(0.8)
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
        prior = np.array([1.0, 0.0, 0.0, 0.0])

        voi = compute_voi(prior, table, 0, uniform_category_prior(), config)

        print(f"Certain-posterior VOI: {voi:.10f} (expected = 0.0)")
        assert voi == pytest.approx(0.0, abs=1e-10)

    def test_more_uncertain_prior_higher_voi(self):
        """More uncertain prior -> higher VOI.

        Uniform [0.25]*4 is maximally uncertain (flat), [0.7,0.1,0.1,0.1] is peaked.
        Information is more valuable when you're more uncertain.
        """
        table = self._table_with_reliability(0.7)
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))
        cat_post = uniform_category_prior()

        voi_uniform = compute_voi(uniform_answer_prior(), table, 0, cat_post, config)
        voi_peaked = compute_voi(
            np.array([0.7, 0.1, 0.1, 0.1]), table, 0, cat_post, config,
        )

        print(f"VOI (uniform prior): {voi_uniform:.4f}")
        print(f"VOI (peaked prior):  {voi_peaked:.4f}")
        assert voi_uniform > voi_peaked


# =============================================================================
# Group 3: Decision Logic Hand-Worked Example
# =============================================================================


class TestDecisionHandWorked:
    """Full hand-worked VOI and action-selection calculation.

    Setup: prior = [0.4, 0.3, 0.2, 0.1], one tool with r=0.8, cost=1, full coverage.

    Hand computation:
      EU(submit) = 0.4*10 + 0.6*(-5) = 1.0
      EU(abstain) = 0

      r = 0.8, wrong_lik = (1-0.8)/3 = 1/15

      P(resp=j):
        j=0: 54/150   j=1: 43/150   j=2: 32/150   j=3: 21/150

      EU* per response:
        j=0: 25/3     j=1: 325/43   j=2: 25/4     j=3: 25/7

      E[EU* after] = 54/150 * 25/3 + 43/150 * 325/43 + 32/150 * 25/4 + 21/150 * 25/7
                   = 18/6 + 13/6 + 8/6 + 3/6 = 42/6 = 7.0

      VOI = 7.0 - 1.0 = 6.0
      Net EU(query) = 6.0 - 1.0 (cost) = 5.0
    """

    PRIOR = np.array([0.4, 0.3, 0.2, 0.1])
    R = 0.8
    WRONG_LIK = (1.0 - R) / 3.0  # 1/15

    @staticmethod
    def _setup():
        """Table with E[r]=0.8 everywhere (Beta(4,1)), full coverage, cost=1."""
        table = make_reliability_table(1)
        table[0, :, 0] = 4.0
        table[0, :, 1] = 1.0
        cat_post = uniform_category_prior()
        return table, cat_post

    def test_eu_submit_and_abstain(self):
        """EU(submit) = 0.4*10 + 0.6*(-5) = 1.0; EU(abstain) = 0."""
        eu_sub = eu_submit(self.PRIOR)
        eu_abs = eu_abstain()

        print(f"EU(submit)  = {eu_sub:.4f} (expected 1.0)")
        print(f"EU(abstain) = {eu_abs:.4f} (expected 0.0)")

        assert eu_sub == pytest.approx(1.0)
        assert eu_abs == pytest.approx(0.0)

    def test_response_probabilities(self):
        """P(resp=j) = prior[j]*r + (1-prior[j])*wrong_lik."""
        p_resp = np.array([
            self.PRIOR[j] * self.R + (1.0 - self.PRIOR[j]) * self.WRONG_LIK
            for j in range(NUM_CANDIDATES)
        ])
        expected = np.array([54.0 / 150, 43.0 / 150, 32.0 / 150, 21.0 / 150])

        print(f"P(response): {p_resp}")
        print(f"Expected:    {expected}")
        print(f"Sum:         {p_resp.sum():.10f}")

        assert_allclose(p_resp, expected, atol=1e-10)
        assert p_resp.sum() == pytest.approx(1.0)

    def test_voi_matches_hand_computation(self):
        """Code VOI must match the hand-derived value of 6.0."""
        table, cat_post = self._setup()
        config = ToolConfig(cost=0.0, coverage_by_category=np.ones(NUM_CATEGORIES))

        # Hand-computed EU* per response
        eu_per_resp = np.array([25.0 / 3, 325.0 / 43, 25.0 / 4, 25.0 / 7])
        p_resp = np.array([54.0 / 150, 43.0 / 150, 32.0 / 150, 21.0 / 150])

        expected_eu_after = float(np.dot(p_resp, eu_per_resp))
        voi_hand = expected_eu_after - eu_star(self.PRIOR)

        print(f"E[EU* after query] = {expected_eu_after:.6f} (expected 7.0)")
        print(f"VOI (hand)         = {voi_hand:.6f} (expected 6.0)")

        assert expected_eu_after == pytest.approx(7.0, abs=1e-10)
        assert voi_hand == pytest.approx(6.0)

        voi_code = compute_voi(self.PRIOR, table, 0, cat_post, config)
        print(f"VOI (code)         = {voi_code:.6f}")

        assert voi_code == pytest.approx(6.0, abs=1e-4)

    def test_select_action_picks_query(self):
        """EU(query) = eu_current + VOI - cost = 1.0 + 6.0 - 1.0 = 6.0 > EU(submit) = 1.0."""
        table, cat_post = self._setup()
        config = ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES))

        state = QuestionState(
            answer_posterior=self.PRIOR,
            category_posterior=cat_post,
            used_tools=frozenset(),
            tool_responses={},
        )

        action = select_action(state, table, [config])

        print(f"Action: {action.action_type.name}, tool={action.tool_idx}, EU={action.eu:.4f}")
        print(f"Expected: QUERY, tool=0, EU=6.0")

        assert action.action_type == ActionType.QUERY
        assert action.tool_idx == 0
        assert action.eu == pytest.approx(6.0, abs=1e-4)


# =============================================================================
# Group 4: Bayesian Update Hand-Worked Example
# =============================================================================


class TestBayesianUpdateHandWorked:
    """Hand-verified Bayes' rule update.

    Prior [0.25]*4, r=0.8, tool returns candidate 0.

    Likelihoods: L(0|x_0)=0.8, L(0|x_i!=0)=(1-0.8)/3 = 1/15
    Unnormalised: [0.25*0.8, 0.25/15, 0.25/15, 0.25/15] = [0.2, 1/60, 1/60, 1/60]
    Sum = 0.2 + 3/60 = 0.2 + 0.05 = 0.25
    Normalised: [0.8, 1/15, 1/15, 1/15]
    """

    def test_likelihood_structure(self):
        """L(correct) = r = 0.8, L(wrong) = (1-r)/3 = 1/15."""
        r = 0.8
        wrong_lik = (1.0 - r) / (NUM_CANDIDATES - 1)

        print(f"L(correct) = {r}")
        print(f"L(wrong)   = {wrong_lik:.10f} (expected {1/15:.10f})")

        assert wrong_lik == pytest.approx(1.0 / 15.0)

    def test_unnormalised_values(self):
        """Unnormalised posterior sums to 0.25."""
        prior = uniform_answer_prior()
        r = 0.8
        wrong_lik = (1.0 - r) / 3.0

        unnorm = prior * np.array([r, wrong_lik, wrong_lik, wrong_lik])

        print(f"Unnormalised: {unnorm}")
        print(f"Sum: {unnorm.sum():.10f} (expected 0.25)")

        assert_allclose(unnorm, [0.2, 0.25 / 15, 0.25 / 15, 0.25 / 15], atol=1e-10)
        assert unnorm.sum() == pytest.approx(0.25)

    def test_normalised_posterior(self):
        """Normalised posterior = [0.8, 1/15, 1/15, 1/15]."""
        posterior = update_answer_posterior(uniform_answer_prior(), response_idx=0, r_effective=0.8)
        expected = np.array([0.8, 1.0 / 15, 1.0 / 15, 1.0 / 15])

        print(f"Posterior: {posterior}")
        print(f"Expected:  {expected}")

        assert_allclose(posterior, expected, atol=1e-10)
        assert posterior.sum() == pytest.approx(1.0)


# =============================================================================
# Group 5: Category Conditioning Check
# =============================================================================


class TestCategoryConditioning:
    """Verify two-stage update correctly conditions on response type.

    Simplified 2-category setup:
      P(numerical)=0.3 at index 1, P(other)=0.7 at index 4.
      Tool covers only numerical: coverage = [0,1,0,0,0].

    Tool returning an answer -> P(numerical) collapses to 1.0.
    effective_reliability must use the UPDATED category posterior.
    """

    CAT_POST = np.array([0.0, 0.3, 0.0, 0.0, 0.7])
    COVERAGE = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    def test_category_collapses_on_answer(self):
        """Getting an answer with coverage [0,1,0,0,0] -> P(numerical) = 1.0.

        Weights = coverage = [0,1,0,0,0].
        Updated = [0*0, 0.3*1, 0*0, 0*0, 0.7*0] = [0, 0.3, 0, 0, 0].
        Normalised: [0, 1, 0, 0, 0].
        """
        updated = update_category_posterior_on_response(
            self.CAT_POST, self.COVERAGE, got_answer=True,
        )

        print(f"Original: {self.CAT_POST}")
        print(f"Updated:  {updated}")
        print(f"P(numerical) = {updated[1]:.4f} (expected 1.0)")

        assert_allclose(updated, [0.0, 1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_effective_reliability_with_updated_vs_original(self):
        """After category collapse, r_eff ~ 0.99, NOT the unconditioned 0.647.

        Tool reliability: numerical (idx 1) = Beta(100,1) -> E[r] = 100/101.
        Other categories remain Beta(1,1) -> E[r] = 0.5.

        Correct (updated posterior [0,1,0,0,0]):
          r_eff = 1.0 * (100/101) = 0.9901

        Wrong (original posterior [0, 0.3, 0, 0, 0.7]):
          r_eff = 0.3 * (100/101) + 0.7 * 0.5 = 0.6470
        """
        table = make_reliability_table(1)
        table[0, 1, 0] = 100.0  # alpha for numerical
        table[0, 1, 1] = 1.0    # beta for numerical

        updated_cat = update_category_posterior_on_response(
            self.CAT_POST, self.COVERAGE, got_answer=True,
        )

        r_correct = effective_reliability(table, 0, updated_cat)
        r_wrong = effective_reliability(table, 0, self.CAT_POST)

        print(f"r_eff (updated posterior): {r_correct:.6f} (expected {100/101:.6f})")
        print(f"r_eff (original posterior): {r_wrong:.6f} (expected {0.3*100/101 + 0.7*0.5:.6f})")

        assert r_correct == pytest.approx(100.0 / 101.0, abs=1e-4)
        assert r_wrong == pytest.approx(0.3 * (100.0 / 101.0) + 0.7 * 0.5, abs=1e-4)
        assert r_correct > r_wrong
