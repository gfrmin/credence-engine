"""Tests for EU-based action selection and state management."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from credence.inference.beta_posterior import (
    make_reliability_table,
    uniform_answer_prior,
    uniform_category_prior,
)

NUM_CANDIDATES = 4
NUM_CATEGORIES = 5
from credence.inference.decision import (
    Action,
    ActionType,
    QuestionState,
    apply_reliability_updates,
    apply_tool_response,
    compute_binary_reliability_updates,
    compute_reliability_updates,
    initial_question_state,
    select_action,
)
from credence.inference.voi import ScoringRule, ToolConfig


# --- Helper: standard 4-tool configs matching the spec ---

def make_spec_tool_configs() -> list[ToolConfig]:
    """Tool configs from the spec: A(cost=1), B(cost=2), C(cost=1), D(cost=2)."""
    return [
        ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES)),         # A
        ToolConfig(cost=2.0, coverage_by_category=np.array([0.65, 0.3, 0.35, 0.55, 0.2])),  # B
        ToolConfig(cost=1.0, coverage_by_category=np.array([0.0, 1.0, 0.0, 0.0, 0.0])),  # C
        ToolConfig(cost=2.0, coverage_by_category=np.ones(NUM_CATEGORIES)),         # D
    ]


# --- Action selection ---


def test_confident_posterior_submits():
    """Confident posterior (p=0.95) should submit."""
    cat_post = uniform_category_prior()
    state = initial_question_state(cat_post)
    state = state._replace(answer_posterior=np.array([0.95, 0.02, 0.02, 0.01]))
    table = make_reliability_table(4)
    configs = make_spec_tool_configs()
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.SUBMIT
    assert action.answer_idx == 0


def test_uniform_posterior_queries_cheap_informative_tool():
    """Uniform posterior + cheap informative tool -> QUERY."""
    cat_post = uniform_category_prior()
    state = initial_question_state(cat_post)
    table = make_reliability_table(4)
    table[:, :, 0] = 9.0  # All tools E = 0.9
    configs = make_spec_tool_configs()
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.QUERY


def test_uniform_posterior_no_tools_left_abstains():
    """Uniform posterior + all tools used -> ABSTAIN."""
    cat_post = uniform_category_prior()
    state = QuestionState(
        answer_posterior=uniform_answer_prior(),
        category_posterior=cat_post,
        used_tools=frozenset({0, 1, 2, 3}),
        tool_responses={0: 0, 1: 1, 2: None, 3: 2},
    )
    table = make_reliability_table(4)
    configs = make_spec_tool_configs()
    action = select_action(state, table, configs)
    # eu_submit for uniform = -1.25, eu_abstain = 0 -> ABSTAIN
    assert action.action_type == ActionType.ABSTAIN


def test_expensive_tools_low_voi_abstains():
    """Expensive tools with low VOI -> ABSTAIN over QUERY."""
    cat_post = uniform_category_prior()
    state = initial_question_state(cat_post)
    table = make_reliability_table(1)
    # Uninformative tool: r = 0.25
    table[0, :, 0] = 1.0
    table[0, :, 1] = 3.0
    # Very expensive
    configs = [ToolConfig(cost=100.0, coverage_by_category=np.ones(NUM_CATEGORIES))]
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.ABSTAIN


def test_submit_tiebreak_over_abstain():
    """At p=1/3, eu_submit = 0 = eu_abstain. Tie-break: SUBMIT wins (>= comparison)."""
    post = np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6])
    cat_post = uniform_category_prior()
    state = QuestionState(
        answer_posterior=post,
        category_posterior=cat_post,
        used_tools=frozenset({0, 1, 2, 3}),
        tool_responses={},
    )
    table = make_reliability_table(4)
    configs = make_spec_tool_configs()
    action = select_action(state, table, configs)
    # eu_submit = 0.0 = eu_abstain, SUBMIT wins tie
    assert action.action_type == ActionType.SUBMIT


# --- State management ---


def test_initial_state_uniform_answer():
    """initial_question_state produces uniform answer posterior."""
    cat_post = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
    state = initial_question_state(cat_post)
    assert_allclose(state.answer_posterior, 0.25)
    assert_allclose(state.category_posterior, cat_post)
    assert state.used_tools == frozenset()
    assert state.tool_responses == {}


def test_apply_tool_response_with_answer_updates_posterior():
    """apply_tool_response with a candidate answer updates answer posterior."""
    cat_post = uniform_category_prior()
    state = initial_question_state(cat_post)
    table = make_reliability_table(4)
    table[0, :, 0] = 9.0  # E = 0.9
    config = ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES))

    new_state = apply_tool_response(state, 0, 2, table, config)
    # Answer posterior should concentrate on candidate 2
    assert new_state.answer_posterior[2] > 0.5
    assert new_state.answer_posterior.sum() == pytest.approx(1.0)


def test_apply_tool_response_with_none_updates_category():
    """apply_tool_response with None updates category posterior only."""
    cat_post = uniform_category_prior()
    state = initial_question_state(cat_post)
    table = make_reliability_table(4)
    # Tool C coverage
    coverage = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    config = ToolConfig(cost=1.0, coverage_by_category=coverage)

    new_state = apply_tool_response(state, 2, None, table, config)
    # Category posterior should eliminate numerical
    assert new_state.category_posterior[1] == pytest.approx(0.0)
    # Answer posterior should be unchanged
    assert_allclose(new_state.answer_posterior, state.answer_posterior)


def test_apply_tool_response_adds_to_used_tools():
    """apply_tool_response adds tool to used_tools."""
    state = initial_question_state(uniform_category_prior())
    table = make_reliability_table(4)
    config = ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES))

    new_state = apply_tool_response(state, 1, 0, table, config)
    assert 1 in new_state.used_tools
    assert new_state.tool_responses[1] == 0


def test_apply_tool_response_immutability():
    """Original state is unchanged after apply_tool_response."""
    state = initial_question_state(uniform_category_prior())
    original_ans = state.answer_posterior.copy()
    original_cat = state.category_posterior.copy()
    table = make_reliability_table(4)
    table[0, :, 0] = 9.0
    config = ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES))

    apply_tool_response(state, 0, 1, table, config)

    assert_allclose(state.answer_posterior, original_ans)
    assert_allclose(state.category_posterior, original_cat)
    assert state.used_tools == frozenset()
    assert state.tool_responses == {}


# --- Reliability updates ---


def test_reliability_updates_correct_submission():
    """Correct submission: agreeing tool -> True, disagreeing -> False."""
    responses = {0: 2, 1: 2, 2: 0, 3: None}
    updates = compute_reliability_updates(
        submitted_answer_idx=2, was_correct=True, tool_responses=responses,
    )
    assert updates[0] is True   # agreed with correct answer
    assert updates[1] is True
    assert updates[2] is False  # disagreed
    assert updates[3] is None   # returned None


def test_reliability_updates_wrong_submission():
    """Wrong submission: agreeing tool -> False, disagreeing -> None."""
    responses = {0: 1, 1: 3, 2: None}
    updates = compute_reliability_updates(
        submitted_answer_idx=1, was_correct=False, tool_responses=responses,
    )
    assert updates[0] is False  # agreed with wrong answer -> definitely wrong
    assert updates[1] is None   # disagreed -> might be right, might be different wrong
    assert updates[2] is None   # returned None


def test_reliability_updates_abstain():
    """Abstain: no ground truth, all None."""
    responses = {0: 2, 1: 0}
    updates = compute_reliability_updates(
        submitted_answer_idx=None, was_correct=None, tool_responses=responses,
    )
    assert all(v is None for v in updates.values())


def test_apply_reliability_updates_modifies_table():
    """apply_reliability_updates produces correct alpha/beta changes."""
    table = make_reliability_table(4)
    cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # point-mass on factual
    updates = {0: True, 1: False, 2: None}
    new_table = apply_reliability_updates(table, updates, cat_post)
    # Tool 0 correct: alpha[0][factual] += 1.0
    assert new_table[0, 0, 0] == pytest.approx(2.0)
    assert new_table[0, 0, 1] == pytest.approx(1.0)
    # Tool 1 wrong: beta[1][factual] += 1.0
    assert new_table[1, 0, 0] == pytest.approx(1.0)
    assert new_table[1, 0, 1] == pytest.approx(2.0)
    # Tool 2 None: unchanged
    assert new_table[2, 0, 0] == pytest.approx(1.0)
    assert new_table[2, 0, 1] == pytest.approx(1.0)
    # Original unchanged
    assert_allclose(table, make_reliability_table(4))


# --- Integration: micro-scenario ---


def test_micro_scenario_full_decision_loop():
    """Micro-scenario: 2 tools, known reliabilities, trace full loop for 1 question.

    Tool 0: cheap (cost=1), r=0.9 for all categories, full coverage.
    Tool 1: expensive (cost=5), r=0.95 for all categories, full coverage.

    With uniform prior, agent should query tool 0 first (high VOI, low cost).
    After tool 0 responds with candidate 0, posterior should concentrate.
    Then agent should submit (VOI of tool 1 - cost < EU_submit).
    """
    table = make_reliability_table(2)
    table[0, :, 0] = 9.0   # E = 0.9
    table[0, :, 1] = 1.0
    table[1, :, 0] = 19.0  # E = 0.95
    table[1, :, 1] = 1.0

    configs = [
        ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES)),
        ToolConfig(cost=5.0, coverage_by_category=np.ones(NUM_CATEGORIES)),
    ]

    state = initial_question_state(uniform_category_prior())

    # Step 1: should query tool 0
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.QUERY
    assert action.tool_idx == 0

    # Simulate tool 0 responding with candidate 0
    state = apply_tool_response(state, 0, 0, table, configs[0])
    assert state.answer_posterior[0] > 0.8

    # Step 2: should submit (not query expensive tool 1)
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.SUBMIT
    assert action.answer_idx == 0


def test_voi_exceeding_cost_triggers_query():
    """VOI > cost should trigger a query even when eu_current is positive.

    Regression test for the 2x comparison bug: select_action used to compare
    (VOI - cost) against eu_current instead of (eu_current + VOI - cost).
    This meant VOI had to exceed (cost + eu_current) instead of just cost.
    """
    # Setup: moderate posterior where eu_submit > 0
    post = np.array([0.5, 0.2, 0.2, 0.1])
    cat_post = uniform_category_prior()
    state = QuestionState(
        answer_posterior=post,
        category_posterior=cat_post,
        used_tools=frozenset(),
        tool_responses={},
    )

    # Scoring with meaningful penalty so VOI is nonzero
    scoring = ScoringRule(reward_correct=1.0, penalty_wrong=-0.5, reward_abstain=-0.05)

    # Tool with r_eff=0.7 and low cost
    table = make_reliability_table(1)
    table[0, :, 0] = 7.0   # alpha=7, beta=3 → r_eff=0.7
    table[0, :, 1] = 3.0

    configs = [ToolConfig(cost=0.02, coverage_by_category=np.ones(NUM_CATEGORIES))]

    action = select_action(state, table, configs, scoring)

    # With the fix, VOI (~0.21) > cost (0.02) → query
    # Without the fix, VOI - cost (0.19) < eu_submit (0.25) → wrongly submits
    assert action.action_type == ActionType.QUERY
    assert action.tool_idx == 0


def test_learning_over_questions():
    """After several questions, agent learns to prefer reliable tool."""
    table = make_reliability_table(2)
    cat_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # all factual

    # Simulate 10 questions where tool 0 is always correct, tool 1 always wrong
    for _ in range(10):
        table = apply_reliability_updates(
            table, {0: True, 1: False}, cat_post,
        )

    # Tool 0 should now have much higher reliability than tool 1
    from credence.inference.beta_posterior import expected_reliability
    r0 = expected_reliability(table[0, 0, 0], table[0, 0, 1])
    r1 = expected_reliability(table[1, 0, 0], table[1, 0, 1])
    assert r0 > 0.8
    assert r1 < 0.2

    # With equal cost, agent should query tool 0 over tool 1
    configs = [
        ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES)),
        ToolConfig(cost=1.0, coverage_by_category=np.ones(NUM_CATEGORIES)),
    ]
    state = initial_question_state(cat_post)
    action = select_action(state, table, configs)
    assert action.action_type == ActionType.QUERY
    assert action.tool_idx == 0


# --- Binary reliability updates ---


def test_binary_updates_correct():
    """Correct outcome: responding tools get True, None-tools get None."""
    responses: dict[int, int | None] = {0: 2, 1: 0, 2: None}
    updates = compute_binary_reliability_updates(responses, was_correct=True)
    assert updates[0] is True
    assert updates[1] is True
    assert updates[2] is None


def test_binary_updates_wrong():
    """Wrong outcome: responding tools get False, None-tools get None."""
    responses: dict[int, int | None] = {0: 1, 1: None}
    updates = compute_binary_reliability_updates(responses, was_correct=False)
    assert updates[0] is False
    assert updates[1] is None


def test_binary_updates_unknown():
    """Unknown outcome: all tools get None."""
    responses: dict[int, int | None] = {0: 0, 1: 2}
    updates = compute_binary_reliability_updates(responses, was_correct=None)
    assert all(v is None for v in updates.values())
