"""Tests for baseline agents and LangChain agents (sans LLM calls).

Verifies all agents run through the benchmark without errors, produce valid
results, and satisfy expected score orderings.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.baselines import (
    AllToolsAgent,
    OracleAgent,
    RandomAgent,
    SingleBestToolAgent,
)
from src.agents.bayesian_agent import BayesianAgent
from src.agents.common import DecisionStep
from src.environment.benchmark import run_benchmark
from src.environment.questions import Question, get_questions
from src.environment.tools import make_spec_tools, tool_config_for


# --- Fixtures ---

@pytest.fixture
def spec_tools():
    return make_spec_tools()


@pytest.fixture
def tool_configs(spec_tools):
    return [tool_config_for(t) for t in spec_tools]


@pytest.fixture
def questions_5():
    return get_questions(seed=42)[:5]


@pytest.fixture
def questions_50():
    return get_questions(seed=42)


# --- Helper to verify AgentResult-like records ---

def _assert_valid_records(result):
    """Check that benchmark records have valid structure."""
    for rec in result.records:
        assert rec.action_type in ("submit", "abstain")
        assert isinstance(rec.tools_queried, tuple)
        assert isinstance(rec.tool_responses, dict)
        assert isinstance(rec.reward, float)
        assert isinstance(rec.tool_cost, float)
        if rec.action_type == "submit":
            assert rec.submitted_answer is not None
            assert rec.was_correct is not None
        else:
            assert rec.submitted_answer is None
            assert rec.was_correct is None


# --- RandomAgent tests ---

class TestRandomAgent:
    def test_runs_5_questions(self, spec_tools, questions_5):
        agent = RandomAgent(num_tools=4, seed=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        assert len(result.records) == 5
        _assert_valid_records(result)

    def test_queries_one_tool_per_question(self, spec_tools, questions_5):
        agent = RandomAgent(num_tools=4, seed=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert len(rec.tools_queried) == 1

    def test_always_submits(self, spec_tools, questions_5):
        agent = RandomAgent(num_tools=4, seed=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.action_type == "submit"

    def test_belief_snapshot(self, spec_tools, questions_5):
        agent = RandomAgent(num_tools=4, seed=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.belief_snapshot is not None


# --- AllToolsAgent tests ---

class TestAllToolsAgent:
    def test_runs_5_questions(self, spec_tools, questions_5):
        agent = AllToolsAgent(num_tools=4)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        assert len(result.records) == 5
        _assert_valid_records(result)

    def test_queries_all_tools(self, spec_tools, questions_5):
        agent = AllToolsAgent(num_tools=4)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert len(rec.tools_queried) == 4

    def test_always_submits(self, spec_tools, questions_5):
        agent = AllToolsAgent(num_tools=4)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.action_type == "submit"

    def test_tool_cost(self, spec_tools, questions_5):
        """Each question costs 1+3+1+2 = 7 points in tools."""
        agent = AllToolsAgent(num_tools=4)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.tool_cost == 7.0


# --- SingleBestToolAgent tests ---

class TestSingleBestToolAgent:
    def test_runs_5_questions(self, spec_tools, questions_5):
        agent = SingleBestToolAgent(tool_idx=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        assert len(result.records) == 5
        _assert_valid_records(result)

    def test_always_queries_tool_a(self, spec_tools, questions_5):
        agent = SingleBestToolAgent(tool_idx=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.tools_queried == (0,)

    def test_tool_cost_is_1(self, spec_tools, questions_5):
        agent = SingleBestToolAgent(tool_idx=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            assert rec.tool_cost == 1.0


# --- OracleAgent tests ---

class TestOracleAgent:
    def test_runs_5_questions(self, spec_tools, tool_configs, questions_5):
        agent = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        assert len(result.records) == 5
        _assert_valid_records(result)

    def test_has_true_reliabilities(self, spec_tools, tool_configs):
        agent = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        # Tool C: numerical reliability should be near 1.0
        r_table = agent.reliability_table
        alpha = r_table[2, 1, 0]  # tool C, numerical, alpha
        beta = r_table[2, 1, 1]   # tool C, numerical, beta
        expected_r = alpha / (alpha + beta)
        assert abs(expected_r - 1.0) < 0.01

    def test_does_not_learn(self, spec_tools, tool_configs, questions_5):
        """Oracle's reliability table should not change after questions."""
        agent = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        table_before = agent.reliability_table.copy()
        run_benchmark(agent, spec_tools, questions_5, seed=42)
        np.testing.assert_array_equal(agent.reliability_table, table_before)


# --- Score ordering tests (50 questions for statistical stability) ---

class TestScoreOrdering:
    def test_oracle_beats_bayesian(self, spec_tools, tool_configs, questions_50):
        """Oracle (perfect info) should score >= Bayesian (learned info)."""
        oracle = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        bayesian = BayesianAgent(tool_configs=tool_configs)

        r_oracle = run_benchmark(oracle, spec_tools, questions_50, seed=42)
        r_bayesian = run_benchmark(bayesian, spec_tools, questions_50, seed=42)

        # Oracle has perfect info, so it should do at least as well.
        # Allow small tolerance for stochastic tool responses.
        assert r_oracle.total_score >= r_bayesian.total_score - 10.0, (
            f"Oracle ({r_oracle.total_score:.1f}) should beat "
            f"Bayesian ({r_bayesian.total_score:.1f})"
        )

    def test_all_tools_more_accurate_than_random(self, spec_tools, questions_50):
        """AllTools (majority vote) should have higher accuracy than Random.

        AllTools pays 7 points/question in tool costs, so its net score may be
        lower than Random's. But its raw accuracy (reward ignoring costs) should
        be higher — that's the point of querying all tools.
        """
        random_agent = RandomAgent(num_tools=4, seed=0)
        all_tools = AllToolsAgent(num_tools=4)

        r_random = run_benchmark(random_agent, spec_tools, questions_50, seed=42)
        r_all = run_benchmark(all_tools, spec_tools, questions_50, seed=42)

        assert r_all.total_reward > r_random.total_reward, (
            f"AllTools reward ({r_all.total_reward:.1f}) should beat "
            f"Random reward ({r_random.total_reward:.1f})"
        )

    def test_oracle_beats_single_best(self, spec_tools, tool_configs, questions_50):
        """Oracle should beat single-tool strategy."""
        oracle = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        single = SingleBestToolAgent(tool_idx=0)

        r_oracle = run_benchmark(oracle, spec_tools, questions_50, seed=42)
        r_single = run_benchmark(single, spec_tools, questions_50, seed=42)

        assert r_oracle.total_score >= r_single.total_score - 5.0, (
            f"Oracle ({r_oracle.total_score:.1f}) should beat "
            f"SingleBest ({r_single.total_score:.1f})"
        )


# --- Decision trace tests ---

class TestDecisionTrace:
    def test_random_has_trace(self, spec_tools, questions_5):
        agent = RandomAgent(num_tools=4, seed=0)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            snapshot = rec.belief_snapshot
            assert snapshot is not None

    def test_all_tools_has_trace(self, spec_tools, questions_5):
        agent = AllToolsAgent(num_tools=4)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            snapshot = rec.belief_snapshot
            assert snapshot is not None

    def test_oracle_has_trace(self, spec_tools, tool_configs, questions_5):
        agent = OracleAgent(tools=list(spec_tools), tool_configs=tool_configs)
        result = run_benchmark(agent, spec_tools, questions_5, seed=42)
        for rec in result.records:
            snapshot = rec.belief_snapshot
            assert snapshot is not None
            assert "answer_posterior" in snapshot


# --- LangChain agents (import-only, no LLM calls) ---

class TestLangChainImports:
    """Verify LangChain agents can be imported and constructed without LLM calls."""

    def test_langchain_agent_import(self):
        from src.agents.langchain_agent import LangChainAgent
        agent = LangChainAgent()
        assert agent.name == "langchain_react"

    def test_langchain_enhanced_import(self):
        from src.agents.langchain_enhanced import LangChainEnhancedAgent
        agent = LangChainEnhancedAgent()
        assert agent.name == "langchain_enhanced"

    def test_langchain_agent_protocol(self):
        """LangChainAgent has the required protocol methods."""
        from src.agents.langchain_agent import LangChainAgent
        agent = LangChainAgent()
        assert hasattr(agent, "on_question_start")
        assert hasattr(agent, "choose_action")
        assert hasattr(agent, "on_tool_response")
        assert hasattr(agent, "on_question_end")
        assert hasattr(agent, "name")

    def test_langchain_enhanced_has_history(self):
        from src.agents.langchain_enhanced import LangChainEnhancedAgent
        agent = LangChainEnhancedAgent()
        assert hasattr(agent, "_history")
        assert agent._history == []

    def test_enhanced_system_prompt_includes_strategy(self):
        from src.agents.langchain_enhanced import ENHANCED_SYSTEM_PROMPT
        assert "STRATEGY GUIDANCE" in ENHANCED_SYSTEM_PROMPT
        assert "TOOL SELECTION" in ENHANCED_SYSTEM_PROMPT
        assert "calculator is perfect for maths" in ENHANCED_SYSTEM_PROMPT
