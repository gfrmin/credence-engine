"""Credence: Bayesian decision-theoretic agents."""

# Inference layer
from credence.inference.beta_posterior import (
    AnswerPosterior,
    CategoryPosterior,
    ReliabilityTable,
    make_reliability_table,
    uniform_answer_prior,
    uniform_category_prior,
    effective_reliability,
    expected_reliability,
    update_answer_posterior,
    update_category_posterior_on_response,
    update_reliability_table,
)
from credence.inference.voi import (
    ScoringRule,
    ToolConfig,
    eu_submit,
    eu_abstain,
    eu_star,
    compute_voi,
)
from credence.inference.decision import (
    ActionType,
    Action,
    QuestionState,
    initial_question_state,
    select_action,
    apply_tool_response,
    compute_reliability_updates,
    apply_reliability_updates,
)

# Agent
from credence.agents.bayesian_agent import BayesianAgent
from credence.agents.common import AgentResult, DecisionStep

__all__ = [
    # Beta posteriors
    "AnswerPosterior", "CategoryPosterior", "ReliabilityTable",
    "make_reliability_table", "uniform_answer_prior", "uniform_category_prior",
    "effective_reliability", "expected_reliability",
    "update_answer_posterior", "update_category_posterior_on_response",
    "update_reliability_table",
    # VOI / EU
    "ScoringRule", "ToolConfig",
    "eu_submit", "eu_abstain", "eu_star", "compute_voi",
    # Decision loop
    "ActionType", "Action", "QuestionState",
    "initial_question_state", "select_action", "apply_tool_response",
    "compute_reliability_updates", "apply_reliability_updates",
    # Agent
    "BayesianAgent", "AgentResult", "DecisionStep",
]
