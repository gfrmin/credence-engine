"""Credence: Bayesian decision-theoretic agents."""

# Inference layer
from src.inference.beta_posterior import (
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
from src.inference.voi import (
    ScoringRule,
    ToolConfig,
    eu_submit,
    eu_abstain,
    eu_star,
    compute_voi,
)
from src.inference.decision import (
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
from src.agents.bayesian_agent import BayesianAgent
from src.agents.common import AgentResult, DecisionStep

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
