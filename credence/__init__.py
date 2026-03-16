"""Credence: Bayesian decision-theoretic agents.

Inference is handled by the Julia Credence DSL via juliacall.
This package provides the Python agent, types, and benchmark harness.
"""

# Types and configuration
from credence.inference.voi import (
    ScoringRule,
    ToolConfig,
)
from credence.inference.decision import (
    Action,
    ActionType,
    compute_binary_reliability_updates,
    compute_reliability_updates,
)

# Bridge
from credence.julia_bridge import CredenceBridge

# Agent
from credence.agents.bayesian_agent import BayesianAgent
from credence.agents.common import AgentResult, DecisionStep

__all__ = [
    # Types
    "ScoringRule", "ToolConfig",
    "ActionType", "Action",
    "compute_binary_reliability_updates", "compute_reliability_updates",
    # Bridge
    "CredenceBridge",
    # Agent
    "BayesianAgent", "AgentResult", "DecisionStep",
]
