# Credence

**Bayesian decision-theoretic agents — a reusable library and an empirical benchmark.**

Credence provides domain-agnostic Bayesian agents that learn tool reliability from
experience, use value-of-information calculations to decide when to query and when
to abstain, and adapt when reliability changes. No heuristics, no hardcoded routing
— just expected utility maximisation.

The project also includes a head-to-head benchmark comparing the Bayesian agent
against LangChain ReAct agents on a multi-tool question-answering task.

## Using Credence as a Library

```bash
# Core library (numpy + scipy only)
uv sync

# With benchmark dependencies (matplotlib, langchain, etc.)
uv sync --extra benchmark

# With dev tools (pytest, ruff, etc.)
uv sync --extra dev
```

### Facade import

```python
from credence import BayesianAgent, ToolConfig, ScoringRule
```

### Minimal custom-domain example

```python
import numpy as np
from credence import BayesianAgent, ToolConfig, ScoringRule

# Define your domain's categories and tools
categories = ("plot", "character", "setting", "theme")
num_cats = len(categories)
tools = [
    ToolConfig(cost=1.0, coverage_by_category=np.ones(num_cats)),   # wiki_search
    ToolConfig(cost=3.0, coverage_by_category=np.ones(num_cats)),   # deep_read
    ToolConfig(cost=2.0, coverage_by_category=np.ones(num_cats)),   # summary_llm
]

# Category inference: question text -> probability vector over categories
# (or use make_keyword_category_infer_fn for keyword-based heuristics)
def infer_category(text: str) -> np.ndarray:
    return np.ones(num_cats) / num_cats  # uniform prior

agent = BayesianAgent(
    tool_configs=tools,
    categories=categories,
    category_infer_fn=infer_category,
)
```

Categories, tools, scoring rules, and the category inference function are all
injected — the agent itself is domain-agnostic.

## The Benchmark

A head-to-head comparison between a Bayesian decision-theoretic agent and LangChain
ReAct agents on a multi-tool question-answering task.

Both agents have access to the same four tools with heterogeneous, category-dependent
reliability. The Bayesian agent learns tool reliability from experience and uses
value-of-information calculations to decide when to query, when to cross-verify,
and when to abstain. The LangChain agent lets the LLM decide.

### Running experiments

```bash
# Run tests
uv run pytest -v

# Run all experiments (stationary + drift + ablation)
uv run python -m experiments.run_full_comparison

# Or run individually
uv run python -m experiments.run_stationary --seeds 20
uv run python -m experiments.run_drift --seeds 20
uv run python -m experiments.run_ablation --seeds 20
```

LangChain agents require a local Ollama instance with `llama3.1` (default).
Set `CREDENCE_LLM_PROVIDER=openai` or `CREDENCE_LLM_PROVIDER=anthropic` for API-based models.

### Key Results

| Agent | Score | Accuracy | Tools/Q |
|---|---|---|---|
| oracle | +188.2 | 70.6% | 1.08 |
| **bayesian** | **+112.6** | **59.6%** | **0.99** |
| langchain_react | -7.4 | 64.0% | 3.22 |
| langchain_enhanced | -68.2 | 66.0% | 3.94 |

The Bayesian agent outscores LangChain by 120 points despite lower accuracy —
it queries ~1 tool per question instead of ~3.2, and strategically abstains on
low-confidence questions. Enhanced prompting makes LangChain *worse* by triggering
more tool calls without proportional accuracy gains.

See [`results/RESULTS.md`](results/RESULTS.md) for full results across all three
experiments (stationary, drift, ablation).

## Project Structure

```
credence/
├── credence/                    # Facade package — public API re-exports
│   └── __init__.py
├── src/
│   ├── inference/               # Domain-agnostic Bayesian inference layer
│   │   ├── beta_posterior.py    # Beta-Bernoulli reliability tracking
│   │   ├── voi.py               # Value of information, ScoringRule, ToolConfig
│   │   └── decision.py          # EU-based decision logic
│   ├── agents/
│   │   ├── bayesian_agent.py    # Domain-agnostic Bayesian agent
│   │   ├── common.py            # Shared agent interface
│   │   ├── baselines.py         # Random, all-tools, oracle agents
│   │   ├── langchain_agent.py   # Standard LangChain ReAct agent
│   │   └── langchain_enhanced.py # LangChain with best-effort prompting
│   ├── environment/             # Benchmark-specific: simulated tools and questions
│   │   ├── benchmark.py         # The benchmark harness
│   │   ├── tools.py             # Simulated tools with known reliability
│   │   ├── questions.py         # Question bank with ground truth
│   │   └── categories.py        # Category definitions, make_keyword_category_infer_fn
│   ├── analysis/
│   │   ├── metrics.py           # Score, calibration, cost metrics
│   │   └── visualisation.py     # Plots and dashboards
│   └── utils/
│       └── logging.py           # Structured logging for analysis
├── tests/
├── experiments/
│   ├── run_stationary.py        # Main experiment: stationary reliabilities
│   ├── run_drift.py             # Extension: tool reliability drift
│   └── run_ablation.py          # Ablation studies
├── results/                     # Generated plots and data
└── pyproject.toml               # Core deps vs [benchmark] vs [dev] extras
```

## Design Principles

1. **Everything is expected utility maximisation** — no heuristics
2. **No hacks** — if it doesn't work, fix the model
3. **LLM outputs are data** — with quantified uncertainty
4. **The benchmark must be fair** — LangChain gets every advantage
5. **Be honest about parameters** — every number is justified

See `CLAUDE.md` for architecture and development guidelines.
See `SPEC.md` for the full mathematical specification.
