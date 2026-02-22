# Bayesian Tool Agent Benchmark

**Thesis:** Current LLM "agents" are expensive flowcharts. Decision theory does it better.

## The Experiment

A head-to-head comparison between a Bayesian decision-theoretic agent and LangChain 
ReAct agents on a multi-tool question-answering task.

Both agents have access to the same four tools with heterogeneous, category-dependent 
reliability. The Bayesian agent learns tool reliability from experience and uses 
value-of-information calculations to decide when to query, when to cross-verify, 
and when to abstain. The LangChain agent lets the LLM decide.

## Quick Start

```bash
# Install dependencies
uv sync

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

## Key Results

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

See `CLAUDE.md` for architecture and design principles.
See `SPEC.md` for the full mathematical specification.

## Design Principles

1. **Everything is expected utility maximisation** — no heuristics
2. **No hacks** — if it doesn't work, fix the model
3. **LLM outputs are data** — with quantified uncertainty
4. **The benchmark must be fair** — LangChain gets every advantage
5. **Be honest about parameters** — every number is justified
