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
pip install -e .

# Run the stationary benchmark
python experiments/run_stationary.py

# Run with drift
python experiments/run_drift.py

# Run ablation studies
python experiments/run_ablation.py
```

## Key Results

(To be populated after running experiments)

## Project Structure

See `CLAUDE.md` for architecture and design principles.
See `SPEC.md` for the full mathematical specification.

## Design Principles

1. **Everything is expected utility maximisation** — no heuristics
2. **No hacks** — if it doesn't work, fix the model
3. **LLM outputs are data** — with quantified uncertainty
4. **The benchmark must be fair** — LangChain gets every advantage
5. **Be honest about parameters** — every number is justified
