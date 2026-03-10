# CLAUDE.md — Credence

## What This Project Is

**Credence** is a reusable Bayesian decision-theoretic agent library and an empirical
benchmark demonstrating that decision theory outperforms prompt engineering for
tool-using agents.

The library provides domain-agnostic Bayesian agents: categories, tools, scoring rules,
and category inference are all injected — the agent and inference layer have no hardcoded
domain knowledge. The benchmark instantiates this library for a multi-tool QA task and
compares against LangChain ReAct agents.

---

## Design Principles (NON-NEGOTIABLE)

### 1. Everything is Expected Utility Maximisation

Every decision the Bayesian agent makes — which tool to query, whether to cross-verify,
whether to submit or abstain — is:

    a* = argmax_a E[U | a, beliefs]

No special cases. No "if confused, query all tools." Just EU.

### 2. No Hacks

If the agent behaves badly, the solution is better modelling, not bolted-on fixes.

**Forbidden:**
- Exploration bonuses
- Hardcoded tool routing ("always use calculator for maths")
- Retry logic that isn't derived from VOI
- "If stuck, try all tools"
- Any rule that isn't derived from EU maximisation

**Instead:** Figure out why EU maximisation gives the wrong answer. Fix the model.

### 3. LLM Outputs Are Data

When the Bayesian agent uses an LLM (e.g. to classify question category), the output
is an observation with uncertainty, not a command. The agent maintains beliefs about
LLM reliability per category and updates from experience.

### 4. The Benchmark Must Be Fair

- Use LangChain idiomatically — recommended patterns, good prompts, good model
- Don't sandbag the LangChain agent
- Include enhanced LangChain variants (self-reflection, reliability-aware prompts)
- Report all results honestly, including cases where LangChain does well

### 5. Be Honest About Parameters

Every parameter in the Bayesian agent must be justified:
- **Tool cost:** Reflects real API cost asymmetry
- **Priors:** Beta(1,1) uniform — we genuinely don't know tool reliability
- **Forgetting rate λ:** Only used in drift variant; justified by non-stationarity prior

---

## Architecture Overview

Everything lives under the `credence` package (PyPI: `credence-agents`).

```
credence/
├── credence/                    # The package
│   ├── __init__.py              # Public API re-exports
│   ├── py.typed
│   ├── inference/               # Domain-agnostic inference layer
│   │   ├── beta_posterior.py    # Beta-Bernoulli reliability tracking
│   │   ├── voi.py               # Value of information, ScoringRule, ToolConfig
│   │   └── decision.py          # EU-based decision logic
│   ├── agents/
│   │   ├── bayesian_agent.py    # Domain-agnostic Bayesian agent
│   │   ├── common.py            # Shared agent interface (AgentResult, DecisionStep)
│   │   ├── baselines.py         # Random, all-tools, oracle agents
│   │   ├── langchain_agent.py   # Standard LangChain ReAct agent
│   │   └── langchain_enhanced.py # LangChain with best-effort prompting
│   ├── environment/             # Benchmark-specific
│   │   ├── benchmark.py         # The benchmark harness
│   │   ├── tools.py             # Simulated tools with known reliability
│   │   ├── questions.py         # Question bank with ground truth
│   │   └── categories.py        # CATEGORIES tuple, make_keyword_category_infer_fn()
│   └── analysis/
│       ├── metrics.py           # Score, calibration, cost metrics
│       └── visualisation.py     # Plots and dashboards
├── tests/
├── experiments/
│   ├── run_stationary.py        # Main experiment: stationary reliabilities
│   ├── run_drift.py             # Extension: tool reliability drift
│   └── run_ablation.py          # Ablation studies
├── results/                     # Generated plots and data
└── pyproject.toml               # Core deps vs [benchmark] vs [dev] extras
```

### Decoupled Architecture

The `BayesianAgent` constructor accepts injected domain knowledge:

```python
BayesianAgent(
    tool_configs: list[ToolConfig],        # Tool names and costs
    categories: tuple[str, ...] | None,    # Domain categories (injected)
    category_infer_fn: Callable | None,    # Question text → category prior (injected)
    forgetting: float = 1.0,               # Exponential forgetting rate
)
```

The inference layer (`beta_posterior.py`, `voi.py`, `decision.py`) works with integer
indices for tools and categories — no domain strings. `ScoringRule` and `ToolConfig`
are the only configuration types needed to parameterise the entire decision loop.

`make_keyword_category_infer_fn()` in `categories.py` is a convenience for the benchmark;
custom domains provide their own `category_infer_fn`.

---

## Common Mistakes to Avoid

### WRONG: Hardcoding domain knowledge in the agent
```python
# BAD — domain-specific routing inside the agent
if question.looks_numerical():
    tool = "calculator"
```

### RIGHT: Inject domain knowledge, decide via EU
```python
# GOOD — domain categories and inference injected at construction
agent = BayesianAgent(
    tool_configs=tool_configs,
    categories=("factual", "numerical", "recent", "misconception", "reasoning"),
    category_infer_fn=my_category_infer_fn,
)
# Tool selection happens via EU maximisation inside select_action()
```

### WRONG: Treating LLM classification as certain
```python
# BAD — trusting LLM category without uncertainty
category = llm.classify(question)  # treated as ground truth
```

### RIGHT: Category as uncertain observation
```python
# GOOD — LLM classification updates beliefs, doesn't determine them
category_distribution = update_category_beliefs(
    prior=uniform_over_categories,
    observation=llm.classify(question),
    llm_reliability=self.category_reliability
)
```

### WRONG: Arbitrary verification threshold
```python
# BAD — magic number
if confidence < 0.7:
    cross_verify()
```

### RIGHT: VOI-driven verification
```python
# GOOD — cross-verify when EU says to
voi_verify = expected_eu_after_query(tool_b) - current_eu
if voi_verify > cost_tool_b:
    cross_verify(tool_b)
```

### WRONG: Making the LangChain agent deliberately bad
```python
# BAD — strawman
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
# with a terrible prompt
```

### RIGHT: Give LangChain every advantage
```python
# GOOD — use the best available patterns
agent = initialize_agent(
    tools, llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    # with a carefully crafted system prompt that explains
    # the scoring, tool characteristics, and the value of abstention
)
```

---

## Testing Strategy

1. **Unit test the inference layer first** — Beta posteriors, VOI calculations, decision logic
2. **Test with a trivial environment** — 2 tools, 5 questions, known reliabilities
3. **Verify against oracle** — with true reliabilities given, Bayesian agent should match oracle
4. **Then integrate** — full benchmark with all agents

---

## Key Invariants

- VOI is always non-negative (information can't hurt in expectation)
- Beta posteriors are always valid (alpha > 0, beta > 0)
- EU(submit) + EU(abstain) + EU(query) must be computed over the SAME belief state
- Tool costs are subtracted from EU(query), not from score retrospectively
- Ground truth for reliability updates: ONLY from final answer feedback
- The Bayesian agent never "peeks" at true tool reliability
- The agent and inference layer contain NO domain-specific constants
