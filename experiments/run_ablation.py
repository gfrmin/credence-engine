"""Experiment 3: Ablation studies (SPEC §7.3).

Six Bayesian agent variants with components disabled to measure contribution.

Usage:
    uv run python -m experiments.run_ablation --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from src.agents.bayesian_agent import BayesianAgent, infer_category_prior
from src.analysis.metrics import accuracy, total_score, tool_calls_per_question
from src.analysis.visualisation import score_comparison_bar, tool_calls_comparison
from src.environment.benchmark import BenchmarkResult, run_benchmark
from src.environment.questions import get_questions
from src.environment.tools import make_spec_tools, tool_config_for
from src.environment.categories import NUM_CATEGORIES
from src.inference.decision import Action, ActionType


RESULTS_DIR = Path("results")


# --- Ablation variants ---

class NoVOIAgent(BayesianAgent):
    """Always queries cheapest applicable tool instead of using VOI."""

    def __init__(self, tool_configs, name="no_voi"):
        super().__init__(tool_configs=tool_configs, name=name)

    def choose_action(self) -> Action:
        assert self._state is not None
        self._step += 1

        # If no tools queried yet, pick cheapest unused
        unused = [i for i in range(self.num_tools) if i not in self._state.used_tools]
        if unused and not self._state.tool_responses:
            cheapest = min(unused, key=lambda i: self.tool_configs[i].cost)
            action = Action(ActionType.QUERY, tool_idx=cheapest)
        else:
            # After one tool, defer to normal EU-based submit/abstain
            from src.inference.voi import eu_submit, eu_abstain
            eu_s = eu_submit(self._state.answer_posterior)
            eu_a = eu_abstain()
            if eu_s >= eu_a:
                best = int(np.argmax(self._state.answer_posterior))
                action = Action(ActionType.SUBMIT, answer_idx=best)
            else:
                action = Action(ActionType.ABSTAIN)

        from src.agents.common import DecisionStep
        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=str(action),
        ))
        return action


class NoCategoryAgent(BayesianAgent):
    """Treats all categories as identical (uniform category prior, never updated)."""

    def __init__(self, tool_configs, name="no_category"):
        super().__init__(tool_configs=tool_configs, name=name)

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        # Force uniform category prior regardless of question text
        from src.inference.decision import initial_question_state
        uniform = np.full(NUM_CATEGORIES, 1.0 / NUM_CATEGORIES)
        self._state = initial_question_state(uniform)
        self._trace = []
        self._step = 0
        self._question_text = ""


class NoAbstentionAgent(BayesianAgent):
    """Must always submit — abstention disabled."""

    def __init__(self, tool_configs, name="no_abstention"):
        super().__init__(tool_configs=tool_configs, name=name)

    def choose_action(self) -> Action:
        action = super().choose_action()
        if action.action_type == ActionType.ABSTAIN:
            best = int(np.argmax(self._state.answer_posterior))
            return Action(ActionType.SUBMIT, answer_idx=best)
        return action


class FixedReliabilityAgent(BayesianAgent):
    """No learning — uses prior reliability throughout."""

    def __init__(self, tool_configs, name="fixed_reliability"):
        super().__init__(tool_configs=tool_configs, name=name)

    def on_question_end(self, was_correct):
        # Skip reliability update
        pass


class SingleToolAgent(BayesianAgent):
    """Only queries one tool per question — no cross-verification."""

    def __init__(self, tool_configs, name="no_crossverify"):
        super().__init__(tool_configs=tool_configs, name=name)
        self._queried_this_q = False

    def on_question_start(self, question_id, candidates, num_tools, question_text=""):
        super().on_question_start(question_id, candidates, num_tools, question_text=question_text)
        self._queried_this_q = False

    def choose_action(self) -> Action:
        action = super().choose_action()
        if action.action_type == ActionType.QUERY:
            if self._queried_this_q:
                # Force submit/abstain instead of second query
                from src.inference.voi import eu_submit, eu_abstain
                eu_s = eu_submit(self._state.answer_posterior)
                eu_a = eu_abstain()
                if eu_s >= eu_a:
                    best = int(np.argmax(self._state.answer_posterior))
                    return Action(ActionType.SUBMIT, answer_idx=best)
                return Action(ActionType.ABSTAIN)
            self._queried_this_q = True
        return action


def make_ablation_agents(tool_configs):
    """Create all ablation variants."""
    return [
        ("full_agent", lambda: BayesianAgent(tool_configs=tool_configs, name="full_agent")),
        ("no_voi", lambda: NoVOIAgent(tool_configs=tool_configs)),
        ("no_category", lambda: NoCategoryAgent(tool_configs=tool_configs)),
        ("no_abstention", lambda: NoAbstentionAgent(tool_configs=tool_configs)),
        ("fixed_reliability", lambda: FixedReliabilityAgent(tool_configs=tool_configs)),
        ("no_crossverify", lambda: SingleToolAgent(tool_configs=tool_configs)),
    ]


def run_ablation(n_seeds: int = 20) -> dict[str, list[BenchmarkResult]]:
    """Run all ablation variants."""
    spec_tools = make_spec_tools()
    tool_configs = [tool_config_for(t) for t in spec_tools]
    agents = make_ablation_agents(tool_configs)

    results: dict[str, list[BenchmarkResult]] = {}

    for seed in range(n_seeds):
        questions = get_questions(seed=seed)
        for name, factory in agents:
            agent = factory()
            result = run_benchmark(agent, spec_tools, questions, seed=seed)
            if name not in results:
                results[name] = []
            results[name].append(result)

        print(f"  Seed {seed} done", file=sys.stderr)

    return results


def ablation_table(results: dict[str, list[BenchmarkResult]]) -> str:
    """Format ablation comparison table."""
    header = f"{'Variant':<20} {'Score':>12} {'Accuracy':>10} {'Tools/Q':>10}"
    lines = [header, "-" * len(header)]

    full_score = np.mean([total_score(r) for r in results.get("full_agent", [])])

    for name, runs in results.items():
        scores = [total_score(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        tools = [tool_calls_per_question(r) for r in runs]
        delta = np.mean(scores) - full_score

        lines.append(
            f"{name:<20} "
            f"{np.mean(scores):>7.1f}±{np.std(scores):.1f} "
            f"{np.mean(accs):>7.3f}±{np.std(accs):.3f} "
            f"{np.mean(tools):>7.2f}±{np.std(tools):.2f}"
            + (f"  ({delta:+.1f})" if name != "full_agent" else "")
        )

    return "\n".join(lines)


def save_results(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Save ablation results as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for agent_name, runs in results.items():
        data[agent_name] = [
            {
                "seed": r.seed,
                "total_score": r.total_score,
                "total_reward": r.total_reward,
                "total_tool_cost": r.total_tool_cost,
                "records": [
                    {
                        "question_id": rec.question_id,
                        "category": rec.category,
                        "action_type": rec.action_type,
                        "was_correct": rec.was_correct,
                        "reward": rec.reward,
                        "tool_cost": rec.tool_cost,
                        "tools_queried": list(rec.tools_queried),
                    }
                    for rec in r.records
                ],
            }
            for r in runs
        ]

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(data, f, indent=2)


def save_plots(results: dict[str, list[BenchmarkResult]], out_dir: Path) -> None:
    """Generate and save ablation plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = score_comparison_bar(results)
    fig.savefig(out_dir / "ablation_comparison.png")
    plt.close(fig)

    fig = tool_calls_comparison(results)
    fig.savefig(out_dir / "ablation_tool_calls.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    args = parser.parse_args()

    print(f"Running ablation with {args.seeds} seeds...", file=sys.stderr)
    results = run_ablation(n_seeds=args.seeds)

    out_dir = RESULTS_DIR / "ablation"
    save_results(results, out_dir)
    save_plots(results, out_dir)

    print("\n" + ablation_table(results))
    print(f"\nResults saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
