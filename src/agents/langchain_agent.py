"""Standard LangChain ReAct agent for the benchmark.

Uses the system prompt from SPEC §4.1 verbatim. Wraps SimulatedTools as
LangChain Tools. Supports both OpenAI and Anthropic via CREDENCE_LLM_PROVIDER.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from src.agents.common import AgentResult, DecisionStep
from src.inference.decision import Action, ActionType

# Tool descriptions matching SPEC §4.1
TOOL_DESCRIPTIONS = (
    "Fast web search. Good for factual questions. Cost: 1 point.",
    "Curated database. Very reliable when it has results, "
    "but returns 'no result' for questions outside its coverage. Cost: 3 points.",
    "Perfect for numerical computation. Returns 'not applicable' "
    "for non-numerical questions. Cost: 1 point.",
    "Ask your own knowledge directly. Moderately reliable. Cost: 2 points.",
)

TOOL_NAMES = ("quick_search", "knowledge_base", "calculator", "llm_direct")

# SPEC §4.1 system prompt — verbatim
SYSTEM_PROMPT = """\
You are a question-answering agent with access to four tools:
- quick_search: Fast web search. Good for factual questions. Cost: 1 point.
- knowledge_base: Curated database. Very reliable when it has results, \
but returns "no result" for questions outside its coverage. Cost: 3 points.
- calculator: Perfect for numerical computation. Returns "not applicable" \
for non-numerical questions. Cost: 1 point.
- llm_direct: Ask your own knowledge directly. Moderately reliable. Cost: 2 points.

SCORING:
- Correct answer: +10 points
- Wrong answer: -5 points
- "I don't know": 0 points
- Each tool use costs points as listed above

Your goal is to MAXIMISE total score across all questions.
Be selective about which tools you use — every query costs points.
If you're not confident, it's better to say "I don't know" (0 points) \
than guess wrong (-5 points).

For each question, select from the 4 provided candidate answers, or abstain.
"""


def _get_llm():
    """Create LLM based on CREDENCE_LLM_PROVIDER env var."""
    provider = os.environ.get("CREDENCE_LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0.0)


class LangChainAgent:
    """Standard LangChain ReAct agent using SPEC §4.1 prompt."""

    def __init__(self, name: str = "langchain_react"):
        self.name = name
        self._candidates: tuple[str, ...] = ()
        self._num_tools: int = 0
        self._tools_queried: list[int] = []
        self._tool_responses: dict[int, int | None] = {}
        self._trace: list[DecisionStep] = []
        self._step: int = 0
        self._question_id: str = ""
        self._pending_action: Action | None = None

    def on_question_start(
        self, question_id: str, candidates: tuple[str, ...], num_tools: int,
    ) -> None:
        self._question_id = question_id
        self._candidates = candidates
        self._num_tools = num_tools
        self._tools_queried = []
        self._tool_responses = {}
        self._trace = []
        self._step = 0
        self._pending_action = None

    def choose_action(self) -> Action:
        self._step += 1

        if self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
            self._trace.append(DecisionStep(
                step=self._step, eu_submit=0.0, eu_abstain=0.0,
                eu_query={}, chosen_action=_action_str(action),
            ))
            return action

        # Build the LLM prompt
        action = self._ask_llm()
        self._trace.append(DecisionStep(
            step=self._step, eu_submit=0.0, eu_abstain=0.0,
            eu_query={}, chosen_action=_action_str(action),
        ))
        return action

    def _ask_llm(self) -> Action:
        """Query the LLM for the next action."""
        llm = _get_llm()

        # Build user message
        parts = [f"Question: {self._question_id}"]
        parts.append(f"Candidates: {list(self._candidates)}")

        if self._tool_responses:
            parts.append("\nTool results so far:")
            for t_idx, resp in self._tool_responses.items():
                tool_name = TOOL_NAMES[t_idx] if t_idx < len(TOOL_NAMES) else f"tool_{t_idx}"
                if resp is None:
                    parts.append(f"  {tool_name}: no result / not applicable")
                else:
                    parts.append(f"  {tool_name}: {self._candidates[resp]}")

        available = [i for i in range(self._num_tools) if i not in self._tool_responses]
        if available:
            tool_list = ", ".join(
                TOOL_NAMES[i] if i < len(TOOL_NAMES) else f"tool_{i}"
                for i in available
            )
            parts.append(f"\nAvailable tools (not yet queried): {tool_list}")

        parts.append(
            "\nRespond with EXACTLY one of:\n"
            "- QUERY <tool_name> (to query a tool)\n"
            "- SUBMIT <candidate_index> (0-3, to submit an answer)\n"
            "- ABSTAIN (if unsure)"
        )

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n".join(parts))]

        response = llm.invoke(messages)
        return self._parse_llm_response(response.content)

    def _parse_llm_response(self, content: str) -> Action:
        """Parse LLM text response into an Action."""
        text = content.strip().upper()

        if "ABSTAIN" in text:
            return Action(ActionType.ABSTAIN)

        if "SUBMIT" in text:
            for ch in text:
                if ch.isdigit():
                    idx = int(ch)
                    if 0 <= idx < len(self._candidates):
                        return Action(ActionType.SUBMIT, answer_idx=idx)
            return Action(ActionType.SUBMIT, answer_idx=0)

        if "QUERY" in text:
            text_lower = content.strip().lower()
            for i, name in enumerate(TOOL_NAMES):
                if name in text_lower and i not in self._tool_responses:
                    return Action(ActionType.QUERY, tool_idx=i)
            # Fallback: query first available
            for i in range(self._num_tools):
                if i not in self._tool_responses:
                    return Action(ActionType.QUERY, tool_idx=i)

        # Default: submit candidate 0
        return Action(ActionType.SUBMIT, answer_idx=0)

    def on_tool_response(self, tool_idx: int, response: int | None) -> None:
        self._tools_queried.append(tool_idx)
        self._tool_responses[tool_idx] = response

    def on_question_end(self, was_correct: bool | None) -> None:
        pass

    def get_belief_snapshot(self) -> dict | None:
        return {
            "tool_responses": dict(self._tool_responses),
            "tools_queried": list(self._tools_queried),
        }


def _action_str(action: Action) -> str:
    if action.action_type == ActionType.SUBMIT:
        return f"submit({action.answer_idx})"
    elif action.action_type == ActionType.ABSTAIN:
        return "abstain"
    else:
        return f"query({action.tool_idx})"
