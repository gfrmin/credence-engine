"""Domain-specific category definitions for the Q&A benchmark."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

CATEGORIES = ("factual", "numerical", "recent_events", "misconceptions", "reasoning")
NUM_CATEGORIES = len(CATEGORIES)


def make_keyword_category_infer_fn(
    categories: tuple[str, ...] = CATEGORIES,
) -> Callable[[str], NDArray]:
    """Returns the benchmark's keyword-based category prior function.

    The returned function maps question text to a probability distribution
    over categories, using regex patterns for numerical, recent_events,
    misconceptions, and reasoning. Unmatched questions get a mild boost
    toward 'factual'.
    """
    numerical_pattern = re.compile(
        r"(?:\bcalcul|\bcomput|\bhow many\b|\bhow much\b|\bwhat is \d|"
        r"\bsquare root\b|\bsum of\b|\barea\b|\bradius\b|"
        r"\binvest|\btip on\b|\d+\s*[\+\-\*\/\%\^]\s*\d+|\d+%)",
        re.IGNORECASE,
    )
    recent_pattern = re.compile(
        r"\b(202[0-9]|recent|latest|current|this year|last year|"
        r"who won the 20|hosted the 20|released .* in 20)\b",
        re.IGNORECASE,
    )
    misconception_pattern = re.compile(
        r"(?:\btrue or false\b|\bcommon belief\b|\bmyth\b|"
        r"\bdo .* really\b|\bis it true\b|\bdoes .* have a\b|"
        r"\bvisible from space\b|percent of the brain|"
        r"\bmemory span\b|\bmother reject|\bonly use\b)",
        re.IGNORECASE,
    )
    reasoning_pattern = re.compile(
        r"\b(if .* then|therefore|conclude|logic|implies|"
        r"can we conclude|probability|minimum number|"
        r"how long does it take .* machines|"
        r"all but|overtake|missing dollar|"
        r"counterfeit)\b",
        re.IGNORECASE,
    )

    n = len(categories)
    cat_index = {name: i for i, name in enumerate(categories)}

    def infer(question_text: str) -> NDArray:
        weights = np.ones(n, dtype=np.float64)

        if "numerical" in cat_index and numerical_pattern.search(question_text):
            weights[cat_index["numerical"]] += 9.0
        if "recent_events" in cat_index and recent_pattern.search(question_text):
            weights[cat_index["recent_events"]] += 9.0
        if "misconceptions" in cat_index and misconception_pattern.search(question_text):
            weights[cat_index["misconceptions"]] += 9.0
        if "reasoning" in cat_index and reasoning_pattern.search(question_text):
            weights[cat_index["reasoning"]] += 9.0

        # If nothing matched strongly, factual gets a mild boost (most common category)
        if weights.max() == 1.0 and "factual" in cat_index:
            weights[cat_index["factual"]] += 1.0

        return weights / weights.sum()

    return infer
