"""Splitting a response into reasoning steps."""

from __future__ import annotations

from typing import List, Sequence, Tuple

STEP_SEPARATOR = "\n\n"


def split_steps(text: str) -> List[str]:
    """Steps are blank-line-separated blocks."""
    text = (text or "").strip()
    if not text:
        return []
    blocks = [part.strip() for part in text.split(STEP_SEPARATOR) if part.strip()]
    return blocks if blocks else [text]


def prefix_from_steps(steps: Sequence[str], rollback_step: int) -> str:
    """Text of the first `rollback_step` steps, ready for continuation."""
    keep = max(0, min(rollback_step, len(steps)))
    if keep == 0:
        return ""
    return STEP_SEPARATOR.join(steps[:keep]) + STEP_SEPARATOR


def step_char_bounds(response: str, steps: Sequence[str]) -> List[Tuple[int, int]]:
    """Locate each split step inside the raw response text."""
    bounds, pos = [], 0
    for step in steps:
        idx = response.find(step, pos)
        if idx < 0:
            idx = pos
        bounds.append((idx, idx + len(step)))
        pos = idx + len(step)
    return bounds
