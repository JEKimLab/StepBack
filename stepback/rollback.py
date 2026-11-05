"""Rollback rules.

`rollback_step` is the number of draft steps to keep:
    0 resamples from scratch; 2 keeps steps[0:2] and resamples from step 3.

If a signal finds no error, `fallback` decides: `last` keeps all but the last
step (resample only the final answer step), `start` resamples from scratch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from .config import StepBackConfig


def _fallback_step(n_steps: int, fallback: str) -> int:
    return max(n_steps - 1, 0) if fallback == "last" else 0


def first_prm_drop(
    step_scores: Sequence[float],
    threshold: float = 0.05,
    fallback: str = "last",
) -> Tuple[int, bool]:
    """Keep steps before the first step whose PRM score drops by > threshold.

    Returns (rollback_step, triggered).
    """
    n = len(step_scores)
    for i in range(1, n):
        if step_scores[i - 1] - step_scores[i] > threshold:
            return i, True
    return _fallback_step(n, fallback), False


def first_nll_jump(
    step_nlls: Sequence[float],
    threshold: float = 0.1,
    fallback: str = "last",
) -> Tuple[int, bool]:
    """Keep steps before the first step whose mean NLL rises by > threshold.

    Returns (rollback_step, triggered).
    """
    n = len(step_nlls)
    for i in range(1, n):
        if step_nlls[i] - step_nlls[i - 1] > threshold:
            return i, True
    return _fallback_step(n, fallback), False


def rollback_from_tau(tau: Optional[int], n_steps: int, fallback: str) -> int:
    """Map a 1-indexed first-error step to the number of steps to keep."""
    if tau is not None and tau >= 1:
        return max(0, min(tau - 1, n_steps - 1))
    return _fallback_step(n_steps, fallback)


def choose_rollback_step(
    config: StepBackConfig,
    question: str,
    steps: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    gold_answer: str = "",
) -> Tuple[int, Dict[str, Any]]:
    """Return (rollback_step, info) for one draft under the configured signal."""
    n_steps = len(steps)
    signal = config.rollback_signal
    if n_steps == 0:
        return 0, {"signal": signal, "reason": "empty draft"}

    if signal == "last":
        return max(n_steps - 1, 0), {"signal": "last"}

    if signal == "fixed":
        keep = max(0, min(config.fixed_rollback_step, n_steps))
        return keep, {"signal": "fixed"}

    if signal == "nll":
        keep, triggered = first_nll_jump(
            scores, config.nll_threshold, config.fallback
        )
        return keep, {
            "signal": "nll",
            "step_nll": [round(s, 6) for s in scores],
            "threshold": config.nll_threshold,
            "triggered": triggered,
            "fallback": config.fallback,
        }

    if signal == "prm":
        keep, triggered = first_prm_drop(
            scores, config.prm_threshold, config.fallback
        )
        return keep, {
            "signal": "prm",
            "step_prm": [round(s, 6) for s in scores],
            "threshold": config.prm_threshold,
            "triggered": triggered,
            "fallback": config.fallback,
        }

    from .signals.oracle import call_gpt_first_error

    tau, info = call_gpt_first_error(
        question=question,
        gold_answer=gold_answer,
        steps=steps,
        model=config.gpt_model,
        temperature=config.gpt_temperature,
    )
    keep = rollback_from_tau(tau, n_steps, config.fallback)
    info.update({"signal": "gpt", "first_error_step": tau, "fallback": config.fallback})
    return keep, info
