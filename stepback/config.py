"""Run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"

ROLLBACK_SIGNALS = ("nll", "prm", "last", "fixed", "gpt")
FALLBACKS = ("last", "start")


@dataclass
class StepBackConfig:
    # Policy model.
    model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Rollback.
    rollback_signal: str = "nll"
    nll_threshold: float = 0.1
    prm_threshold: float = 0.05
    prm_model: str = PRM_MODEL_ID
    prm_device: str = "cuda:0"
    fallback: str = "last"
    fixed_rollback_step: int = -1
    gpt_model: str = "gpt-5.1"
    gpt_temperature: float = 0.0

    # Answer budget: n_drafts * n_suffixes candidate answers.
    n_drafts: int = 16
    n_suffixes: int = 2

    # Sampling.
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: int = -1

    # vLLM engine.
    max_model_len: int = 4096
    gpu_memory_utilization: Optional[float] = None
    max_num_batched_tokens: int = 512

    def validate(self) -> None:
        if self.rollback_signal not in ROLLBACK_SIGNALS:
            raise ValueError(f"Unknown rollback signal: {self.rollback_signal}")
        if self.fallback not in FALLBACKS:
            raise ValueError(f"Unknown fallback: {self.fallback}")
        if self.rollback_signal == "fixed" and self.fixed_rollback_step < 0:
            raise ValueError("--fixed-rollback-step is required for fixed mode")
        if self.n_drafts < 1 or self.n_suffixes < 1:
            raise ValueError("n_drafts and n_suffixes must be >= 1")

    def resolved_gpu_memory(self) -> float:
        """vLLM memory fraction; leave room for the PRM when it shares the GPU."""
        if self.gpu_memory_utilization is not None:
            return self.gpu_memory_utilization
        return 0.45 if self.rollback_signal == "prm" else 0.9
