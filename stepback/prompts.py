"""Chat prompts and stop tokens for the policy model."""

from __future__ import annotations

from typing import List

MATH_USER_PROMPT = (
    "Solve the following math problem. Present the final answer "
    "in the format: Final Answer: \\boxed{{your_answer}}.\n"
    "Problem: {question}\n"
    "Answer:"
)
SYSTEM_PROMPT = "You are a helpful assistant."


def is_llama(model_id: str) -> bool:
    return "llama" in model_id.lower()


def build_prompt(model_id: str, question: str) -> str:
    """Chat prompt used in the paper (fixed system prompt for all models)."""
    user_msg = MATH_USER_PROMPT.format(question=question)
    if is_llama(model_id):
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def stop_tokens_for(model_id: str) -> List[str]:
    if is_llama(model_id):
        return ["<|eot_id|>", "<|end_of_text|>"]
    return ["<|im_end|>", "<|endoftext|>"]
