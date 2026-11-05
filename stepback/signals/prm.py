"""PRM signal: step scores from Qwen2.5-Math-PRM-7B."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
STEP_TAG = "<extra_0>"


def load_prm(model_id: str, device: str) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer, DynamicCache

    # Qwen2.5-Math-PRM remote code calls an API removed in newer transformers.
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = (
            lambda self, *a, **kw: self.get_seq_length()
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer


def score_step_prm(
    prm: Tuple[Any, Any],
    question: str,
    steps_list: Sequence[Sequence[str]],
) -> List[List[float]]:
    """Per-step PRM probability of correctness for each draft."""
    import torch
    import torch.nn.functional as F

    model, tokenizer = prm
    sep_id = tokenizer.encode(STEP_TAG)[0]
    results = []
    for steps in steps_list:
        messages = [
            {"role": "system", "content": PRM_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": STEP_TAG.join(steps) + STEP_TAG},
        ]
        conv = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = torch.tensor(
            [tokenizer.encode(conv)], dtype=torch.long
        ).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=input_ids)[0]
        probs = F.softmax(logits, dim=-1)[0]
        mask = input_ids[0] == sep_id
        results.append(probs[mask][:, 1].float().cpu().tolist())
    return results
