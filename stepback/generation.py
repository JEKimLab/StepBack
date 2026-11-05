"""vLLM engine and sampling."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def load_vllm(
    model_id: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int = 512,
) -> Any:
    from vllm import LLM

    # Chunked prefill bounds the (tokens x vocab) logprob tensors built when
    # scoring step NLL, so long drafts fit in the memory vLLM leaves free.
    return LLM(
        model=model_id,
        trust_remote_code=True,
        dtype="half",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=True,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def sample_texts(
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Sequence[str],
    seed: int,
) -> Tuple[List[List[str]], int]:
    """Sample n completions per prompt; return (texts, generated tokens).

    With seed >= 0, prompt i uses seed + i.
    """
    from vllm import SamplingParams

    sampling_params = [
        SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=list(stop),
            seed=(seed + i) if seed >= 0 else None,
        )
        for i in range(len(prompts))
    ]
    # Prompts already contain special tokens; avoid a second BOS.
    token_prompts = [
        {"prompt_token_ids": tokenizer.encode(p, add_special_tokens=False)}
        for p in prompts
    ]
    outputs = llm.generate(token_prompts, sampling_params=sampling_params)
    texts = [[s.text for s in out.outputs] for out in outputs]
    n_tokens = sum(len(s.token_ids) for out in outputs for s in out.outputs)
    return texts, n_tokens
