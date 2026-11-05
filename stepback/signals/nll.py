"""NLL signal: step-level NLL of the draft under the policy model (vLLM)."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ..steps import step_char_bounds


def compute_step_mean_nll(
    token_logprobs: Sequence[float],
    token_offsets: Sequence[int],
    bounds: Sequence[Tuple[int, int]],
) -> List[float]:
    """Mean negative log-likelihood of the tokens that start inside each step."""
    nlls = []
    ti = 0
    for start, end in bounds:
        while ti < len(token_offsets) and token_offsets[ti] < start:
            ti += 1
        values = []
        scan = ti
        while scan < len(token_offsets) and token_offsets[scan] < end:
            values.append(token_logprobs[scan])
            scan += 1
        nlls.append(-sum(values) / len(values) if values else 0.0)
    return nlls


def score_step_nlls(
    llm: Any,
    tokenizer: Any,
    base_prompt: str,
    drafts: Sequence[str],
    steps_list: Sequence[Sequence[str]],
) -> List[List[float]]:
    """Per-step mean NLL of each draft under the policy model."""
    from vllm import SamplingParams

    encodings = [
        tokenizer(
            base_prompt + draft,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        for draft in drafts
    ]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    # One draft per call: prompt logprobs materialise a (tokens x vocab)
    # tensor, which can exceed the memory vLLM leaves free when batched.
    outputs = [
        llm.generate(
            [{"prompt_token_ids": enc["input_ids"]}],
            sampling_params=params,
            use_tqdm=False,
        )[0]
        for enc in encodings
    ]

    resp_offset = len(base_prompt)
    results = []
    for draft, steps, enc, out in zip(drafts, steps_list, encodings, outputs):
        logprobs, offsets = [], []
        for pos, ((start, end), token_id) in enumerate(
            zip(enc["offset_mapping"], enc["input_ids"])
        ):
            if end <= resp_offset or pos == 0:
                continue
            entry = out.prompt_logprobs[pos]
            logprobs.append(float(entry[token_id].logprob))
            offsets.append(max(0, start - resp_offset))
        bounds = step_char_bounds(draft, steps)
        results.append(compute_step_mean_nll(logprobs, offsets, bounds))
    return results
