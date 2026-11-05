"""StepBack: draft -> localize -> roll back -> resample -> vote."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .answers import answers_match, extract_answer, majority_vote
from .config import StepBackConfig
from .generation import load_vllm, sample_texts
from .prompts import build_prompt, stop_tokens_for
from .rollback import choose_rollback_step
from .signals import load_prm, score_step_nlls, score_step_prm
from .steps import prefix_from_steps, split_steps


class StepBack:
    """Holds the policy model (and PRM, if used) across questions.

    >>> sb = StepBack(StepBackConfig(model="Qwen/Qwen2.5-3B-Instruct"))
    >>> result = sb.solve("What is 2 + 3?", gold_answer="5")
    """

    def __init__(self, config: Optional[StepBackConfig] = None):
        self.config = config or StepBackConfig()
        self.config.validate()
        c = self.config
        self.llm = load_vllm(
            c.model,
            c.max_model_len,
            c.resolved_gpu_memory(),
            c.max_num_batched_tokens,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.stop = stop_tokens_for(c.model)
        self._prm = None

    @property
    def prm(self):
        if self._prm is None:
            self._prm = load_prm(self.config.prm_model, self.config.prm_device)
        return self._prm

    def _sample(self, prompts: List[str], n: int, seed: int):
        c = self.config
        return sample_texts(
            self.llm,
            self.tokenizer,
            prompts,
            n=n,
            temperature=c.temperature,
            top_p=c.top_p,
            max_tokens=c.max_tokens,
            stop=self.stop,
            seed=seed,
        )

    def _score(self, question, base_prompt, drafts, steps_list):
        signal = self.config.rollback_signal
        if signal == "nll":
            return score_step_nlls(
                self.llm, self.tokenizer, base_prompt, drafts, steps_list
            )
        if signal == "prm":
            return score_step_prm(self.prm, question, steps_list)
        return [None] * len(drafts)

    def solve(self, question: str, gold_answer: str = "") -> Dict[str, Any]:
        c = self.config
        question = question.strip()
        if not question:
            raise ValueError("Empty question")
        if c.rollback_signal == "gpt" and not gold_answer:
            raise ValueError("--rollback-signal gpt is an oracle and needs --gold-answer")

        base_prompt = build_prompt(c.model, question)

        # 1. Drafts.
        draft_groups, draft_tokens = self._sample([base_prompt], c.n_drafts, c.seed)
        draft_texts = draft_groups[0]
        steps_list = [split_steps(t) for t in draft_texts]

        # 2. Step scores for the rollback signal.
        scores_list = self._score(question, base_prompt, draft_texts, steps_list)

        # 3. Rollback points and suffix prompts.
        draft_records: List[Dict[str, Any]] = []
        suffix_prompts: List[str] = []
        for idx, (text, steps, scores) in enumerate(
            zip(draft_texts, steps_list, scores_list)
        ):
            rollback_step, info = choose_rollback_step(
                c, question, steps, scores, gold_answer
            )
            prefix = prefix_from_steps(steps, rollback_step)
            suffix_prompts.append(base_prompt + prefix)
            draft_records.append({
                "draft_idx": idx,
                "draft_text": text,
                "draft_answer": extract_answer(text),
                "steps": steps,
                "rollback_step": rollback_step,
                "kept_prefix": prefix,
                "rollback_info": info,
                "suffixes": [],
            })

        # 4. Resample suffixes: each draft yields its own answer plus
        #    n_suffixes - 1 resampled answers.
        n_resample = max(c.n_suffixes - 1, 0)
        suffix_groups: List[List[str]] = [[] for _ in suffix_prompts]
        suffix_tokens = 0
        if n_resample > 0:
            # Offset so a suffix prompt equal to the base prompt (rollback to
            # step 0) does not replay the draft sample.
            suffix_seed = (c.seed + 1) if c.seed >= 0 else -1
            suffix_groups, suffix_tokens = self._sample(
                suffix_prompts, n_resample, suffix_seed
            )

        # 5. Vote.
        votes: List[str] = []
        suffix_answers: List[str] = []
        for record, suffix_texts in zip(draft_records, suffix_groups):
            votes.append(record["draft_answer"])
            for suffix_idx, suffix_text in enumerate(suffix_texts):
                answer = extract_answer(suffix_text)
                suffix_answers.append(answer)
                votes.append(answer)
                record["suffixes"].append({
                    "suffix_idx": suffix_idx,
                    "suffix_text": suffix_text,
                    "suffix_answer": answer,
                })

        majority_answer = majority_vote(votes)
        return {
            "question": question,
            "gold_answer": gold_answer,
            "model": c.model,
            "rollback_signal": c.rollback_signal,
            "n_drafts": c.n_drafts,
            "n_suffixes": c.n_suffixes,
            "generated_tokens": {
                "drafts": draft_tokens,
                "suffixes": suffix_tokens,
                "total": draft_tokens + suffix_tokens,
            },
            "drafts": draft_records,
            "suffix_answers": suffix_answers,
            "votes": votes,
            "majority_answer": majority_answer,
            "gold_correct": (
                answers_match(majority_answer, gold_answer) if gold_answer else None
            ),
        }


def run_stepback(
    question: str,
    config: Optional[StepBackConfig] = None,
    gold_answer: str = "",
) -> Dict[str, Any]:
    """One-shot helper: load the models and solve a single question."""
    return StepBack(config).solve(question, gold_answer)
