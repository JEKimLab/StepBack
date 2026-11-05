"""Command-line interface: python -m stepback --question "..." """

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Optional, Sequence

from .config import FALLBACKS, ROLLBACK_SIGNALS, StepBackConfig


def build_parser() -> argparse.ArgumentParser:
    d = StepBackConfig()
    p = argparse.ArgumentParser(
        prog="stepback",
        description="StepBack: draft -> localize -> roll back -> resample -> vote.",
    )

    io = p.add_argument_group("input / output")
    io.add_argument("--question", default="")
    io.add_argument("--question-file", default="",
                    help="Text file, or JSON with a 'question'/'problem' field.")
    io.add_argument("--gold-answer", default="",
                    help="Optional reference answer, used to report correctness.")
    io.add_argument("--out", default="outputs/stepback.json")

    rb = p.add_argument_group("rollback")
    rb.add_argument("--rollback-signal", choices=ROLLBACK_SIGNALS,
                    default=d.rollback_signal)
    rb.add_argument("--nll-threshold", type=float, default=d.nll_threshold)
    rb.add_argument("--prm-threshold", type=float, default=d.prm_threshold)
    rb.add_argument("--prm-model", default=d.prm_model)
    rb.add_argument("--prm-device", default=d.prm_device,
                    help="Device for the PRM (e.g. cuda:1 to keep it off the vLLM GPU).")
    rb.add_argument("--fallback", choices=FALLBACKS, default=d.fallback,
                    help="Rollback when the signal detects no error.")
    rb.add_argument("--fixed-rollback-step", type=int, default=d.fixed_rollback_step,
                    help="Number of steps to keep for --rollback-signal fixed.")
    rb.add_argument("--gpt-model", default=d.gpt_model)
    rb.add_argument("--gpt-temperature", type=float, default=d.gpt_temperature)

    gen = p.add_argument_group("generation")
    gen.add_argument("--model", default=d.model)
    gen.add_argument("--n-drafts", type=int, default=d.n_drafts)
    gen.add_argument("--n-suffixes", type=int, default=d.n_suffixes,
                     help="Candidate answers per draft (n_drafts * n_suffixes in total).")
    gen.add_argument("--temperature", type=float, default=d.temperature)
    gen.add_argument("--top-p", type=float, default=d.top_p)
    gen.add_argument("--max-tokens", type=int, default=d.max_tokens)
    gen.add_argument("--seed", type=int, default=d.seed)

    eng = p.add_argument_group("vLLM engine")
    eng.add_argument("--max-model-len", type=int, default=d.max_model_len)
    eng.add_argument("--gpu-memory-utilization", type=float,
                     default=d.gpu_memory_utilization,
                     help="vLLM memory fraction (default 0.9; 0.45 with "
                          "--rollback-signal prm when the PRM shares the GPU).")
    eng.add_argument("--max-num-batched-tokens", type=int,
                     default=d.max_num_batched_tokens,
                     help="vLLM chunked-prefill budget; lower it if NLL scoring "
                          "runs out of GPU memory.")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> StepBackConfig:
    return StepBackConfig(**{f.name: getattr(args, f.name) for f in fields(StepBackConfig)})


def read_question(args: argparse.Namespace) -> str:
    if args.question_file:
        path = Path(args.question_file)
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            return str(data.get("question", data.get("problem", ""))).strip()
        return text
    return args.question.strip()


def main(argv: Optional[Sequence[str]] = None) -> None:
    from .pipeline import run_stepback

    args = parse_args(argv)
    question = read_question(args)
    if not question:
        raise SystemExit("Provide --question or --question-file")
    result = run_stepback(question, config_from_args(args), args.gold_answer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    tokens = result["generated_tokens"]
    print(f"Saved: {out_path}")
    print(f"Majority answer: {result['majority_answer']} "
          f"({len(result['votes'])} votes, {tokens['total']} generated tokens)")
    if result["gold_correct"] is not None:
        print(f"Correct: {result['gold_correct']}")


if __name__ == "__main__":
    main()
