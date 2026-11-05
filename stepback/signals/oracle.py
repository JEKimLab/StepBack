"""First-error oracle: an OpenAI model that sees the gold answer.

Analysis only, not a deployable method.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Sequence, Tuple


def build_first_error_prompt(
    question: str,
    gold_answer: str,
    steps: Sequence[str],
) -> str:
    numbered_steps = "\n".join(
        f"[{i + 1}] {step}" for i, step in enumerate(steps)
    )
    return f"""You are a rigorous math reasoning verifier.

Question:
{question}

Correct final answer: {gold_answer}

The model produced the following step-by-step solution (which arrives at a WRONG final answer):

{numbered_steps}

Judgment criteria (follow strictly):
- Focus ONLY on mathematical operations. For each step, check whether the
  arithmetic (addition, subtraction, multiplication, percentages, etc.) is
  performed correctly on the numbers it uses.
- A step is CORRECT if: (a) its arithmetic is right, AND (b) the numbers it
  uses come from the problem statement or from a preceding step's result.
  It does not matter what the step calls or labels the result.
- A step is INCORRECT only if: (a) it computes an arithmetic operation wrong
  (e.g. 2+3=6), OR (b) it applies a formula/percentage to a number that
  neither the problem statement nor any preceding step provided as the
  appropriate base for that operation (e.g. the problem says "150% of the
  house price" but the step computes 150% of a different quantity).
- Steps that merely restate the problem, outline a plan, set up notation,
  or compute simple sums of previously stated numbers are correct even if
  the label/name they give the result is imprecise.
- Do NOT mark a step wrong for imprecise wording or naming.
- Do NOT mark a step wrong just because it propagates an earlier mistake.

Task:
1. Walk through every step. For each, decide correct/incorrect by the criteria above.
2. If you find an incorrect step, report the FIRST one and explain in one sentence
   what error it introduces.
3. If NO step contains a clear arithmetic or formula error by the criteria above,
   set first_error_step to -1.

Output STRICT JSON (no markdown fences):
{{
  "first_error_step": <int, 1-indexed; or -1 if no clear error found>,
  "total_steps": {len(steps)},
  "reason": "<one sentence explaining the error, or 'no clear error found'>"
}}
"""


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def call_gpt_first_error(
    question: str,
    gold_answer: str,
    steps: Sequence[str],
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 500,
    retries: int = 5,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Ask an OpenAI model for the first erroneous step (1-indexed).

    The client reads OPENAI_API_KEY and, if set, OPENAI_BASE_URL. Failed calls
    are retried with exponential backoff; if all fail, tau is None and the
    caller applies the fallback.
    """
    from openai import OpenAI

    client = OpenAI()
    prompt = build_first_error_prompt(question, gold_answer, steps)

    raw = ""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if hasattr(client, "responses"):
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                raw = (response.output_text or "").strip()
            else:  # older openai SDKs without the Responses API
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                raw = (response.choices[0].message.content or "").strip()
            if raw:
                break
        except Exception as err:  # network / rate-limit / server errors
            last_err = err
        time.sleep(2 ** attempt)

    if not raw:
        return None, {"parsed": None, "raw": "", "error": repr(last_err)}

    parsed = parse_json_object(raw) or {}
    tau = parsed.get("first_error_step")
    if isinstance(tau, int) and not isinstance(tau, bool):
        return tau, {"parsed": parsed, "raw": raw}
    return None, {"parsed": parsed, "raw": raw}
