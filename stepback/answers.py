"""Answer extraction, normalisation and majority voting."""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


def extract_boxed_answer(text: str) -> str:
    idx = (text or "").rfind("\\boxed")
    if idx < 0:
        return ""
    depth = 0
    start = None
    for i in range(idx, len(text)):
        if text[i] == "{":
            if depth == 0:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start + 1 : i].strip()
    return ""


def normalize_answer(answer: str) -> str:
    return str(answer).strip().strip("$").replace(",", "").rstrip(".")


def extract_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed)
    match = re.search(r"Final\s+Answer\s*:\s*(.+?)(?:\n|$)", text or "", re.I)
    if match:
        return normalize_answer(match.group(1))
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if nums:
        return normalize_answer(nums[-1])
    return ""


# LaTeX normalisation adapted from lm-evaluation-harness (Hendrycks MATH).

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr or substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + substr[2:]
            else:
                new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    if string == f"{a}/{b}":
        return f"\\frac{{{a}}}{{{b}}}"
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    """Normalise a LaTeX answer string for comparison."""
    for old, new in [
        ("\n", ""), ("\\!", ""), ("\\\\", "\\"), ("tfrac", "frac"),
        ("dfrac", "frac"), ("\\left", ""), ("\\right", ""),
        ("^{\\circ}", ""), ("^\\circ", ""), ("\\$", ""),
    ]:
        string = string.replace(old, new)
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            string = splits[0]
    string = string.replace("\\%", "").replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string).replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def answers_match(pred: str, gold: str) -> bool:
    """Return True when pred and gold represent the same MATH/GSM8K answer."""
    pred = normalize_answer(pred)
    gold = normalize_answer(gold)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    try:
        if strip_string(pred) == strip_string(gold):
            return True
    except Exception:
        pass
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def majority_vote(answers: Sequence[str]) -> str:
    valid = [a for a in answers if a]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]
