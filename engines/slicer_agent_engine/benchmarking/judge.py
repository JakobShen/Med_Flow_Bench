from __future__ import annotations

import json
import re
from typing import Dict, Optional, Sequence, Set


_GENERIC_ANSWER_RE = re.compile(r"\b(?:ANSWER|CHOICE|OPTION)\b\s*[:\-]?\s*([A-Z])\b", re.IGNORECASE)
_ANSWER_LINE_RE = re.compile(r"\b(?:FINAL\s+)?ANSWER\b", re.IGNORECASE)
_BOXED_CHOICE_RE = re.compile(r"\\boxed\{\s*([A-Z])\s*\}", re.IGNORECASE)
_NUMBERED_COMPACT_RE = re.compile(r"([1-9])([A-Z])", re.IGNORECASE)
_NUMBERED_LINE_RE = re.compile(r"^\s*(?:QUESTION\s*)?([1-9])\s*[:=\-\)]\s*([A-Z])\b", re.IGNORECASE)
_LEADING_CHOICE_RE = re.compile(r"^\s*([A-Z])(?=[\s\.\)\:\-]|$)", re.IGNORECASE)
_PAREN_CHOICE_RE = re.compile(r"\b([A-Z])\)(?=\s)", re.IGNORECASE)
_TRAILING_CHOICE_RE = re.compile(r"\b([A-Z])[\.\)]?\s*$", re.IGNORECASE)


def extract_choice_from_set(text: str, *, valid_choices: Sequence[str]) -> Optional[str]:
    """Extract a multiple-choice answer from free-form model text.

    Supported patterns:
    - JSON containing {"choice": "A"} or {"answer": "B"}
    - Lines like "ANSWER: C" or "Option D"
    - A final line that is just one valid choice token
    """

    if not text:
        return None

    allowed: Set[str] = {str(choice).strip().upper() for choice in valid_choices if str(choice).strip()}
    if not allowed:
        return None

    # 1) Try JSON (common for structured prompting).
    try:
        start = text.find("{")
        end = text.rfind("}")
        if 0 <= start < end:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                for k in ("choice", "answer", "label"):
                    v = obj.get(k)
                    if isinstance(v, str):
                        token = v.strip().upper()
                        if token in allowed:
                            return token
    except Exception:
        pass

    # 2) Prefer the last explicit ANSWER/FINAL ANSWER line. Clinical rationales
    # often discuss another "option B" before stating "FINAL ANSWER: A".
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        answer_label = _ANSWER_LINE_RE.search(line)
        if not answer_label:
            continue
        answer_text = line[answer_label.end() :]
        boxed = _BOXED_CHOICE_RE.findall(answer_text)
        for candidate in reversed(boxed):
            token = candidate.upper()
            if token in allowed:
                return token
        candidates = re.findall(r"\b([A-Z])\b", answer_text.upper())
        for candidate in reversed(candidates):
            if candidate in allowed:
                return candidate

    # 3) A boxed answer near the end is stronger than an option mentioned in
    # the rationale, even when the model omits an ANSWER label.
    for candidate in reversed(_BOXED_CHOICE_RE.findall(text)):
        token = candidate.upper()
        if token in allowed:
            return token

    # 4) Generic labels such as "Choice C" or "Option D". Use the last match
    # because models commonly revise an earlier candidate in their conclusion.
    for match in reversed(list(_GENERIC_ANSWER_RE.finditer(text))):
        token = match.group(1).upper()
        if token in allowed:
            return token

    # 5) Mid-line labels like "The category is A) ..."
    m = _PAREN_CHOICE_RE.search(text)
    if m:
        token = m.group(1).upper()
        if token in allowed:
            return token

    # 6) Final-line fallback, including labels like "B. Oligodendroglioma"
    for ln in reversed(lines[-10:]):
        token = ln.strip().upper()
        if token in allowed:
            return token
        m = _LEADING_CHOICE_RE.match(ln)
        if m:
            token = m.group(1).upper()
            if token in allowed:
                return token
        m = _PAREN_CHOICE_RE.search(ln)
        if m:
            token = m.group(1).upper()
            if token in allowed:
                return token
        m = _TRAILING_CHOICE_RE.search(ln)
        if m:
            token = m.group(1).upper()
            if token in allowed:
                return token

    return None


def extract_choice(text: str) -> Optional[str]:
    """Backward-compatible A/B/C extractor."""

    return extract_choice_from_set(text, valid_choices=("A", "B", "C"))


def extract_numbered_answers(text: str, *, question_ids: Sequence[int]) -> Optional[str]:
    """Extract compact numbered answers like ``1C2D3B4A``.

    Accepts loose formats such as:
      - 1C2D3B4A
      - 1:C 2:D 3:B 4:A
      - 1) C, 2) D, 3) B, 4) A
    """

    if not text:
        return None

    ordered_ids = [str(int(q)) for q in question_ids]
    qset = set(ordered_ids)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    def _complete_answer(candidate: str) -> Optional[str]:
        found: Dict[str, str] = {}
        compact = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        for question_id, answer in _NUMBERED_COMPACT_RE.findall(compact):
            if question_id in qset:
                found[question_id] = answer.upper()
        if set(found) != qset:
            return None
        return "".join(f"{question_id}{found[question_id]}" for question_id in ordered_ids)

    # Prefer the last explicit ANSWER line. Do not accept partial pairs:
    # numbered clinical prose such as "3) Pathological N stage" must not become
    # the bogus answer "3P".
    for line in reversed(lines):
        upper = line.upper()
        matches = list(re.finditer(r"\bANSWER\b\s*[:\-]?\s*(.+)$", upper))
        if not matches:
            continue
        answer = _complete_answer(matches[-1].group(1))
        if answer is not None:
            return answer

    # Support a compact answer wrapped in LaTeX without an ANSWER label.
    for boxed_text in reversed(re.findall(r"\\boxed\{([^}]+)\}", text, flags=re.IGNORECASE)):
        answer = _complete_answer(boxed_text)
        if answer is not None:
            return answer

    # Support a final bare compact line.
    for line in reversed(lines):
        if re.fullmatch(r"(?:\s*[1-9]\s*[A-Z]\s*)+", line, flags=re.IGNORECASE):
            answer = _complete_answer(line)
            if answer is not None:
                return answer

    # Finally accept five explicit per-question lines, but only when the full
    # requested question set is present.
    found: Dict[str, str] = {}
    for line in lines:
        match = _NUMBERED_LINE_RE.match(line)
        if match and match.group(1) in qset:
            found[match.group(1)] = match.group(2).upper()
            continue
        question_match = re.match(r"^\s*(?:QUESTION\s*)?([1-9])\s*[:=\-\)]", line, flags=re.IGNORECASE)
        answer_matches = list(re.finditer(r"\b(?:FINAL\s+)?ANSWER\b\s*[:\-]?\s*([A-Z])\b", line, flags=re.IGNORECASE))
        if question_match and answer_matches and question_match.group(1) in qset:
            found[question_match.group(1)] = answer_matches[-1].group(1).upper()
    if set(found) != qset:
        return None
    return "".join(f"{question_id}{found[question_id]}" for question_id in ordered_ids)
