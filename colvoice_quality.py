"""Text/audio quality metrics shared by the ColVoice UI and local API."""

from __future__ import annotations

import re
import unicodedata


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("’", "'")
    return re.findall(r"[\w']+", text, flags=re.UNICODE)


def compare_transcripts(reference: str, hypothesis: str) -> dict:
    """Return WER-style metrics and concrete word-level differences."""
    expected = normalize_words(reference)
    actual = normalize_words(hypothesis)
    rows = [[0] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    operations = [["eq"] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    for i in range(1, len(expected) + 1):
        rows[i][0] = i
        operations[i][0] = "del"
    for j in range(1, len(actual) + 1):
        rows[0][j] = j
        operations[0][j] = "ins"
    for i, expected_word in enumerate(expected, 1):
        for j, actual_word in enumerate(actual, 1):
            if expected_word == actual_word:
                rows[i][j] = rows[i - 1][j - 1]
                operations[i][j] = "eq"
                continue
            candidates = (
                (rows[i - 1][j] + 1, "del"),
                (rows[i][j - 1] + 1, "ins"),
                (rows[i - 1][j - 1] + 1, "sub"),
            )
            rows[i][j], operations[i][j] = min(candidates, key=lambda item: item[0])

    substitutions = 0
    insertions = 0
    deletions = 0
    missing: list[str] = []
    unexpected: list[str] = []
    i, j = len(expected), len(actual)
    while i or j:
        operation = operations[i][j]
        if operation == "eq":
            i -= 1
            j -= 1
        elif operation == "sub":
            substitutions += 1
            missing.append(expected[i - 1])
            unexpected.append(actual[j - 1])
            i -= 1
            j -= 1
        elif operation == "del":
            deletions += 1
            missing.append(expected[i - 1])
            i -= 1
        else:
            insertions += 1
            unexpected.append(actual[j - 1])
            j -= 1

    denominator = max(1, len(expected))
    wer = (substitutions + insertions + deletions) / denominator
    return {
        "accuracy_percent": round(max(0.0, 1.0 - wer) * 100, 1),
        "wer_percent": round(wer * 100, 1),
        "reference_words": len(expected),
        "recognized_words": len(actual),
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "missing_words": list(reversed(missing[:30])),
        "unexpected_words": list(reversed(unexpected[:30])),
    }
