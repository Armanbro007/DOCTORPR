from __future__ import annotations

from difflib import SequenceMatcher, get_close_matches

try:
    from rapidfuzz import process
except ImportError:  # pragma: no cover - optional dependency fallback
    process = None


def correct_medicine_name(prediction: str, medicine_names: list[str]) -> tuple[str, float]:
    cleaned = (prediction or "").strip()
    if not medicine_names:
        return cleaned, 0.0

    if process is not None:
        match = process.extractOne(cleaned, medicine_names)
        if match:
            return str(match[0]), float(match[1]) / 100.0

    match = get_close_matches(cleaned, medicine_names, n=1, cutoff=0)
    if not match:
        return medicine_names[0], 0.0
    score = SequenceMatcher(None, cleaned.lower(), match[0].lower()).ratio()
    return match[0], score
