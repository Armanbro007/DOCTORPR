from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO

from PIL import Image


@dataclass
class GeminiDecodeResult:
    transcription: str
    medicines: list[str]
    confidence: float
    notes: str
    model: str


def _jpeg_bytes(image: Image.Image) -> bytes:
    rgb = image.convert("RGB")
    rgb.thumbnail((1800, 1800))
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _extract_text(response_data: dict) -> str:
    candidates = response_data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part.get("text"), str)]
    return "\n".join(texts).strip()


def _json_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _candidate_models() -> list[str]:
    configured = os.getenv("GEMINI_VISION_MODEL", "").strip()
    defaults = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    models = [configured] if configured else []
    models.extend(defaults)

    unique_models: list[str] = []
    for model in models:
        if model and model not in unique_models:
            unique_models.append(model)
    return unique_models


def _call_gemini(model: str, api_key: str, payload: dict) -> dict:
    encoded_model = urllib.parse.quote(model, safe="")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{encoded_model}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def decode_prescription_with_gemini(image: Image.Image) -> GeminiDecodeResult:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key from Google AI Studio, then set it in PowerShell."
        )

    prompt = """
You are a medical handwriting transcription assistant for Bangladeshi prescriptions.
Read the handwritten text in the uploaded prescription image.

Return only valid JSON with these fields:
{
  "transcription": "all readable handwritten text, preserving line breaks where useful",
  "medicines": ["medicine name with dose/instruction if visible"],
  "confidence": 0.0,
  "notes": "brief uncertainty notes"
}

Rules:
- Do not invent medicine names or instructions.
- If a word is unclear, write [unclear].
- Include medicine names, dosages, and instructions when visible.
- Do not provide medical advice.
- confidence must be a number from 0 to 1 based on handwriting legibility.
""".strip()

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(_jpeg_bytes(image)).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    last_error = ""
    used_model = ""
    response_data: dict | None = None

    for model in _candidate_models():
        used_model = model
        for attempt in range(2):
            try:
                response_data = _call_gemini(model, api_key, payload)
                last_error = ""
                break
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                last_error = details
                if exc.code in {429, 500, 502, 503, 504} and attempt == 0:
                    time.sleep(2)
                    continue
                break
            except urllib.error.URLError as exc:
                last_error = str(exc)
                break
        if response_data is not None:
            break

    if response_data is None:
        raise RuntimeError(
            "Gemini is temporarily unavailable or rate-limited after trying "
            f"{', '.join(_candidate_models())}. Last error: {last_error}"
        )

    output_text = _extract_text(response_data)
    parsed = _json_from_text(output_text)
    medicines = parsed.get("medicines", [])
    if not isinstance(medicines, list):
        medicines = []

    return GeminiDecodeResult(
        transcription=str(parsed.get("transcription", "")).strip(),
        medicines=[str(item).strip() for item in medicines if str(item).strip()],
        confidence=float(parsed.get("confidence", 0.0)),
        notes=str(parsed.get("notes", "")).strip(),
        model=used_model,
    )
