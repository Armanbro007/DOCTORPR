from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BLANK_TOKEN, MODEL_PATH, VOCAB_PATH
from .dataset import load_labels, medicine_dictionary
from .postprocessing import correct_medicine_name
from .preprocessing import model_input, preprocess_path, preprocess_pil

EXACT_FALLBACK_THRESHOLD = 0.995


@dataclass
class PredictionResult:
    raw_prediction: str
    corrected_prediction: str
    confidence: float
    top_matches: list[tuple[str, float]]
    model_source: str


def _ctc_decode(probabilities: np.ndarray, idx_to_char: dict[int, str]) -> tuple[str, float]:
    best = np.argmax(probabilities, axis=-1)[0]
    best_probs = np.max(probabilities, axis=-1)[0]
    blank_index = len(idx_to_char)
    chars = []
    scores = []
    previous = None
    for index, score in zip(best, best_probs):
        index = int(index)
        if index != previous and index != blank_index:
            chars.append(idx_to_char.get(index, ""))
            scores.append(float(score))
        previous = index
    text = "".join(chars).strip()
    confidence = float(np.mean(scores)) if scores else 0.0
    return text, confidence


@lru_cache(maxsize=1)
def load_trained_model():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    vocab_data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    idx_to_char = {int(index): char for index, char in vocab_data["idx_to_char"].items()}
    return model, idx_to_char


@lru_cache(maxsize=1)
def _fallback_index(max_items: int = 2500) -> tuple[np.ndarray, list[str]]:
    labels = load_labels("train").dropna(subset=["medicine_name", "image_path"])
    if len(labels) > max_items:
        labels = labels.sample(max_items, random_state=42)

    vectors = []
    names = []
    for row in labels.itertuples(index=False):
        path = Path(row.image_path)
        if not path.exists():
            continue
        image = preprocess_path(path)
        vectors.append(image.reshape(-1))
        names.append(str(row.medicine_name))

    if not vectors:
        raise FileNotFoundError("No readable training images found for fallback prediction.")
    matrix = np.vstack(vectors).astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms, names


def _fallback_predict(image: np.ndarray, top_k: int = 3) -> tuple[str, float, list[tuple[str, float]]]:
    matrix, names = _fallback_index()
    vector = image.reshape(1, -1).astype("float32")
    vector = vector / (np.linalg.norm(vector, axis=1, keepdims=True) + 1e-8)
    scores = (matrix @ vector.T).reshape(-1)
    order = np.argsort(scores)[::-1][:top_k]
    top_matches = [(names[i], float(scores[i])) for i in order]
    prediction, confidence = top_matches[0]
    return prediction, confidence, top_matches


def predict_image(image: Image.Image) -> PredictionResult:
    processed = preprocess_pil(image)
    dictionary = medicine_dictionary()
    trained = load_trained_model()

    if trained is not None:
        model, idx_to_char = trained
        probabilities = model.predict(model_input(processed), verbose=0)
        raw_prediction, confidence = _ctc_decode(probabilities, idx_to_char)
        corrected, correction_score = correct_medicine_name(raw_prediction, dictionary)
        final_confidence = confidence * max(correction_score, 0.5)
        return PredictionResult(
            raw_prediction=raw_prediction,
            corrected_prediction=corrected,
            confidence=final_confidence,
            top_matches=[(corrected, correction_score)],
            model_source="CRNN + CTC model",
        )

    raw_prediction, confidence, top_matches = _fallback_predict(processed)
    if confidence < EXACT_FALLBACK_THRESHOLD:
        return PredictionResult(
            raw_prediction="No exact match",
            corrected_prediction="No exact medicine match found",
            confidence=confidence,
            top_matches=top_matches,
            model_source="exact-match fallback",
        )

    corrected, correction_score = correct_medicine_name(raw_prediction, dictionary)
    return PredictionResult(
        raw_prediction=raw_prediction,
        corrected_prediction=corrected,
        confidence=confidence,
        top_matches=top_matches,
        model_source="exact-match fallback",
    )
