from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import IMAGE_HEIGHT, IMAGE_WIDTH


def preprocess_array(image: np.ndarray, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    h, w = thresholded.shape
    scale = min(width / max(w, 1), height / max(h, 1))
    resized_w = max(1, int(w * scale))
    resized_h = max(1, int(h * scale))
    resized = cv2.resize(thresholded, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((height, width), 255, dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas.astype("float32") / 255.0


def preprocess_path(path: str | Path) -> np.ndarray:
    path = Path(path)
    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return preprocess_array(image)


def preprocess_pil(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    array = np.array(rgb)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return preprocess_array(bgr)


def model_input(image: np.ndarray) -> np.ndarray:
    return np.expand_dims(image.T[..., np.newaxis], axis=0)
