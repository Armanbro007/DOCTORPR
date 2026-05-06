from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class TextRegion:
    image: Image.Image
    box: tuple[int, int, int, int]


def _merge_nearby_boxes(boxes: list[tuple[int, int, int, int]], y_gap: int = 18) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    merged: list[tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        if not merged:
            merged.append((x, y, w, h))
            continue

        px, py, pw, ph = merged[-1]
        previous_mid = py + ph / 2
        current_mid = y + h / 2

        if abs(current_mid - previous_mid) <= max(y_gap, ph * 0.65):
            nx = min(px, x)
            ny = min(py, y)
            nr = max(px + pw, x + w)
            nb = max(py + ph, y + h)
            merged[-1] = (nx, ny, nr - nx, nb - ny)
        else:
            merged.append((x, y, w, h))

    return merged


def segment_text_lines(image: Image.Image, max_regions: int = 18) -> list[TextRegion]:
    rgb = image.convert("RGB")
    array = np.array(rgb)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    height, width = binary.shape
    kernel_width = max(18, width // 38)
    kernel_height = max(3, height // 220)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []

    min_area = max(80, int(width * height * 0.00015))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        if h < 8 or w < 20:
            continue
        if h > height * 0.35 or w > width * 0.95:
            continue
        boxes.append((x, y, w, h))

    boxes = _merge_nearby_boxes(boxes, y_gap=max(14, height // 90))
    boxes = sorted(boxes, key=lambda box: (box[1], box[0]))[:max_regions]

    regions: list[TextRegion] = []
    padding_x = max(8, width // 90)
    padding_y = max(6, height // 140)
    for x, y, w, h in boxes:
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(width, x + w + padding_x)
        bottom = min(height, y + h + padding_y)
        crop = rgb.crop((left, top, right, bottom))
        regions.append(TextRegion(image=crop, box=(left, top, right, bottom)))

    return regions
