from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .predict import PredictionResult, predict_image
from .segmentation import segment_text_lines


@dataclass
class FullPageResult:
    crop_index: int
    box: tuple[int, int, int, int]
    prediction: PredictionResult


def read_full_prescription(image: Image.Image) -> list[FullPageResult]:
    regions = segment_text_lines(image)
    results: list[FullPageResult] = []

    for index, region in enumerate(regions, start=1):
        prediction = predict_image(region.image)
        results.append(
            FullPageResult(
                crop_index=index,
                box=region.box,
                prediction=prediction,
            )
        )

    return results
