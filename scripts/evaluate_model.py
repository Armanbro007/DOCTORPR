from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from src.dataset import load_labels
from src.predict import predict_image


def edit_distance(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def main() -> None:
    df = load_labels("test")
    total = 0
    exact = 0
    top3 = 0
    cer_distance = 0
    cer_chars = 0
    confusion = Counter()

    for row in df.itertuples(index=False):
        path = Path(row.image_path)
        if not path.exists():
            continue
        expected = str(row.medicine_name).strip()
        result = predict_image(Image.open(path))
        predicted = result.corrected_prediction.strip()
        total += 1
        exact += int(predicted.lower() == expected.lower())
        top_names = [name.lower() for name, _ in result.top_matches[:3]]
        top3 += int(expected.lower() in top_names or predicted.lower() == expected.lower())
        cer_distance += edit_distance(predicted.lower(), expected.lower())
        cer_chars += max(1, len(expected))
        confusion[(expected, predicted)] += 1

    print(f"Total evaluated: {total}")
    print(f"Accuracy: {exact / max(total, 1):.4f}")
    print(f"Top-3 accuracy: {top3 / max(total, 1):.4f}")
    print(f"Character Error Rate: {cer_distance / max(cer_chars, 1):.4f}")
    print("\nMost common mistakes")
    for (expected, predicted), count in confusion.most_common(20):
        if expected.lower() != predicted.lower():
            print(f"{expected} -> {predicted}: {count}")


if __name__ == "__main__":
    main()
