from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.dataset import find_dataset_root, load_all_labels


def image_size(path: str) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def main() -> None:
    dataset_root = find_dataset_root()
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    df = load_all_labels(dataset_root)
    sizes = df["image_path"].apply(image_size)
    df["width"] = sizes.apply(lambda item: item[0])
    df["height"] = sizes.apply(lambda item: item[1])

    summary = {
        "dataset_root": str(dataset_root),
        "total_images": int(len(df)),
        "unique_medicine_names": int(df["medicine_name"].nunique()),
        "missing_labels": int(df["medicine_name"].isna().sum()),
        "missing_image_files": int((~df["image_path"].map(lambda path: Path(path).exists())).sum()),
    }

    print("Dataset summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nMost common medicine names")
    print(df["medicine_name"].value_counts().head(20))

    df.to_csv(output_dir / "dataset_exploration.csv", index=False)
    pd.Series(summary).to_csv(output_dir / "dataset_summary.csv", header=["value"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["width"].dropna().plot(kind="hist", bins=30, ax=axes[0], title="Image width distribution")
    df["height"].dropna().plot(kind="hist", bins=30, ax=axes[1], title="Image height distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "image_size_distribution.png", dpi=160)

    top_counts = df["medicine_name"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_counts.sort_values().plot(kind="barh", ax=ax, title="Top 20 medicine names")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)

    print(f"\nSaved exploration outputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
