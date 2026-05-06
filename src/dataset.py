from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATASET_GLOB, PROJECT_ROOT


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    labels_path: Path
    image_dir: Path


def find_dataset_root(project_root: Path = PROJECT_ROOT) -> Path:
    candidates = [path for path in project_root.glob(DATASET_GLOB) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "Dataset folder not found. Expected a folder like "
            "'Doctor's Handwritten Prescription BD dataset' in the project root."
        )
    return candidates[0]


def split_info(split: str, dataset_root: Path | None = None) -> DatasetSplit:
    root = dataset_root or find_dataset_root()
    split_key = split.lower()
    folder_map = {
        "train": ("Training", "training_labels.csv", "training_words"),
        "training": ("Training", "training_labels.csv", "training_words"),
        "validation": ("Validation", "validation_labels.csv", "validation_words"),
        "valid": ("Validation", "validation_labels.csv", "validation_words"),
        "test": ("Testing", "testing_labels.csv", "testing_words"),
        "testing": ("Testing", "testing_labels.csv", "testing_words"),
    }
    if split_key not in folder_map:
        raise ValueError(f"Unknown split '{split}'. Use train, validation, or test.")

    split_folder, labels_name, image_dir_name = folder_map[split_key]
    split_root = root / split_folder
    return DatasetSplit(
        name=split_key,
        labels_path=split_root / labels_name,
        image_dir=split_root / image_dir_name,
    )


def load_labels(split: str = "train", dataset_root: Path | None = None) -> pd.DataFrame:
    info = split_info(split, dataset_root)
    df = pd.read_csv(info.labels_path)
    df.columns = [column.strip().lower() for column in df.columns]
    df = df.rename(columns={"image": "image_name"})
    if "medicine_name" not in df.columns:
        raise ValueError(f"{info.labels_path} must contain a MEDICINE_NAME column.")
    df["image_path"] = df["image_name"].apply(lambda name: str(info.image_dir / str(name)))
    df["medicine_name"] = df["medicine_name"].astype(str).str.strip()
    if "generic_name" in df.columns:
        df["generic_name"] = df["generic_name"].astype(str).str.strip()
    return df


def load_all_labels(dataset_root: Path | None = None) -> pd.DataFrame:
    frames = []
    for split in ["train", "validation", "test"]:
        try:
            frame = load_labels(split, dataset_root)
            frame["split"] = split
            frames.append(frame)
        except FileNotFoundError:
            continue
    if not frames:
        raise FileNotFoundError("No dataset label CSV files were found.")
    return pd.concat(frames, ignore_index=True)


def medicine_dictionary(extra_csv: Path | None = None) -> list[str]:
    names = set()
    try:
        labels = load_all_labels()
        names.update(labels["medicine_name"].dropna().astype(str).str.strip())
    except FileNotFoundError:
        pass

    csv_path = extra_csv or PROJECT_ROOT / "medicine_list.csv"
    if csv_path.exists():
        values = pd.read_csv(csv_path).iloc[:, 0].dropna().astype(str).str.strip()
        names.update(values)

    return sorted(name for name in names if name)
