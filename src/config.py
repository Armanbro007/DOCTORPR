from pathlib import Path

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
BLANK_TOKEN = "[blank]"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "crnn_ctc.keras"
VOCAB_PATH = MODEL_DIR / "vocab.json"
DATASET_GLOB = "Doctor*Prescription*dataset"
