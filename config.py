from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACE_ROOT = ROOT / "datasets" / "images"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFESTS = ROOT / "data" / "manifests"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
LIVE = ROOT / "live_test"
SEED = 42
COMMANDS = ("ON", "OFF", "GO", "STOP")
UNSUPPORTED = ("YES", "NO")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ROLES = {"user_01": "Admin", "user_02": "Employee", "user_03": "Guest"}


def make_folders():
    for folder in (RAW, PROCESSED, MANIFESTS, MODELS, RESULTS / "metrics",
                   RESULTS / "confusion_matrices"):
        folder.mkdir(parents=True, exist_ok=True)