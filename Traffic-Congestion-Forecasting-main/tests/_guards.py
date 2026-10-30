"""
Skip guards shared by the test modules: tests that need the dataset or the trained files SKIP (with a clear reason)
instead of erroring when those files are absent (the dataset is git-ignored, so a fresh clone does not contain it).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _isolation  # noqa: E402,F401  (COMMUNITY_DB_PATH -> a temporary database: tests never touch the real community.db)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "Banglore_traffic_Dataset.csv"
MODEL_FILES = [ROOT / "model" / n for n in ("model.h5", "scaler.pkl", "lstm_features.json",
                                            "xgboost_model.json", "xgboost_features.json")]
OUTPUT_FILES = [ROOT / "model" / n for n in ("xgboost_predictions.csv", "lstm_predictions.csv", "baseline_predictions.csv",
                                             "metrics.json", "evaluation_report.json")]

needs_data = unittest.skipUnless(
    DATA_FILE.is_file(), f"dataset not found: {DATA_FILE} (place Banglore_traffic_Dataset.csv in data/raw/)")
needs_models = unittest.skipUnless(
    all(p.is_file() for p in MODEL_FILES), "trained model files not found in model/ (see README)")
needs_outputs = unittest.skipUnless(
    all(p.is_file() for p in OUTPUT_FILES), "saved evaluation outputs not found in model/ (see README)")


def needs_all(cls):
    """Class decorator: dataset + trained models + saved evaluation outputs."""
    return needs_data(needs_models(needs_outputs(cls)))
