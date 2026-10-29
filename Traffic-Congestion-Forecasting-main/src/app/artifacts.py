"""
Trained-model artifacts the live demo needs, and clear errors when one is missing or unreadable.

Nothing here writes or regenerates files: it only reports what is missing and how to restore it.
"""
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"

_LSTM_FIX = ("Restore it from your own backup of the validated model files, or retrain with "
             "`python -m src.modelTraining.trainLstm2` (retraining OVERWRITES the LSTM files and changes the results: back them up first). "
             "Do not use model/original_backup: it is the old, discarded model.")
_XGB_FIX = ("Restore it from your own backup of the validated model files, or retrain with "
            "`python -m src.modelTraining.trainXgboost` (retraining OVERWRITES the XGBoost files and changes the results: back them up first). "
            "Do not use model/original_backup: it is the old, discarded model.")

# label -> (file name, how to get it back)
REQUIRED = {
    "LSTM model": ("model.h5", _LSTM_FIX),
    "LSTM scaler": ("scaler.pkl", _LSTM_FIX),
    "LSTM feature configuration": ("lstm_features.json", _LSTM_FIX),
    "XGBoost model": ("xgboost_model.json", _XGB_FIX),
    "XGBoost feature configuration": ("xgboost_features.json", _XGB_FIX),
}


class ArtifactError(RuntimeError):
    """A trained-model file is missing, unreadable or inconsistent; the message says how to fix it."""


def missing(model_dir=None) -> list:
    """[(label, file name, hint)] for every required artifact that does not exist."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    return [(label, name, hint) for label, (name, hint) in REQUIRED.items() if not (d / name).is_file()]


def describe_missing(items) -> str:
    return "; ".join(f"{label} (`{name}`)" for label, name, _ in items)


def require(labels, model_dir=None):
    """Raise ArtifactError if any of the given artifacts (by label) is missing."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    gone = [(l, REQUIRED[l][0], REQUIRED[l][1]) for l in labels if not (d / REQUIRED[l][0]).is_file()]
    if gone:
        raise ArtifactError(f"Required model file(s) missing in {d}: {describe_missing(gone)}. {gone[0][2]}")


def read_json(name: str, label: str, model_dir=None) -> dict:
    """Read a JSON artifact with a clear error instead of a raw FileNotFoundError / JSONDecodeError."""
    d = Path(model_dir) if model_dir else MODEL_DIR
    path = d / name
    if not path.is_file():
        raise ArtifactError(f"{label} `{name}` not found in {d}. {REQUIRED.get(label, ('', ''))[1]}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ArtifactError(f"{label} `{name}` is not valid JSON ({e}). {REQUIRED.get(label, ('', ''))[1]}") from e
