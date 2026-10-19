import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import zeek_feature_extraction as zfe

MODELS_DIR = zfe.PROJECT_ROOT / "models" / "zeek_bilstm"
NOT_FOR_EVALUATION_FILE = "NOT_FOR_EVALUATION.txt"

CATEGORY_LEVELS = {
    "proto": ["tcp", "udp"],
    "qtype_name": sorted(set(zfe.QTYPE_NAMES.values()) | {"OTHER"}),
}


def encode_inputs(features, input_columns=None):
    X = features.copy()
    for column, levels in CATEGORY_LEVELS.items():
        if column not in X:
            continue
        values = X[column].astype(str)
        unknown = set(values.unique()) - set(levels)
        if unknown:
            raise ValueError(f"Unexpected {column} values: {sorted(unknown)}")
        X[column] = pd.Categorical(values, categories=levels)
    encoded = pd.get_dummies(X, drop_first=True, dtype="float64")
    if input_columns is not None:
        encoded = encoded.reindex(columns=list(input_columns), fill_value=0.0)
    return encoded


def to_tensor(matrix):
    matrix = np.asarray(matrix, dtype="float32")
    return matrix.reshape((matrix.shape[0], 1, matrix.shape[1]))


def git_info():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=zfe.PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def environment_info():
    import keras
    import sklearn
    import tensorflow as tf
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def save_artifacts(directory, models, scaler, label_encoder, feature_columns, input_columns, info):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("model_*.keras"):
        old.unlink()
    model_files = []
    for number, model in enumerate(models, start=1):
        name = f"model_{number}.keras"
        model.save(directory / name)
        model_files.append(name)
    joblib.dump(scaler, directory / "scaler.joblib")
    joblib.dump(label_encoder, directory / "label_encoder.joblib")

    input_columns = list(input_columns)
    features = {
        "model": "zeek_bilstm",
        "feature_columns": list(feature_columns),
        "input_columns": input_columns,
        "categorical_levels": {c: levels for c, levels in CATEGORY_LEVELS.items() if c in feature_columns},
        "qtype_name_dummy_columns": [c for c in input_columns if c.startswith("qtype_name_")],
        "label_classes": [str(c) for c in label_encoder.classes_],
        "window_seconds": zfe.WINDOW_SECONDS,
        "rejoin_max_gap_seconds": zfe.REJOIN_MAX_GAP_SECONDS,
        "models": model_files,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "versions": environment_info(),
        **info,
    }
    (directory / "features.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    marker = directory / NOT_FOR_EVALUATION_FILE
    if info.get("not_for_evaluation"):
        marker.write_text((info.get("note") or "Not for evaluation.") + "\n", encoding="utf-8")
    elif marker.exists():
        marker.unlink()
    return directory


def load_artifacts(directory):
    import keras

    directory = Path(directory)
    features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    return {
        "models": [keras.models.load_model(directory / name) for name in features["models"]],
        "scaler": joblib.load(directory / "scaler.joblib"),
        "label_encoder": joblib.load(directory / "label_encoder.joblib"),
        "features": features,
        "directory": directory,
    }


def prepare_model_input(frame, artifacts):
    spec = artifacts["features"]
    if spec["window_seconds"] != zfe.WINDOW_SECONDS:
        raise ValueError(f"Model was trained with {spec['window_seconds']} s windows, "
                         f"extractor uses {zfe.WINDOW_SECONDS} s")
    ml = zfe.to_ml_frame(frame, label_col=None, features=spec["feature_columns"])
    encoded = encode_inputs(ml, spec["input_columns"])
    scaled = artifacts["scaler"].transform(encoded)
    return to_tensor(scaled)
