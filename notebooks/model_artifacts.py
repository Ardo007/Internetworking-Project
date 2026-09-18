"""
Model inputs and saved artefacts for the Zeek DNS tunnelling model
==================================================================
encode_inputs turns the model feature columns (see
zeek_feature_extraction.FEATURE_COLUMNS) into the numeric input matrix. The
categorical columns get a fixed set of levels, so the one-hot columns are the
same whatever categories a data set happens to contain; training and live
scoring both go through this function.

Each trained configuration is saved to models/zeek_bilstm/<name>/:
  model_1.keras ... model_N.keras   Keras 3 models, one per training run
  scaler.joblib                     StandardScaler fitted on the training rows
  label_encoder.joblib              LabelEncoder for Label (benign / tunnel)
  features.json                     feature order, input (one-hot) columns,
                                    window length, row caps and sampling
                                    seed, training date, git commit and
                                    library versions
load_artifacts reads the folder back; prepare_model_input applies the saved
scaler and column order to a feature frame.
"""
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

#: Levels of the categorical features. One-hot encoding drops the first
#: level of each ("tcp", "A"), as the original notebook's drop_first did.
CATEGORY_LEVELS = {
    "proto": ["tcp", "udp"],
    "qtype_name": sorted(set(zfe.QTYPE_NAMES.values()) | {"OTHER"}),
}


def encode_inputs(features, input_columns=None):
    """One-hot encode the categorical feature columns.

    `features` holds model feature columns only (e.g. from to_ml_frame).
    With `input_columns` (the training columns), the result is reindexed
    to exactly those columns, missing ones filled with 0.
    """
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
    """(rows, features) -> (rows, 1, features), the BiLSTM input shape."""
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
    """Save models, scaler, label encoder and features.json to `directory`.

    `info` is merged into features.json (config, feature set, row caps,
    sampling seed, training rows, hyperparameters, ...).
    """
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
    return directory


def load_artifacts(directory):
    """Load a folder written by save_artifacts.

    Returns a dict with models (list of Keras models), scaler,
    label_encoder and features (the parsed features.json).
    """
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
    """Feature frame (with domain aggregates) -> scaled model input tensor,
    using the column order and scaler saved with the model."""
    spec = artifacts["features"]
    if spec["window_seconds"] != zfe.WINDOW_SECONDS:
        raise ValueError(f"Model was trained with {spec['window_seconds']} s windows, "
                         f"extractor uses {zfe.WINDOW_SECONDS} s")
    ml = zfe.to_ml_frame(frame, label_col=None, features=spec["feature_columns"])
    encoded = encode_inputs(ml, spec["input_columns"])
    scaled = artifacts["scaler"].transform(encoded)
    return to_tensor(scaled)
