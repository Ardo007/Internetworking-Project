import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))
import zeek_feature_extraction as zfe
import model_artifacts as ma

DEFAULT_LOG_PATH = zfe.PROJECT_ROOT / "results" / "web_scoring_log.csv"
LOG_COLUMNS = ["scored_at", "session_id", "window_name", "window_id", "base_domain",
               "n_queries", "mean_tunnel_prob", "n_flagged", "verdict"]


def default_model_dir():
    models_root = zfe.PROJECT_ROOT / "models" / "zeek_bilstm"
    candidates = sorted(models_root.glob("*/final"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No */final model directory under {models_root}")
    return candidates[0]


def load_model(directory, n_models=1):
    import keras

    directory = Path(directory)
    features = json.loads((directory / "features.json").read_text(encoding="utf-8"))
    names = features["models"][:n_models]
    return {
        "models": [keras.models.load_model(directory / name) for name in names],
        "model_names": names,
        "scaler": joblib.load(directory / "scaler.joblib"),
        "label_encoder": joblib.load(directory / "label_encoder.joblib"),
        "features": features,
        "directory": directory,
    }


def score_frame(frame, artifacts, threshold):
    frame = zfe.add_domain_aggregates(frame)
    ml = zfe.to_ml_frame(
        frame, label_col=None,
        extra_cols=["capture_id", "window_start", "window_id", "base_domain"],
        features=artifacts["features"]["feature_columns"],
    )
    tensor = ma.prepare_model_input(frame, artifacts)
    probs = np.stack([m.predict(tensor, verbose=0)[:, 1] for m in artifacts["models"]], axis=0)
    row_prob = probs.mean(axis=0)

    per_row = ml[["window_id", "base_domain"]].copy()
    per_row["tunnel_prob"] = row_prob
    per_row["flagged"] = row_prob >= threshold

    summary = per_row.groupby(["window_id", "base_domain"], as_index=False).agg(
        n_queries=("tunnel_prob", "size"),
        mean_tunnel_prob=("tunnel_prob", "mean"),
        n_flagged=("flagged", "sum"),
    )
    summary["verdict"] = np.where(summary["mean_tunnel_prob"] >= threshold, "TUNNEL", "benign")
    return summary.sort_values("mean_tunnel_prob", ascending=False)


def score_minute(dns_logs, minute_start, capture_id, artifacts, threshold):
    if not dns_logs:
        return None
    frame = zfe.extract_query_records_from_zeek(
        dns_logs, capture_id, category="own_benign", tool="live", label=0,
    )
    if frame.empty:
        return None
    frame = frame[frame["window_start"] == minute_start]
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        return None
    return score_frame(frame, artifacts, threshold)


class ScoreLog:
    def __init__(self, path=DEFAULT_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.file = self.path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=LOG_COLUMNS)
        if is_new:
            self.writer.writeheader()
            self.file.flush()

    def write(self, session_id, window_name, window_id, summary):
        scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for _, row in summary.iterrows():
            self.writer.writerow({
                "scored_at": scored_at,
                "session_id": session_id,
                "window_name": window_name,
                "window_id": window_id,
                "base_domain": row["base_domain"],
                "n_queries": row["n_queries"],
                "mean_tunnel_prob": row["mean_tunnel_prob"],
                "n_flagged": row["n_flagged"],
                "verdict": row["verdict"],
            })
        self.file.flush()

    def close(self):
        self.file.close()
