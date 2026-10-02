"""
Score a Zeek dns.log against a trained DNS-tunnelling model.

Usage:
  python notebooks\score_capture.py <dns.log-path-or-zeek-output-folder> [--model-dir DIR] [--capture-id NAME]

Feeds one capture's dns.log through the same feature pipeline used for
training (zeek_feature_extraction) and the saved model's exact scaler and
one-hot columns (model_artifacts), then prints a per-(window, domain)
verdict: mean tunnel-probability and vote count across the model's 5 folds.

No label is needed -- that's the point of scoring. category/tool/label
passed to extract_query_records_from_zeek are just bookkeeping; the model
never sees them (model_artifacts.prepare_model_input drops the label and
ignores category/tool).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeek_feature_extraction as zfe
import model_artifacts as ma


def find_dns_log(path):
    path = Path(path)
    if path.is_file():
        return path
    candidate = path / "dns.log"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No dns.log found at or under {path}")


def default_model_dir():
    models_root = zfe.PROJECT_ROOT / "models" / "zeek_bilstm"
    candidates = sorted(models_root.glob("*/final"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No */final model directory under {models_root}")
    return candidates[0]


def score(dns_log_path, model_dir, capture_id=None, threshold=0.5):
    dns_log_path = find_dns_log(dns_log_path)
    capture_id = capture_id or dns_log_path.parent.name

    artifacts = ma.load_artifacts(model_dir)
    if artifacts["features"].get("not_for_evaluation"):
        print(f"[using '{Path(model_dir).parent.name}/final' -- the deploy model, "
              f"trained on every GraphTunnel capture]\n")

    frame = zfe.extract_query_records_from_zeek(
        dns_log_path, capture_id, category="own_benign", tool="live", label=0,
    )
    if frame.empty:
        print("No usable DNS records in this capture (all malformed / unmatched?).")
        return None
    # mDNS/DNS-SD (RFC 6762 ".local", multicast port 5353) never leaves the
    # local link, so it can't carry a tunnel payload and isn't represented
    # in GraphTunnel's training captures -- drop it rather than let it land
    # near the decision boundary undefended.
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        print("No usable DNS records in this capture (only mDNS/.local traffic).")
        return None
    frame = zfe.add_domain_aggregates(frame)

    ml = zfe.to_ml_frame(
        frame, label_col=None,
        extra_cols=["capture_id", "window_start", "window_id", "base_domain"],
        features=artifacts["features"]["feature_columns"],
    )
    tensor = ma.prepare_model_input(frame, artifacts)

    probs = np.stack([m.predict(tensor, verbose=0)[:, 1] for m in artifacts["models"]], axis=0)
    row_prob = probs.mean(axis=0)

    per_row = ml[["capture_id", "window_start", "window_id", "base_domain"]].copy()
    per_row["tunnel_prob"] = row_prob
    per_row["flagged"] = row_prob >= threshold

    summary = per_row.groupby(["capture_id", "window_id", "base_domain"], as_index=False).agg(
        window_start=("window_start", "first"),
        n_queries=("tunnel_prob", "size"),
        mean_tunnel_prob=("tunnel_prob", "mean"),
        n_flagged=("flagged", "sum"),
    )
    summary["window_start"] = pd.to_datetime(summary["window_start"], unit="s")
    summary["verdict"] = np.where(summary["mean_tunnel_prob"] >= threshold, "TUNNEL", "benign")
    summary = summary.sort_values("mean_tunnel_prob", ascending=False)

    print(f"Capture: {capture_id}  ({len(frame)} DNS records -> {len(summary)} window/domain rows)")
    with pd.option_context("display.max_rows", 30, "display.width", 160):
        print(summary.to_string(index=False))
    print()
    print(summary["verdict"].value_counts().rename("windows").to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dns_log", help="Path to a dns.log file, or a Zeek output folder containing one")
    parser.add_argument("--model-dir", default=None,
                        help="Model artifacts folder (default: newest models/zeek_bilstm/*/final)")
    parser.add_argument("--capture-id", default=None, help="Label for this capture (default: folder name)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    model_dir = Path(args.model_dir) if args.model_dir else default_model_dir()
    score(args.dns_log, model_dir, args.capture_id, args.threshold)


if __name__ == "__main__":
    main()
