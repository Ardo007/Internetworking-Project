"""
Score a synthetic dns.log reconstructed from a non-Zeek source (e.g. a
Wireshark CSV export) against a trained DNS-tunnelling model.

Same as score_capture.py, with one addition: response_min_ttl can't be
recovered from a Wireshark one-line "Info" summary (no raw TTL values are
printed there), so rows missing it are imputed with the model's own
training-set mean for that feature -- which becomes exactly 0 after
scaling, i.e. a neutral "no signal" value rather than one that could bias
the verdict either way.

Usage:
  python notebooks\\score_external_csv.py <synthetic-dns.log-path> --model-dir DIR [--capture-id NAME] [--true-label tunnel|benign]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeek_feature_extraction as zfe
import model_artifacts as ma


def score(dns_log_path, model_dir, capture_id, threshold=0.5, true_label=None):
    artifacts = ma.load_artifacts(model_dir)
    if artifacts["features"].get("not_for_evaluation"):
        print(f"[using '{Path(model_dir).name}' -- the deploy model, "
              f"trained on every GraphTunnel capture]\n")

    frame = zfe.extract_query_records_from_zeek(
        dns_log_path, capture_id, category="external_test", tool="dnscat2_mta", label=1,
    )
    if frame.empty:
        print("No usable DNS records in this capture.")
        return None
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        print("No usable DNS records in this capture (only mDNS/.local traffic).")
        return None
    frame = zfe.add_domain_aggregates(frame)

    # Impute response_min_ttl (not recoverable from this source) with the
    # model's own training mean for that column -> exactly 0 after scaling.
    spec = artifacts["features"]
    if frame["response_min_ttl"].isna().any():
        ttl_idx = spec["input_columns"].index("response_min_ttl")
        ttl_mean = artifacts["scaler"].mean_[ttl_idx]
        n_missing = frame["response_min_ttl"].isna().sum()
        frame["response_min_ttl"] = frame["response_min_ttl"].fillna(ttl_mean)
        print(f"[note] imputed response_min_ttl for {n_missing}/{len(frame)} rows "
              f"with training mean ({ttl_mean:.2f}) -- not recoverable from this source\n")

    ml = zfe.to_ml_frame(
        frame, label_col=None,
        extra_cols=["capture_id", "window_start", "window_id", "base_domain"],
        features=spec["feature_columns"],
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
    with pd.option_context("display.max_rows", 60, "display.width", 160):
        print(summary.to_string(index=False))
    print()
    print(summary["verdict"].value_counts().rename("windows").to_string())

    print(f"\nPer-row: {len(per_row)} queries, {per_row['flagged'].sum()} flagged tunnel "
          f"({100 * per_row['flagged'].mean():.1f}%), mean tunnel_prob={per_row['tunnel_prob'].mean():.4f}")

    if true_label is not None:
        correct = (summary["verdict"] == ("TUNNEL" if true_label == "tunnel" else "benign")).mean()
        print(f"Known ground truth: {true_label}  ->  {100*correct:.1f}% of windows verdict-correct")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dns_log", help="Path to the synthetic dns.log")
    parser.add_argument("--model-dir", required=True, help="Model artifacts folder")
    parser.add_argument("--capture-id", default=None, help="Label for this capture (default: file's parent folder name)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--true-label", choices=["tunnel", "benign"], default=None,
                        help="If you know the ground truth, report accuracy against it")
    args = parser.parse_args()
    capture_id = args.capture_id or Path(args.dns_log).parent.name
    score(args.dns_log, Path(args.model_dir), capture_id, args.threshold, args.true_label)


if __name__ == "__main__":
    main()
