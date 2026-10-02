"""Score every Zeek dns.log under an external-evaluation folder tree
against a trained model, and summarize accuracy per tool.

Walks Data/zeek/<dataset_name>/**/dns.log (as written by
Ardashes_scripts/process_external_pcaps.py). The ground-truth label for
each capture is inferred from its top-level subfolder name: captures
under a "benign" folder are benign, everything else is tunnel -- override
with --benign-folder if yours is named differently.

Usage:
  python notebooks\\score_external_batch.py <dataset_name> --model-dir DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeek_feature_extraction as zfe
import model_artifacts as ma

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def score_one(dns_log_path, capture_id, artifacts, threshold):
    frame = zfe.extract_query_records_from_zeek(
        dns_log_path, capture_id, category="external_test", tool="external", label=None,
    )
    if frame.empty:
        return None, "no usable DNS records"
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        return None, "only mDNS/.local traffic"
    frame = zfe.add_domain_aggregates(frame)

    spec = artifacts["features"]
    for col in ("response_min_ttl", "response_rcode"):
        if frame[col].isna().any():
            idx = spec["input_columns"].index(col)
            mean = artifacts["scaler"].mean_[idx]
            frame[col] = frame[col].fillna(mean)

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
    return per_row, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="Folder name under Data/zeek/ (e.g. external_tu2023)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benign-folder", default="benign",
                        help="Top-level subfolder name whose captures are ground-truth benign")
    args = parser.parse_args()

    zeek_root = PROJECT_ROOT / "Data" / "zeek" / args.dataset_name
    dns_logs = sorted(zeek_root.rglob("dns.log"))
    if not dns_logs:
        print(f"No dns.log files found under {zeek_root}", file=sys.stderr)
        sys.exit(1)

    artifacts = ma.load_artifacts(Path(args.model_dir))
    if artifacts["features"].get("not_for_evaluation"):
        print(f"[using '{Path(args.model_dir).name}' -- the deploy model, "
              f"trained on every GraphTunnel capture]\n")

    rows = []
    for dns_log in dns_logs:
        capture_dir = dns_log.parent
        relative = capture_dir.relative_to(zeek_root)
        capture_id = relative.as_posix()
        tool = relative.parts[0] if relative.parts else "unknown"
        true_label = "benign" if tool == args.benign_folder else "tunnel"

        per_row, skip_reason = score_one(dns_log, capture_id, artifacts, args.threshold)
        if per_row is None:
            print(f"[skip] {capture_id}  ({skip_reason})")
            continue

        n = len(per_row)
        n_flagged = int(per_row["flagged"].sum())
        mean_prob = per_row["tunnel_prob"].mean()
        pred_label = "tunnel" if mean_prob >= args.threshold else "benign"
        correct = pred_label == true_label
        rows.append({
            "tool": tool, "capture": capture_id, "true_label": true_label,
            "n_queries": n, "n_flagged": n_flagged, "pct_flagged": 100 * n_flagged / n,
            "mean_tunnel_prob": mean_prob, "pred_label": pred_label, "correct": correct,
        })
        mark = "OK" if correct else "WRONG"
        print(f"[{mark:5s}] {capture_id:45s} true={true_label:7s} pred={pred_label:7s} "
              f"mean_prob={mean_prob:.4f}  {n_flagged}/{n} queries flagged")

    if not rows:
        print("Nothing scored.")
        return

    results = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("Per-tool summary:")
    by_tool = results.groupby("tool", as_index=False).agg(
        captures=("capture", "size"),
        correct=("correct", "sum"),
        mean_tunnel_prob=("mean_tunnel_prob", "mean"),
        total_queries=("n_queries", "sum"),
        total_flagged=("n_flagged", "sum"),
    )
    by_tool["accuracy"] = by_tool["correct"] / by_tool["captures"]
    with pd.option_context("display.max_rows", 30, "display.width", 160):
        print(by_tool.to_string(index=False))

    print(f"\nOverall: {results['correct'].sum()}/{len(results)} captures verdict-correct "
          f"({100 * results['correct'].mean():.1f}%)")

    out_path = PROJECT_ROOT / "results" / f"external_eval_{args.dataset_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
