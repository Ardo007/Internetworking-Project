"""
Compute EER and F1 for the DNS-tunnelling detector (EXTERNAL + GraphTunnel self-check).

Two independent evaluations are produced here, because only one of them is a
valid measure of real-world performance:

  EXTERNAL
    liam_run2 scored against data it never trained on:
      - Tu et al. (2023) multi-tool capture (Data/zeek/external_tu2023,
        8 tools + benign)
      - the malware-traffic-analysis.net dnscat2 capture, reconstructed from
        its Wireshark CSV (Data/dns-tunneling-analysis-dnscat2-main)
    This EER/F1 is a genuine generalisation measure.

  GRAPHTUNNEL SELF-CHECK
    liam_run2 scored against config B's held-out GraphTunnel rows
    (test + heldout roles). liam_run2's own features.json says it was
    trained on EVERY GraphTunnel capture ("not_for_evaluation": true,
    "config_description": "final model: every capture is training data").
    These rows were in its training set, so this number is NOT a valid
    generalisation metric -- it's reported only as a sanity check (it
    should come out close to perfect; if it doesn't, something is actually
    wrong) and is clearly labelled as such everywhere it's printed.
    zeek_experiments._require_evaluable() raises on this exact situation
    for the project's own evaluation helpers; this script deliberately
    bypasses that guard for the self-check and prints the warning instead.

Per-row (window, domain) probabilities are written to results/ as CSVs so
you can recompute metrics, plot ROC curves, etc. without rerunning models.

Usage (from the activated venv, at the repo root):
  python notebooks\\compute_formal_metrics.py
  python notebooks\\compute_formal_metrics.py --model-dir "models\\zeek_bilstm\\liam_run2"
  python notebooks\\compute_formal_metrics.py --skip-graphtunnel   # external only, skip/avoid rebuilding the big feature cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeek_feature_extraction as zfe
import model_artifacts as ma


def eer_and_f1(probs, labels, threshold=0.5):
    """labels: bool array, True = tunnel. probs: tunnel probability in [0, 1].

    EER and F1 both need rows of both classes to mean anything (EER is a
    FPR/FNR crossover; F1 needs both precision and recall to be defined).
    A single-class subset (e.g. one source that's 100% tunnel) gets "n/a"
    for those two fields instead of crashing or silently reporting 0%.
    """
    from sklearn.metrics import roc_curve, f1_score

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    has_both_classes = bool(labels.any() and (~labels).any())

    pred = probs >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    if has_both_classes:
        f1 = float(f1_score(labels, pred))
        fpr, tpr, thresholds = roc_curve(labels, probs)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2)
        eer_threshold = float(thresholds[idx])
    else:
        f1 = float("nan")
        eer = float("nan")
        eer_threshold = float("nan")

    return {
        "n": len(labels), "n_tunnel": int(labels.sum()), "n_benign": int((~labels).sum()),
        "eer": eer, "eer_threshold": eer_threshold,
        "f1": f1, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "single_class": not has_both_classes,
    }


def print_metrics(name, m):
    print(f"\n=== {name} ===")
    print(f"  rows: {m['n']} ({m['n_tunnel']} tunnel / {m['n_benign']} benign)")
    if m["single_class"]:
        accuracy = (m["tp"] + m["tn"]) / m["n"] if m["n"] else float("nan")
        print(f"  EER: n/a (only one class present in this subset)")
        print(f"  F1 @ 0.5: n/a (only one class present)   accuracy @ 0.5 = {accuracy:.4f}")
    else:
        print(f"  EER: {m['eer'] * 100:.3f}%  (crossover threshold {m['eer_threshold']:.4f})")
        print(f"  F1 @ 0.5: {m['f1']:.4f}   precision={m['precision']:.4f}  recall={m['recall']:.4f}")
    print(f"  confusion @ 0.5: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def score_dns_log(dns_log_path, capture_id, artifacts):
    """Per-(window, domain) tunnel_prob for one dns.log, ensembled across the model's folds."""
    frame = zfe.extract_query_records_from_zeek(
        dns_log_path, capture_id, category="own_benign", tool="external", label=0,
    )
    if frame.empty:
        return None
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        return None
    frame = zfe.add_domain_aggregates(frame)

    tensor = ma.prepare_model_input(frame, artifacts)
    probs = np.stack([m.predict(tensor, verbose=0)[:, 1] for m in artifacts["models"]], axis=0).mean(axis=0)

    out = frame[["capture_id", "window_id", "base_domain"]].copy()
    out["tunnel_prob"] = probs
    return out.groupby(["capture_id", "window_id", "base_domain"], as_index=False)["tunnel_prob"].mean()


def external_rows(artifacts, data_root):
    """Per-(capture, window, domain) tunnel_prob + true label for every external dataset found."""
    rows = []

    tu_root = data_root / "zeek" / "external_tu2023"
    if tu_root.exists():
        for dns_log in sorted(tu_root.glob("*/*/dns.log")):
            tool_folder = dns_log.parent.parent.name
            capture_id = f"tu2023/{dns_log.parent.name}"
            per_wd = score_dns_log(dns_log, capture_id, artifacts)
            if per_wd is None:
                print(f"  [skip] {capture_id}: no usable DNS records")
                continue
            per_wd["true_tunnel"] = tool_folder != "benign"
            per_wd["source"] = "tu2023"
            per_wd["tool"] = tool_folder
            rows.append(per_wd)
    else:
        print(f"  [note] {tu_root} not found, skipping Tu2023")

    mta_log = data_root / "dns-tunneling-analysis-dnscat2-main" / "synthetic_dns.log"
    if mta_log.exists():
        per_wd = score_dns_log(mta_log, "mta_dnscat2", artifacts)
        if per_wd is not None:
            per_wd["true_tunnel"] = True
            per_wd["source"] = "mta_dnscat2"
            per_wd["tool"] = "dnscat2"
            rows.append(per_wd)
    else:
        print(f"  [note] {mta_log} not found, skipping the malware-traffic-analysis.net capture")

    if not rows:
        return pd.DataFrame(columns=["capture_id", "window_id", "base_domain", "tunnel_prob",
                                     "true_tunnel", "source", "tool"])
    return pd.concat(rows, ignore_index=True)


def graphtunnel_self_check_rows(artifacts, config="B"):
    """liam_run2 scored on config B's held-out (test + heldout) GraphTunnel rows.

    These rows were in liam_run2's own training set (it trains on every
    capture) -- see the module docstring. Not a valid generalisation metric.
    """
    import dataset_splits as ds
    import zeek_experiments as ze

    table = ze.load_features(verbose=True)
    masks = ze.split_masks(table, config)
    meta = ze.evaluation_meta(table, config, masks["evaluation"])
    X = ma.prepare_model_input(table.loc[masks["evaluation"]], artifacts)
    probs = np.stack(
        [m.predict(X, batch_size=8192, verbose=0)[:, 1] for m in artifacts["models"]], axis=0
    ).mean(axis=0)
    out = meta.copy()
    out["tunnel_prob"] = probs
    out["true_tunnel"] = out["Label"].eq("tunnel")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(zfe.PROJECT_ROOT / "models" / "zeek_bilstm" / "liam_run2"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-graphtunnel", action="store_true",
                        help="Skip the GraphTunnel self-check (builds/loads the full feature "
                             "cache, which can take a while the first time).")
    args = parser.parse_args()

    results_dir = zfe.PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    artifacts = ma.load_artifacts(args.model_dir)
    print(f"Loaded {len(artifacts['models'])} models from {args.model_dir}")
    if artifacts["features"].get("not_for_evaluation"):
        print(f"[note] {artifacts['features'].get('note', '')}\n")

    print("Scoring external datasets (Tu2023 + malware-traffic-analysis.net dnscat2) ...")
    ext = external_rows(artifacts, zfe.PROJECT_ROOT / "Data")
    if len(ext):
        ext_path = results_dir / "formal_metrics_external_rows.csv"
        ext.to_csv(ext_path, index=False)
        print(f"  -> {len(ext)} (window, domain) rows across {ext['capture_id'].nunique()} captures "
              f"(wrote {ext_path})")

        m_ext = eer_and_f1(ext["tunnel_prob"], ext["true_tunnel"], args.threshold)
        print_metrics("EXTERNAL -- valid generalisation metric (liam_run2 never trained on this data)", m_ext)

        for source, g in ext.groupby("source"):
            m = eer_and_f1(g["tunnel_prob"], g["true_tunnel"], args.threshold)
            print_metrics(f"  external / {source}", m)
    else:
        print("  No external rows scored -- check the Data/ paths above.")

    if not args.skip_graphtunnel:
        print("\nScoring GraphTunnel config B held-out rows (SELF-CHECK ONLY, see module docstring) ...")
        gt = graphtunnel_self_check_rows(artifacts)
        gt_path = results_dir / "formal_metrics_graphtunnel_selfcheck_rows.csv"
        gt.to_csv(gt_path, index=False)
        m_gt = eer_and_f1(gt["tunnel_prob"], gt["true_tunnel"], args.threshold)
        print_metrics(
            "GRAPHTUNNEL SELF-CHECK -- NOT a valid generalisation metric "
            "(liam_run2 trained on these exact rows; sanity check only)", m_gt)
        print(f"  (wrote {gt_path})")
    else:
        print("\n[skipped] GraphTunnel self-check (--skip-graphtunnel)")

    print("\nDone.")


if __name__ == "__main__":
    main()
