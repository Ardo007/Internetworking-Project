"""
Experiment helpers for the Zeek DNS tunnelling notebook
=======================================================
load_features     the feature table plus, per configuration, the split,
                  role and train/val sampling flag of every row. Cached in
                  notebooks/dataset_zeek/ (gitignored) and rebuilt when the
                  manifest, Zeek logs, splits.csv, caps/seed or extraction
                  code change.
evaluate          metrics for one model's predictions on the test and
                  held-out rows (accuracy, false positive rates, recall per
                  unseen tool / platform capture).
window_size_rates the same rates split by how many queries the row's window
                  holds.
summarise         min / avg / max across runs.
write_report      results/zeek_run.md.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_splits as ds
import dns_feature_extraction
import zeek_feature_extraction as zfe

NOTEBOOK_DIR = Path(__file__).resolve().parent
CACHE_DIR = NOTEBOOK_DIR / "dataset_zeek"
FEATURES_CSV = CACHE_DIR / "features.csv"
CACHE_META = CACHE_DIR / "features_meta.json"
RESULTS_DIR = zfe.PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "zeek_run.md"

ANALYSIS_COLUMNS = ["response_latency"]
CACHE_BOOKKEEPING = ["ts", "window_start", "window_id", "window_query_count", "capture_id",
                     "category", "tool", "family", "base_domain", "rejoined"]
_CACHE_DTYPES = {
    "proto": "str", "qtype_name": "str", "capture_id": "str", "category": "str", "tool": "str",
    "family": "str", "base_domain": "str", "Label": "str", "window_id": "int64",
    "window_query_count": "int64", "rejoined": "bool",
    **{f"{kind}_{config}": "str" for config in ds.CONFIGS for kind in ("split", "role")},
    **{f"sample_{config}": "bool" for config in ds.CONFIGS},
}

WINDOW_SIZE_BUCKETS = ((0, 9, "<10"), (10, 99, "10-99"), (100, math.inf, ">=100"))

#: Rows scored per role; every rate is the share of rows predicted "tunnel"
#: (a false positive rate for benign roles, recall for tunnel roles).
ROLE_GROUPS = {
    "fpr_normal": ("FPR held-out normal", "fpr"),
    "fpr_wildcard": ("FPR held-out wildcard", "fpr"),
    "fpr_own_benign": ("FPR held-out own_benign", "fpr"),
    "unseen_tool": ("Recall unseen tools (unknownTunnel)", "recall"),
    "unseen_platform": ("Recall unseen platform (crossEndPoint, iodine on Android)", "recall"),
    "held_out_family": ("Recall held-out tunnel family", "recall"),
}
PER_TOOL_ROLES = ("unseen_tool", "unseen_platform")

#: The PCAP-based notebook's numbers (notebooks/README.md). Different data
#: and a four-file held-out set, so not directly comparable.
PCAP_BASELINE = {
    "held-out accuracy": "99.8%",
    "false positive rate (wildcard)": "0%",
    "recall on unseen tools": "99.3-100%",
    "held-out set": "cobalstrike, ozymandns, AndIodine-TXT, wildcard_00000 (dnspot and two wildcard "
                    "captures were in training)",
}


# ------------------------------------------------------------ features --

def _normalised_bytes(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def _fingerprint(manifest_rows):
    digest = hashlib.sha256()
    for source in (ds.SPLITS_PATH, zfe.__file__, ds.__file__, dns_feature_extraction.__file__, __file__):
        digest.update(_normalised_bytes(source))
    for row in manifest_rows:
        for path in zfe._dns_log_paths(row):
            stat = path.stat()
            digest.update(f"{row['capture_id']}|{row['label']}|{path}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    digest.update(json.dumps({"caps": ds.ROW_CAPS, "seed": ds.SAMPLING_SEED, "window": zfe.WINDOW_SECONDS,
                              "rejoin": zfe.REJOIN_MAX_GAP_SECONDS}, sort_keys=True).encode())
    return digest.hexdigest()


def load_features(rebuild=False, n_jobs=8, verbose=True):
    """Feature table with split_<config>, role_<config> and sample_<config>.

    Model features are filled as in to_ml_frame. sample_<config> marks the
    train/val rows kept by the per-window row caps.
    """
    manifest = zfe.read_manifest() + zfe.own_benign_manifest_rows()
    fingerprint = _fingerprint(manifest)
    if not rebuild and FEATURES_CSV.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            # Only empty cells are missing: "NULL" is a qtype, not a missing value.
            table = pd.read_csv(FEATURES_CSV, dtype=_CACHE_DTYPES, keep_default_na=False, na_values=[""],
                                float_precision="round_trip")
            if verbose:
                print(f"Loaded {len(table):,} rows from {FEATURES_CSV} (built {meta['built_at']})")
            table.attrs["capture_stats"] = meta["capture_stats"]
            return table
        if verbose:
            print("Cached features are out of date, rebuilding.")

    df = zfe.build_dataset_from_manifest(manifest, n_jobs=n_jobs, verbose=False)
    splits = ds.read_splits()
    df["family"] = df["capture_id"].map(splits.drop_duplicates("capture_id").set_index("capture_id")["family"])
    table = zfe.to_ml_frame(df, features=zfe.FEATURE_COLUMNS + ANALYSIS_COLUMNS, extra_cols=CACHE_BOOKKEEPING)
    for config in ds.CONFIGS:
        assignment = ds.assign_splits(df, splits, config)
        table[f"split_{config}"] = assignment["split"]
        table[f"role_{config}"] = assignment["role"]
        table[f"sample_{config}"] = ds.fit_sample_mask(df, assignment)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_CSV, index=False, lineterminator="\n")
    meta = {
        "fingerprint": fingerprint,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(table),
        "window_seconds": zfe.WINDOW_SECONDS,
        "row_caps": ds.ROW_CAPS,
        "sampling_seed": ds.SAMPLING_SEED,
        "capture_stats": df.attrs["capture_stats"],
    }
    CACHE_META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    table.attrs["capture_stats"] = meta["capture_stats"]
    if verbose:
        print(f"Built {len(table):,} rows and wrote {FEATURES_CSV}")
    return table


def split_masks(table, config):
    """Boolean row masks for one configuration: train and val (sampled),
    test, heldout, evaluation (test + heldout)."""
    split = table[f"split_{config}"]
    sample = table[f"sample_{config}"]
    return {
        "train": sample & split.eq("train"),
        "val": sample & split.eq("val"),
        "test": split.eq("test").fillna(False),
        "heldout": split.eq("heldout").fillna(False),
        "evaluation": split.isin(["test", "heldout"]),
    }


def evaluation_meta(table, config, rows):
    """Columns evaluate() needs, for the given rows."""
    meta = table.loc[rows, ["Label", "category", "capture_id", "tool", "family", "window_query_count"]].copy()
    meta["split"] = table.loc[rows, f"split_{config}"]
    meta["role"] = table.loc[rows, f"role_{config}"]
    return meta


# ---------------------------------------------------------- evaluation --

def _wildcard_index(meta):
    index = meta["capture_id"].str.extract(r"_(\d{5})_\d{14}$")[0]
    return pd.to_numeric(index, errors="coerce")


def _groups(meta):
    """name -> row mask, for every rate evaluate() reports."""
    truth = meta["Label"].eq("tunnel")
    test = meta["split"].eq("test")
    groups = {"test_benign": test & ~truth, "test_tunnel": test & truth}
    for role in ROLE_GROUPS:
        rows = meta["role"].eq(role)
        if rows.any():
            groups[role] = rows
        if role == "fpr_wildcard":
            wildcard_late = rows & (_wildcard_index(meta) >= 7)
            if wildcard_late.any():
                groups["fpr_wildcard_00007_00012"] = wildcard_late
    return groups


def evaluate(meta, predicted_tunnel):
    """Metrics for one model. `predicted_tunnel` is a boolean array aligned
    with `meta` (from evaluation_meta)."""
    pred = pd.Series(np.asarray(predicted_tunnel, dtype=bool), index=meta.index)
    truth = meta["Label"].eq("tunnel")
    metrics = {}
    test = meta["split"].eq("test")
    if test.any():
        metrics["test_accuracy"] = float((pred[test] == truth[test]).mean())
    # A collapsed run predicts one class for every row of the test set (or,
    # without test rows, of everything scored).
    metrics["collapsed"] = float(pred[test if test.any() else slice(None)].nunique() == 1)
    for name, rows in _groups(meta).items():
        metrics[name] = float(pred[rows].mean())
    for role in PER_TOOL_ROLES + ("held_out_family",):
        rows = meta["role"].eq(role)
        for tool, rate in pred[rows].groupby(meta.loc[rows, "tool"]).mean().items():
            metrics[f"{role}/{tool}"] = float(rate)
    return metrics


def window_size_rates(meta, predicted_tunnel):
    """Rates per group and window-size bucket (queries in the row's window)."""
    pred = pd.Series(np.asarray(predicted_tunnel, dtype=bool), index=meta.index)
    size = meta["window_query_count"]
    rows = []
    for name, mask in _groups(meta).items():
        for low, high, label in WINDOW_SIZE_BUCKETS:
            in_bucket = mask & size.between(low, high)
            rows.append({"group": name, "bucket": label, "rows": int(in_bucket.sum()),
                         "rate": float(pred[in_bucket].mean()) if in_bucket.any() else math.nan})
    return pd.DataFrame(rows)


def summarise(runs):
    """min / avg / max of each metric across runs (list of evaluate() dicts)."""
    frame = pd.DataFrame(runs)
    return pd.DataFrame({"min": frame.min(), "avg": frame.mean(), "max": frame.max()})


# --------------------------------------------------------------- report --

def _pct(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.2f}%"


def _spread(summary, metric):
    if summary is None or metric not in summary.index:
        return "n/a"
    row = summary.loc[metric]
    return f"{_pct(row['avg'])} ({_pct(row['min'])} – {_pct(row['max'])})"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _metric_label(metric):
    if metric == "test_accuracy":
        return "In-distribution test accuracy"
    if metric == "test_benign":
        return "In-distribution test FPR (normal 00055–00061)"
    if metric == "test_tunnel":
        return "In-distribution test recall (tunnel test segments)"
    if metric == "fpr_wildcard_00007_00012":
        return "FPR held-out wildcard 00007–00012"
    if metric == "fpr_wildcard":
        return "FPR held-out wildcard (all held-out captures)"
    return ROLE_GROUPS.get(metric, (metric,))[0]


def write_report(context, path=REPORT_PATH):
    """Render results/zeek_run.md from the notebook's collected results.

    context keys: environment, git, settings, counts (config -> DataFrame
    from dataset_splits.split_counts), runs (name -> result dict with
    summary, window_sizes, epochs, seconds, config, feature_set,
    input_columns, train_rows), lofo (family -> result dict), started_at,
    finished_at.
    """
    runs = context["runs"]
    settings = context["settings"]
    out = ["# Zeek BiLSTM run", ""]
    out += [f"Generated by `notebooks/dns_tunneling_bilstm_model.ipynb` on {context['finished_at']} "
            f"(started {context['started_at']}).", ""]
    git = context["git"]
    out += [f"- Git commit: `{git['commit']}`" + (" (working tree had uncommitted changes)" if git["dirty"] else ""),
            "- Versions: " + ", ".join(f"{k} {v}" for k, v in context["environment"].items() if k != "platform"),
            f"- Platform: {context['environment']['platform']}", ""]

    out += ["## How to reproduce", "",
            "```text",
            "python Ardashes_scripts/process_pcaps.py        # Zeek logs + manifest",
            "python Ardashes_scripts/reorder_and_rezeek.py   # time-sort out-of-order pcaps, re-run Zeek",
            "python notebooks/dataset_splits.py              # splits.csv (committed; only if the data changed)",
            "# then run notebooks/dns_tunneling_bilstm_model.ipynb top to bottom",
            "```", ""]

    out += ["## Settings", "",
            md_table(["setting", "value"], [[k, f"`{v}`"] for k, v in settings.items()]), ""]

    out += ["## Data per split", "",
            "Rows are DNS records. Train and val rows are sampled per (capture, window) with the row caps above; "
            "test and held-out rows are scored in full. Windows are 60 s.", ""]
    for config, counts in context["counts"].items():
        out += [f"**Config {config}** ({ds.CONFIG_DESCRIPTIONS[config]})", ""]
        rows = [[r.split, r.role, r.Label, r.category, f"{r.rows:,}", f"{r.windows:,}", r.captures,
                 "" if pd.isna(r.sampled_rows) else f"{int(r.sampled_rows):,}"] for r in counts.itertuples()]
        out += [md_table(["split", "role", "class", "category", "rows", "windows", "captures", "sampled rows"], rows), ""]

    main = {name: runs[name] for name in ("B", "A") if name in runs}
    out += ["## Configs B (primary) and A (stress test)", "",
            "Each cell is the average over the runs, with min – max in brackets. "
            "Rates are the share of rows classified as tunnel: false positive rates for benign rows, "
            "recall for tunnel rows.", ""]
    metrics = ["test_accuracy", "test_benign", "test_tunnel", "fpr_normal", "fpr_wildcard_00007_00012",
               "fpr_wildcard", "fpr_own_benign", "unseen_tool", "unseen_platform"]
    rows = []
    for metric in metrics:
        if not any(metric in r["summary"].index for r in main.values()):
            continue
        rows.append([_metric_label(metric)] + [_spread(r["summary"], metric) for r in main.values()])
    rows.append(["Collapsed runs (one class on the whole test set)"] +
                [f"{int(r['summary'].loc['collapsed', 'avg'] * len(r['epochs']))}/{len(r['epochs'])}" for r in main.values()])
    rows.append(["Epochs trained (min–max)"] + [f"{min(r['epochs'])}–{max(r['epochs'])}" for r in main.values()])
    rows.append(["Training rows (benign / tunnel)"] +
                [f"{r['train_rows']['benign']:,} / {r['train_rows']['tunnel']:,}" for r in main.values()])
    out += [md_table(["metric"] + [f"config {name}" for name in main], rows), "",
            "In config A all 13 wildcard captures are held out; in config B only 00007–00012. "
            "The 00007–00012 row compares the two on the same captures.", ""]

    for role, title in (("unseen_tool", "Recall per unseen tool (unknownTunnel)"),
                        ("unseen_platform", "Recall per unseen-platform capture (crossEndPoint: iodine on Android)")):
        tools = sorted({m.split("/", 1)[1] for r in main.values() for m in r["summary"].index if m.startswith(role + "/")})
        rows = [[tool] + [_spread(r["summary"], f"{role}/{tool}") for r in main.values()] for tool in tools]
        out += [f"## {title}", "", md_table([role.replace("_", " ")] + [f"config {n}" for n in main], rows), ""]

    out += ["## By window size", "",
            "Average rate over the runs, split by the number of queries in the row's 60 s window "
            "(all domains). Rows = rows in the bucket.", ""]
    for name, result in main.items():
        sizes = result["window_sizes"]
        out += [f"**Config {name}**", ""]
        rows = []
        for group in sizes["group"].unique():
            cells = [_metric_label(group)]
            for _, _, bucket in WINDOW_SIZE_BUCKETS:
                cell = sizes[(sizes["group"] == group) & (sizes["bucket"] == bucket)].iloc[0]
                cells.append("–" if cell["rows"] == 0 else f"{_pct(cell['rate'])} ({cell['rows']:,})")
            rows.append(cells)
        out += [md_table(["group"] + [b for _, _, b in WINDOW_SIZE_BUCKETS], rows), ""]

    ablations = [name for name in runs if name.startswith("B") ]
    if len(ablations) > 1:
        out += ["## Feature-set ablations (config B)", "",
                "Same rows, splits and training setup; only the model's input columns change.", ""]
        metrics = ["test_accuracy", "fpr_normal", "fpr_wildcard", "unseen_tool", "unseen_platform"]
        rows = []
        for name in ablations:
            result = runs[name]
            rows.append([f"`{result['feature_set']}` ({len(result['input_columns'])} inputs)"] +
                        [_spread(result["summary"], m) for m in metrics])
        out += [md_table(["feature set"] + [_metric_label(m) for m in metrics], rows), ""]
        for name in ablations:
            feature_set = runs[name]["feature_set"]
            out += [f"- `{feature_set}`: {', '.join(zfe.FEATURE_SETS[feature_set])}"]
        out += [""]

    lofo = context.get("lofo") or {}
    if lofo:
        out += ["## Leave-one-tunnel-family-out cross-validation", "",
                "Config B benign data. Each fold trains on the train segments of four tunnel families "
                "(val on their val segments) and is scored on every row of the fifth family's captures "
                "plus the held-out normal and wildcard captures.", ""]
        rows = []
        for family, result in lofo.items():
            s = result["summary"]
            rows.append([family, f"{result['family_rows']:,}", _spread(s, "held_out_family"),
                         _spread(s, "fpr_normal"), _spread(s, "fpr_wildcard"),
                         f"{int(s.loc['collapsed', 'avg'] * len(result['epochs']))}/{len(result['epochs'])}"
                         if "collapsed" in s.index else "n/a"])
        out += [md_table(["held-out family", "rows", "recall", "FPR held-out normal", "FPR held-out wildcard",
                          "collapsed"], rows), ""]
        per_tool = []
        for family, result in lofo.items():
            for metric in result["summary"].index:
                if metric.startswith("held_out_family/"):
                    per_tool.append([family, metric.split("/", 1)[1], _spread(result["summary"], metric)])
        out += [md_table(["family", "capture", "recall"], per_tool), ""]

    out += ["## Comparison with the PCAP-based model", "",
            md_table(["PCAP-based notebook (README)", "value"], [[k, v] for k, v in PCAP_BASELINE.items()]), "",
            "The Zeek-based numbers above use a different, larger evaluation: 6 held-out normal chunks, "
            "6 or 13 held-out wildcard captures, all 6 unknownTunnel tools and all 5 crossEndPoint captures, "
            "with every tool's in-distribution test data taken from a later part of its own capture.", ""]

    for note in context.get("notes", []):
        out += [f"- {note}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path
