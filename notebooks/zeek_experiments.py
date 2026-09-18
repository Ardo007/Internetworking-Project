"""
Experiment helpers for the Zeek DNS tunnelling notebook
=======================================================
load_features       the feature table (every feature, not only the default
                    set) plus, per configuration, the split, role and
                    train/val sampling flag of every row. Cached in
                    notebooks/dataset_zeek/ (gitignored) and rebuilt when the
                    manifest, Zeek logs, splits.csv, caps/seed or extraction
                    code change.
split_masks,        row masks for a configuration or a leave-one-family-out
lofo_masks          fold.
evaluate            metrics for one model's predictions on the test and
                    held-out rows (accuracy, false positive rates, recall per
                    unseen tool / platform capture).
window_size_rates   the same rates split by how many queries the row's window
                    holds.
score_models,       evaluate every model of a configuration, in memory or
rescore_saved       from a saved models/zeek_bilstm/... folder.
save_result,        per-configuration results as JSON in
load_run            results/runs/<run>/<name>.json (committed).
write_run_section   render one run's section of results/zeek_run.md, next
                    to a reference run, with loss-curve figures in
                    results/figures/. Other sections of the file are left
                    untouched.
"""
import hashlib
import json
import math
import re
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
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"

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
MAIN_CONFIGS = ("B", "A")


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

    Holds every feature (zfe.ALL_FEATURE_COLUMNS), filled as in to_ml_frame,
    so any feature set can be trained from it. sample_<config> marks the
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
    table = zfe.to_ml_frame(df, features=zfe.ALL_FEATURE_COLUMNS + ANALYSIS_COLUMNS, extra_cols=CACHE_BOOKKEEPING)
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


def lofo_masks(table, family, config=ds.PRIMARY_CONFIG):
    """Masks and relabel function for a leave-one-tunnel-family-out fold.

    Trains on `config`'s train/val rows minus every row of `family`'s
    captures, and scores all rows of those captures (role
    "held_out_family") plus the held-out normal and wildcard captures.
    """
    masks = split_masks(table, config)
    in_family = table["category"].eq("tunnel") & table["family"].eq(family)
    if not in_family.any():
        raise ValueError(f"No tunnel captures of family {family!r}")
    benign_heldout = table[f"role_{config}"].isin(["fpr_normal", "fpr_wildcard"])
    fold = {
        "train": masks["train"] & ~in_family,
        "val": masks["val"] & ~in_family,
        "test": in_family,
        "evaluation": in_family | benign_heldout,
    }

    def relabel(meta):
        meta = meta.copy()
        meta["split"] = "heldout"
        meta.loc[in_family.loc[meta.index], "role"] = "held_out_family"
        return meta

    return fold, relabel


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


def score_models(models, X_eval, meta, label_encoder, batch_size=4096):
    """evaluate() and window_size_rates() for every model.

    Returns (per-model metric dicts, window-size rates averaged over models).
    """
    tunnel = list(label_encoder.classes_).index("tunnel")
    per_run, sizes = [], []
    for model in models:
        predicted = model.predict(X_eval, batch_size=batch_size, verbose=0).argmax(axis=1) == tunnel
        per_run.append(evaluate(meta, predicted))
        sizes.append(window_size_rates(meta, predicted))
    window_sizes = (pd.concat(sizes).groupby(["group", "bucket"], sort=False)
                    .agg(rows=("rows", "first"), rate=("rate", "mean")).reset_index())
    return per_run, window_sizes


def rescore_saved(table, directory, config, masks=None, relabel=None):
    """Score the models saved in `directory` (model_artifacts.save_artifacts)
    on `config`'s evaluation rows, or on `masks["evaluation"]`.

    Returns a result dict like the notebook's run_configuration, without
    loss histories (they aren't saved with the models).
    """
    import model_artifacts as ma

    artifacts = ma.load_artifacts(directory)
    spec = artifacts["features"]
    masks = {**split_masks(table, config), **(masks or {})}
    meta = evaluation_meta(table, config, masks["evaluation"])
    if relabel is not None:
        meta = relabel(meta)
    X_eval = ma.prepare_model_input(table.loc[masks["evaluation"]], artifacts)
    per_run, window_sizes = score_models(artifacts["models"], X_eval, meta, artifacts["label_encoder"])
    return {
        "name": spec.get("name", Path(directory).name),
        "config": config,
        "feature_set": spec.get("feature_set"),
        "summary": summarise(per_run),
        "per_run": per_run,
        "window_sizes": window_sizes,
        "epochs": spec.get("epochs_trained"),
        "histories": None,
        "input_columns": spec["input_columns"],
        "train_rows": spec.get("training_rows"),
        "val_rows": spec.get("validation_rows"),
        "settings": {"epoch_cap": spec.get("hyperparameters", {}).get("epochs"),
                     "hyperparameters": spec.get("hyperparameters"),
                     "row_caps": spec.get("row_caps"), "sampling_seed": spec.get("sampling_seed"),
                     "window_seconds": spec.get("window_seconds"),
                     "feature_columns": spec.get("feature_columns"),
                     "splits_sha256": spec.get("splits_sha256")},
        "git": spec.get("git"),
        "environment": spec.get("versions"),
        "trained_at": spec.get("trained_at"),
        "source": "re-scored from the saved models in "
                  f"`{Path(directory).parent.relative_to(zfe.PROJECT_ROOT).as_posix()}/<name>/`",
    }


# ------------------------------------------------------ training curves --

def training_curves(histories, epoch_cap, compare_at=50):
    """Per model: stopped epoch, best epoch (lowest val_loss), whether early
    stopping fired, val_loss at `compare_at` and at the best epoch, and the
    final learning rate."""
    rows = []
    for number, history in enumerate(histories or [], start=1):
        val = history["val_loss"]
        best = int(np.argmin(val))
        rows.append({
            "model": number,
            "stopped_epoch": len(val),
            "best_epoch": best + 1,
            "early_stopped": len(val) < epoch_cap,
            f"val_loss_at_{compare_at}": val[compare_at - 1] if len(val) >= compare_at else math.nan,
            "best_val_loss": val[best],
            "last_val_loss": val[-1],
            "train_loss_at_best": history["loss"][best],
            "final_learning_rate": (history.get("learning_rate") or [math.nan])[-1],
        })
    return pd.DataFrame(rows)


def plot_loss_curves(result, path, title, mark_epoch=50):
    """One panel per model: training and validation loss per epoch, with the
    old epoch cap and the best epoch marked."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=False)
    histories = result["histories"]
    fig, axes = plt.subplots(1, len(histories), figsize=(3.6 * len(histories), 3.0), sharey=True, squeeze=False)
    for ax, (number, history) in zip(axes[0], enumerate(histories, start=1)):
        epochs = np.arange(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"], label="train loss", color="#4374B3")
        ax.plot(epochs, history["val_loss"], label="val loss", color="#E8710A")
        best = int(np.argmin(history["val_loss"])) + 1
        ax.axvline(mark_epoch, color="grey", linestyle="--", linewidth=1, label=f"epoch {mark_epoch}")
        ax.plot([best], [history["val_loss"][best - 1]], "o", color="#E8710A", markersize=5, label="best val loss")
        ax.set_yscale("log")
        ax.set_title(f"model {number}: stopped at {len(epochs)}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("loss (log scale)", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ------------------------------------------------------------ results --

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_result(run_name, name, result, runs_dir=RUNS_DIR):
    """Write one configuration's result to results/runs/<run>/<name>.json."""
    keep = ("config", "feature_set", "per_run", "epochs", "histories", "input_columns", "train_rows",
            "val_rows", "settings", "git", "environment", "trained_at", "source", "seconds", "family_rows")
    payload = {"run": run_name, "name": name, **{k: result.get(k) for k in keep if k in result}}
    payload["summary"] = result["summary"].to_dict(orient="index")
    payload["window_sizes"] = result["window_sizes"].to_dict(orient="records")
    path = Path(runs_dir) / run_name / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1), encoding="utf-8")
    return path


def load_run(run_name, runs_dir=RUNS_DIR):
    """name -> result for every configuration saved under results/runs/<run>/."""
    results = {}
    for path in sorted((Path(runs_dir) / run_name).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = pd.DataFrame.from_dict(result["summary"], orient="index")
        result["summary"] = summary.astype(float)
        result["window_sizes"] = pd.DataFrame(result["window_sizes"])
        results[result["name"]] = result
    if not results:
        raise FileNotFoundError(f"No results in {Path(runs_dir) / run_name}")
    return results


# --------------------------------------------------------------- report --

def _pct(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.2f}%"


def _spread(summary, metric):
    if summary is None or metric not in summary.index:
        return "n/a"
    row = summary.loc[metric]
    return f"{_pct(row['avg'])} ({_pct(row['min'])} – {_pct(row['max'])})"


def _collapsed(result):
    runs = result.get("per_run") or []
    return f"{sum(int(r.get('collapsed', 0)) for r in runs)}/{len(runs)}" if runs else "n/a"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _metric_label(metric):
    labels = {
        "test_accuracy": "In-distribution test accuracy",
        "test_benign": "In-distribution test FPR (normal 00055–00061)",
        "test_tunnel": "In-distribution test recall (tunnel test segments)",
        "fpr_wildcard_00007_00012": "FPR held-out wildcard 00007–00012",
        "fpr_wildcard": "FPR held-out wildcard (all held-out captures)",
        "collapsed": "Collapsed runs (share of runs)",
    }
    return labels.get(metric) or ROLE_GROUPS.get(metric, (metric,))[0]


def upsert_section(path, section_id, markdown):
    """Replace the text between <!-- section:<id>:start/end --> markers in
    `path` (or append it), leaving the rest of the file untouched."""
    path = Path(path)
    start, end = f"<!-- section:{section_id}:start -->", f"<!-- section:{section_id}:end -->"
    block = f"{start}\n{markdown.strip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text)
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _columns(name, run, reference, run_label, reference_label, extra_reference):
    columns = []
    if name in reference:
        columns.append((f"{reference_label} {name}", reference[name]))
    for extra in extra_reference.get(name, ()):
        if extra in reference:
            columns.append((f"{reference_label} {extra}", reference[extra]))
    columns.append((f"{run_label} {name}", run[name]))
    return columns


def render_run_section(run_name, title, description=(), reference_name=None, run_label=None,
                       reference_label=None, extra_reference=None, figures_dir=FIGURES_DIR,
                       runs_dir=RUNS_DIR, figure_link_base="figures"):
    """Markdown for one run, with reference-run columns next to it."""
    run = load_run(run_name, runs_dir)
    reference = load_run(reference_name, runs_dir) if reference_name else {}
    run_label = run_label or run_name
    reference_label = reference_label or reference_name or ""
    extra_reference = extra_reference or {}
    first = next(iter(run.values()))
    out = [f"## {title}", ""]
    out += list(description) + [""] if description else []
    git = first.get("git") or {}
    out += [f"- Run `{run_name}`: configurations {', '.join(f'`{n}`' for n in run)}; "
            f"trained {min(r.get('trained_at') or '' for r in run.values())} – "
            f"{max(r.get('trained_at') or '' for r in run.values())}.",
            f"- Git commit: `{git.get('commit')}`" + (" (working tree had uncommitted changes)" if git.get("dirty") else ""),
            "- Versions: " + ", ".join(f"{k} {v}" for k, v in (first.get("environment") or {}).items() if k != "platform"),
            f"- Models: `models/zeek_bilstm/{run_name}/<name>/`; per-configuration results: "
            f"`results/runs/{run_name}/<name>.json`."]
    if reference:
        ref_first = next(iter(reference.values()))
        out += [f"- Reference columns ({reference_label}): {ref_first.get('source') or 'trained'}."]
    out += [""]

    # settings that differ
    ref_settings = (next(iter(reference.values())).get("settings") or {}) if reference else {}
    rows = []
    for key in ("epoch_cap", "row_caps", "sampling_seed", "window_seconds", "splits_sha256"):
        value = first["settings"].get(key)
        rows.append([key] + ([f"`{ref_settings.get(key)}`"] if reference else []) + [f"`{value}`"])
    rows.append(["feature set"] + ([f"`{next(iter(reference.values())).get('feature_set')}`"] if reference else [])
                + [f"`{first.get('feature_set')}` ({len(first['settings'].get('feature_columns') or [])} features)"])
    out += ["### Settings", "",
            md_table(["setting"] + ([reference_label] if reference else []) + [run_label], rows), ""]

    # training length
    curves = {name: training_curves(r.get("histories"), r["settings"]["epoch_cap"]) for name, r in run.items()
              if r.get("histories")}
    if curves:
        cap = first["settings"]["epoch_cap"]
        out += ["### Training length", "",
                f"Epoch cap {cap} (early stopping on val loss, patience 5, best weights restored; "
                "learning rate halved after 2 epochs without improvement). "
                "`stopped` is the last epoch trained, `best` the epoch with the lowest val loss (the weights kept). "
                "The last column shows how much the best val loss improved on the value at epoch 50.", ""]
        rows = []
        for name, table in curves.items():
            ref_epochs = reference.get(name, {}).get("epochs") if reference else None
            gain = table["val_loss_at_50"] - table["best_val_loss"]
            rows.append([f"`{name}`",
                         ", ".join(map(str, table["stopped_epoch"])),
                         ", ".join(map(str, table["best_epoch"])),
                         f"{int(table['early_stopped'].sum())}/{len(table)}",
                         ", ".join(f"{v:.4f}" for v in table["val_loss_at_50"]),
                         ", ".join(f"{v:.4f}" for v in table["best_val_loss"]),
                         ", ".join(f"{v:.1e}" for v in table["final_learning_rate"]),
                         ", ".join("n/a" if math.isnan(v) else f"{v:.4f}" for v in gain)]
                        + ([", ".join(map(str, ref_epochs)) if ref_epochs else "n/a"] if reference else []))
        out += [md_table(["configuration", "stopped", "best", "early-stopped", "val loss @ 50", "best val loss",
                          "final LR", "val loss gain after 50"] + ([f"{reference_label} epochs"] if reference else []),
                         rows), ""]
        for name, result in run.items():
            if not result.get("histories"):
                continue
            figure = Path(figures_dir) / f"{run_name}_{name}_loss.png"
            plot_loss_curves(result, figure, f"{run_label} {name}: training and validation loss")
            out += [f"![{run_label} {name} loss curves]({figure_link_base}/{figure.name})", ""]

    # main configurations
    mains = [n for n in MAIN_CONFIGS if n in run]
    for name in mains:
        cols = _columns(name, run, reference, run_label, reference_label, extra_reference)
        out += [f"### Config {name}", "",
                "Average over the runs, min – max in brackets. Rates are the share of rows classified as tunnel.", ""]
        metrics = ["test_accuracy", "test_benign", "test_tunnel", "fpr_normal", "fpr_wildcard_00007_00012",
                   "fpr_wildcard", "fpr_own_benign", "unseen_tool", "unseen_platform"]
        rows = [[_metric_label(m)] + [_spread(r["summary"], m) for _, r in cols]
                for m in metrics if any(m in r["summary"].index for _, r in cols)]
        rows.append(["Collapsed runs"] + [_collapsed(r) for _, r in cols])
        rows.append(["Epochs trained"] + [", ".join(map(str, r.get("epochs") or [])) for _, r in cols])
        out += [md_table(["metric"] + [label for label, _ in cols], rows), ""]
        for role, heading in (("unseen_tool", "Recall per unseen tool"),
                              ("unseen_platform", "Recall per unseen-platform capture (iodine on Android)")):
            tools = sorted({m.split("/", 1)[1] for _, r in cols for m in r["summary"].index if m.startswith(role + "/")})
            rows = [[tool] + [_spread(r["summary"], f"{role}/{tool}") for _, r in cols] for tool in tools]
            out += [f"**{heading}, config {name}**", "", md_table(["capture"] + [label for label, _ in cols], rows), ""]
        out += [f"**By window size, config {name}**: average rate per bucket of queries in the row's 60 s window "
                "(empty buckets left out).", ""]
        groups = list(dict.fromkeys(g for _, r in cols for g in r["window_sizes"]["group"]))
        rows = []
        for group in groups:
            for _, _, bucket in WINDOW_SIZE_BUCKETS:
                cells, bucket_rows = [], 0
                for _, r in cols:
                    sizes = r["window_sizes"]
                    cell = sizes[(sizes["group"] == group) & (sizes["bucket"] == bucket)]
                    empty = cell.empty or cell["rows"].iloc[0] == 0
                    bucket_rows = bucket_rows or (0 if empty else int(cell["rows"].iloc[0]))
                    cells.append("–" if empty else _pct(cell["rate"].iloc[0]))
                if bucket_rows:
                    rows.append([_metric_label(group), bucket, f"{bucket_rows:,}"] + cells)
        out += [md_table(["group", "queries in window", "rows"] + [label for label, _ in cols], rows), ""]

    # ablations
    ablations = [n for n in run if n.startswith("B-")]
    if ablations:
        metrics = ["test_accuracy", "fpr_normal", "fpr_wildcard", "unseen_tool", "unseen_platform"]
        rows = [[f"`{run[n]['feature_set']}` ({len(run[n]['input_columns'])} inputs)"]
                + [_spread(run[n]["summary"], m) for m in metrics] for n in ["B"] + ablations if n in run]
        out += ["### Feature-set ablations (config B)", "",
                md_table(["feature set"] + [_metric_label(m) for m in metrics], rows), ""]

    # leave-one-family-out
    folds = [n for n in run if n.startswith("lofo-")]
    if folds:
        labels = ([reference_label] if any(f in reference for f in folds) else []) + [run_label]
        out += ["### Leave-one-tunnel-family-out cross-validation", "",
                "Config B benign data; each fold trains on four tunnel families and is scored on every row of the "
                "fifth family's captures plus the held-out normal and wildcard captures.", ""]
        header = ["held-out family", "rows"]
        for metric in ("recall", "FPR normal", "FPR wildcard"):
            header += [f"{metric} · {label}" for label in labels]
        rows = []
        for fold in folds:
            pair = ([reference[fold]] if fold in reference else ([None] if len(labels) > 1 else [])) + [run[fold]]
            row = [fold.removeprefix("lofo-"), f"{run[fold].get('family_rows', 0):,}"]
            for metric in ("held_out_family", "fpr_normal", "fpr_wildcard"):
                row += [_spread(r["summary"], metric) if r else "n/a" for r in pair]
            rows.append(row)
        out += [md_table(header, rows), ""]
        rows = []
        for fold in folds:
            pair = ([reference.get(fold)] if len(labels) > 1 else []) + [run[fold]]
            tools = sorted(m.split("/", 1)[1] for m in run[fold]["summary"].index if m.startswith("held_out_family/"))
            for tool in tools:
                rows.append([fold.removeprefix("lofo-"), tool]
                            + [_spread(r["summary"], f"held_out_family/{tool}") if r else "n/a" for r in pair])
        out += [md_table(["family", "capture"] + [f"recall · {label}" for label in labels], rows), ""]
    return "\n".join(out)


def write_run_section(run_name, title, report_path=REPORT_PATH, **kwargs):
    """Render a run's section and write it into `report_path`."""
    report_path = Path(report_path)
    markdown = render_run_section(run_name, title, figures_dir=report_path.parent / "figures", **kwargs)
    return upsert_section(report_path, run_name, markdown)
