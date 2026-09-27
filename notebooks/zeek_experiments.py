"""
Experiment helpers for the Zeek DNS tunnelling notebook
=======================================================
load_features       the feature table (every feature, not only the default
                    set) plus, per configuration, the split, role and
                    train/val sampling flag of every row. Cached in
                    notebooks/dataset_zeek/ (gitignored) and rebuilt when the
                    manifest, Zeek logs, splits.csv, caps/seed or extraction
                    code change.
split_masks,        row masks for a configuration, a leave-one-family-out
lofo_masks,         fold, or the final model (every capture is training
final_masks         data; not for evaluation).
evaluate            metrics for one model's predictions on the test and
                    held-out rows (accuracy, false positive rates, recall per
                    unseen tool / platform capture).
window_size_rates   the same rates split by how many queries the row's window
                    holds.
score_models,       evaluate every model of a configuration, in memory or
rescore_saved       from a saved models/zeek_bilstm/... folder.
neutralised_scores  how saved models respond when one input is neutralised,
                    or (forced_feature_scores) set to fixed values;
                    inference only.
save_result,       per-configuration results as JSON in
load_run            results/runs/<run>/<name>.json (committed);
                    save_training_record writes the final model's record
                    (no metrics), which load_run skips.
write_run_report    render one run's section of results/zeek_run.md from
                    results/runs/<run>/run.json: a run next to a reference
                    run, a comparison of leave-one-family-out runs, or the
                    results at the default settings gathered from several
                    runs. Other sections of the file are left untouched.
"""
import hashlib
import json
import math
import os
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


#: Bump when load_features changes what it writes. (The rest of this module,
#: e.g. the report code, doesn't affect the cache, so its source isn't hashed.)
CACHE_VERSION = 2


def _fingerprint(manifest_rows):
    digest = hashlib.sha256()
    for source in (ds.SPLITS_PATH, zfe.__file__, ds.__file__, dns_feature_extraction.__file__):
        digest.update(_normalised_bytes(source))
    digest.update(f"cache version {CACHE_VERSION}".encode())
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

    meta = {
        "fingerprint": fingerprint,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(table),
        "window_seconds": zfe.WINDOW_SECONDS,
        "row_caps": ds.ROW_CAPS,
        "sampling_seed": ds.SAMPLING_SEED,
        "capture_stats": df.attrs["capture_stats"],
    }
    table.attrs["capture_stats"] = meta["capture_stats"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Write-then-rename, so a reader (or a second process building the
        # same cache) never sees a half-written file.
        _write_atomic(FEATURES_CSV, lambda path: table.to_csv(path, index=False, lineterminator="\n"))
        _write_atomic(CACHE_META, lambda path: path.write_text(json.dumps(meta, indent=1), encoding="utf-8"))
    except PermissionError:
        # Another process has the cache open (Windows); keep this table in memory.
        if verbose:
            print(f"Built {len(table):,} rows; {FEATURES_CSV} is in use, so the cache was not updated")
        return table
    if verbose:
        print(f"Built {len(table):,} rows and wrote {FEATURES_CSV}")
    return table


def _write_atomic(path, write):
    """Call write(temporary path), then move it over `path` in one step."""
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


#: Written into the final model's features.json, NOT_FOR_EVALUATION.txt and
#: results record.
FINAL_MODEL_NOTE = (
    "Final model, not for evaluation. It was trained on every GraphTunnel capture, including the unseen tools "
    "(unknownTunnel) and the unseen platform (crossEndPoint), so no GraphTunnel data is left that it hasn't "
    "been trained on, and scoring it there measures nothing. Use it to score new traffic. Its evaluated "
    "counterpart is config B at the same settings (see results/zeek_run.md).")


def final_masks(table, config=ds.PRIMARY_CONFIG):
    """Row masks for the final model, which trains on every capture.

    train: the rows ds.cap_mask keeps in every window of every capture,
    whatever its split or category (unseen tools, the unseen platform and
    gap windows included): the same rows `config` samples wherever it
    samples, and the same caps and seed everywhere else. val: `config`'s
    validation rows, which are training rows here too. Build_model needs a
    validation loss for early stopping and the learning-rate schedule; here
    it can't show overfitting, and nothing about this model is evaluated.
    test and evaluation are the val rows as well, only so the notebook's
    input preparation has rows to transform.
    """
    train = ds.cap_mask(table)
    sampled = table[f"split_{config}"].isin(ds.SAMPLED_SPLITS).fillna(False).astype(bool)
    if not train[sampled].equals(table.loc[sampled, f"sample_{config}"]):
        raise ValueError(f"ds.cap_mask disagrees with the cached sample_{config}; rebuild the feature cache")
    val = split_masks(table, config)["val"]
    return {"train": train, "val": val, "test": val, "evaluation": val}


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
    _require_evaluable(artifacts)
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


def _require_evaluable(artifacts):
    if artifacts["features"].get("not_for_evaluation"):
        raise ValueError(f"{artifacts.get('directory', 'this model')} was trained on every capture "
                         "(not_for_evaluation); scoring it on GraphTunnel data measures nothing")


def _saved_model_inputs(table, directory, config, masks, relabel):
    """(artifacts, evaluation meta, scaled inputs, tunnel class index) for
    scoring the models saved in `directory` on `config`'s evaluation rows."""
    import model_artifacts as ma

    artifacts = ma.load_artifacts(directory)
    _require_evaluable(artifacts)
    masks = {**split_masks(table, config), **(masks or {})}
    meta = evaluation_meta(table, config, masks["evaluation"])
    if relabel is not None:
        meta = relabel(meta)
    X = ma.prepare_model_input(table.loc[masks["evaluation"]], artifacts)
    return artifacts, meta, X, list(artifacts["label_encoder"].classes_).index("tunnel")


def _score_inputs(models, X, meta, tunnel):
    return summarise([evaluate(meta, model.predict(X, batch_size=8192, verbose=0).argmax(axis=1) == tunnel)
                      for model in models])


def neutralised_scores(table, directory, config, features, masks=None, relabel=None):
    """Score saved models with each of `features`, and then all of them,
    replaced by the training mean (0 after scaling).

    Inference only: it shows how much the saved models rely on each feature,
    which retraining without the feature doesn't isolate (the retrained model
    adapts). Returns [{"variant", "neutralised", "summary"}], the first being
    the models as trained.
    """
    artifacts, meta, X, tunnel = _saved_model_inputs(table, directory, config, masks, relabel)
    columns = artifacts["features"]["input_columns"]
    variants = [("as trained", [])] + [(f"without {f}", [f]) for f in features]
    if len(features) > 1:
        variants.append(("without both" if len(features) == 2 else "without all", list(features)))
    out = []
    for label, dropped in variants:
        X_variant = X.copy()
        for feature in dropped:
            X_variant[:, 0, columns.index(feature)] = 0.0
        out.append({"variant": label, "neutralised": dropped,
                    "summary": _score_inputs(artifacts["models"], X_variant, meta, tunnel)})
    return out


def forced_feature_scores(table, directory, config, feature, values, masks=None, relabel=None):
    """Score saved models as recorded, then with `feature` set to each of
    `values` on every scored row (scaled with the saved scaler).

    Inference only, like neutralised_scores: only that one input changes, so
    it shows how the models' decisions depend on the feature's value.
    Returns [{"variant", "value", "summary"}], the first being the rows as
    recorded (value None).
    """
    artifacts, meta, X, tunnel = _saved_model_inputs(table, directory, config, masks, relabel)
    column = artifacts["features"]["input_columns"].index(feature)
    scaler = artifacts["scaler"]
    out = []
    for value in [None] + list(values):
        X_variant = X.copy()
        if value is not None:
            X_variant[:, 0, column] = (value - scaler.mean_[column]) / scaler.scale_[column]
        out.append({"variant": "as recorded" if value is None else f"set to {value}", "value": value,
                    "summary": _score_inputs(artifacts["models"], X_variant, meta, tunnel)})
    return out


def save_diagnostic(run_name, name, payload, runs_dir=RUNS_DIR):
    """Write results/runs/<run>/diagnostics/<name>.json (not a configuration
    result, so load_run ignores it). Summaries are stored as dicts."""
    def encode(value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="index")
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value
    path = Path(runs_dir) / run_name / "diagnostics" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(encode(payload)), indent=1)
    _write_atomic(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    return path


def load_diagnostic(run_name, name, runs_dir=RUNS_DIR):
    return json.loads((Path(runs_dir) / run_name / "diagnostics" / f"{name}.json").read_text(encoding="utf-8"))


def write_feature_neutralisation(run_name, table, models_root, names, features, description,
                                 config=ds.PRIMARY_CONFIG, runs_dir=RUNS_DIR):
    """neutralised_scores for each saved configuration in `names` (folds use
    their leave-one-family-out rows), plus how often each traffic type has
    more than one query type per domain window. Written to
    results/runs/<run>/diagnostics/feature_neutralisation.json."""
    results = {}
    for name in names:
        masks, relabel = lofo_masks(table, name.removeprefix("lofo-"), config) if name.startswith("lofo-") else (None, None)
        results[name] = neutralised_scores(table, Path(models_root) / name, config, features, masks, relabel)
    groups = {"normal (train)": table["category"].eq("normal") & table[f"split_{config}"].eq("train"),
              f"wildcard (train, config {config})": table["category"].eq("wildcard") & table[f"split_{config}"].eq("train")}
    for category in ("tunnel", "unknownTunnel", "crossEndPoint"):
        for tool in sorted(table.loc[table["category"].eq(category), "tool"].unique()):
            groups[tool] = table["category"].eq(category) & table["tool"].eq(tool)
    feature_values = {name: {"rows": int(mask.sum()),
                             "share_above_1": float(100 * (table.loc[mask, "domain_qtype_diversity"] > 1).mean())}
                      for name, mask in groups.items()}
    models = Path(models_root).relative_to(zfe.PROJECT_ROOT).as_posix() + "/<name>/"
    return save_diagnostic(run_name, "feature_neutralisation", {
        "description": description, "models": models, "features": list(features), "results": results,
        "feature_values": feature_values}, runs_dir)


def write_forced_feature_probe(run_name, diagnostic, table, models_root, names, feature, values, title,
                               description, config=ds.PRIMARY_CONFIG, runs_dir=RUNS_DIR):
    """forced_feature_scores for each saved configuration in `names` (folds
    use their leave-one-family-out rows), written to
    results/runs/<run>/diagnostics/<diagnostic>.json with the heading and
    text of its report subsection."""
    results = {}
    for name in names:
        masks, relabel = lofo_masks(table, name.removeprefix("lofo-"), config) if name.startswith("lofo-") else (None, None)
        results[name] = forced_feature_scores(table, Path(models_root) / name, config, feature, values, masks, relabel)
    models = Path(models_root).relative_to(zfe.PROJECT_ROOT).as_posix() + "/<name>/"
    return save_diagnostic(run_name, diagnostic, {
        "title": title, "description": description, "models": models, "feature": feature, "values": list(values),
        "results": results}, runs_dir)


def render_forced_feature_probe(data):
    """Markdown tables for a write_forced_feature_probe diagnostic: one for
    the main configurations, one for the leave-one-family-out folds."""
    variants = [v["variant"] for v in next(iter(data["results"].values()))]
    out = [f"### {data['title']}", "", data["description"], ""]
    for name, entries in data["results"].items():
        if name.startswith("lofo-"):
            continue
        by_variant = {e["variant"]: e["summary"] for e in entries}
        metrics = ["fpr_normal", "fpr_wildcard", "unseen_tool", "unseen_tool/ozymandns", "unseen_tool/cobalstrike",
                   "unseen_platform"]
        rows = [[_metric_label(m) if "/" not in m else f"&nbsp;&nbsp;{m.split('/', 1)[1]}"]
                + [_pct(by_variant[v].get(m, {}).get("avg", math.nan)) for v in variants] for m in metrics]
        out += [f"Config `{name}` models (`{data['models'].replace('<name>', name)}`), average over their 5 models:",
                "", md_table(["metric"] + variants, rows), ""]
    rows = []
    for name, entries in data["results"].items():
        if not name.startswith("lofo-"):
            continue
        by_variant = {e["variant"]: e["summary"] for e in entries}
        for metric, label in (("held_out_family", name.removeprefix("lofo-") + " recall"),
                              ("fpr_normal", "&nbsp;&nbsp;FPR normal"), ("fpr_wildcard", "&nbsp;&nbsp;FPR wildcard")):
            rows.append([label] + [_pct(by_variant[v].get(metric, {}).get("avg", math.nan)) for v in variants])
    if rows:
        out += [f"Fold models (`{data['models']}`), average over their 5 models:", "",
                md_table(["fold"] + variants, rows), ""]
    return out


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


def result_path(run_name, name, runs_dir=RUNS_DIR):
    """results/runs/<run>/<name>.json. The file is written last, after the
    configuration's models are saved, so it marks the configuration done."""
    return Path(runs_dir) / run_name / f"{name}.json"


def save_result(run_name, name, result, runs_dir=RUNS_DIR):
    """Write one configuration's result to results/runs/<run>/<name>.json."""
    keep = ("config", "feature_set", "per_run", "epochs", "histories", "input_columns", "train_rows",
            "val_rows", "settings", "git", "environment", "trained_at", "source", "seconds", "family_rows")
    payload = {"run": run_name, "name": name, **{k: result.get(k) for k in keep if k in result}}
    payload["summary"] = result["summary"].to_dict(orient="index")
    payload["window_sizes"] = result["window_sizes"].to_dict(orient="records")
    path = result_path(run_name, name, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=1)
    _write_atomic(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    return path


def save_training_record(run_name, name, record, runs_dir=RUNS_DIR):
    """Write results/runs/<run>/<name>.json for a model that is not
    evaluated (the final model): training rows, settings and loss
    histories, no metrics, and "not_for_evaluation": true. Like a result
    file it is written last and marks the model as done; load_run skips
    it."""
    payload = {"run": run_name, "name": name, "not_for_evaluation": True, **record}
    path = result_path(run_name, name, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=1)
    _write_atomic(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))
    return path


def load_training_record(run_name, name, runs_dir=RUNS_DIR):
    return json.loads(result_path(run_name, name, runs_dir).read_text(encoding="utf-8"))


#: Per-run report settings (title, description, reference run, layout) in
#: results/runs/<run>/run.json; not a configuration result.
RUN_METADATA = "run.json"


def _parse_result(result):
    result["summary"] = pd.DataFrame.from_dict(result["summary"], orient="index").astype(float)
    result["window_sizes"] = pd.DataFrame(result["window_sizes"])
    return result


def load_result(run_name, name, runs_dir=RUNS_DIR):
    """One configuration's result, as load_run returns it."""
    result = json.loads(result_path(run_name, name, runs_dir).read_text(encoding="utf-8"))
    if result.get("not_for_evaluation"):
        raise ValueError(f"{run_name}/{name} has no evaluation results (not_for_evaluation)")
    return _parse_result(result)


def load_run(run_name, runs_dir=RUNS_DIR):
    """name -> result for every evaluated configuration saved under
    results/runs/<run>/ (not run.json, not the final model's record)."""
    results = {}
    for path in sorted((Path(runs_dir) / run_name).glob("*.json")):
        if path.name == RUN_METADATA:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("not_for_evaluation"):
            continue
        results[result["name"]] = _parse_result(result)
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


#: Rows of a main configuration's metric table, in order.
MAIN_METRICS = ["test_accuracy", "test_benign", "test_tunnel", "fpr_normal", "fpr_wildcard_00007_00012",
                "fpr_wildcard", "fpr_own_benign", "unseen_tool", "unseen_platform"]
PER_CAPTURE_TABLES = (("unseen_tool", "Recall per unseen tool"),
                      ("unseen_platform", "Recall per unseen-platform capture (iodine on Android)"))


def _metric_rows(cols, metrics):
    """One row per metric any of the (label, result) columns has."""
    return [[_metric_label(m)] + [_spread(r["summary"], m) for _, r in cols]
            for m in metrics if any(m in r["summary"].index for _, r in cols)]


def _per_capture_rows(cols, role):
    """One row per capture of `role` (e.g. every unseen tool)."""
    tools = sorted({m.split("/", 1)[1] for _, r in cols for m in r["summary"].index if m.startswith(role + "/")})
    return [[tool] + [_spread(r["summary"], f"{role}/{tool}") for _, r in cols] for tool in tools]


def _window_size_rows(cols):
    """group, bucket, rows, then the average rate of each column; empty
    buckets left out."""
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
    return rows


def upsert_section(path, section_id, markdown, at_top=False):
    """Replace the text between <!-- section:<id>:start/end --> markers in
    `path`, leaving the rest of the file untouched. A new section is
    appended, or with `at_top` inserted after the file's first line (its
    title)."""
    path = Path(path)
    start, end = f"<!-- section:{section_id}:start -->", f"<!-- section:{section_id}:end -->"
    block = f"{start}\n{markdown.strip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end) + r"\n?", re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text)
    elif at_top and text:
        title, _, rest = text.partition("\n")
        text = f"{title}\n\n{block}\n{rest.lstrip(chr(10))}"
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
                "The `best vs epoch 50` column is the relative change in val loss from epoch 50 to the best epoch.", ""]
        rows = []
        for name, table in curves.items():
            ref_epochs = reference.get(name, {}).get("epochs") if reference else None
            change = (table["best_val_loss"] - table["val_loss_at_50"]) / table["val_loss_at_50"]
            rows.append([f"`{name}`",
                         ", ".join(map(str, table["stopped_epoch"])),
                         ", ".join(map(str, table["best_epoch"])),
                         f"{int(table['early_stopped'].sum())}/{len(table)}",
                         ", ".join("n/a" if math.isnan(v) else f"{v:.2e}" for v in table["val_loss_at_50"]),
                         ", ".join(f"{v:.2e}" for v in table["best_val_loss"]),
                         ", ".join("n/a" if math.isnan(v) else f"{100 * v:+.0f}%" for v in change),
                         ", ".join(f"{v:.0e}" for v in table["final_learning_rate"])]
                        + ([", ".join(map(str, ref_epochs)) if ref_epochs else "n/a"] if reference else []))
        out += [md_table(["configuration", "stopped", "best", "early-stopped", "val loss @ 50", "best val loss",
                          "best vs epoch 50", "final LR"] + ([f"{reference_label} epochs"] if reference else []),
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
        rows = _metric_rows(cols, MAIN_METRICS)
        rows.append(["Collapsed runs"] + [_collapsed(r) for _, r in cols])
        rows.append(["Epochs trained"] + [", ".join(map(str, r.get("epochs") or [])) for _, r in cols])
        out += [md_table(["metric"] + [label for label, _ in cols], rows), ""]
        for role, heading in PER_CAPTURE_TABLES:
            out += [f"**{heading}, config {name}**", "",
                    md_table(["capture"] + [label for label, _ in cols], _per_capture_rows(cols, role)), ""]
        out += [f"**By window size, config {name}**: average rate per bucket of queries in the row's 60 s window "
                "(empty buckets left out).", ""]
        out += [md_table(["group", "queries in window", "rows"] + [label for label, _ in cols],
                         _window_size_rows(cols)), ""]

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


def _avg(result, metric):
    summary = result["summary"]
    return summary.loc[metric, "avg"] if metric in summary.index else math.nan


def _pp(value):
    return "n/a" if math.isnan(value) else f"{100 * value:+.2f} pp"


def _row_weighted(results, folds):
    """Share of all held-out-family rows detected, over the given folds."""
    rows = {fold: results[fold].get("family_rows") or 0 for fold in folds}
    total = sum(rows.values())
    return sum(rows[f] * _avg(results[f], "held_out_family") for f in folds) / total if total else math.nan


def render_lofo_comparison(run_name, title, columns, description=(), effects=(), highlight_captures=(),
                           diagnostic=None, probe=None, findings=(), runs_dir=RUNS_DIR):
    """Markdown comparing the leave-one-family-out folds of several runs.

    columns: [{"run", "label", "features", "epochs"}], the last one being
    this run. effects: [{"label", "from", "to"}] (run names), shown as the
    difference in average recall. diagnostic: name of a file in
    results/runs/<run>/diagnostics/ written from neutralised_scores; probe:
    one written by write_forced_feature_probe.
    """
    runs = {c["run"]: load_run(c["run"], runs_dir) for c in columns}
    this = runs[run_name]
    folds = [n for n in this if n.startswith("lofo-")]
    labels = {c["run"]: c["label"] for c in columns}
    head = [f"{c['label']} ({c['features']} features, {c['epochs']} epochs)" for c in columns]
    first = this[folds[0]]
    git = first.get("git") or {}
    out = [f"## {title}", ""] + list(description) + [""]
    out += [f"- Run `{run_name}`: {', '.join(f'`{f}`' for f in folds)}; trained "
            f"{min(this[f].get('trained_at') or '' for f in folds)} – {max(this[f].get('trained_at') or '' for f in folds)}.",
            f"- Git commit: `{git.get('commit')}`" + (" (working tree had uncommitted changes)" if git.get("dirty") else ""),
            "- Versions: " + ", ".join(f"{k} {v}" for k, v in (first.get("environment") or {}).items() if k != "platform"),
            f"- Models: `models/zeek_bilstm/{run_name}/<fold>/`; results: `results/runs/{run_name}/<fold>.json`.", ""]

    rows = []
    for key in ("epoch_cap", "row_caps", "sampling_seed", "window_seconds", "splits_sha256"):
        rows.append([key] + [f"`{runs[c['run']][folds[0]]['settings'].get(key)}`" for c in columns])
    rows.append(["feature set"] + [f"`{runs[c['run']][folds[0]].get('feature_set')}`" for c in columns])
    out += ["### Settings", "", md_table(["setting"] + [c["label"] for c in columns], rows), ""]

    # the 2x2 grid of row-weighted recall
    feature_values = list(dict.fromkeys(c["features"] for c in columns))
    epoch_values = sorted({c["epochs"] for c in columns})
    grid = []
    for features in feature_values:
        cells = []
        for epochs in epoch_values:
            match = [c for c in columns if c["features"] == features and c["epochs"] == epochs]
            cells.append(f"{match[0]['label']}: **{_pct(_row_weighted(runs[match[0]['run']], folds))}**" if match
                         else "not run")
        grid.append([f"{features} features"] + cells)
    total_rows = sum(this[f].get("family_rows") or 0 for f in folds)
    out += ["### Unseen-family recall", "",
            f"Share of all {total_rows:,} held-out-family rows detected, over the five folds (each fold's average "
            "over its 5 models, weighted by the family's rows):", "",
            md_table([""] + [f"{e} epochs" for e in epoch_values], grid), ""]

    header = ["held-out family", "rows"] + head + [e["label"] for e in effects]
    rows = []
    for fold in folds:
        row = [fold.removeprefix("lofo-"), f"{this[fold].get('family_rows', 0):,}"]
        row += [_spread(runs[c["run"]][fold]["summary"], "held_out_family") for c in columns]
        row += [_pp(_avg(runs[e["to"]][fold], "held_out_family") - _avg(runs[e["from"]][fold], "held_out_family"))
                for e in effects]
        rows.append(row)
    row = ["**all five (row-weighted)**", f"{total_rows:,}"] + [_pct(_row_weighted(runs[c["run"]], folds)) for c in columns]
    row += [_pp(_row_weighted(runs[e["to"]], folds) - _row_weighted(runs[e["from"]], folds)) for e in effects]
    rows.append(row)
    out += ["Per family: average recall over the 5 models (min – max). The last columns are differences in the "
            "average, in percentage points.", "", md_table(header, rows), ""]

    header = ["held-out family"] + [f"FPR normal · {c['label']}" for c in columns] + \
             [f"FPR wildcard · {c['label']}" for c in columns]
    rows = [[fold.removeprefix("lofo-")] + [_pct(_avg(runs[c["run"]][fold], "fpr_normal")) for c in columns]
            + [_pct(_avg(runs[c["run"]][fold], "fpr_wildcard")) for c in columns] for fold in folds]
    out += ["False positive rates on the held-out benign captures stay near zero in every run:", "",
            md_table(header, rows), ""]

    captures = []
    for fold in folds:
        for metric in this[fold]["summary"].index:
            if metric.startswith("held_out_family/"):
                captures.append((fold, metric.split("/", 1)[1]))
    order = {c: i for i, c in enumerate(highlight_captures)}
    captures.sort(key=lambda fc: (order.get(fc[1], len(order)), fc[0], fc[1]))
    header = ["capture", "family"] + head + [e["label"] for e in effects]
    rows = []
    for fold, capture in captures:
        metric = f"held_out_family/{capture}"
        name = f"**{capture}**" if capture in order else capture
        rows.append([name, fold.removeprefix("lofo-")] + [_spread(runs[c["run"]][fold]["summary"], metric) for c in columns]
                    + [_pp(_avg(runs[e["to"]][fold], metric) - _avg(runs[e["from"]][fold], metric)) for e in effects])
    out += ["### Per capture", "",
            "Recall on each capture of the held-out family" + (f" ({', '.join(highlight_captures)} first)"
                                                              if highlight_captures else "") + ":", "",
            md_table(header, rows), ""]

    if diagnostic:
        data = load_diagnostic(run_name, diagnostic, runs_dir)
        variants = [v["variant"] for v in next(iter(data["results"].values()))]
        out += ["### Which of the two features?", "", data["description"], ""]
        rows = []
        for name, entries in data["results"].items():
            if not name.startswith("lofo-"):
                continue
            by_variant = {e["variant"]: e["summary"] for e in entries}
            metrics = [("held_out_family", name.removeprefix("lofo-") + " recall")]
            metrics += [(f"held_out_family/{c}", f"&nbsp;&nbsp;{c}") for c in highlight_captures
                        if f"held_out_family/{c}" in by_variant["as trained"]]
            metrics += [("fpr_wildcard", "&nbsp;&nbsp;FPR wildcard")]
            for metric, label in metrics:
                rows.append([label] + [_pct(by_variant[v].get(metric, {}).get("avg", math.nan)) for v in variants])
        if rows:
            out += [f"Run 1 fold models (`{data['models']}`), average over their 5 models:", "",
                    md_table(["fold / capture"] + variants, rows), ""]
        for name, entries in data["results"].items():
            if name.startswith("lofo-"):
                continue
            by_variant = {e["variant"]: e["summary"] for e in entries}
            metric_names = ["unseen_tool", "unseen_tool/ozymandns", "unseen_tool/cobalstrike", "unseen_platform",
                            "fpr_normal", "fpr_wildcard"]
            rows = [[_metric_label(m) if "/" not in m else f"&nbsp;&nbsp;{m.split('/', 1)[1]}"]
                    + [_pct(by_variant[v].get(m, {}).get("avg", math.nan)) for v in variants] for m in metric_names]
            out += [f"Run 1 config `{name}` models, average over their 5 models:", "",
                    md_table(["metric"] + variants, rows), ""]
        if data.get("feature_values"):
            out += ["Share of rows whose domain has more than one query type in its 60 s window "
                    "(`domain_qtype_diversity` > 1):", "",
                    md_table(["traffic", "rows", "> 1 query type"],
                             [[k, f"{v['rows']:,}", f"{v['share_above_1']:.2f}%"] for k, v in data["feature_values"].items()]),
                    ""]
    if probe:
        out += render_forced_feature_probe(load_diagnostic(run_name, probe, runs_dir))

    curves = {fold: training_curves(this[fold].get("histories"), this[fold]["settings"]["epoch_cap"]) for fold in folds
              if this[fold].get("histories")}
    if curves:
        rows = [[f"`{fold}`", ", ".join(map(str, c["stopped_epoch"])), ", ".join(map(str, c["best_epoch"])),
                 ", ".join(f"{v:.2e}" for v in c["best_val_loss"])] for fold, c in curves.items()]
        out += ["### Training length", "",
                md_table(["fold", "stopped", "best", "best val loss"], rows), ""]

    if findings:
        out += ["### Findings", ""] + list(findings) + [""]
    return "\n".join(out)


def splits_sha256(path=None):
    """sha256 of splits.csv with LF line endings, as the notebook records it."""
    return hashlib.sha256(_normalised_bytes(path or ds.SPLITS_PATH)).hexdigest()


def check_default_settings(results):
    """Problems (a list of strings) with gathering `results` (name ->
    result) as the results at the default settings: every one must share
    config B's epoch cap and hyperparameters and have today's row caps,
    sampling seed, window and splits.csv; main configurations and folds must
    use the default features, and each ablation its own subset of them."""
    problems = []
    base = results["B"]["settings"]
    expected = {"epoch_cap": base.get("epoch_cap"), "hyperparameters": base.get("hyperparameters"),
                "row_caps": ds.ROW_CAPS, "sampling_seed": ds.SAMPLING_SEED, "window_seconds": zfe.WINDOW_SECONDS,
                "splits_sha256": splits_sha256()}
    for name, result in results.items():
        settings = result["settings"]
        for key, value in expected.items():
            if settings.get(key) != value:
                problems.append(f"{name}: {key} is {settings.get(key)!r}, expected {value!r}")
        features = settings.get("feature_columns")
        if name.startswith("B-"):
            feature_set = name.removeprefix("B-")
            if features != zfe.FEATURE_SETS.get(feature_set) or not set(features) <= set(zfe.FEATURE_COLUMNS):
                problems.append(f"{name}: features are not {feature_set!r}, a subset of the default")
        elif features != zfe.FEATURE_COLUMNS:
            problems.append(f"{name}: features are not the default set")
    return problems


def _window_size_grid(result):
    """group x bucket table of 'rate (rows)' for one configuration."""
    sizes = result["window_sizes"]
    buckets = [label for _, _, label in WINDOW_SIZE_BUCKETS]
    rows = []
    for group in dict.fromkeys(sizes["group"]):
        cells = []
        for bucket in buckets:
            cell = sizes[(sizes["group"] == group) & (sizes["bucket"] == bucket)]
            empty = cell.empty or cell["rows"].iloc[0] == 0
            cells.append("–" if empty else f"{_pct(cell['rate'].iloc[0])} ({int(cell['rows'].iloc[0]):,})")
        rows.append([_metric_label(group)] + cells)
    return md_table(["group"] + buckets, rows)


def _rows_text(rows):
    return " / ".join(f"{rows.get(label, 0):,}" for label in ("benign", "tunnel")) if rows else "n/a"


def render_default_results(title, configurations, description=(), final=None, findings=(), runs_dir=RUNS_DIR):
    """Markdown for the results at the default settings, gathered from the
    runs that trained them.

    configurations: [{"name", "run", "result", "note"?}]. name is how the
    section refers to it ("B", "A", "B-<feature set>", "lofo-<family>");
    run and result locate results/runs/<run>/<result>.json. final:
    {"run", "result"} of the final model's training record. Raises
    ValueError when a configuration or the final model doesn't have the
    default settings (check_default_settings), so the section can't mix
    settings.
    """
    sources = {c["name"]: c for c in configurations}
    results = {c["name"]: load_result(c["run"], c["result"], runs_dir) for c in configurations}
    record = load_training_record(final["run"], final["result"], runs_dir) if final else None
    problems = check_default_settings({**results, **({"final": record} if record else {})})
    if problems:
        raise ValueError("Not all at the default settings:\n  " + "\n  ".join(problems))
    b = results["B"]
    settings = b["settings"]
    architecture = {k: v for k, v in settings["hyperparameters"].items() if k not in ("epochs", "early_stopping")}
    model_seed = next((r["settings"]["model_seed"] for r in results.values() if "model_seed" in r["settings"]), 0)
    out = [f"## {title}", ""] + list(description) + [""]

    out += ["### Default settings", "", md_table(["setting", "value"], [
        ["feature set", f"`{b['feature_set']}`: {len(zfe.FEATURE_COLUMNS)} features (every feature except "
                        f"{' and '.join(f'`{c}`' for c in zfe.FEATURE_GROUPS['artefact_suspect'])}), "
                        f"{len(b['input_columns'])} model inputs after one-hot encoding"],
        ["epoch cap", f"{settings['epoch_cap']}, with early stopping on the validation loss (patience 5, best weights "
                      "restored); the learning rate is halved after 2 epochs without improvement, down to 1e-05"],
        ["hyperparameters", f"`{architecture}`"],
        ["class weights", "sklearn `balanced`, computed on the training rows"],
        ["decision rule", "argmax of the softmax output (benign / tunnel)"],
        ["models per configuration", f"{len(b['per_run'])}, each trained from scratch; tables give the average over "
                                     "them with min – max in brackets"],
        ["window", f"{settings['window_seconds']} s per domain aggregate"],
        ["train/val row caps per (capture, window)", f"`{settings['row_caps']}`"],
        ["sampling seed · model seed", f"{settings['sampling_seed']} · {model_seed}"],
        ["splits.csv sha256", f"`{settings['splits_sha256']}`"],
    ]), "", "Features: " + ", ".join(zfe.FEATURE_COLUMNS) + ".", ""]

    rows = []
    for name, result in results.items():
        git = result.get("git") or {}
        rows.append([f"`{name}`", f"`{sources[name]['run']}/{sources[name]['result']}`",
                     (result.get("trained_at") or "")[:10],
                     f"`{(git.get('commit') or '')[:7]}`" + (" (uncommitted changes)" if git.get("dirty") else ""),
                     ", ".join(map(str, result.get("epochs") or [])), sources[name].get("note", "")])
    out += ["### Where each result comes from", "",
            md_table(["configuration", "result file (results/runs/…)", "trained", "commit", "epochs trained", "note"],
                     rows), ""]

    mains = [(f"config {n}", results[n]) for n in MAIN_CONFIGS if n in results]
    out += ["### Configs B and A", "",
            "Config B (primary) has wildcard hard negatives in training and validation; config A (stress test) has "
            "none, and holds out all 13 wildcard captures instead of 00007–00012. The 00007–00012 row compares the two "
            "on the same captures. Rates are the share of rows classified as tunnel: false positive rates for benign "
            "rows, recall for tunnel rows.", ""]
    rows = _metric_rows(mains, MAIN_METRICS)
    rows.append(["Collapsed runs (one class on the whole test set)"] + [_collapsed(r) for _, r in mains])
    rows.append(["Training rows (benign / tunnel)"] + [_rows_text(r.get("train_rows")) for _, r in mains])
    out += [md_table(["metric"] + [label for label, _ in mains], rows), ""]
    for role, heading in PER_CAPTURE_TABLES:
        out += [f"**{heading}**", "", md_table(["capture"] + [label for label, _ in mains],
                                                _per_capture_rows(mains, role)), ""]
    out += ["**By window size**: rate by the number of queries in the row's 60 s window (all domains), rows in "
            "brackets.", ""]
    for label, result in mains:
        out += [f"*{label[0].upper()}{label[1:]}*", "", _window_size_grid(result), ""]

    ablations = [n for n in results if n.startswith("B-")]
    if ablations:
        metrics = ["test_accuracy", "fpr_normal", "fpr_wildcard", "unseen_tool", "unseen_platform"]
        rows = []
        for name in ["B"] + ablations:
            r = results[name]
            label = "`all` (the default)" if name == "B" else f"`{name.removeprefix('B-')}`"
            rows.append([label, f"{len(r['settings']['feature_columns'])} ({len(r['input_columns'])})"]
                        + [_spread(r["summary"], m) for m in metrics])
        out += ["### Feature-set ablations (config B)", "",
                "Same rows, splits and training; only the model inputs change. Every set is a subset of the default.",
                "", md_table(["feature set", "features (inputs)"] + [_metric_label(m) for m in metrics], rows), ""]
        out += [f"- `{name.removeprefix('B-')}`: {', '.join(results[name]['settings']['feature_columns'])}"
                for name in ablations] + [""]

    folds = [n for n in results if n.startswith("lofo-")]
    if folds:
        rows = [[fold.removeprefix("lofo-"), f"{results[fold].get('family_rows', 0):,}"]
                + [_spread(results[fold]["summary"], m) for m in ("held_out_family", "fpr_normal", "fpr_wildcard")]
                + [_collapsed(results[fold])] for fold in folds]
        total = sum(results[f].get("family_rows") or 0 for f in folds)
        rows.append(["**all five (row-weighted)**", f"{total:,}", f"**{_pct(_row_weighted(results, folds))}**",
                     "", "", ""])
        out += ["### Leave-one-tunnel-family-out cross-validation", "",
                "Config B's benign data; each fold trains on four tunnel families and is scored on every row of the "
                "fifth family's captures and on the held-out normal and wildcard captures.", "",
                md_table(["held-out family", "rows", "recall", "FPR held-out normal", "FPR held-out wildcard",
                          "collapsed"], rows), ""]
        rows = [[fold.removeprefix("lofo-"), metric.split("/", 1)[1], _spread(results[fold]["summary"], metric)]
                for fold in folds for metric in sorted(results[fold]["summary"].index)
                if metric.startswith("held_out_family/")]
        out += [md_table(["family", "capture", "recall"], rows), ""]

    if record:
        by_category = record.get("train_rows_by_category") or {}
        captures = record.get("captures_by_category") or {}
        rows = [[category, f"{captures.get(category, 0)}", f"{count:,}"] for category, count in by_category.items()]
        rows.append(["**total**", f"**{sum(captures.values())}**", f"**{sum(by_category.values()):,}**"])
        git = record.get("git") or {}
        n_models = len(record.get("epochs") or [])
        model_files = "`model_1.keras`" + (f" … `model_{n_models}.keras`" if n_models > 1 else "")
        out += ["### Final model (not for evaluation)", "",
                "Trained on every GraphTunnel capture, including the unseen tools (unknownTunnel) and the unseen "
                "platform (crossEndPoint), so no GraphTunnel data is left that it hasn't been trained on, and it has "
                "no results here. It is the model for scoring new traffic; config B above is its evaluated "
                "counterpart.", "",
                f"- Saved to `{record.get('models_dir')}`: {model_files}, `scaler.joblib`, `label_encoder.joblib`, "
                "`features.json` (`\"not_for_evaluation\": true`) and `NOT_FOR_EVALUATION.txt`. The scoring "
                "functions in `zeek_experiments` refuse to score it.",
                f"- Trained {(record.get('trained_at') or '')[:10]} at commit `{(git.get('commit') or '')[:7]}`"
                + (" (uncommitted changes)" if git.get("dirty") else "") + " with the settings above; epochs "
                f"trained: {', '.join(map(str, record.get('epochs') or []))}.",
                f"- Training rows: every capture, capped per window like the other configurations "
                f"(benign / tunnel {_rows_text(record.get('train_rows'))}). Early stopping and the learning-rate "
                f"schedule monitor config B's validation rows ({_rows_text(record.get('val_rows'))}), which are "
                "training rows here too.", "",
                md_table(["category", "captures", "training rows"], rows), ""]

    if findings:
        out += ["### Findings", ""] + list(findings) + [""]
    return "\n".join(out)


def write_run_report(run_name, report_path=REPORT_PATH, runs_dir=RUNS_DIR):
    """Write a run's section of results/zeek_run.md using the report settings
    in results/runs/<run>/run.json: layout "single" (default; title,
    description, reference run and labels), "lofo_comparison" (see
    render_lofo_comparison) or "defaults" (render_default_results; written
    near the top of the file, as section spec["section"])."""
    spec = json.loads((Path(runs_dir) / run_name / RUN_METADATA).read_text(encoding="utf-8"))
    layout = spec.pop("layout", "single")
    report_path = Path(report_path)
    if layout == "single":
        return write_run_section(run_name, report_path=report_path, runs_dir=runs_dir, **spec)
    if layout == "lofo_comparison":
        return upsert_section(report_path, run_name, render_lofo_comparison(run_name, runs_dir=runs_dir, **spec))
    if layout == "defaults":
        section = spec.pop("section", run_name)
        return upsert_section(report_path, section, render_default_results(runs_dir=runs_dir, **spec), at_top=True)
    raise ValueError(f"Unknown report layout {layout!r} in {run_name}/{RUN_METADATA}")
