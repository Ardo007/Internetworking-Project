import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

import zeek_feature_extraction as zfe

SPLITS_PATH = zfe.PROJECT_ROOT / "Data" / "processed" / "GraphTunnel" / "splits.csv"

CONFIGS = ("A", "B")
PRIMARY_CONFIG = "B"
CONFIG_DESCRIPTIONS = {
    "A": "stress test: no wildcard in training, all wildcard held out",
    "B": "primary: wildcard hard negatives in training",
}
SPLIT_NAMES = ("train", "val", "test", "heldout")

NORMAL_BLOCKS = [("train", 0, 47), ("val", 48, 54), ("test", 55, 61), ("heldout", 62, 67)]
WILDCARD_BLOCKS = {
    "A": [("heldout", 0, 12)],
    "B": [("train", 0, 6), ("heldout", 7, 12)],
}
WILDCARD_B_VAL_CANDIDATES = (0, 6)
TIME_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

ROW_CAPS = {"default": 200, "wildcard": 1000}
SAMPLING_SEED = 0
SAMPLED_SPLITS = ("train", "val")
GAP_WINDOWS = 1
TIME_SPLIT_CATEGORIES = ("tunnel", zfe.OWN_BENIGN)

ROLES = {
    "train": "fit",
    "val": "model_selection",
    "test": "in_distribution_test",
}
HELDOUT_ROLES = {
    "normal": "fpr_normal",
    "wildcard": "fpr_wildcard",
    zfe.OWN_BENIGN: "fpr_own_benign",
    "unknownTunnel": "unseen_tool",
    "crossEndPoint": "unseen_platform",
}

TUNNEL_FAMILIES = ("DNS-shell", "dnscat2", "dnspot", "iodine", "tuns")

COLUMNS = [
    "config", "capture_id", "category", "tool", "family", "label", "split", "role",
    "whole_capture", "first_window", "last_window", "t_start", "t_end",
    "t_start_utc", "t_end_utc", "window_seconds", "rule",
]

_INDEX = re.compile(r"_(\d{5})_\d{14}$")


def capture_index(capture_id):
    match = _INDEX.search(capture_id)
    if not match:
        raise ValueError(f"No _<index>_<timestamp> suffix in capture_id: {capture_id}")
    return int(match.group(1))


def tool_family(category, tool):
    if category == "tunnel":
        for family in TUNNEL_FAMILIES:
            if tool.lower().startswith(family.lower()):
                return family
        raise ValueError(f"Tunnel tool {tool!r} doesn't belong to a known family {TUNNEL_FAMILIES}")
    if category == "crossEndPoint":
        return "iodine (Android)"
    return tool


def capture_table(df):
    g = df.groupby("capture_id", sort=True)
    table = pd.DataFrame({
        "category": g["category"].first().astype(str),
        "tool": g["tool"].first().astype(str),
        "label": g["Label"].first().astype(str),
        "first_ts": g["ts"].min(),
        "last_ts": g["ts"].max(),
        "window_origin": g["window_start"].min(),
        "n_windows": g["window_id"].max() + 1,
    })
    return table.reset_index()


def timeline_segments(lengths, fractions=TIME_FRACTIONS, gap=GAP_WINDOWS):
    total = sum(lengths)
    available = total - 2 * gap
    if available < 3:
        raise ValueError(f"Too few windows ({total}) for a {'/'.join(fractions)} time split")
    n_train = max(1, math.floor(fractions["train"] * available + 0.5))
    n_val = max(1, math.floor(fractions["val"] * available + 0.5))
    n_test = available - n_train - n_val
    if n_test < 1:
        n_train -= 1 - n_test
        n_test = 1
    blocks = [("train", 0, n_train - 1),
              ("val", n_train + gap, n_train + gap + n_val - 1),
              ("test", n_train + n_val + 2 * gap, total - 1)]

    segments = []
    offset = 0
    for position, length in enumerate(lengths):
        for split, start, end in blocks:
            first, last = max(start, offset), min(end, offset + length - 1)
            if first <= last:
                segments.append((position, split, first - offset, last - offset))
        offset += length
    return segments


def _blocks_by_index(capture_id, blocks):
    index = capture_index(capture_id)
    for split, low, high in blocks:
        if low <= index <= high:
            return split, f"file index {low:05d}-{high:05d}"
    raise ValueError(f"{capture_id}: index {index} is not covered by {blocks}")


def wildcard_validation_capture(captures):
    low, high = WILDCARD_B_VAL_CANDIDATES
    wildcard = captures[captures["category"] == "wildcard"].copy()
    if wildcard.empty:
        return None
    wildcard["index"] = wildcard["capture_id"].map(capture_index)
    candidates = wildcard[wildcard["index"].between(low, high)]
    return candidates.sort_values(["n_windows", "index"], ascending=[False, True])["capture_id"].iloc[0]


def make_splits(captures, window_seconds=zfe.WINDOW_SECONDS):
    rows = []

    def add(config, capture, split, rule, first=None, last=None):
        whole = first is None
        first = 0 if whole else first
        last = capture["n_windows"] - 1 if whole else last
        t_start = capture["window_origin"] + first * window_seconds
        t_end = capture["window_origin"] + (last + 1) * window_seconds
        rows.append({
            "config": config,
            "capture_id": capture["capture_id"],
            "category": capture["category"],
            "tool": capture["tool"],
            "family": tool_family(capture["category"], capture["tool"]),
            "label": capture["label"],
            "split": split,
            "role": HELDOUT_ROLES[capture["category"]] if split == "heldout" else ROLES[split],
            "whole_capture": whole,
            "first_window": int(first),
            "last_window": int(last),
            "t_start": t_start,
            "t_end": t_end,
            "t_start_utc": pd.Timestamp(t_start, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "t_end_utc": pd.Timestamp(t_end, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            "window_seconds": window_seconds,
            "rule": rule,
        })

    unknown = set(captures["category"]) - set(zfe.TRAINING_CATEGORIES) - set(zfe.EVALUATION_ONLY_CATEGORIES)
    if unknown:
        raise ValueError(f"No split rule for categories: {sorted(unknown)}")
    wildcard_val = wildcard_validation_capture(captures)

    for config in CONFIGS:
        for _, capture in captures.sort_values("capture_id").iterrows():
            category = capture["category"]
            if category == "normal":
                split, rule = _blocks_by_index(capture["capture_id"], NORMAL_BLOCKS)
                add(config, capture, split, "normal " + rule)
            elif category == "wildcard" and config == "B" and capture["capture_id"] == wildcard_val:
                low, high = WILDCARD_B_VAL_CANDIDATES
                add(config, capture, "val", f"wildcard config B val: most windows of file index {low:05d}-{high:05d}")
            elif category == "wildcard":
                split, rule = _blocks_by_index(capture["capture_id"], WILDCARD_BLOCKS[config])
                add(config, capture, split, f"wildcard config {config} " + rule)
            elif category == "tunnel":
                for _, split, first, last in timeline_segments([capture["n_windows"]]):
                    add(config, capture, split, "tunnel time segment 70/15/15, 1-window gaps", first, last)
            elif category in zfe.EVALUATION_ONLY_CATEGORIES:
                add(config, capture, "heldout", "evaluation only")

        own = captures[captures["category"] == zfe.OWN_BENIGN].sort_values("first_ts")
        if len(own):
            add(config, own.iloc[-1], "heldout", "own_benign most recent complete session")
            earlier = own.iloc[:-1]
            if len(earlier):
                for position, split, first, last in timeline_segments(earlier["n_windows"].tolist()):
                    add(config, earlier.iloc[position], split,
                        "own_benign earlier sessions, contiguous time blocks 70/15/15, 1-window gaps",
                        first, last)

    splits = pd.DataFrame(rows, columns=COLUMNS)
    validate_splits(splits)
    return splits


def validate_splits(splits):
    missing = set(COLUMNS) - set(splits.columns)
    if missing:
        raise ValueError(f"splits table is missing columns: {sorted(missing)}")
    if set(splits["config"]) - set(CONFIGS):
        raise ValueError(f"Unknown configs: {sorted(set(splits['config']) - set(CONFIGS))}")
    if set(splits["split"]) - set(SPLIT_NAMES):
        raise ValueError(f"Unknown split names: {sorted(set(splits['split']) - set(SPLIT_NAMES))}")
    if (splits["first_window"] > splits["last_window"]).any():
        raise ValueError("A segment ends before it starts")

    problems = []
    captures_per_config = splits.groupby("config")["capture_id"].apply(frozenset)
    if captures_per_config.nunique() > 1:
        problems.append("configs cover different captures")

    for (config, capture_id), rows in splits.groupby(["config", "capture_id"], sort=False):
        category = rows["category"].iloc[0]
        if rows["category"].nunique() > 1 or rows["label"].nunique() > 1:
            problems.append(f"{config} {capture_id}: inconsistent category/label")
        if len(rows) > 1 and (category not in TIME_SPLIT_CATEGORIES or rows["whole_capture"].any()):
            problems.append(f"{config} {capture_id}: in {len(rows)} splits {sorted(rows['split'])}")
            continue
        ordered = rows.sort_values("first_window")
        previous_last = None
        previous_split = None
        for _, segment in ordered.iterrows():
            if previous_last is not None:
                gap = segment["first_window"] - previous_last - 1
                if gap < 0:
                    problems.append(f"{config} {capture_id}: {previous_split} and {segment['split']} overlap")
                elif gap < GAP_WINDOWS:
                    problems.append(f"{config} {capture_id}: no gap between {previous_split} and {segment['split']}")
            previous_last, previous_split = segment["last_window"], segment["split"]
        if ordered["split"].duplicated().any():
            problems.append(f"{config} {capture_id}: split repeated within one capture")

    evaluation_only = splits["category"].isin(zfe.EVALUATION_ONLY_CATEGORIES)
    if (splits.loc[evaluation_only, "split"] != "heldout").any():
        problems.append("an evaluation-only capture is used for train/val/test")
    wildcard_a = (splits["config"] == "A") & (splits["category"] == "wildcard")
    if (splits.loc[wildcard_a, "split"] != "heldout").any():
        problems.append("config A uses wildcard outside the held-out set")
    if problems:
        raise ValueError("Invalid splits:\n  " + "\n  ".join(problems))


def write_splits(splits, path=SPLITS_PATH):
    validate_splits(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False, float_format="%.0f", lineterminator="\n")


def read_splits(path=SPLITS_PATH):
    splits = pd.read_csv(path, dtype={"whole_capture": bool})
    validate_splits(splits)
    return splits


def assign_splits(df, splits, config, window_seconds=zfe.WINDOW_SECONDS):
    rows = splits[splits["config"] == config]
    if rows.empty:
        raise ValueError(f"No splits for config {config!r}")
    unknown = set(df["capture_id"].unique()) - set(rows["capture_id"])
    if unknown:
        raise ValueError(f"Captures missing from the splits table: {sorted(unknown)[:5]}")
    if (rows["window_seconds"] != window_seconds).any():
        raise ValueError(f"splits table was built for a different window length than {window_seconds}s")

    out = pd.DataFrame({"split": pd.Series(pd.NA, index=df.index, dtype="str"),
                        "role": pd.Series(pd.NA, index=df.index, dtype="str")})
    whole = rows[rows["whole_capture"]].set_index("capture_id")
    in_whole = df["capture_id"].isin(whole.index)
    out.loc[in_whole, "split"] = df.loc[in_whole, "capture_id"].map(whole["split"])
    out.loc[in_whole, "role"] = df.loc[in_whole, "capture_id"].map(whole["role"])

    segments = rows[~rows["whole_capture"]]
    for capture_id, capture_segments in segments.groupby("capture_id"):
        in_capture = df.index[df["capture_id"] == capture_id]
        starts = df.loc[in_capture, "window_start"]
        for _, segment in capture_segments.iterrows():
            inside = in_capture[(starts >= segment["t_start"]) & (starts < segment["t_end"])]
            out.loc[inside, "split"] = segment["split"]
            out.loc[inside, "role"] = segment["role"]
    return out


def sampling_keys(df, seed=SAMPLING_SEED):
    keys = pd.Series(np.nan, index=df.index)
    for capture_id, index in df.groupby("capture_id", sort=False).groups.items():
        ordered = df.loc[index, "ts"].sort_values(kind="stable").index
        rng = np.random.default_rng([seed, zlib.crc32(str(capture_id).encode("utf-8"))])
        keys.loc[ordered] = rng.random(len(ordered))
    return keys


def row_caps_for(categories, caps=ROW_CAPS):
    return categories.astype(str).map(lambda c: caps.get(c, caps["default"]))


def cap_mask(df, caps=ROW_CAPS, seed=SAMPLING_SEED):
    keys = sampling_keys(df, seed)
    rank = keys.groupby([df["capture_id"], df["window_id"]]).rank(method="first")
    return rank <= row_caps_for(df["category"], caps)


def cap_rows_per_window(df, caps=ROW_CAPS, seed=SAMPLING_SEED):
    return df[cap_mask(df, caps, seed)]


def fit_sample_mask(df, assignment, caps=ROW_CAPS, seed=SAMPLING_SEED):
    return assignment["split"].isin(SAMPLED_SPLITS).fillna(False).astype(bool) & cap_mask(df, caps, seed)


def split_counts(df, assignment, sample=None):
    frame = pd.DataFrame({
        "split": assignment["split"], "role": assignment["role"], "Label": df["Label"].astype(str),
        "category": df["category"].astype(str), "capture_id": df["capture_id"], "window_id": df["window_id"],
    }).dropna(subset=["split"])
    keys = ["split", "role", "Label", "category"]
    counts = frame.groupby(keys).agg(rows=("capture_id", "size"), captures=("capture_id", "nunique"))
    counts["windows"] = frame.drop_duplicates(keys + ["capture_id", "window_id"]).groupby(keys).size()
    if sample is not None:
        counts["sampled_rows"] = frame[sample.loc[frame.index]].groupby(keys).size()
    order = {name: i for i, name in enumerate(SPLIT_NAMES)}
    counts = counts.reset_index()
    counts["_order"] = counts["split"].map(order)
    return counts.sort_values(["_order", "role", "Label", "category"]).drop(columns="_order")


def main():
    manifest = zfe.read_manifest() + zfe.own_benign_manifest_rows()
    df = zfe.build_dataset_from_manifest(manifest, n_jobs=8, verbose=False)
    splits = make_splits(capture_table(df))
    write_splits(splits)
    print(f"Wrote {SPLITS_PATH} ({len(splits)} rows)")
    print(f"Row caps per window for train/val: {ROW_CAPS}, sampling seed {SAMPLING_SEED}")
    for config in CONFIGS:
        assignment = assign_splits(df, splits, config)
        counts = split_counts(df, assignment, fit_sample_mask(df, assignment))
        print(f"\nConfig {config} ({CONFIG_DESCRIPTIONS[config]})")
        print(counts.to_string(index=False))
        train = counts[counts["split"] == "train"]
        benign = train.loc[train["Label"] == "benign", "sampled_rows"].sum()
        tunnel = train.loc[train["Label"] == "tunnel", "sampled_rows"].sum()
        wildcard = train.loc[train["category"] == "wildcard", "sampled_rows"].sum()
        print(f"sampled train rows: benign {benign:,.0f} / tunnel {tunnel:,.0f} = {benign / tunnel:.2f}:1; "
              f"wildcard {wildcard:,.0f} = {100 * wildcard / benign:.1f}% of benign")


if __name__ == "__main__":
    sys.exit(main())
