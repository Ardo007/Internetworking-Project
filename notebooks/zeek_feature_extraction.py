import csv
import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dns_feature_extraction import DNS_QTYPES, _base_domain, _lexical_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "Data" / "processed" / "GraphTunnel" / "capture_manifest.csv"
LIVE_ZEEK_DIR = PROJECT_ROOT / "datas" / "zeek"

WINDOW_SECONDS = 60

REJOIN_MAX_GAP_SECONDS = 30

QTYPE_NAMES = {**DNS_QTYPES, 64: "SVCB", 65: "HTTPS"}

LABEL_NAMES = {0: "benign", 1: "tunnel"}
TRAINING_CATEGORIES = ("normal", "tunnel", "wildcard", "own_benign")
EVALUATION_ONLY_CATEGORIES = ("unknownTunnel", "crossEndPoint")
OWN_BENIGN = "own_benign"

_QTYPE_NAME_SET = set(QTYPE_NAMES.values())
_ZEEK_ESCAPE = re.compile(rb"\\x([0-9a-fA-F]{2})")
_REQUIRED_FIELDS = ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto")

_RESPONSE_FIELDS = ("rcode", "rcode_name", "AA", "TC", "RA", "Z", "answers", "TTLs", "rejected")

_CHUNK_NAME = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{5})_(?P<timestamp>\d{14})$")


def load_dns_log(path):
    records = []
    malformed = 0
    with open(path, "rb") as file:
        for raw in file:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(record, dict) or any(f not in record for f in _REQUIRED_FIELDS):
                malformed += 1
                continue
            records.append(record)
    return records, malformed


def load_dns_logs(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records = []
    malformed = 0
    for path in paths:
        chunk_records, chunk_malformed = load_dns_log(path)
        records.extend(chunk_records)
        malformed += chunk_malformed
    return records, malformed


def decode_query(query):
    if not query:
        return ""
    if "\\" in query or not query.isascii():
        raw = _ZEEK_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), query.encode("utf-8"))
        query = raw.decode("latin-1")
    return query.rstrip(".").lower()


def qtype_name_of(record):
    name = record.get("qtype_name")
    if name in _QTYPE_NAME_SET:
        return name
    return QTYPE_NAMES.get(record.get("qtype"), "OTHER")


def _is_query_only(record):
    return "rcode" not in record


def _is_response_only(record):
    return "rcode" in record and "qtype" not in record


def _transaction_key(record):
    return (record["proto"], record["id.orig_h"], record["id.orig_p"],
            record["id.resp_h"], record["id.resp_p"], record.get("trans_id"))


def join_split_transactions(records, max_gap=REJOIN_MAX_GAP_SECONDS):
    responses = defaultdict(list)
    for index, record in enumerate(records):
        if _is_response_only(record):
            responses[_transaction_key(record)].append(index)
    for indices in responses.values():
        indices.sort(key=lambda i: records[i]["ts"])

    used = set()
    merged = {}
    for index in sorted((i for i, r in enumerate(records) if _is_query_only(r)),
                        key=lambda i: records[i]["ts"]):
        query = records[index]
        candidates = responses.get(_transaction_key(query))
        if not candidates:
            continue
        query_name = (query.get("query") or "").lower()
        for candidate in candidates:
            if candidate in used:
                continue
            response = records[candidate]
            gap = response["ts"] - query["ts"]
            if gap < 0:
                continue
            if gap > max_gap:
                break
            if "query" in response and response["query"].lower() != query_name:
                continue
            combined = dict(query)
            for field in _RESPONSE_FIELDS:
                if field in response:
                    combined[field] = response[field]
            if response.get("answers") and gap > 0:
                combined["rtt"] = gap
            combined["rejoined"] = True
            merged[index] = combined
            used.add(candidate)
            break

    out = [merged.get(i, record) for i, record in enumerate(records) if i not in used]
    out.sort(key=lambda r: r["ts"])
    return out, len(merged)


def drop_unmatched_responses(records):
    kept = [r for r in records if not _is_response_only(r)]
    return kept, len(records) - len(kept)


def prepare_records(records, max_gap=REJOIN_MAX_GAP_SECONDS):
    records, rejoined = join_split_transactions(records, max_gap)
    records, dropped = drop_unmatched_responses(records)
    return records, {"rejoined": rejoined, "unmatched_responses_dropped": dropped}


def records_to_frame(records):
    qnames = [decode_query(r.get("query")) for r in records]
    lexical_cache = {}
    lexical = []
    for qname in qnames:
        features = lexical_cache.get(qname)
        if features is None:
            features = lexical_cache[qname] = _lexical_features(qname)
        lexical.append(features)

    nan = math.nan
    frame = pd.DataFrame({
        "ts": np.array([r["ts"] for r in records], dtype="float64"),
        "uid": [r["uid"] for r in records],
        "qname": qnames,
        "base_domain": [_base_domain(q) or "" for q in qnames],
        "proto": [r["proto"] for r in records],
        "qtype_name": [qtype_name_of(r) for r in records],
        "response_ancount": np.array([len(r.get("answers") or ()) for r in records], dtype="float64"),
        "response_min_ttl": np.array([min(r["TTLs"]) if r.get("TTLs") else nan for r in records], dtype="float64"),
        "response_rcode": np.array([r.get("rcode", nan) for r in records], dtype="float64"),
        "response_latency": np.array([r.get("rtt", nan) for r in records], dtype="float64"),
        "rejected": np.array([bool(r.get("rejected", False)) for r in records], dtype=bool),
        "rejoined": np.array([bool(r.get("rejoined", False)) for r in records], dtype=bool),
    })
    lexical_frame = pd.DataFrame.from_records(lexical, columns=list(_lexical_features("")))
    return pd.concat([frame, lexical_frame], axis=1)


def assign_windows(df, window_seconds=WINDOW_SECONDS, group_col="capture_id"):
    out = df.copy()
    window_start = np.floor(out["ts"] / window_seconds) * window_seconds
    first = window_start.groupby(out[group_col]).transform("min")
    out["window_start"] = window_start
    out["window_id"] = ((window_start - first) / window_seconds).round().astype("int64")
    out["window_query_count"] = out.groupby([group_col, "window_id"])["ts"].transform("size").astype("int64")
    return out


def extract_query_records_from_zeek(dns_log_path, capture_id, category, tool, label,
                                    window_seconds=WINDOW_SECONDS):
    records, malformed = load_dns_logs(dns_log_path)
    records, stats = prepare_records(records)
    frame = records_to_frame(records)
    frame.insert(1, "capture_id", capture_id)
    frame.insert(2, "category", category)
    frame.insert(3, "tool", tool)
    frame["Label"] = LABEL_NAMES[int(label)] if str(label).isdigit() else label
    frame = assign_windows(frame, window_seconds)
    frame.attrs.update(malformed_lines=malformed, **stats)
    return frame


DOMAIN_KEYS = ("capture_id", "window_id", "base_domain")


def _entropy_by_group(values, keys):
    counts = pd.concat([keys, values.rename("_value")], axis=1).groupby(
        list(keys.columns) + ["_value"], sort=False).size()
    group_levels = list(range(len(keys.columns)))
    p = counts / counts.groupby(level=group_levels, sort=False).transform("sum")
    return (-(p * np.log2(p))).groupby(level=group_levels, sort=False).sum()


def add_domain_aggregates(df, group_cols=DOMAIN_KEYS):
    keys = list(group_cols)
    work = df[keys].copy()
    work["qname"] = df["qname"]
    work["qtype_name"] = df["qtype_name"]
    work["qname_len"] = df["qname_len"]
    work["entropy"] = df["entropy"]
    work["ts"] = df["ts"]
    work["is_txt"] = (df["qtype_name"] == "TXT").astype("float64")
    work["is_null"] = (df["qtype_name"] == "NULL").astype("float64")
    work["is_nxdomain"] = (df["response_rcode"] == 3).astype("float64")
    work["no_response"] = df["response_rcode"].isna().astype("float64")
    work["is_rejected"] = df["rejected"].astype("float64")

    agg = work.groupby(keys, sort=False).agg(
        domain_query_count=("qname", "size"),
        domain_unique_qnames=("qname", "nunique"),
        domain_qtype_diversity=("qtype_name", "nunique"),
        domain_avg_qname_len=("qname_len", "mean"),
        domain_std_qname_len=("qname_len", "std"),
        domain_avg_entropy=("entropy", "mean"),
        domain_txt_ratio=("is_txt", "mean"),
        domain_null_ratio=("is_null", "mean"),
        domain_first_ts=("ts", "min"),
        domain_last_ts=("ts", "max"),
        nxdomain_ratio=("is_nxdomain", "mean"),
        no_response_ratio=("no_response", "mean"),
        rejected_ratio=("is_rejected", "mean"),
    )
    answered = df["response_rcode"].notna()
    entropy = _entropy_by_group(df.loc[answered, "response_rcode"], df.loc[answered, keys])
    agg["rcode_entropy"] = entropy.reindex(agg.index, fill_value=0.0).abs()

    agg["domain_duration"] = (agg["domain_last_ts"] - agg["domain_first_ts"]).clip(lower=1.0)
    agg["domain_query_rate"] = agg["domain_query_count"] / agg["domain_duration"]
    agg["domain_unique_subdomain_ratio"] = agg["domain_unique_qnames"] / agg["domain_query_count"]
    agg["domain_std_qname_len"] = agg["domain_std_qname_len"].fillna(0.0)
    agg = agg.drop(columns=["domain_first_ts", "domain_last_ts"])

    return df.merge(agg.reset_index(), on=keys, how="left", validate="many_to_one")


FEATURE_GROUPS = {
    "transport": ["proto"],
    "lexical": [
        "qname_len", "label_count", "max_label_len", "avg_label_len", "first_label_len",
        "digit_ratio", "hex_ratio", "unique_char_ratio", "entropy", "first_label_entropy",
        "qtype_name",
    ],
    "response": ["response_ancount", "response_min_ttl", "response_rcode"],
    "domain_volume": [
        "domain_query_count", "domain_unique_qnames", "domain_query_rate",
        "domain_duration", "domain_unique_subdomain_ratio",
    ],
    "domain_shape": [
        "domain_avg_qname_len", "domain_std_qname_len", "domain_avg_entropy",
        "domain_txt_ratio", "domain_null_ratio", "domain_qtype_diversity",
    ],
    "blackhole": ["nxdomain_ratio", "no_response_ratio", "rejected_ratio", "rcode_entropy"],
    "artefact_suspect": ["domain_qtype_diversity", "no_response_ratio"],
}

ALL_FEATURE_COLUMNS = list(dict.fromkeys(c for group in FEATURE_GROUPS.values() for c in group))

FEATURE_COLUMNS = [c for c in ALL_FEATURE_COLUMNS if c not in FEATURE_GROUPS["artefact_suspect"]]

FEATURE_SETS = {
    "all": ALL_FEATURE_COLUMNS,
    "lexical_only": FEATURE_GROUPS["lexical"],
    "domain_volume_shape": FEATURE_GROUPS["domain_volume"] + FEATURE_GROUPS["domain_shape"],
    "all_minus_artefact_suspect": FEATURE_COLUMNS,
    "domain_volume_shape_minus_artefact_suspect": [
        c for c in FEATURE_GROUPS["domain_volume"] + FEATURE_GROUPS["domain_shape"]
        if c not in FEATURE_GROUPS["artefact_suspect"]],
}
DEFAULT_FEATURE_SET = "all_minus_artefact_suspect"

BOOKKEEPING_COLUMNS = [
    "ts", "window_start", "window_id", "window_query_count",
    "capture_id", "category", "tool", "base_domain", "uid", "qname", "rejoined",
    "response_latency",
]

_MISSING_RESPONSE_VALUES = {
    "response_ancount": 0.0,
    "response_min_ttl": -1.0,
    "response_rcode": -1.0,
    "response_latency": -1.0,
}


def to_ml_frame(df, label_col="Label", extra_cols=(), features=FEATURE_COLUMNS):
    cols = list(features) + [c for c in extra_cols if c not in features]
    if label_col is not None:
        cols.append(label_col)
    out = df[cols].copy()
    for column, value in _MISSING_RESPONSE_VALUES.items():
        if column in out:
            out[column] = out[column].fillna(value)
    return out


def read_manifest(path=MANIFEST_PATH):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"capture_id", "category", "tool", "label", "zeek_dns_log"}
    missing = required.difference(rows[0] if rows else required)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def find_live_sessions(zeek_dir=LIVE_ZEEK_DIR, idle_seconds=300, now=None):
    zeek_dir = Path(zeek_dir)
    if not zeek_dir.is_dir():
        return []
    chunks = []
    for folder in zeek_dir.iterdir():
        match = _CHUNK_NAME.match(folder.name)
        if not folder.is_dir() or not match or folder.name.endswith("_Backup"):
            continue
        started = datetime.strptime(match["timestamp"], "%Y%m%d%H%M%S").timestamp()
        chunks.append((match["prefix"], started, int(match["index"]), folder))
    chunks.sort(key=lambda c: (c[0], c[1], c[2]))

    sessions = []
    for prefix, started, index, folder in chunks:
        current = sessions[-1] if sessions else None
        if current is None or current["prefix"] != prefix or index != current["last_index"] + 1:
            current = {"prefix": prefix, "session_id": folder.name, "chunks": [], "dns_logs": [],
                       "start": started}
            sessions.append(current)
        current["chunks"].append(folder.name)
        current["last_index"] = index
        current["last_chunk_start"] = started
        if (folder / "dns.log").is_file():
            current["dns_logs"].append(folder / "dns.log")

    now = time.time() if now is None else now
    for i, session in enumerate(sessions):
        later = any(s["prefix"] == session["prefix"] and s["start"] > session["start"] for s in sessions[i + 1:])
        session["complete"] = later or now - session["last_chunk_start"] > idle_seconds
        del session["prefix"], session["last_index"]
    sessions.sort(key=lambda s: s["start"])
    return sessions


def own_benign_manifest_rows(zeek_dir=LIVE_ZEEK_DIR, include_incomplete=False, **session_kwargs):
    rows = []
    for session in find_live_sessions(zeek_dir, **session_kwargs):
        if not session["dns_logs"] or not (session["complete"] or include_incomplete):
            continue
        rows.append({
            "capture_id": f"{OWN_BENIGN}/{session['session_id']}",
            "category": OWN_BENIGN,
            "tool": OWN_BENIGN,
            "label": "0",
            "zeek_dns_log": list(session["dns_logs"]),
            "session_start": session["start"],
            "complete": session["complete"],
        })
    return rows


def check_tool_categories(manifest_rows):
    known = set(TRAINING_CATEGORIES) | set(EVALUATION_ONLY_CATEGORIES)
    unknown = {r["category"] for r in manifest_rows} - known
    if unknown:
        raise ValueError(f"Unknown categories in manifest: {sorted(unknown)}")

    categories_by_tool = defaultdict(set)
    for row in manifest_rows:
        categories_by_tool[row["tool"].casefold()].add(row["category"])
    overlap = sorted(
        tool for tool, cats in categories_by_tool.items()
        if cats & set(TRAINING_CATEGORIES) and cats & set(EVALUATION_ONLY_CATEGORIES)
    )
    if overlap:
        raise ValueError(
            "Tools appear in both training and evaluation-only categories: "
            + ", ".join(f"{t} {sorted(categories_by_tool[t])}" for t in overlap)
        )
    dns2tcp_key = categories_by_tool.get("dns2tcp-key", set())
    if dns2tcp_key - {"unknownTunnel"}:
        raise ValueError(f"dns2tcp-key must only appear under unknownTunnel, found in {sorted(dns2tcp_key)}")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _dns_log_paths(row):
    value = row["zeek_dns_log"]
    paths = [value] if isinstance(value, (str, Path)) else list(value)
    resolved = [_resolve(p) for p in paths]
    for path in resolved:
        if path.parent.name.endswith("_Backup"):
            raise ValueError(f"Manifest points at a _Backup folder: {path}")
    return resolved


def _build_capture(row, window_seconds):
    frame = extract_query_records_from_zeek(
        _dns_log_paths(row), row["capture_id"], row["category"], row["tool"], int(row["label"]),
        window_seconds=window_seconds,
    )
    stats = dict(capture_id=row["capture_id"], category=row["category"], records=len(frame),
                 **frame.attrs)
    return add_domain_aggregates(frame), stats


def build_dataset_from_manifest(manifest_rows, window_seconds=WINDOW_SECONDS, n_jobs=1, verbose=True):
    manifest_rows = list(manifest_rows)
    check_tool_categories(manifest_rows)
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_build_capture, manifest_rows, [window_seconds] * len(manifest_rows)))
    else:
        results = [_build_capture(row, window_seconds) for row in manifest_rows]

    frames = []
    stats = []
    for frame, capture_stats in results:
        if verbose:
            print(f"{capture_stats['capture_id']}: {capture_stats['records']} records, "
                  f"{capture_stats['rejoined']} rejoined, "
                  f"{capture_stats['unmatched_responses_dropped']} unmatched responses dropped, "
                  f"{capture_stats['malformed_lines']} malformed lines skipped")
        frames.append(frame)
        stats.append(capture_stats)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.attrs["capture_stats"] = stats
    return combined
