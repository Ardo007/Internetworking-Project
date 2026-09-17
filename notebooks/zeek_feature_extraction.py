"""
Zeek dns.log Feature Extraction
===============================
Builds the per-query feature table for the DNS tunnelling model from Zeek
``dns.log`` files (JSON, one record per line, ``LogAscii::use_json=T``)
instead of parsing PCAPs. The lexical features, the qtype vocabulary and the
base-domain rule are imported from ``dns_feature_extraction`` so both paths
compute them identically.

Pipeline, per capture:
  1. load_dns_log             read the JSON lines; skip and count malformed ones
  2. join_split_transactions  merge a query Zeek logged without its response
                              with the response it logged separately
  3. records_to_frame         one row per record: lexical and response
                              features plus bookkeeping columns
  4. assign_windows           window_id = WINDOW_SECONDS time window
  5. add_domain_aggregates    per (capture_id, window_id, base_domain)

Every model feature (FEATURE_COLUMNS) is computed from dns.log alone.
conn.log is not used: its resp_bytes is per connection, and a single UDP
connection can carry a whole tunnel session (up to 135,677 queries in
GraphTunnel), so there is no reliable per-query response size.

Windows. Domain aggregates are computed per window rather than per capture,
so they don't depend on how long a capture happened to be. Window boundaries
are aligned to the Unix epoch (window_start = floor(ts / W) * W) and
window_id counts windows from the first one in the capture.

Split transactions. Zeek logs a query and its response as two records when it
can't pair them: it gives up on a connection once 50 transaction IDs are
waiting for an answer (DNS::max_pending_query_ids), it times a DNS
connection out after 10 s without packets (dns_session_timeout), and a
transaction can be cut in two by a capture boundary. join_split_transactions
merges a query-only
record (no rcode) with the earliest response-only record (an rcode but no
qtype, since Zeek only takes qtype from the request) that has the same
protocol, addresses, ports and trans_id, the same query when the response has
one, and arrives at most REJOIN_MAX_GAP_SECONDS after the query.

Live scoring. run_zeek.py writes one Zeek folder per 30-second capture chunk
(datas/zeek/<chunk>/). Live scoring must join consecutive 30 s chunks into
windows of WINDOW_SECONDS before computing domain aggregates: concatenate the
dns.log records of consecutive chunks, run join_split_transactions over the
concatenation (a transaction can straddle two chunks), then assign windows
and compute the aggregates. Aggregates computed on one 30 s chunk on its own
would not match what the model was trained on.

Query decoding. Zeek writes every byte of a DNS name that isn't printable
ASCII as ``\\xNN``. decode_query turns those escapes back into single latin-1
characters and lowercases the name, which reproduces the hand-rolled PCAP
parser exactly (checked on all 105 GraphTunnel captures). A literal backslash
followed by ``x`` and two hex digits would be indistinguishable from an
escape; no GraphTunnel capture contains one.
"""
import csv
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dns_feature_extraction import DNS_QTYPES, _base_domain, _lexical_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "Data" / "processed" / "GraphTunnel" / "capture_manifest.csv"

#: Length of the time window the domain aggregates are computed over. Two
#: live-capture chunks; the shortest GraphTunnel tunnel capture (1,577 s)
#: still yields ~3 validation and ~3 test windows after the time split.
WINDOW_SECONDS = 60

#: Maximum time between a query-only record and the response-only record it
#: is merged with.
REJOIN_MAX_GAP_SECONDS = 30

LABEL_NAMES = {0: "benign", 1: "tunnel"}
TRAINING_CATEGORIES = ("normal", "tunnel", "wildcard")
EVALUATION_ONLY_CATEGORIES = ("unknownTunnel", "crossEndPoint")

_QTYPE_NAMES = set(DNS_QTYPES.values())
_ZEEK_ESCAPE = re.compile(rb"\\x([0-9a-fA-F]{2})")
_REQUIRED_FIELDS = ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto")

# Copied from the response half when a split transaction is merged.
_RESPONSE_FIELDS = ("rcode", "rcode_name", "AA", "TC", "RA", "Z", "answers", "TTLs", "rejected")

# -------------------------------------------------------------- loading --

def load_dns_log(path):
    """Read a JSON dns.log.

    Returns (records, malformed): the records as dicts, in file order, and
    the number of lines skipped because they weren't a JSON object with the
    fields every Zeek DNS record has. Blank lines are ignored.
    """
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


def decode_query(query):
    """Zeek `query` string -> the name the PCAP parser produces.

    Undoes Zeek's JSON escaping (``\\xNN`` -> the byte; any UTF-8 passed
    through -> its bytes), decodes latin-1 (one byte, one character),
    strips a trailing dot and lowercases. A missing query becomes "".
    """
    if not query:
        return ""
    if "\\" in query or not query.isascii():
        raw = _ZEEK_ESCAPE.sub(lambda m: bytes([int(m.group(1), 16)]), query.encode("utf-8"))
        query = raw.decode("latin-1")
    return query.rstrip(".").lower()


def qtype_name_of(record):
    """qtype_name restricted to the DNS_QTYPES vocabulary, else "OTHER"."""
    name = record.get("qtype_name")
    if name in _QTYPE_NAMES:
        return name
    # Zeek names a few types differently from DNS_QTYPES (e.g. 255 is "*").
    return DNS_QTYPES.get(record.get("qtype"), "OTHER")


# ------------------------------------------------ split transactions --

def _is_query_only(record):
    return "rcode" not in record


def _is_response_only(record):
    return "rcode" in record and "qtype" not in record


def _transaction_key(record):
    return (record["proto"], record["id.orig_h"], record["id.orig_p"],
            record["id.resp_h"], record["id.resp_p"], record.get("trans_id"))


def join_split_transactions(records, max_gap=REJOIN_MAX_GAP_SECONDS):
    """Merge query-only records with the response-only record Zeek logged
    separately for the same transaction.

    A query-only record (no rcode) is merged with the earliest unused
    response-only record (rcode but no qtype) that has the same protocol,
    addresses, ports and trans_id, the same query if the response has one,
    and a timestamp between the query's and the query's + max_gap.

    The merged record keeps the query's ts, uid, query, qtype and RD, takes
    the response's rcode, flags, answers, TTLs and rejected, and gets
    rtt = response ts - query ts when the response has answers (Zeek only
    sets rtt for answered queries). It is marked ``rejoined=True``.

    Returns (records sorted by ts, number of merges).
    """
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


# ----------------------------------------------------- record -> row --

def records_to_frame(records):
    """One row per Zeek DNS record: features and per-record bookkeeping."""
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
        "rejected": np.array([bool(r.get("rejected", False)) for r in records]),
        "rejoined": np.array([bool(r.get("rejoined", False)) for r in records]),
    })
    lexical_frame = pd.DataFrame.from_records(lexical, columns=list(_lexical_features("")))
    return pd.concat([frame, lexical_frame], axis=1)


def assign_windows(df, window_seconds=WINDOW_SECONDS, group_col="capture_id"):
    """Add window_start, window_id and window_query_count.

    window_start is the epoch-aligned start of the row's window; window_id
    counts windows from the first window of the row's `group_col` group;
    window_query_count is the number of records in that window.
    """
    out = df.copy()
    window_start = np.floor(out["ts"] / window_seconds) * window_seconds
    first = window_start.groupby(out[group_col]).transform("min")
    out["window_start"] = window_start
    out["window_id"] = ((window_start - first) / window_seconds).round().astype("int64")
    out["window_query_count"] = out.groupby([group_col, "window_id"])["ts"].transform("size").astype("int64")
    return out


def extract_query_records_from_zeek(dns_log_path, capture_id, category, tool, label,
                                    window_seconds=WINDOW_SECONDS):
    """Parse one capture's dns.log into a per-record DataFrame.

    One row per record left after join_split_transactions, with lexical and
    response features, window assignment and bookkeeping columns
    (ts, capture_id, category, tool, base_domain, ...). Domain aggregates
    are added by add_domain_aggregates. ``df.attrs`` holds the number of
    malformed lines skipped and of split transactions merged.
    """
    records, malformed = load_dns_log(dns_log_path)
    records, rejoined = join_split_transactions(records)
    frame = records_to_frame(records)
    frame.insert(1, "capture_id", capture_id)
    frame.insert(2, "category", category)
    frame.insert(3, "tool", tool)
    frame["Label"] = LABEL_NAMES[int(label)] if str(label).isdigit() else label
    frame = assign_windows(frame, window_seconds)
    frame.attrs.update(malformed_lines=malformed, rejoined=rejoined)
    return frame


# ---------------------------------------------------- domain aggregates --

DOMAIN_KEYS = ("capture_id", "window_id", "base_domain")


def _entropy_by_group(values, keys):
    """Shannon entropy (bits) of `values` within each group of `keys`."""
    counts = pd.concat([keys, values.rename("_value")], axis=1).groupby(
        list(keys.columns) + ["_value"], sort=False).size()
    group_levels = list(range(len(keys.columns)))
    p = counts / counts.groupby(level=group_levels, sort=False).transform("sum")
    return (-(p * np.log2(p))).groupby(level=group_levels, sort=False).sum()


def add_domain_aggregates(df, group_cols=DOMAIN_KEYS):
    """Add per-(capture, window, base_domain) features back onto every row."""
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
    # rcode_entropy: diversity of the response codes the domain received in
    # the window. Records without a response are left out (that is what
    # no_response_ratio measures); a window with no responses scores 0.
    answered = df["response_rcode"].notna()
    entropy = _entropy_by_group(df.loc[answered, "response_rcode"], df.loc[answered, keys])
    agg["rcode_entropy"] = entropy.reindex(agg.index, fill_value=0.0).abs()

    # Floor the duration at 1 second so a domain seen once or twice within a
    # fraction of a second doesn't get an absurd instantaneous query rate.
    agg["domain_duration"] = (agg["domain_last_ts"] - agg["domain_first_ts"]).clip(lower=1.0)
    agg["domain_query_rate"] = agg["domain_query_count"] / agg["domain_duration"]
    agg["domain_unique_subdomain_ratio"] = agg["domain_unique_qnames"] / agg["domain_query_count"]
    agg["domain_std_qname_len"] = agg["domain_std_qname_len"].fillna(0.0)
    agg = agg.drop(columns=["domain_first_ts", "domain_last_ts"])

    return df.merge(agg.reset_index(), on=keys, how="left", validate="many_to_one")


# ------------------------------------------------------------ ML frame --

#: Columns handed to the model. Identifiers (IPs, ports, query, uid,
#: trans_id, capture_id, tool) are deliberately excluded so the model can't
#: memorise a host, session or domain string.
FEATURE_COLUMNS = [
    "proto", "qtype_name",
    "qname_len", "label_count", "max_label_len", "avg_label_len", "first_label_len",
    "digit_ratio", "hex_ratio", "unique_char_ratio", "entropy", "first_label_entropy",
    "response_ancount", "response_min_ttl", "response_rcode", "response_latency",
    "domain_query_count", "domain_unique_qnames", "domain_qtype_diversity",
    "domain_avg_qname_len", "domain_std_qname_len", "domain_avg_entropy",
    "domain_txt_ratio", "domain_null_ratio", "domain_duration",
    "domain_query_rate", "domain_unique_subdomain_ratio",
    "nxdomain_ratio", "no_response_ratio", "rejected_ratio", "rcode_entropy",
]

#: Kept alongside the features for splitting and analysis, never modelled.
BOOKKEEPING_COLUMNS = [
    "ts", "window_start", "window_id", "window_query_count",
    "capture_id", "category", "tool", "base_domain", "uid", "qname", "rejoined",
]


def to_ml_frame(df, label_col="Label", extra_cols=()):
    """Select the model columns and fill response fields that are missing
    when a query got no (answered) response. `extra_cols` keeps bookkeeping
    columns (e.g. capture_id, window_id) for splitting and analysis."""
    cols =FEATURE_COLUMNS + [c for c in extra_cols if c not in FEATURE_COLUMNS] + [label_col]
    out = df[cols].copy()
    out["response_ancount"] = out["response_ancount"].fillna(0.0)
    for column in ("response_min_ttl", "response_rcode", "response_latency"):
        out[column] = out[column].fillna(-1.0)
    return out


# ------------------------------------------------------------- manifest --

def read_manifest(path=MANIFEST_PATH):
    """Rows of capture_manifest.csv (written by process_pcaps.py) as dicts."""
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"capture_id", "category", "tool", "label", "zeek_dns_log"}
    missing = required.difference(rows[0] if rows else required)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def check_tool_categories(manifest_rows):
    """Fail loudly if a tool is used both for training and as an unseen tool.

    Tool names seen in the training categories (normal/tunnel/wildcard) must
    not appear in the evaluation-only ones (unknownTunnel/crossEndPoint), and
    dns2tcp-key may only appear under unknownTunnel.
    """
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


def _build_capture(row, window_seconds):
    dns_log = _resolve(row["zeek_dns_log"])
    if dns_log.parent.name.endswith("_Backup"):
        raise ValueError(f"Manifest points at a _Backup folder: {dns_log}")
    frame = extract_query_records_from_zeek(
        dns_log, row["capture_id"], row["category"], row["tool"], int(row["label"]),
        window_seconds=window_seconds,
    )
    stats = dict(capture_id=row["capture_id"], category=row["category"], records=len(frame),
                 **frame.attrs)
    return add_domain_aggregates(frame), stats


def build_dataset_from_manifest(manifest_rows, window_seconds=WINDOW_SECONDS, n_jobs=1, verbose=True):
    """Build the combined feature table for every capture in the manifest.

    Returns one DataFrame (one row per DNS record, domain aggregates
    included). ``df.attrs["capture_stats"]`` lists, per capture, the record
    count, malformed lines skipped and split transactions merged.
    """
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
                  f"{capture_stats['rejoined']} rejoined, {capture_stats['malformed_lines']} malformed lines skipped")
        frames.append(frame)
        stats.append(capture_stats)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.attrs["capture_stats"] = stats
    return combined
