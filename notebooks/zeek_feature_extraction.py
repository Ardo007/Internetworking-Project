"""
Zeek dns.log Feature Extraction
===============================
Builds the per-query feature table for the DNS tunnelling model from Zeek
``dns.log`` files (JSON, one record per line, ``LogAscii::use_json=T``)
instead of parsing PCAPs. The lexical features, the qtype vocabulary and the
base-domain rule are imported from ``dns_feature_extraction`` so both paths
compute them identically.

Pipeline, per capture:
  1. load_dns_logs              read the JSON lines of one dns.log, or of
                                several consecutive live chunks; skip and
                                count malformed lines
  2. prepare_records            join_split_transactions, then
                                drop_unmatched_responses
  3. records_to_frame           one row per record: lexical and response
                                features plus bookkeeping columns
  4. assign_windows             window_id = WINDOW_SECONDS time window
  5. add_domain_aggregates      per (capture_id, window_id, base_domain)

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
merges a query-only record (no rcode) with the earliest response-only record
(an rcode but no qtype, since Zeek only takes qtype from the request) that
has the same protocol, addresses, ports and trans_id, the same query when the
response has one, and arrives at most REJOIN_MAX_GAP_SECONDS after the query.

Unmatched responses. A response-only record still left after the join is a
response whose query isn't in the data (it was sent before the capture
started, or its record is missing). Zeek doesn't know its qtype and takes
its query name from the answer section, so drop_unmatched_responses removes
it before rows and domain aggregates are built.

Live scoring. run_zeek.py writes one Zeek folder per 30-second capture chunk
(datas/zeek/<chunk>/). Live scoring must join consecutive 30 s chunks into
windows of WINDOW_SECONDS before computing domain aggregates: load the
dns.log files of consecutive chunks together (load_dns_logs accepts a list),
run prepare_records over the concatenation (a transaction can straddle two
chunks), then assign windows and compute the aggregates. Aggregates computed
on one 30 s chunk on its own would not match what the model was trained on.
find_live_sessions groups chunk folders into capture sessions, and
own_benign_manifest_rows turns them into manifest rows for the "own_benign"
category.

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

#: Length of the time window the domain aggregates are computed over. Two
#: live-capture chunks; the shortest GraphTunnel tunnel capture (1,577 s)
#: still yields ~3 validation and ~3 test windows after the time split.
WINDOW_SECONDS = 60

#: Maximum time between a query-only record and the response-only record it
#: is merged with.
REJOIN_MAX_GAP_SECONDS = 30

#: qtype categories: DNS_QTYPES plus the service-binding types, which modern
#: clients ask for alongside A/AAAA. Everything else is "OTHER".
QTYPE_NAMES = {**DNS_QTYPES, 64: "SVCB", 65: "HTTPS"}

LABEL_NAMES = {0: "benign", 1: "tunnel"}
#: "own_benign" is benign traffic from the live pipeline (datas/zeek).
TRAINING_CATEGORIES = ("normal", "tunnel", "wildcard", "own_benign")
EVALUATION_ONLY_CATEGORIES = ("unknownTunnel", "crossEndPoint")
OWN_BENIGN = "own_benign"

_QTYPE_NAME_SET = set(QTYPE_NAMES.values())
_ZEEK_ESCAPE = re.compile(rb"\\x([0-9a-fA-F]{2})")
_REQUIRED_FIELDS = ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto")

# Copied from the response half when a split transaction is merged.
_RESPONSE_FIELDS = ("rcode", "rcode_name", "AA", "TC", "RA", "Z", "answers", "TTLs", "rejected")

# dumpcap ring-buffer chunk names, as in Ardashes_scripts/run_zeek.py:
# <prefix>_<00001>_<YYYYmmddHHMMSS>
_CHUNK_NAME = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{5})_(?P<timestamp>\d{14})$")

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


def load_dns_logs(paths):
    """load_dns_log for one path or a list of paths (consecutive live
    chunks), concatenated in the order given."""
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
    """qtype_name restricted to the QTYPE_NAMES vocabulary, else "OTHER"."""
    name = record.get("qtype_name")
    if name in _QTYPE_NAME_SET:
        return name
    # Zeek names a few types differently (e.g. 255 is "*").
    return QTYPE_NAMES.get(record.get("qtype"), "OTHER")


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


def drop_unmatched_responses(records):
    """Remove response-only records (rcode but no qtype).

    Run after join_split_transactions: what is left is a response whose
    query isn't in the data. Returns (records, number removed).
    """
    kept = [r for r in records if not _is_response_only(r)]
    return kept, len(records) - len(kept)


def prepare_records(records, max_gap=REJOIN_MAX_GAP_SECONDS):
    """join_split_transactions + drop_unmatched_responses.

    Returns (records sorted by ts, {"rejoined": n, "unmatched_responses_dropped": m}).
    """
    records, rejoined = join_split_transactions(records, max_gap)
    records, dropped = drop_unmatched_responses(records)
    return records, {"rejoined": rejoined, "unmatched_responses_dropped": dropped}


# ----------------------------------------------------- record -> row --

def records_to_frame(records):
    """One row per Zeek DNS record: features and per-record bookkeeping.

    response_latency (Zeek's rtt) is kept for analysis but isn't a model
    feature: it measures the resolver and network the traffic went through.
    """
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
    """Parse one capture into a per-record DataFrame.

    `dns_log_path` is one dns.log, or a list of dns.log files from
    consecutive live chunks, which are treated as one capture. One row per
    record left after prepare_records, with lexical and response features,
    window assignment and bookkeeping columns (ts, capture_id, category,
    tool, base_domain, ...). Domain aggregates are added by
    add_domain_aggregates. ``df.attrs`` holds the number of malformed lines
    skipped, split transactions merged and unmatched responses dropped.
    """
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

#: Model features by group, for ablation runs. Together the groups other
#: than "artefact_suspect" list every model feature exactly once.
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
    # Artefact-suspect: features whose GraphTunnel separation looks like a
    # property of how the captures were recorded rather than of tunnelling.
    # domain_qtype_diversity: the normal crawl asked only for A records while
    # the wildcard traffic asked for A and AAAA. no_response_ratio: the
    # wildcard capture left ~37% of queries unanswered, normal ~0%.
    "artefact_suspect": ["domain_qtype_diversity", "no_response_ratio"],
}

#: Every feature the extractor computes, in model order. Identifiers (IPs,
#: ports, query, uid, trans_id, capture_id, tool) are deliberately excluded
#: so a model can't memorise a host, session or domain string.
#: response_latency is excluded because it depends on the resolver and
#: network, not on the traffic.
ALL_FEATURE_COLUMNS = list(dict.fromkeys(c for group in FEATURE_GROUPS.values() for c in group))

#: Default model inputs: every feature except the artefact-suspect pair. In
#: run 1 (results/zeek_run.md) dropping domain_qtype_diversity and
#: no_response_ratio raised unseen-tool recall from 97.09% to 99.18% with
#: false positive rates of 0.00% on held-out normal and wildcard. Both are
#: still computed and stay available through FEATURE_SETS["all"].
FEATURE_COLUMNS = [c for c in ALL_FEATURE_COLUMNS if c not in FEATURE_GROUPS["artefact_suspect"]]

#: Feature sets for the ablation runs. "all" is run 1's default.
FEATURE_SETS = {
    "all": ALL_FEATURE_COLUMNS,
    "lexical_only": FEATURE_GROUPS["lexical"],
    "domain_volume_shape": FEATURE_GROUPS["domain_volume"] + FEATURE_GROUPS["domain_shape"],
    "all_minus_artefact_suspect": FEATURE_COLUMNS,
}
DEFAULT_FEATURE_SET = "all_minus_artefact_suspect"

#: Kept alongside the features for splitting and analysis, never modelled.
BOOKKEEPING_COLUMNS = [
    "ts", "window_start", "window_id", "window_query_count",
    "capture_id", "category", "tool", "base_domain", "uid", "qname", "rejoined",
    "response_latency",
]

# Value used for a response field when there is no (answered) response.
_MISSING_RESPONSE_VALUES = {
    "response_ancount": 0.0,
    "response_min_ttl": -1.0,
    "response_rcode": -1.0,
    "response_latency": -1.0,
}


def to_ml_frame(df, label_col="Label", extra_cols=(), features=FEATURE_COLUMNS):
    """Select the model columns and fill response fields that are missing
    when a query got no (answered) response. `features` picks a feature set
    (see FEATURE_SETS); `extra_cols` keeps bookkeeping columns (e.g.
    capture_id, window_id) for splitting and analysis. label_col=None
    leaves the label out (live data has none)."""
    cols = list(features) + [c for c in extra_cols if c not in features]
    if label_col is not None:
        cols.append(label_col)
    out = df[cols].copy()
    for column, value in _MISSING_RESPONSE_VALUES.items():
        if column in out:
            out[column] = out[column].fillna(value)
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


def find_live_sessions(zeek_dir=LIVE_ZEEK_DIR, idle_seconds=300, now=None):
    """Group live-pipeline chunk folders (datas/zeek/<chunk>/) into sessions.

    A session is one capture_live.py run: chunks with the same prefix whose
    dumpcap indices are consecutive when ordered by chunk timestamp (dumpcap
    restarts at 00001 on every run). A missing index ends the session. A
    chunk folder without dns.log is a chunk with no DNS traffic and stays in
    the session.

    A session is complete when a later session with the same prefix exists
    or its last chunk started more than `idle_seconds` before `now`
    (default: the current time); otherwise capture may still be running.

    Returns a list of dicts ordered by start time: session_id (the first
    chunk's name), chunks, dns_logs (existing files, in order), start and
    last_chunk_start (epoch seconds, from the chunk names, local time), and
    complete.
    """
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
    """Manifest rows for benign live-pipeline captures ("own_benign").

    One row per session from find_live_sessions (complete sessions only,
    unless include_incomplete). `zeek_dns_log` is the list of the session's
    chunk dns.log files; build_dataset_from_manifest treats them as one
    capture. Sessions without any dns.log are skipped.
    """
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
    """Fail loudly if a tool is used both for training and as an unseen tool.

    Tool names seen in the training categories (normal/tunnel/wildcard/
    own_benign) must not appear in the evaluation-only ones
    (unknownTunnel/crossEndPoint), and dns2tcp-key may only appear under
    unknownTunnel.
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
    """Build the combined feature table for every capture in the manifest.

    Rows come from read_manifest() and, for live captures,
    own_benign_manifest_rows(); a row's zeek_dns_log may be a list of
    consecutive chunk logs. Returns one DataFrame (one row per DNS record,
    domain aggregates included). ``df.attrs["capture_stats"]`` lists, per
    capture, the record count, malformed lines skipped, split transactions
    merged and unmatched responses dropped.
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
                  f"{capture_stats['rejoined']} rejoined, "
                  f"{capture_stats['unmatched_responses_dropped']} unmatched responses dropped, "
                  f"{capture_stats['malformed_lines']} malformed lines skipped")
        frames.append(frame)
        stats.append(capture_stats)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.attrs["capture_stats"] = stats
    return combined
