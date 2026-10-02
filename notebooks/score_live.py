"""
Continuously score live DNS traffic from capture_live.py + run_zeek.py.

run_zeek.py writes one Zeek output folder per 30-second dumpcap chunk
(datas/zeek/<chunk>/dns.log). Per zeek_feature_extraction's "Live scoring"
note, aggregates computed on a single 30s chunk would not match what the
model was trained on (WINDOW_SECONDS=60) -- so this script watches
datas/zeek/, waits for chunks to arrive in consecutive pairs (2 x 30s =
60s), and scores each pair together as one capture, printing only domains
flagged TUNNEL as they're found. Ctrl+C stops it and scores any leftover
unpaired chunk before exiting.

Usage (run alongside capture_live.py and run_zeek.py, each in their own
terminal):
  python notebooks\\score_live.py [--model-dir DIR] [--zeek-dir DIR]
                                   [--threshold T] [--interval SECONDS]
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeek_feature_extraction as zfe
import model_artifacts as ma
import score_capture as sc

#: A capture_live.py run is a "session" in find_live_sessions' terms: chunks
#: with consecutive dumpcap indices. dumpcap restarts its index at 00001 on
#: every run, so pairing chunks by raw index alone (ignoring which run they
#: came from) can join a stale leftover chunk from a previous run with a
#: fresh one that happens to share an index number. find_live_sessions
#: already solves this the same way the training-side own_benign pipeline
#: does, so reuse it here instead of re-deriving the grouping.
IDLE_SECONDS = 300

DEFAULT_LOG_PATH = zfe.PROJECT_ROOT / "results" / "live_scoring_log.csv"
LOG_COLUMNS = ["scored_at", "session_id", "window_name", "window_id", "base_domain",
               "n_queries", "mean_tunnel_prob", "n_flagged", "verdict"]


def open_log(log_path):
    """Open log_path for appending, writing a header only if it's new/empty.
    Rows from every run of this script accumulate in the same file, so it
    reads as one continuous history rather than a scattered file per run."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists() or log_path.stat().st_size == 0
    file = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
    if is_new:
        writer.writeheader()
        file.flush()
    return file, writer


def log_summary(log, session_id, window_name, summary):
    if log is None or summary.empty:
        return
    file, writer = log
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for _, row in summary.iterrows():
        writer.writerow({
            "scored_at": scored_at,
            "session_id": session_id,
            "window_name": window_name,
            "window_id": row["window_id"],
            "base_domain": row["base_domain"],
            "n_queries": row["n_queries"],
            "mean_tunnel_prob": row["mean_tunnel_prob"],
            "n_flagged": row["n_flagged"],
            "verdict": row["verdict"],
        })
    file.flush()


def score_window(chunk_folders, window_name, artifacts, threshold, session_id=None, log=None):
    dns_logs = [f / "dns.log" for f in chunk_folders if (f / "dns.log").is_file()]
    if not dns_logs:
        print(f"[{window_name}] no DNS traffic in this window", flush=True)
        return

    frame = zfe.extract_query_records_from_zeek(
        dns_logs, window_name, category="own_benign", tool="live", label=0,
    )
    if frame.empty:
        print(f"[{window_name}] no usable DNS records", flush=True)
        return
    # mDNS/DNS-SD (RFC 6762 ".local", multicast port 5353) never leaves the
    # local link -- it can't carry a tunnel payload to an off-network
    # server, and it isn't represented in GraphTunnel's training captures,
    # so the model has no real prior on it. Drop it before scoring rather
    # than let it land near the decision boundary.
    frame = frame[~frame["qname"].str.casefold().str.endswith(".local")]
    if frame.empty:
        print(f"[{window_name}] no usable DNS records (only mDNS/.local traffic)", flush=True)
        return
    frame = zfe.add_domain_aggregates(frame)

    ml = zfe.to_ml_frame(
        frame, label_col=None,
        extra_cols=["capture_id", "window_start", "window_id", "base_domain"],
        features=artifacts["features"]["feature_columns"],
    )
    tensor = ma.prepare_model_input(frame, artifacts)
    probs = np.stack([m.predict(tensor, verbose=0)[:, 1] for m in artifacts["models"]], axis=0)
    row_prob = probs.mean(axis=0)

    per_row = ml[["window_id", "base_domain"]].copy()
    per_row["tunnel_prob"] = row_prob
    per_row["flagged"] = row_prob >= threshold

    summary = per_row.groupby(["window_id", "base_domain"], as_index=False).agg(
        n_queries=("tunnel_prob", "size"),
        mean_tunnel_prob=("tunnel_prob", "mean"),
        n_flagged=("flagged", "sum"),
    )
    summary["verdict"] = np.where(summary["mean_tunnel_prob"] >= threshold, "TUNNEL", "benign")
    summary = summary.sort_values("mean_tunnel_prob", ascending=False)
    log_summary(log, session_id, window_name, summary)

    flagged = summary[summary["verdict"] == "TUNNEL"]
    if len(flagged):
        print(f"[{window_name}] {len(flagged)}/{len(summary)} domain(s) flagged TUNNEL:", flush=True)
        with pd.option_context("display.max_rows", 20, "display.width", 160):
            print(flagged.to_string(index=False))
    elif len(summary):
        print(f"[{window_name}] {len(summary)} domain(s), all benign "
              f"(max prob {summary['mean_tunnel_prob'].max():.4f})", flush=True)
    else:
        print(f"[{window_name}] no domains scored", flush=True)


def watch(zeek_dir, artifacts, threshold, interval, log_path=None):
    cursor = {}  # session_id -> number of that session's chunks already scored
    log = None
    if log_path is not None:
        log_file, log_writer = open_log(log_path)
        log = (log_file, log_writer)
        print(f"[score] logging every scored window to {log_path}", flush=True)
    print(f"[score] watching {zeek_dir} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            for session in zfe.find_live_sessions(zeek_dir, idle_seconds=IDLE_SECONDS):
                sid = session["session_id"]
                names = session["chunks"][cursor.get(sid, 0):]
                while len(names) >= 2:
                    pair_names, names = names[:2], names[2:]
                    folders = [zeek_dir / name for name in pair_names]
                    window_name = "+".join(pair_names)
                    score_window(folders, window_name, artifacts, threshold, sid, log)
                    cursor[sid] = cursor.get(sid, 0) + 2
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[score] stopping...", flush=True)
        for session in zfe.find_live_sessions(zeek_dir, idle_seconds=IDLE_SECONDS):
            sid = session["session_id"]
            names = session["chunks"][cursor.get(sid, 0):]
            if names:
                folders = [zeek_dir / name for name in names]
                window_name = "+".join(names)
                print(f"[score] flushing leftover chunk(s) for {sid}: {window_name}", flush=True)
                score_window(folders, window_name, artifacts, threshold, sid, log)
    finally:
        if log is not None:
            log[0].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=None,
                        help="Model artifacts folder (default: newest models/zeek_bilstm/*/final)")
    parser.add_argument("--zeek-dir", default=None,
                        help="Folder to watch for per-chunk Zeek output (default: datas/zeek)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    parser.add_argument("--log-file", default=None,
                        help=f"CSV to append every scored window/domain row to "
                             f"(default: {DEFAULT_LOG_PATH})")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else sc.default_model_dir()
    artifacts = ma.load_artifacts(model_dir)
    if artifacts["features"].get("not_for_evaluation"):
        print(f"[using '{Path(model_dir).parent.name}/final' -- the deploy model, "
              f"trained on every GraphTunnel capture]\n", flush=True)

    zeek_dir = Path(args.zeek_dir) if args.zeek_dir else zfe.LIVE_ZEEK_DIR
    log_path = None if args.no_log else Path(args.log_file) if args.log_file else DEFAULT_LOG_PATH
    watch(zeek_dir, artifacts, args.threshold, args.interval, log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
