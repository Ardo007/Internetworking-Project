import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dpkt

import process_pcaps
from process_pcaps import MANIFEST_PATH, PROJECT_ROOT, RAW_ROOT, ZEEK_ROOT, usable_log


INTERIM_ROOT = PROJECT_ROOT / "Data" / "interim" / "GraphTunnel"
REPORT_PATH = INTERIM_ROOT / "reorder_report.csv"
BACKUP_SUFFIX = "_Backup"

WINDOWS_REORDERCAP_LOCATIONS = [
    Path(r"C:\Program Files\Wireshark\reordercap.exe"),
    Path(r"C:\Program Files (x86)\Wireshark\reordercap.exe"),
]


def find_reordercap():
    on_path = shutil.which("reordercap")
    if on_path:
        return Path(on_path)
    for candidate in WINDOWS_REORDERCAP_LOCATIONS:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "reordercap was not found. Install Wireshark (which bundles "
        "reordercap) or add reordercap to PATH."
    )


def open_reader(file):
    magic = file.read(4)
    file.seek(0)
    if magic == b"\x0a\x0d\x0d\x0a":
        return dpkt.pcapng.Reader(file)
    return dpkt.pcap.Reader(file)


def scan_order(pcap):
    packets = 0
    backwards = 0
    max_lag = 0.0
    newest = None
    previous = None
    with pcap.open("rb") as file:
        for timestamp, _ in open_reader(file):
            packets += 1
            if previous is not None and timestamp < previous:
                backwards += 1
            if newest is not None and timestamp < newest:
                max_lag = max(max_lag, newest - timestamp)
            newest = timestamp if newest is None else max(newest, timestamp)
            previous = timestamp
    return packets, backwards, max_lag


def read_manifest():
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def paths_for(row):
    raw_pcap = PROJECT_ROOT / row["raw_pcap"]
    relative = raw_pcap.relative_to(RAW_ROOT)
    sorted_pcap = INTERIM_ROOT / relative
    output_directory = ZEEK_ROOT / relative.parent / raw_pcap.stem
    backup_directory = output_directory.with_name(output_directory.name + BACKUP_SUFFIX)
    return raw_pcap, sorted_pcap, output_directory, backup_directory


def check_capture(row):
    raw_pcap, _, _, _ = paths_for(row)
    packets, backwards, max_lag = scan_order(raw_pcap)
    return {
        "capture_id": row["capture_id"],
        "category": row["category"],
        "packets": packets,
        "backwards_steps": backwards,
        "max_lag_s": round(max_lag, 6),
    }


def reprocess_capture(row, reordercap):
    raw_pcap, sorted_pcap, output_directory, backup_directory = paths_for(row)
    output_ready = usable_log(output_directory / "dns.log") and usable_log(
        output_directory / "conn.log"
    )
    if backup_directory.exists() and output_ready:
        return "skipped (already reprocessed)"

    sorted_pcap.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(reordercap), str(raw_pcap), str(sorted_pcap)],
        check=True,
        capture_output=True,
        text=True,
    )
    packets, backwards, _ = scan_order(sorted_pcap)
    expected_packets, _, _ = scan_order(raw_pcap)
    if backwards or packets != expected_packets:
        raise RuntimeError(
            f"Sorted copy is not usable: {sorted_pcap} "
            f"({packets}/{expected_packets} packets, {backwards} backwards steps)"
        )

    if not backup_directory.exists():
        if not output_directory.is_dir():
            raise FileNotFoundError(
                f"No Zeek output to back up for {row['capture_id']}: {output_directory}. "
                "Run process_pcaps.py first."
            )
        output_directory.rename(backup_directory)

    print(f"[zeek] {sorted_pcap.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    process_pcaps.run_zeek(sorted_pcap, output_directory)
    if not usable_log(output_directory / "dns.log") or not usable_log(
        output_directory / "conn.log"
    ):
        raise RuntimeError(
            f"Zeek did not create usable dns.log and conn.log files for {sorted_pcap}"
        )
    return "reordered + zeek re-run"


def write_report(rows):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["capture_id", "category", "packets", "backwards_steps", "max_lag_s", "action"]
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only report which captures are out of order; change nothing",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Captures scanned / reprocessed in parallel (default: 4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = read_manifest()

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        checks = list(pool.map(check_capture, manifest))

    out_of_order = [c for c in checks if c["backwards_steps"]]
    print(f"{len(out_of_order)} of {len(checks)} captures have out-of-order timestamps:")
    for check in out_of_order:
        print(
            f"  {check['capture_id']}: {check['backwards_steps']} backwards steps, "
            f"max lag {check['max_lag_s']:.6f}s"
        )

    if args.check_only:
        return

    reordercap = find_reordercap()
    process_pcaps.check_docker()
    affected = {c["capture_id"] for c in out_of_order}
    rows_to_process = [row for row in manifest if row["capture_id"] in affected]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        actions = dict(
            zip(
                (row["capture_id"] for row in rows_to_process),
                pool.map(reprocess_capture, rows_to_process, [reordercap] * len(rows_to_process)),
            )
        )

    for check in checks:
        check["action"] = actions.get(check["capture_id"], "in order")
    write_report(checks)
    print(f"Complete: {sum(a.startswith('reordered') for a in actions.values())} reprocessed, "
          f"{sum(a.startswith('skipped') for a in actions.values())} skipped.")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
