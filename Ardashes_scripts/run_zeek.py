"""Feed capture chunks from datas/captures through a long-lived Zeek container.

Designed to run alongside capture_live.py: dumpcap keeps rotating chunks
into datas/captures while this script watches that folder and processes
each chunk as it becomes safe to read.

Sealing rule: a ring-buffer chunk is only read once dumpcap has moved on to
the next one in its sequence (chunk N is processed once chunk N+1 exists).
Reading a chunk dumpcap still has open would race its writes. Files that
don't match dumpcap's ring-buffer naming (e.g. a capture dropped in by
hand) are treated as already complete.

A single Zeek container is started once and reused via `docker exec` for
every chunk, instead of paying the 1-3s `docker run` startup cost on every
rotation.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIRECTORY = PROJECT_ROOT / "datas" / "captures"
ZEEK_DIRECTORY = PROJECT_ROOT / "datas" / "zeek"
ZEEK_IMAGE = "zeek/zeek:lts"
CONTAINER_NAME = "dns-tunnel-live-zeek"
SUPPORTED_SUFFIXES = {".pcap", ".pcapng"}
DEFAULT_POLL_INTERVAL_SECONDS = 5

# dumpcap ring-buffer naming: <prefix>_<00001>_<YYYYmmddHHMMSS>.<ext>
RING_BUFFER_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<index>\d{5})_(?P<timestamp>\d{14})$"
)


def check_docker():
    if shutil.which("docker") is None:
        raise RuntimeError("Docker was not found. Install or start Docker Desktop.")

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Docker did not respond within 15 seconds. "
            "Wait for Docker Desktop to finish starting and try again."
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Docker is installed but its engine is unavailable. "
            f"Start Docker Desktop and try again. {detail}"
        )


def start_container():
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    mount = f"type=bind,source={PROJECT_ROOT},target=/work"
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--mount", mount,
            ZEEK_IMAGE,
            "sleep", "infinity",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def stop_container():
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )


def usable_output(output_directory):
    if not output_directory.is_dir():
        return False
    return any(
        log.stat().st_size > 0 for log in output_directory.glob("*.log")
    )


def discover_sealed_captures(capture_directory=CAPTURE_DIRECTORY):
    """Return capture files that are safe to read, oldest first.

    A file matching dumpcap's ring-buffer naming is sealed once the next
    index in its sequence exists. Files with any other name are treated as
    already complete (manual drop-ins, single imported pcaps, etc).
    """
    if not capture_directory.is_dir():
        return []

    all_files = sorted(
        (
            path
            for path in capture_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )

    highest_index = {}
    for path in all_files:
        match = RING_BUFFER_PATTERN.match(path.stem)
        if not match:
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        highest_index[prefix] = max(highest_index.get(prefix, -1), index)

    sealed = []
    for path in all_files:
        match = RING_BUFFER_PATTERN.match(path.stem)
        if not match:
            sealed.append(path)
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        if index < highest_index[prefix]:
            sealed.append(path)

    return sealed


def process_capture(capture, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    container_capture = "/work/" + capture.relative_to(PROJECT_ROOT).as_posix()
    container_output = "/work/" + output_directory.relative_to(PROJECT_ROOT).as_posix()

    subprocess.run(
        [
            "docker", "exec",
            "-w", container_output,
            CONTAINER_NAME,
            "zeek", "-C", "-r", container_capture,
            "LogAscii::use_json=T",
        ],
        check=True,
    )
    return sorted(output_directory.glob("*.log"), key=lambda path: path.name.casefold())


def run_pass(capture_directory, zeek_directory):
    processed = 0
    for capture in discover_sealed_captures(capture_directory):
        output_directory = zeek_directory / capture.stem
        if usable_output(output_directory):
            continue

        print(f"[zeek] {capture.name}")
        logs = process_capture(capture, output_directory)
        if logs:
            for log in logs:
                print(f"  {log}")
        else:
            print("  Zeek completed but did not create any log files.")
        processed += 1

    return processed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir", type=Path, default=CAPTURE_DIRECTORY,
        help="Folder to watch for capture chunks",
    )
    parser.add_argument(
        "--zeek-dir", type=Path, default=ZEEK_DIRECTORY,
        help="Folder to write per-chunk Zeek logs into",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Seconds between polls in watch mode (default: {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process currently sealed chunks once and exit, instead of watching",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_docker()
    start_container()

    try:
        if args.once:
            processed = run_pass(args.capture_dir, args.zeek_dir)
            print(f"Complete: {processed} chunk(s) processed.")
            return

        print(f"[zeek] watching {args.capture_dir} (Ctrl+C to stop)")
        while True:
            processed = run_pass(args.capture_dir, args.zeek_dir)
            if processed == 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[zeek] stopping...")
    finally:
        stop_container()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
