"""Continuously capture live traffic to disk using dumpcap's ring buffer.

Chunk rotation is kept short (default 30s) so files become available for
downstream processing quickly. The statistical analysis window that later
consumes these chunks is a separate concern and spans many of them -- this
script only produces the raw chunks. dumpcap performs rotation, filenames,
and disk-capping itself, so no timing loop is needed here and there is no
packet loss at rotation boundaries.

Rotated files land in the output folder using dumpcap's own naming
convention: ``<prefix>_<00001>_<YYYYmmddHHMMSS>.pcapng``. Once the ring
buffer cap is reached, dumpcap deletes the oldest chunk before starting the
next one, so the chunk-processing pipeline must keep up with capture speed
or it will lose unprocessed chunks.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "datas" / "captures"
DEFAULT_DURATION_SECONDS = 30
DEFAULT_RING_BUFFER_FILES = 120
DEFAULT_PREFIX = "capture"

WINDOWS_DUMPCAP_LOCATIONS = [
    Path(r"C:\Program Files\Wireshark\dumpcap.exe"),
    Path(r"C:\Program Files (x86)\Wireshark\dumpcap.exe"),
]


def find_dumpcap():
    on_path = shutil.which("dumpcap")
    if on_path:
        return Path(on_path)
    for candidate in WINDOWS_DUMPCAP_LOCATIONS:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "dumpcap was not found. Install Wireshark (which bundles dumpcap "
        "and the Npcap driver) or add dumpcap to PATH."
    )


def list_interfaces(dumpcap):
    subprocess.run([str(dumpcap), "-D"], check=True)


def build_command(dumpcap, interface, duration, files, output_dir, prefix):
    return [
        str(dumpcap),
        "-i", interface,
        "-b", f"duration:{duration}",
        "-b", f"files:{files}",
        "-w", str(output_dir / f"{prefix}.pcapng"),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface",
        help="Interface number or name, as shown by --list-interfaces",
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List capture interfaces and exit",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Seconds per rotated chunk (default: {DEFAULT_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=DEFAULT_RING_BUFFER_FILES,
        help=(
            "Ring buffer size in chunks; dumpcap deletes the oldest chunk "
            f"past this cap (default: {DEFAULT_RING_BUFFER_FILES})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Folder to write capture chunks into",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Base filename for capture chunks",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dumpcap = find_dumpcap()

    if args.list_interfaces:
        list_interfaces(dumpcap)
        return 0

    if not args.interface:
        raise RuntimeError(
            "--interface is required (use --list-interfaces to see choices)"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        dumpcap, args.interface, args.duration, args.files, args.output_dir, args.prefix
    )

    print(f"[capture] {' '.join(command)}")
    print(f"[capture] writing rotated chunks to {args.output_dir} (Ctrl+C to stop)")

    process = subprocess.Popen(command)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("[capture] stopping (letting dumpcap close its current chunk)...")
        process.wait()

    return process.returncode


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
