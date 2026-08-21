"""Process GraphTunnel PCAPs with Zeek and generate a label manifest."""

import csv
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "Data" / "raw" / "GraphTunnel"
ZEEK_ROOT = PROJECT_ROOT / "Data" / "zeek" / "GraphTunnel"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "GraphTunnel"
    / "capture_manifest.csv"
)
ZEEK_IMAGE = "zeek/zeek:lts"

LABELS = {
    "normal": 0,
    "wildcard": 0,
    "tunnel": 1,
    "unknowntunnel": 1,
    "unkowntunnel": 1,  # Spelling used by the source repository.
    "crossendpoint": 1,
}


def usable_log(path):
    return path.is_file() and path.stat().st_size > 0


def check_docker():
    if shutil.which("docker") is None:
        raise RuntimeError("Docker was not found. Install or start Docker Desktop.")

    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            "Docker is installed but its engine is unavailable. "
            f"Start Docker Desktop and try again. {detail}"
        )


def run_zeek(pcap, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    container_pcap = "/work/" + pcap.relative_to(PROJECT_ROOT).as_posix()
    container_output = (
        "/work/" + output_directory.relative_to(PROJECT_ROOT).as_posix()
    )
    mount = f"type=bind,source={PROJECT_ROOT},target=/work"

    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--mount",
            mount,
            "-w",
            container_output,
            ZEEK_IMAGE,
            "zeek",
            "-C",
            "-r",
            container_pcap,
            "LogAscii::use_json=T",
        ],
        check=True,
    )


def write_manifest(rows):
    columns = [
        "capture_id",
        "category",
        "tool",
        "label",
        "pcap_bytes",
        "raw_pcap",
        "zeek_dns_log",
    ]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(MANIFEST_PATH)


def main():
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Raw GraphTunnel folder not found: {RAW_ROOT}")

    pcaps = sorted(
        (path for path in RAW_ROOT.rglob("*") if path.suffix.casefold() == ".pcap"),
        key=lambda path: str(path).casefold(),
    )
    if not pcaps:
        raise FileNotFoundError(f"No PCAP files found under: {RAW_ROOT}")

    rows = []
    docker_checked = False
    processed = 0
    skipped = 0

    for pcap in pcaps:
        relative_pcap = pcap.relative_to(RAW_ROOT)
        if len(relative_pcap.parts) < 2:
            raise ValueError(f"PCAP must be inside a category folder: {pcap}")

        category = relative_pcap.parts[0]
        category_key = category.casefold()
        if category_key not in LABELS:
            raise ValueError(f"Unknown GraphTunnel category: {category}")

        output_directory = ZEEK_ROOT / relative_pcap.parent / pcap.stem
        dns_log = output_directory / "dns.log"
        conn_log = output_directory / "conn.log"

        if usable_log(dns_log) and usable_log(conn_log):
            print(f"[skip] {relative_pcap.as_posix()}")
            skipped += 1
        else:
            if not docker_checked:
                check_docker()
                docker_checked = True
            print(f"[zeek] {relative_pcap.as_posix()}")
            run_zeek(pcap, output_directory)
            if not usable_log(dns_log) or not usable_log(conn_log):
                raise RuntimeError(
                    f"Zeek did not create usable dns.log and conn.log files for {pcap}"
                )
            processed += 1

        if category_key in {"normal", "wildcard"}:
            tool = category
        elif len(relative_pcap.parts) > 2:
            tool = relative_pcap.parts[1]
        else:
            tool = pcap.stem

        rows.append(
            {
                "capture_id": relative_pcap.with_suffix("").as_posix(),
                "category": category,
                "tool": tool,
                "label": LABELS[category_key],
                "pcap_bytes": pcap.stat().st_size,
                "raw_pcap": pcap.relative_to(PROJECT_ROOT).as_posix(),
                "zeek_dns_log": dns_log.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    write_manifest(rows)
    print(f"Complete: {processed} processed, {skipped} skipped.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
