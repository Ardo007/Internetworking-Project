"""Run Zeek (via Docker) over an external PCAP folder, independent of the
GraphTunnel training manifest.

This dataset (and others like it) splits each capture into two one-way
files, "<base>_up.pcapng" (client->server) and "<base>_down.pcapng"
(server->client), captured from each host separately. Zeek needs to see
both directions in one file to pair a query with its response (otherwise
every record comes out query-only, with no response/rcode/TTL features),
so before running Zeek this script merges each *_up/*_down pair with
mergecap (bundled with Wireshark, same installation capture_live.py
already looks for) into "<base>.pcapng", and runs Zeek on the merged file.
A file with no up/down counterpart (nothing to merge) is Zeek'd as-is.

Expects PCAPs under Data/raw/<dataset_name>/<tool>/<file>.pcap(ng), writes
Zeek JSON logs to Data/zeek/<dataset_name>/<tool>/<file>/dns.log.

Usage:
  python Ardashes_scripts\\process_external_pcaps.py <dataset_name>
  (e.g. python Ardashes_scripts\\process_external_pcaps.py external_tu2023)
"""
import re
import time
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZEEK_IMAGE = "zeek/zeek:lts"

WINDOWS_MERGECAP_LOCATIONS = [
    Path(r"C:\Program Files\Wireshark\mergecap.exe"),
    Path(r"C:\Program Files (x86)\Wireshark\mergecap.exe"),
]
_UP_DOWN = re.compile(r"^(?P<base>.+)_(?P<direction>up|down)$", re.IGNORECASE)


def find_mergecap():
    on_path = shutil.which("mergecap")
    if on_path:
        return Path(on_path)
    for candidate in WINDOWS_MERGECAP_LOCATIONS:
        if candidate.is_file():
            return candidate
    return None


def usable_log(path):
    return path.is_file() and path.stat().st_size > 0


def check_docker():
    if shutil.which("docker") is None:
        raise RuntimeError("Docker was not found. Install or start Docker Desktop.")
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Docker is installed but its engine is unavailable. Start Docker Desktop and try again. {detail}")


def run_zeek(pcap, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    container_pcap = "/work/" + pcap.relative_to(PROJECT_ROOT).as_posix()
    container_output = "/work/" + output_directory.relative_to(PROJECT_ROOT).as_posix()
    mount = f"type=bind,source={PROJECT_ROOT},target=/work"
    subprocess.run(
        ["docker", "run", "--rm", "--mount", mount, "-w", container_output,
         ZEEK_IMAGE, "zeek", "-C", "-r", container_pcap, "LogAscii::use_json=T"],
        check=True,
    )


def pair_up_down(pcaps, merged_dir, mergecap):
    """Group *_up/*_down pairs and merge each into merged_dir. Returns the
    list of PCAPs to actually run Zeek on (merged pairs + unpaired files)."""
    by_base = {}
    singles = []
    for pcap in pcaps:
        m = _UP_DOWN.match(pcap.stem)
        if m:
            by_base.setdefault((pcap.parent, m.group("base")), {})[m.group("direction").lower()] = pcap
        else:
            singles.append(pcap)

    to_process = list(singles)
    for (parent, base), sides in by_base.items():
        if "up" in sides and "down" in sides and mergecap:
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged = merged_dir / f"{base}.pcapng"
            if not usable_log(merged):
                print(f"[merge] {base}")
                subprocess.run(
                    [str(mergecap), "-w", str(merged), str(sides["up"]), str(sides["down"])],
                    check=True,
                )
                # Docker Desktop (WSL2) bind mounts can lag slightly behind a
                # just-written host file -- wait for it to actually appear
                # and settle (size stops growing) before handing it to Zeek.
                for attempt in range(20):
                    if usable_log(merged):
                        size1 = merged.stat().st_size
                        time.sleep(0.5)
                        if merged.stat().st_size == size1:
                            break
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"mergecap did not produce a stable output file: {merged}")
            to_process.append(merged)
        else:
            # No counterpart (or no mergecap available) -- process one-way as-is.
            to_process.extend(sides.values())
    return sorted(to_process)


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_external_pcaps.py <dataset_name>", file=sys.stderr)
        sys.exit(1)
    dataset_name = sys.argv[1]
    raw_root = PROJECT_ROOT / "Data" / "raw" / dataset_name
    zeek_root = PROJECT_ROOT / "Data" / "zeek" / dataset_name
    merged_root = PROJECT_ROOT / "Data" / "interim" / f"{dataset_name}_merged"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Not found: {raw_root}")

    mergecap = find_mergecap()
    if not mergecap:
        print("[warn] mergecap not found -- up/down pairs will be processed one-way "
              "(response features will be mostly absent for those captures)")

    all_pcaps = sorted(
        p for p in raw_root.rglob("*") if p.suffix.casefold() in (".pcap", ".pcapng")
    )
    if not all_pcaps:
        raise FileNotFoundError(f"No PCAP files found under: {raw_root}")

    # Group+merge per tool subfolder so merged files land next to their source.
    by_folder = {}
    for p in all_pcaps:
        by_folder.setdefault(p.parent, []).append(p)

    pcaps = []
    for folder, group in by_folder.items():
        rel = folder.relative_to(raw_root)
        pcaps.extend(pair_up_down(group, merged_root / rel, mergecap))

    docker_checked = False
    processed = skipped = 0
    for pcap in pcaps:
        try:
            relative = pcap.relative_to(raw_root)
        except ValueError:
            relative = pcap.relative_to(merged_root)
        output_directory = zeek_root / relative.parent / pcap.stem
        dns_log = output_directory / "dns.log"
        if usable_log(dns_log):
            print(f"[skip] {relative.as_posix()}")
            skipped += 1
            continue
        if not docker_checked:
            check_docker()
            docker_checked = True
        print(f"[zeek] {relative.as_posix()}")
        run_zeek(pcap, output_directory)
        if not usable_log(dns_log):
            print(f"  (no DNS traffic found in {pcap.name})")
        processed += 1

    print(f"\nComplete: {processed} processed, {skipped} skipped.")
    print(f"Zeek logs under: {zeek_root}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
