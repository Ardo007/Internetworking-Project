"""Zeek for the web app (web version of Ardashes_scripts/run_zeek.py).

Same approach as the CLI script: one long-lived zeek/zeek:lts container
with the project folder mounted at /work, and one `docker exec zeek -r`
per capture chunk. It uses its own container name, so starting or stopping
the web app never removes a container the CLI pipeline is using.
"""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZEEK_IMAGE = "zeek/zeek:lts"
CONTAINER_NAME = "dns-tunnel-web-zeek"
SUPPORTED_SUFFIXES = {".pcap", ".pcapng"}

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
        raise RuntimeError(
            "Docker is installed but its engine is unavailable. "
            "Start Docker Desktop and try again."
        )


def container_running():
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def start_container():
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    mount = f"type=bind,source={PROJECT_ROOT},target=/work"
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--mount", mount,
            ZEEK_IMAGE,
            "sleep", "infinity",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not start the Zeek container: {result.stderr.strip()}")


def stop_container():
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )


def chunk_start(path):
    """Epoch seconds a dumpcap chunk started at, from its file name
    (local time, as dumpcap writes it), or None for other names."""
    match = RING_BUFFER_PATTERN.match(Path(path).stem)
    if not match:
        return None
    return datetime.strptime(match["timestamp"], "%Y%m%d%H%M%S").timestamp()


def discover_sealed_captures(capture_directory, all_sealed=False):
    """Return capture chunks that are safe to read, oldest first.

    While dumpcap is running, chunk N is sealed once chunk N+1 exists.
    all_sealed=True (dumpcap has exited) returns every chunk.
    """
    if not capture_directory.is_dir():
        return []

    chunks = sorted(
        (
            path
            for path in capture_directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() in SUPPORTED_SUFFIXES
            and RING_BUFFER_PATTERN.match(path.stem)
        ),
        key=lambda path: int(RING_BUFFER_PATTERN.match(path.stem)["index"]),
    )
    return chunks if all_sealed else chunks[:-1]


def newest_chunk(capture_directory):
    """The chunk dumpcap is currently writing (highest index), or None."""
    chunks = discover_sealed_captures(capture_directory, all_sealed=True)
    return chunks[-1] if chunks else None


def process_capture(capture, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    container_capture = "/work/" + capture.relative_to(PROJECT_ROOT).as_posix()
    container_output = "/work/" + output_directory.relative_to(PROJECT_ROOT).as_posix()

    result = subprocess.run(
        [
            "docker", "exec",
            "-w", container_output,
            CONTAINER_NAME,
            "zeek", "-C", "-r", container_capture,
            "LogAscii::use_json=T",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Zeek failed on {capture.name}: {result.stderr.strip() or result.stdout.strip()}"
        )
    dns_log = output_directory / "dns.log"
    return dns_log if dns_log.is_file() else None
