import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path


CHUNK_SECONDS = 30
RING_BUFFER_FILES = 120
PREFIX = "capture"

WINDOWS_DUMPCAP_LOCATIONS = [
    Path(r"C:\Program Files\Wireshark\dumpcap.exe"),
    Path(r"C:\Program Files (x86)\Wireshark\dumpcap.exe"),
]

_INTERFACE_LINE = re.compile(r"^\s*\d+\.\s+(?P<id>\S+)(?:\s+\((?P<name>.*)\))?\s*$")


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


def parse_interfaces(text):
    interfaces = []
    for line in text.splitlines():
        match = _INTERFACE_LINE.match(line)
        if not match:
            continue
        device = match["id"]
        if sys.platform == "win32" and not device.startswith("\\Device\\NPF_"):
            continue
        interfaces.append({"id": device, "name": match["name"] or device})
    return interfaces


def list_interfaces(dumpcap=None):
    dumpcap = dumpcap or find_dumpcap()
    result = subprocess.run([str(dumpcap), "-D"], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"dumpcap -D failed: {result.stderr.strip() or result.stdout.strip()}")
    return parse_interfaces(result.stdout)


def build_command(dumpcap, interface, output_dir):
    return [
        str(dumpcap),
        "-i", interface,
        "-q",
        "-b", f"interval:{CHUNK_SECONDS}",
        "-b", f"files:{RING_BUFFER_FILES}",
        "-w", str(output_dir / f"{PREFIX}.pcapng"),
    ]


def start_capture(interface, output_dir, log_path, dumpcap=None):
    dumpcap = dumpcap or find_dumpcap()
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.Popen(
            build_command(dumpcap, interface, output_dir),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=flags,
        )


def stop_capture(process, timeout=10):
    if process.poll() is not None:
        return process.returncode
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        return process.wait(timeout=timeout)
