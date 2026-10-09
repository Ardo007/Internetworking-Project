"""dumpcap control for the web app (web version of Ardashes_scripts/capture_live.py).

Differences from the CLI script:
- chunks rotate on the clock (``-b interval:30``), so every chunk after the
  first starts at :00 or :30 and two chunks make one calendar minute -- the
  same epoch-aligned minute windows assign_windows uses for training data;
- interfaces are passed by device id (``\\Device\\NPF_{...}``) rather than
  by number, since Npcap renumbers interfaces whenever an adapter is added;
- dumpcap runs in its own process group so the app can stop it with
  CTRL_BREAK (dumpcap's Ctrl+C handling: it closes the current chunk
  cleanly) without the app's own Ctrl+C reaching it.
"""

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

# `dumpcap -D` line: "5. \Device\NPF_{GUID} (Wi-Fi)"
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
    """Parse `dumpcap -D` output into [{id, name}].

    On Windows only Npcap devices (\\Device\\NPF_...) are kept, which drops
    extcap pseudo-interfaces (etwdump, sshdump, ...) that aren't network
    adapters. Elsewhere every listed interface is kept.
    """
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
        "-q",  # no running packet count in dumpcap.log
        "-b", f"interval:{CHUNK_SECONDS}",
        "-b", f"files:{RING_BUFFER_FILES}",
        "-w", str(output_dir / f"{PREFIX}.pcapng"),
    ]


def start_capture(interface, output_dir, log_path, dumpcap=None):
    """Start dumpcap writing rotating chunks into output_dir; its console
    output goes to log_path. Returns the Popen."""
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
    """Ask dumpcap to stop (it closes the current chunk first); kill it if
    it hasn't exited after `timeout` seconds."""
    if process.poll() is not None:
        return process.returncode
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        process.terminate()
        return process.wait(timeout=timeout)
