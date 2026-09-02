"""Run Zeek on every finished capture in datas/captures."""

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIRECTORY = PROJECT_ROOT / "datas" / "captures"
ZEEK_DIRECTORY = PROJECT_ROOT / "datas" / "zeek"
ZEEK_IMAGE = "zeek/zeek:lts"
SUPPORTED_SUFFIXES = {".pcap", ".pcapng"}


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


def prepare_paths(capture, output_directory):
    capture = capture.resolve()
    output_directory = output_directory.resolve()

    if not capture.is_file():
        raise FileNotFoundError(f"Capture file not found: {capture}")
    if capture.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Capture must be a PCAP or PCAPNG file: {capture}")

    if output_directory.exists():
        if not output_directory.is_dir():
            raise ValueError(f"Output path is not a folder: {output_directory}")
        if any(output_directory.iterdir()):
            raise ValueError(f"Output folder must be empty: {output_directory}")
    else:
        output_directory.mkdir(parents=True)

    return capture, output_directory


def discover_captures(capture_directory=CAPTURE_DIRECTORY):
    if not capture_directory.is_dir():
        raise FileNotFoundError(f"Capture folder not found: {capture_directory}")

    captures = sorted(
        (
            path
            for path in capture_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not captures:
        raise FileNotFoundError(
            f"No PCAP or PCAPNG files found in: {capture_directory}"
        )
    return captures


def prepare_jobs(captures, zeek_directory=ZEEK_DIRECTORY):
    jobs = []
    output_directories = set()

    for capture in captures:
        output_directory = zeek_directory / capture.stem
        if output_directory in output_directories:
            raise ValueError(
                f"Multiple captures would use the same output folder: {output_directory}"
            )
        output_directories.add(output_directory)
        jobs.append(prepare_paths(capture, output_directory))

    return jobs


def docker_command(capture, output_directory):
    input_mount = f"type=bind,source={capture.parent},target=/input,readonly"
    output_mount = f"type=bind,source={output_directory},target=/output"
    container_capture = f"/input/{capture.name}"

    return [
        "docker",
        "run",
        "--rm",
        "--mount",
        input_mount,
        "--mount",
        output_mount,
        "-w",
        "/output",
        ZEEK_IMAGE,
        "zeek",
        "-C",
        "-r",
        container_capture,
        "LogAscii::use_json=T",
    ]


def process_capture(capture, output_directory):
    subprocess.run(docker_command(capture, output_directory), check=True)
    return sorted(output_directory.glob("*.log"), key=lambda path: path.name.casefold())


def main():
    jobs = prepare_jobs(discover_captures())
    check_docker()

    for capture, output_directory in jobs:
        print(f"[zeek] {capture.name}")
        logs = process_capture(capture, output_directory)
        if logs:
            for log in logs:
                print(f"  {log}")
        else:
            print("  Zeek completed but did not create any log files.")

    print(f"Complete: {len(jobs)} capture(s) processed.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
