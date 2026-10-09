"""
Local web app for live DNS tunnel detection.

Pick a capture interface, press Start, and the page shows BENIGN or TUNNEL
DETECTED for every minute of traffic until you press Stop. Runs on this
machine only (127.0.0.1); Docker Desktop must be running for Zeek.

Usage:
  python webapp\\app.py [--model-dir DIR] [--threshold T] [--port N]
                        [--log-file CSV | --no-log] [--no-browser]
"""
import argparse
import os
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

import capture  # noqa: E402
import scoring  # noqa: E402
from detector import AlreadyRunning, Detector  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(detector):
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/interfaces")
    def interfaces():
        try:
            return jsonify(capture.list_interfaces())
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return jsonify(error=str(error)), 500

    @app.post("/api/start")
    def start():
        interface = (request.get_json(silent=True) or {}).get("interface")
        if not interface:
            return jsonify(error="Choose an interface first."), 400
        try:
            detector.start(interface)
        except AlreadyRunning as error:
            return jsonify(error=str(error)), 409
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return jsonify(error=str(error)), 503
        return jsonify(detector.state())

    @app.post("/api/stop")
    def stop():
        detector.stop()
        return jsonify(detector.state())

    @app.get("/api/state")
    def state():
        return jsonify(detector.state())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=None,
                        help="Model artifacts folder (default: newest models/zeek_bilstm/*/final)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-file", default=None,
                        help=f"CSV to append every scored minute/domain row to "
                             f"(default: {scoring.DEFAULT_LOG_PATH})")
    parser.add_argument("--no-log", action="store_true", help="Disable CSV logging")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the page automatically")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else scoring.default_model_dir()
    print(f"[webapp] loading model from {model_dir} ...", flush=True)
    artifacts = scoring.load_model(model_dir)
    model_label = f"{model_dir.parent.name}/{model_dir.name} ({artifacts['model_names'][0]})"

    log = None if args.no_log else scoring.ScoreLog(Path(args.log_file) if args.log_file else scoring.DEFAULT_LOG_PATH)
    if log is not None:
        print(f"[webapp] logging every scored minute to {log.path}", flush=True)

    detector = Detector(artifacts, args.threshold, log, model_label)
    app = create_app(detector)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[webapp] open {url} (Ctrl+C to quit)", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break shuts down cleanly too, like Ctrl+C
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
    finally:
        print("[webapp] shutting down (stopping capture and the Zeek container)...", flush=True)
        detector.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
