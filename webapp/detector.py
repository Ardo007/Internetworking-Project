"""The live detector behind the web app: runs dumpcap, feeds each sealed
chunk through Zeek and scores every calendar minute as soon as all of its
traffic has been processed.

One run = one Start..Stop. Each run gets its own folder,
datas/web/<run_id>/{captures,zeek}, outside datas/zeek, so web sessions
are never picked up as own_benign training data and chunks from an earlier
run can't be mixed into a new one.

Minute scheduling: dumpcap rotates on the clock (:00 and :30), so the chunk
being written marks how far processing has got -- every chunk before it is
sealed and has been through Zeek. Minute M is scored once that point
(the "horizon") reaches M+60. The chunk before the minute is loaded too,
so a query/response pair split across the minute boundary is re-joined.
"""

import math
import threading
import time
from collections import deque

import capture
import scoring
import zeek_runner as zr

RUNS_DIR = zr.PROJECT_ROOT / "datas" / "web"
POLL_SECONDS = 2
HISTORY_LENGTH = 30
TOP_DOMAINS = 5
#: Rough Zeek + scoring time after a minute ends, for the "next verdict" countdown.
PROCESSING_ALLOWANCE_SECONDS = 5
#: The first/last minute of a run is only scored if at least this much of
#: it was captured -- a few seconds of traffic is too little for a verdict.
MIN_PARTIAL_SECONDS = 30

ACTIVE_STATUSES = ("starting", "capturing", "stopping")


class AlreadyRunning(RuntimeError):
    pass


class Run:
    """Everything belonging to one Start..Stop."""

    def __init__(self, run_id, interface, process, started):
        self.run_id = run_id
        self.interface = interface
        self.process = process
        self.started = started
        self.directory = RUNS_DIR / run_id
        self.captures_dir = self.directory / "captures"
        self.zeek_dir = self.directory / "zeek"
        self.first_window = math.floor(started / 60) * 60
        if self.first_window + 60 - started < MIN_PARTIAL_SECONDS:
            self.first_window += 60
        self.next_window = self.first_window
        self.chunks = []  # [(start epoch, dns.log path or None)], in capture order
        self.processed = set()  # chunk file names
        self.stop_requested = threading.Event()
        self.thread = None

    def dumpcap_log_tail(self, chars=600):
        try:
            text = (self.directory / "dumpcap.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text.strip()[-chars:]

    def dns_logs_for(self, minute_start):
        """dns.logs of the chunks overlapping [minute_start - 30, minute_start + 60):
        the minute itself plus the chunk before it."""
        low, high = minute_start - capture.CHUNK_SECONDS, minute_start + 60
        logs = []
        for i, (start, dns_log) in enumerate(self.chunks):
            end = self.chunks[i + 1][0] if i + 1 < len(self.chunks) else math.inf
            if dns_log is not None and start < high and end > low:
                logs.append(dns_log)
        return logs


class Detector:
    def __init__(self, artifacts, threshold=0.5, log=None, model_label=""):
        self.artifacts = artifacts
        self.threshold = threshold
        self.log = log
        self.model_label = model_label
        self._lock = threading.Lock()
        self._run = None
        self._container_ready = False
        self._state = self._idle_state()

    # ------------------------------------------------------------ state --

    @staticmethod
    def _idle_state():
        return {
            "status": "idle",
            "error": None,
            "warning": None,
            "interface": None,
            "run_id": None,
            "started_at": None,
            "next_verdict_at": None,
            "latest": None,
            "history": deque(maxlen=HISTORY_LENGTH),
            "last_tunnel_at": None,
        }

    def _set(self, **changes):
        with self._lock:
            self._state.update(changes)

    def state(self):
        with self._lock:
            state = dict(self._state)
            state["history"] = list(state["history"])
        state["server_time"] = time.time()
        state["threshold"] = self.threshold
        state["model"] = self.model_label
        return state

    # ---------------------------------------------------------- control --

    def start(self, interface_id):
        with self._lock:
            if self._state["status"] in ACTIVE_STATUSES:
                raise AlreadyRunning("A capture is already running -- stop it first.")
            self._state["status"] = "starting"
            self._state["error"] = None
        try:
            names = {i["id"]: i["name"] for i in capture.list_interfaces()}
            if interface_id not in names:
                raise ValueError(f"Unknown interface: {interface_id}")
            zr.check_docker()
            if not (self._container_ready and zr.container_running()):
                zr.start_container()
                self._container_ready = True

            started = time.time()
            run_id = "web_" + time.strftime("%Y%m%d%H%M%S", time.localtime(started))
            directory = RUNS_DIR / run_id
            directory.mkdir(parents=True, exist_ok=True)
            process = capture.start_capture(interface_id, directory / "captures", directory / "dumpcap.log")
        except Exception as error:
            self._set(status="idle", error=str(error))
            raise

        run = Run(run_id, {"id": interface_id, "name": names[interface_id]}, process, started)
        with self._lock:
            self._state = self._idle_state()
            self._state.update(
                status="capturing",
                interface=run.interface,
                run_id=run_id,
                started_at=started,
                next_verdict_at=run.first_window + 60 + PROCESSING_ALLOWANCE_SECONDS,
            )
            self._run = run
        run.thread = threading.Thread(target=self._worker, args=(run,), name=f"detector-{run_id}", daemon=True)
        run.thread.start()

    def stop(self):
        """Ask the current run to stop. Returns at once; the worker stops
        dumpcap, scores the last (partial) minute and goes back to idle."""
        with self._lock:
            run = self._run
            if run is None or self._state["status"] not in ("starting", "capturing"):
                return
            self._state["status"] = "stopping"
        run.stop_requested.set()

    def shutdown(self, timeout=60):
        """Stop any run and remove the Zeek container (app exit)."""
        run = self._run
        if run is not None:
            run.stop_requested.set()
            if run.thread is not None:
                run.thread.join(timeout)
            if run.process.poll() is None:
                capture.stop_capture(run.process)
        if self._container_ready:
            zr.stop_container()
            self._container_ready = False
        if self.log is not None:
            self.log.close()

    # ----------------------------------------------------------- worker --

    def _worker(self, run):
        try:
            while not run.stop_requested.is_set():
                if run.process.poll() is not None:
                    raise RuntimeError("dumpcap stopped unexpectedly. " + run.dumpcap_log_tail())
                self._process_chunks(run, all_sealed=False)
                self._score_ready(run, horizon=self._horizon(run))
                run.stop_requested.wait(POLL_SECONDS)

            self._set(status="stopping")
            capture.stop_capture(run.process)
            stopped = time.time()
            self._process_chunks(run, all_sealed=True)
            self._score_ready(run, horizon=stopped, final=True)
            self._set(status="idle", next_verdict_at=None)
        except Exception as error:  # surface anything to the page instead of dying silently
            if run.process.poll() is None:
                capture.stop_capture(run.process)
            self._set(status="error", error=str(error), next_verdict_at=None)

    def _process_chunks(self, run, all_sealed):
        for chunk in zr.discover_sealed_captures(run.captures_dir, all_sealed=all_sealed):
            if chunk.name in run.processed:
                continue
            try:
                dns_log = zr.process_capture(chunk, run.zeek_dir / chunk.stem)
            except (OSError, RuntimeError) as error:
                # Skip the chunk rather than stall every later minute on it.
                dns_log = None
                self._set(warning=str(error))
            run.processed.add(chunk.name)
            run.chunks.append((zr.chunk_start(chunk), dns_log))

    @staticmethod
    def _horizon(run):
        """Start of the oldest chunk not yet through Zeek (normally the one
        dumpcap is writing): all traffic before it has been processed."""
        pending = [
            zr.chunk_start(chunk)
            for chunk in zr.discover_sealed_captures(run.captures_dir, all_sealed=True)
            if chunk.name not in run.processed
        ]
        return min(pending) if pending else run.started

    def _score_ready(self, run, horizon, final=False):
        """Score every minute that ends by `horizon`. With final=True
        (dumpcap has stopped), also score the minute `horizon` falls in."""
        while True:
            minute = run.next_window
            end = minute + 60
            if end > horizon and not (final and horizon - minute >= MIN_PARTIAL_SECONDS):
                break
            partial = minute < run.started or (final and end > horizon)
            summary = scoring.score_minute(
                run.dns_logs_for(minute), minute, run.run_id, self.artifacts, self.threshold,
            )
            self._publish(run, minute, summary, partial)
            run.next_window = end
        if not final:
            self._set(next_verdict_at=run.next_window + 60 + PROCESSING_ALLOWANCE_SECONDS)

    def _publish(self, run, minute, summary, partial):
        if summary is None:
            verdict, flagged, top, max_prob, n_queries = "no_dns", [], [], None, 0
        else:
            def rows(frame):
                return [
                    {
                        "domain": row.base_domain,
                        "prob": float(row.mean_tunnel_prob),
                        "n_queries": int(row.n_queries),
                        "n_flagged": int(row.n_flagged),
                    }
                    for row in frame.itertuples()
                ]
            flagged = rows(summary[summary["verdict"] == "TUNNEL"])
            top = rows(summary.head(TOP_DOMAINS))
            verdict = "TUNNEL" if flagged else "benign"
            max_prob = float(summary["mean_tunnel_prob"].max())
            n_queries = int(summary["n_queries"].sum())
            if self.log is not None:
                window_name = f"{run.run_id}@{time.strftime('%H:%M', time.localtime(minute))}"
                self.log.write(run.run_id, window_name, int((minute - run.first_window) // 60), summary)

        latest = {
            "window_start": minute,
            "window_end": minute + 60,
            "verdict": verdict,
            "partial": partial,
            "max_prob": max_prob,
            "n_queries": n_queries,
            "n_domains": 0 if summary is None else len(summary),
            "flagged": flagged,
            "top": top,
            "scored_at": time.time(),
        }
        with self._lock:
            self._state["latest"] = latest
            self._state["history"].append(
                {"window_start": minute, "verdict": verdict, "max_prob": max_prob, "partial": partial}
            )
            if verdict == "TUNNEL":
                self._state["last_tunnel_at"] = minute
