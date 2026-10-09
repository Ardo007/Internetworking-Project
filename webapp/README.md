# Web app: live DNS tunnel detection

A local web page for the live scoring pipeline. Pick a capture interface,
press **Start**, and the page shows **BENIGN** or **TUNNEL DETECTED** for
every minute of traffic until you press **Stop**. It runs on this machine
only (`127.0.0.1`).

It is a self-contained alternative to the three-terminal CLI pipeline in
[`LIVE_SCORING.md`](../LIVE_SCORING.md); nothing outside `webapp/` was changed,
so the CLI scripts still work exactly as before.

## Prerequisites

Same as the CLI pipeline:

- **Wireshark** (for `dumpcap` and the Npcap driver).
- **Docker Desktop running** -- Zeek runs in a `zeek/zeek:lts` container.
- The project's **venv** with `notebooks/requirements.txt` installed, plus Flask:

```
pip install -r webapp\requirements.txt
```

## Running it

```
python webapp\app.py
```

The model loads (a few seconds), then the page opens in your browser at
<http://127.0.0.1:8000/>. Choose an interface by name (e.g. "Wi-Fi"), press
Start. **Ctrl+C** in the terminal stops any running capture, removes the Zeek
container and exits.

Options: `--model-dir DIR` (default: newest `models/zeek_bilstm/*/final`),
`--threshold 0.5`, `--port 8000`, `--log-file CSV` / `--no-log`,
`--no-browser`.

## What the page shows

- **Banner** -- the verdict for the most recent scored minute: red
  **TUNNEL DETECTED** if any domain's mean tunnel probability is at or above
  the threshold, otherwise green **BENIGN** ("no DNS traffic" also counts as
  benign). The first verdict appears about a minute after Start, then one
  per minute.
- **Domains** -- the flagged domains plus the highest-scoring others in that
  minute, with query counts and probabilities.
- **Last 30 minutes** -- one dot per minute (hover for details), and when a
  tunnel was last seen.

## How it works

```text
capture.py      dumpcap -b interval:30   -> datas/web/<run>/captures/*.pcapng
zeek_runner.py  docker exec zeek -r       -> datas/web/<run>/zeek/<chunk>/dns.log
scoring.py      features + model_1.keras  -> verdict per minute (+ results/web_scoring_log.csv)
detector.py     ties them together in one background thread, polled by the page
app.py          Flask: /api/interfaces, /api/start, /api/stop, /api/state, and the page
```

- **One model.** Only `model_1.keras` of the final model's five is loaded.
- **Calendar minutes.** Training data was windowed into epoch-aligned
  minutes (`assign_windows`). dumpcap rotates on the clock (:00 and :30), and
  each minute is scored once both of its chunks are through Zeek, using
  only that minute's queries. (The CLI's `score_live.py` pairs chunks from
  whenever capture started, so its 60 s windows straddle two minutes.)
  The chunk before the minute is loaded too, so a query and response split
  across the boundary are re-joined. A first or last minute with under 30 s
  captured is skipped; one with more is scored and marked "partial".
- **Feature code is shared, not copied.** `scoring.py` imports
  `zeek_feature_extraction` and `model_artifacts` from `notebooks/`, so live
  features are computed exactly as in training. `capture.py` and
  `zeek_runner.py` are web versions of `capture_live.py` and `run_zeek.py`.
- **Interfaces by device id.** The page lists interfaces by name and passes
  dumpcap the `\Device\NPF_{...}` id, not the number, so the
  interface-renumbering gotcha in `LIVE_SCORING.md` doesn't apply.
- **Own folder, own container.** Each run writes to `datas/web/<run_id>/`,
  never `datas/zeek/`, so web sessions (including tunnel demos) are never
  picked up as `own_benign` training data. The Zeek container is
  `dns-tunnel-web-zeek`, so the web app and the CLI's `run_zeek.py` can run
  at the same time.

## Notes

- **Reverse DNS is flagged.** PTR lookups (`*.in-addr.arpa`) score near 1.0
  -- e.g. Windows `nslookup` first looks up the DNS server's own address,
  which is enough to turn the banner red. Four of the five final models do
  this, so it is a model blind spot, not a one-model artefact.
- **Disk use.** dumpcap keeps the last 120 chunks (one hour) of pcaps per
  run; Zeek output is kept for the whole run. Delete old `datas/web/<run>`
  folders when you no longer need them.
- **Log.** Every scored (minute, domain) row is appended to
  `results/web_scoring_log.csv`, with the same columns as the CLI's
  `live_scoring_log.csv`.
