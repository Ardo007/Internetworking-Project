# Live scoring

Scores DNS traffic as it happens: capture packets, turn them into Zeek `dns.log`
records, extract the same features used for training, and score them with the
trained model -- all in near real time. This fills in the "scorer not built
yet" gap left in `DATA_PIPELINE.md`'s live-capture diagram; `score_live.py`
is that scorer.

```text
Ardashes_scripts/capture_live.py -> datas/captures/<chunk>.pcapng   30 s ring-buffer chunks
Ardashes_scripts/run_zeek.py     -> datas/zeek/<chunk>/dns.log      one Zeek pass per chunk
notebooks/score_live.py          -> console + results/live_scoring_log.csv
     joins consecutive chunk PAIRS into 60 s windows (matching WINDOW_SECONDS
     the model trained on), scores each with the model ensemble, prints any
     domain flagged TUNNEL, and appends every (window, domain) row scored to
     the CSV log.
```

All three run at once, each in its own terminal, each watching the folder the
previous one writes into. None of them talk to each other directly -- the
filesystem is the handoff.

## Prerequisites

- **Wireshark** installed (provides `dumpcap`/`mergecap` and the Npcap driver).
  `capture_live.py` looks for `dumpcap` on `PATH`, then falls back to
  `C:\Program Files\Wireshark\dumpcap.exe`.
- **Docker Desktop** running. `run_zeek.py` starts one long-lived
  `zeek/zeek:lts` container and reuses it for every chunk (`docker exec`)
  instead of paying container-startup cost per chunk -- check Docker Desktop
  is actually up before starting it, or you'll get a clear error instead of a
  hang.
- The project's **venv activated** (`.venv\Scripts\Activate.ps1` or similar),
  with `requirements.txt` installed.
- A trained model folder: `model_1.keras`...`model_N.keras`, `scaler.joblib`,
  `label_encoder.joblib`, `features.json`. The repo's convention is
  `models/zeek_bilstm/<run>/final/` (that's the one path under `models/` that
  isn't gitignored). **`score_live.py` and `score_capture.py` only
  auto-detect a model at that exact pattern** (`models/zeek_bilstm/*/final`,
  newest by modified time) -- if your model lives anywhere else (e.g.
  `models/zeek_bilstm/liam_run2/`), you must pass `--model-dir` explicitly
  every time, as shown below.
- The right capture interface. See the gotcha below -- don't assume a
  previously-used interface number is still correct.

## Running it

**1. List interfaces and pick the right one** (do this fresh every time you
set up on a machine that's had any networking change -- see gotcha below):

```
python Ardashes_scripts\capture_live.py --list-interfaces
```

**2. Terminal 1 -- capture.** Replace `12` with your interface number from
step 1:

```
python Ardashes_scripts\capture_live.py --interface 12
```

Writes 30-second rotating chunks to `datas\captures\`. Ctrl+C lets the
current chunk finish writing before stopping.

**3. Terminal 2 -- Zeek.**

```
python Ardashes_scripts\run_zeek.py
```

Watches `datas\captures\`, and once dumpcap has moved on to chunk *N+1*
(sealing chunk *N*), runs it through Zeek into `datas\zeek\<chunk>\dns.log`.
Add `--once` to process whatever's currently sealed and exit, instead of
watching continuously.

**4. Terminal 3 -- score.** Point `--model-dir` at your model folder:

```
python notebooks\score_live.py --model-dir "models\zeek_bilstm\liam_run2"
```

Watches `datas\zeek\`, pairs up consecutive 30s chunks into 60s windows, and
scores each window the moment both its chunks are ready.

**Stopping:** Ctrl+C any of the three independently, in any order.
`score_live.py` scores any leftover unpaired chunk before it exits, so you
won't lose the tail end of a session. `capture_live.py` and `run_zeek.py` can
be left running across multiple scoring sessions if you want.

## Reading the output

Console, per 60s window:

```
[capture_00011_...+capture_00012_...] 2/4 domain(s) flagged TUNNEL:
window_id base_domain  n_queries  mean_tunnel_prob  n_flagged verdict
        0    local.lan         74          0.996472         74  TUNNEL
        0 in-addr.arpa        148          0.626164        148  TUNNEL
```

or, when nothing's suspicious: `all benign (max prob 0.0421)`.

Every row scored (flagged or not) is also appended to
`results\live_scoring_log.csv`:

| column | meaning |
|---|---|
| `scored_at` | UTC timestamp the window was scored |
| `session_id` | the `capture_live.py` run this chunk came from |
| `window_name` | the two chunk filenames joined by `+` |
| `window_id`, `base_domain` | which 60s window / domain this row summarizes |
| `n_queries` | DNS queries to that domain in the window |
| `mean_tunnel_prob` | model's tunnel probability, averaged across the 5-fold ensemble and across queries |
| `n_flagged` | how many of those queries individually scored >= threshold |
| `verdict` | `TUNNEL` if `mean_tunnel_prob >= threshold` (default 0.5), else `benign` |

This log accumulates across every run of `score_live.py` -- it's one
continuous history, not one file per session.

## Known gotchas (hit these during setup)

- **Wrong interface number.** Installing VMware Workstation (or any new
  virtual adapter) shifts Npcap's interface numbering. A number that was
  correct last week can silently point at the wrong adapter -- you'll see
  `run_zeek.py` chugging along but `score_live.py` reporting "no DNS traffic
  in this window" for everything. Always re-run `--list-interfaces` after
  any networking/VM software change, and match by adapter *name*
  (e.g. "Ethernet 4"), not by remembering last time's number.
- **Windows Defender can quarantine capture files.** If a file in the
  capture/Zeek output folders is named after an attack tool (testing with
  real tunnelling tools can produce this), Defender's real-time protection
  may silently delete it moments after creation. If files vanish right
  after being written, add a folder exclusion:
  `Add-MpPreference -ExclusionPath "<project folder>"` (elevated PowerShell).
- **Docker/WSL2 filesystem lag.** If you're also running batch processing
  (e.g. `process_external_pcaps.py`) that writes files Docker then reads,
  a just-written file can briefly be invisible inside the container. Not
  usually an issue for the live pipeline itself since `run_zeek.py` already
  waits for a chunk to be sealed before reading it.
- **Model not found / wrong model loaded silently.** `score_live.py` and
  `score_capture.py` default to the newest `models/zeek_bilstm/*/final`
  folder if you don't pass `--model-dir`. If your model lives elsewhere,
  always pass `--model-dir` explicitly -- otherwise you may silently score
  against an old or missing model.

## Performance

Measured from real `live_scoring_log.csv` runs (excluding early sessions run
before the interface bug above was fixed): mean per-window latency 7.67s
against a 60s window, i.e. **RTF (real-time factor) mean = 0.128** -- the
pipeline processes each window in about 1/8th of its real-world duration,
roughly **7.8x faster than real-time**. It comfortably keeps up with live
traffic; chunk/window backlog shouldn't build up under normal load.

## After a live session

- **Score a specific capture after the fact** (not live) with
  `notebooks\score_capture.py <path-to-dns.log-or-folder> --model-dir <dir>`.
- **Formal EER / F1 metrics**, pooling live/external results with
  GraphTunnel's own held-out data: `notebooks\compute_formal_metrics.py`.
  Note it reports two separate numbers -- an EXTERNAL one (valid, model
  never trained on that data) and a GRAPHTUNNEL SELF-CHECK one (not valid as
  a generalisation measure if your model folder is a "final" model trained
  on every GraphTunnel capture -- see that script's docstring).
- **Feed today's session back into training data.** Live sessions land under
  category `own_benign` automatically the next time `notebooks\dataset_splits.py`
  is run (it scans the live Zeek output folder for new sessions). This
  reshuffles which `own_benign` session counts as held-out for the benign
  false-positive check -- harmless, but worth knowing if you're comparing
  before/after results.
