# Data pipeline

Two flows share one feature extractor: the GraphTunnel corpus is used to train
the model, and the live capture pipeline produces the same kind of Zeek logs
for scoring later.

**Training**

```text
download GraphTunnel
 └─ Data/raw/GraphTunnel/<category>/*.pcap          105 captures, deduplicated layout
     └─ Ardashes_scripts/process_pcaps.py           Zeek per capture -> Data/zeek/GraphTunnel/...,
        │                                           Data/processed/GraphTunnel/capture_manifest.csv
        └─ Ardashes_scripts/reorder_and_rezeek.py   time-sort out-of-order captures, re-run Zeek
            └─ notebooks/zeek_feature_extraction.py dns.log -> one row per DNS record,
                │                                   with per-domain 60 s window aggregates
                └─ notebooks/dataset_splits.py      Data/processed/GraphTunnel/splits.csv (committed)
                    └─ notebooks/dns_tunneling_bilstm_model.ipynb
                        ├─ notebooks/dataset_zeek/     cached feature table (gitignored)
                        ├─ models/zeek_bilstm/<name>/  models, scaler, features.json (gitignored)
                        └─ results/zeek_run.md         metrics (committed)
```

**Live capture**

```text
Ardashes_scripts/capture_live.py -> datas/captures/<chunk>.pcapng   30 s ring-buffer chunks
Ardashes_scripts/run_zeek.py     -> datas/zeek/<chunk>/dns.log
 └─ notebooks/zeek_feature_extraction.py    the same extractor; consecutive chunks are
     │                                      joined into 60 s windows before aggregating
     └─ scoring against models/zeek_bilstm/<config>/   (scorer not built yet)
```

Benign live sessions can also become training data, as category `own_benign`
(see [Splits](#splits)).

## 1. The GraphTunnel corpus

Source: [DNS-Tunnel-Datasets](https://github.com/ggyggy666/DNS-Tunnel-Datasets)
([paper](https://ieeexplore.ieee.org/document/10636232)), about 658 MB here.

Place the captures under `Data/raw/GraphTunnel/<category>/`, keeping only each
category's top-level `.pcap` files: the per-tool subfolders upstream hold the
same traffic split into chunks, so processing both would count it twice.
`normal` is the exception, as its captures live in the upstream `normal/normal/`
folder and are consecutive chunks of one long session.

| category | files | label | what it is |
|---|---|---|---|
| `normal` | 68 | 0 | benign lookups of the Cloudflare top-1M list, consecutive chunks named `normal_<index>_<timestamp>` |
| `tunnel` | 13 | 1 | DNS-shell, dnscat2 (3), dnspot, iodine (7), tuns |
| `unknownTunnel` | 6 | 1 | tools kept out of training: cobalstrike, dns2tcp-key, dns2tcp-txt, ozymandns, tcp-over-dns (2) |
| `crossEndPoint` | 5 | 1 | iodine on Android (`AndIodine-*`): an unseen platform, not an unseen tool |
| `wildcard` | 13 | 0 | benign wildcard DNS that structurally resembles tunnelling |

The upstream repository spells one folder `unkownTunnel`; both spellings are
accepted. `Data/` is gitignored apart from `splits.csv`, so every collaborator
regenerates the logs and features locally.

## 2. Zeek logs and the manifest: `process_pcaps.py`

```text
python Ardashes_scripts/process_pcaps.py
```

For each PCAP the program runs Zeek in the `zeek/zeek:lts` Docker image and
creates a dedicated output directory beneath `Data/zeek/GraphTunnel`:

```text
Data/raw/GraphTunnel/normal/example.pcap
Data/zeek/GraphTunnel/normal/example/dns.log
Data/zeek/GraphTunnel/normal/example/conn.log
```

PCAPs are discovered recursively, so nested split-capture folders work too. A
capture is skipped when its `dns.log` and `conn.log` already exist and are
non-empty. Every run recreates the manifest
`Data/processed/GraphTunnel/capture_manifest.csv`, which maps each PCAP to its
Zeek log and ground-truth label (`normal` and `wildcard` are 0; `tunnel`,
`unknownTunnel`/`unkownTunnel` and `crossEndPoint` are 1). Unknown category
names cause an error rather than a guessed label. Original PCAPs and existing
Zeek logs are never modified.

The manifest is the single source of truth for which captures exist and what
they are labelled. (The placeholder
`testAndEval/TestingAndEval/capture_manifest.csv` is unrelated and unused.)

## 3. Out-of-order PCAPs are time-sorted before Zeek

Run once, after `process_pcaps.py`:

```text
python Ardashes_scripts/reorder_and_rezeek.py --check-only   # report only
python Ardashes_scripts/reorder_and_rezeek.py
```

**Why.** Some GraphTunnel captures are not stored in timestamp order. All 68
`normal` chunks contain packets timestamped up to 2.6 hours later than the
packets around them. Zeek keeps its clock at the newest timestamp it has seen,
so on those files:

- `dns.log` `ts` is Zeek's clock, not the packet time. In `normal_00000`,
  29,383 records had only 136 distinct timestamps, so time windows were
  meaningless.
- Connections look idle and are expired straight away, so a query and its
  response end up in different connections and are logged as two records,
  neither with `rtt`. Only 115 of those 29,383 records had `rtt`, against most
  records in the (time-ordered) tunnel captures.

Live captures are written in time order, so a model trained on the unsorted
output would learn an artefact ("no rcode / no rtt means benign") that never
occurs live.

**What the script does.** It scans every packet of every capture in the
manifest. A capture is out of order if any timestamp is earlier than the one
before it. Only those captures are processed:

1. `reordercap` (bundled with Wireshark) writes a time-sorted copy to
   `Data/interim/GraphTunnel/<same relative path>`. `Data/raw` is never
   modified.
2. The existing Zeek folder is renamed to `<stem>_Backup`. Anything that reads
   the Zeek logs ignores folders ending in `_Backup`.
3. Zeek runs on the sorted copy into the original folder, using
   `process_pcaps.run_zeek` (same image and flags), so the manifest's
   `zeek_dns_log` paths stay valid.

It writes `Data/interim/GraphTunnel/reorder_report.csv` (packets, backwards
steps and maximum lag per capture, plus the action taken). Re-running is safe:
a capture that already has a `_Backup` folder and usable logs is skipped.

On the current dataset, 81 of 105 captures are out of order: the 68 `normal`
chunks (maximum lag 8 s to 2.6 h) and the 13 `wildcard` captures (maximum lag
4-40 microseconds, harmless but sorted anyway because the rule is strict). The
24 tunnel-family captures are already in order.

Effect on the 68 `normal` captures: `dns.log` shrank from 1,973,175 to
1,013,648 records, within 0.2-2.5% of the queries in the PCAPs (the remainder
is responses whose query is not in the capture). Distinct timestamps went from
0.2% to 98% of records, every `ts` is now a real packet time, and `rtt` is
present on 89% of records (99% of answered ones), against 0.1% before. The
wildcard logs came out identical.

## 4. Features: `zeek_feature_extraction`

The extractor reads the JSON `dns.log` files listed in the manifest and
produces one row per DNS record:

- **lexical** features of the query name (length, label structure, character
  mix, entropy) and the query type. Zeek writes non-printable bytes as `\xNN`;
  they are decoded back to bytes, which reproduces the older PCAP parser's
  names on all 105 captures.
- **response** features: answer count, minimum TTL, rcode.
- **per-domain, per-window aggregates** over records sharing a base domain
  inside the same 60-second window (`WINDOW_SECONDS`): volume, name shape and
  the NXDOMAIN / no-response / rejected / rcode-entropy ratios. Aggregating per
  window instead of per capture keeps these independent of how long a capture
  ran, and is computable live from consecutive chunks.

Two clean-ups happen first. Zeek logs a query and its response separately when
it cannot pair them (long-lived tunnel connections that exceed its 50 pending
queries, responses slower than its 10 s DNS session timeout, or a capture
boundary); such pairs are re-joined when protocol, addresses, ports,
transaction ID and query name match within 30 seconds. A response still left
without its query is dropped, because Zeek does not record its query type.

`conn.log` is not used: its `resp_bytes` is per connection, and one UDP
connection can carry a whole tunnel session (up to 135,677 queries here), so
there is no reliable per-query response size.

For live scoring, load the `dns.log` files of consecutive 30-second chunks
together (`load_dns_logs` accepts a list), re-join split transactions over the
whole stream, then assign windows and aggregate. `find_live_sessions` groups
chunk folders into capture sessions.

## 5. Splits

`notebooks/dataset_splits.py` is the project's only split rule (the old
row-level `testAndEval/TestingAndEval/split_dataset.py` was removed). Run it
after the Zeek logs are in place:

```text
python notebooks/dataset_splits.py
```

It builds the feature table, writes `Data/processed/GraphTunnel/splits.csv`
(committed) and prints row and window counts per class, split and
configuration. Each row of `splits.csv` assigns a whole capture, or one
contiguous range of 60-second windows of a capture, to `train`, `val`, `test`
or `heldout` in configuration `A` or `B`:

| category | rule |
|---|---|
| normal | file index 00000-00047 train, 00048-00054 val, 00055-00061 test, 00062-00067 held out (false positive rate) |
| tunnel | per capture, first ~70% of its windows train, next ~15% val, last ~15% test, one unused window between segments |
| wildcard | config B (primary): of 00000-00006 the capture with the most windows is val and the other six train, 00007-00012 held out. Config A (stress test): all 13 held out |
| unknownTunnel | held out (unseen tools) |
| crossEndPoint | held out (unseen platform: iodine on Android) |
| own_benign | live-pipeline sessions from `datas/zeek/<chunk>/`: most recent complete session held out, earlier sessions split 70/15/15 by contiguous time blocks |

Splits never cut through a window, and `validate_splits` rejects any window
assigned to two splits. Train and val rows are then sampled per (capture,
window) — at most 200 rows, or 1,000 for wildcard — with a fixed seed, so busy
windows can't dominate fitting. The aggregates are computed on all rows before
sampling, and test and held-out rows are always scored in full.

`own_benign` captures are found by
`zeek_feature_extraction.find_live_sessions`: consecutive 30-second chunks of
one `capture_live.py` run form one capture, and a session only counts once it
has finished. None exist yet. Note that `capture_live.py` keeps only the last
120 chunks, so copy a session out of `datas/captures` before relying on it.

## 6. Training and results

Open `notebooks/dns_tunneling_bilstm_model.ipynb` and run it top to bottom
(see `notebooks/README.md` for the environment). It caches the feature table in
`notebooks/dataset_zeek/`, trains the configurations and writes:

- `models/zeek_bilstm/<name>/` — one folder per configuration with
  `model_*.keras`, `scaler.joblib`, `label_encoder.joblib` and `features.json`
  (feature order, one-hot input columns, window length, row caps, sampling
  seed, training date, git commit and library versions). Gitignored:
  regenerate by running the notebook.
- `results/zeek_run.md` — the metrics of the run, committed.

## 7. Live capture pipeline

Two scripts, meant to run side by side: `capture_live.py` writes rotating
capture chunks to disk, and `run_zeek.py` watches that folder and turns each
finished chunk into Zeek logs.

```text
datas/
├── captures/   <- capture_live.py writes rotated .pcapng chunks here
└── zeek/       <- run_zeek.py writes one log folder per processed chunk
```

Requirements: Wireshark installed (it provides `dumpcap` and the Npcap driver)
for `capture_live.py`, and Docker Desktop plus the `zeek/zeek:lts` image for
`run_zeek.py`.

### Capture: `capture_live.py`

Wraps `dumpcap`'s own ring buffer. No timing loop is needed — dumpcap handles
rotation, filenames, and disk-capping itself, and there's no packet loss at
rotation boundaries.

```text
python Ardashes_scripts/capture_live.py --list-interfaces
python Ardashes_scripts/capture_live.py --interface 6
```

- `--duration` (default 30s): how often a new chunk starts. This is a capture
  cadence, not the statistical analysis window — tunneling signal is
  detected over minutes of traffic spanning many chunks, not within one. The
  model's window is `WINDOW_SECONDS` (60 s), so scoring joins consecutive
  chunks.
- `--files` (default 120): ring buffer cap. Once reached, dumpcap deletes the
  oldest chunk before starting the next, so this is also your disk cap. The
  downstream processor must keep up with capture speed, or it will lose
  chunks it hasn't read yet.
- `--output-dir` (default `datas/captures`), `--prefix` (default `capture`).

Chunks are named the way dumpcap always names ring-buffer files:
`<prefix>_<00001>_<YYYYmmddHHMMSS>.pcapng`. Stop with Ctrl+C; dumpcap closes
its current chunk cleanly before exiting.

### Processing: `run_zeek.py`

```text
python Ardashes_scripts/run_zeek.py            # watch mode (default)
python Ardashes_scripts/run_zeek.py --once     # process what's sealed, then exit
```

1. Checks Docker is installed and responding, then starts one long-lived
   `zeek/zeek:lts` container (`sleep infinity`) with the whole project mounted
   at `/work`. Every chunk is processed via `docker exec` against that same
   container instead of paying a `docker run` startup cost (1-3s) on every
   rotation.
2. **Sealing rule:** a chunk matching dumpcap's ring-buffer naming is only
   read once the next index in its sequence exists — reading a chunk dumpcap
   still has open would race its writes. A file that doesn't match that
   naming (e.g. dropped in by hand) is treated as already complete.
   - Corollary: the *last* chunk of a capture that has stopped for good never
     gets a "next index" and is never processed. This is fine while
     `capture_live.py` keeps running, but a one-off/finished capture set
     using the same naming convention (GraphTunnel's own captures included)
     will permanently strand its highest-indexed file unless it was already
     processed before capture stopped.
3. For each sealed chunk, it creates `datas/zeek/<chunk stem>/` and runs Zeek
   there with JSON logging (`-C -r <chunk> LogAscii::use_json=T`) — the same
   flags `process_pcaps.py` uses, so the logs are interchangeable.
4. A chunk is skipped if its output folder already contains a non-empty
   `.log` file — reprocessing is safe to re-run and idempotent.
5. In watch mode it polls every `--interval` seconds (default 5) and runs
   until Ctrl+C, which tears down the long-lived container before exiting.
   `--once` processes whatever is currently sealed and exits without
   watching. `--capture-dir` / `--zeek-dir` override the default folders.

The exact logs depend on the traffic in each chunk — Zeek creates `dns.log`
only when it finds DNS activity, so a chunk with no recognised traffic may
produce no log files at all.
