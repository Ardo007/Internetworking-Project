# Live capture pipeline

Two scripts, meant to run side by side: `capture_live.py` writes rotating
capture chunks to disk, and `run_zeek.py` watches that folder and turns each
finished chunk into Zeek logs.

```text
datas/
├── captures/   <- capture_live.py writes rotated .pcapng chunks here
└── zeek/       <- run_zeek.py writes one log folder per processed chunk
```

## Requirements

- Python 3
- Wireshark installed (provides `dumpcap` and the Npcap driver), for
  `capture_live.py`
- Docker Desktop installed and running, for `run_zeek.py`
- The `zeek/zeek:lts` Docker image (Docker downloads it automatically if needed)

## 1. Capture: `capture_live.py`

Wraps `dumpcap`'s own ring buffer. No timing loop is needed — dumpcap handles
rotation, filenames, and disk-capping itself, and there's no packet loss at
rotation boundaries.

```text
python scripts/capture_live.py --list-interfaces
python scripts/capture_live.py --interface 6
```

- `--duration` (default 30s): how often a new chunk starts. This is a capture
  cadence, not the statistical analysis window — tunneling signal is
  detected over minutes of traffic spanning many chunks, not within one.
- `--files` (default 120): ring buffer cap. Once reached, dumpcap deletes the
  oldest chunk before starting the next, so this is also your disk cap. The
  downstream processor must keep up with capture speed, or it will lose
  chunks it hasn't read yet.
- `--output-dir` (default `datas/captures`), `--prefix` (default `capture`).

Chunks are named the way dumpcap always names ring-buffer files:
`<prefix>_<00001>_<YYYYmmddHHMMSS>.pcapng`. Stop with Ctrl+C; dumpcap closes
its current chunk cleanly before exiting.

## 2. Processing: `run_zeek.py`

```text
python scripts/run_zeek.py            # watch mode (default)
python scripts/run_zeek.py --once     # process what's sealed, then exit
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
   there with JSON logging (`-C -r <chunk> LogAscii::use_json=T`).
4. A chunk is skipped if its output folder already contains a non-empty
   `.log` file — reprocessing is safe to re-run and idempotent.
5. In watch mode it polls every `--interval` seconds (default 5) and runs
   until Ctrl+C, which tears down the long-lived container before exiting.
   `--once` processes whatever is currently sealed and exits without
   watching. `--capture-dir` / `--zeek-dir` override the default folders.

The exact logs depend on the traffic in each chunk — Zeek creates `dns.log`
only when it finds DNS activity, so a chunk with no recognised traffic may
produce no log files at all.




// Redundant for now
# GraphTunnel ingestion

Place GraphTunnel PCAP files beneath `Data/raw/GraphTunnel` while preserving their dataset category folders. Then run:

```text
python scripts/process_pcaps.py
```

For each PCAP, the program runs Zeek in the `zeek/zeek:lts` Docker image and creates a dedicated output directory beneath `Data/zeek/GraphTunnel`.

For example:

```text
Data/raw/GraphTunnel/normal/example.pcap
Data/zeek/GraphTunnel/normal/example/dns.log
Data/zeek/GraphTunnel/normal/example/conn.log
```

The program recursively discovers PCAPs, so nested split-capture folders are supported. A capture is skipped when its expected `dns.log` and `conn.log` already exist and are non-empty.

Every run recreates:

```text
Data/processed/GraphTunnel/capture_manifest.csv
```

The manifest maps each PCAP to its Zeek log and ground-truth label:

- `normal` and `wildcard`: label `0`
- `tunnel`, `unknownTunnel`/`unkownTunnel`, and `crossEndPoint`: label `1`

Unknown category names cause an error rather than receiving a guessed label. Original PCAPs and existing Zeek logs are not modified.

## Out-of-order PCAPs are time-sorted before Zeek

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
