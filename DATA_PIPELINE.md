# Zeek processing

The processor runs Zeek against every completed packet capture in a
fixed input folder. It does not assign labels or depend on the GraphTunnel
dataset structure.

## Requirements

- Python 3
- Docker Desktop installed and running
- The `zeek/zeek:lts` Docker image (Docker downloads it automatically if needed)

## Input and output folders

Place `.pcap` and `.pcapng` files directly inside `datas/captures`:

```text
datas/
├── captures/
│   ├── capture_001.pcapng
│   └── capture_002.pcap
└── zeek/
```

The script scans files directly inside `datas/captures`; it does not scan nested
folders. Files with other extensions are ignored.

From the project root, run:

```text
python scripts/run_zeek.py
```

## How the script works

1. It finds all PCAP and PCAPNG files in `datas/captures` and sorts them by
   filename.
2. It creates a separate output folder for each capture using the filename
   without its extension. For example, `capture_001.pcapng` uses
   `datas/zeek/capture_001`.
3. It checks that Docker is installed and that the Docker engine is responding.
4. For each capture, it starts a temporary `zeek/zeek:lts` container. The input
   folder is mounted read-only, while the capture's output folder is mounted as
   writable.
5. Zeek reads the saved capture and writes JSON-formatted protocol logs to the
   output folder. The temporary container is removed after processing, but the
   generated logs remain on the computer.
6. The script prints each generated log path and reports how many captures were
   processed.

The resulting structure looks like:

```text
datas/
└── zeek/
    ├── capture_001/
    │   ├── conn.log
    │   └── dns.log
    └── capture_002/
        ├── conn.log
        └── dns.log
```

The exact logs depend on the traffic in each capture. For example, Zeek creates
`dns.log` only when it finds DNS activity. A successful capture with no
recognised traffic may produce no log files.

Only place completed captures in `datas/captures`; never process a file that a
capture tool is still writing. Existing non-empty output folders are not
overwritten, so they must be moved or removed before processing the same capture
again. Two input files with the same base name, such as `sample.pcap` and
`sample.pcapng`, are rejected because they would share one output folder.




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
