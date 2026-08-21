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
