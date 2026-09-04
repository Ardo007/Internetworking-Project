"""
DNS Tunneling Feature Extraction
=================================
Parses DNS traffic out of PCAP / PCAPNG capture files and builds a
per-query feature table suitable for rule-based, classical-ML and
deep-learning DNS tunnel detectors.

Handles the mixed capture formats (legacy pcap and pcapng) and mixed
link layers (Ethernet, Linux "cooked" SLL, raw IP) found in the
DNS-Tunnel-Datasets repository.

Feature groups produced per DNS query:
  * lexical      - qname length/entropy/character composition, label
                    (subdomain) structure -- the classic signals used
                    to spot encoded payloads riding in DNS labels.
  * response     - size, TTL, answer count, rcode and latency of the
                    matching DNS response (when one is observed).
  * domain/session - aggregated over all queries seen for the same
                    base domain within a capture: query rate, unique
                    subdomain count, query-type diversity, etc. This
                    is what actually distinguishes a tunnel (many
                    unique high-entropy subdomains, machine-gunned)
                    from ordinary lookups to the same domain.
"""
import math
import socket
import struct
from collections import Counter

import dpkt
import pandas as pd

DNS_QTYPES = {
    1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 10: "NULL", 12: "PTR",
    15: "MX", 16: "TXT", 28: "AAAA", 33: "SRV", 41: "OPT", 255: "ANY",
}
HEX_CHARS = set("0123456789abcdef")

# -------------------------------------------------------------- helpers --

def shannon_entropy(s):
    """Character-level Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def open_pcap_reader(fh):
    """Return (reader, datalink) for either a classic pcap or pcapng file."""
    magic = fh.read(4)
    fh.seek(0)
    if magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",
                 b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
        reader = dpkt.pcap.Reader(fh)
        return reader, reader.datalink()
    elif magic == b"\x0a\x0d\x0d\x0a":
        reader = dpkt.pcapng.Reader(fh)
        return reader, reader.datalink()
    raise ValueError(f"Unrecognized capture format (magic={magic!r})")


def _l2_to_ip(datalink, buf):
    try:
        if datalink == dpkt.pcap.DLT_EN10MB:
            data = dpkt.ethernet.Ethernet(buf).data
        elif datalink == dpkt.pcap.DLT_LINUX_SLL:
            data = dpkt.sll.SLL(buf).data
        elif datalink == dpkt.pcap.DLT_RAW:
            data = dpkt.ip.IP(buf)
        else:
            return None
        if isinstance(data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return data
    except Exception:
        return None
    return None


def _dns_payload(ip):
    l4 = ip.data
    if isinstance(l4, dpkt.udp.UDP) and (l4.sport == 53 or l4.dport == 53):
        return "udp", l4.sport, l4.dport, l4.data
    if isinstance(l4, dpkt.tcp.TCP) and (l4.sport == 53 or l4.dport == 53):
        payload = l4.data
        if len(payload) > 2:  # DNS-over-TCP has a 2-byte length prefix
            payload = payload[2:]
        return "tcp", l4.sport, l4.dport, payload
    return None, None, None, None


def _ip_to_str(raw):
    try:
        return socket.inet_ntoa(raw) if len(raw) == 4 else socket.inet_ntop(socket.AF_INET6, raw)
    except Exception:
        return None


# ---------------------------------------------------------- DNS parsing --
#
# NOTE: dpkt.dns.DNS() decodes every label as strict UTF-8 and raises
# UnicodeDecodeError on anything else. DNS tunneling tools routinely pack
# raw/near-random bytes into query labels and into NULL/TXT answer data
# (that's the whole point of a tunnel) -- on this dataset that made dpkt
# silently drop 46-97% of messages for the iodine-family captures, i.e.
# exactly the traffic this project cares most about. So DNS messages are
# parsed by hand below: header fields via struct, and names via raw
# length-prefixed labels decoded latin-1 (a lossless 1-byte<->1-char
# mapping) instead of UTF-8, so arbitrary bytes never abort parsing.

def _read_name(payload, offset, max_jumps=20):
    """Read a (possibly compressed) DNS name starting at `offset`.
    Returns (name_as_latin1_str, offset_immediately_after_the_name)."""
    labels = []
    return_offset = None
    jumps = 0
    pos = offset
    n = len(payload)
    while pos < n:
        length = payload[pos]
        if length == 0:
            pos += 1
            break
        if (length & 0xC0) == 0xC0:  # compression pointer
            if pos + 1 >= n:
                break
            ptr = ((length & 0x3F) << 8) | payload[pos + 1]
            if return_offset is None:
                return_offset = pos + 2
            if ptr >= n or jumps > max_jumps:
                break
            pos = ptr
            jumps += 1
            continue
        pos += 1
        labels.append(payload[pos:pos + length].decode("latin-1"))
        pos += length
    end_offset = return_offset if return_offset is not None else pos
    return ".".join(labels), end_offset


def parse_dns_message(payload):
    """
    Minimal, tolerant DNS message parser. Returns a dict of header fields
    plus the first question's (qname, qtype) and, for responses, the
    minimum TTL seen among answer records -- or None if even the 12-byte
    header doesn't fit.
    """
    if len(payload) < 12:
        return None
    txid, flags, qdcount, ancount, nscount, arcount = struct.unpack("!HHHHHH", payload[:12])
    qr = (flags >> 15) & 0x1
    opcode = (flags >> 11) & 0xF
    rcode = flags & 0xF

    qname, qtype = None, None
    min_ttl = None
    offset = 12
    try:
        if qdcount:
            qname, offset = _read_name(payload, offset)
            qname = qname.rstrip(".").lower()
            if offset + 4 <= len(payload):
                qtype = struct.unpack("!H", payload[offset:offset + 2])[0]
                offset += 4  # qtype + qclass

        if qr == 1 and ancount:
            ttls = []
            for _ in range(min(ancount, 64)):  # cap: tolerate truncated/garbled captures
                _, offset = _read_name(payload, offset)
                if offset + 10 > len(payload):
                    break
                rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", payload[offset:offset + 10])
                offset += 10
                ttls.append(ttl)
                offset += rdlen
                if offset > len(payload):
                    break
            if ttls:
                min_ttl = min(ttls)
    except Exception:
        pass  # keep whatever header/question fields we already extracted

    return {
        "txid": txid, "qr": qr, "opcode": opcode, "rcode": rcode,
        "qdcount": qdcount, "ancount": ancount, "nscount": nscount, "arcount": arcount,
        "qname": qname, "qtype": qtype, "min_ttl": min_ttl,
    }


# --------------------------------------------------------- pcap parsing --

def iter_dns_messages(pcap_path):
    """Yield one dict per successfully parsed DNS message (query or response)."""
    with open(pcap_path, "rb") as fh:
        reader, datalink = open_pcap_reader(fh)
        for ts, buf in reader:
            ip = _l2_to_ip(datalink, buf)
            if ip is None:
                continue
            proto, sport, dport, payload = _dns_payload(ip)
            if payload is None:
                continue
            dns = parse_dns_message(payload)
            if dns is None:
                continue

            yield {
                "ts": ts,
                "proto": proto,
                "src_ip": _ip_to_str(ip.src), "sport": sport,
                "dst_ip": _ip_to_str(ip.dst), "dport": dport,
                "txid": dns["txid"],
                "qr": dns["qr"],            # 0 = query, 1 = response
                "opcode": dns["opcode"],
                "rcode": dns["rcode"],
                "qdcount": dns["qdcount"], "ancount": dns["ancount"],
                "nscount": dns["nscount"], "arcount": dns["arcount"],
                "qname": dns["qname"], "qtype": dns["qtype"],
                "min_ttl": dns["min_ttl"],
                "pkt_len": len(buf),
            }


# --------------------------------------------------------- lexical feats --

def _lexical_features(qname):
    if not qname:
        return dict(qname_len=0, label_count=0, max_label_len=0, avg_label_len=0.0,
                    first_label_len=0, digit_ratio=0.0, hex_ratio=0.0,
                    unique_char_ratio=0.0, entropy=0.0, first_label_entropy=0.0)
    labels = qname.split(".")
    label_lens = [len(l) for l in labels]
    n = len(qname)
    digits = sum(c.isdigit() for c in qname)
    hexch = sum(c in HEX_CHARS for c in qname)
    return dict(
        qname_len=n,
        label_count=len(labels),
        max_label_len=max(label_lens),
        avg_label_len=sum(label_lens) / len(label_lens),
        first_label_len=label_lens[0],
        digit_ratio=digits / n,
        hex_ratio=hexch / n,
        unique_char_ratio=len(set(qname)) / n,
        entropy=shannon_entropy(qname),
        first_label_entropy=shannon_entropy(labels[0]),
    )


def _base_domain(qname, keep_labels=2):
    if not qname:
        return None
    labels = qname.split(".")
    return ".".join(labels[-keep_labels:]) if len(labels) >= keep_labels else qname


# ------------------------------------------------------ per-file dataset --

def extract_query_records(pcap_path, label=None):
    """
    Parse one capture file into a per-query DataFrame: one row per DNS
    question, enriched with lexical features and (when found) the
    matching response's size / TTL / rcode / latency.
    """
    messages = list(iter_dns_messages(pcap_path))
    if not messages:
        return pd.DataFrame()

    df = pd.DataFrame(messages)
    queries = df[df["qr"] == 0].copy()
    responses = df[df["qr"] == 1].copy()
    if queries.empty:
        return pd.DataFrame()

    queries["match_key"] = list(zip(queries["txid"], queries["src_ip"], queries["sport"],
                                     queries["dst_ip"], queries["dport"]))
    if not responses.empty:
        responses["match_key"] = list(zip(responses["txid"], responses["dst_ip"], responses["dport"],
                                           responses["src_ip"], responses["sport"]))
        responses = responses.sort_values("ts").drop_duplicates("match_key", keep="first")
        resp_lookup = responses.set_index("match_key")[["ts", "pkt_len", "ancount", "min_ttl", "rcode"]]
        resp_lookup = resp_lookup.rename(columns={
            "ts": "resp_ts", "pkt_len": "response_size", "ancount": "response_ancount",
            "min_ttl": "response_min_ttl", "rcode": "response_rcode"})
        queries = queries.join(resp_lookup, on="match_key")
        queries["response_latency"] = queries["resp_ts"] - queries["ts"]
    else:
        for c in ["response_size", "response_ancount", "response_min_ttl",
                  "response_rcode", "response_latency"]:
            queries[c] = pd.NA

    lex = queries["qname"].apply(_lexical_features).apply(pd.Series)
    queries = pd.concat([queries.reset_index(drop=True), lex.reset_index(drop=True)], axis=1)
    queries["qtype_name"] = queries["qtype"].map(DNS_QTYPES).fillna("OTHER")
    queries["base_domain"] = queries["qname"].apply(_base_domain)
    queries["pcap_file"] = str(pcap_path)
    if label is not None:
        queries["Label"] = label

    return queries.drop(columns=["match_key"])


# ---------------------------------------------------- domain aggregates --

def add_domain_aggregates(df, group_cols=("pcap_file", "base_domain")):
    """Add per-(file, base_domain) session/window features back onto every row."""
    group_cols = list(group_cols)
    agg = df.groupby(group_cols).agg(
        domain_query_count=("qname", "count"),
        domain_unique_qnames=("qname", "nunique"),
        domain_qtype_diversity=("qtype_name", "nunique"),
        domain_avg_qname_len=("qname_len", "mean"),
        domain_std_qname_len=("qname_len", "std"),
        domain_avg_entropy=("entropy", "mean"),
        domain_txt_ratio=("qtype_name", lambda s: (s == "TXT").mean()),
        domain_null_ratio=("qtype_name", lambda s: (s == "NULL").mean()),
        domain_first_ts=("ts", "min"),
        domain_last_ts=("ts", "max"),
    ).reset_index()

    # Floor the window at 1 second so a domain seen only once or twice in a
    # fraction of a second doesn't produce an absurd instantaneous "rate"
    # (a handful of samples divided by a near-zero duration would otherwise
    # dwarf every genuinely bursty tunnel domain).
    agg["domain_duration"] = (agg["domain_last_ts"] - agg["domain_first_ts"]).clip(lower=1.0)
    agg["domain_query_rate"] = agg["domain_query_count"] / agg["domain_duration"]
    agg["domain_unique_subdomain_ratio"] = agg["domain_unique_qnames"] / agg["domain_query_count"]
    agg["domain_std_qname_len"] = agg["domain_std_qname_len"].fillna(0)

    return df.merge(agg, on=group_cols, how="left")


# ------------------------------------------------------------ ML frame --

#: columns handed to the downstream classical-ML / deep-learning models.
#: (identifiers like ip/port/qname/txid are deliberately excluded so the
#: model can't just memorize a host or domain string.)
FEATURE_COLUMNS = [
    "proto", "qtype_name",
    "qname_len", "label_count", "max_label_len", "avg_label_len", "first_label_len",
    "digit_ratio", "hex_ratio", "unique_char_ratio", "entropy", "first_label_entropy",
    "response_size", "response_ancount", "response_min_ttl", "response_rcode", "response_latency",
    "domain_query_count", "domain_unique_qnames", "domain_qtype_diversity",
    "domain_avg_qname_len", "domain_std_qname_len", "domain_avg_entropy",
    "domain_txt_ratio", "domain_null_ratio", "domain_duration",
    "domain_query_rate", "domain_unique_subdomain_ratio",
]


def to_ml_frame(df, label_col="Label", extra_cols=()):
    """
    Select modeling columns and fill the response-side NaNs that occur
    when a query's matching response wasn't captured. `extra_cols` lets
    you keep bookkeeping columns (e.g. 'pcap_file', 'source_group') for
    analysis without feeding them to the model.
    """
    cols = FEATURE_COLUMNS + list(extra_cols) + [label_col]
    out = df[cols].copy()
    out["response_size"] = out["response_size"].fillna(0)
    out["response_ancount"] = out["response_ancount"].fillna(0)
    out["response_min_ttl"] = out["response_min_ttl"].fillna(-1)
    out["response_rcode"] = out["response_rcode"].fillna(-1)
    out["response_latency"] = out["response_latency"].fillna(-1)
    return out


def build_dataset(file_label_pairs, verbose=True):
    """
    file_label_pairs: iterable of (pcap_path, label) tuples.
    Returns one combined, feature-enriched DataFrame (one row per DNS query).
    """
    frames = []
    for path, label in file_label_pairs:
        if verbose:
            print(f"Parsing {path}  ->  label={label}")
        recs = extract_query_records(path, label=label)
        if verbose:
            print(f"   {len(recs)} DNS queries extracted")
        if not recs.empty:
            frames.append(recs)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = add_domain_aggregates(combined)
    return combined
