import json
import math
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import zeek_feature_extraction as zfe
from dns_feature_extraction import _lexical_features, parse_dns_message

FIXTURE = Path(__file__).parent / "fixtures" / "dns_sample.log"
WINDOW_START = 1693717020.0  # a multiple of 60, so the fixture starts a window
TUNNEL_QNAME = "tqä\x01z.t.example.net"


@pytest.fixture(scope="module")
def capture():
    return zfe.extract_query_records_from_zeek(FIXTURE, "tunnel/sample", "tunnel", "sample-tool", 1)


@pytest.fixture(scope="module")
def aggregated(capture):
    return zfe.add_domain_aggregates(capture)


def row(frame, uid):
    rows = frame[frame["uid"] == uid]
    assert len(rows) == 1, f"expected one row for {uid}, found {len(rows)}"
    return rows.iloc[0]


# ------------------------------------------------------------- loading --

def test_load_dns_log_skips_and_counts_malformed_lines():
    records, malformed = zfe.load_dns_log(FIXTURE)
    assert malformed == 1
    assert len(records) == 11  # the blank line is ignored, not counted
    assert "CBroken" not in {r["uid"] for r in records}


def test_load_dns_log_rejects_json_that_is_not_a_record(tmp_path):
    log = tmp_path / "dns.log"
    log.write_text('[1, 2]\n{"ts": 1.0}\n"text"\n', encoding="utf-8")
    records, malformed = zfe.load_dns_log(log)
    assert records == []
    assert malformed == 3


# ------------------------------------------------------ field mapping --

def test_bookkeeping_columns(capture):
    # 11 records: one split pair merged, two unmatched responses dropped
    assert len(capture) == 8
    assert capture.attrs == {"malformed_lines": 1, "rejoined": 1, "unmatched_responses_dropped": 2}
    assert set(capture["capture_id"]) == {"tunnel/sample"}
    assert set(capture["category"]) == {"tunnel"}
    assert set(capture["tool"]) == {"sample-tool"}
    assert set(capture["Label"]) == {"tunnel"}
    assert capture["ts"].is_monotonic_increasing
    assert set(zfe.BOOKKEEPING_COLUMNS) <= set(capture.columns)


def test_answered_a_query(capture):
    r = row(capture, "CNormalA")
    assert r["proto"] == "udp"
    assert r["qtype_name"] == "A"
    assert r["qname"] == "www.example.com"
    assert r["base_domain"] == "example.com"
    assert r["response_ancount"] == 2
    assert r["response_min_ttl"] == 120.0
    assert r["response_rcode"] == 0
    assert r["response_latency"] == pytest.approx(0.02)
    assert not r["rejected"]
    for name, value in _lexical_features("www.example.com").items():
        assert r[name] == pytest.approx(value)


def test_nxdomain(capture):
    r = row(capture, "CNxdomain")
    assert r["response_rcode"] == 3
    assert r["response_ancount"] == 0
    assert math.isnan(r["response_min_ttl"])
    assert math.isnan(r["response_latency"])  # Zeek only sets rtt when there are answers
    assert r["rejected"]


def test_query_without_response(capture):
    r = row(capture, "CNoResponse")
    assert r["qtype_name"] == "AAAA"
    assert r["response_ancount"] == 0
    assert math.isnan(r["response_rcode"])
    assert math.isnan(r["response_min_ttl"])
    assert math.isnan(r["response_latency"])


def test_to_ml_frame_fills_missing_response_fields(capture):
    ml = zfe.to_ml_frame(zfe.add_domain_aggregates(capture), extra_cols=["uid"])
    r = ml[ml["uid"] == "CNoResponse"].iloc[0]
    assert (r["response_ancount"], r["response_min_ttl"], r["response_rcode"]) == (0, -1, -1)
    assert list(ml.columns) == zfe.FEATURE_COLUMNS + ["uid", "Label"]
    assert not ml[zfe.FEATURE_COLUMNS].isna().any().any()


def test_to_ml_frame_feature_subset(aggregated):
    ml = zfe.to_ml_frame(aggregated, features=zfe.FEATURE_SETS["lexical_only"],
                         extra_cols=["capture_id", "response_latency"])
    assert list(ml.columns) == zfe.FEATURE_GROUPS["lexical"] + ["capture_id", "response_latency", "Label"]
    assert ml.loc[row(aggregated, "CNoResponse").name, "response_latency"] == -1


@pytest.mark.parametrize("record, expected", [
    ({"qtype": 16, "qtype_name": "TXT"}, "TXT"),
    ({"qtype": 65, "qtype_name": "HTTPS"}, "HTTPS"),
    ({"qtype": 64, "qtype_name": "SVCB"}, "SVCB"),
    ({"qtype": 65}, "HTTPS"),
    ({"qtype": 25, "qtype_name": "KEY"}, "OTHER"),
    ({"qtype": 65399, "qtype_name": "query-65399"}, "OTHER"),
    ({"qtype": 255, "qtype_name": "*"}, "ANY"),
    ({}, "OTHER"),
])
def test_qtype_name_vocabulary(record, expected):
    assert zfe.qtype_name_of(record) == expected


# ------------------------------------------------------ feature columns --

def test_identifiers_are_never_features():
    identifiers = {"uid", "qname", "query", "trans_id", "capture_id", "tool", "category",
                   "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "ts", "base_domain",
                   "window_id", "window_start", "window_query_count", "Label"}
    assert not identifiers & set(zfe.FEATURE_COLUMNS)
    assert not set(zfe.BOOKKEEPING_COLUMNS) & set(zfe.FEATURE_COLUMNS)
    assert "response_size" not in zfe.FEATURE_COLUMNS
    assert "response_latency" not in zfe.FEATURE_COLUMNS


def test_feature_groups_partition_the_features():
    partition = [c for name, group in zfe.FEATURE_GROUPS.items() if name != "artefact_suspect" for c in group]
    assert sorted(partition) == sorted(zfe.FEATURE_COLUMNS)
    assert len(partition) == len(set(partition))
    assert set(zfe.FEATURE_GROUPS["artefact_suspect"]) == {"domain_qtype_diversity", "no_response_ratio"}
    assert "domain_qtype_diversity" in zfe.FEATURE_GROUPS["domain_shape"]
    assert "qtype_name" in zfe.FEATURE_GROUPS["lexical"]


def test_feature_sets():
    sets = zfe.FEATURE_SETS
    assert sets["all"] == zfe.FEATURE_COLUMNS
    assert sets["lexical_only"] == zfe.FEATURE_GROUPS["lexical"]
    assert set(sets["domain_volume_shape"]) == set(zfe.FEATURE_GROUPS["domain_volume"] + zfe.FEATURE_GROUPS["domain_shape"])
    assert set(sets["all_minus_artefact_suspect"]) == set(zfe.FEATURE_COLUMNS) - {"domain_qtype_diversity", "no_response_ratio"}
    for features in sets.values():
        assert set(features) <= set(zfe.FEATURE_COLUMNS)


# ------------------------------------------------------ escape decoding --

def test_escaped_tunnel_query_is_decoded(capture):
    r = row(capture, "CTunnelTxt")
    assert r["qname"] == TUNNEL_QNAME
    assert r["qtype_name"] == "TXT"
    assert r["base_domain"] == "example.net"
    assert r["response_min_ttl"] == 0.0
    for name, value in _lexical_features(TUNNEL_QNAME).items():
        assert r[name] == pytest.approx(value)


@pytest.mark.parametrize("zeek, expected", [
    ("plain.example.com", "plain.example.com"),
    ("Mixed.Example.COM.", "mixed.example.com"),
    ("a\\x00b\\xffc", "a\x00b\xffc"),
    ("\\xC4\\xe4", "ää"),
    ("café", "cafã©"),  # UTF-8 passed through -> its two bytes, lowercased
    ("", ""),
    (None, ""),
])
def test_decode_query(zeek, expected):
    assert zfe.decode_query(zeek) == expected


def test_decoding_matches_the_pcap_parser():
    labels = [b"Tq\xc4\x01z", b"t", b"example", b"net"]
    qname = b"".join(bytes([len(label)]) + label for label in labels) + b"\x00"
    message = struct.pack("!HHHHHH", 7, 0x0100, 1, 0, 0, 0) + qname + struct.pack("!HH", 16, 1)
    parsed = parse_dns_message(message)["qname"]
    # Zeek lowercases ASCII and escapes the other bytes.
    assert zfe.decode_query("tq\\xc4\\x01z.t.example.net") == parsed == TUNNEL_QNAME


# -------------------------------------------------- split transactions --

def record(ts, *, trans_id=1, port=5000, query="q.example.com", kind="query", uid=None, **extra):
    base = {"ts": ts, "uid": uid or f"C{kind}{ts}", "id.orig_h": "10.0.0.2", "id.orig_p": port,
            "id.resp_h": "10.0.0.1", "id.resp_p": 53, "proto": "udp", "trans_id": trans_id}
    if query is not None:
        base["query"] = query
    if kind == "query":
        base.update(qtype=1, qtype_name="A", RD=True, rejected=False)
    elif kind == "response":
        base.update(rcode=0, rcode_name="NOERROR", RD=False, RA=True, rejected=False)
    elif kind == "paired":
        base.update(qtype=1, qtype_name="A", rcode=3, rcode_name="NXDOMAIN", RD=True, rejected=True)
    base.update(extra)
    return base


def test_split_pair_in_fixture_is_merged(capture):
    r = row(capture, "CSplitQuery")
    assert r["rejoined"]
    assert r["ts"] == WINDOW_START + 3.0
    assert r["qtype_name"] == "TXT"
    assert r["response_rcode"] == 0
    assert r["response_ancount"] == 1
    assert r["response_latency"] == pytest.approx(2.5)
    assert "CSplitResponse" not in set(capture["uid"])
    assert capture["rejoined"].sum() == 1


def test_fixture_pairs_that_must_not_merge(capture):
    # 31 s apart / different trans_id: the queries stay unanswered and the
    # responses, having no query in the data, are dropped.
    assert math.isnan(row(capture, "CLateQuery")["response_rcode"])
    assert math.isnan(row(capture, "CMismatchQuery")["response_rcode"])
    assert not {"CLateResponse", "CMismatchResponse"} & set(capture["uid"])


def test_rejoin_merges_response_fields():
    query = record(100.0)
    response = record(101.5, kind="response", answers=["192.0.2.1"], TTLs=[30.0], AA=True)
    out, merged = zfe.join_split_transactions([query, response])
    assert merged == 1
    assert len(out) == 1
    combined = out[0]
    assert combined["ts"] == 100.0
    assert combined["uid"] == query["uid"]
    assert combined["qtype"] == 1 and combined["RD"] is True
    assert combined["rcode"] == 0 and combined["RA"] is True and combined["AA"] is True
    assert combined["answers"] == ["192.0.2.1"] and combined["TTLs"] == [30.0]
    assert combined["rtt"] == pytest.approx(1.5)
    assert combined["rejoined"] is True


@pytest.mark.parametrize("gap, merges", [(0.0, True), (30.0, True), (30.001, False), (31.0, False), (-0.5, False)])
def test_rejoin_time_limit(gap, merges):
    out, merged = zfe.join_split_transactions([record(100.0), record(100.0 + gap, kind="response")])
    assert merged == int(merges)
    assert len(out) == (1 if merges else 2)


def test_rejoin_requires_same_trans_id():
    out, merged = zfe.join_split_transactions([record(100.0, trans_id=1), record(101.0, trans_id=2, kind="response")])
    assert merged == 0 and len(out) == 2


def test_rejoin_requires_same_addresses_and_ports():
    other_port = record(101.0, port=5001, kind="response")
    other_server = record(101.0, kind="response")
    other_server["id.resp_h"] = "10.0.0.9"
    other_proto = record(101.0, kind="response", proto="tcp")
    for response in (other_port, other_server, other_proto):
        out, merged = zfe.join_split_transactions([record(100.0), response])
        assert merged == 0 and len(out) == 2


def test_rejoin_requires_same_query_when_response_has_one():
    out, merged = zfe.join_split_transactions([record(100.0), record(101.0, query="other.example.com", kind="response")])
    assert merged == 0 and len(out) == 2
    # case differences don't count
    out, merged = zfe.join_split_transactions([record(100.0), record(101.0, query="Q.Example.com", kind="response")])
    assert merged == 1


def test_rejoin_accepts_response_without_query():
    response = record(102.0, query=None, kind="response", rcode=2, rcode_name="SERVFAIL", rejected=True)
    out, merged = zfe.join_split_transactions([record(100.0), response])
    assert merged == 1
    assert out[0]["rcode"] == 2 and out[0]["rejected"] is True
    assert out[0]["query"] == "q.example.com"
    assert "rtt" not in out[0]  # no answers, so no rtt (same as Zeek)


def test_rejoin_never_consumes_a_paired_record():
    # A record with both qtype and rcode was paired by Zeek; it isn't a
    # separately logged response even though it has no rtt.
    out, merged = zfe.join_split_transactions([record(100.0), record(101.0, kind="paired")])
    assert merged == 0 and len(out) == 2


def test_rejoin_pairs_in_time_order():
    queries = [record(100.0), record(104.0)]
    responses = [record(102.0, kind="response", rcode=0), record(105.0, kind="response", rcode=2)]
    out, merged = zfe.join_split_transactions(responses + queries)
    assert merged == 2
    assert [(r["ts"], r["rcode"]) for r in out] == [(100.0, 0), (104.0, 2)]


def test_unmatched_responses_are_dropped_after_the_join():
    records = [record(100.0), record(101.0, kind="response"),        # merged
               record(200.0, trans_id=2, kind="response"),            # no query
               record(300.0, trans_id=3, kind="paired"),              # paired by Zeek, kept
               record(400.0, trans_id=4)]                             # no response, kept
    out, stats = zfe.prepare_records(records)
    assert stats == {"rejoined": 1, "unmatched_responses_dropped": 1}
    assert [r["ts"] for r in out] == [100.0, 300.0, 400.0]


# ---------------------------------------------------- window assignment --

def test_window_assignment():
    ts = [WINDOW_START - 0.001, WINDOW_START, WINDOW_START + 59.999, WINDOW_START + 60.0,
          WINDOW_START + 30.0, WINDOW_START + 200.0]
    frame = pd.DataFrame({"ts": ts, "capture_id": ["a"] * 4 + ["b"] * 2})
    out = zfe.assign_windows(frame, window_seconds=60)
    assert out["window_id"].tolist() == [0, 1, 1, 2, 0, 3]
    assert out["window_start"].tolist() == [WINDOW_START - 60, WINDOW_START, WINDOW_START,
                                            WINDOW_START + 60, WINDOW_START, WINDOW_START + 180]
    assert out["window_query_count"].tolist() == [1, 2, 2, 1, 1, 1]


def test_fixture_windows(capture):
    assert capture.loc[capture["uid"] == "CNormalB", "window_id"].tolist() == [1]
    assert (capture.loc[capture["uid"] != "CNormalB", "window_id"] == 0).all()
    assert capture.loc[capture["window_id"] == 0, "window_query_count"].unique().tolist() == [7]
    assert capture.loc[capture["window_id"] == 1, "window_query_count"].unique().tolist() == [1]


# ---------------------------------------------------- domain aggregates --

def test_domain_aggregates_per_window(aggregated):
    # example.com, window 0: www, nosuch, lost, split, late, mismatch
    r = row(aggregated, "CNormalA")
    assert r["domain_query_count"] == 6
    assert r["domain_unique_qnames"] == 6
    assert r["domain_unique_subdomain_ratio"] == 1.0
    assert r["domain_qtype_diversity"] == 3  # A, AAAA, TXT
    assert r["domain_avg_qname_len"] == pytest.approx(102 / 6)
    assert r["domain_std_qname_len"] == pytest.approx(np.std([15, 18, 16, 17, 16, 20], ddof=1))
    assert r["domain_txt_ratio"] == pytest.approx(1 / 6)
    assert r["domain_null_ratio"] == 0
    assert r["domain_duration"] == pytest.approx(11.5)
    assert r["domain_query_rate"] == pytest.approx(6 / 11.5)

    # the same domain in the next window is aggregated on its own
    later = row(aggregated, "CNormalB")
    assert later["domain_query_count"] == 1
    assert later["domain_std_qname_len"] == 0
    assert later["domain_duration"] == 1.0
    assert later["domain_query_rate"] == 1.0


def test_new_per_domain_ratios(aggregated):
    r = row(aggregated, "CNormalA")
    assert r["nxdomain_ratio"] == pytest.approx(1 / 6)
    assert r["no_response_ratio"] == pytest.approx(3 / 6)  # lost, late query, mismatch query
    assert r["rejected_ratio"] == pytest.approx(1 / 6)
    # rcodes of the answered records: 0 (www), 3 (nosuch), 0 (split)
    assert r["rcode_entropy"] == pytest.approx(-(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3)))

    tunnel = row(aggregated, "CTunnelTxt")  # example.net, window 0: 1 record
    assert tunnel["domain_query_count"] == 1
    assert tunnel["domain_txt_ratio"] == 1
    assert (tunnel["nxdomain_ratio"], tunnel["no_response_ratio"], tunnel["rejected_ratio"]) == (0, 0, 0)
    assert tunnel["rcode_entropy"] == 0


def test_rcode_entropy_is_zero_without_responses():
    frame = zfe.records_to_frame([record(100.0), record(101.0, trans_id=2)])
    frame["capture_id"] = "c"
    out = zfe.add_domain_aggregates(zfe.assign_windows(frame))
    assert out["no_response_ratio"].tolist() == [1.0, 1.0]
    assert out["rcode_entropy"].tolist() == [0.0, 0.0]


def test_aggregates_keep_every_row(capture, aggregated):
    assert len(aggregated) == len(capture)
    aggregate_columns = [c for c in zfe.FEATURE_COLUMNS if c.startswith("domain_") or c.endswith(("_ratio", "_entropy"))]
    assert not aggregated[aggregate_columns].isna().any().any()


# ------------------------------------------------------ manifest checks --

def manifest_row(category, tool):
    return {"capture_id": f"{category}/{tool}", "category": category, "tool": tool, "label": "1",
            "zeek_dns_log": f"Data/zeek/{category}/{tool}/dns.log"}


def test_tool_categories_ok():
    zfe.check_tool_categories([manifest_row("tunnel", "iodine-txt"), manifest_row("normal", "normal"),
                               manifest_row("own_benign", "own_benign"),
                               manifest_row("unknownTunnel", "dns2tcp-key"),
                               manifest_row("crossEndPoint", "AndIodine-TXT")])


def test_tool_in_training_and_evaluation_categories_fails():
    with pytest.raises(ValueError, match="iodine-txt"):
        zfe.check_tool_categories([manifest_row("tunnel", "iodine-txt"),
                                   manifest_row("crossEndPoint", "Iodine-TXT")])


def test_dns2tcp_key_outside_unknown_tunnel_fails():
    with pytest.raises(ValueError, match="dns2tcp-key"):
        zfe.check_tool_categories([manifest_row("crossEndPoint", "dns2tcp-key")])


def test_unknown_category_fails():
    with pytest.raises(ValueError, match="Unknown categories"):
        zfe.check_tool_categories([manifest_row("mystery", "x")])


def test_build_dataset_from_manifest(tmp_path):
    folder = tmp_path / "tunnel" / "sample"
    folder.mkdir(parents=True)
    (folder / "dns.log").write_bytes(FIXTURE.read_bytes())
    rows = [{"capture_id": "tunnel/sample", "category": "tunnel", "tool": "sample", "label": "1",
             "zeek_dns_log": str(folder / "dns.log")}]
    df = zfe.build_dataset_from_manifest(rows, verbose=False)
    assert len(df) == 8
    assert set(zfe.FEATURE_COLUMNS) <= set(df.columns)
    assert df.attrs["capture_stats"] == [{"capture_id": "tunnel/sample", "category": "tunnel", "records": 8,
                                          "malformed_lines": 1, "rejoined": 1,
                                          "unmatched_responses_dropped": 2}]


def test_build_dataset_refuses_backup_folders(tmp_path):
    folder = tmp_path / "sample_Backup"
    folder.mkdir()
    (folder / "dns.log").write_bytes(FIXTURE.read_bytes())
    rows = [{"capture_id": "tunnel/sample", "category": "tunnel", "tool": "sample", "label": "1",
             "zeek_dns_log": str(folder / "dns.log")}]
    with pytest.raises(ValueError, match="_Backup"):
        zfe.build_dataset_from_manifest(rows, verbose=False)


# ------------------------------------------------ live chunk folders --

def chunk_name(index, started):
    return f"capture_{index:05d}_{started:%Y%m%d%H%M%S}"


def write_chunk(zeek_dir, index, started, records):
    folder = zeek_dir / chunk_name(index, started)
    folder.mkdir(parents=True)
    if records is not None:
        (folder / "dns.log").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    else:
        (folder / "conn.log").write_text("{}\n", encoding="utf-8")  # a chunk without DNS traffic
    return folder


@pytest.fixture
def live_dir(tmp_path):
    """Two capture_live.py sessions of 30 s chunks.

    Session 1: chunks 1-3; a transaction straddles chunks 1 and 2 and
    chunk 3 has no DNS traffic. Session 2 restarts at index 1 an hour later.
    """
    zeek_dir = tmp_path / "datas" / "zeek"
    t0 = datetime(2026, 9, 18, 10, 0, 0)
    base = t0.timestamp()
    write_chunk(zeek_dir, 1, t0, [record(base + 5, trans_id=1, uid="Ca"),
                                   record(base + 29, trans_id=2, query="x.tunnel.test", uid="Cq")])
    write_chunk(zeek_dir, 2, t0.replace(second=30), [
        record(base + 31, trans_id=2, query="x.tunnel.test", kind="response", uid="Cr",
               answers=["TXT 1 a"], TTLs=[0.0]),
        record(base + 45, trans_id=3, query="y.tunnel.test", uid="Cb")])
    write_chunk(zeek_dir, 3, t0.replace(minute=1), None)
    later = t0.replace(hour=11)
    write_chunk(zeek_dir, 1, later, [record(later.timestamp() + 1, trans_id=9, uid="Cc")])
    write_chunk(zeek_dir, 2, later.replace(second=30), [record(later.timestamp() + 31, trans_id=10, uid="Cd")])
    (zeek_dir / (chunk_name(3, later.replace(minute=1)) + "_Backup")).mkdir()
    return zeek_dir, base, later.timestamp()


def test_find_live_sessions(live_dir):
    zeek_dir, first_start, second_start = live_dir
    sessions = zfe.find_live_sessions(zeek_dir, now=second_start + 45)
    assert [s["session_id"] for s in sessions] == [
        "capture_00001_20260918100000", "capture_00001_20260918110000"]
    first, second = sessions
    assert first["chunks"] == ["capture_00001_20260918100000", "capture_00002_20260918100030",
                               "capture_00003_20260918100100"]
    assert [p.parent.name for p in first["dns_logs"]] == first["chunks"][:2]
    assert first["start"] == first_start
    assert first["complete"]          # a later session exists
    assert not second["complete"]     # last chunk started 15 s before `now`
    assert zfe.find_live_sessions(zeek_dir, now=second_start + 3600)[1]["complete"]


def test_missing_chunk_index_starts_a_new_session(tmp_path):
    t0 = datetime(2026, 9, 18, 10, 0, 0)
    for index, second in [(1, 0), (2, 30), (4, 30)]:
        write_chunk(tmp_path, index, t0.replace(minute=index // 2, second=second), [record(t0.timestamp() + index)])
    sessions = zfe.find_live_sessions(tmp_path, now=t0.timestamp())
    assert [len(s["chunks"]) for s in sessions] == [2, 1]


def test_live_session_is_one_capture_across_chunks(live_dir):
    zeek_dir, base, second_start = live_dir
    rows = zfe.own_benign_manifest_rows(zeek_dir, now=second_start + 45)
    assert [r["capture_id"] for r in rows] == ["own_benign/capture_00001_20260918100000"]  # complete only
    assert rows[0]["category"] == "own_benign" and rows[0]["label"] == "0"
    assert len(zfe.own_benign_manifest_rows(zeek_dir, include_incomplete=True, now=second_start + 45)) == 2

    df = zfe.build_dataset_from_manifest(rows, verbose=False)
    assert df.attrs["capture_stats"][0]["rejoined"] == 1   # query in chunk 1, response in chunk 2
    assert df["uid"].tolist() == ["Ca", "Cq", "Cb"]
    assert set(df["Label"]) == {"benign"}
    # windows follow the clock, not the 30 s chunks
    assert df["window_start"].tolist() == [math.floor(base / 60) * 60] * 3
    tunnel_domain = df[df["base_domain"] == "tunnel.test"]
    assert tunnel_domain["domain_query_count"].tolist() == [2, 2]
    assert row(df, "Cq")["response_latency"] == pytest.approx(2.0)
