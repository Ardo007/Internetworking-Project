import pandas as pd
import pytest

import dataset_splits as ds
import zeek_feature_extraction as zfe

W = zfe.WINDOW_SECONDS
ORIGIN = 1_700_000_040.0  # a multiple of 60


def captures_table():
    rows = []

    def add(capture_id, category, tool, label, n_windows, first_ts=ORIGIN):
        rows.append({"capture_id": capture_id, "category": category, "tool": tool, "label": label,
                     "first_ts": first_ts, "last_ts": first_ts + n_windows * W - 1,
                     "window_origin": first_ts, "n_windows": n_windows})

    for i in range(68):
        add(f"normal/normal_{i:05d}_20230805150331", "normal", "normal", "benign", 30)
    for i in range(13):
        add(f"wildcard/wildcard_{i:05d}_20231011215521", "wildcard", "wildcard", "benign", 4)
    add("tunnel/iodine-private", "tunnel", "iodine-private", "tunnel", 27)
    add("tunnel/dnscat2-mx", "tunnel", "dnscat2-mx", "tunnel", 145)
    add("unknownTunnel/dns2tcp-key", "unknownTunnel", "dns2tcp-key", "tunnel", 192)
    add("crossEndPoint/AndIodine-TXT", "crossEndPoint", "AndIodine-TXT", "tunnel", 141)
    for n, (session, start) in enumerate([("capture_00001_20260918100000", 0),
                                          ("capture_00001_20260919100000", 86400),
                                          ("capture_00001_20260920100000", 2 * 86400)]):
        add(f"own_benign/{session}", "own_benign", "own_benign", "benign", 10 + n, ORIGIN + start)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits():
    return ds.make_splits(captures_table())


def only(splits, config, capture_id):
    return splits[(splits["config"] == config) & (splits["capture_id"] == capture_id)]


# --------------------------------------------------------------- rules --

def test_timeline_segments_single_capture():
    # iodine-private: 27 windows -> 25 usable after two one-window gaps
    segments = ds.timeline_segments([27])
    assert segments == [(0, "train", 0, 17), (0, "val", 19, 22), (0, "test", 24, 26)]


def test_timeline_segments_across_captures():
    segments = ds.timeline_segments([10, 11])
    covered = {(pos, w) for pos, _, first, last in segments for w in range(first, last + 1)}
    assert len(covered) == 21 - 2  # everything except the two gap windows
    assert [s[1] for s in segments] == ["train", "train", "val", "test"]
    assert segments[0] == (0, "train", 0, 9)
    assert segments[-1][0] == 1 and segments[-1][3] == 10


def test_timeline_segments_too_short():
    with pytest.raises(ValueError):
        ds.timeline_segments([4])


def test_normal_blocks(splits):
    for config in ds.CONFIGS:
        normal = splits[(splits["config"] == config) & (splits["category"] == "normal")]
        by_split = normal.groupby("split")["capture_id"].apply(lambda s: sorted(map(ds.capture_index, s)))
        assert by_split["train"] == list(range(0, 48))
        assert by_split["val"] == list(range(48, 55))
        assert by_split["test"] == list(range(55, 62))
        assert by_split["heldout"] == list(range(62, 68))
        assert set(normal.loc[normal["split"] == "heldout", "role"]) == {"fpr_normal"}


def test_wildcard_configs(splits):
    a = splits[(splits["config"] == "A") & (splits["category"] == "wildcard")]
    assert set(a["split"]) == {"heldout"} and len(a) == 13
    b = splits[(splits["config"] == "B") & (splits["category"] == "wildcard")]
    by_split = b.groupby("split")["capture_id"].apply(lambda s: sorted(map(ds.capture_index, s)))
    assert by_split["train"] == [0, 1, 2, 3, 4, 5]
    assert by_split["val"] == [6]
    assert by_split["heldout"] == list(range(7, 13))


def test_tunnel_time_segments(splits):
    segments = only(splits, "B", "tunnel/iodine-private").sort_values("first_window")
    assert segments["split"].tolist() == ["train", "val", "test"]
    assert segments[["first_window", "last_window"]].values.tolist() == [[0, 17], [19, 22], [24, 26]]
    assert segments["t_start"].tolist() == [ORIGIN, ORIGIN + 19 * W, ORIGIN + 24 * W]
    assert segments["t_end"].tolist() == [ORIGIN + 18 * W, ORIGIN + 23 * W, ORIGIN + 27 * W]
    assert not segments["whole_capture"].any()
    assert set(splits.loc[splits["category"] == "tunnel", "family"]) == {"iodine", "dnscat2"}


def test_evaluation_only_categories(splits):
    held = splits[splits["category"].isin(["unknownTunnel", "crossEndPoint"])]
    assert set(held["split"]) == {"heldout"}
    assert set(held.loc[held["category"] == "unknownTunnel", "role"]) == {"unseen_tool"}
    cross = held[held["category"] == "crossEndPoint"]
    assert set(cross["role"]) == {"unseen_platform"}
    assert set(cross["family"]) == {"iodine (Android)"}


def test_own_benign_sessions(splits):
    own = splits[(splits["config"] == "B") & (splits["category"] == "own_benign")]
    latest = own[own["capture_id"] == "own_benign/capture_00001_20260920100000"]
    assert latest["split"].tolist() == ["heldout"] and latest["role"].tolist() == ["fpr_own_benign"]
    earlier = own[own["capture_id"] != "own_benign/capture_00001_20260920100000"]
    # sessions of 10 and 11 windows form one 21-window timeline
    assert sorted(zip(earlier["capture_id"].str[-14:], earlier["split"])) == [
        ("20260918100000", "train"), ("20260919100000", "test"),
        ("20260919100000", "train"), ("20260919100000", "val")]


def test_configs_share_everything_but_wildcard(splits):
    columns = ["capture_id", "split", "first_window", "last_window"]
    a = splits[(splits["config"] == "A") & (splits["category"] != "wildcard")][columns].reset_index(drop=True)
    b = splits[(splits["config"] == "B") & (splits["category"] != "wildcard")][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_category_has_no_rule():
    table = captures_table()
    table.loc[0, "category"] = "mystery"
    with pytest.raises(ValueError, match="mystery"):
        ds.make_splits(table)


# ---------------------------------------------------- overlap checks --

def test_capture_in_two_splits_is_rejected(splits):
    duplicate = only(splits, "B", "normal/normal_00000_20230805150331").assign(split="test")
    with pytest.raises(ValueError, match="in 2 splits"):
        ds.validate_splits(pd.concat([splits, duplicate]))


def test_overlapping_time_segments_are_rejected(splits):
    broken = splits.copy()
    val = (broken["config"] == "B") & (broken["capture_id"] == "tunnel/iodine-private") & (broken["split"] == "val")
    broken.loc[val, "first_window"] = 17
    with pytest.raises(ValueError, match="overlap"):
        ds.validate_splits(broken)


def test_segments_without_gap_are_rejected(splits):
    broken = splits.copy()
    val = (broken["config"] == "B") & (broken["capture_id"] == "tunnel/iodine-private") & (broken["split"] == "val")
    broken.loc[val, "first_window"] = 18
    with pytest.raises(ValueError, match="no gap"):
        ds.validate_splits(broken)


def test_evaluation_only_capture_in_training_is_rejected(splits):
    broken = splits.copy()
    broken.loc[broken["category"] == "unknownTunnel", "split"] = "train"
    with pytest.raises(ValueError, match="evaluation-only"):
        ds.validate_splits(broken)


def test_config_a_wildcard_training_is_rejected(splits):
    broken = splits.copy()
    broken.loc[(broken["config"] == "A") & (broken["category"] == "wildcard"), "split"] = "train"
    with pytest.raises(ValueError, match="config A"):
        ds.validate_splits(broken)


def test_committed_splits_file():
    """No capture, and no tunnel time segment, is in more than one split."""
    splits = ds.read_splits()  # validates
    assert set(splits["config"]) == {"A", "B"}
    assert (splits["window_seconds"] == W).all()
    for config, rows in splits.groupby("config"):
        # expand every row to the windows it covers; each window has one split
        windows = [(r.capture_id, w) for r in rows.itertuples() for w in range(r.first_window, r.last_window + 1)]
        assert len(windows) == len(set(windows)), f"config {config}: a window is in two splits"
        whole = rows[rows["category"] != "tunnel"]
        assert not whole["capture_id"].duplicated().any()
        assert whole["whole_capture"].all()
        assert rows["capture_id"].nunique() == 105
        tunnel = rows[rows["category"] == "tunnel"]
        assert tunnel.groupby("capture_id")["split"].apply(lambda s: sorted(s) == ["test", "train", "val"]).all()
    assert set(splits.loc[splits["category"] == "tunnel", "family"]) == set(ds.TUNNEL_FAMILIES)


# ------------------------------------------------------- assignment --

def frame_for(capture_id, category, label, n_windows, rows_per_window=3):
    rows = []
    for w in range(n_windows):
        for k in range(rows_per_window):
            rows.append({"capture_id": capture_id, "category": category, "Label": label,
                         "ts": ORIGIN + w * W + k, "window_start": ORIGIN + w * W, "window_id": w})
    return pd.DataFrame(rows)


def test_assign_splits_leaves_gap_windows_unassigned(splits):
    df = pd.concat([frame_for("tunnel/iodine-private", "tunnel", "tunnel", 27),
                    frame_for("wildcard/wildcard_00006_20231011215521", "wildcard", "benign", 4)],
                   ignore_index=True)
    b = ds.assign_splits(df, splits, "B")
    tunnel = b[df["capture_id"] == "tunnel/iodine-private"]
    per_window = tunnel.groupby(df.loc[tunnel.index, "window_id"])["split"].first()
    assert per_window.isna().tolist() == [w in (18, 23) for w in range(27)]
    assert per_window.loc[0] == "train" and per_window.loc[19] == "val" and per_window.loc[26] == "test"
    assert set(b.loc[df["category"] == "wildcard", "split"]) == {"val"}
    a = ds.assign_splits(df, splits, "A")
    assert set(a.loc[df["category"] == "wildcard", "role"]) == {"fpr_wildcard"}


def test_assign_splits_rejects_unknown_captures(splits):
    df = frame_for("tunnel/not-in-splits", "tunnel", "tunnel", 3)
    with pytest.raises(ValueError, match="missing from the splits table"):
        ds.assign_splits(df, splits, "B")


def test_assign_splits_rejects_other_window_length(splits):
    df = frame_for("tunnel/iodine-private", "tunnel", "tunnel", 3)
    with pytest.raises(ValueError, match="window length"):
        ds.assign_splits(df, splits, "B", window_seconds=30)


def test_cap_rows_per_window():
    df = pd.concat([frame_for("a", "normal", "benign", 2, rows_per_window=10),
                    frame_for("b", "normal", "benign", 1, rows_per_window=2)], ignore_index=True)
    capped = ds.cap_rows_per_window(df, 4, seed=1)
    assert capped.groupby(["capture_id", "window_id"]).size().tolist() == [4, 4, 2]
    pd.testing.assert_frame_equal(capped, ds.cap_rows_per_window(df, 4, seed=1))
