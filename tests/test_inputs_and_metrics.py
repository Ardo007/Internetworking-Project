import math

import numpy as np
import pandas as pd
import pytest

import model_artifacts as ma
import zeek_experiments as zx


# ------------------------------------------------------- input encoding --

def features(proto, qtype):
    return pd.DataFrame({"proto": proto, "qname_len": [10.0] * len(proto), "qtype_name": qtype})


def test_one_hot_columns_do_not_depend_on_the_data():
    only_a = ma.encode_inputs(features(["udp", "udp"], ["A", "A"]))
    mixed = ma.encode_inputs(features(["tcp", "udp"], ["TXT", "HTTPS"]))
    assert list(only_a.columns) == list(mixed.columns)
    qtype_levels = ma.CATEGORY_LEVELS["qtype_name"]
    assert list(only_a.columns) == ["qname_len", "proto_udp"] + [f"qtype_name_{q}" for q in qtype_levels[1:]]
    assert qtype_levels[0] == "A" and ma.CATEGORY_LEVELS["proto"][0] == "tcp"  # dropped by drop_first
    assert mixed.loc[0, "proto_udp"] == 0 and mixed.loc[1, "proto_udp"] == 1
    assert mixed.loc[0, "qtype_name_TXT"] == 1 and mixed.loc[1, "qtype_name_HTTPS"] == 1
    assert only_a.filter(like="qtype_name_").to_numpy().sum() == 0  # "A" is the all-zero level


def test_encode_inputs_reindexes_to_training_columns():
    train = ma.encode_inputs(features(["udp"], ["A"]))
    subset = ma.encode_inputs(features(["udp"], ["MX"])[["qtype_name", "qname_len"]], list(train.columns))
    assert list(subset.columns) == list(train.columns)
    assert subset.loc[0, "proto_udp"] == 0  # missing column filled with 0


def test_encode_inputs_rejects_unknown_categories():
    with pytest.raises(ValueError, match="qtype_name"):
        ma.encode_inputs(features(["udp"], ["KEY"]))  # the extractor maps KEY to OTHER


def test_to_tensor_shape():
    tensor = ma.to_tensor(np.ones((4, 3)))
    assert tensor.shape == (4, 1, 3) and tensor.dtype == np.float32


# -------------------------------------------------------------- metrics --

def meta_and_predictions():
    rows = [
        # split, role, Label, category, capture_id, tool, window size, predicted tunnel
        ("test", "in_distribution_test", "benign", "normal", "normal/normal_00055_20230806134019", "normal", 150, False),
        ("test", "in_distribution_test", "benign", "normal", "normal/normal_00055_20230806134019", "normal", 150, True),
        ("test", "in_distribution_test", "tunnel", "tunnel", "tunnel/tuns", "tuns", 5, True),
        ("test", "in_distribution_test", "tunnel", "tunnel", "tunnel/tuns", "tuns", 150, True),
        ("heldout", "fpr_normal", "benign", "normal", "normal/normal_00062_20230806144212", "normal", 150, False),
        ("heldout", "fpr_wildcard", "benign", "wildcard", "wildcard/wildcard_00003_20231011220812", "wildcard", 150, True),
        ("heldout", "fpr_wildcard", "benign", "wildcard", "wildcard/wildcard_00008_20231011222232", "wildcard", 150, False),
        ("heldout", "unseen_tool", "tunnel", "unknownTunnel", "unknownTunnel/ozymandns", "ozymandns", 50, False),
        ("heldout", "unseen_tool", "tunnel", "unknownTunnel", "unknownTunnel/cobalstrike", "cobalstrike", 150, True),
        ("heldout", "unseen_tool", "tunnel", "unknownTunnel", "unknownTunnel/cobalstrike", "cobalstrike", 150, True),
        ("heldout", "unseen_platform", "tunnel", "crossEndPoint", "crossEndPoint/AndIodine-TXT", "AndIodine-TXT", 150, True),
    ]
    frame = pd.DataFrame(rows, columns=["split", "role", "Label", "category", "capture_id", "tool",
                                        "window_query_count", "predicted"])
    frame["family"] = frame["tool"]
    return frame.drop(columns="predicted"), frame["predicted"].to_numpy()


def test_evaluate():
    meta, predicted = meta_and_predictions()
    m = zx.evaluate(meta, predicted)
    assert m["test_accuracy"] == pytest.approx(3 / 4)
    assert m["test_benign"] == pytest.approx(1 / 2)       # false positive rate
    assert m["test_tunnel"] == pytest.approx(1.0)         # recall
    assert m["fpr_normal"] == 0.0
    assert m["fpr_wildcard"] == pytest.approx(1 / 2)
    assert m["fpr_wildcard_00007_00012"] == 0.0           # only wildcard_00008 is in that range
    assert m["unseen_tool"] == pytest.approx(2 / 3)
    assert m["unseen_tool/ozymandns"] == 0.0 and m["unseen_tool/cobalstrike"] == 1.0
    assert m["unseen_platform"] == 1.0 and m["unseen_platform/AndIodine-TXT"] == 1.0
    assert m["collapsed"] == 0.0


def test_collapsed_run_is_flagged():
    meta, _ = meta_and_predictions()
    assert zx.evaluate(meta, np.ones(len(meta), dtype=bool))["collapsed"] == 1.0


def test_window_size_rates():
    meta, predicted = meta_and_predictions()
    rates = zx.window_size_rates(meta, predicted).set_index(["group", "bucket"])
    assert rates.loc[("test_tunnel", "<10"), "rows"] == 1
    assert rates.loc[("test_tunnel", "<10"), "rate"] == 1.0
    assert rates.loc[("unseen_tool", "10-99"), "rate"] == 0.0     # ozymandns, window of 50
    assert rates.loc[("unseen_tool", ">=100"), "rate"] == 1.0
    assert rates.loc[("fpr_normal", "<10"), "rows"] == 0
    assert math.isnan(rates.loc[("fpr_normal", "<10"), "rate"])


def test_summarise():
    summary = zx.summarise([{"x": 0.2, "y": 1.0}, {"x": 0.4, "y": 1.0}, {"x": 0.9, "y": 1.0}])
    assert summary.loc["x"].tolist() == pytest.approx([0.2, 0.5, 0.9])
    assert summary.loc["y"].tolist() == [1.0, 1.0, 1.0]
