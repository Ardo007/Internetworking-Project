import json
import math

import numpy as np
import pandas as pd
import pytest

import zeek_experiments as zx


def history(val_losses):
    return {"loss": [v * 0.9 for v in val_losses], "val_loss": list(val_losses),
            "learning_rate": [1e-3] * (len(val_losses) - 1) + [1.25e-4]}


def result(name, metrics, histories=None, epochs=None, feature_set="all_minus_artefact_suspect", **extra):
    summary = pd.DataFrame({"min": metrics, "avg": metrics, "max": metrics}).astype(float)
    sizes = pd.DataFrame([{"group": "unseen_tool", "bucket": b, "rows": 10, "rate": metrics.get("unseen_tool", 0.5)}
                          for _, _, b in zx.WINDOW_SIZE_BUCKETS])
    return {"name": name, "config": "B", "feature_set": feature_set, "summary": summary, "per_run": [metrics],
            "window_sizes": sizes, "epochs": epochs or [len(h["val_loss"]) for h in histories or []],
            "histories": histories, "input_columns": ["a", "b"], "train_rows": {"benign": 2, "tunnel": 1},
            "settings": {"epoch_cap": 150, "row_caps": {"default": 200}, "sampling_seed": 0, "window_seconds": 60,
                         "feature_columns": ["a", "b"], "splits_sha256": "abc"},
            "git": {"commit": "deadbeef", "dirty": False}, "environment": {"python": "3.13"},
            "trained_at": "2026-09-18T00:00:00+00:00", "source": "test", **extra}


# ------------------------------------------------------ training curves --

def test_training_curves():
    curves = zx.training_curves([history([0.5, 0.4, 0.3, 0.35, 0.36]), history([0.5] * 6)], epoch_cap=6, compare_at=3)
    first, second = curves.iloc[0], curves.iloc[1]
    assert (first["stopped_epoch"], first["best_epoch"], first["early_stopped"]) == (5, 3, True)
    assert first["val_loss_at_3"] == 0.3 and first["best_val_loss"] == 0.3 and first["last_val_loss"] == 0.36
    assert first["final_learning_rate"] == 1.25e-4
    assert (second["stopped_epoch"], second["best_epoch"], second["early_stopped"]) == (6, 1, False)


def test_training_curves_before_the_comparison_epoch():
    curves = zx.training_curves([history([0.5, 0.4])], epoch_cap=150)
    assert math.isnan(curves.iloc[0]["val_loss_at_50"])


# ------------------------------------------------------------ sections --

def test_upsert_section_appends_then_replaces_only_its_section(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Title\n\nrun 1 table\n", encoding="utf-8")
    zx.upsert_section(report, "run2", "## Run 2\n\nfirst version")
    zx.upsert_section(report, "run3", "## Run 3")
    zx.upsert_section(report, "run2", "## Run 2\n\nsecond version")
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nrun 1 table\n")
    assert "second version" in text and "first version" not in text
    assert text.index("## Run 2") < text.index("## Run 3")
    assert text.count("<!-- section:run2:start -->") == 1


# ------------------------------------------------------------- results --

def test_save_and_load_run(tmp_path):
    original = result("B", {"unseen_tool": 0.99, "fpr_normal": 0.0001}, [history([0.5, 0.4, 0.45])],
                      family_rows=12)
    zx.save_result("run2", "B", original, runs_dir=tmp_path)
    loaded = zx.load_run("run2", runs_dir=tmp_path)["B"]
    pd.testing.assert_frame_equal(loaded["summary"], original["summary"])
    pd.testing.assert_frame_equal(loaded["window_sizes"], original["window_sizes"], check_dtype=False)
    assert loaded["histories"] == original["histories"]
    assert loaded["epochs"] == [3] and loaded["family_rows"] == 12 and loaded["run"] == "run2"


def test_load_run_needs_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        zx.load_run("missing", runs_dir=tmp_path)


def test_run_json_drives_the_report_and_is_not_a_configuration(tmp_path):
    runs = tmp_path / "runs"
    zx.save_result("run2", "B", result("B", {"test_accuracy": 0.99}, epochs=[50]), runs_dir=runs)
    (runs / "run2" / zx.RUN_METADATA).write_text(
        json.dumps({"title": "Run 2: a title", "description": ["why it ran"], "run_label": "run 2"}), encoding="utf-8")
    assert list(zx.load_run("run2", runs_dir=runs)) == ["B"]
    report = tmp_path / "zeek_run.md"
    zx.write_run_report("run2", report_path=report, runs_dir=runs)
    text = report.read_text(encoding="utf-8")
    assert "## Run 2: a title" in text and "why it ran" in text and "| metric | run 2 B |" in text
    (runs / "run2" / zx.RUN_METADATA).write_text(json.dumps({"title": "x", "layout": "mystery"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mystery"):
        zx.write_run_report("run2", report_path=report, runs_dir=runs)


def test_save_result_leaves_no_temporary_files(tmp_path):
    zx.save_result("run2", "B", result("B", {"test_accuracy": 0.99}, epochs=[50]), runs_dir=tmp_path)
    assert [p.name for p in (tmp_path / "run2").iterdir()] == ["B.json"]


def test_render_run_section_next_to_reference(tmp_path):
    runs = tmp_path / "runs"
    reference = {"B": result("B", {"test_accuracy": 0.99, "unseen_tool": 0.97, "unseen_tool/ozymandns": 0.2},
                             epochs=[50] * 2, feature_set="all"),
                 "B-all_minus_artefact_suspect": result("B-all_minus_artefact_suspect",
                                                        {"test_accuracy": 0.99, "unseen_tool": 0.99}, epochs=[50] * 2),
                 "lofo-dnspot": result("lofo-dnspot", {"held_out_family": 0.04, "fpr_normal": 0.0}, epochs=[50])}
    new = {"B": result("B", {"test_accuracy": 0.999, "unseen_tool": 0.995, "unseen_tool/ozymandns": 0.6},
                       [history([0.5] * 50 + [0.4, 0.45, 0.46, 0.47, 0.48, 0.49]), history([0.3] * 20)]),
           "lofo-dnspot": result("lofo-dnspot", {"held_out_family": 0.5, "fpr_normal": 0.0, "held_out_family/dnspot": 0.5},
                                 [history([0.2, 0.1])], family_rows=26687)}
    for run, results in (("run1", reference), ("run2", new)):
        for name, r in results.items():
            zx.save_result(run, name, r, runs_dir=runs)
    markdown = zx.render_run_section("run2", "Run 2", description=["why"], reference_name="run1",
                                     run_label="run 2", reference_label="run 1",
                                     extra_reference={"B": ["B-all_minus_artefact_suspect"]},
                                     runs_dir=runs, figures_dir=tmp_path / "figures")
    assert "## Run 2" in markdown and "why" in markdown
    assert "| metric | run 1 B | run 1 B-all_minus_artefact_suspect | run 2 B |" in markdown
    assert "99.50%" in markdown and "97.00%" in markdown
    assert "ozymandns" in markdown and "60.00%" in markdown
    assert "### Training length" in markdown and "56, 20" in markdown  # stopped epochs of the two models
    assert "| dnspot | 26,687 |" in markdown and "4.00%" in markdown and "50.00%" in markdown
    assert (tmp_path / "figures" / "run2_B_loss.png").exists()
    assert (tmp_path / "figures" / "run2_lofo-dnspot_loss.png").exists()


def lofo_result(name, recall, captures, family_rows, epochs=(50,), fpr=0.0):
    metrics = {"held_out_family": recall, "fpr_normal": fpr, "fpr_wildcard": fpr,
               **{f"held_out_family/{c}": v for c, v in captures.items()}}
    return result(name, metrics, [history([0.5] * e) for e in epochs], family_rows=family_rows)


def test_render_lofo_comparison(tmp_path):
    runs = tmp_path / "runs"
    data = {
        "old": {"lofo-iodine": lofo_result("lofo-iodine", 0.6, {"iodine-NULL": 0.8, "iodine-a": 0.0}, 300),
                "lofo-tuns": lofo_result("lofo-tuns", 1.0, {"tuns": 1.0}, 100)},
        "long": {"lofo-iodine": lofo_result("lofo-iodine", 0.35, {"iodine-NULL": 0.1, "iodine-a": 0.0}, 300, (150,)),
                 "lofo-tuns": lofo_result("lofo-tuns", 0.9, {"tuns": 0.9}, 100, (150,))},
        "new": {"lofo-iodine": lofo_result("lofo-iodine", 0.4, {"iodine-NULL": 0.2, "iodine-a": 0.0}, 300),
                "lofo-tuns": lofo_result("lofo-tuns", 0.9, {"tuns": 0.9}, 100)},
    }
    for run, results in data.items():
        for name, r in results.items():
            zx.save_result(run, name, r, runs_dir=runs)
    zx.save_diagnostic("new", "neutral", {
        "description": "diagnostic text", "models": "models/x/<name>/", "features": ["f"],
        "results": {"lofo-iodine": [
            {"variant": "as trained", "summary": {"held_out_family": {"avg": 0.6}, "fpr_wildcard": {"avg": 0.0},
                                                   "held_out_family/iodine-NULL": {"avg": 0.8}}},
            {"variant": "without f", "summary": {"held_out_family": {"avg": 0.45}, "fpr_wildcard": {"avg": 0.02},
                                                  "held_out_family/iodine-NULL": {"avg": 0.3}}}]},
        "feature_values": {"wildcard": {"rows": 10, "share_above_1": 99.998}}}, runs_dir=runs)
    (runs / "new" / zx.RUN_METADATA).write_text(json.dumps({
        "layout": "lofo_comparison", "title": "Run 3",
        "columns": [{"run": "old", "label": "run 1", "features": "30", "epochs": 50},
                    {"run": "long", "label": "run 2", "features": "28", "epochs": 150},
                    {"run": "new", "label": "run 3", "features": "28", "epochs": 50}],
        "effects": [{"label": "feature change", "from": "old", "to": "new"},
                    {"label": "more epochs", "from": "new", "to": "long"}],
        "highlight_captures": ["iodine-NULL"], "diagnostic": "neutral", "findings": ["the finding"]}), encoding="utf-8")
    report = tmp_path / "zeek_run.md"
    report.write_text("# Report\n\nrun 1 text\n", encoding="utf-8")
    zx.write_run_report("new", report_path=report, runs_dir=runs)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Report\n\nrun 1 text\n")
    # 2x2 grid of row-weighted recall: (300*0.6 + 100*1.0) / 400 = 70%
    assert "| 30 features | run 1: **70.00%** | not run |" in text
    assert "| 28 features | run 3: **52.50%** | run 2: **48.75%** |" in text
    assert "| iodine | 300 | 60.00% (60.00% – 60.00%) | 35.00% (35.00% – 35.00%) | 40.00% (40.00% – 40.00%) " \
           "| -20.00 pp | -5.00 pp |" in text
    assert "| **all five (row-weighted)** | 400 | 70.00% | 48.75% | 52.50% | -17.50 pp | -3.75 pp |" in text
    assert text.index("**iodine-NULL**") < text.index("| iodine-a |")          # highlighted captures first
    assert "diagnostic text" in text and "| &nbsp;&nbsp;iodine-NULL | 80.00% | 30.00% |" in text
    assert "| wildcard | 10 | 100.00% |" in text
    assert "### Findings" in text and "the finding" in text


def test_neutralised_scores(monkeypatch):
    import model_artifacts as ma

    class Encoder:
        classes_ = ["benign", "tunnel"]

    class Model:
        # predicts tunnel when input column 1 (the feature under test) is positive
        def predict(self, X, batch_size=None, verbose=0):
            tunnel = X[:, 0, 1] > 0
            return np.stack([~tunnel, tunnel], axis=1).astype(float)

    table = pd.DataFrame({"Label": ["tunnel", "tunnel", "benign"], "category": ["unknownTunnel"] * 2 + ["normal"],
                          "capture_id": ["u/a", "u/b", "n/c"], "tool": ["a", "b", "normal"], "family": ["a", "b", "normal"],
                          "window_query_count": [100, 100, 100], "split_B": ["heldout"] * 3,
                          "role_B": ["unseen_tool", "unseen_tool", "fpr_normal"], "sample_B": [False] * 3})
    X = np.array([[[0.5, 1.0, 2.0]], [[0.5, 1.0, -1.0]], [[0.5, -1.0, 1.0]]])
    monkeypatch.setattr(ma, "load_artifacts", lambda d: {"features": {"input_columns": ["x", "f1", "f2"]},
                                                         "models": [Model()], "label_encoder": Encoder()})
    monkeypatch.setattr(ma, "prepare_model_input", lambda frame, art: X[: len(frame)])
    out = zx.neutralised_scores(table, "ignored", "B", ["f1", "f2"])
    assert [o["variant"] for o in out] == ["as trained", "without f1", "without f2", "without both"]
    recall = {o["variant"]: o["summary"].loc["unseen_tool", "avg"] for o in out}
    assert recall == {"as trained": 1.0, "without f1": 0.0, "without f2": 1.0, "without both": 0.0}
    assert X[0, 0, 1] == 1.0  # the caller's matrix is not modified


# ------------------------------------------------------ leave one out --

def test_lofo_masks():
    table = pd.DataFrame({
        "category": ["tunnel", "tunnel", "tunnel", "normal", "normal", "wildcard", "unknownTunnel"],
        "family": ["dnspot", "dnspot", "tuns", "normal", "normal", "wildcard", "ozymandns"],
        "split_B": ["train", pd.NA, "train", "train", "heldout", "heldout", "heldout"],
        "role_B": ["fit", pd.NA, "fit", "fit", "fpr_normal", "fpr_wildcard", "unseen_tool"],
        "sample_B": [True, False, True, True, False, False, False],
    })
    masks, relabel = zx.lofo_masks(table, "dnspot", "B")
    assert masks["train"].tolist() == [False, False, True, True, False, False, False]
    assert masks["test"].tolist() == [True, True, False, False, False, False, False]      # gap rows too
    assert masks["evaluation"].tolist() == [True, True, False, False, True, True, False]  # no unknownTunnel
    meta = relabel(table.loc[masks["evaluation"], ["role_B"]].rename(columns={"role_B": "role"}))
    assert meta["role"].tolist() == ["held_out_family", "held_out_family", "fpr_normal", "fpr_wildcard"]
    assert set(meta["split"]) == {"heldout"}
    with pytest.raises(ValueError):
        zx.lofo_masks(table, "iodine", "B")
