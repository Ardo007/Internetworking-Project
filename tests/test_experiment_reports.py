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


# --------------------------------------------------------- final model --

def test_final_masks_train_on_every_capture():
    import dataset_splits as ds

    rows = []
    for capture, category, split in (("n", "normal", "train"), ("h", "normal", "heldout"), ("u", "unknownTunnel", "heldout"),
                                     ("t", "tunnel", "val"), ("g", "tunnel", None)):
        for window in range(2):
            for i in range(300):
                rows.append({"capture_id": capture, "category": category, "window_id": window, "ts": window * 60.0 + i,
                             "split_B": split})
    table = pd.DataFrame(rows)
    capped = ds.cap_mask(table)
    table["sample_B"] = capped & table["split_B"].isin(["train", "val"])
    masks = zx.final_masks(table, "B")
    kept = masks["train"].groupby([table["capture_id"], table["window_id"]]).sum()
    assert set(kept) == {200}                       # every window of every capture, held-out and gap rows too
    assert masks["val"].equals(table["sample_B"] & table["split_B"].eq("val"))
    assert (masks["train"] | ~masks["val"]).all()   # validation rows are training rows too
    assert masks["test"].equals(masks["val"]) and masks["evaluation"].equals(masks["val"])
    table.loc[table.index[0], "sample_B"] = not table.loc[table.index[0], "sample_B"]
    with pytest.raises(ValueError, match="rebuild"):
        zx.final_masks(table, "B")


def test_training_record_is_not_a_result(tmp_path):
    zx.save_result("run4", "A", result("A", {"test_accuracy": 0.99}, epochs=[50]), runs_dir=tmp_path)
    zx.save_training_record("run4", "final", {"epochs": [50], "train_rows": {"benign": 3, "tunnel": 2}},
                            runs_dir=tmp_path)
    assert list(zx.load_run("run4", runs_dir=tmp_path)) == ["A"]
    assert zx.load_training_record("run4", "final", runs_dir=tmp_path)["not_for_evaluation"] is True
    assert zx.load_result("run4", "A", runs_dir=tmp_path)["summary"].loc["test_accuracy", "avg"] == 0.99
    with pytest.raises(ValueError, match="not_for_evaluation"):
        zx.load_result("run4", "final", runs_dir=tmp_path)


class Encoder:
    classes_ = ["benign", "tunnel"]


class ThresholdModel:
    # predicts tunnel when input column 1 is positive
    def predict(self, X, batch_size=None, verbose=0):
        tunnel = X[:, 0, 1] > 0
        return np.stack([~tunnel, tunnel], axis=1).astype(float)


def scored_table():
    return pd.DataFrame({"Label": ["tunnel", "tunnel", "benign"], "category": ["unknownTunnel"] * 2 + ["normal"],
                         "capture_id": ["u/a", "u/b", "n/c"], "tool": ["a", "b", "normal"], "family": ["a", "b", "normal"],
                         "window_query_count": [100, 100, 100], "split_B": ["heldout"] * 3,
                         "role_B": ["unseen_tool", "unseen_tool", "fpr_normal"], "sample_B": [False] * 3})


def test_forced_feature_scores(monkeypatch):
    import model_artifacts as ma

    class Scaler:
        mean_ = np.array([0.0, 2.0, 0.0])
        scale_ = np.array([1.0, 0.5, 1.0])

    X = np.array([[[0.5, 1.0, 2.0]], [[0.5, -1.0, -1.0]], [[0.5, -1.0, 1.0]]])
    monkeypatch.setattr(ma, "load_artifacts", lambda d: {"features": {"input_columns": ["x", "f", "g"]},
                                                         "models": [ThresholdModel()], "label_encoder": Encoder(),
                                                         "scaler": Scaler()})
    monkeypatch.setattr(ma, "prepare_model_input", lambda frame, art: X[: len(frame)])
    out = zx.forced_feature_scores(scored_table(), "ignored", "B", "f", [1, 3])
    assert [o["variant"] for o in out] == ["as recorded", "set to 1", "set to 3"]
    # scaled: (1 - 2) / 0.5 = -2 -> benign everywhere; (3 - 2) / 0.5 = 2 -> tunnel everywhere
    rates = {o["variant"]: (o["summary"].loc["unseen_tool", "avg"], o["summary"].loc["fpr_normal", "avg"]) for o in out}
    assert rates == {"as recorded": (0.5, 0.0), "set to 1": (0.0, 0.0), "set to 3": (1.0, 1.0)}
    assert X[1, 0, 1] == -1.0  # the caller's matrix is not modified


def test_scoring_refuses_the_final_model(monkeypatch):
    import model_artifacts as ma

    monkeypatch.setattr(ma, "load_artifacts", lambda d: {"features": {"not_for_evaluation": True, "input_columns": ["f"]},
                                                         "directory": d})
    for score in (lambda: zx.neutralised_scores(scored_table(), "models/final", "B", ["f"]),
                  lambda: zx.forced_feature_scores(scored_table(), "models/final", "B", "f", [1]),
                  lambda: zx.rescore_saved(scored_table(), "models/final", "B")):
        with pytest.raises(ValueError, match="not_for_evaluation"):
            score()


def test_render_forced_feature_probe():
    def summary(**metrics):
        return {k: {"avg": v} for k, v in metrics.items()}

    data = {"title": "What the count does", "description": "probe text", "models": "models/x/<name>/",
            "feature": "f", "values": [2],
            "results": {"B": [{"variant": "as recorded", "summary": summary(unseen_tool=0.97, fpr_normal=0.0)},
                              {"variant": "set to 2", "summary": summary(unseen_tool=0.0002, fpr_normal=0.0)}],
                        "lofo-iodine": [{"variant": "as recorded", "summary": summary(held_out_family=0.58)},
                                        {"variant": "set to 2", "summary": summary(held_out_family=0.03)}]}}
    text = "\n".join(zx.render_forced_feature_probe(data))
    assert text.startswith("### What the count does\n\nprobe text")
    assert "| Recall unseen tools (unknownTunnel) | 97.00% | 0.02% |" in text
    assert "Config `B` models (`models/x/B/`)" in text
    assert "| iodine recall | 58.00% | 3.00% |" in text


# ------------------------------------------------- default-settings section --

def default_result(name, metrics, feature_set="all_minus_artefact_suspect", epochs=(50,), **extra):
    import dataset_splits as ds
    import zeek_feature_extraction as zfe

    r = result(name, metrics, [history([0.5] * e) for e in epochs], feature_set=feature_set, **extra)
    r["settings"] = {"epoch_cap": 50, "hyperparameters": {"epochs": 50, "n_neurons": 24, "early_stopping": True},
                     "row_caps": ds.ROW_CAPS, "sampling_seed": ds.SAMPLING_SEED, "model_seed": 0,
                     "window_seconds": zfe.WINDOW_SECONDS, "feature_columns": zfe.FEATURE_SETS[feature_set],
                     "splits_sha256": "abc"}
    r["per_run"] = [metrics] * 5
    return r


def test_upsert_section_at_top(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Title\n\nrun 1 text\n", encoding="utf-8")
    zx.upsert_section(report, "defaults", "## Defaults\n\nfirst", at_top=True)
    zx.upsert_section(report, "defaults", "## Defaults\n\nsecond", at_top=True)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\n<!-- section:defaults:start -->\n## Defaults\n\nsecond\n")
    assert text.endswith("<!-- section:defaults:end -->\n\nrun 1 text\n") and "first" not in text


def test_render_default_results(tmp_path, monkeypatch):
    monkeypatch.setattr(zx, "splits_sha256", lambda path=None: "abc")
    runs = tmp_path / "runs"
    main = {"test_accuracy": 0.9999, "fpr_normal": 0.0, "fpr_wildcard": 0.0, "unseen_tool": 0.99,
            "unseen_tool/ozymandns": 1.0, "unseen_platform/AndIodine-TXT": 0.993}
    zx.save_result("old", "B-all_minus_artefact_suspect", default_result("B-all_minus_artefact_suspect", main),
                   runs_dir=runs)
    zx.save_result("old", "B-lexical_only", default_result("B-lexical_only", {**main, "unseen_tool": 0.998},
                                                           feature_set="lexical_only"), runs_dir=runs)
    zx.save_result("new", "A", default_result("A", {**main, "fpr_wildcard": 0.2}), runs_dir=runs)
    zx.save_result("folds", "lofo-iodine", default_result("lofo-iodine", {
        "held_out_family": 0.4, "fpr_normal": 0.0, "fpr_wildcard": 0.003, "held_out_family/iodine-NULL": 0.2},
        family_rows=300), runs_dir=runs)
    zx.save_result("folds", "lofo-tuns", default_result("lofo-tuns", {
        "held_out_family": 0.9, "fpr_normal": 0.0, "fpr_wildcard": 0.0, "held_out_family/tuns": 0.9},
        family_rows=100), runs_dir=runs)
    final = {
        "epochs": [50, 50], "models_dir": "models/zeek_bilstm/new/final/", "note": "the final note",
        "train_rows": {"benign": 10, "tunnel": 20}, "val_rows": {"benign": 1, "tunnel": 2},
        "train_rows_by_category": {"normal": 10, "tunnel": 20}, "captures_by_category": {"normal": 68, "tunnel": 13},
        "trained_at": "2026-09-27T00:00:00+00:00", "git": {"commit": "1234567890"},
        "settings": default_result("final", {})["settings"]}
    zx.save_training_record("new", "final", final, runs_dir=runs)
    configurations = [
        {"name": "B", "run": "old", "result": "B-all_minus_artefact_suspect", "note": "reused"},
        {"name": "A", "run": "new", "result": "A"},
        {"name": "B-lexical_only", "run": "old", "result": "B-lexical_only", "note": "reused"},
        {"name": "lofo-iodine", "run": "folds", "result": "lofo-iodine"},
        {"name": "lofo-tuns", "run": "folds", "result": "lofo-tuns"},
    ]
    (runs / "new" / zx.RUN_METADATA).write_text(json.dumps({
        "layout": "defaults", "section": "defaults", "title": "Results at the default settings",
        "description": ["intro"], "configurations": configurations, "final": {"run": "new", "result": "final"},
        "findings": ["1. a finding"]}), encoding="utf-8")
    report = tmp_path / "zeek_run.md"
    report.write_text("# Report\n\nrun 1 text\n", encoding="utf-8")
    zx.write_run_report("new", report_path=report, runs_dir=runs)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Report\n\n<!-- section:defaults:start -->\n## Results at the default settings\n\nintro")
    assert text.rstrip().endswith("run 1 text")
    assert "| sampling seed · model seed | 0 · 0 |" in text
    assert "| `B` | `old/B-all_minus_artefact_suspect` |" in text and "reused |" in text
    assert "| metric | config B | config A |" in text
    assert "| FPR held-out wildcard (all held-out captures) | 0.00% (0.00% – 0.00%) | 20.00% (20.00% – 20.00%) |" in text
    assert "| ozymandns | 100.00% (100.00% – 100.00%) | 100.00% (100.00% – 100.00%) |" in text
    assert "| AndIodine-TXT |" in text and "*Config B*" in text and "*Config A*" in text
    assert "| `all` (the default) | 28 (2) |" in text and "| `lexical_only` | 11 (2) |" in text
    assert "| **all five (row-weighted)** | 400 | **52.50%** |" in text   # (300 * 0.4 + 100 * 0.9) / 400
    assert "| iodine | iodine-NULL | 20.00% (20.00% – 20.00%) |" in text
    assert "### Final model (not for evaluation)" in text and "no results here" in text
    assert "`models/zeek_bilstm/new/final/`" in text and "| **total** | **81** | **30** |" in text
    assert "### Findings" in text and "1. a finding" in text

    # a configuration trained with other settings can't be gathered into the section
    other = default_result("A", main)
    other["settings"]["epoch_cap"] = 150
    zx.save_result("new", "A", other, runs_dir=runs)
    with pytest.raises(ValueError, match="A: epoch_cap is 150"):
        zx.write_run_report("new", report_path=report, runs_dir=runs)
    zx.save_result("new", "A", default_result("A", main, feature_set="all"), runs_dir=runs)
    with pytest.raises(ValueError, match="A: features are not the default set"):
        zx.write_run_report("new", report_path=report, runs_dir=runs)
    zx.save_result("new", "A", default_result("A", main), runs_dir=runs)
    final["settings"] = {**final["settings"], "sampling_seed": 1}
    zx.save_training_record("new", "final", final, runs_dir=runs)
    with pytest.raises(ValueError, match="final: sampling_seed is 1"):
        zx.write_run_report("new", report_path=report, runs_dir=runs)
