import math

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
