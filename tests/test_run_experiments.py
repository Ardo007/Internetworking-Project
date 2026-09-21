import hashlib
import json
import re
from pathlib import Path

import pytest

import dataset_splits as ds
import run_experiments as rx
import zeek_feature_extraction as zfe

# Source hashes of the model cells used for runs 1 and 2 (commit a29a0de).
UNCHANGED_CELLS = {
    "7cf0bad9-cffa-4881-ad01-5521eb858205": "7f0ece562655b21e11714f1c3dc694cd8c75b107af35755f750313c36f5456bc",  # Build_model
    "ddf24191-a7df-43b3-9363-bab3a0d0d677": "b8d6220222e2e9612732f4f022d985cb614de34a32fb58ecfaf9e6d70985becd",  # Build_experiment
}


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rx, "SANITY_DIR", tmp_path / "results" / "executed" / "sanity")
    return tmp_path


def notebook_cells():
    nb = json.loads(rx.NOTEBOOK.read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in nb["cells"]}


def test_model_cells_are_unchanged():
    cells = notebook_cells()
    for cell_id, digest in UNCHANGED_CELLS.items():
        assert hashlib.sha256(cells[cell_id].encode()).hexdigest() == digest, cell_id


def test_settings_cell_defines_every_overridden_setting():
    settings = notebook_cells()[rx.SETTINGS_CELL]
    code = rx.override_code(rx.parse_args(["--run", "r", "--lofo", "iodine", "--epochs", "50",
                                           "--feature-set", "all"]), ["lofo-iodine"])
    overridden = set(re.findall(r"^([A-Z_]+) = ", code, re.MULTILINE))
    defined = set(re.findall(r"^([A-Z_]+) = ", settings, re.MULTILINE))
    assert overridden <= defined, overridden - defined


def test_runner_constants_match_the_modules():
    assert rx.TUNNEL_FAMILIES == ds.TUNNEL_FAMILIES
    assert set(rx.FEATURE_SETS) == set(zfe.FEATURE_SETS)


def test_names_and_overrides(roots):
    args = rx.parse_args(["--run", "run3", "--lofo", "DNS-shell", "tuns", "--configs", "B",
                          "--ablations", "lexical_only", "--epochs", "50"])
    assert rx.configuration_names(args) == ["B", "B-lexical_only", "lofo-DNS-shell", "lofo-tuns"]
    namespace = {"FEATURE_SET": "all_minus_artefact_suspect", "EPOCH_CAP": 150, "print": lambda *a, **k: None}
    exec(rx.override_code(args, ["lofo-tuns", "B-lexical_only"]), namespace)
    assert namespace["RUN_NAME"] == "run3"
    assert namespace["RUN_CONFIGS"] == []
    assert namespace["RUN_ABLATION_SETS"] == ["lexical_only"]
    assert namespace["RUN_LOFO_FAMILIES"] == ["tuns"]
    assert namespace["EPOCH_CAP"] == 50 and namespace["N_OF_MODELS"] == 5 and namespace["TRAINING_VERBOSE"] == 1
    assert namespace["RENDER_REPORT"] is False
    assert namespace["MODELS_ROOT"] == roots / "models" / "zeek_bilstm"
    assert namespace["RUNS_ROOT"] == roots / "results" / "runs"
    assert namespace["FEATURE_SET"] == "all_minus_artefact_suspect"  # not overridden unless asked


def test_done_configurations_are_skipped_unless_overwrite(roots):
    done = roots / "results" / "runs" / "run3" / "lofo-dnspot.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}", encoding="utf-8")
    args = rx.parse_args(["--run", "run3", "--lofo", "dnspot", "iodine"])
    assert rx.plan(args) == (["lofo-iodine"], ["lofo-dnspot"])
    args = rx.parse_args(["--run", "run3", "--lofo", "dnspot", "iodine", "--overwrite"])
    assert rx.plan(args) == (["lofo-dnspot", "lofo-iodine"], [])


def test_sanity_defaults(roots):
    args = rx.parse_args(["--sanity"])
    assert (args.run, args.epochs, args.models, args.lofo, args.overwrite) == ("sanity", 1, 1, ["dnscat2"], True)
    models_root, runs_root = rx.output_roots(args)
    assert models_root == roots / "results" / "executed" / "sanity" / "models"
    assert runs_root == roots / "results" / "executed" / "sanity" / "runs"


@pytest.mark.parametrize("argv", [[], ["--run", "r"], ["--run", "r", "--lofo", "iodine", "--models", "0"],
                                  ["--lofo", "iodine"], ["--run", "r", "--lofo", "nope"]])
def test_bad_arguments(argv):
    with pytest.raises(SystemExit):
        rx.parse_args(argv)


def test_summary_line(tmp_path):
    path = tmp_path / "lofo-iodine.json"
    path.write_text(json.dumps({
        "name": "lofo-iodine", "epochs": [50, 48], "seconds": 600,
        "per_run": [{"collapsed": 0.0}, {"collapsed": 1.0}],
        "summary": {"held_out_family": {"avg": 0.4031}, "fpr_normal": {"avg": 0.0},
                    "fpr_wildcard": {"avg": 0.0029}}}), encoding="utf-8")
    line = rx.summary_line(path)
    assert "held_out_family 40.31%" in line and "collapsed 1/2" in line and "10.0 min" in line
