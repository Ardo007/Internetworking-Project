"""
Run the model notebook for selected configurations, from the command line
=========================================================================
Executes notebooks/dns_tunneling_bilstm_model.ipynb top to bottom in a fresh
.venv kernel, with its settings overridden so that only the requested
configurations are trained, and streams the notebook's output to the console:
Keras' per-epoch progress, then each configuration's results. Build_model and
the rest of the notebook run exactly as written.

Examples, from the repo root:

  .\\.venv\\Scripts\\python.exe notebooks\\run_experiments.py --sanity
  .\\.venv\\Scripts\\python.exe notebooks\\run_experiments.py --run run3_default_50ep --epochs 50 --lofo DNS-shell

A configuration is done once results/runs/<run>/<name>.json exists; that file
is written after the configuration's models are saved. Done configurations
are skipped, so re-running a command is safe; --overwrite retrains them and
replaces their models and results. A configuration stopped part-way leaves
no results file and is trained from scratch next time.

Each invocation writes:
  models/zeek_bilstm/<run>/<name>/          models, scaler, label encoder, features.json
  results/runs/<run>/<name>.json            metrics and per-epoch loss histories
  results/executed/<run>__<names>.ipynb     executed notebook copy (gitignored)
  results/executed/<run>__<names>.log       everything printed to the console (gitignored)
  results/executed/<run>__<names>.kernel.log   the kernel process's own warnings (gitignored)
--sanity writes the same under results/executed/sanity/ instead.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
NOTEBOOK = NOTEBOOK_DIR / "dns_tunneling_bilstm_model.ipynb"
EXECUTED_DIR = PROJECT_ROOT / "results" / "executed"
SANITY_DIR = EXECUTED_DIR / "sanity"
SETTINGS_CELL = "4d5a16f7"
TUNNEL_FAMILIES = ("DNS-shell", "dnscat2", "dnspot", "iodine", "tuns")
FEATURE_SETS = ("all_minus_artefact_suspect", "all", "lexical_only", "domain_volume_shape")
MAIN_CONFIGS = ("B", "A")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="run name: models/zeek_bilstm/<run>/, results/runs/<run>/")
    parser.add_argument("--configs", nargs="+", choices=MAIN_CONFIGS, default=[],
                        help="main configurations to train")
    parser.add_argument("--ablations", nargs="+", choices=FEATURE_SETS, default=[],
                        help="train config B with these feature sets (names B-<set>)")
    parser.add_argument("--lofo", nargs="+", choices=TUNNEL_FAMILIES, default=[],
                        help="leave-one-tunnel-family-out folds to train (names lofo-<family>)")
    parser.add_argument("--epochs", type=int, help="epoch cap passed to Build_model (default: the notebook's)")
    parser.add_argument("--models", type=int, default=5, help="models per configuration (default 5)")
    parser.add_argument("--feature-set", choices=FEATURE_SETS,
                        help="feature set of the main configurations and folds (default: the notebook's)")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1,
                        help="Keras progress: 0 silent, 1 progress bar per epoch (default), 2 one line per epoch")
    parser.add_argument("--overwrite", action="store_true",
                        help="retrain configurations that already have results, replacing them")
    parser.add_argument("--sanity", action="store_true",
                        help="quick setup check: 1 model, 1 epoch, fold dnscat2 unless --lofo is given, "
                             "written to results/executed/sanity/")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the settings, train nothing")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.sanity:
        args.run = args.run or "sanity"
        args.epochs = 1 if args.epochs is None else args.epochs
        args.models = 1
        args.overwrite = True
        if not (args.configs or args.ablations or args.lofo):
            args.lofo = ["dnscat2"]
    if not args.run:
        parser.error("--run is required (or use --sanity)")
    if not (args.configs or args.ablations or args.lofo):
        parser.error("nothing to train: give --configs, --ablations and/or --lofo")
    if args.models < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error("--models and --epochs must be at least 1")
    return args


def output_roots(args):
    """(models root, results root) for this invocation."""
    if args.sanity:
        return SANITY_DIR / "models", SANITY_DIR / "runs"
    return PROJECT_ROOT / "models" / "zeek_bilstm", PROJECT_ROOT / "results" / "runs"


def configuration_names(args):
    return (list(args.configs) + [f"B-{fs}" for fs in args.ablations]
            + [f"lofo-{family}" for family in args.lofo])


def plan(args):
    """(names to train, names skipped because their results exist)."""
    _, runs_root = output_roots(args)
    to_train, skipped = [], []
    for name in configuration_names(args):
        done = (runs_root / args.run / f"{name}.json").exists()
        (skipped if done and not args.overwrite else to_train).append(name)
    return to_train, skipped


def override_code(args, to_train):
    """Python run right after the notebook's settings cell."""
    models_root, runs_root = output_roots(args)
    lines = [
        "from pathlib import Path",
        f"RUN_NAME = {args.run!r}",
        f"RUN_CONFIGS = {[n for n in to_train if n in MAIN_CONFIGS]!r}",
        f"RUN_ABLATION_SETS = {[n.removeprefix('B-') for n in to_train if n.startswith('B-')]!r}",
        f"RUN_LOFO_FAMILIES = {[n.removeprefix('lofo-') for n in to_train if n.startswith('lofo-')]!r}",
        f"N_OF_MODELS = {args.models}",
        f"TRAINING_VERBOSE = {args.verbose}",
        f"MODELS_ROOT = Path({str(models_root)!r})",
        f"RUNS_ROOT = Path({str(runs_root)!r})",
        "RENDER_REPORT = False",
    ]
    if args.epochs is not None:
        lines.append(f"EPOCH_CAP = {args.epochs}")
    if args.feature_set:
        lines.append(f"FEATURE_SET = {args.feature_set!r}")
    lines.append('print(f"settings overridden by run_experiments.py: run {RUN_NAME}, configs {RUN_CONFIGS}, '
                 'ablations {RUN_ABLATION_SETS}, folds {RUN_LOFO_FAMILIES}, epoch cap {EPOCH_CAP}, '
                 'feature set {FEATURE_SET}, {N_OF_MODELS} models each")')
    return "\n".join(lines) + "\n"


def git_state():
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return None, None


class Console:
    """Print to the console and append to a log file."""

    def __init__(self, log_path):
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
        self.log.write(text)
        self.log.flush()

    def close(self):
        self.log.close()


def run_notebook(notebook_path, overrides, executed_path, console, kernel_log_path):
    """Execute the notebook in a fresh kernel, streaming output. Returns True if every cell ran."""
    from jupyter_client.manager import start_new_kernel

    nb = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    position = next(i for i, c in enumerate(nb["cells"]) if c.get("id") == SETTINGS_CELL) + 1
    nb["cells"].insert(position, {"cell_type": "code", "id": "run-experiments-overrides", "metadata": {},
                                  "execution_count": None, "outputs": [],
                                  "source": ("# Settings overridden by run_experiments.py\n" + overrides)
                                  .splitlines(keepends=True)})
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    kernel_log = open(kernel_log_path, "w", encoding="utf-8")
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(Path(notebook_path).parent), startup_timeout=180,
                              stdout=kernel_log, stderr=kernel_log)
    ok = True
    try:
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        for number, cell in enumerate(code_cells, start=1):
            source = "".join(cell["source"])
            first = next((line for line in source.splitlines() if line.strip()), "")[:70]
            console.write(f"\n--- cell {number}/{len(code_cells)} [{cell['id'][:8]}] {first}\n")
            outputs = []

            def hook(msg, outputs=outputs):
                kind, content = msg["msg_type"], msg["content"]
                if kind == "stream":
                    console.write(content["text"])
                    if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream", "name": content["name"], "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    data = content["data"]
                    console.write("[figure]\n" if "image/png" in data else data.get("text/plain", "") + "\n")
                    out = {"output_type": kind, "data": data, "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        out["execution_count"] = content["execution_count"]
                    outputs.append(out)
                elif kind == "error":
                    console.write("\n".join(content["traceback"]) + "\n")
                    outputs.append({"output_type": "error", "ename": content["ename"],
                                    "evalue": content["evalue"], "traceback": content["traceback"]})

            reply = kc.execute_interactive(source, timeout=None, output_hook=hook, store_history=True)
            for out in outputs:
                if out["output_type"] == "stream":
                    out["text"] = out["text"].splitlines(keepends=True)
            cell["outputs"] = outputs
            cell["execution_count"] = reply["content"].get("execution_count")
            if reply["content"]["status"] != "ok":
                console.write(f"\nCell {number} failed: {reply['content'].get('ename')}: "
                              f"{reply['content'].get('evalue')}\n")
                ok = False
                break
    finally:
        executed_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        kc.stop_channels()
        km.shutdown_kernel(now=True)
        kernel_log.close()
    return ok


def summary_line(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = result["summary"]
    wanted = (["held_out_family", "fpr_normal", "fpr_wildcard"] if result["name"].startswith("lofo-")
              else ["test_accuracy", "fpr_normal", "fpr_wildcard", "unseen_tool", "unseen_platform"])
    metrics = ", ".join(f"{m} {100 * summary[m]['avg']:.2f}%" for m in wanted if m in summary)
    collapsed = sum(int(r.get("collapsed", 0)) for r in result.get("per_run", []))
    return (f"{result['name']}: epochs {result.get('epochs')}, {metrics}, "
            f"collapsed {collapsed}/{len(result.get('per_run', []))}, {result.get('seconds', 0) / 60:.1f} min")


def main(argv=None):
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    models_root, runs_root = output_roots(args)
    to_train, skipped = plan(args)
    commit, dirty = git_state()
    print(f"run {args.run}: train {to_train or 'nothing'}"
          + (f"; skip {skipped} (results exist; --overwrite retrains)" if skipped else ""))
    print(f"models -> {models_root / args.run}\\<name>\\   results -> {runs_root / args.run}\\<name>.json")
    print(f"git {commit}" + (" (uncommitted changes: results will be recorded as dirty)" if dirty else " (clean)"))
    if args.dry_run:
        print("\nsettings that would be injected after the notebook's settings cell:\n")
        print(override_code(args, to_train))
        return 0
    if not to_train:
        print("Nothing to do.")
        return 0

    out_dir = SANITY_DIR if args.sanity else EXECUTED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.run}__{'+'.join(to_train)}"
    console = Console(out_dir / f"{stem}.log")
    started = time.time()
    try:
        ok = run_notebook(args.notebook, override_code(args, to_train), out_dir / f"{stem}.ipynb", console,
                          out_dir / f"{stem}.kernel.log")
    except KeyboardInterrupt:
        console.write("\nStopped (Ctrl+C). Configurations that hadn't finished have no results file and will be "
                      "trained from scratch next time.\n")
        console.close()
        return 130
    console.write(f"\n{'RUN OK' if ok else 'RUN FAILED'} after {(time.time() - started) / 60:.1f} min\n")
    for name in to_train:
        path = runs_root / args.run / f"{name}.json"
        console.write(("  " + summary_line(path) if path.exists() else f"  {name}: no results written") + "\n")
    console.write(f"log: {out_dir / (stem + '.log')}\n")
    console.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
