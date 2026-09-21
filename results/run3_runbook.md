# Run 3 runbook: leave-one-family-out folds, 28 features, 50 epochs

Run 3 retrains the five leave-one-tunnel-family-out folds with the default 28
features (`all_minus_artefact_suspect`) and run 1's epoch cap of 50. Splits,
row caps (`{'default': 200, 'wildcard': 1000}`), sampling seed 0, model seed 0,
early stopping and `Build_model` are all as in run 1, so the feature set is the
only difference from run 1's family results. With run 2 (28 features, 150
epochs) this completes the comparison:

| | 50 epochs | 150 epochs |
|---|---|---|
| 30 features | run 1 | – |
| 28 features | **run 3** | run 2 |

All commands run from the repo root in PowerShell. Each one starts a fresh
kernel, runs `notebooks/dns_tunneling_bilstm_model.ipynb` top to bottom with
its settings overridden to train only the named fold, and streams the output,
including Keras' progress for every epoch.

## Before you start

1. **Commit the prepared code.** Training records the git commit in every
   result. The runner's first line says `git <commit> (clean)` when it's
   committed, or `(uncommitted changes: results will be recorded as dirty)`.

   ```powershell
   git add notebooks/run_experiments.py notebooks/zeek_experiments.py notebooks/dns_tunneling_bilstm_model.ipynb tests/test_run_experiments.py tests/test_experiment_reports.py results/runs/run2_default_150ep/run.json results/run3_runbook.md
   git commit -m "feat: run the model notebook from the command line; prepare run 3" -m "run_experiments.py executes the notebook for selected configurations or leave-one-family-out folds with per-epoch progress; finished configurations are skipped unless --overwrite. Notebook settings (epoch cap, feature set, output folders, verbosity) are overridable, report settings move to results/runs/<run>/run.json, and result files are written atomically."
   git status
   ```

2. The feature cache (`notebooks/dataset_zeek/`) is already built. If any code
   changes, the first command rebuilds it (about 50 s).

3. Results and models of runs 1 and 2 are not touched: run 3 writes only under
   `models/zeek_bilstm/run3_default_50ep/` and `results/runs/run3_default_50ep/`.

## Step 0: sanity check (about 1–2 minutes)

```powershell
.\.venv\Scripts\python.exe notebooks\run_experiments.py --sanity
```

- **What it does:** runs the whole notebook with the dnscat2 fold (the smallest
  training set) for 1 model and 1 epoch, writing to a scratch folder.
- **Expected output:**
  1. The header line, then `--- cell k/21 ...` as each notebook cell starts, and
     `Loaded 1,914,892 rows from ...features.csv`.
  2. The split tables, then
     `=== lofo-dnscat2: training 1 models one after another ...`.
  3. `Epoch 1/1` and a progress bar filling to `1546/1546`, ending in a line
     like `... accuracy: 0.99xx - loss: 0.02xx - val_accuracy: 0.9999 - val_loss: 0.008x - learning_rate: 0.0010`.
  4. `lofo-dnscat2: 0.x min, epochs [1] ...`, a small results table,
     `RUN OK after ~1-2 min`, and a summary line ending in `collapsed 0/1`.
- **It worked if:** it ends with `RUN OK`, `val_loss` is roughly 0.005–0.02,
  `val_accuracy` is at least 0.99, and the header says `(clean)`.
- **Files:** everything goes under `results/executed/sanity/` (gitignored, safe
  to delete): `models/sanity/lofo-dnscat2/`, `runs/sanity/lofo-dnscat2.json`
  and the `sanity__lofo-dnscat2` `.ipynb`, `.log` and `.kernel.log` files.
  Nothing is written to `models/zeek_bilstm/` or `results/runs/`. Re-running
  Step 0 always overwrites it.

## Steps 1–5: one fold each

| step | command |
|---|---|
| 1 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run3_default_50ep --epochs 50 --lofo DNS-shell` |
| 2 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run3_default_50ep --epochs 50 --lofo dnscat2` |
| 3 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run3_default_50ep --epochs 50 --lofo dnspot` |
| 4 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run3_default_50ep --epochs 50 --lofo iodine` |
| 5 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run3_default_50ep --epochs 50 --lofo tuns` |

**What each step does.** It trains 5 models on config B's benign data plus the
other four tunnel families. It then scores them on every row of the held-out
family's captures (its recall) and on the held-out normal and wildcard
captures (false positive rates).

| step | fold | training rows (benign / tunnel) | steps per epoch | run-1 time (5 models) | run-1 epochs |
|---|---|---|---|---|---|
| 1 | DNS-shell | 132,514 / 93,682 | 1,768 | 39.5 min | 50, 50, 50, 50, 50 |
| 2 | dnscat2 | 132,514 / 65,329 | 1,546 | 26.4 min | 29, 30, 50, 50, 28 |
| 3 | dnspot | 132,514 / 91,002 | 1,747 | 39.2 min | 50, 50, 50, 50, 50 |
| 4 | iodine | 132,514 / 74,112 | 1,615 | 36.5 min | 50, 50, 50, 50, 50 |
| 5 | tuns | 132,514 / 102,595 | 1,837 | 41.9 min | 50, 50, 50, 50, 50 |

**How long.** About 30–45 minutes per step when it has the machine to itself
(run-1 times plus a minute to start), so about 3 hours for all five in a row.
You can run several steps at once in separate PowerShell windows; each is
then slower (roughly 1.5–2×), but the total is shorter.

**Progress output.**
1. The header and notebook cells, as in Step 0.
2. `=== lofo-<family>: training 5 models one after another (each starts again at epoch 1), epoch cap 50, ...`.
3. For each model in turn, `Epoch 1/50` … `Epoch 50/50`, each with a progress
   bar and a closing line of `accuracy`, `loss`, `val_accuracy`, `val_loss`
   and `learning_rate`. Nothing names the model: a new `Epoch 1/50` means the
   next of the 5 has started. A model can end before 50 if early stopping
   fires; run 1's dnscat2 fold did that at around epoch 30.
4. After the fifth model, about 10–20 s of scoring, then
   `lofo-<family>: NN min, epochs [...]`, a table with the fold's
   `held_out_family`, `fpr_normal`, `fpr_wildcard` and `collapsed`,
   `RUN OK`, and a summary line.

**Files each step writes.**
- `models/zeek_bilstm/run3_default_50ep/lofo-<family>/`: `model_1.keras` …
  `model_5.keras`, `scaler.joblib`, `label_encoder.joblib`, `features.json`
  (gitignored).
- `results/runs/run3_default_50ep/lofo-<family>.json`: metrics and per-epoch
  loss histories. It is written last, so it marks the fold as done.
- `results/executed/run3_default_50ep__lofo-<family>.ipynb`, `.log` (the
  console output) and `.kernel.log` (gitignored).

**Independent and resumable.**
- **Independent:** every step is its own process, reseeds before training and
  only reads the feature cache, so the steps can run in any order or at the
  same time.
- **Re-running a finished step skips it.** You'll see
  `skip ['lofo-<family>'] (results exist; --overwrite retrains)`. Add
  `--overwrite` to retrain it; that replaces its model files and results file.
- **Stopping a step part-way** (Ctrl+C or closing the window) leaves no results
  file for that fold, so it doesn't count. Running the same command again
  trains it from scratch and overwrites its model files. Finished folds are
  unaffected.
- **Stopping after, say, two folds** leaves two complete, valid results.

## What healthy and bad look like

**Healthy.** These are run 2's median values at the same epochs; it had the same
features, data and seeds, so run 3 should look much the same, though not
identical, because TensorFlow on CPU isn't bit-for-bit reproducible.

| fold | val_loss epoch 1 | epoch 5 | epoch 10 | epoch 25 | epoch 50 | val_accuracy |
|---|---|---|---|---|---|---|
| DNS-shell | 7.2e-03 | 1.3e-03 | 7.1e-04 | 3.5e-04 | 3.1e-04 | 1.0000 from epoch 1 |
| dnscat2 | 8.4e-03 | 2.0e-03 | 1.2e-03 | 5.5e-04 | 4.9e-04 | 0.9999–1.0000 |
| dnspot | 7.6e-03 | 1.7e-03 | 1.4e-03 | 4.2e-04 | 3.4e-04 | 1.0000 from epoch 1 |
| iodine | 7.1e-03 | 1.2e-03 | 6.4e-04 | 2.3e-04 | 1.9e-04 | 1.0000 from epoch 1 |
| tuns | 7.2e-03 | 1.6e-03 | 9.3e-04 | 3.9e-04 | 3.3e-04 | 1.0000 from epoch 1 |

Training `loss` starts around 0.02 and stays above `val_loss`, because dropout,
L2 and class weights only apply during training. `learning_rate` falls from
0.0010 in halving steps down to 1e-05.

**Bad: stop (Ctrl+C) and tell me.**
- **A collapsed model:** `val_accuracy` frozen at the fold's one-class share
  for several epochs, usually with `val_loss` stuck near 0.69 and `accuracy`
  near 0.5. The summary line afterwards shows `collapsed 1/5` or more.

  | fold | val_accuracy if everything is called benign | … if everything is called tunnel |
  |---|---|---|
  | DNS-shell | 0.484 | 0.516 |
  | dnscat2 | 0.545 | 0.455 |
  | dnspot | 0.489 | 0.511 |
  | iodine | 0.558 | 0.442 |
  | tuns | 0.449 | 0.551 |

- **Loss not moving:** `val_loss` still above about 5e-03 at epoch 5, climbing
  for several epochs in a row, or `nan`.
- **An error:** a traceback followed by `RUN FAILED`. The full output is in
  the `.log` file named on the last line.

**Not a failure.** The held-out family's recall is only computed at the end;
it can't be seen during training. Validation contains only the families
being trained on, which is why `val_accuracy` sits at about 1.0000 from the
first epoch. A low recall at the end is a result (run 1: DNS-shell 6.47%,
dnspot 4.10%), not a broken run.

## When all five are done

`Get-ChildItem results\runs\run3_default_50ep` should list five
`lofo-*.json` files. Tell me, and I'll do the rest: read the results, write
run 3's section of `results/zeek_run.md` with the comparison against runs 1
and 2, update the docs, and tell you what to commit.
