# Run 4 runbook: config A, the domain volume/shape ablation and the final model

Run 4 trains what the default settings still lack. The defaults are the 28-feature
set `all_minus_artefact_suspect`, an epoch cap of 50, row caps
`{'default': 200, 'wildcard': 1000}`, sampling seed 0, model seed 0 before each
configuration, and early stopping and `Build_model` as in runs 1–3.

| step | trains | why run 1's version can't be reused |
|---|---|---|
| 1 | config A (stress test), 5 models | run 1's A had all 30 features |
| 2 | config B with `domain_volume_shape_minus_artefact_suspect` (10 features), 5 models | run 1's `domain_volume_shape` (11) included `domain_qtype_diversity`, which isn't in the default |
| 3 | the **final model** on all 105 captures, 5 models, **not for evaluation** | new |

**Reused, not retrained.** With `all` now meaning the 28-feature default, three
results already exist at these settings:

- **Config B, which is also the `all` row of the ablation table**, is run 1's
  `B-all_minus_artefact_suspect`.
- **The `lexical_only` ablation** is run 1's `B-lexical_only`. The lexical group
  never held either artefact-suspect feature.
- **The five leave-one-family-out folds** are run 3's.

Both run 1 models have the same features, training and validation rows,
hyperparameters and epoch cap as a new run would. Their saved scalers match
scalers fitted on today's training rows exactly, which confirms the rows are
the same. The one difference: run 1 set the random seed once per notebook run,
not before each configuration, so their starting weights differ from what a
fresh run would draw. That is the same kind of difference as between the five
models of one configuration. If you'd rather have them reseeded like everything
else, add a step with `--configs B --ablations lexical_only` (about 65 min), and
I'll point the results section at those instead.

All commands run from the repo root in PowerShell. As in run 3, each one starts
a fresh kernel, runs `notebooks/dns_tunneling_bilstm_model.ipynb` top to bottom
with its settings overridden to train only what the step names, and streams
the output, including Keras' progress for every epoch.

## Before you start

1. **Commit the prepared code.** Training records the git commit in every
   result. The runner's first line says `git <commit> (clean)` when everything
   is committed, or `(uncommitted changes: results will be recorded as dirty)`.

   ```powershell
   git add notebooks/dataset_splits.py notebooks/dns_tunneling_bilstm_model.ipynb notebooks/model_artifacts.py notebooks/run_experiments.py notebooks/zeek_experiments.py notebooks/zeek_feature_extraction.py notebooks/README.md DATA_PIPELINE.md tests/test_experiment_reports.py tests/test_inputs_and_metrics.py tests/test_run_experiments.py tests/test_splits.py tests/test_zeek_feature_extraction.py results/runs/run3_default_50ep/run.json results/runs/run3_default_50ep/diagnostics/query_type_count.json results/zeek_run.md results/runs/run4_default_50ep/run.json results/run4_runbook.md
   git commit -m "feat: prepare run 4 (config A, 10-feature ablation, final model)" -m "The final model trains on every capture at the default settings and is marked not for evaluation (features.json, NOT_FOR_EVALUATION.txt, a training record without metrics; scoring functions refuse it). The results section at the default settings is rendered from the runs that trained each configuration and refuses results with other settings. Run 3's write-up gains the query-type probe of run 1's 30-feature models and the argument that the 62.88% is optimistic."
   git status
   ```

2. The feature cache (`notebooks/dataset_zeek/`) is already built for this code.
   If any code changes, the first command rebuilds it (about 50 s).

3. Runs 1–3 are not touched. Run 4 writes only under
   `models/zeek_bilstm/run4_default_50ep/` and `results/runs/run4_default_50ep/`,
   which already holds `run.json`, the report settings for the results section.

## Step 0: sanity check (about 3 minutes)

```powershell
.\.venv\Scripts\python.exe notebooks\run_experiments.py --sanity --configs A --ablations domain_volume_shape_minus_artefact_suspect --final
```

- **What it does:** runs the whole notebook with 1 model and 1 epoch for each
  of the three, writing to a scratch folder. I ran it once already. Rerun it
  after committing, as a check on your machine and on the committed code.
- **Expected output:**
  1. `run sanity: train ['A', 'B-domain_volume_shape_minus_artefact_suspect', 'final']`,
     the `git` line, then `--- cell k/22 ...` as each notebook cell starts and
     `Loaded 1,914,892 rows from ...features.csv`.
  2. `=== A: training 1 models ...` (41 input columns), then `Epoch 1/1` and a
     progress bar to `1700/1700` ending in
     `accuracy: 0.9964 - loss: 0.0199 - val_accuracy: 1.0000 - val_loss: 0.0073 - learning_rate: 0.0010`.
  3. `=== B-domain_volume_shape_minus_artefact_suspect: ...` (10 input columns),
     `1869/1869`, ending in
     `accuracy: 0.9907 - loss: 0.0311 - val_accuracy: 0.9975 - val_loss: 0.0240 - learning_rate: 0.0010`.
  4. `=== final: training 1 models ... on every capture (105 captures: {'crossEndPoint': 5, 'normal': 68, 'tunnel': 13, 'unknownTunnel': 6, 'wildcard': 13}), NOT FOR EVALUATION, ...`
     with `train rows {'tunnel': 336852, 'benign': 202628}`, then `4215/4215`
     ending in
     `accuracy: 0.9964 - loss: 0.0190 - val_accuracy: 0.9999 - val_loss: 0.0046 - learning_rate: 0.0010`
     and `final: ... (not for evaluation)`.
  5. `RUN OK after ~3 min` and three summary lines. A's shows `fpr_wildcard`
     near 89%: config A has no wildcard in training, and after one epoch it
     calls most wildcard traffic tunnelling. That isn't a problem; run 1's A
     reached 20.68% after 50 epochs. The last line is
     `final: epochs [1], trained on 105 captures, 202,628 benign / 336,852 tunnel rows, not for evaluation, ...`.
- **It worked if:** it ends with `RUN OK`, the header says `(clean)`, and the
  three `val_loss` values are 0.0073, 0.0240 and 0.0046. Training is
  deterministic on this machine (run 3 reproduced run 2's validation losses
  exactly), so they should match to the last digit shown.
- **Files:** everything goes under `results/executed/sanity/` (gitignored, safe
  to delete), including `models/sanity/final/NOT_FOR_EVALUATION.txt`. Nothing is
  written to `models/zeek_bilstm/` or `results/runs/`.

## Steps 1–3: one per command

| step | command |
|---|---|
| 1 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run4_default_50ep --epochs 50 --configs A` |
| 2 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run4_default_50ep --epochs 50 --ablations domain_volume_shape_minus_artefact_suspect` |
| 3 | `.\.venv\Scripts\python.exe notebooks\run_experiments.py --run run4_default_50ep --epochs 50 --final` |

| step | trains | training rows (benign / tunnel) | validation rows (benign / tunnel) | steps per epoch | time alone (5 models) |
|---|---|---|---|---|---|
| 1 | `A` | 110,904 / 106,680 | 12,016 / 22,621 | 1,700 | about 40 min (run 1's A: 38 min) |
| 2 | `B-domain_volume_shape_minus_artefact_suspect` | 132,514 / 106,680 | 18,232 / 22,621 | 1,869 | 35–45 min (run 1's 11-feature version: 36 min) |
| 3 | `final` | 202,628 / 336,852, from all 105 captures | 18,232 / 22,621, which are training rows too | 4,215 | about 1.5 h |

**What each step does.**

- **Step 1** trains config A: no wildcard in training or validation, all 13
  wildcard captures held out. It then scores every test and held-out row.
- **Step 2** trains config B with only the 10 per-domain volume and shape
  features, then scores the same rows as config B.
- **Step 3** trains on the capped rows of every window of every capture:
  normal, wildcard, the 13 tunnel captures, the 6 unseen-tool captures and the
  5 unseen-platform captures. Early stopping and the learning-rate schedule
  still monitor config B's validation rows, but those are training rows here
  too. Nothing is scored.

**How long.** About 3 hours one after another. Step 3 takes most of that, so
running the steps side by side in separate windows (each then 1.5–2× slower)
saves little.

**Progress output.** As in run 3:
1. The header and notebook cells.
2. A `=== <name>: training 5 models one after another ...` line.
3. `Epoch 1/50` … `Epoch 50/50` for each model; a new `Epoch 1/50` means the
   next model has started.
4. After the fifth model:
   - **Steps 1 and 2:** scoring (10–30 s), `<name>: NN min, epochs [...]`,
     the results tables, `RUN OK` and a summary line.
   - **Step 3:** no scoring and no results table, just
     `final: NN min, epochs [...], saved to ...\final (not for evaluation)`,
     `RUN OK`, and
     `final: epochs [...], trained on 105 captures, 202,628 benign / 336,852 tunnel rows, not for evaluation, NN min`.

**Files each step writes.**
- `models/zeek_bilstm/run4_default_50ep/<name>/`: `model_1.keras` …
  `model_5.keras`, `scaler.joblib`, `label_encoder.joblib`, `features.json`
  (gitignored). For `final`, `features.json` also has
  `"not_for_evaluation": true`, and the folder holds `NOT_FOR_EVALUATION.txt`.
- `results/runs/run4_default_50ep/<name>.json`, written last, so it marks the
  step as done. For A and the ablation it holds metrics and per-epoch loss
  histories. `final.json` holds only training rows, settings and loss
  histories, with `"not_for_evaluation": true`.
- `results/executed/run4_default_50ep__<name>.ipynb`, `.log` and
  `.kernel.log` (gitignored).

**Independent and resumable**, as in run 3:
- **Independent:** the steps can run in any order or at the same time.
- **A finished step is skipped when re-run.** `--overwrite` retrains it.
- **A step stopped part-way** leaves no results file. Running the same command
  again trains it from scratch.

## What healthy and bad look like

**Healthy.**
- **Model 1's first epoch** ends exactly as in Step 0: `val_loss` 0.0073 (A),
  0.0240 (the ablation) and 0.0046 (final).
- **A and final** should look like run 2's config B, which had the same
  features and seed. Its median `val_loss` was 7.3e-03 at epoch 1, 1.8e-03 at
  epoch 5, 1.2e-03 at epoch 10, 4.3e-04 at epoch 25 and 3.7e-04 at epoch 50,
  with `val_accuracy` 1.0000 from epoch 1. A's validation rows have no
  wildcard, and the final model's are also training rows, so both may end a
  little lower.
- **The final model** will most likely run all 50 epochs in every model:
  its validation loss keeps falling because the model trains on those rows.
- **The ablation** starts higher (0.0240, `val_accuracy` about 0.997–0.999)
  and may stop early. Run 1's 11-feature version stopped at epochs 35 and 27
  in two of its five models.
- In every step, training `loss` stays above `val_loss`, and
  `learning_rate` halves from 0.0010 down to 1e-05.

**Bad: stop (Ctrl+C) and tell me.**
- **A collapsed model:** `val_accuracy` frozen at the one-class share for
  several epochs, usually with `val_loss` near 0.69 and `accuracy` near 0.5.
  The summary line afterwards shows `collapsed 1/5` or more (steps 1 and 2).

  | step | val_accuracy if everything is called benign | … if everything is called tunnel |
  |---|---|---|
  | 1 (A) | 0.347 | 0.653 |
  | 2 (ablation) | 0.446 | 0.554 |
  | 3 (final) | 0.446 | 0.554 |

- **Loss not moving:** `val_loss` no lower at epoch 5 than at epoch 1,
  climbing for several epochs in a row, or `nan`.
- **An error:** a traceback followed by `RUN FAILED`. The full output is in
  the `.log` file named on the last line.

**Not a failure.**
- **Config A's false positives on held-out wildcard** are its point as a
  stress test (run 1's A: 20.68%).
- **The final model has no results table:** nothing is left out to score it on.

## When all three are done

`Get-ChildItem results\runs\run4_default_50ep` should list `A.json`,
`B-domain_volume_shape_minus_artefact_suspect.json`, `final.json` and `run.json`.
Tell me, and I'll do the rest:
- read the results;
- write the results section at the top of `results/zeek_run.md` (the default
  settings stated once; configs A and B; per-tool and per-capture held-out
  recall; the window-size breakdown; the ablations; the leave-one-family-out
  table; and the final model). The section refuses any result that wasn't
  trained at the default settings;
- update the docs and tell you what to commit.
