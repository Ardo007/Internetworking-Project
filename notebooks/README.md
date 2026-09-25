# DNS Tunnelling Detection Model (BiLSTM + Multi-Head Attention)

Deep-learning component of the risk-scoring framework: a binary (`benign`/`tunnel`)
classifier trained on the [DNS-Tunnel-Datasets](https://github.com/ggyggy666/DNS-Tunnel-Datasets)
corpus (the same source referred to as "GraphTunnel" elsewhere in this repo).

Features come from Zeek `dns.log` files, the same logs the live capture pipeline
writes, so the trained model can score live traffic later. `DATA_PIPELINE.md`
describes the whole flow, from downloading the corpus to the saved models.

## Setup

TensorFlow 2.21 supports Python 3.13 but not 3.14, so the project uses
Python 3.13 in a virtual environment at the repo root. On Windows, install it
with the [Python Install Manager](https://docs.python.org/3/using/windows.html),
then run these from the repo root:

```text
py install 3.13
py -V:3.13 -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate on Linux/Mac
python -m pip install -r notebooks/requirements.txt
```

Open the notebook in VS Code or Jupyter and select the `.venv` interpreter
(Python 3.13) as the kernel. Run the tests from the repo root with
`.\.venv\Scripts\python.exe -m pytest`.

VS Code's Pylance may underline `tensorflow.keras` imports. That is an editor
warning only; the imports resolve to Keras 3 at runtime.

## Running it

The Zeek logs and the manifest have to exist first (see `DATA_PIPELINE.md`):

```text
python Ardashes_scripts/process_pcaps.py        # Zeek logs + capture_manifest.csv
python Ardashes_scripts/reorder_and_rezeek.py   # time-sort out-of-order pcaps, re-run Zeek
```

Then train, in either of two ways:

- **In the notebook:** run `dns_tunneling_bilstm_model.ipynb` top to bottom.
  It trains what its settings cell lists (`RUN_NAME`, `RUN_CONFIGS`,
  `RUN_ABLATION_SETS`, `RUN_LOFO_FAMILIES`, `RUN_FINAL`, `FEATURE_SET`,
  `EPOCH_CAP`, `N_OF_MODELS`).
- **From the command line, one configuration or fold per command**, with
  Keras' progress for every epoch:

  ```text
  .\.venv\Scripts\python.exe notebooks\run_experiments.py --sanity                     # 1-2 min setup check
  .\.venv\Scripts\python.exe notebooks\run_experiments.py --run <name> --epochs 50 --configs B
  .\.venv\Scripts\python.exe notebooks\run_experiments.py --run <name> --epochs 50 --lofo iodine
  .\.venv\Scripts\python.exe notebooks\run_experiments.py --run <name> --epochs 50 --final
  ```

  This runs the same notebook with its settings overridden. A configuration
  whose results already exist is skipped (`--overwrite` retrains it), so an
  interrupted series can simply be re-run. `results/run3_runbook.md` and
  `results/run4_runbook.md` are worked examples.

On first use the feature table is built (about 50 s for 1.9 M records) and
cached in `dataset_zeek/`. It is rebuilt only when the Zeek logs, `splits.csv`,
the caps/seed or the extraction code change. One configuration (5 models, up to
50 epochs) takes about 30–45 minutes on CPU.

## What is in this folder

| file | what it does |
|---|---|
| `zeek_feature_extraction.py` | Zeek `dns.log` → one row per DNS record, with per-domain 60 s window aggregates. Also groups live chunk folders into capture sessions. |
| `dataset_splits.py` | the project's single split rule; writes `Data/processed/GraphTunnel/splits.csv` and the per-window row sampling |
| `model_artifacts.py` | one-hot encoding with fixed levels, and saving/loading models with `features.json` |
| `zeek_experiments.py` | feature cache, held-out metrics, per-run results and the `results/zeek_run.md` sections |
| `run_experiments.py` | runs the notebook from the command line for selected configurations or folds |
| `dns_feature_extraction.py` | the older PCAP parser. Kept: the Zeek path imports its lexical features and base-domain rule, so both compute them identically. |
| `dns_tunneling_bilstm_model.ipynb` | the model: data prep, `Build_model` / `Build_experiment`, evaluation, ablations, cross-validation, saving |

## Runs, results and models

Every training run has a name. For each configuration it trained there is:

- `models/zeek_bilstm/<run>/<name>/` with `model_1.keras` … `model_5.keras`,
  `scaler.joblib`, `label_encoder.joblib` and `features.json` (feature order,
  one-hot input columns, window length, row caps, sampling seed, training date,
  git commit and library versions). Gitignored: regenerate by re-running the
  run's configurations (notebook or `run_experiments.py` with the same
  `--run`, `--epochs` and `--feature-set`). Load one with
  `model_artifacts.load_artifacts(...)`.
- `results/runs/<run>/<name>.json`: the metrics and per-epoch loss histories
  (committed). `results/runs/<run>/run.json` holds the run's report settings,
  and `zeek_experiments.write_run_report("<run>")` writes the run's section of
  `results/zeek_run.md`.

| run | features | epoch cap | trained | models |
|---|---|---|---|---|
| `run1_all_50ep` | 30 (`all`) | 50 | B, A, three ablations, five folds | `models/zeek_bilstm/<name>/` |
| `run2_default_150ep` | 28 (default) | 150 | B, five folds | `models/zeek_bilstm/run2_default_150ep/<name>/` |
| `run3_default_50ep` | 28 (default) | 50 | five folds | `models/zeek_bilstm/run3_default_50ep/<name>/` |
| `run4_default_50ep` | 28 (default) | 50 | A, the domain volume/shape ablation, the final model | `models/zeek_bilstm/run4_default_50ep/<name>/` |

The results at the default settings (28 features, epoch cap 50) come from
three runs: config B and the `lexical_only` ablation from run 1 (trained with
the same features, rows and settings), the folds from run 3, and the rest from
run 4. `results/runs/run4_default_50ep/run.json` lists them, and
`write_run_report("run4_default_50ep")` writes them as one section at the top
of `results/zeek_run.md`, refusing any result with other settings.

**The final model** (`models/zeek_bilstm/run4_default_50ep/final/`) is trained
with the default settings on every capture, including the unseen tools and
platform, so it is **not for evaluation**: nothing in GraphTunnel is left that
it hasn't seen. Its `features.json` has `"not_for_evaluation": true`, the folder
holds `NOT_FOR_EVALUATION.txt`, the scoring functions refuse it, and
`results/runs/run4_default_50ep/final.json` has only its training rows and
loss histories. It is the model to score live traffic with; config B is its
evaluated counterpart.

## Current status

The defaults are the 28-feature set `all_minus_artefact_suspect`, an epoch cap
of 50 and config B (wildcard hard negatives in training). With those settings,
average over 5 models (the full tables are in the section at the top of
`results/zeek_run.md`):

| metric | config B | config A |
|---|---|---|
| In-distribution test accuracy | 99.99% | 99.99% |
| False positive rate, held-out normal | 0.00% | 0.00% |
| False positive rate, held-out wildcard (00007–00012) | 0.00% | 51.63% |
| Recall, unseen tools (unknownTunnel) | 99.18% (ozymandns 100%, cobalstrike 88.1%) | 99.96% |
| Recall, unseen platform (iodine on Android) | 99.60% | 99.49% |
| Recall, unseen tunnel families (leave-one-family-out, all rows) | 56.16% | – |

What the runs show (details in `results/zeek_run.md`):

- **Wildcard hard negatives are what keep false positives at zero.** Without
  them (config A) the default model calls 52–56% of held-out wildcard traffic
  tunnelling. Run 1's 30-feature config A flagged 14–21%, but only because of
  `domain_qtype_diversity` (see below).
- **Neither feature group alone matches the default.** Lexical features alone
  find slightly more unseen tools but flag 0.51% of held-out wildcard. The
  domain volume/shape features alone miss a quarter of dns2tcp-key.
- **The feature set is a trade-off, decided by one feature.** Dropping
  `domain_qtype_diversity` and `no_response_ratio` raised unseen-tool recall
  from 97.1% to 99.2% (ozymandns 19.9% → 100%) but lowered unseen-family recall
  from 62.9% to 56.2% (mostly iodine's NULL and private captures). Run 3 traced
  both effects to `domain_qtype_diversity`. In GraphTunnel it only marks the
  wildcard captures, and it lets any tunnel that mixes query types pass as
  benign, as ozymandns does, so the default leaves it out. The 62.9% is also
  optimistic: the 30-feature models' verdicts follow that query-type count
  almost completely, a pattern real clients (which ask A, AAAA and HTTPS for
  ordinary names) don't share, so GraphTunnel can't show what it would cost
  on live traffic.
- **Training longer doesn't help.** 150 epochs lowered the validation loss by
  about 30% but didn't improve held-out results, so the cap stays at 50.
  Validation data only contains known tools, so it can't show when a model
  generalises to new ones.
- **Unseen tunnel families are the weak spot.** Holding a family out of
  training, dnscat2 and tuns are still found (98–99%), iodine only partly (42%)
  and DNS-shell and dnspot hardly at all (under 1%).
- **Sparse windows are the other one.** In windows with fewer than 10 queries,
  unseen-tool recall is about 20%, against over 99% in windows of 100 or more.

## Next steps

- Score live traffic with the final model (`datas/zeek/<chunk>/` → the same
  extractor → `models/zeek_bilstm/run4_default_50ep/final/`).
- Improve generalisation to unseen families with data rather than features:
  more tunnel families and record types, and realistic benign traffic from the
  live pipeline (`own_benign`). A validation set that holds out a family too
  would let model selection see generalisation.
- Add the rule-based and classical-ML methods the project brief calls for;
  this notebook covers only the deep-learning part.
