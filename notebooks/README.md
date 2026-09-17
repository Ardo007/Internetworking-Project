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

Then run `dns_tunneling_bilstm_model.ipynb` top to bottom. On first run it
builds the feature table (about 45 s for 1.9 M records) and caches it in
`dataset_zeek/`; afterwards it reloads the cache and rebuilds only when the
Zeek logs, `splits.csv`, the caps/seed or the extraction code change. Training
all ten configurations takes several hours on CPU; reduce `N_OF_MODELS` for a
quicker pass.

## What is in this folder

| file | what it does |
|---|---|
| `zeek_feature_extraction.py` | Zeek `dns.log` → one row per DNS record, with per-domain 60 s window aggregates. Also groups live chunk folders into capture sessions. |
| `dataset_splits.py` | the project's single split rule; writes `Data/processed/GraphTunnel/splits.csv` and the per-window row sampling |
| `model_artifacts.py` | one-hot encoding with fixed levels, and saving/loading models with `features.json` |
| `zeek_experiments.py` | feature cache, held-out metrics and the `results/zeek_run.md` report |
| `dns_feature_extraction.py` | the older PCAP parser. Kept: the Zeek path imports its lexical features and base-domain rule, so both compute them identically. |
| `dns_tunneling_bilstm_model.ipynb` | the model: data prep, `Build_model` / `Build_experiment`, evaluation, ablations, cross-validation, saving |

## Regenerating the models

Running the notebook writes one folder per configuration under
`models/zeek_bilstm/` (gitignored, so everyone regenerates them locally):

```text
models/zeek_bilstm/B/          primary configuration (wildcard hard negatives)
models/zeek_bilstm/A/          stress test (no wildcard in training)
models/zeek_bilstm/B-<set>/    feature-set ablations
models/zeek_bilstm/lofo-<family>/   leave-one-family-out folds
```

Each folder holds `model_1.keras` … `model_5.keras`, `scaler.joblib`,
`label_encoder.joblib` and `features.json` (feature order, one-hot input
columns, window length, row caps, sampling seed, training date, git commit and
library versions). Load one with
`model_artifacts.load_artifacts("models/zeek_bilstm/B")`.

## Current status

Metrics of the latest run are in `results/zeek_run.md` (5 runs per
configuration; average with min–max across runs). Config B is the primary one:

| metric | config B | config A (no wildcard in training) |
|---|---|---|
| In-distribution test accuracy | 99.99% | 99.99% |
| False positive rate, held-out normal | 0.01% | 0.01% |
| False positive rate, held-out wildcard (00007–00012) | 0.00% | 14.16% (2.13–28.89) |
| Recall, unseen tools (unknownTunnel) | 97.09% | 97.10% |
| Recall, unseen platform (iodine on Android) | 99.39% | 99.38% |
| Collapsed runs | 0/5 | 0/5 |

What the run says, beyond the headline numbers:

- **Wildcard hard negatives are what keeps false positives near zero.** Without
  them (config A) 14–21% of held-out wildcard traffic is called tunnelling.
- **Per-tool recall is uneven.** ozymandns is missed almost entirely (19.9%)
  and cobalstrike partly (92.5%), while the other four unseen tools are at
  99.98–100%.
- **Generalisation to unseen tunnelling *families* is much weaker than to
  unseen tools of a known family.** Leave-one-family-out recall is 99.9% for
  dnscat2 and tuns but 6.5% for DNS-shell, 4.1% for dnspot and 58.2% for
  iodine, at an unchanged false positive rate. The in-distribution numbers
  alone would hide this.
- **Dropping the two artefact-suspect features helps.** Removing
  `domain_qtype_diversity` and `no_response_ratio` raises unseen-tool recall
  from 97.1% to 99.2% while keeping both false positive rates at 0.00–0.01%.
- **Sparse windows are the weak spot.** In windows with fewer than 10 queries,
  unseen-tool recall is 27.5%, against 97.3% in windows of 100 or more.

## Next steps

- Score live traffic with a saved model (`datas/zeek/<chunk>/` → the same
  extractor → `models/zeek_bilstm/B/`).
- Treat the "unseen family" numbers as the honest generalisation estimate, and
  look at the features the weak families share.
- Add the rule-based and classical-ML methods the project brief calls for;
  this notebook covers only the deep-learning part.
