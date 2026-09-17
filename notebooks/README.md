# DNS Tunnelling Detection Model (BiLSTM + Multi-Head Attention)

Deep-learning component of the risk-scoring framework: a binary (`benign`/`tunnel`)
classifier trained on the [DNS-Tunnel-Datasets](https://github.com/ggyggy666/DNS-Tunnel-Datasets)
PCAP corpus (the same source referred to as "GraphTunnel" elsewhere in this repo).

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

Then open `dns_tunneling_bilstm_model.ipynb` and run top to bottom. Part 1 will
auto-clone the ~845MB PCAP dataset on first run (skipped if `dataset/*.csv`
already exist from a previous run) and extract features into `dataset/`;
both are gitignored, so every collaborator regenerates them locally rather
than pulling large/derived files through git.

## Note for the team: two parallel feature-extraction paths

This notebook's `dns_feature_extraction.py` parses PCAPs directly (hand-rolled
struct-level DNS parsing, chosen because `dpkt`'s strict-UTF8 decoder silently
dropped up to 97% of messages on some tunnelling tools — see the notebook's
own "Notes, caveats" section for detail). That's a **different path** from
`scripts/process_pcaps.py` + `run_zeek.py`, which extracts features via Zeek's
`dns.log`/`conn.log`. Both currently produce their own feature schema from the
same underlying PCAP source.

Worth a team discussion before going further: keep both approaches (e.g. as
an ML-model-specific pipeline vs. the live-capture/Zeek pipeline), or converge
on one shared feature-extraction path so `src/evaluation/metrics.py` and this
model's evaluation are computed from the exact same feature definitions.

## Current status

Iteratively debugged against a genuinely held-out set (unseen tunnelling
tools + benign traffic that's structurally tunnel-like) rather than trusting
in-distribution accuracy alone:

| Metric | Initial model | Current |
|---|---|---|
| Held-out accuracy | 59.4% | 99.8% |
| False-positive rate (benign traffic that looks tunnel-like) | 88.7% | 0% |
| Recall on unseen tunnelling tools | 99.3-100% | 99.3-100% |
| Training stability (collapsed runs, out of 10) | n/a | 0/10 |

See the notebook's markdown cells for the full diagnosis of each failure mode
and why each fix (hard-negative mining, capacity/regularization, hard-positive
mining) was needed.

## Next steps

- Scale extraction to the full dataset (currently a curated ~10% subset)
- Reconcile with the Zeek-based feature path (see note above)
- Add the rule-based and classical-ML methods called for in the project brief
  -- this notebook currently covers only the deep-learning piece
