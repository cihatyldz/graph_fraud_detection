# SAFE-G: split-aware evaluation of graph-derived features for fraud detection

Reproducibility package for the manuscript *"SAFE-G: Split-aware evaluation of graph-derived features for fraud
detection, and why entity coverage decides whether they help"* (C. Yıldız, H. Çoban Eroğlu, Doğuş University).

The notebooks are provided **with their outputs**: every number in the manuscript can be traced to a printed
cell output, and the scripts in `paper/` regenerate all manuscript tables and Figure 1 from those outputs.

## Contents

| Path | Purpose | Manuscript |
|---|---|---|
| `notebooks/ieee_cis_pipeline_v11_D1.ipynb` | IEEE-CIS: audit, falsification test, P1/P2/P3 grids (10 models), strong baseline (A_full), leakage injection, coverage-aware remedies (router, κ-mask), κ-stratified metrics, imputation invariance, corrected statistics, resampling, threshold analysis, SHAP, cost, **P4 rolling-origin** | Tables 1–7, 9, A1, A4 |
| `notebooks/elliptic_safeg_E4.ipynb` | Elliptic++: data checks, wallet-feature leakage check, audit, falsification test, P1/P3/P4 grids, remedies, κ-stratified metrics, corrected statistics, **strong baseline (A+AGG) with router / κ-mask** (E4); loads the GNN results of E3 | Tables 1, 3–8, A3, A4, Fig. 1 |
| `notebooks/elliptic_safeg_E3_gnn.ipynb` | Elliptic++: GCN and GraphSAGE training under the same partitions (3 seeds), seed variance, latency measurement | Table 8, Sect. 5.7 |
| `notebooks/banksim_leakfree_v11_D1.ipynb` | BankSim: audit (shortcut case), falsification test, P1/P2/P3 grids, leakage injection, remedies, corrected statistics, resampling, SHAP | Tables 1–4, 9, A2 |
| `paper/safeg_numbers.py` | Parses the frozen notebook outputs (`*.txt`) into data structures | – |
| `paper/tables.py`, `paper/figures.py` | Generate every LaTeX table and Figure 1 from `safeg_numbers.py` (no hand-typed numbers) | all tables, Fig. 1 |
| `paper/verify_claims.py` | Recomputes the key numbers quoted in the text and checks them | – |
| `paper/dump_outputs.py` | Re-creates `ieee.txt`, `ell.txt`, `ell_e3.txt`, `bs.txt` from the executed notebooks | – |

Each notebook starts with a revision log (in Turkish) documenting every methodological change and its rationale.

## Data

The datasets are not redistributed.

* **IEEE-CIS Fraud Detection** – `train_transaction.csv`, `train_identity.csv` from <https://www.kaggle.com/c/ieee-fraud-detection>; place them in the working directory.
* **Elliptic++** – from the authors' repository <https://github.com/git-disl/EllipticPlusPlus>; the notebook downloads the folder with `gdown` if it is missing (`ELLIPTIC_DIR`).
* **BankSim** – `bs140513_032310.csv` (Kaggle: *Synthetic data from a financial payment system*); place it in the working directory.

## Running

The notebooks were run on Google Colab (Python 3, CPU for the tabular grids, GPU for the GNNs). Set `USE_DRIVE = True`
to keep checkpoints on Google Drive; with `RESUME = True` completed folds are reloaded. Approximate run times
(FAST_MODE): IEEE-CIS ≈ 6–7 h, BankSim ≈ 4–5 h, Elliptic++ tabular ≈ 2 h, GNNs ≈ 30 min on a GPU.

To regenerate the manuscript tables from the executed notebooks:

```bash
cd paper
python dump_outputs.py ../notebooks/ieee_cis_pipeline_v11_D1.ipynb > ieee.txt
python dump_outputs.py ../notebooks/elliptic_safeg_E4.ipynb       > ell.txt
python dump_outputs.py ../notebooks/elliptic_safeg_E3_gnn.ipynb   > ell_e3.txt
python dump_outputs.py ../notebooks/banksim_leakfree_v11_D1.ipynb > bs.txt
python verify_claims.py
python tables.py > tables.tex
python figures.py
```

## Determinism and fingerprints

Two sources of run-to-run variation were found and removed (manuscript Sect. 4.4):

1. **IEEE-CIS entity subsample.** Entities are sorted by identifier before the seeded sample is drawn; rows are
   sorted stably by `(TransactionDT, TransactionID)`. Fingerprint: 200,000 rows, 6,764 positives, 73,965 entities,
   SHA-1 of sorted TransactionIDs `8e8c441d93937f9fd8c3456e5be1469c63a5e1cf`.
2. **Entity-grouped folds (P2).** Older scikit-learn versions assign equally sized groups to folds with an unstable
   sort. `GroupKFoldDet` implements the algorithm of scikit-learn ≥ 1.7 `GroupKFold(shuffle=False)` with a stable
   sort. P2 fold-assignment fingerprints (first 12 hex digits): IEEE-CIS `64f56d1834d0`, BankSim `7d2c4d05d580`.

BankSim data fingerprint: 594,643 rows, 7,200 positives, SHA-1 `728aa437723424422f63448c9d9b0579e47cb96d`.
The notebooks write these fingerprints to the checkpoint directory and refuse to mix checkpoints from a different
sample or split. Re-runs in fresh sessions reproduced the random and temporal protocols exactly (BankSim: all ten
models; IEEE-CIS: tree models; MLP and logistic regression vary in the third decimal with the hardware).

## Licence and citation

Code repository: <https://github.com/cihatyldz/graph_fraud_detection> (see the repository for the licence).
Please cite the manuscript and this archive (Zenodo DOI).
