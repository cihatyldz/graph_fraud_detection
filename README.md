# Graph-Based Feature Engineering for Financial Fraud Detection

> **Leak-Free Graph Feature Engineering Pipeline for Credit Card / Banking Fraud Detection**

A rigorous, reproducible machine learning pipeline that investigates the contribution of **graph-based features** (centrality metrics, community detection) to financial fraud detection. The project employs a strict **leak-free** evaluation protocol — all features (graph, behavioral aggregates, community fraud rates, imputations, categorical encodings) are computed exclusively from training data within each cross-validation fold.

---

## 📌 Highlights

- **Ablation study** with four feature sets (A → D) to isolate the pure contribution of graph features
- **Three evaluation protocols**: Stratified K-Fold, GroupKFold (entity-level), Out-of-Time (temporal)
- **Ten classifiers** compared: LR, NB, DT, KNN, SVM, RF, GBDT (HistGradientBoosting), XGBoost, LightGBM, MLP
- **Statistical significance** tests: paired t-test, Wilcoxon signed-rank, Cohen's d
- **SHAP-based feature attribution** for interpretability
- **Class imbalance** strategies: None, RandomUnder, NearMiss, SMOTE, SMOTETomek
- **Artifact audit** (PaySim lessons learned): single-feature AUC scan, rule detector, degeneration check
- **Automated leak self-test** that verifies features depend on training fold

---

## 🗂️ Project Structure

```
├── README.md
├── banksim_leakfree_v1.ipynb      # 🔑 Primary project — BankSim dataset
├── ieee_cis_pipeline_v1.ipynb     #    Validation project — IEEE-CIS dataset
└── data/
    ├── bs140513_032310.csv         # BankSim raw data
    ├── train_transaction.csv       # IEEE-CIS transaction data
    ├── train_identity.csv          # IEEE-CIS identity data
    ├── results_*.csv               # BankSim experiment results
    ├── ieee_results_*.csv          # IEEE-CIS experiment results
    ├── *_shap_importance.csv       # SHAP feature importance
    ├── *_timing_analysis.json      # Inference timing benchmarks
    ├── paper_tables_*.xlsx         # Paper-ready summary tables
    └── ...
```

---

## 🔬 Methodology

### Feature Set Design (Ablation Study)

| Feature Set | Description | Purpose |
|:-----------:|:------------|:--------|
| **Set A** | Raw transaction features (amounts, categories, IDs, time deltas) | Baseline |
| **Set B** | A + Behavioral aggregates (velocity, unique counts, avg amounts) | Measures behavioral contribution |
| **Set C** | B + Graph features (centrality + community) | Full model |
| **Set D** | A + Graph features only (no behavioral aggregates) | **Isolates pure graph contribution** |

### Graph Construction

The transaction graph is built **exclusively from training data** in each fold:

- **BankSim**: Bipartite graph — Customer ↔ Merchant edges weighted by transaction count and total amount
- **IEEE-CIS**: Multi-relational graph — `card1 ↔ addr1` and `card1 ↔ P_emaildomain` edges

### Graph Features Extracted

| Category | Features |
|:---------|:---------|
| **Centrality (Cat. 1)** | Weighted degree, PageRank, Betweenness centrality (sampled, k=300), Eigenvector centrality, Clustering coefficient |
| **Community (Cat. 2)** | Louvain community ID, `community_fraud_rate` (from training labels only) |
| **Behavioral (Cat. 3)** | Transaction count, avg/std/max amount, unique addresses, unique emails, time span, velocity, amount ratios |

### Leak Prevention Protocol

All of the following are computed **only from each fold's training partition**:

1. **Graph construction** — only training transactions
2. **Centrality metrics** — computed on the training graph
3. **Louvain communities** — detected on the training graph
4. **`community_fraud_rate`** — derived from training labels only
5. **Behavioral aggregates** — calculated from training transactions
6. **Missing value imputation** — training median/mode
7. **Categorical encoding** — training-derived dictionary; unseen values → `-1`

**Cold-start policy**: For unseen entities in the test set, centrality defaults to `0.0`, community to `-1`, `community_fraud_rate` to training prior, and aggregates to training median.

---

## 📊 Evaluation Protocols

| Protocol | Description | Leak Prevented |
|:---------|:------------|:---------------|
| **P1 — Stratified 5-Fold CV** | Standard random stratified split | Target leakage, look-ahead bias |
| **P2 — GroupKFold** | Customer/card-based grouping — same entity never in both train and test | Entity leakage |
| **P3 — Out-of-Time** | Temporal split (80/20) — train on past, test on future | Temporal leakage |

### Metrics

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, PR-AUC (critical for imbalanced data)
- MCC (Matthews Correlation Coefficient)
- FPR (False Positive Rate)
- Precision@1%, Recall@1%
- Full confusion matrix (TP, FP, TN, FN)

---

## 📈 Key Results

### BankSim (Primary Project)

**Dataset**: 594,643 transactions · 4,112 customers · 50 merchants · 7,200 fraud (1.21%) · Imbalance ratio 1:81

#### Protocol 1 — Stratified 5-Fold CV (F1-Score)

| Model | Set A (Raw) | Set B (A+Behavioral) | Set C (B+Graph) | Set D (A+Graph) | Δ(C−A) | Δ(D−A) |
|:------|:-----------:|:--------------------:|:---------------:|:---------------:|:------:|:------:|
| RF | 0.8025 | 0.8628 | **0.8705** | 0.8652 | +0.0680 | +0.0627 |
| LightGBM | 0.8206 | 0.8391 | **0.8561** | 0.8521 | +0.0355 | +0.0315 |
| MLP | 0.8012 | 0.8562 | **0.8615** | 0.8548 | +0.0603 | +0.0536 |
| GBDT | 0.8185 | 0.8505 | 0.8481 | **0.8540** | +0.0296 | +0.0355 |
| XGBoost | 0.8095 | 0.8423 | 0.8373 | 0.8388 | +0.0278 | +0.0293 |

#### Protocol 3 — Out-of-Time (Best Model: MLP on Set C)

```
Precision: 0.8757  |  Recall: 0.8357  |  F1: 0.8553
AUC-ROC:  0.9989  |  PR-AUC: 0.9339  |  MCC: 0.8540
```

### IEEE-CIS (Validation Project)

**Dataset**: 590,540 transactions · 3.5% fraud rate · Real e-commerce data

#### Protocol 1 — Stratified 5-Fold CV (F1-Score)

| Model | Set A (Raw) | Set B (A+Behavioral) | Set C (B+Graph) | Set D (A+Graph) | Δ(D−A) |
|:------|:-----------:|:--------------------:|:---------------:|:---------------:|:------:|
| LightGBM | 0.3999 | 0.4188 | **0.4200** | 0.4144 | +0.0145 |
| XGBoost | 0.4191 | 0.4263 | **0.4352** | 0.4310 | +0.0119 |
| RF | **0.4224** | 0.4341 | 0.4277 | 0.4289 | +0.0065 |
| MLP | 0.3136 | 0.3334 | **0.3385** | 0.3369 | +0.0233 |

> **Key finding**: Graph features provide consistent but modest improvements in the leak-free IEEE-CIS pipeline. The contribution is smaller than in BankSim due to the dataset's inherent characteristics (25.4% single-transaction entities, sparser graph structure).

---

## 🛡️ Artifact Audit (Lessons from PaySim)

Before running experiments, an automated audit checks for data artifacts:

| Check | Description | BankSim Result | IEEE-CIS Result |
|:------|:------------|:---------------|:----------------|
| **Single-feature AUC scan** | Any feature with AUC > 0.95 alone? | ✅ No artifact | ✅ No artifact (max 0.69) |
| **Rule detector** | Simple threshold rules achieving high F1? | ✅ Rules insufficient | ✅ Best rule F1=0.18 |
| **Degeneration check** | Behavioral aggregates collapsing to constants? | ✅ Information preserved | ✅ Aggregates carry info |

**Comparison**: PaySim was rejected because a two-condition rule achieved F1=0.989 — the problem didn't require learning.

---

## ⚡ Getting Started

### Prerequisites

```bash
pip install pandas numpy networkx python-louvain scikit-learn xgboost lightgbm imbalanced-learn shap scipy matplotlib seaborn
```

### Dependencies

| Package | Purpose |
|:--------|:--------|
| `pandas`, `numpy` | Data manipulation |
| `networkx` | Graph construction & centrality |
| `python-louvain` | Community detection (Louvain) |
| `scikit-learn` | ML models, metrics, preprocessing |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `imbalanced-learn` | Class imbalance strategies (SMOTE, etc.) |
| `shap` | Feature attribution |
| `scipy` | Statistical tests |
| `matplotlib`, `seaborn` | Visualization |

### Running the Notebooks

#### BankSim (Primary)

1. Place `bs140513_032310.csv` in the project root (or `data/` directory)
2. Open `banksim_leakfree_v1.ipynb` in Jupyter / Google Colab
3. Configure parameters at the top of the notebook:

```python
FAST_MODE = True       # True: ~4h | False: ~7h (full results)
SUBSAMPLE_N = None     # None → full data | 200000 → faster first run
N_FOLDS = 5
SEED = 42
```

4. Run all cells sequentially. Checkpoints are saved automatically for Colab crash recovery.

#### IEEE-CIS (Validation)

1. Place `train_transaction.csv` and `train_identity.csv` in the project root
2. Open `ieee_cis_pipeline_v1.ipynb`
3. Configure parameters:

```python
FAST_MODE = True
SUBSAMPLE_N = 200000   # None → full 590K data (~6-8h)
GRAPH_RELATIONS = [('card1','addr1'), ('card1','P_emaildomain')]
```

4. Run all cells. The IEEE-CIS pipeline includes the same leak-free protocol, artifact audit, and statistical tests.

### Checkpoint / Resume

Both notebooks support checkpoint-based resume for long runs (especially on Google Colab):

```python
CHECKPOINT_DIR = 'checkpoints'  # or 'checkpoints_ieee'
RESUME = True                    # True: resume from last completed fold
```

For Google Drive persistence:
```python
from google.colab import drive; drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/fraud_checkpoints'
```

---

## 📖 Notebook Sections

### BankSim (`banksim_leakfree_v1.ipynb`)

| # | Section | Description |
|:-:|:--------|:------------|
| 1 | Setup & Configuration | Install dependencies, set hyperparameters |
| 2 | Data Loading & EDA | Load BankSim data, integrity checks, class distribution |
| 3 | Graph Construction | Build bipartite Customer–Merchant graph (training-only) |
| 4 | Feature Engineering | Compute centrality, community, behavioral aggregates (leak-free) |
| 5 | Feature Sets & Metrics | Define A/B/C/D sets, comprehensive metric suite |
| 6 | Protocol 1 — Stratified 5-Fold CV | Leak-free random cross-validation |
| 7 | Protocol 2 — GroupKFold | Customer-based entity leakage prevention |
| 8 | Protocol 3 — Out-of-Time | Temporal train/test split |
| 9 | Protocol Comparison | Side-by-side F1 ablation tables + contribution decomposition |
| 10 | Leak Self-Test | Automated verification that features depend on training fold |
| 11 | Class Imbalance Strategies | None, RandomUnder, NearMiss, SMOTE, SMOTETomek |
| 12 | Best Model Detailed Analysis | Confusion matrix, PR curve, threshold analysis |
| 13 | Statistical Significance | Paired t-test, Wilcoxon, Cohen's d |
| 14 | SHAP Feature Attribution | Per-category contribution breakdown |
| 15 | Inference Timing | Algorithm-level latency, transactions/second |

### IEEE-CIS (`ieee_cis_pipeline_v1.ipynb`)

| # | Section | Description |
|:-:|:--------|:------------|
| 1 | Setup & Configuration | Parameters, graph relation definitions |
| 2 | Data Loading | Load & merge transaction + identity tables, memory optimization |
| 3 | Prevalence-Preserving Sampling | Random subsample that maintains original fraud rate |
| 4 | Raw Feature Groups | Define A_NUMERIC, A_CATEGORICAL, D columns, Vesta C columns |
| 5 | Leak-Free Feature Computation (Core) | `compute_fold_features()` — all features from training only |
| 6 | Feature Sets & Metrics | A/B/C/D + Vesta C reference set |
| 7 | Artifact Audit | Single-feature AUC, rule detector, degeneration check |
| 8 | Leak Self-Test | Automated leak verification |
| 9-12 | Three Evaluation Protocols | Same as BankSim: Stratified, GroupKFold, Out-of-Time |
| 13 | Statistical Significance | Paired tests with Cohen's d effect sizes |
| 14 | Class Imbalance | Sampling strategy comparison |
| 15 | Best Model Analysis | Detailed metrics, confusion matrix, PR curve |
| 16 | SHAP Attribution | Feature importance by category |
| 17 | Timing Analysis | Inference performance benchmarks |

---

## 🔧 Configuration Reference

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `SEED` | 42 | Random seed for reproducibility |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `FAST_MODE` | True | Use fewer estimators for speed |
| `SUBSAMPLE_N` | None | Subsample size (None = full data) |
| `BETWEENNESS_K` | 300 | Betweenness centrality sampling size |
| `TEMPORAL_CUT` | 144 (BankSim) / 80th percentile (IEEE-CIS) | Train/test temporal boundary |
| `KNN_TRAIN_CAP` | 50,000 | Max training samples for KNN (quadratic scaling) |
| `SAMPLING_FOLDS` | 3 | Folds for class imbalance experiments |
| `THRESHOLD_MODE` | 'tuned' | F1-maximizing threshold from validation subset |

---

## 📝 Citation

If you use this code in your research, please cite:

```
@article{graphml-fraud-detection,
  title={Graph-Based Feature Engineering for Financial Fraud Detection: A Leak-Free Evaluation},
  year={2026}
}
```

---

## 📄 License

This project is intended for academic and research purposes.

---

## 🙏 Acknowledgments

- **BankSim** dataset — Lopez-Rojas et al.
- **IEEE-CIS Fraud Detection** dataset — Vesta Corporation / IEEE Computational Intelligence Society
- **NetworkX** — Graph analysis library
- **python-louvain** — Community detection
- **SHAP** — Explainable AI
