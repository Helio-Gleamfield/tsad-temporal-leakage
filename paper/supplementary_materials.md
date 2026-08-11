# Supplementary Materials — Temporal Data Leakage Paper

## S1. Full Experimental Results

### S1.1 Per-Method Leakage Gaps (ΔAUC = AUC_random − AUC_temporal)

| Dataset | Z-score | IsolationForest | OCSVM | LSTM-AE | Transformer-AE |
|---------|---------|-----------------|-------|---------|-----------------|
| SWaT | +0.134 | −0.139 | +0.043 | −0.254 | −0.165 |
| TEP | +0.020 | +0.024 | +0.028 | +0.021 | +0.053 |
| MSL | +0.067 | +0.036 | −0.138 | +0.104 | +0.054 |
| SMAP | +0.016 | +0.124 | +0.092 | −0.005 | +0.009 |
| SMD | +0.065 | +0.118 | +0.043 | +0.090 | +0.150 |

### S1.2 Temporal-Split AUC (mean of 3 seeds)

| Dataset | Z-score | IsolationForest | OCSVM | LSTM-AE | Transformer-AE |
|---------|---------|-----------------|-------|---------|-----------------|
| SWaT | 0.712 | 0.822 | 0.773 | 0.869 | 0.909 |
| TEP | 0.793 | 0.740 | 0.683 | 0.803 | 0.777 |
| MSL | 0.578 | 0.520 | 0.635 | 0.537 | 0.559 |
| SMAP | 0.458 | 0.329 | 0.360 | 0.503 | 0.481 |
| SMD | 0.775 | 0.446 | 0.490 | 0.698 | 0.682 |

### S1.3 Random-Split AUC (mean of 3 seeds)

| Dataset | Z-score | IsolationForest | OCSVM | LSTM-AE | Transformer-AE |
|---------|---------|-----------------|-------|---------|-----------------|
| SWaT | 0.846 | 0.683 | 0.816 | 0.615 | 0.743 |
| TEP | 0.813 | 0.764 | 0.711 | 0.824 | 0.830 |
| MSL | 0.645 | 0.556 | 0.498 | 0.641 | 0.613 |
| SMAP | 0.473 | 0.453 | 0.452 | 0.498 | 0.490 |
| SMD | 0.840 | 0.564 | 0.533 | 0.788 | 0.832 |

## S2. Overlap Ablation Results (SWaT, LSTM-AE)

| Stride S | Overlap ρ | n_windows | Temporal AUC | Random AUC | ΔAUC |
|----------|-----------|-----------|-------------|-----------|------|
| 32 | 87.5% | 2000 | 0.762 | 0.677 | −0.085 |
| 64 | 75.0% | 2000 | 0.882 | 0.791 | −0.091 |
| 128 | 50.0% | 2000 | 0.847 | 0.705 | −0.142 |
| 256 | 0.0% | 1757 | 0.827 | 0.766 | −0.062 |

## S3. Experimental Configuration

### S3.1 Hardware
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- CUDA: 12.8, PyTorch 2.x with BF16 AMP
- OS: Windows 11, Python 3.12

### S3.2 Hyperparameters
- Window length: L = 256
- Window stride: S = 64 (default), S ∈ {32, 64, 128, 256} for ablation
- Training epochs: 15 (LSTM-AE), 15 (Transformer-AE)
- Optimizer: Adam, learning rate 1e-3
- LSTM-AE: bidirectional, 64 hidden units
- Transformer-AE: 2 layers, d_model = min(dim, 128), nhead = min(4, dim//8)
- Random seeds: 42, 43, 44
- Subsampling: stratified, max 2000 windows per dataset

### S3.3 Evaluation Metrics
- AUC-ROC: Area under the ROC curve
- AUC-PR: Area under the Precision-Recall curve
- F1-score: At 90th-percentile anomaly score threshold
- ΔAUC: AUC_random − AUC_temporal (leakage gap)
- Spearman ρ: Rank correlation between random-split and temporal-split method rankings

## S4. Data Availability

All datasets used in this study are publicly available:
- SWaT: iTrust Lab, Singapore University of Technology and Design
- TEP: Harvard Dataverse (Downs & Vogel, 1993)
- MSL, SMAP: NASA Open Data Portal (Hundman et al., KDD 2018)
- SMD: Large internet company (Su et al., KDD 2019)

## S5. Code Availability

All experiment scripts, trained model checkpoints, and analysis code will be released at:
`https://github.com/[TBD]/tsad-temporal-leakage`

The repository includes:
- `experiment_p0_comprehensive.py`: Full experiment pipeline (5 datasets × 5 methods)
- `experiment_p0_figures.py`: Publication-quality figure generation
- `pipeline_post_experiment.py`: Automated result integration into manuscript
- All trained model checkpoints and raw experiment outputs
