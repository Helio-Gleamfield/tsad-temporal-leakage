# Temporal Data Leakage in Time Series Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and experiments for the manuscript:

> **When Benchmarks Lie: Temporal Data Leakage Reverses Time Series Anomaly Detection Rankings**
>
> Xicheng Zeng, College of Finance and Statistics, Hunan University

---

## Overview

The standard evaluation protocol for time series anomaly detection (TSAD) randomly partitions overlapping sliding windows into training and test sets. We show this practice introduces systematic temporal data leakage that can **reverse method rankings** — selecting the worst-performing method for deployment.

This repository contains all experiment scripts, analysis code, and figure generation code to reproduce the results.

## Key Findings

- **Absolute leakage gap** |ΔAUC| averages 0.080 across 25 method–dataset pairs
- **Ranking reversal**: Spearman ρ = −0.7 on SWaT (the most widely used benchmark)
- **Bidirectional distortion**: Random split can either inflate or deflate performance
- **TSP fix**: One line of code — split the raw time series before generating windows

## Project Structure

```
├── src/vicsynad/          # Core library (config, data loaders, modules)
├── scripts/               # All experiment and figure generation scripts
│   ├── experiment_p0_comprehensive.py    # Main 5×5×3 experiment
│   ├── experiment_p2_vlm_direct.py      # VLM direct vision experiment
│   ├── experiment_p2_crossdomain.py     # Cross-domain experiment
│   ├── experiment_p3_fast.py            # 9-method sklearn experiment
│   ├── experiment_p3_multisplit.py      # Multi-split robustness
│   ├── figures_final.py                 # Publication figure generation
│   └── graphical_abstract.py            # Graphical abstract generation
├── experiments/           # Experiment outputs (JSON)
├── figures/               # Generated figures (PDF + PNG)
├── paper/                 # LaTeX manuscript and supplementary materials
│   ├── manuscript_final.tex             # Authoritative manuscript
│   ├── supplementary_materials.tex      # Supplementary tables
│   └── cover_letter_final.md           # Cover letter
└── .env.example           # API key configuration template
```

## Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA (optional; CPU-only works for most experiments)
- See `requirements.txt` for package dependencies

### Installation
```bash
pip install -r requirements.txt

# For VLM experiments: set your API key
cp .env.example .env
# Edit .env and add your DashScope API key
```

### Reproduce Main Results
```bash
# Full experiment (5 datasets × 5 methods × 3 seeds, ~8 min on GPU)
python -X utf8 scripts/experiment_p0_comprehensive.py

# Generate statistics
python -X utf8 scripts/compute_stats.py

# Generate figures
python -X utf8 scripts/figures_final.py
```

### Verify Pipeline
```bash
python -X utf8 scripts/verify_pipeline.py
```

## Hardware Notes

Developed on NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM), CUDA 12.8. Most experiments run on CPU. Windows-specific notes:
- Use `num_workers=0` (multiprocessing with >0 workers crashes on Windows)
- Use `python -X utf8` flag for GBK encoding compatibility

## Data

All datasets are publicly available:
- **SWaT**: iTrust Lab, Singapore University of Technology and Design
- **TEP**: Harvard Dataverse (Downs & Vogel, 1993)
- **MSL, SMAP**: NASA Open Data Portal
- **SMD**: KDD 2019
- **MITDB**: PhysioNet
- **Daphnet**: UCI Machine Learning Repository
- **CICIDS**: Canadian Institute for Cybersecurity

Set `TSAD_DATA_ROOT` environment variable to point to your data directory.

## License

MIT — see LICENSE file.

## Citation

If you use this code, please cite:
```
Zeng, X. "When Benchmarks Lie: Temporal Data Leakage Reverses
Time Series Anomaly Detection Rankings." Science Bulletin, 2026.
```
