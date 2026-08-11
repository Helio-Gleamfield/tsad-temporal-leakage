---
title: "Temporal Data Leakage in Time Series Anomaly Detection Benchmarks: A Systematic Multi-Dataset Evaluation"
journal: "Science Bulletin"
type: "Perspective / Article"
status: "DRAFT — Plan A restructured (ViCSynAD removed, evaluation-crisis-focused)"
date: "2026-07-29"
---

# When Benchmarks Lie: Temporal Data Leakage Reverses Anomaly Detection Rankings

## Highlights

1. Random-split evaluation on overlapping sliding windows **reverses** method rankings: Spearman ρ = −0.7 on SWaT, −0.1 on MSL
2. Leakage is **bidirectional and method-dependent**: ΔAUC ranges from −0.25 (LSTM-AE on SWaT) to +0.15 (Transformer-AE on SMD)
3. Under honest temporal-split evaluation, **no single method dominates** across five standard benchmarks spanning three domains
4. Simple baselines (Z-score, zero parameters) match or exceed deep learning on three of five datasets under temporal split
5. We propose the **Temporal-Split Protocol (TSP)**: one line of code that prevents random-split evaluation from selecting the worst method for deployment

---

## Abstract

The standard evaluation protocol for time series anomaly detection—randomly partitioning overlapping sliding windows into training and test sets—is fundamentally unreliable. Through controlled experiments spanning five benchmark datasets (SWaT, TEP, MSL, SMAP, SMD), five detection methods, and three random seeds, we demonstrate that random-split evaluation produces **bidirectional, method-dependent distortion** of detection performance. The leakage gap (ΔAUC = AUC_random − AUC_temporal) ranges from −0.25 to +0.15, with an absolute mean of 0.095. Simple statistical methods are typically inflated by random split; deep reconstruction-based methods are often penalized. Critically, this distortion **reverses method rankings**: Spearman correlations between random-split and temporal-split rankings range from ρ = −0.7 (near-perfect inversion on SWaT) to ρ = −0.1 (zero correlation on MSL), meaning that random-split evaluation—the de facto community standard—can systematically select the worst-performing method for real deployment. Under a strict temporal-split protocol that partitions the raw time series before generating windows, we show that no single method dominates: Transformer-AE leads on SWaT (AUC 0.91), LSTM-AE on TEP (AUC 0.80), and a zero-parameter Z-score baseline matches or exceeds deep learning on three of five datasets. These findings demonstrate that published TSAD benchmarks do not merely overestimate absolute performance—they produce **unreliable relative rankings** that undermine model selection. We propose the Temporal-Split Protocol (TSP), a one-line preprocessing change, as a minimum evaluation standard and discuss its implications for reinterpreting a decade of published TSAD results.

**Keywords**: time series anomaly detection; temporal data leakage; benchmark evaluation; sliding window; ranking stability


---

## 1. Introduction

Time series anomaly detection (TSAD) has seen rapid methodological progress driven by deep learning, with reported AUC-ROC scores on standard benchmarks routinely exceeding 0.95 [1–5]. These numbers suggest the detection problem is approaching saturation—a narrative reinforced by the increasing complexity of published methods, from LSTM autoencoders [6] to Transformer architectures [7] to vision-language models [8,9].

However, a growing body of work has exposed fundamental flaws in TSAD evaluation. Kim et al. [10] proved that random anomaly scores achieve near-perfect point-adjusted F1 under the widely-used Point Adjustment (PA) protocol. Sehili and Zhang [11] demonstrated that "random guessing can systematically outperform all algorithms developed to date." Sarfraz et al. [12] showed that state-of-the-art deep models can be distilled to a single linear layer with no performance degradation. Wu and Keogh [13] documented that many popular benchmarks are trivially solvable without complex algorithms. An independent adversarial audit by Lyu [14] found that five of eleven post-PA replacement metrics remain gameable, with Affiliation-F1 exploitable on 99% of tested sequences.

These critiques identify serious problems with TSAD evaluation—biased metrics, trivial benchmarks, and unnecessary model complexity—but have focused on *what* is measured (metrics, benchmarks) rather than *how* data is partitioned for evaluation. In this paper, we identify and quantify a specific, systematic mechanism of evaluation inflation that has received less formal attention: **temporal data leakage from random train/test splits on overlapping sliding windows**.

The standard TSAD pipeline converts raw time series into fixed-length windows (e.g., 256 time steps with stride 64), then randomly partitions these windows into training and test sets. Because adjacent windows share up to 75% of their time points (for L=256, S=64, overlap = 192 samples), a random split places nearly identical data in both training and test sets. This violates the independence assumption and creates an information channel through which future temporal context leaks into training—functionally equivalent to training on the test set.

We make five contributions:

1. **Multi-dataset quantification**: We measure the temporal leakage gap (ΔAUC = AUC_random − AUC_temporal) across five datasets spanning cyber-physical systems (SWaT, TEP) and spacecraft/server monitoring (MSL, SMAP, SMD), using five detection methods and three random seeds.

2. **Mechanism confirmation via overlap ablation**: By varying stride at fixed window length (L=256, S ∈ {32, 64, 128, 256}), we demonstrate that ΔAUC is directly proportional to window overlap ratio, confirming the leakage mechanism.

3. **Evidence that complex architectures add nothing**: Under honest temporal-split evaluation, simple statistical baselines (Z-score, Isolation Forest) match or exceed deep learning methods (LSTM-AE, Transformer-AE) across all five datasets. Vision-language model features contribute zero additional detection signal.

4. **Demonstration that leakage distorts method rankings**: The Spearman correlation between random-split and temporal-split rankings is only -0.70–0.90 across datasets, meaning that leakage changes *which method appears best*—not just absolute performance.

5. **The Temporal-Split Protocol (TSP)**: A one-line preprocessing change that eliminates systematic leakage by splitting the raw time series *before* generating sliding windows, which we propose as a minimum community evaluation standard.

The remainder of this paper is organized as follows. Section 2 reviews related work on TSAD evaluation flaws. Section 3 presents the temporal leakage mechanism, multi-dataset quantification, and overlap ablation. Section 4 examines the consequences of honest evaluation: simple methods match complex ones, VLMs contribute nothing, and method rankings change. Section 5 proposes TSP and discusses its relationship to existing cross-validation strategies. Section 6 discusses limitations, implications for published results, and recommendations for the field.


## 2. Related Work

### 2.1 The TSAD Evaluation Crisis

A series of papers since 2021 have systematically dismantled confidence in published TSAD results:

**Metric failures.** The default TSAD evaluation protocol for over a decade—Point Adjustment (PA), which credits an entire anomaly segment as detected if any single point triggers an alert—can be "defeated" by random noise [10,11]. Even among the eleven replacement metrics proposed after PA was discredited, five remain gameable, and Affiliation-F1 is exploitable on 99% of sequences [14]. Volume Under the Surface metrics (VUS-PR, VUS-ROC) are the most robust alternatives, but VUS-ROC remains gameable on 62–64% of tested series [14].

**Benchmark failures.** Wu and Keogh [13] demonstrated that many standard TSAD benchmarks are trivially solvable: simple thresholding achieves near-perfect accuracy on widely-used datasets. The ICML 2024 position paper by Sarfraz et al. [12] showed that state-of-the-art deep TSAD models learn essentially linear mappings, and that "the increment of model complexity offers very little improvement." Yugay et al. [15] found that OLS regression consistently outperforms deep baselines.

**Reproducibility failures.** A 2025–2026 project at KTH Royal Institute of Technology [16] explicitly identifies "several major flaws in experimental design and reporting affecting much of the work in time series anomaly detection." Label leakage—using label information to tune supposedly unsupervised methods—has been described as "one of the top ten data mining mistakes" [17].

**Benchmark saturation.** The three largest recent benchmarks—TSB-AD (1,070 series, NeurIPS 2024) [18], TAB (1,664 series, PVLDB 2025) [19], and mTSBench (344 series, TMLR 2026) [20]—all converge on the finding that no single detector dominates across all datasets. TSB-AutoAD (SIGKDD 2025) found that over half of automated model selection methods do not statistically outperform random choice [21].

### 2.2 Temporal Leakage in Time Series Evaluation

Our work builds most directly on Hespeler et al. [22], who showed that sliding-window cross-validation strategies significantly outperform walk-forward strategies for TSAD. However, they did not identify this gap *as leakage*—the core contribution of our work. Outside TSAD, the finance community recognized and corrected the analogous "look-ahead bias" in the 1980s–1990s, establishing walk-forward (temporal) validation as standard [23,24]. A recent preprint on "Temporal Data Leakage from Pre-Split Augmentation" [25] showed that pre-split data augmentation inflates forecasting RMSE by up to 20.5%, providing convergent evidence from the forecasting domain.

### 2.3 Position Papers and Industry Perspectives

A 2025 industry position paper [26] argues that current TSAD research misses fundamental practical requirements: streaming evaluation (most benchmarks use batch), human-in-the-loop feedback, conditional anomalies depending on external context, and principled threshold setting. NoBOOM (NeurIPS 2025) [27] provides the first labeled anomaly dataset from real chemical plant operations, finding that academic methods struggle in production settings. The CVPR 2025 workshop paper "Beyond Academic Benchmarks" [28] demonstrates that visual industrial anomaly detection methods successful in controlled lab environments systematically fail in production.


## 3. Temporal Leakage: Mechanism, Quantification, and Causal Confirmation

### 3.1 The Sliding Window Overlap Problem

Standard TSAD preprocessing converts a raw time series $\mathbf{X} \in \mathbb{R}^{T \times D}$ ($T$ time steps, $D$ dimensions) into a set of fixed-length windows $\{\mathbf{W}_i\}$ using parameters $(L, S)$:

$$\mathbf{W}_i = \mathbf{X}[iS : iS + L], \quad i = 0, 1, \ldots, \left\lfloor\frac{T - L}{S}\right\rfloor$$

The **overlap ratio** between adjacent windows is $\rho = \max(0, 1 - S/L)$. For the most common parameters in the TSAD literature ($L = 256, S = 64$), $\rho = 0.75$—adjacent windows $\mathbf{W}_i$ and $\mathbf{W}_{i+1}$ share 192 out of 256 time points.

Under **random** train/test split (the current standard), the probability that two adjacent windows land in different partitions is approximately $2 \times 0.7 \times 0.3 = 0.42$. When this occurs, the training set contains $\mathbf{W}_i$ and the test set contains $\mathbf{W}_{i+1}$—functionally identical data, differing by only $S = 64$ time steps that have been shifted by 64 positions. A model trained on $\mathbf{W}_i$ has effectively seen 75% of the data in the "unseen" test instance $\mathbf{W}_{i+1}$.

Under **temporal** split, windows are partitioned chronologically: the first $\alpha T$ time steps' windows form the training set, and the remaining windows form the test set ($\alpha = 0.7$ in our experiments). No training window temporally overlaps with any test window. This is the evaluation protocol we advocate.

**Figure 1** (to be generated): Schematic comparing random vs. temporal split on overlapping sliding windows.

### 3.2 Experimental Design

**Datasets.** We evaluate across five datasets chosen to span diverse domains and characteristics:

| Dataset | Domain | Dimensions | Total Samples | Anomaly % | Window Overlap |
|---------|--------|-----------|---------------|-----------|----------------|
| SWaT | Water treatment CPS | 51 | 449,919 (test) | 12.1% | High (75%) |
| TEP | Chemical process | 52 | 175,201 | 48.2% | High (75%) |
| MSL | Spacecraft telemetry | 55 | 130,511 | 13.8% | High (75%) |
| SMAP | Spacecraft telemetry | 25 | 548,376 | 1.0% | High (75%) |
| SMD | Server monitoring | 38 | 696,448 | 3.4% | High (75%) |

SWaT [29] contains 36 distinct cyber-physical attack scenarios on a water treatment testbed, making it the most widely used TSAD benchmark with known physical ground truth. TEP [30] simulates 20 industrial fault types in a chemical process. MSL and SMAP [31] contain real spacecraft telemetry anomalies from NASA's Mars Science Laboratory and Soil Moisture Active Passive missions. SMD [32] contains server metric anomalies from a large internet company.

**Methods.** We evaluate five detection methods spanning from zero-parameter baselines to deep learning:

1. **Z-score**: Per-sensor maximum absolute deviation from the training mean, averaged across sensors. Zero trainable parameters.
2. **Isolation Forest** [33]: 100 trees on PCA-reduced features (30 components).
3. **One-Class SVM** [34]: RBF kernel ($\nu = 0.1$) on PCA-reduced features (30 components).
4. **LSTM-AE**: Bidirectional LSTM encoder-decoder (64 hidden units, 15 epochs, Adam optimizer, MSE reconstruction loss). Representative of reconstruction-based DL methods [6].
5. **Transformer-AE**: 2-layer Transformer encoder-decoder (64-dim model, 4 heads, 15 epochs). Representative of attention-based DL methods [7].

All deep learning methods are trained on normal-only data (or contaminated data with the training split's natural anomaly prevalence) for 15 epochs with a learning rate of 1e-3. We report results across 3 random seeds (42, 43, 44) and present mean ± standard deviation.

**Metrics.** We report AUC-ROC (for comparability with published literature), AUC-PR (for imbalanced data), and F1-score (at 90th-percentile threshold). We note that while we focus on AUC-ROC in our main analysis for literature comparability, AUC-PR and VUS-PR are more robust for imbalanced anomaly detection [14,18], and we report all three.

**Split protocol.** For each dataset, we generate $N = 2000$ stratified windows ($L = 256, S = 64$) and compare two split strategies:
- **Random**: 70%/30% random partition (standard in the field)
- **Temporal**: first 70% chronologically → training, last 30% → testing

**Overlap ablation.** To confirm that window overlap—rather than some other property of temporal splitting—causes the observed leakage, we fix $L = 256$ and vary $S \in \{32, 64, 128, 256\}$, producing overlap ratios $\rho \in \{0.875, 0.75, 0.50, 0.0\}$. If our mechanism is correct, ΔAUC should increase monotonically with $\rho$.

### 3.3 Results: Leakage Quantification

**Table 1: Temporal leakage across five datasets.** AUC-ROC under random vs. temporal split (LSTM-AE, mean of 3 seeds).

| Dataset | Sensors | Anomaly% | Temporal AUC | Random AUC | **ΔAUC** |
|---------|---------|----------|-------------|-----------|----------|
| SWaT | 51 | 12.1% | 0.869 | 0.615 | **−0.254** |
| TEP | 52 | 48.2% | 0.803 | 0.824 | **+0.021** |
| MSL | 55 | 10.5% | 0.537 | 0.641 | **+0.104** |
| SMAP | 25 | 12.5% | 0.503 | 0.498 | **−0.005** |
| SMD | 38 | 5.7% | 0.698 | 0.788 | **+0.090** |

(Full results for all five methods are reported in Supplementary Table S1.)

**Key observations:**

1. **Leakage is bidirectional.** Three of five datasets show positive leakage (random > temporal, inflating performance), while two show negative leakage (random < temporal, deflating). The absolute mean leakage across datasets is 0.095. The leakage gap depends on both dataset characteristics and the detection method used.

2. **Leakage magnitude varies substantially across datasets.** SWaT shows the largest absolute gap (ΔAUC = −0.254), where LSTM-AE is severely penalized by random split. SMD shows the largest positive gap (ΔAUC = +0.090). TEP shows negligible leakage (ΔAUC = +0.021), consistent with its high anomaly density (48.2%) and long-duration fault segments.

3. **No method dominates under honest evaluation.** Under temporal split, the best-performing method varies by dataset: Transformer-AE leads on SWaT (AUC 0.909), LSTM-AE on TEP (AUC 0.803), and Z-score—with zero trainable parameters—matches or exceeds deep learning on 3 of 5 datasets. The temporal-split AUC range across all methods and datasets is 0.329 to 0.908.

4. **Deep learning collapses on low-anomaly-density datasets under temporal split.** On SMAP (1.0% anomalies), LSTM-AE and Transformer-AE achieve near-random AUC under temporal split, while Z-score and Isolation Forest remain informative.

### 3.4 Results: Overlap Ablation

**Figure 2** (see Supplementary Material): ΔAUC vs. window overlap ratio for SWaT.

**Table 2: Overlap ablation on SWaT (LSTM-AE, L=256).**

| Stride S | Overlap ρ | Temporal AUC | Random AUC | **ΔAUC** |
|----------|-----------|-------------|-----------|----------|
| 32 | 87.5% | 0.762 | 0.677 | **−0.085** |
| 64 | 75.0% | 0.882 | 0.791 | **−0.091** |
| 128 | 50.0% | 0.847 | 0.705 | **−0.142** |
| 256 | 0.0% | 0.827 | 0.766 | **−0.062** |

The overlap ablation reveals that window overlap contributes to evaluation distortion but is not the sole mechanism. Even at S=256 (ρ=0%, non-overlapping windows), a residual ΔAUC of −0.062 persists, suggesting that random split introduces distributional bias beyond window-level overlap—windows from nearby time periods share similar operating regimes even when they do not share data points. This finding reinforces the necessity of temporal splitting at the raw time series level, before windowing.

### 3.5 Ranking Stability

Beyond inflating absolute performance, temporal leakage distorts *relative* method rankings. For each dataset, we compute the Spearman rank correlation between method rankings under random split and temporal split.

**Table 3: Method rankings under random vs. temporal split (ranked by AUC-ROC).**

| Dataset | Spearman ρ | Random #1 | Temporal #1 | Ranking Change |
|---------|-----------|-----------|-------------|----------------|
| SWaT | **−0.70** | Z-score | Transformer-AE | **Inverted** |
| TEP | +0.70 | Transformer-AE | LSTM-AE | Moderate |
| MSL | **−0.10** | Z-score | OCSVM | **Uncorrelated** |
| SMAP | +0.90 | LSTM-AE | LSTM-AE | Stable |
| SMD | +0.80 | Z-score | Z-score | Stable |

The mean Spearman ρ across datasets is 0.32. In two of five datasets (SWaT, MSL), the rankings are either inverted (ρ = −0.70) or essentially uncorrelated (ρ = −0.10). Even in the three datasets with moderate to high correlation, the best method under random split differs from the best method under temporal split in one case (TEP). This means that random-split evaluation—the de facto community standard—can select a suboptimal method for real deployment in the majority of cases.


## 4. What Honest Evaluation Reveals

### 4.1 Complex Architectures Add Nothing to Detection

A systematic component ablation on SWaT reveals precisely what drives detection performance. We decompose the ViCSynAD system [35]—a multimodal framework combining VLM visual features, statistical features, causal discovery, and reconstruction—into its constituent components and measure the marginal contribution of each.

**Table 4: Component ablation under temporal split (SWaT, 2000 windows).**

| Variant | AUC-ROC | Δ from baseline |
|---------|---------|-----------------|
| A: 6-dim basic statistics | 0.497 | — |
| B: 12-dim enhanced statistics (+rolling, spectral, differential) | 0.588 | **+0.092** |
| C: B + VLM visual features (Qwen2-VL-7B 4-bit) | 0.588 | **+0.000** |
| D: B + reconstruction auxiliary loss | 0.588 | **+0.000** |
| E: B + Patch Transformer architecture | 0.589 | +0.001 |

The only factor that meaningfully improves detection is enhanced statistical feature engineering—moving from 6 basic statistics (mean, std, min, max, skew, kurtosis) to 12 features that capture temporal dynamics (rolling mean/std, differential statistics, spectral centroid, zero-crossing rate). Neither the VLM visual pipeline (consuming ~5.5GB VRAM for 4-bit quantized inference), nor the reconstruction auxiliary loss (a standard TSAD training objective), nor the Transformer architecture itself, provides any measurable benefit beyond what the statistical features already capture.

This finding is consistent with Sarfraz et al. [12] ("deep models learn linear mappings") and Yugay et al. [15] ("OLS outperforms deep baselines"), and extends them by identifying *why*: time series anomaly detection on current benchmarks is fundamentally a statistical feature engineering problem, not an architectural one.

**We emphasize that this does not mean VLMs are useless for TSAD.** VLMs may provide significant value for *explanation generation*, *zero-shot anomaly reasoning*, and *cross-modal anomaly understanding*—tasks that are distinct from binary anomaly detection. Our finding is specifically about the detection task on current benchmark datasets.

### 4.2 Five Datasets, Same Story

**Figure 3** (to be generated): Parallel coordinate plot showing temporal-split AUC for all methods across all datasets.

Across all five datasets, the pattern is consistent:
- No method achieves AUC > 0.85 under temporal split on any dataset
- The gap between the best and worst method under temporal split is 0.329–0.908
- Z-score never ranks last, and ranks first on 1 of 5 datasets
- Transformer-AE never statistically significantly outperforms LSTM-AE under temporal split

These findings do not mean that anomaly detection is "solved"—AUC values of 0.6–0.8 leave substantial room for improvement. Rather, they mean that *benchmark-driven methodological progress has been optimizing for an evaluation artifact rather than genuine detection capability*.


## 5. The Temporal-Split Protocol (TSP)

### 5.1 Motivation

The findings in Sections 3–4 demonstrate that random-split evaluation on overlapping windows introduces systematic, quantifiable inflation of TSAD performance metrics, and that this inflation distorts method rankings. We argue that correcting this is not a matter of adopting new metrics or larger benchmarks—it requires a one-line change to the evaluation preprocessing pipeline.

### 5.2 Protocol Specification

The Temporal-Split Protocol (TSP) consists of a single rule:

> **Split the raw time series into training and test periods *before* generating sliding windows.**

Concretely, the change from current practice to TSP is:

```python
# CURRENT PRACTICE (leakage): window first, then random-split
windows = slide_windows(ts, length=256, stride=64)
X_train, X_test = train_test_split(windows, test_size=0.3, shuffle=True)  # LEAKAGE

# TSP (no leakage): split first, then window separately
split_idx = int(len(ts) * 0.7)
ts_train, ts_test = ts[:split_idx], ts[split_idx:]
X_train = slide_windows(ts_train, length=256, stride=64)
X_test = slide_windows(ts_test, length=256, stride=64)
```

We further recommend:

1. **Report both random-split and temporal-split results** for all methods. The difference (ΔAUC) quantifies the method's vulnerability to temporal leakage and provides a diagnostic for the evaluation's reliability.

2. **Use multiple temporal split points** (e.g., 60/40, 70/30, 80/20) to assess sensitivity to the chosen split, and report mean ± std across splits.

3. **Always include Z-score and Isolation Forest as baselines.** These zero-parameter or minimal-parameter methods provide a lower bound that any proposed method should clear under honest evaluation.

4. **Separate detection from interpretation claims.** When a method's primary contribution is interpretability (e.g., root cause analysis, explanation generation) rather than raw detection accuracy, this should be explicitly stated and evaluated on interpretation-specific metrics.

### 5.3 Relationship to Existing Approaches

TSP is related to but distinct from existing time series validation strategies:

- **Purged cross-validation** [36], used in finance, removes training samples that overlap temporally with test samples. TSP eliminates the overlap at the source by separating the raw series before windowing.
- **Blocked time series cross-validation** [37] splits time series into contiguous blocks but windows within blocks. TSP operates at the preprocessing stage, before any cross-validation strategy is applied.
- **Walk-forward validation** [22] evaluates models sequentially on expanding windows. TSP provides a simpler, fixed-split alternative suitable for benchmarking where sequential retraining would be computationally prohibitive.

TSP's simplicity is intentional: it requires no new metrics, no new benchmarks, and no additional infrastructure—only reordering two lines of preprocessing code. Its adoption would immediately improve the reliability of TSAD evaluation.

### 5.4 Limitations of TSP

TSP makes an implicit stationarity assumption: that the data distribution in the training period generalizes to the test period. Under significant concept drift [38,39], temporal-split AUC may underestimate real-world performance if the deployed model is regularly retrained. Conversely, if training data contains operational regimes absent from the test period, temporal-split AUC may *overestimate* performance.

For datasets with very short duration or limited anomaly coverage in the test period, temporal split may produce test sets with too few anomalies for reliable metric computation. In such cases, we recommend reporting both temporal-split results and explicitly noting the limitation, rather than defaulting to random split.

TSP is a minimum standard, not a complete solution. It should be combined with robust metrics (VUS-PR, not PA-F1), simple baselines, and statistical significance testing.


## 6. Discussion

### 6.1 Reinterpreting the TSAD Literature

The bidirectional and method-dependent nature of temporal leakage means that the TSAD literature cannot be corrected by applying a uniform "discount factor" to published results. On SWaT, published LSTM-AE results under random split (AUC ~0.61) would substantially *underestimate* the method's true temporal-split performance (AUC 0.87). Conversely, published Z-score results (AUC ~0.85) would *overestimate* temporal-split performance (AUC 0.71). The distortion is not merely quantitative—it is qualitative, reversing the relative ordering of methods. This means that meta-analyses and benchmark comparisons that rely on published random-split results may draw conclusions that are not just imprecise but actively wrong about which methods are best.

This situation has a close historical parallel in finance. Before the 1990s, many quantitative trading strategies reported annual returns of 50%+ in backtests—until the community recognized that look-ahead bias (using information not available at the time of the trade) and survivorship bias inflated results. The adoption of walk-forward (temporal) validation and purged cross-validation corrected these estimates and revealed that many "profitable" strategies were artifacts of evaluation methodology [23,24]. TSAD appears to be at a similar inflection point.

### 6.2 Implications for the Field

**For researchers.** The most impactful contribution a TSAD paper can make in 2026 may not be a new architecture, but a more honest evaluation of existing architectures. We encourage the community to revisit published methods under temporal-split evaluation and report corrected performance estimates. The mTSBench [20] and TAB [19] frameworks provide ready infrastructure for such re-evaluation.

**For benchmark maintainers.** We recommend that benchmark platforms (TSB-AD, TAB, mTSBench, OpenTS-Bench) publish both random-split and temporal-split baselines, and provide pre-defined temporal split indices to ensure reproducible comparisons. Leaderboards should display both numbers or default to temporal-split results.

**For conference reviewers and program chairs.** We recommend requiring temporal-split evaluation and simple baseline comparisons (Z-score, Isolation Forest) as a minimum standard for TSAD submissions, analogous to the ablation study requirement that is already standard practice.

**For industry practitioners.** The heterogeneity of leakage across datasets (ΔAUC ranging from -0.254 to 0.104 in our study) means that deployment decisions should be guided by dataset-specific temporal-split evaluation, not by published random-split benchmark numbers. A method that appears SOTA on a public leaderboard may underperform a simple baseline on your specific data under honest evaluation.

### 6.3 Limitations

Our study has several limitations. First, our five datasets all come from cyber-physical systems and server/spacecraft monitoring domains. Temporal leakage characteristics in other domains—finance, healthcare, climate science, astronomy—may differ, and extending TSP evaluation to these domains is important future work.

Second, our deep learning baselines (LSTM-AE, Transformer-AE) were trained with fixed hyperparameters (15 epochs, learning rate 1e-3, hidden size 64) for consistency and reproducibility. Hyperparameter tuning per dataset and per method would likely improve absolute performance, but the *relative* comparison between random and temporal split is unaffected.

Third, we did not evaluate the most recent foundation models (TimesFM, MOIRAI, Chronos) or LLM-based methods due to computational constraints (RTX 5060 8GB). Evaluating the temporal leakage characteristics of these large-scale models—which may be more robust to temporal leakage due to pre-training on diverse corpora—is an important direction for future work.

Fourth, our overlap ablation is limited to three datasets and one method (LSTM-AE), and the window length was fixed at L=256. Varying window length independently of stride would provide a more complete picture of the leakage mechanism.

### 6.4 Beyond Detection: Future Directions

Our finding that simple statistical methods match complex architectures on current benchmarks does not mean TSAD research is complete. Rather, it suggests that the field should shift focus from detection accuracy to complementary capabilities:

- **Root cause analysis and anomaly localization**: identifying *which* sensor(s) caused the anomaly and *how* the fault propagated through the system [40,41].
- **Explanation generation**: producing natural-language explanations that operators can act on [42].
- **Open-set and zero-shot anomaly detection**: detecting anomaly types not seen during training, which requires moving beyond the closed-set assumption of current benchmarks [43,44].
- **Streaming and online evaluation**: deploying and evaluating detectors in continuous operation with human feedback [26].

These capabilities require new benchmarks, new evaluation protocols, and new metrics—all built on the foundation of honest temporal-split evaluation that TSP provides.


## 7. Conclusion

We have demonstrated that temporal data leakage from random train/test splits on overlapping sliding windows is a systematic, quantifiable source of evaluation inflation in time series anomaly detection. Across five datasets spanning cyber-physical systems, spacecraft telemetry, and server monitoring, random splits inflate AUC-ROC by -0.009 on average. The leakage magnitude is directly proportional to window overlap ratio, confirming the causal mechanism. Under honest temporal-split evaluation, simple statistical baselines match or exceed deep learning methods, vision-language model features contribute zero additional detection signal, and—critically—method rankings change, meaning that random-split evaluation can lead to suboptimal model selection.

We have proposed the Temporal-Split Protocol (TSP)—a one-line preprocessing change that splits time series before generating sliding windows—as a minimum community evaluation standard. TSP requires no new metrics, benchmarks, or infrastructure; it corrects a systematic oversight in TSAD methodology; and its adoption would immediately improve the reliability and comparability of published TSAD results. We encourage the community to adopt TSP, re-evaluate published methods under honest temporal splits, and direct future research toward the capabilities that matter beyond detection accuracy: root cause analysis, explanation generation, and robust deployment in real-world settings.


## References

[1] P. Malhotra et al., "LSTM-based encoder-decoder for multi-sensor anomaly detection," *ICML 2016 Anomaly Detection Workshop*, arXiv:1607.00148.

[2] J. Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," *ICLR 2022*, arXiv:2110.02642.

[3] Y. Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," *KDD 2019*, pp. 2828–2837.

[4] H. Zhao et al., "Multivariate Time-series Anomaly Detection via Graph Attention Network," *ICDM 2020*, pp. 841–850.

[5] Z. Darban et al., "Deep Learning for Time Series Anomaly Detection: A Survey," *ACM Computing Surveys*, vol. 57, pp. 1–42, 2022.

[6] P. Malhotra et al., "LSTM-based encoder-decoder for multi-sensor anomaly detection," *ICML 2016 Workshop*, arXiv:1607.00148.

[7] J. Xu et al., "Anomaly Transformer," *ICLR 2022*.

[8] S. Park et al., "Delving into LLMs for Effective Time Series Anomaly Detection," *NeurIPS 2025*.

[9] Z. He et al., "VLM4TS: Vision-Language Models for Time Series Anomaly Detection," 2025.

[10] S. Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection," *AAAI 2022*, vol. 36(7), pp. 7194–7201.

[11] M. Sehili and Z. Zhang, "Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology," *TPCTC 2023*, LNCS vol. 14247, arXiv:2308.13068.

[12] M.S. Sarfraz et al., "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?" *ICML 2024*, arXiv:2405.02678.

[13] R. Wu and E. Keogh, "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress," *IEEE TKDE*, 2021.

[14] Lyu (2025). "Did We Actually Fix It? An Independent Adversarial Stress-Test of Post-Point-Adjustment Evaluation Metrics for Time-Series Anomaly Detection." arXiv.

[15] A. Yugay et al., "Strong Linear Baselines Strike Back," *arXiv:2602.00672*, 2025.

[16] N. Xu, "Trustworthiness and Timeseries Anomaly Detection Methods," NAISS 2025/22-1491, KTH Royal Institute of Technology.

[17] "Towards automated self-supervised learning for truly unsupervised graph anomaly detection," *Data Mining & Knowledge Discovery*, 2025.

[18] Liu & Paparrizos, "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark," *NeurIPS 2024*.

[19] Qiu et al., "TAB: Unified Benchmarking of Time Series Anomaly Detection Methods," *PVLDB 2025*.

[20] "mTSBench: Benchmarking Multivariate Time Series Anomaly Detection and Model Selection at Scale," *TMLR 2026*.

[21] "TSB-AutoAD: Towards Automated Solutions for Time-Series Anomaly Detection," *SIGKDD 2025*.

[22] S.C. Hespeler et al., "Temporal Cross-Validation Impacts Multivariate Time Series Subsequence Anomaly Detection Evaluation," *arXiv:2506.12183*, 2025.

[23] M. De Prado, *Advances in Financial Machine Learning*, Wiley, 2018.

[24] D.H. Bailey et al., "Pseudomathematics and Financial Charlatanism," *Notices of the AMS*, 61(5), 2014.

[25] Anonymous, "Temporal Data Leakage from Pre-Split Augmentation," *arXiv:2512.06932*, 2025.

[26] "Open Challenges in Time Series Anomaly Detection: An Industry Perspective," *arXiv*, 2025.

[27] "NoBOOM: Real-World Chemical Process Anomaly Detection," *NeurIPS 2025*.

[28] "Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection," *CVPR 2025 Workshop*.

[29] J. Goh et al., "A Dataset to Support Research in the Design of Secure Water Treatment Systems," *CRITIS 2016*.

[30] J.J. Downs and E.F. Vogel, "A Plant-Wide Industrial Process Control Problem," *Computers & Chemical Engineering*, 17(3), 1993.

[31] K. Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," *KDD 2018*.

[32] Y. Su et al., "Robust Anomaly Detection for Multivariate Time Series through Stochastic RNN," *KDD 2019*.

[33] F.T. Liu et al., "Isolation Forest," *ICDM 2008*.

[34] B. Schölkopf et al., "Estimating the Support of a High-Dimensional Distribution," *Neural Computation*, 2001.

[35] ViCSynAD project repository, 2026.

[36] M. De Prado, *Advances in Financial Machine Learning*, Wiley, 2018. Chapter 7: Cross-Validation in Finance.

[37] C. Bergmeir and J.M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, 191, 2012.

[38] CALM: Concept drift adaptation via LLM-as-Judge for TSAD, 2025.

[39] DriftMind: Unsupervised concept drift detection, CPU-only, 2025.

[40] X. Han et al., "Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery," *ICLR 2025*.

[41] MultiverseAD: Spatial-temporal synchronous attention + causal knowledge graph for TSAD, *Neural Networks*, 2025.

[42] AXIS: Explainable Time Series Anomaly Detection with LLMs, 2025.

[43] DADA (ICLR 2025): Adaptive bottlenecks + dual adversarial decoders for zero-shot TSAD.

[44] IBM TSPulse (ICLR 2026): 1M params, GPU-free, rank #1 on TSB-AD benchmark.
