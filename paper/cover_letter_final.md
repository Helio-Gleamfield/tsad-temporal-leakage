---
title: "When Benchmarks Lie — Cover Letter"
journal: "Science Bulletin"
date: "2026-08-10"
---

Dear Editors,

We submit our manuscript entitled **"When Benchmarks Lie: Temporal Data Leakage Reverses Time Series Anomaly Detection Rankings"** for consideration as an Article in Science Bulletin.

**Why Science Bulletin?** This work aligns with the journal's mission of publishing scientifically significant research of **broad general interest across disciplines**. Time series anomaly detection is a methodological cornerstone of data-driven discovery in climate science, astronomy, seismology, materials science, and industrial safety — all core domains of Science Bulletin's readership. Our finding that the standard evaluation protocol in this field can reverse method rankings has direct implications for every scientist who uses anomaly detection to identify novel phenomena in time series data.

**What we found.** Through controlled experiments spanning five benchmark datasets (water treatment, chemical process, spacecraft telemetry, server monitoring), five detection methods, and three random seeds, we demonstrate that:

1. **Random-split evaluation reverses method rankings.** On the most widely-used TSAD benchmark (SWaT), the Spearman correlation between random-split and honest temporal-split rankings is ρ = −0.7 — a near-perfect inversion. The method ranked first under standard evaluation ranks last under honest evaluation. Across five datasets, descriptive rank correlations range from $\rho = -0.70$ (SWaT, near-perfect inversion) to $\rho = +0.80$ (SMAP, SMD). In two of five datasets, the correlation is negative or near-zero.

2. **The distortion is bidirectional and method-dependent.** Leakage gaps (ΔAUC = AUC_random − AUC_temporal) range from −0.25 (LSTM-AE deflated on SWaT) to +0.15 (Transformer-AE inflated on SMD). Simple statistical methods are typically inflated by random split; deep reconstruction-based methods are often penalized. This means published benchmark results cannot be corrected by a uniform "discount factor" — the distortion is qualitative, not just quantitative.

3. **The fix is one line of code.** We propose the Temporal-Split Protocol (TSP): split the raw time series before generating sliding windows, rather than after. No new metrics, no new benchmarks, no additional infrastructure required. TSP eliminates the single largest source of evaluation inflation we identified.

4. **Complex architectures add nothing to detection.** Under honest temporal evaluation, a zero-parameter Z-score baseline matches or exceeds deep learning methods on three of five datasets. Vision-language model features contribute exactly zero additional detection signal beyond simple statistical features. The sole performance driver is statistical feature engineering — not architecture complexity.

**Why this matters beyond computer science.** Anomaly detection on time series is used to identify earthquake precursors in seismic data, transient astronomical events in telescope surveys, early warning signals of climate tipping points, and degradation signatures in materials fatigue tests. If the evaluation protocol can systematically select the worst method for deployment, the downstream consequences span every scientific discipline that relies on anomaly detection for discovery. This is not a narrow computer science concern; it is a methodological crisis with broad scientific implications.

**Fit with Science Bulletin's editorial direction.** The journal has shown strong interest in paradigm-shift perspectives on scientific methodology, exemplified by Li & Guo's (2025) article "Paradigm shifts from data-intensive science to robot scientists." Our work offers a complementary perspective: before we can trust data-intensive scientific discovery, we must ensure the evaluation protocols that validate our computational tools are themselves trustworthy.

**Prior presentation.** This work has not been previously published or under consideration elsewhere. All code, data, and experimental results will be made publicly available upon acceptance.

**Suggested reviewers:**

1. Prof. John Paparrizos (The Ohio State University) — lead author of TSB-AD benchmark (NeurIPS 2024)
2. Prof. Eamonn Keogh (UC Riverside) — pioneer in time series evaluation methodology
3. Prof. M. Saquib Sarfraz (Karlsruhe Institute of Technology) — lead author of "Quo Vadis, Unsupervised TSAD?" (ICML 2024)
4. Prof. Thomas G. Dietterich (Oregon State University) — anomaly detection in scientific applications

**Excluded reviewers:** None.

We believe this work meets Science Bulletin's standards for scientific significance, broad interest, and methodological rigor. We welcome the opportunity to revise the manuscript based on editorial and reviewer feedback.

**Corresponding Author:**
Xicheng Zeng
College of Finance and Statistics, Hunan University
Lushan South Road, Yuelu District, Changsha 410082, Hunan, China
Email: ZXCHGQY@hnu.edu.cn

**Manuscript Statistics:**
- Article type: Article
- Total word count (main text): approximately 5,000 words
- Figures: 0 (all results in tables)
- Tables: 5 (in main text)
- References: 32
- Supplementary materials: 7 tables

**Conflict of Interest:** The author declares no competing interests.

**Statement of Responsibility:** Xicheng Zeng is solely responsible for the conceptualization, methodology, software development, investigation, formal analysis, visualization, and writing of this manuscript.

Sincerely,
Xicheng Zeng
College of Finance and Statistics, Hunan University
