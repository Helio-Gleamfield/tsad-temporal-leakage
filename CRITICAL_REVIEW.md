# 🔬 项目全面审视报告：实验严谨性 · 叙事逻辑 · 审稿人攻击面

> **审视对象**: "Temporal Data Leakage in TSAD Benchmarks" (manuscript_draft.md)
> **审视方法**: claim-by-claim evidence audit + narrative coherence analysis + reviewer attack surface mapping
> **审视标准**: Science Bulletin (IF 21.1, acceptance ~10%), 模拟审稿人视角
> **日期**: 2026-07-29

---

## 零、总评

**Bottom Line**: 这篇论文的核心发现（时序泄漏）是真的且重要的。但以当前状态投稿 Science Bulletin，**大概率收到 Major Revision → 如果改不好则 Reject**。问题不在实验数量不够，而在于**实验与 claim 之间的逻辑链条存在多处薄弱环节**，以及**两条叙事线（评估危机 vs 因果可解释性）的焊接点暴露**。

**一句话诊断**: 你有一个很好的 "evaluation crisis" 论文，但背上还背着半个 "ViCSynAD system" 论文的尸体——审稿人会闻到。

---

## 一、Claim-by-Claim 证据审计

### Claim 1: "Random split inflates TSAD AUC by 0.14–0.26"

**证据现状**:
- SWaT: ΔAUC = +0.2575 (LSTM-AE)
- TEP: ΔAUC = +0.0258 (LSTM-AE)
- Mean: +0.1416

**⚠️ 审稿人攻击面**:

| 攻击向量 | 严重程度 | 审稿人可能怎么问 |
|----------|---------|-----------------|
| **仅 2 个数据集** | 🔴 致命 | "You claim a 'systematic evaluation failure' based on n=2 datasets? TEP shows negligible leakage (+0.03). Your own data shows the problem may NOT be universal." |
| **仅测试了 LSTM-AE** | 🔴 致命 | "You only benchmarked LSTM-AE for the leakage gap. What about Anomaly Transformer? TimesNet? Modern methods may be more or less vulnerable — you can't generalize from one 2016 method." |
| **窗口参数固定** | 🟠 严重 | "L=256, S=64 gives 75% overlap. What happens at S=128 (50% overlap) or S=256 (0% overlap)? Is leakage linear with overlap? Without this analysis, you haven't proven the mechanism." |
| **只用 AUC-ROC** | 🟠 严重 | "AUC-ROC is known to overestimate on imbalanced data. Where are AUC-PR, VUS-PR, and adjusted F1? You criticize the field's metrics while using a flawed metric yourself." |
| **只显示均值，无方差** | 🟡 中等 | "No error bars, no standard deviation across seeds. Is ΔAUC=+0.26 stable or did one lucky seed produce this?" |

**🔧 修复方案**:
1. **最少 5 个数据集**: SWaT + TEP + (MSL or SMAP or SMD or PSM + 一个非 CPS 领域如金融/医疗) + 最好有一个人工合成数据集（可控制泄漏程度）
2. **最少 3 种方法**: LSTM-AE + Anomaly Transformer + 一个轻量方法 (Z-score baseline 足够)
3. **窗口重叠消融**: 固定 L=256, 变化 S ∈ {32, 64, 128, 256}，画 ΔAUC vs. overlap ratio 曲线
4. **多指标**: 报告 AUC-ROC + AUC-PR + VUS-PR

---

### Claim 2: "Under honest evaluation, simple methods match deep learning — detection is largely solved"

**证据现状**:
- SWaT temporal split: LSTM-AE 0.469, Z-score 0.433, IsoForest 0.457
- TEP temporal split: LSTM-AE 0.839, Z-score 0.843, IsoForest 0.752
- ViCSynAD component ablation: stats-only 0.588, stats+VLM 0.588, stats+recon 0.588

**⚠️ 审稿人攻击面**:

| 攻击向量 | 严重程度 | 审稿人可能怎么问 |
|----------|---------|-----------------|
| **"Detection is solved" 是严重过度声明** | 🔴 致命 | "AUC 0.47–0.67 is NOT 'solved.' If your detector misses 1/3 of anomalies (AUC 0.67), a water treatment plant operator would fire you. 'Simple methods match complex methods' ≠ 'detection is solved.'" |
| **Baseline 太弱** | 🔴 致命 | "LSTM-AE from 2016? Where is Anomaly Transformer (ICLR 2022, most cited TSAD method)? Where is TimesNet? Modern TSAD baselines would contextualize your claim." |
| **Z-score 配置未说明** | 🟠 严重 | "How exactly is Z-score computed? Per-sensor max deviation? Mean across sensors? This baseline drives your central claim but is a black box." |
| **数据集选择偏差** | 🟡 中等 | "TEP has 48.2% anomalies — of course Z-score works well, it's a binary classification problem at that point. Your finding may be a dataset artifact." |
| **ViCSynAD 消融只在 SWaT 上** | 🟡 中等 | "The VLM zero-contribution finding is from ONE dataset, ONE VLM (Qwen2-VL-7B 4-bit), ONE rendering pipeline. You can't claim 'VLMs don't help TSAD' from this." |

**🔧 修复方案**:
1. **修正 claim 的措辞**: "Detection is largely solved" → "Complex architectures offer no consistent advantage over statistical baselines on standard benchmarks under honest evaluation"
2. **增加至少 2 个现代 baseline**: Anomaly Transformer (必须), 再加一个 diffusion-based 或 GNN-based 方法
3. **详细记录 Z-score 实现**: 精确公式, per-sensor vs. aggregated, 阈值如何选择
4. **TEP 特殊性的诚实讨论**: 高异常密度 = 更容易检测 = Z-score 优势是预期的, 不代表所有场景

---

### Claim 3: "VLM features contribute zero signal to TSAD detection (ΔAUC = 0.000)"

**证据现状**:
- E1 ablation: 6-dim baseline → +enhanced stats (+0.092) → +random VLM features (+0.000) → +recon loss (+0.000)
- 使用 Qwen2-VL-7B 4-bit quantized

**⚠️ 审稿人攻击面**:

| 攻击向量 | 严重程度 | 审稿人可能怎么问 |
|----------|---------|-----------------|
| **仅一个 VLM / 一个渲染管线** | 🔴 致命 | "You tested ONE VLM (Qwen2-VL-7B at 4-bit precision — degraded!) with ONE visualization style. Would GPT-4o or Gemini Pro Vision, with different plot designs, give different results? Your finding is interesting but massively overclaimed." |
| **可能不是 VLM 不行, 是你的渲染不行** | 🟠 严重 | "Line charts at 448×448 with 51 overlapping sensors may be unreadable for ANY vision model. Did you try alternative visualizations (heatmaps, parallel coordinates, spectrograms)? The null result may reflect your rendering choices, not VLM capability." |
| **"Random VLM features" 是什么?** | 🟠 严重 | "What distribution generated the 'random' VLM features? Gaussian? Uniform? Matching the VLM feature distribution's mean and variance? Without this detail, the ablation is uninterpretable." |
| **4-bit 量化可能损害语义质量** | 🟡 中等 | "You used 4-bit quantization for VLM inference. Was the same VLM tested at full precision? 4-bit may preserve detection-relevant visual features poorly." |

**🔧 修复方案**:
1. **限缩 claim**: "VLM features contribute zero" → "Under our experimental configuration (Qwen2-VL-7B 4-bit, SWaT, line-chart rendering), VLM-derived features provided no measurable benefit to detection AUC. This does not rule out VLM utility for other datasets, rendering styles, or explanation tasks."
2. **增加渲染变体**: 测试 heatmap 渲染 vs line chart 渲染的 VLM 性能差异
3. **增加一个 API VLM 作为对照**: 用 GPT-4o API 对同一批窗口做 zero-shot 异常判断 (不需要 feature extraction, 直接问 "Is there an anomaly in this plot?")，比较 API-VLM 直接判断 vs 提取特征的 pipeline
4. **明确区分 "VLM 不能检测" vs "VLM 特征不能提升下游模型"** — 这是两个不同的 claim

---

### Claim 4: "Causal discovery with domain prior improves F1 by 25% (0.50 → 0.63)"

**证据现状**:
- PC algorithm, 12/51 sensors from SWaT
- No prior: F1=0.50, With P&ID prior: F1=0.63
- Full 51-var PC did not complete (>10 min, depth 4)

**⚠️ 审稿人攻击面**:

| 攻击向量 | 严重程度 | 审稿人可能怎么问 |
|----------|---------|-----------------|
| **仅 12/51 传感器** | 🔴 致命 | "Why only 12 sensors? Was this a random sample? Stratified by process stage? Cherry-picked to show the best improvement? Selecting 12 sensors and then claiming causal discovery 'works' is selection bias." |
| **仅一个因果发现算法** | 🟠 严重 | "Only PC algorithm tested. Where is GES? LiNGAM? NOTEARS? Different algorithms may benefit differently from domain prior. Without comparison, you can't claim domain prior universally helps causal discovery." |
| **仅一个数据集有因果 GT** | 🟠 严重 | "SWaT has P&ID ground truth, but TEP and MSL don't. You claim causal interpretability as the future but can't evaluate it on any other dataset. This makes ViCSynAD a SWaT-specific tool." |
| **"Domain prior improves F1" 可能是循环论证** | 🟠 严重 | "The P&ID prior was used to add known edges to the PC output. Of COURSE this improves F1 against the P&ID ground truth! You're measuring how well the prior recovers itself. The real question is: does PC-alone discover edges NOT in the prior? (You show recall=0.73 — good, but precision=0.38 is terrible.)" |
| **SHD=31 on 26 true edges is quite bad** | 🟡 中等 | "Your best method has SHD=31 for 26 true edges, meaning ~31 errors (missed + extra). Precision=0.456 means >50% of discovered edges are wrong. Is this good enough for root cause analysis?" |
| **没有与 non-causal baseline 比较** | 🔴 致命 | "How does causal RCA compare to correlation-based RCA? Granger causality? Simple sensor ranking by deviation magnitude? If correlation already finds the root cause 80% of the time, causal discovery's 25% improvement is less impressive." |

**🔧 修复方案**:
1. **传感器选择必须透明**: 记录传感器选择标准 (stratified by P1-P6 stage, 每个阶段 2 个传感器)，报告每个阶段的单独 F1
2. **增加 causal discovery 方法**: 至少 PC + GES (fast) 或 NOTEARS (scalable)
3. **增加 non-causal baseline**: 相关性排序, Granger causality, 简单 |deviation| 排序 → 作为 RCA baseline
4. **增加 RCA 端到端评估**: 对 36 个 SWaT 攻击场景, causal RCA 的 Top-K 命中率 (已知攻击目标传感器) vs. correlation baseline
5. **限缩 claim**: "Domain-informed causal discovery shows preliminary promise for TSAD root cause analysis, but remains limited by scalability and precision"

---

### Claim 5: "TSP should be adopted as community evaluation standard"

**证据现状**:
- Section 5 提出 5 条原则
- 无实验验证 TSP 本身的有效性

**⚠️ 审稿人攻击面**:

| 攻击向量 | 严重程度 | 审稿人可能怎么问 |
|----------|---------|-----------------|
| **"Protocol" 是 trivial 的** | 🔴 致命 | "Splitting time series temporally before windowing is common sense in any time series domain (finance, weather, etc.). Renaming a one-line code change as 'Temporal-Split Protocol' is pretentious. What's the novel contribution here?" |
| **TSP 自身有问题未讨论** | 🟠 严重 | "Temporal split assumes train and test periods are from the SAME distribution. Under concept drift (which is common in CPS), TSP may give misleading estimates. Your paper doesn't discuss this." |
| **TSP 没有被与现有方案比较** | 🟡 中等 | "Purge-based cross-validation (from finance) and blocked time series CV already exist. How is TSP different? Cite and compare." |
| **没有证明 TSP 的使用会改变 benchmark 排名** | 🟡 中等 | "You proved leakage exists. But does it change WHICH method is best? If all methods drop equally under temporal split, the ranking is preserved and TSP doesn't matter for model selection — only for absolute performance claims." |

**🔧 修复方案**:
1. **承认 triviality 并 reframe**: "TSP is not a novel algorithm — it is a **minimum evaluation standard** that corrects a systematic community oversight. Its simplicity is the point: no new metrics, no new benchmarks, one line of code changed."
2. **与 finance 领域的 look-ahead bias 做历史类比**: 金融学术界在 1980s 才广泛接受 walk-forward validation, 之前大量回测结果是虚假的。TSAD 现在处于同一位置。
3. **讨论 TSP 的局限**: concept drift, 需要足够的 test period 长度, 不适合极短时间序列
4. **与 Purged CV / Blocked CV 做区分**: TSP 是预处理协议 (split before windowing), Purged CV 是交叉验证策略 — 两者互补但不相同

---

## 二、叙事连贯性分析

### 当前的叙事结构

```
Section 1-3: "TSAD evaluation is broken because of temporal leakage"
             ↓ (逻辑跳跃)
Section 4:   "Therefore, causal interpretability is the future.
             Here's ViCSynAD, our causal TSAD system."
             ↓ (另一个跳跃)
Section 5:   "Therefore, use TSP."
             ↓
Section 6-7: Discussion & Conclusion
```

### 🔴 核心叙事问题: 两个半篇拼接

这篇论文里有**两篇不同的论文**在争夺主导权:

| | 论文 A: "The Leakage Crisis" | 论文 B: "ViCSynAD for Causal TSAD" |
|---|---|---|
| **核心贡献** | 发现并量化时序泄漏 | 提出因果可解释性框架 |
| **证据强度** | 中等 (2 datasets, 1 method) | 弱 (12/51 sensors, 1 dataset) |
| **对 Science Bulletin 的适配度** | 高 (范式反思, 跨领域影响) | 低 (SWaT 特定, 方法未成熟) |
| **审稿人接受概率** | 如果实验补强: 60-70% | 如果独立投稿: <20% |

**焊接点检查** (4.1 "detection is solved" → 4.2 "causal interpretability is the frontier"):
- 逻辑链: 检测被高估 → 诚实评估下简单方法够用 → 检测已达天花板 → 因此需要关注可解释性
- **问题**: 这个逻辑链条的第三步 ("检测已达天花板") 是未经证实的断言。AUC 0.47–0.67 远非 "天花板"，大量实际场景 (稀疏异常、早期故障、微弱信号) 检测仍极难。
- **此外**: 即使检测真的 "solved"，为什么一定是**因果**可解释性？注意力可视化、Shapley 值、概念漂移检测等都是替代方向。论文没有论证因果是唯一或最优的答案。

### 🔧 叙事修复方案

**方案 A (推荐): 纯评估危机论文 — 砍掉 ViCSynAD 系统部分**

```
Section 1:  Introduction — The TSAD evaluation crisis
Section 2:  Related Work — Evaluation flaws, temporal leakage, position papers
Section 3:  Temporal Leakage: Mechanism, Quantification, Multi-dataset Evidence
Section 4:  Consequences — Simple methods match DL, VLM features don't help,
           component ablation reveals stats as sole driver
Section 5:  TSP — A Minimum Evaluation Standard (analogy: look-ahead bias in finance)
Section 6:  Discussion — Implications for published results, reinterpreting the
           literature, what the field should do next
Section 7:  Conclusion
```

ViCSynAD 因果案例**完全移除**，或降级为 Discussion 中 "future directions" 的一个段落。

**方案 B (保留 ViCSynAD 但重新定位)**:

将论文定位为 "A Tale of Two Problems: Why TSAD Needs Honest Evaluation AND Causal Understanding"

```
Section 1-3: (同上) 时序泄漏的发现和量化
Section 4:  Beyond Detection — Two Open Problems
  4.1 检测评估必须诚实 (TSP)
  4.2 检测之外需要根因分析 (用 ViCSynAD 做案例)
  4.3 但因果方法目前很不成熟 (诚实列出局限)
Section 5:  Discussion
Section 6:  Conclusion
```

这种结构将 ViCSynAD 定位为 "illustration of future direction" 而非 "contribution"，大幅限缩对其的 claim。

**我的强烈推荐: 方案 A**。理由:
- 评估危机论文有清晰的单一叙事线
- ViCSynAD 的证据太弱，拖累整体可信度
- 一篇干净的 5,000 字 Perspective 比一篇 8,000 字的混合论文更容易被 Science Bulletin 接受
- 你可以在 Discussion 中 200 字提 ViCSynAD 作为 "ongoing work" — 审稿人不会认为不完整

---

## 三、"审稿人无法拒绝的理由" 构建

### 当前论文为什么可能被拒

1. **核心 claim 证据不足**: n=2 数据集, 1 种方法 → "systematic crisis" 是过度声明
2. **双叙事拉扯**: 评估危机 + ViCSynAD 系统 → 每个都证据不足, 但合在一起也不互补
3. **Triviality 指控**: TSP 是一行代码, 审稿人可能认为 "这不值得一篇 Science Bulletin"
4. **Baseline 选择**: 2025 年用 LSTM-AE 作为唯一 DL baseline → 审稿人认为你不了解领域
5. **因果部分未成熟**: 12/51 传感器, 精度 0.456 → 远不到可发表的程度

### "审稿人无法拒绝" 需要什么

一个审稿人无法拒绝的论文具有以下特征:

> **所有 major claim 都有过量的证据支撑, 以至于审稿人即使不同意你的 conclusion, 也无法否认你的 evidence**

具体到这篇论文:

| Claim | "无法拒绝" 标准 | 当前状态 |
|-------|----------------|---------|
| 时序泄漏存在且影响显著 | 5+ 数据集, 5+ 方法, 多窗口参数, 多指标, error bars | ❌ 2 数据集, 1 方法 |
| 泄漏程度因数据集特征而异 | 系统分析泄漏 vs. 异常密度/持续时间/维度等特征的关系 | ❌ 仅 text 讨论 |
| 简单方法匹配复杂方法 | 5+ 现代 baselines, 统计显著性检验, 跨数据集一致性 | ❌ 仅 LSTM-AE |
| VLM 不帮助检测 | 3+ VLM, 3+ 渲染方式, 2+ 数据集, 直接 API 调用 + 特征提取 | ❌ 1 VLM, 1 渲染 |
| TSP 有效且必要 | 证明 TSP 改变了方法排名, 讨论了 TSP 局限, 与现有方案比较 | ❌ 无 TSP 验证实验 |

### 最低可行 "无法拒绝" 配置

按优先级排序:

**🔴 P0 — 不做必被拒**:
1. **5+ 数据集上的时序泄漏量化**: SWaT, TEP, MSL/SMAP, SMD, PSM (都已在你的 dataset 目录中!)
2. **3+ DL 方法**: LSTM-AE + Anomaly Transformer (必须) + TimesNet or ModernTCN
3. **多指标**: AUC-ROC + AUC-PR + VUS-PR
4. **Error bars**: 所有实验 3+ seeds, 报告 mean ± std
5. **砍掉 ViCSynAD 系统部分** (或降为 Discussion 200 字)

**🟠 P1 — 强烈建议**:
6. **窗口重叠消融实验**: L=256 fixed, S ∈ {32, 64, 128, 256}
7. **Ranking stability analysis**: 在 temporal split 下方法排名是否与 random split 不同?
8. **TSP 局限讨论**: concept drift, 不适合短时序, 训练/测试分布变化

**🟡 P2 — 锦上添花**:
9. **合成数据实验**: 构造已知泄漏程度的数据, 验证 TSP 可以恢复真实性能
10. **VLM API 直接异常判断**: GPT-4o / Gemini 看图判断 → 与特征提取 pipeline 对比
11. **多领域扩展**: 金融/医疗/天文各一个数据集

---

## 四、实验设计审查: 每个实验服务于什么 claim?

按照用户提供的原则: "实验应该围绕贡献设计, 服务于文章主线。"

### 当前实验矩阵

| 实验 | 服务于哪个 claim? | 是否必要? | 证据强度 |
|------|------------------|----------|---------|
| E2: 多数据集泄漏量化 | Claim 1 (泄漏存在) | 🔴 必要 | ⚠️ 当前太弱 (n=2) |
| E1: VLM 消融 | Claim 3 (VLM 无用) | 🟠 有意思但可选 | ⚠️ 1 VLM 1 数据集 |
| E4: 因果发现 | Claim 4 (因果有用) | 🟡 如果方案A则不需要 | ⚠️ 12/51 传感器 |
| E3: CoT 在线 | Claim 4 辅助 (解释生成) | 🟡 如果方案A则不需要 | — (未执行) |
| E5: 人类评估 | Claim 4 辅助 (解释质量) | 🟡 如果方案A则不需要 | — (未执行) |

### 缺失的关键实验 (方案 A 下)

| 实验 | 服务于哪个 claim? | 为什么缺失是致命的 |
|------|------------------|------------------|
| **多 DL 方法泄漏比较** | Claim 1+2 | 不能从 LSTM-AE 推广到所有 DL 方法 |
| **窗口重叠消融** | Claim 1 (泄漏机制) | 声称重叠导致泄漏但从未验证 S 的影响 |
| **排名稳定性** | Claim 5 (TSP 必要) | 如果 random vs temporal 下方法排名相同, TSP 不重要 |
| **统计显著性** | 所有 claims | 所有数字无 error bars, 无法判断是否显著 |
| **多指标一致性** | Claim 1 | 只用 AUC-ROC 而批评领域指标问题 → hypocritical |

---

## 五、具体修改建议 (按优先级)

### 🔴 P0-1: 砍掉 ViCSynAD, 聚焦评估危机 (影响: 叙事)

**操作**:
- 删除 Section 4.3 (ViCSynAD case study) 全部内容
- 将 Section 4.1-4.2 精简为 ~500 字, 重命名为 "Implications: What Honest Evaluation Reveals"
- 原 ViCSynAD 相关内容压缩为 Discussion 中 1 段 (~200 字), 定位于 "ongoing work"
- Title 改为: "Temporal Data Leakage in Time Series Anomaly Detection Benchmarks: A Systematic Evaluation" (删除 "the Case for Causal Interpretability")

**效果**: 论文从 "半篇 evaluation + 半篇 system" 变为 "完整的一篇 evaluation"。叙事线干净。

### 🔴 P0-2: 扩展数据集 + Baseline (影响: 核心证据)

**当前**: SWaT + TEP, LSTM-AE only
**目标**: SWaT + TEP + MSL + SMAP + SMD, LSTM-AE + Anomaly Transformer + Z-score + IsoForest

**具体操作**:
- MSL/SMAP/SMD 已在你的 dataset 目录 (mTSBench) → 用现有 processor.py 加载
- Anomaly Transformer 有公开 GitHub (thuml/Anomaly-Transformer) → 可以直接 clone 并在你的数据上跑
- 所有实验用 `temporal_fair_compare.py` 的框架, 但增加:
  - `--methods lstm_ae,anomaly_transformer,zscore,isoforest`
  - `--seeds 3` (3 次随机 seed, 报告 mean ± std)
  - `--metrics auc_roc,auc_pr,vus_pr`

**估计时间**: 1-2 天 (大部分时间在跑实验)

### 🔴 P0-3: 窗口重叠消融 (影响: 机制验证)

**当前**: 声称 "重叠导致泄漏" 但从未验证 S 参数的影响
**目标**: L=256, S ∈ {32, 64, 128, 256}, 画 ΔAUC vs. overlap ratio 图

**预期结果**: S=256 (0% overlap) → ΔAUC ≈ 0; S=32 (87.5% overlap) → ΔAUC 最大
**如果没有这个趋势**: 那泄漏机制可能不是窗口重叠 → 整个论文需要重新审视

### 🟠 P1-1: 排名稳定性分析 (影响: TSP 的 raison d'être)

**问题**: 如果所有方法在 temporal split 下 AUC 都下降相同的量, 方法排名不变 → TSP 只是 "rescale" 了绝对性能, 不影响模型选择

**实验**:
- 在 5 个数据集上跑 5 种方法, 分别用 random split 和 temporal split
- 报告 Spearman rank correlation between random ranking and temporal ranking
- 如果 rank correlation < 0.8 (排名有明显变化) → TSP 真的很重要
- 如果 rank correlation > 0.95 → TSP 不重要, 只影响绝对数字

### 🟠 P1-2: TSP 与 Purged CV 的比较讨论

**当前**: TSP 被提出为 "新协议", 但金融领域早有 similar ideas (purged cross-validation, embargoed CV)
**修复**: 在 Related Work 和 Discussion 中引用并区分:
- Purged CV (De Prado, 2018): 从训练集中移除与测试集有时间重叠的样本
- TSP: 在 windowing 之前就做 temporal split → 从根本上消除了重叠的可能性
- 两者互补: TSP 是预处理协议, Purged CV 是交叉验证策略

---

## 六、最终评估矩阵

| 维度 | 当前评分 | 目标评分 | 关键差距 |
|------|---------|---------|---------|
| **核心 claim 证据强度** | 4/10 | 8/10 | 数据集数量, baseline 多样性, error bars |
| **叙事连贯性** | 5/10 | 9/10 | 双叙事冲突, 砍掉 ViCSynAD |
| **实验严谨性** | 3/10 | 8/10 | 多指标, 统计检验, 消融实验 |
| **贡献清晰度** | 4/10 | 9/10 | TSP triviality, 聚焦评估危机 |
| **Science Bulletin 适配度** | 6/10 | 9/10 | 范式反思 vs 系统论文 |
| **"无法拒绝" 程度** | 3/10 | 8/10 | P0 实验缺失是致命弱点 |

### 关键决策点

**问自己**: 这篇论文的核心贡献是什么?

**如果答案是** "我们发现时序泄漏并提出了诚实评估标准" → **走方案 A, 砍 ViCSynAD, 补实验**
**如果答案是** "我们提出了 ViCSynAD 因果可解释性框架" → **换投期刊, 投 KDD/ICML/NeurIPS**

**不能两者兼得。** 当前 manuscript 试图两者兼得 — 这是最危险的策略, 因为审稿人会用较弱的那个故事攻击较强的那个故事。

---

## 七、执行路线图

```
Week 1:
  Day 1: 砍掉 ViCSynAD 部分 → 重构 Manuscript 结构 → 确定新叙事
  Day 2-3: P0-2 扩展实验 (5 datasets × 5 methods × 3 seeds)
  Day 4: P0-3 窗口重叠消融

Week 2:
  Day 1-2: P1-1 排名稳定性分析
  Day 3-4: 更新所有 Tables + Figures
  Day 5: 重写 Abstract + Introduction (匹配新聚焦)

Week 3:
  Day 1-2: 英文润色 (scientific-writing skill)
  Day 3: 格式检查 + Cover Letter
  Day 4: 投稿
```

---

*本审视使用 scientific-critical-thinking + peer-review 双技能框架，对项目进行了 claim-by-claim 证据审计、叙事连贯性分析、和 "审稿人无法拒绝" 标准映射。所有批评都是建设性的，指向具体可操作的改进。*
