# Related Work Research --- Positional Distillation Paper
## Iterations 1-5 --- Full Research Log
**Last Updated:** 2026-03-18

---

## Iteration 1: Initial Bibliography & Paper Collection

(See git history for original iteration 1 content)

---

## Iteration 2: Verification & Gap Filling

### Verified Papers

All arxiv IDs from iteration 1 have been verified via web search. Key corrections:

1. **Wang et al. (NeurIPS 2025) "Beyond the 80/20 Rule"** --- Verified. arXiv:2506.01939. Lead author is Shenzhi Wang (not "Yiming Wang"). Full author list: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin. Confirmed NeurIPS 2025.

2. **Huang et al. SelecTKD** --- Verified. arXiv:2510.24021. Authors: Haiduo Huang, Jiangcheng Song, Yadong Zhang, Pengju Ren.

3. **Tavor et al. "Rethinking Selective KD"** --- Verified. arXiv:2602.01395. Authors: Almog Tavor, Itay Ebenspanger, Neil Cnaan, Mor Geva.

4. **Kim & Baek. TSD-KD** --- Verified. arXiv:2603.13260. Authors: Minsang Kim, Seung Jun Baek.

5. **Xie et al. AdaKD** --- Verified. arXiv:2510.11615. Authors: Xurong Xie, Zhucun Xue, Jiafu Wu, Jian Li, Yabiao Wang, Xiaobin Hu, Yong Liu, Jiangning Zhang. Accepted AAAI 2026.

6. **Jang et al. Veto** --- Verified. arXiv:2601.07155. Authors: Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggu Lim, Taesup Kim. Seoul National University.

7. **Chen et al. "Distilling the Essence"** --- Verified. arXiv:2512.21002. Authors: Wei-Rui Chen, Vignesh Kothapalli, Ata Fatahibaarzi, Hejian Sang, Shao Tang, Qingquan Song, Zhipeng Wang, Muhammad Abdul-Mageed.

8. **Xu et al. SKD (Speculative KD)** --- Verified. arXiv:2410.11325. Authors: Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, Tomas Pfister. Confirmed ICLR 2025.

9. **Wu et al. AKL** --- Verified. arXiv:2404.02657. Authors: Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, Ngai Wong. Confirmed COLING 2025.

10. **Li et al. preplan-and-anchor** --- Verified. arXiv:2510.13554. Authors: Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, Bo Zheng, Junchi Yan.

11. **Yan et al. DASD** --- Verified. arXiv:2601.09088. Authors: Shaotian Yan, Kaiyuan Liu, Chen Shen, Bing Wang, Sinan Fan, Jun Zhang, Yue Wu, Zheng Wang, Jieping Ye.

12. **Li et al. Sequence Length Warmup** --- Verified. arXiv:2108.06084. Authors: Conglong Li, Minjia Zhang, Yuxiong He. Published NeurIPS 2022.

### Newly Added Papers

**DPO (Rafailov et al., NeurIPS 2023)**
- **Full citation:** Rafailov, Sharma, Mitchell, Manning, Ermon, Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. arXiv:2305.18290.
- Relevant as background for alignment; reverse KL connection to mode-seeking behavior.

**Model Collapse (Shumailov et al., Nature 2024)**
- **Full citation:** Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal. "AI models collapse when trained on recursively generated data." Nature 631, 755--759, 2024. DOI:10.1038/s41586-024-07566-y
- Relevant: on-policy generation with iterative training risks a form of distribution collapse; our full-seq repetition pattern is related.

**Scheduled Sampling (Bengio et al., NeurIPS 2015)**
- **Full citation:** Bengio, Vinyals, Jaitly, Shazeer. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." NeurIPS 2015. arXiv:1506.03099.
- Relevant: curriculum from teacher-forced to free-running generation; our progressive schedule is analogous.

**ImitKD (Lin et al., EMNLP 2020)** --- Already in bib. Verified: arXiv:2009.07253. Authors: Alexander Lin, Jeremy Wohlwend, Howard Chen, Tao Lei.

**KD-LoRA (Azimi et al., NeurIPS ENLSP Workshop 2024)** --- Verified. arXiv:2410.20777. Authors: Rambod Azimi, Rishav Rishav, Marek Teichmann, Samira Ebrahimi Kahou. Note: published at NeurIPS ENLSP workshop, not main PMLR.

**Dataset Decomposition / Variable Sequence Length Curriculum (Pouransari et al., NeurIPS 2024)** --- Verified. arXiv:2405.13226. Authors: Hadi Pouransari, Chun-Liang Li, Jen-Hao Rick Chang, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Oncel Tuzel. Apple.

**RLAD (arXiv:2602.22495)** --- Verified. "Reinforcement-aware Knowledge Distillation for LLM Reasoning." Lead author: Zhaoyang Zhang.

### Search for "Prefix Distillation" / "Prefix Loss"

Searched for "prefix distillation", "prefix loss", "early token distillation", "position-weighted loss" in the context of LLM KD. **No prior work uses the exact concept of restricting distillation loss to the first N positional tokens.** The closest works are:
- **"Distilling the Essence"** (Chen et al., 2025): truncates sequences but from a data-efficiency perspective, not a principled positional analysis.
- **SE-KD** (Tavor et al., 2026): selects token positions by student entropy, not by absolute position.
- **SelecTKD** (Huang et al., 2025): selects tokens by teacher verification, not position.

This confirms our contribution is novel: using absolute position as the selection criterion for distillation loss, motivated by empirical analysis of KL concentration.

---

## Iteration 3: Concurrent Work Differentiation

### The 6 Closest Concurrent/Related Works

#### 1. "Rethinking Selective Knowledge Distillation" (Tavor et al., Feb 2026) --- SE-KD
**What they do:** Systematically study selective KD along position, class, and sample axes. Propose student-entropy-guided position selection (SE-KD).
**Difference from us:**
- They select positions *adaptively* based on student entropy; we use *fixed* positional cutoff.
- They don't analyze WHY certain positions matter---no KL concentration analysis.
- They focus on efficiency (70% wall-time reduction); we focus on understanding the mechanism and showing early tokens drive the cascade.
- Our cascade effect analysis (showing changes in first 50 tokens propagate to 1500+ character responses) is unique.
**Our advantage:** Simpler (one hyperparameter N), interpretable (motivated by KL analysis), eliminates instability.

#### 2. "Explain in Your Own Words: TSD-KD" (Kim & Baek, Mar 2026)
**What they do:** Token-selective dual KD combining indirect (preference ranking) and direct (entropy-based token selection) distillation.
**Difference from us:**
- They select tokens by relative confidence between teacher and student; we select by position.
- They use a complex dual-objective framework; we propose a minimal one-line change.
- They don't study the instability problem or cascade effect.
- Different paradigm: they combine preference + distillation; we are pure on-policy distillation.
**Our advantage:** Simplicity, instability analysis, positional KL motivation.

#### 3. "Distilling the Essence: Sequence Truncation" (Chen et al., Dec 2025)
**What they do:** Study compute-quality tradeoffs as function of sequence length. Show 50% truncation retains ~91% performance.
**Difference from us:**
- They truncate *training data sequences* (SFT on shorter teacher outputs); we apply loss masking on student-generated on-policy sequences.
- They study off-policy (supervised) distillation; we study on-policy distillation with reverse KL.
- No analysis of WHY truncation works (KL concentration).
- No cascade effect analysis.
**Our advantage:** On-policy setting, principled KL motivation, cascade analysis, instability elimination.

#### 4. "Stable On-Policy Distillation via Veto" (Jang et al., Jan 2026)
**What they do:** Address on-policy KD instability via geometric bridge distribution in logit space. Suppress harmful gradients on low-confidence tokens.
**Difference from us:**
- They modify the *target distribution* (logit mixing); we modify the *loss scope* (positional masking).
- Complementary approaches: Veto could be combined with positional distillation.
- We additionally provide efficiency gains (3-7x in generation phase).
- Our approach is architecturally simpler (no new hyperparameters beyond N).
**Our advantage:** Simpler, more efficient, provides both stability AND efficiency. Could be combined with Veto.

#### 5. "Beyond the 80/20 Rule" (Wang et al., NeurIPS 2025)
**What they do:** Show ~20% of CoT tokens have high entropy and steer reasoning paths. RL gradients on only these tokens match full performance.
**Difference from us:**
- They study RL (RLVR); we study distillation.
- They identify tokens by entropy; we identify by position.
- Our key finding: early positions correlate with high-entropy strategy tokens, providing a simpler proxy.
- Different mechanism: RL credit assignment vs. KD loss masking.
**Relationship:** Complementary. Their entropy-based finding provides theoretical grounding for why positional selection works---early tokens tend to be the high-entropy decision points.

#### 6. SelecTKD (Huang et al., Oct 2025) and AdaKD (Xie et al., AAAI 2026)
**What they do:** Token-level adaptive distillation. SelecTKD uses teacher verification (propose-and-verify). AdaKD uses learning stability metrics to adaptively weight tokens.
**Difference from us:**
- Both require per-token computation of importance/difficulty metrics; we use zero-cost positional selection.
- Both are "plug-and-play" complexity; ours is a single hyperparameter.
- Neither addresses on-policy instability.
**Our advantage:** Zero overhead for token selection, simpler, addresses instability.

### Summary of Unique Contributions

1. **Empirical discovery**: KL divergence concentrates at early positions (first 50 tokens = 26% of KL signal).
2. **Cascade effect**: Changing first 50 tokens changes the entire 1500+ character response.
3. **Instability elimination**: Positional loss eliminates full-sequence degeneration (boxed repetition).
4. **Simplicity**: One-line modification, one hyperparameter N.
5. **Efficiency**: 3-7x compute reduction in generation phase.
6. **Progressive schedule**: Curriculum over loss scope is novel.

No concurrent work provides all of these together.

---

## Iteration 4: BibTeX

Complete bibtex entries saved to `/CGLab/ziheng/projects/dft-distill/paper/references_extended.bib`.

---

## Iteration 5: Related Work Section

Complete LaTeX related work section saved to `/CGLab/ziheng/projects/dft-distill/paper/related_work_section.tex`.

### Narrative Structure
1. **Knowledge Distillation for Language Models** (~0.4 pages)
   - Hinton et al. -> Kim & Rush -> MiniLLM -> GKD -> DistiLLM -> f-DISTILL -> AKL -> SKD
2. **Token-Level Importance in Learning** (~0.4 pages)
   - Wang et al. 80/20 -> preplan-and-anchor -> Singh pruning -> SelecTKD -> AdaKD -> SE-KD -> TSD-KD
3. **Training Stability and Sequence-Level Issues** (~0.3 pages)
   - Degeneration -> model collapse -> Veto -> Distilling the Essence -> DASD -> exposure bias
4. **Curriculum Learning** (~0.2 pages)
   - Bengio curriculum -> scheduled sampling -> sequence warmup -> dataset decomposition
5. **Mathematical Reasoning Models** (~0.2 pages)
   - Qwen2.5-Math -> Qwen3 -> DeepSeek-R1 -> MATH/MATH-500

---

## Complete Paper List (Verified)

| # | Paper | Venue | arXiv ID | Status |
|---|-------|-------|----------|--------|
| 1 | Hinton et al. 2015, KD | NIPS Workshop 2015 | 1503.02531 | Verified |
| 2 | Kim & Rush 2016, Seq-KD | EMNLP 2016 | 1606.07947 | Verified |
| 3 | Wen et al. 2023, f-DISTILL | ACL 2023 | 2307.15190 | Verified |
| 4 | Agarwal et al. 2024, GKD | ICLR 2024 | 2306.13649 | Verified |
| 5 | Gu et al. 2024, MiniLLM | ICLR 2024 | 2306.08543 | Verified |
| 6 | Ko et al. 2024, DistiLLM | ICML 2024 | 2402.03898 | Verified |
| 7 | Wu et al. 2025, AKL | COLING 2025 | 2404.02657 | Verified |
| 8 | Zhang et al. 2024, DSKD | EMNLP 2024 | 2406.17328 | Verified |
| 9 | Xu et al. 2025, SKD | ICLR 2025 | 2410.11325 | Verified |
| 10 | Lin et al. 2020, ImitKD | EMNLP 2020 | 2009.07253 | Verified |
| 11 | Wang et al. 2025, 80/20 | NeurIPS 2025 | 2506.01939 | Verified |
| 12 | Singh & Hakkani-Tur 2026 | arXiv preprint | 2601.03066 | Verified |
| 13 | Li et al. 2025, preplan-anchor | arXiv preprint | 2510.13554 | Verified |
| 14 | Huang et al. 2025, SelecTKD | arXiv preprint | 2510.24021 | Verified |
| 15 | Xie et al. 2026, AdaKD | AAAI 2026 | 2510.11615 | Verified |
| 16 | Tavor et al. 2026, SE-KD | arXiv preprint | 2602.01395 | Verified |
| 17 | Kim & Baek 2026, TSD-KD | arXiv preprint | 2603.13260 | Verified |
| 18 | Chen et al. 2025, Distilling Essence | arXiv preprint | 2512.21002 | Verified |
| 19 | Jang et al. 2026, Veto | arXiv preprint | 2601.07155 | Verified |
| 20 | Yan et al. 2026, DASD | arXiv preprint | 2601.09088 | Verified |
| 21 | RLAD 2026 | arXiv preprint | 2602.22495 | Verified |
| 22 | Holtzman et al. 2020 | ICLR 2020 | 1904.09751 | Verified |
| 23 | Shumailov et al. 2024 | Nature 2024 | 2305.17493 | Verified |
| 24 | Bengio et al. 2009, Curriculum | ICML 2009 | -- | Verified |
| 25 | Bengio et al. 2015, Sched. Samp. | NeurIPS 2015 | 1506.03099 | Verified |
| 26 | Li et al. 2022, Seq Warmup | NeurIPS 2022 | 2108.06084 | Verified |
| 27 | Pouransari et al. 2024, DataDecomp | NeurIPS 2024 | 2405.13226 | Verified |
| 28 | Hu et al. 2022, LoRA | ICLR 2022 | 2106.09685 | Verified |
| 29 | Rafailov et al. 2023, DPO | NeurIPS 2023 | 2305.18290 | Verified |
| 30 | Wang et al. 2024, f-DPO | ICLR 2024 | 2309.16240 | Verified |
| 31 | Wei et al. 2022, CoT | NeurIPS 2022 | 2201.11903 | Verified |
| 32 | DeepSeek-R1, 2025 | arXiv preprint | 2501.12948 | Verified |
| 33 | Qwen2.5-Math, 2024 | arXiv preprint | 2409.12122 | Verified |
| 34 | Qwen2.5, 2024 | arXiv preprint | 2412.15115 | Verified |
| 35 | Qwen3, 2025 | arXiv preprint | 2505.09388 | Verified |
| 36 | Hendrycks et al. 2021, MATH | NeurIPS 2021 | 2103.03874 | Verified |
| 37 | Lightman et al. 2023, MATH-500 | arXiv preprint | 2305.20050 | Verified |
| 38 | Xu et al. 2024, KD Survey | arXiv preprint | 2402.13116 | Verified |
| 39 | Kwon et al. 2023, vLLM | SOSP 2023 | 2309.06180 | Verified |
| 40 | He & Zhang 2021, Exposure Bias | EMNLP 2021 | -- | Verified |
