# Innovation Notes: Improved Reduced Auditory Filter Banks

**Purpose:** Draft notes for the report / thesis chapter.  
**Paper baseline:** Fayyazi & Sameti, *Exploiting auditory filter models as interpretable convolutional frontends* (SGR_AFM).  
**Code:** `afm_sgr/reduced_new.py`, `run_new_reduced.py`, `new_reduced_pipeline.bat`  
**Results folder:** `results_github/reduced_{A,B,C,D,AD}_timit` (+ `_finetune`) and `results_github/new_reduced_comparison/`

---

## 1. One-sentence claim (for abstract / contribution list)

We propose several alternatives to the paper’s Mean-shift pole reduction that (i) respect auditory frequency resolution, (ii) keep filters that matter for gender classification, and/or (iii) refine the reduced bank with distillation, while **matching the paper’s Reduced model on filter count** so accuracy comparisons are fair.

---

## 2. Motivation (why innovate here)

The paper’s Reduced model (Sec. 3.2.2 / 4.3) works as follows:

1. Train a Cascade Gaussian frontend (80 filters × 2 stages → **160 poles**).
2. Map each `(f1, f2)` band to a complex pole  
   \(x = e^{-\sigma + j\omega_0}\), with  
   \(\omega_0 = 2\pi(f_1+f_2)/(2f_s)\) and \(\sigma = 2\pi(f_2-f_1)/f_s\).
3. Cluster poles with **MeanShift(bandwidth = 0.079)** on \((\mathrm{Re}\,x,\,\mathrm{Im}\,x)\).
4. Replace each cluster by the mean `(f1, f2)` and rebuild a **frozen sinc²** filter bank.
5. Retrain only the CNN body on TIMIT.

### Limitations of that baseline

| # | Limitation | Why it matters for gender SID |
|---|------------|-------------------------------|
| 1 | Clustering is **task-agnostic** | Filters that carry F0 / formant cues can be merged with less useful neighbours. |
| 2 | Distance is **Euclidean on the complex plane**, with a **fixed** radius | ~200 Hz resolution everywhere. At low frequencies (male ≈ 85–180 Hz, female ≈ 165–255 Hz) that is coarse; at high frequencies it is overly fine. |
| 3 | Each Cascade filter is split into **two independent poles** | The real frontend response is the cascade of both stages; clustering poles ignores that. |
| 4 | Reduced cutoffs are **frozen** after clustering | No recovery if cluster centres are slightly wrong. |
| 5 | Filter count \(K\) is an **indirect** side-effect of bandwidth | Hard to compare methods at fixed complexity. |

Our variants address (1)–(4). For (5) we always set \(K\) equal to the number of filters produced by the paper’s Mean-shift run on the same Cascade checkpoint.

---

## 3. Common experimental protocol

- **Teacher / source of filters:** trained Cascade TIMIT model (`results_github/cascade_timit/best.pt`).
- **Reduced frontend type:** sinc², same kernel family as the paper’s Reduced model.
- **Filter count:** \(K =\) `#filters` of paper Mean-shift Reduced (same Cascade).
- **TIMIT:** 50 epochs, RMSprop, same chunking / augmentation as the main pipeline.
- **Fine-tune:** 20 epochs on LibriSpeech (dev-clean split used in this repo), then eval on RAVDESS and VoxCeleb1-test.
- **Metrics:** utterance accuracy, CER, FER (summed log-prob decision, as in the authors’ SIDTrainer).
- **Baselines in the comparison table:** Cascade (80 filters) + paper Reduced (Mean-shift).

Variants:

| ID | Selection of filters | Training of Reduced |
|----|----------------------|---------------------|
| **A** | Task-aware ERB-warped weighted mean-shift | Frozen sinc² + CE |
| **B** | Learnable L0 (hard-concrete) gates | Frozen sinc² + CE |
| **C** | Agglomerative clustering of effective Cascade responses | Frozen sinc² + CE |
| **D** | Paper Mean-shift centres | **Unfrozen** f1/f2 + CE + **KD** from Cascade |
| **AD** | Same selection as A | Unfrozen f1/f2 + CE + KD (proposed combo) |

---

## 4. Method A — Task-aware, ERB-warped pole clustering

### Idea
Keep the paper’s “poles → cluster → sinc²” story, but change **where** and **how strongly** poles attract each other.

### Steps

1. Extract both stages’ cutoffs from the Cascade frontend → 160 bands `(f1, f2)`.
2. Represent each pole in an **auditory feature space** (not the complex plane):
   - \(u = \mathrm{ERB\text{-}rate}(f_c)\), \(f_c = (f_1+f_2)/2\)
   - \(v = \mathrm{ERB\text{-}rate}(f_2) - \mathrm{ERB\text{-}rate}(f_1)\) (width in ERBs)
   - Glasberg–Moore ERB-rate: \(\mathrm{ERB\text{-}rate}(f) = 21.4\log_{10}(1 + 0.00437\,f)\).
3. Compute a **per-filter importance** on TIMIT validation chunks (gender-balanced probe):
   - **Taylor / attribution:** \(\mathbb{E}\big[|\sum_t a_{i,t}\,\partial L/\partial a_{i,t}|\big]\) on the Cascade filter activation \(a_i\) (first-order effect of removing filter \(i\)).
   - **Fisher score** of the filter’s log-energy between male and female chunks:  
     \((\mu_m-\mu_f)^2 / (\sigma_m^2+\sigma_f^2)\).
   - Importance \(w_i\) = average of the two scores after normalising each to mean 1; duplicated for the two stages of each Cascade filter.
4. Run **weighted Gaussian mean-shift** on \((u,v)\) with weights \(w_i\). Bisect the bandwidth until the number of clusters equals \(K\).
5. Cluster centre cutoffs = **weighted** means of member `(f1, f2)`.
6. Build frozen sinc² bank; train CNN body as usual.

### Why this is novel relative to the paper
- Perceptual distance (ERB) instead of complex-plane Euclidean.
- Task weights so gender-relevant poles dominate cluster centres.
- Explicit control of \(K\) (fair comparison).

### Suggested report wording
> “We replace Euclidean Mean-shift on cascade poles with a weighted mean-shift in ERB-rate space, where each pole is weighted by a combination of first-order loss attribution and a gender Fisher score of filter energy. The bandwidth is chosen so that the Reduced bank has the same size as the paper’s Mean-shift bank.”

### Key figure for the report
`results_github/reduced_A_timit/figures/A_taskaware_erb_clusters.png`  
(poles in ERB space; marker size = importance; colour = cluster)

---

## 5. Method B — Learnable L0 gates (hard-concrete)

### Idea
Do not cluster poles after training. Instead, **learn which of the 80 Cascade filters to keep**, with a sparsity budget \(K\).

### Steps

1. Attach a **hard-concrete gate** (Louizos et al., 2018) \(g_i \in [0,1]\) to each Cascade filter output: \(a_i \leftarrow g_i\,a_i\).
2. Fine-tune Cascade + gates for a short schedule (default **10 epochs**) with  
   \(\mathcal{L} = \mathrm{CE} + \lambda\,\mathrm{ReLU}(\sum_i P(g_i\text{ open}) - K)/N\).
3. Keep the \(K\) filters with largest \(\log\alpha_i\) (gate open probability).
4. Convert each kept filter’s **effective** magnitude response to a band `(f1, f2)` (contiguous −3 dB region around the peak).
5. Build frozen sinc² bank at those bands; train CNN body from scratch (50 epochs).

### Why this is novel
- End-to-end, task-driven selection; no hand-chosen complex-plane bandwidth.
- Directly optimises for gender CE under a filter budget.

### Suggested report wording
> “We prune the Cascade bank with differentiable hard-concrete gates under a target sparsity of \(K\) filters, then instantiate a reduced sinc² bank from the magnitude responses of the surviving filters.”

### Key figure
`results_github/reduced_B_timit/figures/B_l0_gates.png`  
(open probabilities; kept filters highlighted)

### Reference to cite
Louizos, Welling, Kingma — *Learning Sparse Neural Networks through \(L_0\) Regularization* (ICLR 2018).

---

## 6. Method C — Clustering on effective Cascade responses

### Idea
Fix the paper’s split of each Cascade filter into two poles: cluster **filters**, not poles, using the **true** combined frequency response.

### Steps

1. For each of the 80 Cascade kernels, compute the magnitude of the FFT of the effective impulse response (both Gaussian stages already convolved in the frontend).
2. L2-normalise responses; form pairwise **cosine distance**.
3. **Agglomerative clustering** (average linkage, precomputed distance) with `n_clusters = K`.
4. For each cluster, average the normalised responses and extract `(f1, f2)` via the −3 dB contiguous band around the peak.
5. Frozen sinc² + CE training as usual.

### Why this is novel
- Matches the physical Cascade filter (product of stages in frequency ≈ cascade in time).
- Similarity is defined on responses that the CNN actually sees, not on pole coordinates.

### Suggested report wording
> “Instead of clustering the 160 stage-wise poles, we cluster the 80 Cascade filters by cosine similarity of their effective magnitude responses and replace each cluster by a single sinc² band matched to the cluster prototype.”

### Key figure
`results_github/reduced_C_timit/figures/C_response_clusters.png`

---

## 7. Method D — Distillation + trainable cutoffs (training innovation)

### Idea
Keep the **paper’s Mean-shift filter centres**, but improve how the Reduced model is trained.

### Changes vs paper Reduced

1. **Knowledge distillation** from the frozen Cascade teacher:  
   \(\mathcal{L} = (1-\alpha)\,\mathrm{CE}(y,\mathrm{student}) + \alpha\,T^2\,\mathrm{KL}(\mathrm{teacher}_T \,\|\, \mathrm{student}_T)\)  
   Defaults: \(\alpha=0.5\), \(T=4\).
2. **Do not freeze** `f1`/`f2` (and amplitude) of the Reduced sinc² bank; allow small adjustments during the 50 TIMIT epochs.

### Why this is novel
- Orthogonal to selection: can be combined with A/B/C (we implement **AD** = A selection + D training).
- Addresses the “frozen, slightly wrong centres” limitation without changing the paper’s clustering.

### Suggested report wording
> “We retain the authors’ Mean-shift Reduced bank but train it with Hinton-style distillation from the Cascade teacher and allow the reduced cutoffs to adapt, recovering capacity lost by freezing the frontend.”

### Reference to cite
Hinton, Vinyals, Dean — *Distilling the Knowledge in a Neural Network* (2015).

---

## 8. Proposed combination (AD)

**Selection:** Method A (task-aware ERB).  
**Training:** Method D (KD + trainable f1/f2).

This is the strongest “ours” candidate for the contribution paragraph:  
*auditory + task-aware reduction*, then *distilled refinement of the reduced bank*.

---

## 9. What to put in the Results section (after runs finish)

Fill from `results_github/new_reduced_comparison/comparison.csv`:

| Model | #filters | TIMIT Acc | RAVDESS Acc | VoxCeleb1 Acc |
|-------|----------|-----------|-------------|---------------|
| Cascade (teacher) | 80 | | | |
| Paper Reduced (Mean-shift) | \(K\) | | | |
| A (ours) | \(K\) | | | |
| B (ours) | \(K\) | | | |
| C (ours) | \(K\) | | | |
| D (ours) | \(K\) | | | |
| AD (ours) | \(K\) | | | |

### Claims you can make if numbers support them

1. At **equal** \(K\), A/B/C beat paper Reduced on TIMIT and/or transfer sets.
2. D / AD improve over frozen Mean-shift / frozen A.
3. Low-frequency filters are better preserved under A (show ERB scatter + filter-bank plots).
4. B’s open probabilities concentrate on low-frequency / high-Fisher filters (qualitative).

### Extra plots already produced per run

- `reduced_filter_bank.png` — sinc² responses after reduction  
- Standard paper analyses: accumulative magnitude, \(f_c\) histogram, \(Q\), t-SNE, learned `f1`/`f2`  
- Method-specific plots listed above  

---

## 10. Limitations / honesty for the report

- Fine-tune uses **LibriSpeech dev-clean** (and a small test-clean slice for val) because `train-clean-360` was not fully available in this project; the paper uses train-clean-360. State this clearly when comparing to paper Tables 7–8.
- \(K\) is taken from one Mean-shift run on our Cascade checkpoint; if that Cascade differs slightly from the authors’, \(K\) may differ from the paper’s reported Reduced size.
- Method B’s gate stage is short (10 epochs); a longer schedule might change which filters survive.
- Sinc² kernels in this codebase (matching the authors) have a non-zero broadband floor from the centre sample = 1; this affects **all** Reduced variants equally.

---

## 11. Suggested structure in the final report

1. **Background:** Cascade + Mean-shift Reduced (paper).  
2. **Critique:** five limitations (table in §2).  
3. **Proposed methods:** A, B, C, D (+ AD).  
4. **Implementation notes:** equal-\(K\) protocol, datasets, training hyperparams.  
5. **Results:** comparison table + figures.  
6. **Discussion:** which limitation each method fixes; best overall (likely AD or A).  
7. **Conclusion:** Reduced banks can be made more perceptually and task-aware without increasing filter count.

---

## 12. Short “contribution bullets” (copy-paste)

- We identify that Mean-shift pole reduction is task-agnostic and uses a frequency-uniform distance that is poorly matched to gender cues at low frequencies.  
- We introduce **task-aware ERB-warped weighted mean-shift** (A) and **response-based agglomerative reduction** (C) as drop-in replacements for pole Mean-shift.  
- We introduce **L0 gate pruning** (B) as an end-to-end alternative that selects \(K\) Cascade filters without clustering.  
- We show that **distillation + unfrozen cutoffs** (D) improve Reduced training and combine naturally with A (**AD**).  
- All methods are evaluated at the **same filter count** as the paper’s Reduced model on TIMIT, RAVDESS, and VoxCeleb1-test.

---

## 13. File map (for Methods appendix)

| File | Role |
|------|------|
| `afm_sgr/reduced.py` | Paper Mean-shift Reduced |
| `afm_sgr/reduced_new.py` | Methods A–D helpers |
| `run_new_reduced.py` | Train / eval / comparison table |
| `new_reduced_pipeline.bat` | Resumable full run of A,B,C,D,AD |
| `afm_sgr/train.py` | Optional KD teacher in `fit()` |

---

*Last updated from the implementation in this repository. Fill numeric results after `new_reduced_pipeline.bat` completes on the trained Cascade checkpoint.*
