# AFM Speaker Gender Recognition

Reproduction and extension of Fayyazi & Shekofteh (2023), *Exploiting auditory filter models as interpretable convolutional frontends for speaker gender recognition* (Applied Acoustics), aligned with the authors’ [SGR_AFM](https://github.com/HosseinFayyazi/SGR_AFM) code. Includes task-aware **Reduced** filter-bank variants (ERB clustering + distillation).

**GitHub:** `afm-speaker-gender-recognition` (code, reports, metrics JSON, figures; datasets and `.pt` checkpoints are not in the repo—see below).

## Quick start

```bash
cd "e:\ACADEMY\SBU\paper implementation\paper1_implement2"
pip install -r requirements.txt

# sanity check (tiny subset, 2 epochs)
python run_pipeline.py --stage smoke --frontend cascade

# full TIMIT training (50 epochs, 1 run)
python run_pipeline.py --stage timit --frontend cascade

# fine-tune on LibriSpeech + eval RAVDESS & VoxCeleb1-test (20 epochs)
python run_pipeline.py --stage finetune --frontend cascade

# Reduced filter-bank model (needs cascade checkpoint)
python run_pipeline.py --stage reduced

# fine-tune Reduced (after reduced TIMIT stage)
python run_pipeline.py --stage finetune --frontend reduced

# everything for one frontend (cascade also builds Reduced)
python run_pipeline.py --stage all --frontend cascade
```

Other frontends: `free`, `mel`, `sinc`, `sinc2`, `gamma`, `gauss`, `cascade`, `reduced`.

`run_all_paper.bat` trains all frontends on TIMIT, builds Reduced, then fine-tunes + evals **free / sinc / sinc2 / gamma / gauss / cascade / reduced** on RAVDESS & VoxCeleb1 (paper Tables 7–8; Mel skipped).

## Data and checkpoints (not in git)

Place datasets under `data/` (TIMIT, LibriSpeech, RAVDESS, VoxCeleb1) as in `afm_sgr/config.py` / your local paths. Training writes `results_github/<run>/` including `best.pt` and `last.pt` (~100+ MB per run). This repository ships **metrics**, **plots**, and **report** sources so results are reproducible without uploading multi‑GB binaries.

## Alignment with the authors' code (`SGR_AFM-main`)

- **Frontends** (`afm_sgr/filters.py`): learned `norm_f1`, `norm_f2` (cutoffs / fs) with
  `f1 = |norm_f1| + 50/fs`, `f2 = f1 + |norm_f2 - f1| + 50/fs`; mel init `f1 = previous mel point (30 Hz first)`, `f2 = mel point`.
  Kernels are the authors' `kernel_sinc / sinc2 / gamma / gauss`, max-normalized then Hamming-windowed.
  Cascade uses two Gaussian stages `(f1,f2)`, `(f3,f4)` initialized on different mel band pairs (`get_f1_f4_new`).
- **Network** (`afm_sgr/model.py`): input LayerNorm, `|conv|` after the first layer, SincNet LayerNorm over `[C, T]`,
  LayerNorm at the MLP input, BatchNorm (momentum 0.05), uniform `±sqrt(0.01/(in+out))` init.
- **Training**: RMSprop (lr 1e-3, alpha 0.95, eps 1e-8), random chunk offset + random gain 0.8-1.2.
  Epochs stay 50 (TIMIT) / 20 (fine-tune); 8 random 200 ms chunks per utterance per epoch.
- **Eval**: utterance decision = argmax of summed chunk log-probabilities; FER averaged per utterance.
- **Data**: TIMIT peak-normalized + trimmed to first/last word (`.WRD`); LibriSpeech silences >= 1 s below -30 dB removed; RAVDESS / VoxCeleb raw.
- **Reduced**: poles `exp(-sigma + j w0)` of both cascade stages -> MeanShift(bandwidth 0.079) -> mean `(f1, f2)` per cluster -> frozen sinc2 bank.

## Outputs

Under `results_github/<run_name>/` (override with env `AFM_RESULTS_DIR`; the old `results/` is not touched):
- `loss_per_epoch.png`, `accuracy_per_epoch.png`, `history.json`
- `best.pt`, `metrics_*.json`
- `figures/`: t-SNE, accumulative magnitude, fc histogram, quality factor, learned `f1_f2_*.png/json`
- `reduced_timit/reduced_filters.json` and `figures/pole_reduction_complex_plane.png`

After the full bat finishes (or anytime):
```bash
python scripts/make_comparison_table.py
```
writes `results_github/comparison_table.csv`, `.md`, and `.png`.

## Resume after shutdown

Training saves `results_github/<run>/last.pt` **after every epoch**.

- Re-run `run_all_paper.bat` → skips finished stages (`STAGE_DONE.json` + metrics), continues unfinished ones from `last.pt`, and skips evals that already have `metrics_*.json`.
- Mid-eval crash (e.g. RAVDESS done, Vox not) → re-run finishes only the missing eval.
- Force a stage from scratch:
  ```bash
  python run_pipeline.py --stage timit --frontend free --force --no-resume
  ```
  or delete that run’s `STAGE_DONE.json` / `last.pt` / `metrics_*.json`.
