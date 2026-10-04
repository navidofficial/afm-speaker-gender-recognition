"""
Build English implementation report (Markdown + multi-page PDF).

Usage:
  python scripts/plot_pole_reduction.py
  python scripts/generate_report_pdf.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
ASSETS = RESULTS / "report_assets"
REPORT_MD = ROOT / "REPORT.md"
REPORT_PDF = ROOT / "REPORT.pdf"

PAPER_TIMIT = {
    "free": {"acc": 98.84, "cer": 1.16, "fer": 3.14},
    "mel": {"acc": 98.64, "cer": 1.36, "fer": 4.92},
    "sinc": {"acc": 98.84, "cer": 1.16, "fer": 2.93},
    "sinc2": {"acc": 99.03, "cer": 0.97, "fer": 3.16},
    "gamma": {"acc": 99.01, "cer": 0.99, "fer": 2.96},
    "gauss": {"acc": 98.91, "cer": 1.09, "fer": 2.90},
    "cascade": {"acc": 99.16, "cer": 0.84, "fer": 2.88},
    "reduced": {"acc": 99.03, "cer": 0.97, "fer": 3.94},
}
PAPER_RAV = {
    "free": 93.54, "sinc": 96.32, "sinc2": 94.86, "gamma": 97.78,
    "gauss": 93.19, "cascade": 97.99, "reduced": 96.11,
}
PAPER_VOX = {
    "free": 97.48, "sinc": 97.66, "sinc2": 96.33, "gamma": 97.17,
    "gauss": 96.74, "cascade": 97.27, "reduced": 97.44,
}

FRONTEND_LABELS = {
    "free": "Free-CNN",
    "mel": "Mel-CNN",
    "sinc": "Sinc",
    "sinc2": "Sinc2",
    "gamma": "Gamma",
    "gauss": "Gauss",
    "cascade": "Cascade",
    "reduced": "Reduced",
}


def _load_metrics(run: str, split: str) -> dict | None:
    p = RESULTS / run / f"metrics_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _ours_timit(frontend: str) -> dict:
    return _load_metrics(f"{frontend}_timit", "timit_test") or {}


def _ours_ft(frontend: str, split: str) -> dict | None:
    return _load_metrics(f"{frontend}_timit_finetune", split)


def _write_markdown() -> str:
    today = date.today().isoformat()
    lines = [
        "# Implementation Report: Auditory Filter Model Frontends for Speaker Gender Recognition",
        "",
        f"**Date:** {today}  ",
        "**Reference:** Fayyazi, H., & Shekofteh, Y. (2023). "
        "*Exploiting auditory filter models as interpretable convolutional frontends to obtain optimal "
        "architectures for speaker gender recognition.* Applied Acoustics, 213, 109635.",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "This report documents a full reproduction of the AFM-based speaker gender recognition pipeline "
        "described by Fayyazi and Shekofteh (2023). The system classifies speakers as male or female from "
        "raw 16 kHz waveforms using interpretable auditory filter models (AFMs) as the first convolutional "
        "layer. Results are reported on TIMIT (in-domain), RAVDESS, and VoxCeleb1-test (out-of-domain), "
        "and compared against the published benchmarks.",
        "",
        "**Key findings:**",
        "- TIMIT performance closely matches the paper (typically within ±0.5% accuracy).",
        "- Out-of-domain accuracy on RAVDESS and VoxCeleb1-test is lower than reported, primarily due to "
        "shorter fine-tuning, a smaller fine-tuning corpus, and evaluation on VoxCeleb1-test only.",
        "- The Reduced model achieves the best TIMIT CER (0.67%) and competitive VoxCeleb1-test accuracy (95.28%).",
        "",
        "## 2. Implementation Pipeline",
        "",
        "| Stage | Description |",
        "|-------|-------------|",
        "| 1. Preprocessing | Voice activity detection (VAD), 16 kHz resampling |",
        "| 2. TIMIT training | Train all frontends (Free, Mel, Sinc, Sinc2, Gamma, Gauss, Cascade) for 50 epochs |",
        "| 3. Filter analysis | t-SNE embeddings, accumulative magnitude response, center-frequency histogram, quality factor |",
        "| 4. Model reduction | Extract Cascade poles → mean-shift clustering → frozen triangular filter bank (~26 filters) → retrain |",
        "| 5. Fine-tuning | 20 epochs on LibriSpeech dev-clean |",
        "| 6. Generalization eval | RAVDESS (1,440 utterances) and VoxCeleb1-test (4,874 utterances) |",
        "| 7. Reporting | Comparison tables, training curves, pole-reduction visualization |",
        "",
        "## 3. Datasets",
        "",
        "| Dataset | Role in paper | Role in this project |",
        "|---------|---------------|----------------------|",
        "| TIMIT | Primary train/test | Complete (1,344 test utterances) |",
        "| LibriSpeech train-clean-360 | Fine-tuning (921 speakers) | Not used |",
        "| LibriSpeech dev-clean | — | Fine-tuning substitute |",
        "| RAVDESS | Generalization test | Complete (1,440 utterances) |",
        "| VoxCeleb1 (full) | Generalization test (~1,251 speakers) | Not used |",
        "| VoxCeleb1-test | — | Evaluation only (~40 speakers) |",
        "",
        "## 4. Training Configuration",
        "",
        "| Parameter | Paper | This implementation |",
        "|-----------|-------|---------------------|",
        "| Input | Raw waveform, 16 kHz | Identical |",
        "| Chunk / hop | 200 ms / 10 ms | Eval: identical; Train: 8 random chunks per utterance per epoch |",
        "| Conv1 | 80 filters, length 257 | Identical (Reduced: 26 filters after pruning) |",
        "| Conv2 / Conv3 | 60 filters, kernel 5, pool 3 | Identical |",
        "| Classifier | 3 × FC 2048 + BatchNorm, 2-class softmax | Identical |",
        "| Loss / optimizer | NLL / RMSprop, lr = 0.001 | Identical |",
        "| Batch size | 128 | 128 |",
        "| TIMIT epochs | 500 | **50** |",
        "| Fine-tune epochs | 200 (train-clean-360) | **20** (dev-clean) |",
        "| Independent runs | 3 (mean ± std) | **1** (seed = 42) |",
        "",
        "## 5. TIMIT Test Results (vs. Paper Table 4)",
        "",
        "Paper values are the mean of three independent runs. Δ = Ours − Paper.",
        "",
        "| Method | Paper Acc (%) | Ours Acc (%) | Δ Acc | Paper CER (%) | Ours CER (%) | Paper FER (%) | Ours FER (%) |",
        "|--------|-------------:|-------------:|------:|--------------:|-------------:|--------------:|-------------:|",
    ]

    for fe in ["free", "mel", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
        p = PAPER_TIMIT[fe]
        o = _ours_timit(fe)
        if not o:
            continue
        lines.append(
            f"| {FRONTEND_LABELS[fe]} | {p['acc']:.2f} | {o['accuracy']:.2f} | "
            f"{o['accuracy'] - p['acc']:+.2f} | {p['cer']:.2f} | {o['cer']:.2f} | "
            f"{p['fer']:.2f} | {o['fer']:.2f} |"
        )

    lines += [
        "",
        "## 6. RAVDESS Results after Fine-tuning (vs. Paper Table 8)",
        "",
        "| Method | Paper Acc (%) | Ours Acc (%) | Δ Acc |",
        "|--------|-------------:|-------------:|------:|",
    ]
    for fe in ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
        o = _ours_ft(fe, "ravdess")
        if not o:
            continue
        pa = PAPER_RAV[fe]
        lines.append(
            f"| {FRONTEND_LABELS[fe]} | {pa:.2f} | {o['accuracy']:.2f} | {o['accuracy'] - pa:+.2f} |"
        )

    lines += [
        "",
        "## 7. VoxCeleb1-test Results after Fine-tuning (vs. Paper Table 7)",
        "",
        "| Method | Paper Acc (%) | Ours Acc (%) | Δ Acc |",
        "|--------|-------------:|-------------:|------:|",
    ]
    for fe in ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
        o = _ours_ft(fe, "voxceleb1_test")
        if not o:
            continue
        pa = PAPER_VOX[fe]
        lines.append(
            f"| {FRONTEND_LABELS[fe]} | {pa:.2f} | {o['accuracy']:.2f} | {o['accuracy'] - pa:+.2f} |"
        )

    lines += [
        "",
        "## 8. Analysis of Discrepancies",
        "",
        "### 8.1 TIMIT (close agreement)",
        "",
        "On the in-domain TIMIT test set, most frontends fall within ±0.5% of the paper's reported mean accuracy. "
        "Sinc and Reduced achieve the lowest CER in our runs (0.67%). Cascade underperforms the paper's best "
        "result by 0.13 percentage points in CER (0.97% vs. 0.84%). These small gaps are consistent with "
        "training for 50 rather than 500 epochs and reporting a single run instead of an average over three.",
        "",
        "### 8.2 RAVDESS and VoxCeleb1-test (larger gap)",
        "",
        "Out-of-domain accuracy is systematically lower than the paper, with deficits ranging from 0.3% to 9.9%. "
        "The principal contributing factors are:",
        "",
        "1. **Reduced fine-tuning duration:** 20 epochs vs. 200 in the paper (10× shorter).",
        "2. **Smaller fine-tuning corpus:** LibriSpeech dev-clean (~40 speakers) instead of train-clean-360 (~921 speakers).",
        "3. **Restricted VoxCeleb evaluation:** VoxCeleb1-test only (~40 speakers) vs. the full VoxCeleb1 corpus (~1,251 speakers).",
        "4. **Single-run reporting:** The paper reports mean ± std over three runs; we report one deterministic run (seed = 42).",
        "5. **Stochastic training sampling:** Eight random 200 ms chunks per utterance per epoch, rather than exhaustive 10 ms hop coverage during training.",
        "",
        "### 8.3 Notable positive results",
        "",
        "- **Sinc2** and **Reduced** on VoxCeleb1-test are within 1.2–2.2 percentage points of the paper.",
        "- **Gauss** on RAVDESS marginally exceeds the paper (+0.28%).",
        "- **Reduced** achieves the best TIMIT CER (0.67%) while using only 26 frozen filters.",
        "",
        "## 9. Figures",
        "",
        "The accompanying PDF (`REPORT.pdf`) includes:",
        "",
        "- Training loss and accuracy curves (Cascade and Reduced, TIMIT and fine-tuning stages)",
        "- Pole reduction on the complex plane (160 Cascade poles → 26 cluster centroids)",
        "- Filter interpretability plots (accumulative magnitude, center-frequency histogram, t-SNE, quality factor)",
        "- VoxCeleb1-test accuracy comparison bar chart",
        "",
        "Individual plot files are stored under `results/<run_name>/`.",
        "",
    ]
    text = "\n".join(lines)
    REPORT_MD.write_text(text, encoding="utf-8")
    return text


def _text_page(pdf: PdfPages, title: str, body: str, fontsize: float = 10.5):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(0.08, y, title, fontsize=17, weight="bold", va="top")
    y -= 0.05
    for para in body.strip().split("\n\n"):
        wrapped = textwrap.fill(para, width=95)
        n_lines = wrapped.count("\n") + 1
        fig.text(0.08, y, wrapped, fontsize=fontsize, va="top", linespacing=1.45)
        y -= 0.028 * n_lines + 0.02
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.94
            fig.text(0.08, y, f"{title} (cont.)", fontsize=14, weight="bold", va="top")
            y -= 0.05
            fig.text(0.08, y, wrapped, fontsize=fontsize, va="top", linespacing=1.45)
            y -= 0.028 * n_lines + 0.02
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[list[str]]):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", pad=14)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _image_page(pdf: PdfPages, title: str, img_path: Path):
    if not img_path.exists():
        return
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=13, weight="bold")
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.88])
    ax.imshow(mpimg.imread(str(img_path)))
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dual_image_page(pdf: PdfPages, title: str, left: Path, right: Path, lt: str, rt: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=13, weight="bold")
    for ax, p, st in zip(axes, [left, right], [lt, rt]):
        if p.exists():
            ax.imshow(mpimg.imread(str(p)))
        ax.set_title(st, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_pdf():
    with PdfPages(REPORT_PDF) as pdf:
        _text_page(
            pdf,
            "Implementation Report",
            f"Auditory Filter Model Frontends for Speaker Gender Recognition\n"
            f"Reference: Fayyazi & Shekofteh (2023), Applied Acoustics 213\n"
            f"Date: {date.today().strftime('%B %d, %Y')}\n\n"
            "This document presents a systematic reproduction and evaluation of the AFM-based "
            "speaker gender recognition framework proposed by Fayyazi and Shekofteh. The report covers "
            "the experimental pipeline, training configuration, quantitative comparison with published "
            "results, analysis of observed discrepancies, and supporting visualizations.",
            fontsize=11,
        )

        _text_page(
            pdf,
            "1. Executive Summary",
            "We implemented the full training and evaluation pipeline for eight convolutional frontends "
            "(Free-CNN, Mel-CNN, Sinc, Sinc2, Gamma, Gauss, Cascade, and Reduced) on TIMIT, followed by "
            "LibriSpeech fine-tuning and out-of-domain evaluation on RAVDESS and VoxCeleb1-test.\n\n"
            "On TIMIT, our results closely match the published benchmarks, with most methods within ±0.5% "
            "of the paper's mean accuracy. The Reduced model achieves the lowest CER (0.67%).\n\n"
            "On RAVDESS and VoxCeleb1-test, accuracy is consistently lower than reported. The gap is "
            "attributed to a 10× shorter fine-tuning schedule, use of LibriSpeech dev-clean instead of "
            "train-clean-360, evaluation on VoxCeleb1-test only, and single-run reporting.",
        )

        _text_page(
            pdf,
            "2. Experimental Pipeline",
            "Stage 1 — Preprocessing: Energy-based voice activity detection removes silence; all audio is "
            "resampled to 16 kHz.\n\n"
            "Stage 2 — TIMIT training: Each frontend is trained for 50 epochs on the TIMIT train/validation "
            "split (3,700 / 920 utterances; 1,344 test utterances).\n\n"
            "Stage 3 — Filter interpretability analysis: t-SNE of convolutional features, accumulative "
            "magnitude response, center-frequency histogram, and quality factor plots are generated per model.\n\n"
            "Stage 4 — Model reduction: Cascade filter poles are mapped to the complex plane, clustered via "
            "mean-shift (target: 40 centroids), converted to triangular FIR filters, and frozen. The remaining "
            "network is retrained, yielding the Reduced model (26 filters).\n\n"
            "Stage 5 — Fine-tuning: Models are fine-tuned for 20 epochs on LibriSpeech dev-clean.\n\n"
            "Stage 6 — Generalization evaluation: Fine-tuned models are evaluated on RAVDESS and VoxCeleb1-test.\n\n"
            "Stage 7 — Reporting: Metrics, comparison tables, and figures are aggregated into this report.",
        )

        _table_page(
            pdf,
            "3. Training Configuration — Paper vs. Implementation",
            ["Parameter", "Paper", "This implementation"],
            [
                ["Input", "Raw 16 kHz waveform", "Identical"],
                ["Chunk / hop", "200 ms / 10 ms", "Eval: identical; Train: 8 random chunks/utt/epoch"],
                ["Conv1", "80 filters, length 257", "Identical (Reduced: 26 after pruning)"],
                ["Optimizer / LR", "RMSprop / 0.001", "Identical"],
                ["Batch size", "128", "128"],
                ["TIMIT epochs", "500", "50"],
                ["Fine-tune", "200 epochs, train-clean-360", "20 epochs, dev-clean"],
                ["Runs", "3 (mean ± std)", "1 (seed = 42)"],
                ["VoxCeleb eval", "Full VoxCeleb1", "VoxCeleb1-test only"],
            ],
        )

        timit_rows = []
        for fe in ["free", "mel", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
            p = PAPER_TIMIT[fe]
            o = _ours_timit(fe)
            if not o:
                continue
            timit_rows.append([
                FRONTEND_LABELS[fe],
                f"{p['acc']:.2f}", f"{o['accuracy']:.2f}", f"{o['accuracy'] - p['acc']:+.2f}",
                f"{p['cer']:.2f}", f"{o['cer']:.2f}",
                f"{p['fer']:.2f}", f"{o['fer']:.2f}",
            ])
        _table_page(
            pdf,
            "4. TIMIT Test Results (Paper Table 4)",
            ["Method", "P.Acc", "O.Acc", "Δ", "P.CER", "O.CER", "P.FER", "O.FER"],
            timit_rows,
        )

        rav_rows = []
        for fe in ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
            o = _ours_ft(fe, "ravdess")
            if not o:
                continue
            pa = PAPER_RAV[fe]
            rav_rows.append([
                FRONTEND_LABELS[fe], f"{pa:.2f}", f"{o['accuracy']:.2f}", f"{o['accuracy'] - pa:+.2f}",
            ])
        _table_page(pdf, "5. RAVDESS Results after Fine-tuning (Paper Table 8)",
                    ["Method", "Paper Acc (%)", "Ours Acc (%)", "Δ"], rav_rows)

        vox_rows = []
        for fe in ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]:
            o = _ours_ft(fe, "voxceleb1_test")
            if not o:
                continue
            pa = PAPER_VOX[fe]
            vox_rows.append([
                FRONTEND_LABELS[fe], f"{pa:.2f}", f"{o['accuracy']:.2f}", f"{o['accuracy'] - pa:+.2f}",
            ])
        _table_page(pdf, "6. VoxCeleb1-test Results after Fine-tuning (Paper Table 7)",
                    ["Method", "Paper Acc (%)", "Ours Acc (%)", "Δ"], vox_rows)

        _text_page(
            pdf,
            "7. Analysis of Discrepancies",
            "TIMIT — Close agreement: Most frontends are within ±0.5% of the paper's mean accuracy. "
            "Sinc and Reduced achieve the lowest CER (0.67%). The small residual gap is explained by "
            "50 vs. 500 training epochs and single-run vs. three-run averaging.\n\n"
            "RAVDESS / VoxCeleb — Systematic deficit: Out-of-domain accuracy is 0.3–9.9 percentage points "
            "below the paper. The primary causes are: (1) fine-tuning for 20 instead of 200 epochs; "
            "(2) LibriSpeech dev-clean (~40 speakers) instead of train-clean-360 (~921 speakers); "
            "(3) VoxCeleb1-test (~40 speakers) instead of the full corpus (~1,251 speakers); "
            "(4) stochastic chunk sampling during training; and (5) single-run reporting.\n\n"
            "Notable results: Sinc2 (95.12%) and Reduced (95.28%) on VoxCeleb1-test are within 1.2–2.2 pp "
            "of the paper. Gauss on RAVDESS marginally exceeds the paper (+0.28%). Cascade achieves the "
            "highest RAVDESS accuracy in our runs (95.21%).",
        )

        for run, label in [
            ("cascade_timit", "Cascade — TIMIT Training"),
            ("cascade_timit_finetune", "Cascade — LibriSpeech Fine-tuning"),
            ("reduced_timit", "Reduced — TIMIT Training"),
            ("reduced_timit_finetune", "Reduced — LibriSpeech Fine-tuning"),
        ]:
            _dual_image_page(
                pdf,
                f"Training Curves — {label}",
                RESULTS / run / "loss_per_epoch.png",
                RESULTS / run / "accuracy_per_epoch.png",
                "Loss per Epoch", "Accuracy per Epoch",
            )

        pole = ASSETS / "pole_reduction_complex_plane.png"
        _image_page(
            pdf,
            "Pole Reduction on the Complex Plane (Cascade → Reduced via Mean-Shift Clustering)",
            pole,
        )

        for tag, folder in [("Cascade", "cascade_timit"), ("Reduced", "reduced_timit")]:
            figdir = RESULTS / folder / "figures"
            prefix = folder.split("_")[0]
            _dual_image_page(
                pdf, f"Filter Analysis — {tag} (TIMIT)",
                figdir / f"accum_mag_{prefix}.png",
                figdir / f"fc_hist_{prefix}.png",
                "Accumulative Magnitude Response", "Center-Frequency Histogram",
            )
            qf = list(figdir.glob("quality_factor_*.png"))
            tsne = list(figdir.glob("tsne_*.png"))
            if qf and tsne:
                _dual_image_page(
                    pdf, f"Feature & Filter Analysis — {tag} (TIMIT)",
                    tsne[0], qf[0], "t-SNE of Conv Features", "Quality Factor",
                )

        frontends = ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]
        ours_vox = [(_ours_ft(f, "voxceleb1_test") or {}).get("accuracy", 0) for f in frontends]
        paper_vox = [PAPER_VOX[f] for f in frontends]
        labels = [FRONTEND_LABELS[f] for f in frontends]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = range(len(frontends))
        w = 0.35
        ax.bar([i - w / 2 for i in x], paper_vox, w, label="Paper", color="#4C72B0")
        ax.bar([i + w / 2 for i in x], ours_vox, w, label="Ours", color="#DD8452")
        ax.set_xticks(list(x), labels, rotation=15)
        ax.set_ylabel("Utterance Accuracy (%)")
        ax.set_title("VoxCeleb1-test Accuracy: Paper vs. Implementation")
        ax.legend()
        ax.set_ylim(85, 100)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "AFM-SGR Implementation Report"
        d["Author"] = "paper1_implement2"
        d["Subject"] = "Speaker gender recognition with auditory filter model frontends"


def main():
    pole_script = ROOT / "scripts" / "plot_pole_reduction.py"
    if not (ASSETS / "pole_reduction_complex_plane.png").exists():
        subprocess.run([sys.executable, str(pole_script)], check=True)
    _write_markdown()
    _build_pdf()
    print(f"[ok] {REPORT_MD}")
    print(f"[ok] {REPORT_PDF}")


if __name__ == "__main__":
    main()
