"""Regenerate Reduced RAVDESS t-SNE with gender-balanced sampling."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm_sgr.analyze import plot_tsne
from afm_sgr.config import get_config
from afm_sgr.datasets import discover_ravdess
from afm_sgr.train import load_model_from_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config(run_name="reduced_timit_finetune", frontend="reduced")
    model = load_model_from_checkpoint(cfg, cfg.run_dir() / "best.pt", device)
    model.eval()

    rav = discover_ravdess(cfg.ravdess_root)
    out = cfg.run_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_tsne(
        model, rav, cfg, device,
        out / "tsne_reduced_ravdess.png",
        "t-SNE Reduced after fine-tune (RAVDESS)",
    )
    report_fig = ROOT / "report_latex" / "figures"
    if report_fig.exists():
        shutil.copy(out / "tsne_reduced_ravdess.png", report_fig / "reduced_ft_tsne.png")
    print("[ok] regenerated Reduced RAVDESS t-SNE")


if __name__ == "__main__":
    main()
