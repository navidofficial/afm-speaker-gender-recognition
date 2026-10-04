"""Regenerate t-SNE figures with gender-balanced sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm_sgr.analyze import plot_tsne
from afm_sgr.config import get_config
from afm_sgr.datasets import discover_timit, split_timit_train_val
from afm_sgr.train import load_model_from_checkpoint


def load_model(run_name: str, device):
    cfg = get_config(run_name=run_name)
    model = load_model_from_checkpoint(cfg, cfg.run_dir() / "best.pt", device)
    model.eval()
    return model, cfg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = get_config()
    utts = discover_timit(cfg0.timit_root, exclude_sa=True)
    _, _, te = split_timit_train_val(utts, cfg0.val_ratio_from_train, cfg0.seed)

    runs = [
        ("cascade_timit", "cascade"),
        ("reduced_timit", "reduced"),
        ("sinc_timit", "sinc"),
        ("gamma_timit", "gamma"),
        ("gauss_timit", "gauss"),
        ("free_timit", "free"),
        ("sinc2_timit", "sinc2"),
        ("mel_timit", "mel"),
    ]
    for run, tag in runs:
        ckpt = Path(cfg0.results_root) / run / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {run}")
            continue
        print(f"=== regenerating t-SNE for {run} ===")
        model, cfg = load_model(run, device)
        out = cfg.run_dir() / "figures"
        out.mkdir(parents=True, exist_ok=True)
        plot_tsne(model, te, cfg, device, out / f"tsne_{tag}.png", f"t-SNE ({tag})")
        print(f"[ok] {out / f'tsne_{tag}.png'}")


if __name__ == "__main__":
    main()
