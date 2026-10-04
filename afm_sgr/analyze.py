from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .config import Config
from .datasets import Utterance
from .model import GenderCNN


def _balanced_utterances(
    utts: List[Utterance],
    max_utts: int = 80,
    seed: int = 42,
) -> List[Utterance]:
    """
    Stratify by gender so t-SNE is not dominated by alphabetical TIMIT order
    (F* speakers come before M*, so taking the first N chunks yields females only).
    """
    males = [u for u in utts if u.gender == 0]
    females = [u for u in utts if u.gender == 1]
    rng = random.Random(seed)
    rng.shuffle(males)
    rng.shuffle(females)
    n_each = max(1, max_utts // 2)
    selected = males[:n_each] + females[:n_each]
    rng.shuffle(selected)
    return selected


@torch.no_grad()
def collect_conv_features(
    model: GenderCNN,
    utts: List[Utterance],
    cfg: Config,
    device: torch.device,
    max_chunks: int = 4000,
):
    from tqdm import tqdm

    from .datasets import WaveCache, EvalChunkDataset

    # Cap chunks per utterance so one long file cannot monopolize the budget
    chunks_per_utt = max(8, max_chunks // max(1, len(utts)))
    cache = WaveCache(cfg.sample_rate, remove_sil=True)
    for u in tqdm(utts, desc="Cache for t-SNE", leave=False):
        cache.get(u)
    ds = EvalChunkDataset(utts, cache, cfg.chunk_len, cfg.hop_len)

    # Subsample index: up to chunks_per_utt random frames per utterance, then gender-balance
    by_utt: Dict[int, List[int]] = {}
    for idx, (u_idx, _c) in enumerate(ds.index):
        by_utt.setdefault(u_idx, []).append(idx)
    rng = np.random.default_rng(42)
    chosen: List[int] = []
    for u_idx, idxs in by_utt.items():
        take = min(chunks_per_utt, len(idxs))
        chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
    rng.shuffle(chosen)

    feats, labels = [], []
    model.eval()
    for start in tqdm(range(0, len(chosen), cfg.batch_size), desc="Extract features", leave=False):
        batch_ids = chosen[start : start + cfg.batch_size]
        xs, ys = [], []
        for i in batch_ids:
            x, y, _ = ds[i]
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys)
        _, f = model(x, return_conv_feat=True)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        if sum(len(a) for a in labels) >= max_chunks:
            break
    X = np.concatenate(feats, axis=0)[:max_chunks]
    y = np.concatenate(labels, axis=0)[:max_chunks]
    return X, y


def plot_tsne(model, utts, cfg, device, out_path: Path, title: str):
    subset = _balanced_utterances(utts, max_utts=80, seed=42)
    X, y = collect_conv_features(model, subset, cfg, device, max_chunks=4000)
    n_m = int((y == 0).sum())
    n_f = int((y == 1).sum())
    print(f"[t-SNE] chunks: male={n_m} female={n_f} (utterances used={len(subset)})")
    if n_m == 0 or n_f == 0:
        raise RuntimeError(
            f"t-SNE gender imbalance: male={n_m} female={n_f}. "
            "Check utterance sampling / label mapping."
        )
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    Z = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Draw male first, female on top with matching alpha so both are visible
    for lab, name, color in [(0, "male", "C0"), (1, "female", "C1")]:
        m = y == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55, label=f"{name} (n={m.sum()})", c=color)
    ax.legend()
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _filters_and_fb(model: GenderCNN, sample_rate: int):
    frontend = model.frontend
    w = frontend.get_filters().detach().cpu()
    if hasattr(frontend, "get_fc_bw"):
        fc, bw, amp = frontend.get_fc_bw()
        fc = fc.detach().cpu().numpy()
        bw = bw.detach().cpu().numpy()
        # Free / Mel report normalized cycles/sample; AFM frontends report Hz
        if fc.max() <= 1.0:
            fc = fc * sample_rate
            bw = bw * sample_rate
    else:
        fc = np.zeros(w.shape[0])
        bw = np.ones(w.shape[0])
    return w.numpy(), fc, bw


def plot_accumulative_magnitude(model, cfg, out_path: Path, title: str):
    w, fc, bw = _filters_and_fb(model, cfg.sample_rate)
    n_fft = 2048
    freqs = np.linspace(0, cfg.sample_rate / 2, n_fft // 2 + 1)
    acc = np.zeros_like(freqs)
    for i in range(w.shape[0]):
        spec = np.abs(np.fft.rfft(w[i], n=n_fft))
        acc += spec / (spec.max() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, acc)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Accumulative magnitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_center_frequency_hist(model, cfg, out_path: Path, title: str):
    _, fc, _ = _filters_and_fb(model, cfg.sample_rate)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fc, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Center frequency (Hz)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quality_factor(model, cfg, out_path: Path, title: str):
    _, fc, bw = _filters_and_fb(model, cfg.sample_rate)
    q = fc / np.maximum(bw, 1.0)
    order = np.argsort(fc)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fc[order], q[order], "o-", markersize=3)
    ax.set_xlabel("Center frequency (Hz)")
    ax.set_ylabel("Quality factor (fc/B)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_learned_cutoffs(model, out_dir: Path, tag: str) -> None:
    """Dump and plot learned (f1, f2) of AFM frontends (cascade: both stages)."""
    frontend = model.frontend
    if not hasattr(frontend, "get_f1_f2"):
        return
    f1, f2 = frontend.get_f1_f2()
    (out_dir / f"f1_f2_{tag}.json").write_text(
        json.dumps({"f1_hz": f1.tolist(), "f2_hz": f2.tolist()}, indent=2), encoding="utf-8"
    )
    order = np.argsort((f1 + f2) / 2.0)
    idx = np.arange(len(f1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.vlines(idx, f1[order], f2[order], color="steelblue", alpha=0.6)
    ax.plot(idx, f1[order], ".", color="C0", markersize=4, label="f1")
    ax.plot(idx, f2[order], ".", color="C3", markersize=4, label="f2")
    ax.set_xlabel("Filter (sorted by center frequency)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Learned cutoffs f1 / f2 ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"f1_f2_{tag}.png", dpi=150)
    plt.close(fig)


def run_paper_analyses(
    model: GenderCNN,
    test_utts: List[Utterance],
    cfg: Config,
    device: torch.device,
    tag: str,
):
    out = cfg.run_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_tsne(model, test_utts, cfg, device, out / f"tsne_{tag}.png", f"t-SNE ({tag})")
    plot_accumulative_magnitude(model, cfg, out / f"accum_mag_{tag}.png", f"Accumulative magnitude ({tag})")
    plot_center_frequency_hist(model, cfg, out / f"fc_hist_{tag}.png", f"Center frequency histogram ({tag})")
    plot_quality_factor(model, cfg, out / f"quality_factor_{tag}.png", f"Quality factor ({tag})")
    save_learned_cutoffs(model, out, tag)
    print(f"[analysis] saved figures under {out}")
