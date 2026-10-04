"""
New ways to build the Reduced filter bank from a trained Cascade model.

All methods produce K sinc2 filters (K = number of filters of the paper's Mean-shift
Reduced model, so comparisons are at an equal filter count):

  A  task-aware, ERB-warped pole clustering
       poles described by (ERB-rate of centre freq, ERB-rate width); each pole weighted by the
       importance of its Cascade filter (Taylor |a * dL/da| + gender Fisher score of the filter
       energy); weighted Gaussian mean-shift, bandwidth searched so that K clusters remain.
  B  learnable L0 gates (hard-concrete) on the Cascade filter outputs, trained with a
       sparsity penalty toward K open gates; the K most-open filters are kept.
  C  agglomerative clustering of the 80 Cascade filters on the cosine distance of their
       effective (two-stage) magnitude responses.
  D  training add-on: knowledge distillation from the Cascade teacher + trainable f1/f2.

A, B and C convert the selected filters / clusters to (f1, f2) bands; sinc2 then reproduces them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ARCH_VERSION, Config
from .datasets import EvalChunkDataset, TrainChunkDataset, Utterance, WaveCache
from .filters import AFMConv1d
from .model import GenderCNN


# ------------------------------------------------------------------ frequency helpers
def hz_to_erb_rate(f_hz: np.ndarray) -> np.ndarray:
    """Glasberg & Moore ERB-rate scale (number of ERBs below f)."""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(f_hz, dtype=np.float64))


def effective_responses(frontend: nn.Module, sample_rate: int, n_fft: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude responses (n_filt, n_fft//2+1) of the frontend kernels and the frequency axis (Hz)."""
    with torch.no_grad():
        h = frontend.get_filters().detach().cpu().double()
    mag = torch.fft.rfft(h, n=n_fft).abs().numpy()
    freqs = np.linspace(0.0, sample_rate / 2.0, mag.shape[1])
    return mag, freqs


def band_from_response(mag: np.ndarray, freqs: np.ndarray, drop_db: float = 3.0, min_hz: float = 20.0) -> Tuple[float, float]:
    """(f1, f2) = contiguous region around the peak that stays within `drop_db` of the peak."""
    valid = freqs >= min_hz
    peak = int(np.flatnonzero(valid)[np.argmax(mag[valid])])
    thr = mag[peak] * 10.0 ** (-drop_db / 20.0)
    lo = peak
    while lo > 0 and mag[lo - 1] >= thr:
        lo -= 1
    hi = peak
    while hi < len(mag) - 1 and mag[hi + 1] >= thr:
        hi += 1
    return float(freqs[lo]), float(freqs[hi])


def to_sinc2_init(f1_hz: np.ndarray, f2_hz: np.ndarray, min_freq_hz: float, sample_rate: int) -> Dict[str, np.ndarray]:
    """
    AFMConv1d adds min_freq_hz to both cutoffs (f1 = |nf1| + m, f2 = f1 + |nf2 - f1| + m), so the
    stored init is shifted down by m to make the realised band equal the requested one.
    """
    m = min_freq_hz
    f1 = np.clip(np.asarray(f1_hz, dtype=np.float64), m, sample_rate / 2.0 - 3 * m)
    f2 = np.maximum(np.asarray(f2_hz, dtype=np.float64), f1 + 2 * m)
    f2 = np.minimum(f2, sample_rate / 2.0)
    order = np.argsort((f1 + f2) / 2.0)
    return {
        "f1_hz": f1[order] - m,
        "f2_hz": f2[order] - m,
        "band_f1_hz": f1[order],
        "band_f2_hz": f2[order],
    }


def _bands_of_filters(frontend: nn.Module, idx: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    mag, freqs = effective_responses(frontend, sample_rate)
    bands = [band_from_response(mag[i], freqs) for i in idx]
    return np.array([b[0] for b in bands]), np.array([b[1] for b in bands])


# ------------------------------------------------------------------ A: task-aware ERB clustering
def _probe_loader(utts: List[Utterance], cfg: Config, max_utts: int) -> DataLoader:
    from .analyze import _balanced_utterances

    sel = _balanced_utterances(utts, max_utts=max_utts, seed=cfg.seed)
    cache = WaveCache(cfg.sample_rate, remove_sil=True)
    ds = EvalChunkDataset(sel, cache, cfg.chunk_len, cfg.chunk_len)  # non-overlapping chunks
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)


def filter_importance(model: GenderCNN, utts: List[Utterance], cfg: Config, device, max_utts: int = 300) -> Dict[str, np.ndarray]:
    """
    Per Cascade filter:
      taylor = mean over chunks of |sum_t a(t) * dL/da(t)|   (first-order loss change if removed)
      fisher = (mu_m - mu_f)^2 / (var_m + var_f) of the chunk log-energy of the filter output
    importance = average of both, each normalised to mean 1.
    """
    loader = _probe_loader(utts, cfg, max_utts)
    model.eval()
    acts = {}

    def hook(_m, _inp, out):
        out.retain_grad()
        acts["a"] = out

    handle = model.frontend.register_forward_hook(hook)
    n = model.frontend.n_filt
    taylor = torch.zeros(n, dtype=torch.float64)
    energies, genders = [], []
    n_chunks = 0
    try:
        for x, y, _ in tqdm(loader, desc="Filter importance", leave=False):
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = F.cross_entropy(model(x), y, reduction="sum")
                loss.backward()
            a = acts["a"]
            taylor += (a.detach() * a.grad).sum(dim=2).abs().sum(dim=0).double().cpu()
            energies.append(torch.log(a.detach().pow(2).mean(dim=2) + 1e-8).double().cpu())
            genders.append(y.cpu())
            n_chunks += x.size(0)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    taylor = (taylor / max(1, n_chunks)).numpy()
    e = torch.cat(energies).numpy()
    g = torch.cat(genders).numpy()
    em, ef = e[g == 0], e[g == 1]
    fisher = (em.mean(0) - ef.mean(0)) ** 2 / (em.var(0) + ef.var(0) + 1e-12)
    t_n = taylor / (taylor.mean() + 1e-12)
    f_n = fisher / (fisher.mean() + 1e-12)
    importance = 0.5 * (t_n + f_n)
    return {"taylor": taylor, "fisher": fisher, "importance": importance}


def weighted_mean_shift(X: np.ndarray, w: np.ndarray, h: float, iters: int = 500, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel mean-shift where every point pulls with weight w; returns (labels, modes)."""
    modes = X.copy()
    for _ in range(iters):
        d2 = ((modes[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        k = w[None, :] * np.exp(-0.5 * d2 / h ** 2)
        new = (k @ X) / k.sum(1, keepdims=True)
        shift = np.abs(new - modes).max()
        modes = new
        if shift < tol:
            break
    labels = -np.ones(len(X), dtype=int)
    centers: List[np.ndarray] = []
    for i in np.argsort(-w):
        for c, ctr in enumerate(centers):
            if np.linalg.norm(modes[i] - ctr) < h / 2.0:
                labels[i] = c
                break
        else:
            centers.append(modes[i])
            labels[i] = len(centers) - 1
    return labels, np.asarray(centers)


def taskaware_erb_clustering(
    f1: np.ndarray,
    f2: np.ndarray,
    weights: np.ndarray,
    target_k: int,
) -> Dict[str, np.ndarray]:
    """Bisection on the ERB bandwidth h until weighted mean-shift leaves target_k clusters."""
    fc = (f1 + f2) / 2.0
    X = np.stack([hz_to_erb_rate(fc), hz_to_erb_rate(f2) - hz_to_erb_rate(f1)], axis=1)
    lo, hi = math.log(0.02), math.log(20.0)
    best = None
    for _ in range(40):
        h = math.exp(0.5 * (lo + hi))
        labels, _ = weighted_mean_shift(X, weights, h)
        k = int(labels.max()) + 1
        if best is None or abs(k - target_k) < abs(best[0] - target_k):
            best = (k, h, labels)
        if k == target_k:
            break
        if k > target_k:
            lo = math.log(h)
        else:
            hi = math.log(h)
    k, h, labels = best
    cf1, cf2 = [], []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        ww = weights[idx] / weights[idx].sum()
        cf1.append(float((ww * f1[idx]).sum()))
        cf2.append(float((ww * f2[idx]).sum()))
    return {"f1": np.array(cf1), "f2": np.array(cf2), "labels": labels, "erb_bandwidth": h, "features": X}


def build_init_A(cascade: GenderCNN, val_utts: List[Utterance], cfg: Config, device, target_k: int) -> Dict:
    imp = filter_importance(cascade, val_utts, cfg, device)
    f1, f2 = cascade.frontend.get_f1_f2()
    w = np.concatenate([imp["importance"], imp["importance"]])
    w = np.clip(w / w.mean(), 0.05, None)
    res = taskaware_erb_clustering(f1, f2, w, target_k)
    init = to_sinc2_init(res["f1"], res["f2"], cfg.min_freq_hz, cfg.sample_rate)
    init["extra"] = {
        "erb_bandwidth": res["erb_bandwidth"],
        "n_source_poles": int(len(f1)),
        "importance": imp["importance"].tolist(),
        "taylor": imp["taylor"].tolist(),
        "fisher": imp["fisher"].tolist(),
    }
    init["_plot"] = {"X": res["features"], "w": w, "labels": res["labels"]}
    return init


# ------------------------------------------------------------------ B: L0 gates
class HardConcreteGate(nn.Module):
    """Louizos et al. (2018) hard-concrete gate, one per filter."""

    def __init__(self, n: int, init_open: float = 0.9, beta: float = 2.0 / 3.0, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((n,), math.log(init_open / (1.0 - init_open))))
        self.beta, self.gamma, self.zeta = beta, gamma, zeta

    def forward(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def prob_open(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self.beta * math.log(-self.gamma / self.zeta))


class GatedFrontend(nn.Module):
    def __init__(self, frontend: nn.Module, gate: HardConcreteGate):
        super().__init__()
        self.inner = frontend
        self.gate = gate
        self.n_filt = frontend.n_filt

    def forward(self, x):
        return self.inner(x) * self.gate().view(1, -1, 1)


def train_l0_gates(
    cascade: GenderCNN,
    train_utts: List[Utterance],
    cfg: Config,
    device,
    target_k: int,
    out_dir: Path,
    epochs: int = 10,
    penalty: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Fine-tune the Cascade model with gates; resumable through out_dir/l0_last.pt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = cascade.frontend.n_filt
    gate = HardConcreteGate(n)
    cascade.frontend = GatedFrontend(cascade.frontend, gate)
    cascade.to(device)

    net_params = [p for name, p in cascade.named_parameters() if "gate." not in name]
    opt_net = torch.optim.RMSprop(net_params, lr=cfg.lr * 0.1, alpha=cfg.rmsprop_alpha, eps=cfg.rmsprop_eps)
    opt_gate = torch.optim.Adam(gate.parameters(), lr=1e-2)
    history = {"ce": [], "expected_open": []}
    start = 1
    last = out_dir / "l0_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        cascade.load_state_dict(ck["model"])
        opt_net.load_state_dict(ck["opt_net"])
        opt_gate.load_state_dict(ck["opt_gate"])
        history = ck["history"]
        start = int(ck["epoch"]) + 1
        print(f"[L0] resuming gate training from epoch {start}/{epochs}")

    cache = WaveCache(cfg.sample_rate, remove_sil=True)
    for u in tqdm(train_utts, desc="Loading/caching audio", leave=False):
        cache.get(u)
    ds = TrainChunkDataset(train_utts, cache, cfg.chunk_len, cfg.hop_len, chunks_per_utt=8, amp_fact=cfg.amp_fact)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(start, epochs + 1):
        cascade.train()
        ce_sum, n_b = 0.0, 0
        pbar = tqdm(loader, desc=f"L0 gates {epoch}/{epochs}")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            opt_net.zero_grad(set_to_none=True)
            opt_gate.zero_grad(set_to_none=True)
            ce = F.cross_entropy(cascade(x), y)
            exp_open = gate.prob_open().sum()
            loss = ce + penalty * F.relu(exp_open - target_k) / n
            loss.backward()
            opt_net.step()
            opt_gate.step()
            ce_sum += ce.item()
            n_b += 1
            pbar.set_postfix(ce=f"{ce_sum / n_b:.4f}", open=f"{exp_open.item():.1f}/{n}")
        history["ce"].append(ce_sum / max(1, n_b))
        history["expected_open"].append(float(gate.prob_open().sum().item()))
        torch.save(
            {
                "model": cascade.state_dict(),
                "opt_net": opt_net.state_dict(),
                "opt_gate": opt_gate.state_dict(),
                "history": history,
                "epoch": epoch,
                "arch": ARCH_VERSION,
            },
            last,
        )

    log_alpha = gate.log_alpha.detach().cpu().numpy()
    keep = np.sort(np.argsort(-log_alpha)[:target_k])
    frontend = cascade.frontend.inner
    cascade.frontend = frontend
    return {"log_alpha": log_alpha, "prob_open": gate.prob_open().detach().cpu().numpy(), "keep": keep, "history": history}


def build_init_B(cascade: GenderCNN, train_utts: List[Utterance], cfg: Config, device, target_k: int, out_dir: Path, epochs: int) -> Dict:
    res = train_l0_gates(cascade, train_utts, cfg, device, target_k, out_dir, epochs=epochs)
    f1, f2 = _bands_of_filters(cascade.frontend, res["keep"], cfg.sample_rate)
    init = to_sinc2_init(f1, f2, cfg.min_freq_hz, cfg.sample_rate)
    init["extra"] = {
        "kept_cascade_filters": res["keep"].tolist(),
        "log_alpha": res["log_alpha"].tolist(),
        "prob_open": res["prob_open"].tolist(),
        "gate_history": res["history"],
    }
    init["_plot"] = {"prob_open": res["prob_open"], "keep": res["keep"]}
    return init


# ------------------------------------------------------------------ C: response clustering
def build_init_C(cascade: GenderCNN, cfg: Config, target_k: int) -> Dict:
    from sklearn.cluster import AgglomerativeClustering

    mag, freqs = effective_responses(cascade.frontend, cfg.sample_rate)
    norm = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + 1e-12)
    dist = np.clip(1.0 - norm @ norm.T, 0.0, None)
    try:
        agg = AgglomerativeClustering(n_clusters=target_k, metric="precomputed", linkage="average")
    except TypeError:
        agg = AgglomerativeClustering(n_clusters=target_k, affinity="precomputed", linkage="average")
    labels = agg.fit_predict(dist)
    f1, f2 = [], []
    for c in range(target_k):
        rep = norm[labels == c].mean(axis=0)
        b1, b2 = band_from_response(rep, freqs)
        f1.append(b1)
        f2.append(b2)
    init = to_sinc2_init(np.array(f1), np.array(f2), cfg.min_freq_hz, cfg.sample_rate)
    init["extra"] = {"labels": labels.tolist()}
    init["_plot"] = {"mag": norm, "freqs": freqs, "labels": labels}
    return init


# ------------------------------------------------------------------ baseline (for D)
def build_init_meanshift(cascade: GenderCNN, cfg: Config) -> Dict:
    from .reduced import build_reduced_init_from_cascade

    base = build_reduced_init_from_cascade(cascade, cfg.sample_rate, cfg.reduced_bandwidth)
    return {
        "f1_hz": np.asarray(base["f1_hz"]),
        "f2_hz": np.asarray(base["f2_hz"]),
        "extra": {"meanshift_bandwidth": cfg.reduced_bandwidth, "n_source_poles": int(len(base["poles"]))},
    }


# ------------------------------------------------------------------ save / load / plots
def save_init(init: Dict, out_dir: Path, method: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "n_filters": int(len(init["f1_hz"])),
        "f1_hz": [float(v) for v in init["f1_hz"]],
        "f2_hz": [float(v) for v in init["f2_hz"]],
    }
    if "band_f1_hz" in init:
        payload["band_f1_hz"] = [float(v) for v in init["band_f1_hz"]]
        payload["band_f2_hz"] = [float(v) for v in init["band_f2_hz"]]
    payload["extra"] = init.get("extra", {})
    path = out_dir / "reduced_filters.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_init(out_dir: Path) -> Optional[Dict]:
    path = out_dir / "reduced_filters.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"f1_hz": np.array(data["f1_hz"]), "f2_hz": np.array(data["f2_hz"]), "method": data.get("method")}


def plot_filter_bank(init: Dict, cfg: Config, out_path: Path, title: str) -> None:
    bank = AFMConv1d(len(init["f1_hz"]), cfg.filt_len, cfg.sample_rate, "sinc2", cfg.min_freq_hz, init=init)
    mag, freqs = effective_responses(bank, cfg.sample_rate)
    mag_db = 20 * np.log10(mag / (mag.max(axis=1, keepdims=True) + 1e-12) + 1e-6)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for row in mag_db:
        ax.plot(freqs, row, lw=0.9, alpha=0.8)
    ax.set_ylim(-40, 2)
    ax.set_xlim(0, cfg.sample_rate / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, per-filter normalised)")
    ax.set_title(f"{title} ({len(init['f1_hz'])} sinc2 filters)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_method_details(method: str, init: Dict, out_dir: Path) -> None:
    info = init.get("_plot")
    if not info:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if method == "A":
        X, w, labels = info["X"], info["w"], info["labels"]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(X[:, 0], X[:, 1], s=20 + 80 * w / w.max(), c=labels, cmap="tab20", alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.set_xlabel("Centre frequency (ERB-rate)")
        ax.set_ylabel("Width (ERB)")
        ax.set_title("A: Cascade poles in ERB space (size = filter importance, colour = cluster)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "A_taskaware_erb_clusters.png", dpi=170)
        plt.close(fig)
    elif method == "B":
        p, keep = info["prob_open"], info["keep"]
        colors = ["C3" if i in set(keep.tolist()) else "C7" for i in range(len(p))]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(np.arange(len(p)), p, color=colors)
        ax.set_xlabel("Cascade filter index")
        ax.set_ylabel("P(gate open)")
        ax.set_title(f"B: L0 gate open probabilities (red = kept, {len(keep)} filters)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "B_l0_gates.png", dpi=170)
        plt.close(fig)
    elif method == "C":
        mag, freqs, labels = info["mag"], info["freqs"], info["labels"]
        order = np.argsort(labels, kind="stable")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(mag[order], aspect="auto", origin="lower", extent=[0, freqs[-1], 0, len(order)], cmap="magma")
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        for b in bounds:
            ax.axhline(b, color="w", lw=0.4)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cascade filters (grouped by cluster)")
        ax.set_title("C: effective Cascade responses grouped by cluster")
        fig.tight_layout()
        fig.savefig(fig_dir / "C_response_clusters.png", dpi=170)
        plt.close(fig)
