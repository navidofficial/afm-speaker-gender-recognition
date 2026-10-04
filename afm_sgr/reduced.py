"""
Mean-shift filter pruning -> Reduced model (paper Sec. 3.2.2 / 4.3), following the authors'
cluster_pole_loc.py + KernelCustomFreezeConv:

  1. take the (f1, f2) cutoffs of both Gaussian stages of every Cascade filter
  2. pole of each band: x = exp(-sigma + j w0), w0 = 2 pi (f1+f2) / (2 fs), sigma = 2 pi (f2-f1) / fs
  3. MeanShift(bandwidth=0.079) on (Re x, Im x)
  4. each cluster -> one filter with f1, f2 = mean of the members' f1, f2 (noise points kept as-is)
  5. Reduced frontend = sinc2 kernels initialised at those (f1, f2), amplitude (f2 - f1) / fs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift

from .filters import AFMConv1d
from .model import GenderCNN


def poles_from_f1_f2(f1_hz: np.ndarray, f2_hz: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    w0 = 2 * np.pi * (f1_hz + f2_hz) / (2 * sample_rate)
    sigma = 2 * np.pi * (f2_hz - f1_hz) / sample_rate
    x = np.exp(-sigma + 1j * w0)
    return np.stack([x.real, x.imag], axis=1), x


def cluster_poles(
    f1_hz: np.ndarray,
    f2_hz: np.ndarray,
    sample_rate: int,
    bandwidth: float = 0.079,
) -> Dict[str, np.ndarray]:
    feats, poles = poles_from_f1_f2(f1_hz, f2_hz, sample_rate)
    labels = MeanShift(bandwidth=bandwidth).fit(feats).labels_
    f1_c, f2_c, centers = [], [], []
    for lbl in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == lbl)
        if lbl == -1:
            for i in idx:
                f1_c.append(f1_hz[i])
                f2_c.append(f2_hz[i])
                centers.append(poles[i])
        else:
            f1_c.append(f1_hz[idx].mean())
            f2_c.append(f2_hz[idx].mean())
            c = feats[idx].mean(axis=0)
            centers.append(complex(c[0], c[1]))
    f1_c, f2_c, centers = np.asarray(f1_c), np.asarray(f2_c), np.asarray(centers)
    order = np.argsort((f1_c + f2_c) / 2.0)
    return {
        "f1_hz": f1_c[order],
        "f2_hz": f2_c[order],
        "centers": centers[order],
        "poles": poles,
        "labels": labels,
    }


def build_reduced_init_from_cascade(
    model: GenderCNN,
    sample_rate: int,
    bandwidth: float = 0.079,
) -> Dict[str, np.ndarray]:
    frontend = model.frontend
    if not (isinstance(frontend, AFMConv1d) and frontend.filter_type == "cascade"):
        raise TypeError("Reduced model needs a trained Cascade frontend")
    f1, f2 = frontend.get_f1_f2()
    return cluster_poles(f1, f2, sample_rate, bandwidth)


def save_reduced_init(init: Dict[str, np.ndarray], out_dir: Path, bandwidth: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reduced_filters.json"
    payload = {
        "n_source_poles": int(len(init["poles"])),
        "n_filters": int(len(init["f1_hz"])),
        "meanshift_bandwidth": bandwidth,
        "f1_hz": [float(v) for v in init["f1_hz"]],
        "f2_hz": [float(v) for v in init["f2_hz"]],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def plot_pole_reduction(init: Dict[str, np.ndarray], out_path: Path) -> None:
    poles, centers = init["poles"], init["centers"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    unit = np.exp(1j * np.linspace(0, 2 * np.pi, 400))
    for ax, pts, title, color in [
        (axes[0], poles, f"Cascade poles (n={len(poles)})", "C0"),
        (axes[1], centers, f"Reduced cluster centers (n={len(centers)})", "C3"),
    ]:
        ax.plot(unit.real, unit.imag, "k--", lw=0.8, alpha=0.5)
        ax.scatter(pts.real, pts.imag, s=28, c=color, edgecolors="white", linewidths=0.4, alpha=0.85)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Pole reduction on the complex plane (Mean-shift clustering)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
