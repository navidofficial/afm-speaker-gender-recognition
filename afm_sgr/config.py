"""Hyperparameters for AFM gender recognition, aligned with the authors' SGR_AFM code."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = Path(os.environ.get("AFM_RESULTS_DIR", PROJECT_ROOT / "results_github"))

# Bump when the network / frontend parameterization changes so stale last.pt files are not resumed.
ARCH_VERSION = "sgr_afm_github_v1"


@dataclass
class Config:
    # Paths
    data_root: Path = DATA_ROOT
    results_root: Path = RESULTS_ROOT
    timit_root: Path = DATA_ROOT / "TIMIT" / "data" / "lisa" / "data" / "timit" / "raw" / "TIMIT"
    libri_root: Path = DATA_ROOT / "LibriSpeech"
    ravdess_root: Path = DATA_ROOT / "RAVDESS"
    vox_root: Path = DATA_ROOT / "VoxCeleb1"
    vox_meta: Path = DATA_ROOT / "VoxCeleb1" / "vox1_meta.csv"

    # Audio / chunking (cw_len=200, cw_shift=10)
    sample_rate: int = 16000
    chunk_ms: float = 200.0
    hop_ms: float = 10.0
    exclude_timit_sa: bool = True

    # Model (KernelCNN_TIMIT.cfg)
    frontend: str = "cascade"  # free|mel|sinc|sinc2|gamma|gauss|cascade|reduced
    n_filt: int = 80
    filt_len: int = 257
    conv2_filt: int = 60
    conv2_kernel: int = 5
    conv3_filt: int = 60
    conv3_kernel: int = 5
    pool_size: int = 3
    fc_dim: int = 2048
    n_classes: int = 2
    leaky_slope: float = 0.2
    min_freq_hz: float = 50.0

    # Train
    batch_size: int = 128
    lr: float = 1e-3
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 1e-8
    amp_fact: float = 0.2  # random gain in [1-amp_fact, 1+amp_fact] per training chunk
    epochs: int = 50
    finetune_epochs: int = 20
    optimizer: str = "rmsprop"
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    val_ratio_from_train: float = 0.2
    early_stop_patience: int = 0

    # Reduced filter bank (cluster_pole_loc.py)
    reduced_bandwidth: float = 0.079
    reduced_freeze: bool = True
    reduced_n_clusters: int = 40  # informational only; MeanShift bandwidth decides the count

    # Eval / analysis
    run_name: str = "cascade_timit"
    save_every: int = 50

    @property
    def chunk_len(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000.0)

    @property
    def hop_len(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000.0)

    def run_dir(self) -> Path:
        d = Path(self.results_root) / self.run_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self):
        out = {}
        for k, v in asdict(self).items():
            out[k] = str(v) if isinstance(v, Path) else v
        return out


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown config key: {k}")
        setattr(cfg, k, v)
    return cfg
