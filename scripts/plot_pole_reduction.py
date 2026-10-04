"""Plot Cascade poles on the complex plane before/after mean-shift reduction."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afm_sgr.config import get_config
from afm_sgr.reduced import build_reduced_init_from_cascade, plot_pole_reduction
from afm_sgr.train import load_model_from_checkpoint


def main():
    cfg = get_config(frontend="cascade", run_name="cascade_timit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(cfg, Path(cfg.results_root) / "cascade_timit" / "best.pt", device)
    model.eval()

    init = build_reduced_init_from_cascade(model, cfg.sample_rate, cfg.reduced_bandwidth)
    out_path = Path(cfg.results_root) / "report_assets" / "pole_reduction_complex_plane.png"
    plot_pole_reduction(init, out_path)
    print(f"[ok] {len(init['poles'])} poles -> {len(init['f1_hz'])} clusters; saved {out_path}")


if __name__ == "__main__":
    main()
