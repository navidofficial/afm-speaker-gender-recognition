"""
New Reduced-model variants built from the already-trained Cascade checkpoint
(results_github/cascade_timit/best.pt). Nothing before pole reduction is retrained.

  A   task-aware ERB-warped weighted mean-shift      (frozen sinc2, CE)
  B   learnable L0 gates on the Cascade filters      (frozen sinc2, CE)
  C   clustering of effective Cascade responses      (frozen sinc2, CE)
  D   paper Mean-shift filters + distillation + trainable f1/f2
  AD  A filters + distillation + trainable f1/f2

Every variant keeps K = number of filters of the paper's Mean-shift Reduced model.

Examples:
  python run_new_reduced.py --variant A --stage timit
  python run_new_reduced.py --variant A --stage finetune
  python run_new_reduced.py --stage table
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afm_sgr.analyze import run_paper_analyses
from afm_sgr.audio import set_seed
from afm_sgr.reduced_new import (
    build_init_A,
    build_init_B,
    build_init_C,
    build_init_meanshift,
    load_init,
    plot_filter_bank,
    plot_method_details,
    save_init,
)
from afm_sgr.train import build_model, fit, load_model_from_checkpoint
from run_pipeline import (
    _device,
    _eval_if_needed,
    _is_done,
    _mark_done,
    _path_cfg,
    prepare_timit,
    stage_finetune,
)

VARIANTS = {
    "A": {"select": "A", "kd": False, "freeze": True, "label": "A: task-aware ERB mean-shift"},
    "B": {"select": "B", "kd": False, "freeze": True, "label": "B: L0 gates"},
    "C": {"select": "C", "kd": False, "freeze": True, "label": "C: response clustering"},
    "D": {"select": "meanshift", "kd": True, "freeze": False, "label": "D: Mean-shift + KD + trainable f1/f2"},
    "AD": {"select": "A", "kd": True, "freeze": False, "label": "A+D: task-aware + KD + trainable f1/f2"},
}


def run_name(variant: str) -> str:
    return f"reduced_{variant}_timit"


def variant_cfg(variant: str, args):
    cfg = _path_cfg(frontend="reduced", run_name=run_name(variant))
    cfg.batch_size = args.batch_size
    cfg.device = args.device
    cfg.num_workers = 0
    cfg.reduced_freeze = VARIANTS[variant]["freeze"]
    if args.epochs is not None:
        cfg.epochs = args.epochs
    return cfg


def load_cascade(cfg, device):
    ckpt = cfg.results_root / "cascade_timit" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Need the trained Cascade model at {ckpt}. Run run_all_paper.bat first.")
    base = _path_cfg(frontend="cascade", run_name="cascade_timit")
    base.batch_size = cfg.batch_size
    base.device = cfg.device
    return load_model_from_checkpoint(base, ckpt, device), base


def get_or_build_init(variant: str, cfg, args, device, tr, va):
    out = cfg.run_dir()
    saved = load_init(out)
    if saved is not None:
        print(f"[init] reusing {out / 'reduced_filters.json'} ({len(saved['f1_hz'])} filters)")
        return saved

    select = VARIANTS[variant]["select"]
    if select == "A" and variant != "A":
        shared = load_init(cfg.results_root / run_name("A"))
        if shared is not None:
            print(f"[init] variant {variant}: reusing the filters selected by variant A")
            src = json.loads((cfg.results_root / run_name("A") / "reduced_filters.json").read_text(encoding="utf-8"))
            src["method"] = VARIANTS[variant]["label"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "reduced_filters.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
            plot_filter_bank(shared, cfg, out / "figures" / "reduced_filter_bank.png", VARIANTS[variant]["label"])
            return shared

    cascade, base_cfg = load_cascade(cfg, device)
    target_k = len(build_init_meanshift(cascade, cfg)["f1_hz"])
    print(f"[init] variant {variant}: building {target_k} filters with method '{select}'")
    if select == "A":
        init = build_init_A(cascade, va, base_cfg, device, target_k)
    elif select == "B":
        init = build_init_B(cascade, tr, base_cfg, device, target_k, out / "l0_gates", epochs=args.gate_epochs)
    elif select == "C":
        init = build_init_C(cascade, cfg, target_k)
    else:
        init = build_init_meanshift(cascade, cfg)
    save_init(init, out, VARIANTS[variant]["label"])
    plot_method_details(select, init, out)
    plot_filter_bank(init, cfg, out / "figures" / "reduced_filter_bank.png", VARIANTS[variant]["label"])
    return init


def stage_timit(variant: str, args):
    cfg = variant_cfg(variant, args)
    set_seed(cfg.seed)
    device = _device(cfg)
    out = cfg.run_dir()
    if not args.force and _is_done(out) and (out / "metrics_timit_test.json").exists():
        print(f"[skip] {out.name} already complete")
        return

    tr, va, te = prepare_timit(cfg)
    init = get_or_build_init(variant, cfg, args, device, tr, va)
    cfg.n_filt = int(len(init["f1_hz"]))
    model = build_model(cfg, frontend_init={"f1_hz": init["f1_hz"], "f2_hz": init["f2_hz"]}).to(device)
    if cfg.reduced_freeze:
        model.freeze_frontend()

    teacher = None
    if VARIANTS[variant]["kd"]:
        teacher, _ = load_cascade(cfg, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    model, _ = fit(
        cfg,
        tr,
        va,
        model=model,
        epochs=cfg.epochs,
        resume=not args.force,
        teacher=teacher,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
    )
    metrics = _eval_if_needed(model, te, cfg, device, "timit_test", force=args.force)
    run_paper_analyses(model, te, cfg, device, tag=f"reduced_{variant}")
    _mark_done(out, "reduced_new", {"variant": variant, "metrics": metrics})


def stage_ft(variant: str, args):
    cfg = variant_cfg(variant, args)
    if not (cfg.run_dir() / "best.pt").exists():
        raise FileNotFoundError(f"Train {run_name(variant)} on TIMIT first")
    stage_finetune(cfg, resume=not args.force, force=args.force)


# ------------------------------------------------------------------ comparison table
def _metric(run_dir: Path, name: str):
    p = run_dir / f"metrics_{name}.json"
    if not p.exists():
        return None
    return round(float(json.loads(p.read_text(encoding="utf-8"))["accuracy"]), 2)


def _n_filters(run_dir: Path, default=None):
    p = run_dir / "reduced_filters.json"
    if p.exists():
        return int(json.loads(p.read_text(encoding="utf-8"))["n_filters"])
    return default


def stage_table(args):
    cfg = _path_cfg()
    res = Path(cfg.results_root)
    entries = [("cascade_timit", "Cascade (teacher, 80 filters)", 80), ("reduced_timit", "Paper Reduced (Mean-shift)", None)]
    entries += [(run_name(v), VARIANTS[v]["label"], None) for v in VARIANTS]
    rows = []
    for rn, label, n in entries:
        d = res / rn
        ft = res / f"{rn}_finetune"
        rows.append(
            {
                "run": rn,
                "method": label,
                "n_filters": _n_filters(d, n),
                "TIMIT_acc_%": _metric(d, "timit_test"),
                "RAVDESS_acc_%": _metric(ft, "ravdess"),
                "VoxCeleb1_acc_%": _metric(ft, "voxceleb1_test"),
            }
        )
    out_dir = res / "new_reduced_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "-" if v is None else str(v)
    lines = ["# Reduced-model variants", "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in fields) + " |" for r in rows]
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sets = ["TIMIT_acc_%", "RAVDESS_acc_%", "VoxCeleb1_acc_%"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, s in enumerate(sets):
        vals = [r[s] if r[s] is not None else 0 for r in rows]
        bars = ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=s.replace("_acc_%", ""))
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['method']}\n({fmt(r['n_filters'])} filters)" for r in rows], fontsize=7)
    ax.set_ylabel("Utterance accuracy (%)")
    lo = min([r[s] for r in rows for s in sets if r[s]] or [50])
    ax.set_ylim(max(0, lo - 10), 101)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Reduced-model variants")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=170)
    plt.close(fig)
    print("\n".join(lines))
    print(f"\n[ok] {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="A")
    p.add_argument("--stage", choices=["timit", "finetune", "all", "table"], default="all")
    p.add_argument("--epochs", type=int, default=None, help="Override the 50 TIMIT epochs")
    p.add_argument("--gate-epochs", type=int, default=10, help="Epochs of L0 gate training (variant B)")
    p.add_argument("--kd-alpha", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true", help="Ignore checkpoints / finished metrics")
    args = p.parse_args()

    if args.stage == "table":
        stage_table(args)
        return
    if args.stage in ("timit", "all"):
        stage_timit(args.variant, args)
    if args.stage in ("finetune", "all"):
        stage_ft(args.variant, args)


if __name__ == "__main__":
    main()
