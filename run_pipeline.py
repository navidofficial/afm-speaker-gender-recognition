"""
Full paper pipeline (single run).

Examples:
  python run_pipeline.py --stage smoke
  python run_pipeline.py --stage timit --frontend cascade
  python run_pipeline.py --stage all --frontend cascade
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afm_sgr.analyze import run_paper_analyses
from afm_sgr.audio import set_seed
from afm_sgr.config import get_config
from afm_sgr.datasets import (
    discover_librispeech,
    discover_ravdess,
    discover_timit,
    discover_voxceleb_test,
    split_timit_train_val,
)
from afm_sgr.reduced import build_reduced_init_from_cascade, plot_pole_reduction, save_reduced_init
from afm_sgr.train import (
    apply_saved_run_cfg,
    build_model,
    evaluate,
    fit,
    load_model_from_checkpoint,
    make_loader,
)


def _device(cfg):
    return torch.device(cfg.device if torch.cuda.is_available() else "cpu")


def prepare_timit(cfg):
    utts = discover_timit(cfg.timit_root, exclude_sa=cfg.exclude_timit_sa)
    tr, va, te = split_timit_train_val(utts, cfg.val_ratio_from_train, cfg.seed)
    print(f"TIMIT utterances: train={len(tr)} val={len(va)} test={len(te)} (total {len(utts)})")
    return tr, va, te


def eval_and_save(model, utts, cfg, device, name: str):
    loader = make_loader(utts, cfg, shuffle=False)
    metrics = evaluate(model, loader, device, desc=f"Eval:{name}")
    out = cfg.run_dir() / f"metrics_{name}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[metrics:{name}] CER={metrics['cer']:.2f}% FER={metrics['fer']:.2f}% Acc={metrics['accuracy']:.2f}%")
    return metrics


def _mark_done(run_dir: Path, stage: str, extra: dict | None = None):
    payload = {"stage": stage, "done": True}
    if extra:
        payload.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STAGE_DONE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_done(run_dir: Path) -> bool:
    return (run_dir / "STAGE_DONE.json").exists()


def _last_epoch(run_dir: Path) -> int:
    last = run_dir / "last.pt"
    if not last.exists():
        return 0
    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        return int(ckpt.get("epoch", 0))
    except Exception:
        return 0


_apply_saved_run_cfg = apply_saved_run_cfg
_load_checkpoint_file = load_model_from_checkpoint


def _load_timit_checkpoint(cfg, device) -> torch.nn.Module:
    return _load_checkpoint_file(cfg, cfg.run_dir() / "best.pt", device)


def _path_cfg(**overrides):
    cfg = get_config(**overrides)
    for k in ["data_root", "results_root", "timit_root", "libri_root", "ravdess_root", "vox_root", "vox_meta"]:
        setattr(cfg, k, Path(getattr(cfg, k)))
    return cfg


def _eval_if_needed(model, utts, cfg, device, name: str, force: bool = False):
    """Resume-friendly eval: skip if metrics_*.json already exists."""
    out = cfg.run_dir() / f"metrics_{name}.json"
    if not force and out.exists():
        metrics = json.loads(out.read_text(encoding="utf-8"))
        print(
            f"[skip-eval:{name}] already have metrics "
            f"CER={metrics.get('cer', 0):.2f}% Acc={metrics.get('accuracy', 0):.2f}%"
        )
        return metrics
    return eval_and_save(model, utts, cfg, device, name)


def stage_timit(cfg, epochs=None, resume: bool = True, force: bool = False):
    set_seed(cfg.seed)
    out = cfg.run_dir()
    n_epochs = epochs if epochs is not None else cfg.epochs
    done_enough = _is_done(out) and (out / "metrics_timit_test.json").exists() and (
        _last_epoch(out) >= n_epochs or not (out / "last.pt").exists()
    )
    if not force and done_enough:
        print(f"[skip] TIMIT already done ({n_epochs} epochs): {out}")
        return None, json.loads((out / "metrics_timit_test.json").read_text(encoding="utf-8"))

    # If target epochs increased, clear done flag and continue from last.pt
    if (out / "STAGE_DONE.json").exists() and _last_epoch(out) < n_epochs and (out / "last.pt").exists():
        print(f"[resume] more epochs requested ({_last_epoch(out)} -> {n_epochs}); continuing")
        (out / "STAGE_DONE.json").unlink(missing_ok=True)

    # Training finished earlier, but eval / STAGE_DONE missing → eval only
    if (
        not force
        and resume
        and (out / "best.pt").exists()
        and _last_epoch(out) >= n_epochs
    ):
        print(f"[resume] TIMIT training complete @ epoch {_last_epoch(out)}; finishing eval if needed")
        device = _device(cfg)
        _, _, te = prepare_timit(cfg)
        model = _load_checkpoint_file(cfg, out / "best.pt", device)
        metrics = _eval_if_needed(model, te, cfg, device, "timit_test", force=force)
        run_paper_analyses(model, te, cfg, device, tag=cfg.frontend)
        _mark_done(out, "timit", {"frontend": cfg.frontend, "epochs": n_epochs, "metrics": metrics})
        return model, metrics

    tr, va, te = prepare_timit(cfg)
    model, history = fit(cfg, tr, va, epochs=epochs, resume=resume and not force)
    device = _device(cfg)
    metrics = _eval_if_needed(model, te, cfg, device, "timit_test", force=force)
    run_paper_analyses(model, te, cfg, device, tag=cfg.frontend)
    _mark_done(out, "timit", {"frontend": cfg.frontend, "epochs": n_epochs, "metrics": metrics})
    return model, metrics


def stage_finetune(cfg, model=None, resume: bool = True, force: bool = False):
    set_seed(cfg.seed)
    device = _device(cfg)

    # Align architecture with the finished TIMIT run before building finetune cfg
    _apply_saved_run_cfg(cfg, cfg.run_dir())

    cfg_ft = get_config(**{**cfg.to_dict(), "run_name": cfg.run_name + "_finetune", "epochs": cfg.finetune_epochs})
    for k in ["data_root", "results_root", "timit_root", "libri_root", "ravdess_root", "vox_root", "vox_meta"]:
        setattr(cfg_ft, k, Path(getattr(cfg_ft, k)))
    cfg_ft.frontend = cfg.frontend
    cfg_ft.n_filt = cfg.n_filt
    cfg_ft.epochs = cfg.finetune_epochs

    out = cfg_ft.run_dir()
    rav_m = out / "metrics_ravdess.json"
    vox_m = out / "metrics_voxceleb1_test.json"
    if not force and rav_m.exists() and vox_m.exists():
        if not _is_done(out):
            _mark_done(out, "finetune", {"frontend": cfg.frontend, "backfilled": True})
        print(f"[skip] Fine-tune already done: {out}")
        return None

    do_resume = resume and not force
    ft_last = out / "last.pt"
    ft_best = out / "best.pt"

    # Prefer in-progress / finished finetune weights over reloading TIMIT
    if model is None:
        if do_resume and ft_last.exists():
            print(f"[resume] fine-tune from {ft_last} @ epoch {_last_epoch(out)}")
            model = _load_checkpoint_file(cfg_ft, ft_last, device)
        elif do_resume and ft_best.exists() and _last_epoch(out) >= cfg.finetune_epochs:
            print(f"[resume] fine-tune training done; loading {ft_best}")
            model = _load_checkpoint_file(cfg_ft, ft_best, device)
        else:
            model = _load_timit_checkpoint(cfg, device)

    # Skip training loop if already finished; still run missing evals
    need_train = force or _last_epoch(out) < cfg.finetune_epochs or not ft_best.exists()
    if need_train:
        libri = discover_librispeech(cfg.libri_root, ["dev-clean", "test-clean"])
        dev = [u for u in libri if u.split == "dev-clean"]
        test_l = [u for u in libri if u.split == "test-clean"]
        speakers = sorted({u.speaker for u in dev})
        import random

        rng = random.Random(cfg.seed)
        rng.shuffle(speakers)
        n_val = max(1, len(speakers) // 5)
        val_spk = set(speakers[:n_val])
        ft_train = [u for u in dev if u.speaker not in val_spk]
        ft_val = [u for u in dev if u.speaker in val_spk] + test_l[:200]
        print(f"Libri fine-tune: train_utts={len(ft_train)} val_utts={len(ft_val)}")

        model, _ = fit(
            cfg_ft,
            ft_train,
            ft_val,
            model=model,
            epochs=cfg.finetune_epochs,
            resume=do_resume,
        )
    else:
        print(f"[resume] skipping fine-tune training (already {_last_epoch(out)}/{cfg.finetune_epochs} epochs)")
        if ft_best.exists():
            model = _load_checkpoint_file(cfg_ft, ft_best, device)

    rav = discover_ravdess(cfg.ravdess_root)
    vox = discover_voxceleb_test(cfg.vox_root, cfg.vox_meta)
    _eval_if_needed(model, rav, cfg_ft, device, "ravdess", force=force)
    _eval_if_needed(model, vox, cfg_ft, device, "voxceleb1_test", force=force)
    run_paper_analyses(model, rav[:], cfg_ft, device, tag=f"{cfg.frontend}_ravdess")
    _mark_done(out, "finetune", {"frontend": cfg.frontend})
    return model


def stage_reduced(cfg, resume: bool = True, force: bool = False):
    set_seed(cfg.seed)
    device = _device(cfg)
    red_dir = cfg.results_root / "reduced_timit"
    if not force and (red_dir / "metrics_timit_test.json").exists():
        if not _is_done(red_dir):
            red_dir.mkdir(parents=True, exist_ok=True)
            _mark_done(red_dir, "reduced", {"backfilled": True})
        print(f"[skip] Reduced already done: {red_dir}")
        return None

    do_resume = resume and not force
    red_cfg = _path_cfg(frontend="reduced", run_name="reduced_timit")
    red_cfg.epochs = cfg.epochs
    red_cfg.batch_size = cfg.batch_size
    red_cfg.num_workers = cfg.num_workers
    red_cfg.device = cfg.device
    _apply_saved_run_cfg(red_cfg, red_dir)

    tr, va, te = prepare_timit(cfg)

    # Resume mid-training / mid-eval without rebuilding mean-shift filters
    if do_resume and ((red_dir / "last.pt").exists() or (red_dir / "best.pt").exists()):
        ckpt = red_dir / "last.pt" if (red_dir / "last.pt").exists() else red_dir / "best.pt"
        print(f"[resume] Reduced from {ckpt.name} @ epoch {_last_epoch(red_dir)}")
        model = _load_checkpoint_file(red_cfg, ckpt, device)
        if _last_epoch(red_dir) < cfg.epochs or force:
            model, _ = fit(
                red_cfg, tr, va, model=model, epochs=cfg.epochs, resume=do_resume
            )
        else:
            print(f"[resume] Reduced training complete; finishing eval if needed")
            if (red_dir / "best.pt").exists():
                model = _load_checkpoint_file(red_cfg, red_dir / "best.pt", device)
        metrics = _eval_if_needed(model, te, red_cfg, device, "timit_test", force=force)
        run_paper_analyses(model, te, red_cfg, device, tag="reduced")
        _mark_done(red_cfg.run_dir(), "reduced", {"metrics": metrics})
        return model

    cascade_dir = cfg.results_root / "cascade_timit"
    ckpt_path = cascade_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Need cascade checkpoint at {ckpt_path}. Run cascade training first.")

    base_cfg = _path_cfg(frontend="cascade", run_name="cascade_timit")
    cascade = load_model_from_checkpoint(base_cfg, ckpt_path, device)

    init = build_reduced_init_from_cascade(cascade, cfg.sample_rate, red_cfg.reduced_bandwidth)
    red_dir.mkdir(parents=True, exist_ok=True)
    save_reduced_init(init, red_dir, red_cfg.reduced_bandwidth)
    plot_pole_reduction(init, red_dir / "figures" / "pole_reduction_complex_plane.png")
    print(
        f"Reduced filter bank: {len(init['poles'])} cascade poles -> {len(init['f1_hz'])} filters "
        f"(MeanShift bandwidth={red_cfg.reduced_bandwidth})"
    )

    red_cfg.n_filt = int(len(init["f1_hz"]))
    red = build_model(red_cfg, frontend_init={"f1_hz": init["f1_hz"], "f2_hz": init["f2_hz"]}).to(device)
    if red_cfg.reduced_freeze:
        red.freeze_frontend()
    model_ft, _ = fit(
        red_cfg, tr, va, model=red, epochs=cfg.epochs, resume=False
    )
    metrics = _eval_if_needed(model_ft, te, red_cfg, device, "timit_test", force=force)
    run_paper_analyses(model_ft, te, red_cfg, device, tag="reduced")
    _mark_done(red_cfg.run_dir(), "reduced", {"metrics": metrics})
    return model_ft


def stage_smoke(cfg):
    """Fast sanity check on a tiny TIMIT subset."""
    set_seed(cfg.seed)
    tr, va, te = prepare_timit(cfg)

    def balanced(utts, n):
        import random

        rng = random.Random(cfg.seed)
        by_g = [[u for u in utts if u.gender == g] for g in (0, 1)]
        for g in by_g:
            rng.shuffle(g)
        return by_g[0][: n // 2] + by_g[1][: n // 2]

    tr, va, te = balanced(tr, 64), balanced(va, 16), balanced(te, 16)
    cfg = get_config(
        frontend=cfg.frontend,
        run_name=f"smoke_{cfg.frontend}",
        epochs=2,
        batch_size=16,
        num_workers=0,
    )
    for k in ["data_root", "results_root", "timit_root", "libri_root", "ravdess_root", "vox_root", "vox_meta"]:
        setattr(cfg, k, Path(getattr(cfg, k)))
    model, _ = fit(cfg, tr, va, epochs=2)
    device = _device(cfg)
    eval_and_save(model, te, cfg, device, "smoke_test")
    print("[smoke] OK")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["smoke", "timit", "finetune", "reduced", "all"], default="smoke")
    p.add_argument("--frontend", default="cascade",
                   choices=["free", "mel", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"])
    p.add_argument("--epochs", type=int, default=None, help="Override paper epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true", default=True, help="Resume from last.pt (default)")
    p.add_argument("--no-resume", action="store_true", help="Ignore last.pt and train from scratch")
    p.add_argument("--force", action="store_true", help="Re-run even if STAGE_DONE.json exists")
    args = p.parse_args()

    resume = not args.no_resume
    force = args.force

    cfg = get_config(
        frontend=args.frontend,
        run_name=f"{args.frontend}_timit",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.stage == "smoke":
        stage_smoke(cfg)
    elif args.stage == "timit":
        stage_timit(cfg, epochs=args.epochs, resume=resume, force=force)
    elif args.stage == "finetune":
        stage_finetune(cfg, resume=resume, force=force)
    elif args.stage == "reduced":
        stage_reduced(cfg, resume=resume, force=force)
    elif args.stage == "all":
        stage_timit(cfg, epochs=args.epochs, resume=resume, force=force)
        stage_finetune(cfg, resume=resume, force=force)
        if args.frontend == "cascade":
            stage_reduced(cfg, resume=resume, force=force)


if __name__ == "__main__":
    main()
