from __future__ import annotations

import json
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
from .datasets import Utterance
from .model import GenderCNN

ARCH_KEYS = ("frontend", "n_filt", "filt_len", "conv2_filt", "conv3_filt", "fc_dim")


def build_model(cfg: Config, frontend_init: Optional[Dict] = None) -> GenderCNN:
    return GenderCNN(
        frontend=cfg.frontend,
        sample_rate=cfg.sample_rate,
        n_filt=cfg.n_filt,
        filt_len=cfg.filt_len,
        conv2_filt=cfg.conv2_filt,
        conv2_kernel=cfg.conv2_kernel,
        conv3_filt=cfg.conv3_filt,
        conv3_kernel=cfg.conv3_kernel,
        pool_size=cfg.pool_size,
        fc_dim=cfg.fc_dim,
        n_classes=cfg.n_classes,
        leaky_slope=cfg.leaky_slope,
        chunk_len=cfg.chunk_len,
        min_freq_hz=cfg.min_freq_hz,
        frontend_init=frontend_init,
    )


def apply_saved_run_cfg(cfg: Config, run_dir: Path) -> None:
    """Overlay architecture keys from a run's config.json (needed for Reduced's filter count)."""
    cfg_path = Path(run_dir) / "config.json"
    if not cfg_path.exists():
        return
    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ARCH_KEYS:
        if key in saved and saved[key] is not None:
            setattr(cfg, key, saved[key])


def model_from_state(cfg: Config, state: dict, device) -> GenderCNN:
    """Rebuild the architecture matching a state dict (filter count is read from the weights)."""
    if "frontend.norm_f1" in state:
        cfg.n_filt = int(state["frontend.norm_f1"].shape[0])
    model = build_model(cfg).to(device)
    model.load_state_dict(state, strict=True)
    if cfg.frontend == "reduced" and cfg.reduced_freeze:
        model.freeze_frontend()
    return model


def load_model_from_checkpoint(cfg: Config, ckpt_path: Path, device) -> GenderCNN:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    apply_saved_run_cfg(cfg, ckpt_path.parent)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("arch") not in (None, ARCH_VERSION):
        raise RuntimeError(
            f"{ckpt_path} was trained with architecture '{ckpt.get('arch')}', current is '{ARCH_VERSION}'"
        )
    return model_from_state(cfg, ckpt["model"], device)


def make_loader(utts: List[Utterance], cfg: Config, shuffle: bool, cache=None) -> DataLoader:
    from .datasets import WaveCache, TrainChunkDataset, EvalChunkDataset

    if cache is None:
        cache = WaveCache(cfg.sample_rate, remove_sil=True)
    for u in tqdm(utts, desc="Loading/caching audio", leave=False):
        cache.get(u)

    if shuffle:
        ds = TrainChunkDataset(utts, cache, cfg.chunk_len, cfg.hop_len, chunks_per_utt=8, amp_fact=cfg.amp_fact)
    else:
        print(f"Building eval index ({len(utts)} utterances, hop={cfg.hop_len})...")
        ds = EvalChunkDataset(utts, cache, cfg.chunk_len, cfg.hop_len)
        print(f"Eval chunks: {len(ds)}")
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,  # required: waveform cache lives in-process
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # BatchNorm cannot train on a batch of one
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
) -> Dict[str, float]:
    """
    SIDTrainer.test protocol: per utterance, frame error = mean chunk error, loss = mean chunk NLL,
    decision = argmax of the summed chunk log-probabilities. FER / loss are averaged over utterances.
    """
    model.eval()
    all_logp, all_y, all_u = [], [], []
    for x, y, u_idx in tqdm(loader, desc=desc, leave=False):
        logits = model(x.to(device))
        all_logp.append(F.log_softmax(logits, dim=1).cpu())
        all_y.append(y)
        all_u.append(u_idx)

    logp = torch.cat(all_logp)
    y = torch.cat(all_y).long()
    u = torch.cat(all_u).long()
    pred = logp.argmax(dim=1)

    _, inv = torch.unique(u, return_inverse=True)
    n_utt = int(inv.max().item()) + 1
    counts = torch.bincount(inv, minlength=n_utt).float()
    frame_err = torch.zeros(n_utt).index_add_(0, inv, (pred != y).float()) / counts
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    utt_loss = torch.zeros(n_utt).index_add_(0, inv, nll) / counts
    sum_logp = torch.zeros(n_utt, logp.shape[1]).index_add_(0, inv, logp)
    utt_y = torch.zeros(n_utt, dtype=torch.long)
    utt_y[inv] = y
    utt_acc = (sum_logp.argmax(dim=1) == utt_y).float().mean().item()

    fer = frame_err.mean().item()
    return {
        "cer": (1.0 - utt_acc) * 100.0,
        "fer": fer * 100.0,
        "accuracy": utt_acc * 100.0,
        "frame_accuracy": (1.0 - fer) * 100.0,
        "loss": utt_loss.mean().item(),
        "n_utt": float(n_utt),
        "n_chunks": float(len(y)),
    }


def kd_loss(student_logits, teacher_logits, y, criterion, alpha: float, temperature: float):
    """Hinton distillation: (1 - alpha) * CE(y) + alpha * T^2 * KL(teacher_T || student_T)."""
    ce = criterion(student_logits, y)
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    )
    return (1.0 - alpha) * ce + alpha * temperature ** 2 * kl


def train_one_epoch(
    model,
    loader,
    optim,
    criterion,
    device,
    epoch: int = 1,
    n_epochs: int = 1,
    teacher: Optional[nn.Module] = None,
    kd_alpha: float = 0.5,
    kd_temperature: float = 4.0,
) -> Tuple[float, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc=f"Train {epoch}/{n_epochs}", leave=True)
    for x, y, _ in pbar:
        x = x.to(device)
        y = y.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(x)
            loss = kd_loss(logits, t_logits, y, criterion, kd_alpha, kd_temperature)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        avg_loss = total_loss / max(1, total)
        avg_acc = total_correct / max(1, total)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc*100:.1f}%")
    return total_loss / max(1, total), total_correct / max(1, total)


def plot_history(history: Dict[str, List[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="train_loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL Loss")
    ax.set_title("Loss per epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_per_epoch.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="train_acc")
    if history.get("val_acc"):
        ax.plot(epochs, history["val_acc"], label="val_accuracy (utt %)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_per_epoch.png", dpi=150)
    plt.close(fig)


def fit(
    cfg: Config,
    train_utts: List[Utterance],
    val_utts: List[Utterance],
    model: Optional[GenderCNN] = None,
    epochs: Optional[int] = None,
    resume: bool = True,
    teacher: Optional[nn.Module] = None,
    kd_alpha: float = 0.5,
    kd_temperature: float = 4.0,
) -> Tuple[GenderCNN, Dict]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = cfg.run_dir()
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    if model is None:
        model = build_model(cfg)
    model = model.to(device)

    from .datasets import WaveCache

    cache = WaveCache(cfg.sample_rate, remove_sil=True)
    train_loader = make_loader(train_utts, cfg, shuffle=True, cache=cache)
    val_loader = make_loader(val_utts, cfg, shuffle=False, cache=cache)

    crit = nn.CrossEntropyLoss()  # == NLLLoss(LogSoftmax(logits))
    optim = torch.optim.RMSprop(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        alpha=cfg.rmsprop_alpha,
        eps=cfg.rmsprop_eps,
    )

    n_epochs = epochs if epochs is not None else cfg.epochs
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_cer": [],
        "val_fer": [],
    }
    best_acc = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    start_epoch = 1

    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt.get("arch") != ARCH_VERSION:
            print(f"[resume] {last_path} is from architecture '{ckpt.get('arch')}'; starting fresh")
        else:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                try:
                    optim.load_state_dict(ckpt["optimizer"])
                except Exception as e:
                    print(f"[resume] optimizer state not loaded ({e}); continuing with fresh optim")
            history = ckpt.get("history", history)
            best_acc = float(ckpt.get("best_acc", -1.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[resume] {cfg.run_name}: loaded {last_path.name} @ epoch {ckpt.get('epoch')}")
            print(f"[resume] continuing from epoch {start_epoch}/{n_epochs} (best_acc={best_acc:.2f})")
            if start_epoch > n_epochs:
                print(f"[resume] already finished {n_epochs} epochs; skipping training loop")
                if best_path.exists():
                    model.load_state_dict(
                        torch.load(best_path, map_location=device, weights_only=False)["model"]
                    )
                return model, history

    for epoch in range(start_epoch, n_epochs + 1):
        print(f"\n===== [{cfg.run_name}] Epoch {epoch}/{n_epochs} =====")
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optim,
            crit,
            device,
            epoch=epoch,
            n_epochs=n_epochs,
            teacher=teacher,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )
        metrics = evaluate(model, val_loader, device, desc=f"Val {epoch}/{n_epochs}")

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(metrics["loss"])
        history["val_acc"].append(metrics["accuracy"])
        history["val_cer"].append(metrics["cer"])
        history["val_fer"].append(metrics["fer"])

        print(
            f"[{cfg.run_name}] epoch {epoch}/{n_epochs} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% "
            f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.2f}% "
            f"CER={metrics['cer']:.2f}% FER={metrics['fer']:.2f}%"
        )

        if metrics["accuracy"] >= best_acc:
            best_acc = metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.to_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "best_acc": best_acc,
                    "arch": ARCH_VERSION,
                },
                best_path,
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_acc": best_acc,
                "cfg": cfg.to_dict(),
                "arch": ARCH_VERSION,
            },
            last_path,
        )
        plot_history(history, out_dir)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if epoch % max(1, cfg.save_every) == 0 or epoch == n_epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch, "arch": ARCH_VERSION}, out_dir / f"epoch_{epoch}.pt")

    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model"]
        )
    plot_history(history, out_dir)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return model, history
