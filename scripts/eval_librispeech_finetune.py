"""
Fast LibriSpeech test-clean eval after fine-tuning.

Uses evenly spaced chunks per utterance (capped) for utterance-level majority vote,
so Acc/CER are meaningful without full 10 ms hop over ~2.6k files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afm_sgr.audio import chunk_waveform
from afm_sgr.config import get_config
from afm_sgr.datasets import WaveCache, discover_librispeech
from afm_sgr.train import load_model_from_checkpoint


def load_finetuned(run_name: str, device):
    cfg = get_config(run_name=run_name)
    model = load_model_from_checkpoint(cfg, cfg.run_dir() / "best.pt", device)
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_utt_level(model, utts, cfg, device, max_chunks_per_utt: int = 40):
    cache = WaveCache(cfg.sample_rate, remove_sil=True)
    utt_ok, frame_ok, frame_n = 0, 0, 0
    for u in tqdm(utts, desc="Libri eval", leave=False):
        wav = cache.get(u)
        chunks = chunk_waveform(wav, cfg.chunk_len, cfg.hop_len)
        n = chunks.shape[0]
        if n == 0:
            continue
        if n > max_chunks_per_utt:
            idx = np.linspace(0, n - 1, max_chunks_per_utt).astype(int)
            chunks = chunks[idx]
        logps = []
        for start in range(0, chunks.shape[0], cfg.batch_size):
            x = chunks[start : start + cfg.batch_size].to(device)
            logps.append(torch.log_softmax(model(x), dim=1).cpu())
        logp = torch.cat(logps)
        pred = logp.argmax(dim=1)
        y = int(u.gender)
        frame_ok += int((pred == y).sum())
        frame_n += int(pred.numel())
        if int(logp.sum(dim=0).argmax().item()) == y:
            utt_ok += 1
    n_utt = len(utts)
    acc = 100.0 * utt_ok / max(1, n_utt)
    cer = 100.0 - acc
    fer = 100.0 * (1.0 - frame_ok / max(1, frame_n))
    return {
        "accuracy": acc,
        "cer": cer,
        "fer": fer,
        "frame_accuracy": 100.0 - fer,
        "n_utt": float(n_utt),
        "n_chunks": float(frame_n),
        "protocol": f"evenly_spaced_max_{max_chunks_per_utt}_chunks_per_utt",
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = get_config()
    test_clean = [
        u
        for u in discover_librispeech(cfg0.libri_root, ["test-clean"])
        if u.split == "test-clean"
    ]
    print(f"LibriSpeech test-clean: {len(test_clean)} utts | device={device}")

    frontends = ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]
    summary = []
    for fe in frontends:
        run = f"{fe}_timit_finetune"
        out = Path(cfg0.results_root) / run
        if not (out / "best.pt").exists():
            print(f"[skip] {run}")
            continue
        metrics_path = out / "metrics_librispeech_test_clean.json"
        print(f"=== {run} ===")
        model, cfg = load_finetuned(run, device)
        m = eval_utt_level(model, test_clean, cfg, device, max_chunks_per_utt=40)
        metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        hist = json.loads((out / "history.json").read_text(encoding="utf-8")) if (out / "history.json").exists() else {}
        best_val = max(hist["val_acc"]) if hist.get("val_acc") else None
        last_tr = hist["train_acc"][-1] * 100 if hist.get("train_acc") else None
        row = {
            "frontend": fe,
            "test_acc": round(m["accuracy"], 2),
            "test_cer": round(m["cer"], 2),
            "test_fer": round(m["fer"], 2),
            "ft_best_val_acc": round(best_val, 2) if best_val is not None else None,
            "ft_last_train_acc": round(last_tr, 2) if last_tr is not None else None,
        }
        summary.append(row)
        print(
            f"  test Acc={row['test_acc']:.2f}% CER={row['test_cer']:.2f}% "
            f"FER={row['test_fer']:.2f}% | FT-val best={row['ft_best_val_acc']}"
        )

    summary_path = Path(cfg0.results_root) / "librispeech_finetune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] {summary_path}")


if __name__ == "__main__":
    main()
