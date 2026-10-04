from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torchaudio


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_audio(path, sample_rate: int = 16000) -> torch.Tensor:
    """Load mono float waveform at target sample rate (supports NIST SPHERE / WAV / FLAC)."""
    path = str(path)
    wav = None
    sr = None
    try:
        import soundfile as sf

        data, sr = sf.read(path, always_2d=False)
        wav = torch.tensor(np.asarray(data), dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
    except Exception:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).float()
        sr = int(sr)

    if int(sr) != sample_rate:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), int(sr), sample_rate).squeeze(0)
    return wav.float()


def remove_silence(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    top_db: float = 30.0,
) -> torch.Tensor:
    """Energy-based silence trimming / removal (simple VAD)."""
    if wav.numel() == 0:
        return wav
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if wav.numel() < frame:
        return wav

    # frame energies
    n = 1 + (wav.numel() - frame) // hop
    frames = wav.unfold(0, frame, hop)[:n]
    energy = frames.pow(2).mean(dim=1).clamp_min(1e-12)
    log_e = 10.0 * torch.log10(energy)
    thresh = log_e.max() - top_db
    keep = log_e >= thresh
    if not bool(keep.any()):
        return wav

    idx = keep.nonzero(as_tuple=False).squeeze(1)
    # merge kept frames into contiguous samples
    pieces = []
    for i in idx.tolist():
        start = i * hop
        pieces.append(wav[start : start + frame])
    out = torch.cat(pieces)
    # de-duplicate overlap by hop-average approx: take non-overlapping hop steps more cleanly
    # Better: build mask
    mask = torch.zeros(wav.numel(), dtype=torch.bool)
    for i in idx.tolist():
        start = i * hop
        mask[start : start + frame] = True
    trimmed = wav[mask]
    return trimmed if trimmed.numel() > 0 else wav


def peak_normalize(wav: torch.Tensor) -> torch.Tensor:
    peak = wav.abs().max()
    return wav / peak if float(peak) > 0 else wav


def trim_to_words(wav: torch.Tensor, wav_path) -> torch.Tensor:
    """TIMIT: keep samples between the first word start and last word end (.WRD)."""
    from pathlib import Path

    p = Path(wav_path)
    for wrd in (p.with_suffix(".WRD"), p.with_suffix(".wrd")):
        if not wrd.exists():
            continue
        lines = [l.split() for l in wrd.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        if not lines:
            break
        beg, end = int(lines[0][0]), int(lines[-1][1])
        seg = wav[beg:end]
        return seg if seg.numel() > 0 else wav
    return wav


def remove_long_silences(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    thresh_db: float = -30.0,
    min_sil_s: float = 1.0,
    frame_ms: float = 10.0,
) -> torch.Tensor:
    """LibriSpeech: drop silent runs >= min_sil_s below thresh_db (ffmpeg silencedetect n=-30dB:d=1)."""
    hop = max(1, int(sample_rate * frame_ms / 1000.0))
    n = wav.numel() // hop
    if n == 0:
        return wav
    peak = wav[: n * hop].view(n, hop).abs().amax(dim=1)
    silent = (peak < 10 ** (thresh_db / 20.0)).int()
    edges = torch.diff(torch.cat([torch.zeros(1, dtype=torch.int32), silent, torch.zeros(1, dtype=torch.int32)]))
    starts = (edges == 1).nonzero(as_tuple=False).flatten().tolist()
    ends = (edges == -1).nonzero(as_tuple=False).flatten().tolist()
    min_frames = int(round(min_sil_s * 1000.0 / frame_ms))
    keep = torch.ones(wav.numel(), dtype=torch.bool)
    for a, b in zip(starts, ends):
        if b - a >= min_frames:
            keep[a * hop : b * hop] = False
    out = wav[keep]
    return out if out.numel() > 0 else wav


def chunk_waveform(
    wav: torch.Tensor,
    chunk_len: int,
    hop_len: int,
) -> torch.Tensor:
    """Return (n_chunks, chunk_len). Pads last chunk if needed when wav shorter than chunk."""
    if wav.numel() < chunk_len:
        pad = chunk_len - wav.numel()
        wav = torch.nn.functional.pad(wav, (0, pad))
        return wav.unsqueeze(0)
    n = 1 + (wav.numel() - chunk_len) // hop_len
    # unfold
    chunks = wav.unfold(0, chunk_len, hop_len)[:n]
    return chunks.contiguous()
