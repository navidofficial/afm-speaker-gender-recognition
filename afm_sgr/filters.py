"""
Auditory filter model frontends (paper Table 1).

The parameterized kernels follow the authors' SGR_AFM code (NetModels/Kernel*Conv.py and
Utils/SignalUtils.kernel_*), vectorized over filters:

  learned:  norm_f1, norm_f2 (cutoffs / fs), amplitude A where the kernel has one
  cutoffs:  f1 = |norm_f1| + 50/fs ;  f2 = f1 + |norm_f2 - f1| + 50/fs   (then * fs -> Hz)
  b = f2 - f1, fc = (f1 + f2) / 2, t in seconds
  sinc   : 2b * sin(b t)/(b t) * cos(2 pi fc t)        symmetric, centre sample = 1
  sinc2  : A * sin(b t)^2/(b t) * cos(2 pi fc t)       symmetric, centre sample = 1
  gamma  : 1e11 * A * t^3 * exp(-2 pi b t) * cos(2 pi fc t),  t = 1/fs .. N/fs
  gauss  : A * exp(-t^2 / (2 sigma^2)) * cos(2 pi fc t), sigma = sqrt(log10 2) / max(b, 1)
  each kernel is divided by its max, then multiplied by a Hamming window.
  cascade: two Gaussian stages (f1,f2) and (f3,f4), each min-max scaled to [-1, 1] and
           mean-removed, convolved together, scaled again, then windowed.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


def mel_freqs(n: int, sample_rate: int) -> np.ndarray:
    """SignalUtils.get_mel_freqs: n points equally spaced in mel between 80 mel and fs/2."""
    high_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel = np.linspace(80.0, high_mel, n)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def init_f1_f2(n_filt: int, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """SignalUtils.get_f1_f2: f1 = previous mel point (30 Hz for the first), f2 = mel point."""
    m = mel_freqs(n_filt, sample_rate)
    f1 = np.roll(m, 1)
    f1[0] = 30.0
    return f1, m.copy()


def init_f1_f4_cascade(n_filt: int, sample_rate: int) -> Tuple[np.ndarray, ...]:
    """SignalUtils.get_f1_f4_new: every filter pairs two different mel bands (i, j>i)."""
    n_mel = int(np.ceil((-1 + np.sqrt(1 + 8 * n_filt)) / 2)) + 1
    m = np.insert(mel_freqs(n_mel, sample_rate), 0, 30.0)
    f1, f2, f3, f4 = (np.zeros(n_filt) for _ in range(4))
    k = 0
    for i in range(len(m) - 1):
        for j in range(i + 1, len(m) - 1):
            if k < n_filt:
                f1[k], f2[k], f3[k], f4[k] = m[i], m[i + 1], m[j], m[j + 1]
                k += 1
    return f1, f2, f3, f4


def mel_filterbank_impulse(
    n_filt: int,
    filt_len: int,
    sample_rate: int,
) -> torch.Tensor:
    """Approximate Mel band-pass FIR filters as impulse responses (n_filt, filt_len)."""
    n_fft = max(512, filt_len)
    mel = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_filt,
        sample_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )  # (n_freqs, n_mels)
    irs = []
    for i in range(n_filt):
        mag = mel[:, i]
        full = torch.zeros(n_fft)
        full[: n_fft // 2 + 1] = mag
        if n_fft % 2 == 0:
            full[n_fft // 2 + 1 :] = mag[1:-1].flip(0)
        else:
            full[n_fft // 2 + 1 :] = mag[1:].flip(0)
        ir = torch.fft.ifft(full).real
        ir = torch.roll(ir, shifts=n_fft // 2)
        if ir.numel() >= filt_len:
            start = (ir.numel() - filt_len) // 2
            ir = ir[start : start + filt_len]
        else:
            pad = filt_len - ir.numel()
            ir = F.pad(ir, (pad // 2, pad - pad // 2))
        ir = ir * torch.hamming_window(filt_len)
        irs.append(ir)
    return torch.stack(irs, dim=0)


def _minmax_center(h: torch.Tensor) -> torch.Tensor:
    lo = h.min(dim=1, keepdim=True).values
    hi = h.max(dim=1, keepdim=True).values
    h = 2.0 * (h - lo) / (hi - lo + 1e-6) - 1.0
    return h - h.mean(dim=1, keepdim=True)


class AFMConv1d(nn.Module):
    """
    Parameterized band-pass frontend with learnable cutoffs (f1, f2) and amplitude.
    filter_type: sinc | sinc2 | gamma | gauss | cascade
    init: optional {"f1_hz": array, "f2_hz": array} (used by the Reduced model).
    """

    def __init__(
        self,
        n_filt: int = 80,
        filt_len: int = 257,
        sample_rate: int = 16000,
        filter_type: str = "sinc",
        min_freq_hz: float = 50.0,
        init: Optional[Dict] = None,
    ):
        super().__init__()
        if filt_len % 2 == 0:
            raise ValueError("filt_len must be odd (symmetric kernels)")
        self.n_filt = n_filt
        self.filt_len = filt_len
        self.sample_rate = sample_rate
        self.filter_type = filter_type.lower()
        self.min_freq_hz = min_freq_hz
        fs = float(sample_rate)

        def P(a):
            return nn.Parameter(torch.as_tensor(np.asarray(a, dtype=np.float64) / fs, dtype=torch.float32))

        if self.filter_type == "cascade":
            f1, f2, f3, f4 = init_f1_f4_cascade(n_filt, sample_rate)
            self.norm_f1, self.norm_f2, self.norm_f3, self.norm_f4 = P(f1), P(f2), P(f3), P(f4)
            self.amp1 = nn.Parameter(torch.ones(n_filt))
            self.amp2 = nn.Parameter(torch.ones(n_filt))
        else:
            if init is not None:
                f1 = np.asarray(init["f1_hz"], dtype=np.float64)
                f2 = np.asarray(init["f2_hz"], dtype=np.float64)
                if len(f1) != n_filt:
                    raise ValueError(f"init has {len(f1)} filters, expected {n_filt}")
            else:
                f1, f2 = init_f1_f2(n_filt, sample_rate)
            self.norm_f1, self.norm_f2 = P(f1), P(f2)
            if self.filter_type == "sinc2":
                self.amp = P(f2 - f1)
            elif self.filter_type in ("gamma", "gauss"):
                self.amp = nn.Parameter(torch.ones(n_filt))
            elif self.filter_type != "sinc":
                raise ValueError(f"Unknown filter_type: {filter_type}")

        half = (filt_len - 1) // 2
        self.register_buffer("t_right", torch.arange(1, half + 1, dtype=torch.float32).view(1, -1) / fs)
        self.register_buffer("t_full", torch.arange(1, filt_len + 1, dtype=torch.float32).view(1, -1) / fs)
        n = torch.linspace(0, filt_len, steps=filt_len)
        self.register_buffer("window", (0.54 - 0.46 * torch.cos(2 * math.pi * n / filt_len)).view(1, -1))

    # ---------------------------------------------------------------- cutoffs
    def _cutoffs(self, nf1: torch.Tensor, nf2: torch.Tensor, clip: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (f1, f2) in Hz, shape (F, 1)."""
        m = self.min_freq_hz / self.sample_rate
        f1 = nf1.abs() + m
        if clip:
            f1 = f1.clamp(0.0, 0.5)
        f2 = f1 + (nf2 - f1).abs() + m
        if clip:
            f2 = f2.clamp(0.0, 0.5)
        return (f1 * self.sample_rate).view(-1, 1), (f2 * self.sample_rate).view(-1, 1)

    # ---------------------------------------------------------------- kernels
    def _symmetric(self, y_right: torch.Tensor) -> torch.Tensor:
        centre = torch.ones(y_right.shape[0], 1, device=y_right.device, dtype=y_right.dtype)
        return torch.cat([y_right.flip(1), centre, y_right], dim=1)

    def _kernel_sinc(self, f1, f2):
        b, fc, t = f2 - f1, (f1 + f2) / 2, self.t_right
        bt = b * t
        return self._symmetric(2 * b * torch.sin(bt) / bt * torch.cos(2 * math.pi * fc * t))

    def _kernel_sinc2(self, amp, f1, f2):
        b, fc, t = f2 - f1, (f1 + f2) / 2, self.t_right
        bt = b * t
        return self._symmetric(amp * torch.sin(bt).pow(2) / bt * torch.cos(2 * math.pi * fc * t))

    def _kernel_gamma(self, amp, f1, f2, order: int = 4):
        b, fc, t = f2 - f1, (f1 + f2) / 2, self.t_full
        return 1e11 * amp * t.pow(order - 1) * torch.exp(-2 * math.pi * b * t) * torch.cos(2 * math.pi * fc * t)

    def _kernel_gauss(self, amp, f1, f2):
        b, fc, t = (f2 - f1).clamp_min(1.0), (f1 + f2) / 2, self.t_full
        sigma = math.sqrt(math.log10(2.0)) / b
        return amp * torch.exp(-t.pow(2) / (2 * sigma.pow(2))) * torch.cos(2 * math.pi * fc * t)

    @staticmethod
    def _max_norm(h: torch.Tensor) -> torch.Tensor:
        return h / h.max(dim=1, keepdim=True).values

    def get_filters(self) -> torch.Tensor:
        """Return kernels (n_filt, filt_len)."""
        kind = self.filter_type
        if kind == "cascade":
            f1, f2 = self._cutoffs(self.norm_f1, self.norm_f2, clip=True)
            f3, f4 = self._cutoffs(self.norm_f3, self.norm_f4, clip=True)
            h1 = _minmax_center(self._kernel_gauss(self.amp1.abs().view(-1, 1), f1, f2))
            h2 = _minmax_center(self._kernel_gauss(self.amp2.abs().view(-1, 1), f3, f4))
            h = F.conv1d(h1.unsqueeze(0), h2.unsqueeze(1), padding=self.filt_len // 2, groups=self.n_filt)
            return _minmax_center(h.squeeze(0)) * self.window

        f1, f2 = self._cutoffs(self.norm_f1, self.norm_f2, clip=False)
        if kind == "sinc":
            h = self._kernel_sinc(f1, f2)
        else:
            amp = self.amp.abs().view(-1, 1)
            if kind == "sinc2":
                h = self._kernel_sinc2(amp, f1, f2)
            elif kind == "gamma":
                h = self._kernel_gamma(amp, f1, f2)
            else:
                h = self._kernel_gauss(amp, f1, f2)
        return self._max_norm(h) * self.window

    # ---------------------------------------------------------------- analysis
    def get_f1_f2(self) -> Tuple[np.ndarray, np.ndarray]:
        """Learned cutoffs in Hz clipped to [0, fs/2]; cascade returns both stages (2 * n_filt)."""
        with torch.no_grad():
            f1, f2 = self._cutoffs(self.norm_f1, self.norm_f2, clip=True)
            if self.filter_type == "cascade":
                f3, f4 = self._cutoffs(self.norm_f3, self.norm_f4, clip=True)
                f1, f2 = torch.cat([f1, f3]), torch.cat([f2, f4])
        return f1.view(-1).cpu().numpy(), f2.view(-1).cpu().numpy()

    def get_fc_bw(self):
        f1, f2 = self.get_f1_f2()
        fc = torch.from_numpy((f1 + f2) / 2.0)
        bw = torch.from_numpy(f2 - f1)
        with torch.no_grad():
            if self.filter_type == "cascade":
                amp = torch.cat([self.amp1.abs(), self.amp2.abs()])
            elif self.filter_type == "sinc":
                amp = torch.ones(self.n_filt)
            else:
                amp = self.amp.abs()
        return fc, bw, amp.detach().cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        filters = self.get_filters().view(self.n_filt, 1, self.filt_len)
        return F.conv1d(x, filters)


class FreeConv1d(nn.Module):
    def __init__(self, n_filt=80, filt_len=257):
        super().__init__()
        self.n_filt = n_filt
        self.conv = nn.Conv1d(1, n_filt, kernel_size=filt_len, stride=1, padding=0, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.conv(x)

    def get_filters(self):
        return self.conv.weight.data.squeeze(1)

    def get_fc_bw(self):
        w = self.get_filters()
        n = w.shape[-1]
        spec = torch.fft.rfft(w, n=n * 4)
        mag = spec.abs()
        freqs = torch.linspace(0, 0.5, mag.shape[-1], device=w.device)
        idx = mag.argmax(dim=-1)
        fc = freqs[idx]  # cycles/sample; caller scales
        bw = torch.ones_like(fc) * 0.05
        amp = w.norm(dim=-1)
        return fc, bw, amp


class MelConv1d(nn.Module):
    def __init__(self, n_filt=80, filt_len=257, sample_rate=16000):
        super().__init__()
        ir = mel_filterbank_impulse(n_filt, filt_len, sample_rate)
        self.register_buffer("filters", ir.view(n_filt, 1, filt_len))
        self.n_filt = n_filt

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, self.filters, stride=1, padding=0)

    def get_filters(self):
        return self.filters.squeeze(1)

    def get_fc_bw(self):
        w = self.get_filters()
        n = w.shape[-1]
        mag = torch.fft.rfft(w, n=n * 4).abs()
        freqs = torch.linspace(0, 0.5, mag.shape[-1], device=w.device)
        idx = mag.argmax(dim=-1)
        fc = freqs[idx]
        bw = torch.ones_like(fc) * 0.05
        amp = w.norm(dim=-1)
        return fc, bw, amp


def build_frontend(
    name: str,
    n_filt=80,
    filt_len=257,
    sample_rate=16000,
    min_freq_hz: float = 50.0,
    init: Optional[Dict] = None,
) -> nn.Module:
    name = name.lower()
    if name == "free":
        return FreeConv1d(n_filt, filt_len)
    if name == "mel":
        return MelConv1d(n_filt, filt_len, sample_rate)
    if name in ("sinc", "sinc2", "gamma", "gauss", "cascade"):
        return AFMConv1d(n_filt, filt_len, sample_rate, filter_type=name, min_freq_hz=min_freq_hz)
    if name == "reduced":
        # KernelCustomFreezeConv: sinc2 kernels initialised at the mean-shift cluster centres
        return AFMConv1d(n_filt, filt_len, sample_rate, filter_type="sinc2", min_freq_hz=min_freq_hz, init=init)
    raise ValueError(f"Unknown frontend: {name}")
