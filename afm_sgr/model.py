from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .filters import build_frontend


class LayerNorm(nn.Module):
    """SincNet / SGR_AFM LayerNorm: statistics over the last axis, affine of shape `features`."""

    def __init__(self, features, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _uniform_linear(in_dim: int, out_dim: int) -> nn.Linear:
    lin = nn.Linear(in_dim, out_dim)
    bound = float(np.sqrt(0.01 / (in_dim + out_dim)))
    nn.init.uniform_(lin.weight, -bound, bound)
    nn.init.zeros_(lin.bias)
    return lin


class GenderCNN(nn.Module):
    """Paper CNN (Fig. 1) following the authors' NetModel + MLP (KernelCNN_TIMIT.cfg)."""

    def __init__(
        self,
        frontend: str = "cascade",
        sample_rate: int = 16000,
        n_filt: int = 80,
        filt_len: int = 257,
        conv2_filt: int = 60,
        conv2_kernel: int = 5,
        conv3_filt: int = 60,
        conv3_kernel: int = 5,
        pool_size: int = 3,
        fc_dim: int = 2048,
        n_classes: int = 2,
        leaky_slope: float = 0.2,
        chunk_len: int = 3200,
        min_freq_hz: float = 50.0,
        frontend_init: Optional[Dict] = None,
    ):
        super().__init__()
        self.frontend_name = frontend.lower()
        self.pool_size = pool_size
        self.ln0 = LayerNorm(chunk_len)
        self.frontend = build_frontend(
            frontend, n_filt, filt_len, sample_rate, min_freq_hz=min_freq_hz, init=frontend_init
        )
        t1 = (chunk_len - filt_len + 1) // pool_size
        t2 = (t1 - conv2_kernel + 1) // pool_size
        t3 = (t2 - conv3_kernel + 1) // pool_size
        self.ln1 = LayerNorm([n_filt, t1])
        self.conv2 = nn.Conv1d(n_filt, conv2_filt, kernel_size=conv2_kernel)
        self.ln2 = LayerNorm([conv2_filt, t2])
        self.conv3 = nn.Conv1d(conv2_filt, conv3_filt, kernel_size=conv3_kernel)
        self.ln3 = LayerNorm([conv3_filt, t3])
        self.act = nn.LeakyReLU(leaky_slope)

        flat = conv3_filt * t3
        self.ln_fc = LayerNorm(flat)
        self.fc1 = _uniform_linear(flat, fc_dim)
        self.bn1 = nn.BatchNorm1d(fc_dim, momentum=0.05)
        self.fc2 = _uniform_linear(fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim, momentum=0.05)
        self.fc3 = _uniform_linear(fc_dim, fc_dim)
        self.bn3 = nn.BatchNorm1d(fc_dim, momentum=0.05)
        self.out = _uniform_linear(fc_dim, n_classes)

    def _conv_tower(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.ln0(x).unsqueeze(1)
        x = self.act(self.ln1(F.max_pool1d(torch.abs(self.frontend(x)), self.pool_size)))
        x = self.act(self.ln2(F.max_pool1d(self.conv2(x), self.pool_size)))
        x = self.act(self.ln3(F.max_pool1d(self.conv3(x), self.pool_size)))
        return x

    def forward(self, x, return_conv_feat: bool = False):
        feat = self._conv_tower(x).reshape(x.size(0), -1)
        h = self.ln_fc(feat)
        h = self.act(self.bn1(self.fc1(h)))
        h = self.act(self.bn2(self.fc2(h)))
        h = self.act(self.bn3(self.fc3(h)))
        logits = self.out(h)
        if return_conv_feat:
            return logits, feat
        return logits

    def freeze_frontend(self):
        for p in self.frontend.parameters():
            p.requires_grad = False
