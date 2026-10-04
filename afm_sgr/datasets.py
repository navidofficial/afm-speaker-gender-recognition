from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import (
    chunk_waveform,
    load_audio,
    peak_normalize,
    remove_long_silences,
    remove_silence,
    trim_to_words,
)


@dataclass
class Utterance:
    path: Path
    speaker: str
    gender: int  # 0 male, 1 female
    split: str
    corpus: str = ""


def _parse_timit_spkrinfo(path: Path) -> Dict[str, int]:
    mapping = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        spk, sex = parts[0].upper(), parts[1].upper()
        if sex in ("M", "F"):
            mapping[spk] = 0 if sex == "M" else 1
    return mapping


def discover_timit(timit_root: Path, exclude_sa: bool = True) -> List[Utterance]:
    spkrinfo = timit_root / "DOC" / "SPKRINFO.TXT"
    gender = _parse_timit_spkrinfo(spkrinfo)
    utts: List[Utterance] = []
    for split_name, folder in (("train", "TRAIN"), ("test", "TEST")):
        root = timit_root / folder
        for wav in sorted(root.rglob("*.WAV")):
            if exclude_sa and wav.stem.upper().startswith("SA"):
                continue
            spk_dir = wav.parent.name.upper()
            key = spk_dir
            if key not in gender and len(key) > 1 and key[0] in ("F", "M"):
                key = key[1:]
            if key not in gender and spk_dir[0] in ("F", "M"):
                gender[spk_dir] = 0 if spk_dir[0] == "M" else 1
                key = spk_dir
            if key not in gender:
                continue
            utts.append(Utterance(wav, spk_dir, gender[key], split_name, "timit"))
    return utts


def split_timit_train_val(
    utts: List[Utterance], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[Utterance], List[Utterance], List[Utterance]]:
    train = [u for u in utts if u.split == "train"]
    test = [u for u in utts if u.split == "test"]
    speakers = sorted({u.speaker for u in train})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_ratio))
    val_spk = set(speakers[:n_val])
    tr = [u for u in train if u.speaker not in val_spk]
    va = [u for u in train if u.speaker in val_spk]
    for u in va:
        u.split = "val"
    return tr, va, test


def discover_librispeech(libri_root: Path, subsets: Sequence[str]) -> List[Utterance]:
    speakers_file = libri_root / "SPEAKERS.TXT"
    gender: Dict[str, int] = {}
    for line in speakers_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        spk, sex = parts[0], parts[1]
        gender[spk] = 0 if sex.upper() == "M" else 1

    utts: List[Utterance] = []
    for subset in subsets:
        root = libri_root / subset
        if not root.exists():
            continue
        for wav in sorted(root.rglob("*.flac")):
            spk = wav.parent.parent.name
            if spk not in gender:
                continue
            utts.append(Utterance(wav, spk, gender[spk], subset, "librispeech"))
    return utts


def discover_ravdess(ravdess_root: Path) -> List[Utterance]:
    utts: List[Utterance] = []
    for wav in sorted(ravdess_root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) < 7:
            continue
        actor = int(parts[-1])
        gender = 0 if actor % 2 == 1 else 1
        utts.append(Utterance(wav, f"Actor_{actor:02d}", gender, "test", "ravdess"))
    return utts


def discover_voxceleb_test(vox_root: Path, meta_csv: Path) -> List[Utterance]:
    gender: Dict[str, int] = {}
    with meta_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spk = row["VoxCeleb1 ID"].strip()
            g = row["Gender"].strip().lower()
            gender[spk] = 0 if g == "m" else 1

    wav_root = vox_root / "wav"
    utts: List[Utterance] = []
    for wav in sorted(wav_root.rglob("*.wav")):
        spk = None
        for p in wav.parts:
            if p in gender:
                spk = p
                break
        if spk is None:
            continue
        utts.append(Utterance(wav, spk, gender[spk], "test", "voxceleb"))
    return utts


class WaveCache:
    """
    Shared waveform cache (not pickled into DataLoader workers).

    Preprocessing follows the authors' data preparation per corpus:
      timit       -> peak normalize, trim to first/last word (.WRD)
      librispeech -> remove silences >= 1 s below -30 dB
      ravdess / voxceleb -> raw
    """

    def __init__(self, sample_rate: int = 16000, remove_sil: bool = True):
        self.sample_rate = sample_rate
        self.remove_sil = remove_sil
        self._cache: Dict[str, torch.Tensor] = {}

    def _preprocess(self, wav: torch.Tensor, path: Path, corpus: str) -> torch.Tensor:
        if corpus == "timit":
            return trim_to_words(peak_normalize(wav), path)
        if corpus == "librispeech":
            return remove_long_silences(wav, self.sample_rate)
        if corpus in ("ravdess", "voxceleb"):
            return wav
        return remove_silence(wav, self.sample_rate) if self.remove_sil else wav

    def get(self, item: Union[Utterance, Path, str], corpus: str = "") -> torch.Tensor:
        if isinstance(item, Utterance):
            path, corpus = item.path, item.corpus
        else:
            path = Path(item)
        key = str(path)
        if key not in self._cache:
            wav = load_audio(path, self.sample_rate)
            self._cache[key] = self._preprocess(wav, path, corpus)
        return self._cache[key]


class TrainChunkDataset(Dataset):
    """
    Training: several random 200 ms chunks per utterance each epoch, taken at a uniformly
    random sample offset and scaled by a random gain in [1-amp_fact, 1+amp_fact].
    """

    def __init__(
        self,
        utterances: List[Utterance],
        cache: WaveCache,
        chunk_len: int = 3200,
        hop_len: int = 160,
        chunks_per_utt: int = 8,
        amp_fact: float = 0.2,
    ):
        self.utterances = utterances
        self.cache = cache
        self.chunk_len = chunk_len
        self.hop_len = hop_len
        self.chunks_per_utt = chunks_per_utt
        self.amp_fact = amp_fact
        self._n = len(utterances) * chunks_per_utt

    def __len__(self):
        return self._n

    def __getitem__(self, idx: int):
        u_idx = idx % len(self.utterances)
        u = self.utterances[u_idx]
        wav = self.cache.get(u)
        n = wav.numel()
        if n <= self.chunk_len + 1:
            x = torch.nn.functional.pad(wav, (0, self.chunk_len - min(n, self.chunk_len)))[: self.chunk_len]
        else:
            beg = np.random.randint(n - self.chunk_len - 1)
            x = wav[beg : beg + self.chunk_len]
        gain = np.random.uniform(1.0 - self.amp_fact, 1.0 + self.amp_fact)
        return x * float(gain), u.gender, u_idx


class EvalChunkDataset(Dataset):
    def __init__(
        self,
        utterances: List[Utterance],
        cache: WaveCache,
        chunk_len: int = 3200,
        hop_len: int = 160,
    ):
        self.utterances = utterances
        self.cache = cache
        self.chunk_len = chunk_len
        self.hop_len = hop_len
        self.index: List[Tuple[int, int]] = []
        try:
            from tqdm import tqdm

            it = tqdm(list(enumerate(utterances)), desc="Indexing eval chunks", leave=False)
        except Exception:
            it = enumerate(utterances)
        for i, u in it:
            wav = cache.get(u)
            n = chunk_waveform(wav, chunk_len, hop_len).shape[0]
            for c in range(n):
                self.index.append((i, c))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        u_idx, c_idx = self.index[idx]
        u = self.utterances[u_idx]
        wav = self.cache.get(u)
        start = c_idx * self.hop_len
        x = wav[start : start + self.chunk_len]
        if x.numel() < self.chunk_len:
            x = torch.nn.functional.pad(x, (0, self.chunk_len - x.numel()))
        return x, u.gender, u_idx


# backward-compatible alias
ChunkDataset = EvalChunkDataset


class UtteranceEvalBundle:
    def __init__(self, dataset: EvalChunkDataset):
        self.dataset = dataset
        self.utt_to_chunks: Dict[int, List[int]] = {}
        for flat_i, (u_i, _) in enumerate(dataset.index):
            self.utt_to_chunks.setdefault(u_i, []).append(flat_i)
