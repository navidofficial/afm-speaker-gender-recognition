import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results_github"
N = 759
for f in ["free", "sinc", "cascade", "gauss", "reduced"]:
    h = json.loads((root / f"{f}_timit_finetune" / "history.json").read_text())
    i = h["val_acc"].index(max(h["val_acc"]))
    acc = h["val_acc"][i]
    c = round(acc * N / 100)
    print(
        f"{f}: best epoch {i + 1} acc={acc:.6f} (~{c}/{N}) "
        f"cer={h['val_cer'][i]:.2f} fer={h['val_fer'][i]:.2f}"
    )
