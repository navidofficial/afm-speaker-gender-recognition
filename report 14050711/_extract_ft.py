import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results_github"
frontends = ["free", "sinc", "sinc2", "gamma", "gauss", "cascade", "reduced"]
print("=== Libri FT (from history) ===")
for f in frontends:
    p = root / f"{f}_timit_finetune" / "history.json"
    h = json.loads(p.read_text())
    ta = h["train_acc"][-1] * 100
    va = max(h["val_acc"])
    print(f"{f:8} train_end={ta:.2f}  best_val={va:.2f}")

print("\n=== OOD after FT ===")
for f in frontends:
    d = root / f"{f}_timit_finetune"
    rav = json.loads((d / "metrics_ravdess.json").read_text())
    vox = json.loads((d / "metrics_voxceleb1_test.json").read_text())
    print(
        f"{f:8} RAV {rav['accuracy']:.2f} {rav['cer']:.2f} {rav['fer']:.2f} | "
        f"Vox {vox['accuracy']:.2f} {vox['cer']:.2f} {vox['fer']:.2f}"
    )

for f in ["reduced", "reduced_A", "reduced_AD", "reduced_D", "reduced_B", "reduced_C"]:
    d = root / f"{f}_timit_finetune"
    rav = json.loads((d / "metrics_ravdess.json").read_text())
    vox = json.loads((d / "metrics_voxceleb1_test.json").read_text())
    print(
        f"{f:12} RAV {rav['accuracy']:.2f} {rav['cer']:.2f} {rav['fer']:.2f} | "
        f"Vox {vox['accuracy']:.2f} {vox['cer']:.2f} {vox['fer']:.2f}"
    )
