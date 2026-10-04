import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)


def bar_compare(methods, paper, ours, title, fname, ylabel="Accuracy (%)"):
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    b1 = ax.bar(x - w / 2, paper, w, label="Paper", color="#4C72B0")
    b2 = ax.bar(x + w / 2, ours, w, label="Ours", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lo = min(paper + ours) - 5
    ax.set_ylim(max(50, lo), 100.5)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.15,
                f"{b.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    fig.tight_layout()
    fig.savefig(out / fname, dpi=170)
    plt.close(fig)


bar_compare(
    ["Free", "Mel", "Sinc", "Sinc2", "Gamma", "Gauss", "Cascade", "Reduced"],
    [98.84, 98.64, 98.84, 99.03, 99.01, 98.91, 99.16, 99.03],
    [98.66, 98.66, 98.88, 98.88, 98.88, 98.88, 99.18, 98.07],
    "TIMIT utterance accuracy: paper vs. this work",
    "bar_timit_acc.png",
)

methods2 = ["Free", "Sinc", "Sinc2", "Gamma", "Gauss", "Cascade", "Reduced"]
libri_methods = ["Free", "Sinc", "Sinc2", "Gamma", "Gauss", "Cascade", "Reduced"]
libri_train = [94.74, 92.08, 91.93, 92.75, 92.36, 92.92, 87.22]
libri_val = [89.46, 89.46, 89.46, 89.46, 89.59, 89.46, 89.46]
x = np.arange(len(libri_methods))
w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.2))
b1 = ax.bar(x - w / 2, libri_train, w, label="Train (epoch 20)", color="#4C72B0")
b2 = ax.bar(x + w / 2, libri_val, w, label="Best val", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(libri_methods, rotation=20, ha="right")
ax.set_ylabel("Accuracy (%)")
ax.set_title("LibriSpeech fine-tuning (dev-clean)")
ax.set_ylim(82, 98)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out / "bar_librispeech_ft.png", dpi=170)
plt.close(fig)

bar_compare(
    methods2,
    [93.54, 96.32, 94.86, 97.78, 93.19, 97.99, 96.11],
    [88.47, 94.65, 93.47, 96.04, 95.21, 95.49, 89.58],
    "RAVDESS accuracy after fine-tuning",
    "bar_ravdess_acc.png",
)
bar_compare(
    methods2,
    [97.48, 97.66, 96.33, 97.17, 96.74, 97.27, 97.44],
    [76.16, 96.96, 95.38, 95.88, 95.69, 96.80, 91.53],
    "VoxCeleb accuracy after fine-tuning (paper: full; ours: test)",
    "bar_vox_acc.png",
)

vars_ = ["Paper MS", "Task-aware\nERB", "Task-aware\ndistilled"]
timit = [98.07, 98.29, 98.88]
rav = [89.58, 91.25, 95.63]
vox = [91.53, 91.83, 93.48]
x = np.arange(len(vars_))
w = 0.25
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (vals, lab) in enumerate([(timit, "TIMIT"), (rav, "RAVDESS"), (vox, "VoxCeleb1")]):
    ax.bar(x + (i - 1) * w, vals, w, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(vars_)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(85, 100.5)
ax.set_title("Reduced models at K=43 filters")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out / "bar_reduced_variants.png", dpi=170)
plt.close(fig)
print("ok")
