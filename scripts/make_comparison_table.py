"""
Aggregate per-run metrics into a comparison table (CSV + Markdown + PNG).

Usage:
  python scripts/make_comparison_table.py
  python scripts/make_comparison_table.py --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afm_sgr.config import RESULTS_ROOT


def collect_rows(results_dir: Path):
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for metrics_path in sorted(run_dir.glob("metrics_*.json")):
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            # run name e.g. cascade_timit / cascade_timit_finetune
            name = run_dir.name
            frontend = name.split("_")[0]
            split = metrics_path.stem.replace("metrics_", "")
            rows.append(
                {
                    "run": name,
                    "frontend": frontend,
                    "eval_set": split,
                    "accuracy_%": round(float(data.get("accuracy", 0.0)), 2),
                    "CER_%": round(float(data.get("cer", 0.0)), 2),
                    "FER_%": round(float(data.get("fer", 0.0)), 2),
                    "frame_accuracy_%": round(float(data.get("frame_accuracy", 0.0)), 2),
                    "n_utt": int(float(data.get("n_utt", 0))),
                    "n_chunks": int(float(data.get("n_chunks", 0))),
                    "has_loss_plot": (run_dir / "loss_per_epoch.png").exists(),
                    "has_acc_plot": (run_dir / "accuracy_per_epoch.png").exists(),
                }
            )
    return rows


def write_csv(rows, path: Path):
    if not rows:
        path.write_text("run,frontend,eval_set,accuracy_%,CER_%,FER_%\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows, path: Path):
    lines = [
        "# Comparison table",
        "",
        "| run | frontend | eval_set | accuracy_% | CER_% | FER_% |",
        "|-----|----------|----------|------------|-------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['frontend']} | {r['eval_set']} | "
            f"{r['accuracy_%']:.2f} | {r['CER_%']:.2f} | {r['FER_%']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png_table(rows, path: Path):
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No metrics_*.json found yet", ha="center", va="center")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    cols = ["run", "frontend", "eval_set", "accuracy_%", "CER_%", "FER_%"]
    cell = [[str(r[c]) for c in cols] for r in rows]
    fig_h = max(2.5, 0.45 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Model comparison (accuracy / CER / FER)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
    )
    args = p.parse_args()
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(results_dir)
    out_csv = results_dir / "comparison_table.csv"
    out_md = results_dir / "comparison_table.md"
    out_png = results_dir / "comparison_table.png"

    write_csv(rows, out_csv)
    write_markdown(rows, out_md)
    write_png_table(rows, out_png)

    print(f"[ok] rows={len(rows)}")
    print(f"  {out_csv}")
    print(f"  {out_md}")
    print(f"  {out_png}")
    if rows:
        print("\nPreview:")
        for r in rows:
            print(
                f"  {r['frontend']:8s} | {r['eval_set']:16s} | "
                f"acc={r['accuracy_%']:6.2f} CER={r['CER_%']:5.2f} FER={r['FER_%']:5.2f}"
            )


if __name__ == "__main__":
    main()
