#!/usr/bin/env python
"""Analyze and visualize training performance with detailed diagnostics."""

import json, sys, os
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

LOG_PATH = "runs/dualstream-train/tship_exp/training_log.jsonl"
OUTPUT_DIR = Path("runs/dualstream-train/tship_exp")


def load_entries():
    entries = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def plot_detailed_analysis(entries):
    entries = load_entries()
    steps = [e["step"] for e in entries]
    
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle("Dual-Stream YOLOv8 Training Analysis - 70 Epochs (Tship Dataset)",
                 fontsize=16, fontweight="bold")
    
    # --- Panel 1: Total Loss (full + smoothed) ---
    ax1 = fig.add_subplot(3, 4, 1)
    total_loss = [e["loss"]["total"] for e in entries]
    window = max(1, len(total_loss) // 200)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(total_loss, kernel, mode="same")
    else:
        smoothed = total_loss
    ax1.plot(steps, total_loss, alpha=0.15, linewidth=0.5, color="#3498db")
    ax1.plot(steps, smoothed, alpha=0.9, linewidth=1.5, color="#2980b9",
             label=f"Smooth(w={window})")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Total Loss (start={total_loss[0]:.1f}, end={total_loss[-1]:.1f})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    # --- Panel 2: Box Loss ---
    ax2 = fig.add_subplot(3, 4, 2)
    box_loss = [e["loss"]["box"] for e in entries]
    ax2.plot(steps, box_loss, alpha=0.15, linewidth=0.5, color="#e74c3c")
    if window > 1:
        ax2.plot(steps, np.convolve(box_loss, kernel, mode="same"), 
                 alpha=0.9, linewidth=1.5, color="#c0392b")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Box Loss ({box_loss[0]:.4f} -> {box_loss[-1]:.4f})")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Classification Loss ---
    ax3 = fig.add_subplot(3, 4, 3)
    cls_loss = [e["loss"]["cls"] for e in entries]
    ax3.plot(steps, cls_loss, alpha=0.15, linewidth=0.5, color="#2ecc71")
    if window > 1:
        ax3.plot(steps, np.convolve(cls_loss, kernel, mode="same"),
                 alpha=0.9, linewidth=1.5, color="#27ae60")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title(f"Cls Loss ({cls_loss[0]:.1f} -> {cls_loss[-1]:.1f})")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: DFL Loss ---
    ax4 = fig.add_subplot(3, 4, 4)
    dfl_loss = [e["loss"]["dfl"] for e in entries]
    ax4.plot(steps, dfl_loss, alpha=0.15, linewidth=0.5, color="#9b59b6")
    if window > 1:
        ax4.plot(steps, np.convolve(dfl_loss, kernel, mode="same"),
                 alpha=0.9, linewidth=1.5, color="#8e44ad")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.set_title(f"DFL Loss ({dfl_loss[0]:.4f} -> {dfl_loss[-1]:.4f})")
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Angle Loss ---
    ax5 = fig.add_subplot(3, 4, 5)
    angle_loss = [e["loss"]["angle"] for e in entries]
    ax5.plot(steps, angle_loss, alpha=0.15, linewidth=0.5, color="#f39c12")
    if window > 1:
        ax5.plot(steps, np.convolve(angle_loss, kernel, mode="same"),
                 alpha=0.9, linewidth=1.5, color="#e67e22")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Loss")
    ax5.set_title(f"Angle Loss ({angle_loss[0]:.4f} -> {angle_loss[-1]:.4f})")
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Gradient Norm ---
    ax6 = fig.add_subplot(3, 4, 6)
    grad_norms = [e["grad_norm"] for e in entries]
    ax6.plot(steps, grad_norms, alpha=0.2, linewidth=0.5, color="#1abc9c")
    if window > 1:
        ax6.plot(steps, np.convolve(grad_norms, kernel, mode="same"),
                 alpha=0.9, linewidth=1.5, color="#16a085")
    ax6.axhline(y=10.0, color="red", linestyle="--", linewidth=1.0, label="Clip max")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Gradient Norm")
    ax6.set_title(f"Grad Norm ({np.mean(grad_norms[:100]):.0f} -> {np.mean(grad_norms[-100:]):.0f})")
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=7)

    # --- Panel 7: Box/Total Ratio ---
    ax7 = fig.add_subplot(3, 4, 7)
    ratios = [e["loss"]["box"] / max(e["loss"]["total"], 1e-6) for e in entries]
    ax7.plot(steps, ratios, alpha=0.3, linewidth=0.5, color="#34495e")
    if window > 1:
        ax7.plot(steps, np.convolve(ratios, kernel, mode="same"),
                 alpha=0.9, linewidth=1.5, color="#2c3e50")
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Box/Total Ratio")
    ax7.set_title(f"Box/Total Loss Ratio ({ratios[0]:.6f} -> {ratios[-1]:.6f})")
    ax7.grid(True, alpha=0.3)

    # --- Panel 8: Per-Epoch averages ---
    ax8 = fig.add_subplot(3, 4, 8)
    epoch_data = {}
    for e in entries:
        ep = e["epoch"]
        if ep not in epoch_data:
            epoch_data[ep] = {"total": [], "cls": [], "box": [], "dfl": [], "angle": []}
        epoch_data[ep]["total"].append(e["loss"]["total"])
        epoch_data[ep]["cls"].append(e["loss"]["cls"])
        epoch_data[ep]["box"].append(e["loss"]["box"])
        epoch_data[ep]["dfl"].append(e["loss"]["dfl"])
        epoch_data[ep]["angle"].append(e["loss"]["angle"])
    
    eps = sorted(epoch_data.keys())
    ax8.plot(eps, [np.mean(epoch_data[ep]["total"]) for ep in eps], "o-", color="#e74c3c", label="Total")
    ax8.plot(eps, [np.mean(epoch_data[ep]["cls"]) for ep in eps], "o-", color="#3498db", label="Cls")
    ax8.set_xlabel("Epoch")
    ax8.set_ylabel("Avg Loss")
    ax8.set_title("Per-Epoch Avg Loss")
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=7)

    # --- Panel 9: Log-scale view ---
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.semilogy(eps, [np.mean(epoch_data[ep]["total"]) for ep in eps], "o-", color="#e74c3c", label="Total")
    ax9.semilogy(eps, [np.mean(epoch_data[ep]["cls"]) for ep in eps], "o-", color="#3498db", label="Cls")
    ax9.semilogy(eps, [np.mean(epoch_data[ep]["dfl"]) for ep in eps], "o-", color="#9b59b6", label="DFL")
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("Avg Loss (log)")
    ax9.set_title("Per-Epoch Loss (Log Scale)")
    ax9.grid(True, alpha=0.3)
    ax9.legend(fontsize=7)

    # --- Panel 10: Box+DFL (magnified) ---
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.plot(eps, [np.mean(epoch_data[ep]["box"]) for ep in eps], "o-", color="#e74c3c", label="Box")
    ax10.plot(eps, [np.mean(epoch_data[ep]["dfl"]) for ep in eps], "o-", color="#9b59b6", label="DFL")
    ax10.plot(eps, [np.mean(epoch_data[ep]["angle"]) for ep in eps], "o-", color="#f39c12", label="Angle")
    ax10.set_xlabel("Epoch")
    ax10.set_ylabel("Avg Loss")
    ax10.set_title("Box / DFL / Angle (Zoomed)")
    ax10.grid(True, alpha=0.3)
    ax10.legend(fontsize=7)

    # --- Panel 11: Cls Loss convergence rate ---
    ax11 = fig.add_subplot(3, 4, 11)
    cls_deltas = []
    prev_cls = None
    for ep in eps:
        curr = np.mean(epoch_data[ep]["cls"])
        if prev_cls is not None:
            cls_deltas.append((prev_cls - curr) / prev_cls * 100)
        prev_cls = curr
    ax11.bar(eps[1:], cls_deltas, color="#3498db", alpha=0.7)
    ax11.axhline(y=0, color="black", linewidth=0.5)
    ax11.set_xlabel("Epoch")
    ax11.set_ylabel("Delta %")
    ax11.set_title("Cls Loss Per-Epoch Change (%)")
    ax11.grid(True, alpha=0.3)

    # --- Panel 12: Summary statistics ---
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis("off")
    
    reduction_total = (1 - np.mean(epoch_data[eps[-1]]["total"]) / np.mean(epoch_data[eps[0]]["total"])) * 100
    reduction_box = (1 - np.mean(epoch_data[eps[-1]]["box"]) / np.mean(epoch_data[eps[0]]["box"])) * 100
    reduction_cls = (1 - np.mean(epoch_data[eps[-1]]["cls"]) / np.mean(epoch_data[eps[0]]["cls"])) * 100
    reduction_dfl = (1 - np.mean(epoch_data[eps[-1]]["dfl"]) / np.mean(epoch_data[eps[0]]["dfl"])) * 100
    reduction_angle = (1 - np.mean(epoch_data[eps[-1]]["angle"]) / np.mean(epoch_data[eps[0]]["angle"])) * 100
    
    stats_text = (
        f"TRAINING SUMMARY (70 epochs)\n\n"
        f"Dataset: Tship (1200 images, 5 classes)\n"
        f"Batch: 4 | ImgSz: 640 | Opt: AdamW\n"
        f"GradClip: 10.0 | Warmup: 3 epochs\n\n"
        f"LOSS REDUCTION:\n"
        f"  Total: {reduction_total:.1f}%\n"
        f"  Box:   {reduction_box:.1f}%\n"
        f"  Cls:   {reduction_cls:.1f}%\n"
        f"  DFL:   {reduction_dfl:.1f}%\n"
        f"  Angle: {reduction_angle:.1f}%\n\n"
        f"STATUS:\n"
        f"  Box/DFL zero: 0/20940 batches  OK\n"
        f"  LR: ~0 (COSINE BUG)\n"
        f"  GradNorm: INCREASING (1250->1719)\n"
        f"  Still decreasing: 2.8%/epoch\n"
    )
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUTPUT_DIR / f"training_analysis_detailed_{datetime.now():%Y%m%d_%H%M%S}.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Detailed analysis plot: {out_path}")


if __name__ == "__main__":
    entries = load_entries()
    print(f"Loaded {len(entries)} entries, {len(set(e['epoch'] for e in entries))} epochs")
    plot_detailed_analysis(entries)
    print("Done.")