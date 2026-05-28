#!/usr/bin/env python
"""
Training Visualization Script for Dual-Stream YOLOv8.

Supports:
  - Offline mode: reads training logs (JSONL/CSV) and generates plots
  - Export mode: exports data to CSV/JSON
  - Batch comparison: compare multiple training runs

Usage:
    # Offline: visualize a completed training log
    python visualize_training.py --log runs/dualstream-train/tship_exp/training_log.jsonl

    # Export data to CSV
    python visualize_training.py --log training_log.jsonl --export-csv output.csv

    # Compare multiple runs
    python visualize_training.py --logs run1/log.jsonl run2/log.jsonl --labels "exp1" "exp2"
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dual-stream YOLOv8 training metrics")

    parser.add_argument("--log", type=str, default=None,
                        help="Path to training log file (JSONL or CSV)")
    parser.add_argument("--logs", type=str, nargs="+", default=None,
                        help="Multiple log files for comparison")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for multiple runs")

    parser.add_argument("--export-csv", type=str, default=None,
                        help="Export data to CSV file")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export data to JSON file")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as log file)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg", "jpg"],
                        help="Output image format")

    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI")
    parser.add_argument("--figsize", type=str, default="20,12",
                        help="Figure size as W,H in inches")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Smoothing window size (0 = no smoothing)")

    parser.add_argument("--title", type=str, default="Dual-Stream YOLOv8 Training Metrics",
                        help="Main title for the plots")

    return parser.parse_args()


def load_jsonl(filepath):
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_csv(filepath):
    import csv
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({k: float(v) if v.replace(".", "").replace("-", "").isdigit() else v for k, v in row.items()})
    return entries


def load_log(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")
    if filepath.suffix == ".jsonl":
        return load_jsonl(filepath)
    elif filepath.suffix == ".csv":
        return load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def smooth(data, window=5):
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def extract_metrics(entries):
    metrics = {
        "epoch": [],
        "batch": [],
        "step": [],
        "total_loss": [],
        "box_loss": [],
        "cls_loss": [],
        "dfl_loss": [],
        "angle_loss": [],
        "lr": [],
        "grad_norm": [],
    }

    for i, e in enumerate(entries):
        if "epoch" in e:
            metrics["epoch"].append(e["epoch"])
        if "batch" in e:
            metrics["batch"].append(e["batch"])
        if "step" in e:
            metrics["step"].append(e["step"])
        else:
            metrics["step"].append(i)

        lt = e.get("loss")
        if isinstance(lt, dict):
            metrics["total_loss"].append(lt.get("total", 0))
            metrics["box_loss"].append(lt.get("box", 0))
            metrics["cls_loss"].append(lt.get("cls", 0))
            metrics["dfl_loss"].append(lt.get("dfl", 0))
            metrics["angle_loss"].append(lt.get("angle", 0))
        else:
            metrics["total_loss"].append(lt if lt is not None else 0)
            metrics["box_loss"].append(e.get("box_loss", 0))
            metrics["cls_loss"].append(e.get("cls_loss", 0))
            metrics["dfl_loss"].append(e.get("dfl_loss", 0))
            metrics["angle_loss"].append(e.get("angle_loss", 0))

        metrics["lr"].append(e.get("lr", 0))
        metrics["grad_norm"].append(e.get("grad_norm", 0))

    return metrics


def filter_nonzero(data, eps=1e-6):
    return [v if v > eps else np.nan for v in data]


def create_plots(metrics_list, labels_list, args, output_dir):
    figsize = tuple(map(int, args.figsize.split(",")))
    fig = plt.figure(figsize=figsize)
    fig.suptitle(args.title, fontsize=16, fontweight="bold")

    has_multiple = len(metrics_list) > 1
    colors = plt.cm.tab10.colors if has_multiple else ["#1f77b4"]
    line_alpha = 0.7 if has_multiple else 0.85

    # 1) Total Loss
    ax1 = fig.add_subplot(2, 4, 1)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["total_loss"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax1.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if has_multiple:
        ax1.legend(fontsize=7)

    # 2) Box Loss
    ax2 = fig.add_subplot(2, 4, 2)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["box_loss"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax2.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Box Loss (ProbIoU)")
    ax2.grid(True, alpha=0.3)

    # 3) Cls Loss
    ax3 = fig.add_subplot(2, 4, 3)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["cls_loss"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax3.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Classification Loss")
    ax3.grid(True, alpha=0.3)

    # 4) DFL Loss
    ax4 = fig.add_subplot(2, 4, 4)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["dfl_loss"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax4.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Loss")
    ax4.set_title("DFL Loss")
    ax4.grid(True, alpha=0.3)

    # 5) Angle Loss
    ax5 = fig.add_subplot(2, 4, 5)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["angle_loss"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax5.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Loss")
    ax5.set_title("Angle Loss")
    ax5.grid(True, alpha=0.3)

    # 6) Learning Rate
    ax6 = fig.add_subplot(2, 4, 6)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["lr"]
        vals = [v for v in vals if v > 1e-12]
        steps_for_lr = steps[:len(vals)] if len(vals) < len(steps) else steps
        vals_for_lr = vals[:len(steps_for_lr)] if len(vals) > len(steps_for_lr) else vals
        if len(steps_for_lr) > len(vals_for_lr):
            steps_for_lr = steps_for_lr[:len(vals_for_lr)]
        elif len(vals_for_lr) > len(steps_for_lr):
            vals_for_lr = vals_for_lr[:len(steps_for_lr)]
        if vals_for_lr:
            ax6.plot(steps_for_lr, vals_for_lr, color=colors[idx % len(colors)],
                     alpha=line_alpha, linewidth=1.2, label=label)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Learning Rate")
    ax6.set_title("Learning Rate")
    ax6.grid(True, alpha=0.3)
    ax6.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # 7) Gradient Norm
    ax7 = fig.add_subplot(2, 4, 7)
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        steps = metrics["step"]
        vals = metrics["grad_norm"]
        vals = filter_nonzero(vals)
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax7.plot(steps, vals, color=colors[idx % len(colors)],
                 alpha=line_alpha, linewidth=1.0, label=label)
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Gradient Norm")
    ax7.set_title("Gradient Norm")
    ax7.grid(True, alpha=0.3)

    # 8) All losses (overlay)
    ax8 = fig.add_subplot(2, 4, 8)
    primary_metrics = metrics_list[0]
    steps = primary_metrics["step"]
    components = {
        "Box": filter_nonzero(primary_metrics["box_loss"]),
        "Cls": filter_nonzero(primary_metrics["cls_loss"]),
        "DFL": filter_nonzero(primary_metrics["dfl_loss"]),
        "Angle": filter_nonzero(primary_metrics["angle_loss"]),
    }
    comp_colors = {"Box": "#e74c3c", "Cls": "#3498db", "DFL": "#2ecc71", "Angle": "#9b59b6"}
    for name, vals in components.items():
        if args.smooth > 1:
            vals = smooth(np.nan_to_num(np.array(vals), nan=0), args.smooth)
        ax8.plot(steps, vals, color=comp_colors[name],
                 alpha=0.8, linewidth=1.0, label=name)
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Loss")
    ax8.set_title("All Loss Components")
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"training_metrics_{timestamp}.{args.format}"
    plt.savefig(str(metrics_path), dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✓ Metrics plot saved to: {metrics_path}")

    # Per-epoch summary plot
    if primary_metrics["epoch"]:
        epoch_steps = sorted(set(zip(primary_metrics["epoch"], primary_metrics["step"])))
        epoch_losses = {}
        for ep, st in epoch_steps:
            if ep not in epoch_losses:
                epoch_losses[ep] = []
            for m in metrics_list:
                idx = m["step"].index(st) if st in m["step"] else None
                if idx is not None:
                    epoch_losses[ep].append(m["total_loss"][idx] if m["total_loss"] else 0)

        epoch_avgs = {ep: np.mean(vals) for ep, vals in epoch_losses.items() if vals}
        if epoch_avgs:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            epochs_sorted = sorted(epoch_avgs.keys())
            ax2.plot(epochs_sorted, [epoch_avgs[ep] for ep in epochs_sorted],
                     "o-", markersize=4, linewidth=1.5, color="#2c3e50")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Average Total Loss")
            ax2.set_title("Per-Epoch Total Loss")
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            epoch_path = output_dir / f"training_epoch_loss_{timestamp}.{args.format}"
            plt.savefig(str(epoch_path), dpi=args.dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            plt.close(fig2)
            print(f"  ✓ Per-epoch plot saved to: {epoch_path}")


def export_to_csv(metrics_list, labels_list, filepath):
    import csv
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["run", "step", "epoch", "batch", "total_loss", "box_loss",
                   "cls_loss", "dfl_loss", "angle_loss", "lr", "grad_norm"]
        writer.writerow(header)
        for run_idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
            n = len(metrics["step"])
            for i in range(n):
                row = [
                    label,
                    metrics["step"][i] if i < len(metrics["step"]) else "",
                    metrics["epoch"][i] if i < len(metrics["epoch"]) else "",
                    metrics["batch"][i] if i < len(metrics["batch"]) else "",
                    metrics["total_loss"][i] if i < len(metrics["total_loss"]) else "",
                    metrics["box_loss"][i] if i < len(metrics["box_loss"]) else "",
                    metrics["cls_loss"][i] if i < len(metrics["cls_loss"]) else "",
                    metrics["dfl_loss"][i] if i < len(metrics["dfl_loss"]) else "",
                    metrics["angle_loss"][i] if i < len(metrics["angle_loss"]) else "",
                    metrics["lr"][i] if i < len(metrics["lr"]) else "",
                    metrics["grad_norm"][i] if i < len(metrics["grad_norm"]) else "",
                ]
                writer.writerow(row)
    print(f"  ✓ Data exported to CSV: {filepath}")


def export_to_json(metrics_list, labels_list, filepath):
    data = {}
    for idx, (metrics, label) in enumerate(zip(metrics_list, labels_list)):
        data[label] = {k: v for k, v in metrics.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✓ Data exported to JSON: {filepath}")


def main():
    args = parse_args()

    if args.logs:
        log_paths = [Path(p) for p in args.logs]
    elif args.log:
        log_paths = [Path(args.log)]
    else:
        print("ERROR: Specify --log or --logs")
        sys.exit(1)

    labels_list = args.labels if args.labels else [f"run_{i}" for i in range(len(log_paths))]
    if len(labels_list) != len(log_paths):
        labels_list = [f"run_{i}" for i in range(len(log_paths))]

    metrics_list = []
    for log_path in log_paths:
        print(f"Loading: {log_path}")
        entries = load_log(log_path)
        print(f"  {len(entries)} entries loaded")
        metrics = extract_metrics(entries)
        metrics_list.append(metrics)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    create_plots(metrics_list, labels_list, args, output_dir)

    if args.export_csv:
        export_to_csv(metrics_list, labels_list, args.export_csv)

    if args.export_json:
        export_to_json(metrics_list, labels_list, args.export_json)

    print("\nDone.")


if __name__ == "__main__":
    main()