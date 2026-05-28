import json, sys
import numpy as np
from pathlib import Path

LOG_PATH = "runs/dualstream-train/tship_exp/training_log.jsonl"

entries = []
with open(LOG_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

print(f"Total entries: {len(entries)}")
last_epoch = entries[-1]["epoch"]
print(f"Epochs trained: {last_epoch}")

epochs = sorted(set(e["epoch"] for e in entries))
print(f"Unique epochs: {epochs}")

# First epoch stats
first_batch = entries[0]
print(f"\n=== First Batch ===")
loss = first_batch["loss"]
print(f"  step={first_batch['step']}, epoch={first_batch['epoch']}, batch={first_batch['batch']}")
print(f"  total={loss['total']:.4f}, box={loss['box']:.4f}, cls={loss['cls']:.4f}, dfl={loss['dfl']:.4f}, angle={loss['angle']:.4f}")
print(f"  lr={first_batch['lr']:.6f}, grad_norm={first_batch['grad_norm']:.2f}")

# Last batch stats
last_batch = entries[-1]
print(f"\n=== Last Batch ===")
loss = last_batch["loss"]
print(f"  step={last_batch['step']}, epoch={last_batch['epoch']}, batch={last_batch['batch']}")
print(f"  total={loss['total']:.4f}, box={loss['box']:.4f}, cls={loss['cls']:.4f}, dfl={loss['dfl']:.4f}, angle={loss['angle']:.4f}")
print(f"  lr={last_batch['lr']:.6f}, grad_norm={last_batch['grad_norm']:.2f}")

# Per-epoch average loss
epoch_losses = {}
for e in entries:
    ep = e["epoch"]
    if ep not in epoch_losses:
        epoch_losses[ep] = {"total": [], "box": [], "cls": [], "dfl": [], "angle": []}
    epoch_losses[ep]["total"].append(e["loss"]["total"])
    epoch_losses[ep]["box"].append(e["loss"]["box"])
    epoch_losses[ep]["cls"].append(e["loss"]["cls"])
    epoch_losses[ep]["dfl"].append(e["loss"]["dfl"])
    epoch_losses[ep]["angle"].append(e["loss"]["angle"])

print(f"\n=== Per-Epoch Average Losses ===")
header = f"{'Epoch':<6} {'Total':<12} {'Box':<10} {'Cls':<12} {'DFL':<10} {'Angle':<10}"
print(header)
print("-" * 60)
for ep in sorted(epoch_losses.keys()):
    avg_t = np.mean(epoch_losses[ep]["total"])
    avg_b = np.mean(epoch_losses[ep]["box"])
    avg_c = np.mean(epoch_losses[ep]["cls"])
    avg_d = np.mean(epoch_losses[ep]["dfl"])
    avg_a = np.mean(epoch_losses[ep]["angle"])
    print(f"{ep:<6} {avg_t:<12.4f} {avg_b:<10.4f} {avg_c:<12.4f} {avg_d:<10.4f} {avg_a:<10.4f}")

# Trend analysis
first_ep = sorted(epoch_losses.keys())[0]
last_ep = sorted(epoch_losses.keys())[-1]
red_t = (1 - np.mean(epoch_losses[last_ep]["total"]) / np.mean(epoch_losses[first_ep]["total"])) * 100
red_b = (1 - np.mean(epoch_losses[last_ep]["box"]) / np.mean(epoch_losses[first_ep]["box"])) * 100
red_c = (1 - np.mean(epoch_losses[last_ep]["cls"]) / np.mean(epoch_losses[first_ep]["cls"])) * 100
red_d = (1 - np.mean(epoch_losses[last_ep]["dfl"]) / np.mean(epoch_losses[first_ep]["dfl"])) * 100
red_a = (1 - np.mean(epoch_losses[last_ep]["angle"]) / np.mean(epoch_losses[first_ep]["angle"])) * 100

print(f"\n=== Loss Reduction (Epoch {first_ep} -> {last_ep}) ===")
print(f"  Total: {red_t:.1f}%")
print(f"  Box:   {red_b:.1f}%")
print(f"  Cls:   {red_c:.1f}%")
print(f"  DFL:   {red_d:.1f}%")
print(f"  Angle: {red_a:.1f}%")

# Box/DFL zero ratio
box_zero = sum(1 for e in entries if e["loss"]["box"] < 1e-6)
dfl_zero = sum(1 for e in entries if e["loss"]["dfl"] < 1e-6)
print(f"\n=== Box/DFL Zero Check ===")
print(f"  Box=0 batches:  {box_zero}/{len(entries)} ({100*box_zero/len(entries):.1f}%)")
print(f"  DFL=0 batches:  {dfl_zero}/{len(entries)} ({100*dfl_zero/len(entries):.1f}%)")

# Gradient norm stats
gn_start = np.mean([e["grad_norm"] for e in entries[:100]])
gn_end = np.mean([e["grad_norm"] for e in entries[-100:]])
print(f"\n=== Gradient Norm ===")
print(f"  Start (first 100): {gn_start:.2f}")
print(f"  End   (last 100):  {gn_end:.2f}")

# LR
print(f"\n=== Learning Rate ===")
print(f"  Start: {entries[0]['lr']:.6f}")
print(f"  End:   {entries[-1]['lr']:.6f}")

# Convergence check - last 3 epochs loss std
last_3_ep_losses = []
for ep in sorted(epoch_losses.keys())[-3:]:
    last_3_ep_losses.append(np.mean(epoch_losses[ep]["total"]))
if len(last_3_ep_losses) >= 2:
    delta = abs(last_3_ep_losses[-1] - last_3_ep_losses[-2]) / last_3_ep_losses[-2] * 100
    print(f"\n=== Convergence ===")
    print(f"  Last 3 epoch avg losses: {[f'{x:.4f}' for x in last_3_ep_losses]}")
    print(f"  Last delta: {delta:.2f}%")
    if delta < 1.0:
        print("  Status: NEARLY CONVERGED (delta < 1%)")
    elif delta < 5.0:
        print("  Status: STILL DECREASING (could benefit from more epochs)")
    else:
        print("  Status: NOT CONVERGED")

# Box/Cls loss ratio
box_cls_ratio_start = np.mean(epoch_losses[first_ep]["box"]) / np.mean(epoch_losses[first_ep]["cls"])
box_cls_ratio_end = np.mean(epoch_losses[last_ep]["box"]) / np.mean(epoch_losses[last_ep]["cls"])
print(f"\n=== Box/Cls Ratio ===")
print(f"  Start: {box_cls_ratio_start:.6f}")
print(f"  End:   {box_cls_ratio_end:.6f}")
if box_cls_ratio_end > 1.0:
    print("  WARNING: Box loss dominates over classification!")