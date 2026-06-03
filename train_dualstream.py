#!/usr/bin/env python
# Ultralytics YOLOv8 Dual-Stream Training Script
# For Visible + Infrared (RGB-T) Object Detection

"""
Training script for dual-stream YOLOv8 models.

This script demonstrates how to train a dual-stream YOLOv8 model for
visible + infrared object detection.

Usage:
    python train_dualstream.py --data dataset_root --epochs 100 --batch 16

Requirements:
    - Paired visible and infrared images
    - YOLO-format annotations
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules.dualstream_model import DualStreamYOLO, DualStreamDetectionModel
from ultralytics.data.dataset_obb import DualStreamOBBDataLoader
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.nms import non_max_suppression
from training_logger import create_logger, log_batch


def parse_args():
    args = SimpleNamespace()
    
    args.data = "./Tship"
    args.nc = 5
    
    args.model = "yolov8-dualstream.yaml"
    args.weights = None
    args.fusion_mode = "concat"
    
    args.epochs = 100
    args.batch = 4
    args.imgsz = 640
    args.device = "0"
    args.workers = 8
    
    args.optimizer = "AdamW"
    args.lr0 = 0.001
    args.lrf = 0.01
    args.momentum = 0.9
    args.weight_decay = 0.0005
    args.warmup_epochs = 3
    args.clip_grad = 10.0
    
    args.hsv_h = 0.015
    args.hsv_s = 0.7
    args.hsv_v = 0.4
    
    args.project = "runs/dualstream-train"
    args.name = "tship_exp"
    args.exist_ok = False
    args.cache = False
    args.resume = False
    args.verbose = True
    args.val_conf = 0.001
    args.val_iou = 0.5
    args.max_det = 300
    
    return args


def match_obb_predictions(pred_cls, target_cls, iou, iouv):
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    if pred_cls.shape[0] == 0 or target_cls.shape[0] == 0:
        return correct

    pred_cls = pred_cls.astype(np.int64)
    target_cls = target_cls.astype(np.int64)
    class_match = target_cls[:, None] == pred_cls[None, :]

    for ti, threshold in enumerate(iouv):
        matches = np.argwhere((iou >= threshold) & class_match)
        if matches.shape[0] == 0:
            continue
        match_ious = iou[matches[:, 0], matches[:, 1]]
        matches = matches[match_ious.argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), ti] = True
    return correct


@torch.no_grad()
def validate_epoch(model, val_loader, device, imgsz, nc, conf_thres=0.001, iou_thres=0.5, max_det=300):
    model.eval()
    val_loss = 0.0
    loss_batches = 0
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)

    for batch in val_loader:
        batch["vis"] = batch["vis"].to(device)
        batch["ir"] = batch["ir"].to(device)
        batch["labels"] = batch["labels"].to(device)

        try:
            loss = model(batch)
            val_loss += loss.item()
            loss_batches += 1
        except Exception as e:
            LOGGER.warning(f"Error in validation loss batch: {e}")

        try:
            preds = model.model._predict_dual({"vis": batch["vis"], "ir": batch["ir"]})
            detections = non_max_suppression(
                preds,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                nc=nc,
                max_det=max_det,
                rotated=True,
            )
            labels = batch["labels"]
            batch_size = batch["vis"].shape[0]

            for si in range(batch_size):
                target = labels[labels[:, 0].long() == si]
                target_cls = target[:, 1].long()
                target_bboxes = target[:, 2:7].clone()
                if target_bboxes.numel():
                    target_bboxes[:, :4] *= imgsz

                det = detections[si]
                if det.shape[0]:
                    pred_bboxes = torch.cat((det[:, :4], det[:, -1:]), dim=1)
                    pred_conf = det[:, 4]
                    pred_cls = det[:, 5].long()
                else:
                    pred_bboxes = torch.zeros(0, 5, device=device)
                    pred_conf = torch.zeros(0, device=device)
                    pred_cls = torch.zeros(0, dtype=torch.long, device=device)

                if pred_bboxes.shape[0] and target_bboxes.shape[0]:
                    iou = batch_probiou(target_bboxes, pred_bboxes).detach().cpu().numpy()
                    tp = match_obb_predictions(
                        pred_cls.detach().cpu().numpy(),
                        target_cls.detach().cpu().numpy(),
                        iou,
                        iouv.detach().cpu().numpy(),
                    )
                else:
                    tp = np.zeros((pred_bboxes.shape[0], iouv.numel()), dtype=bool)

                stats["tp"].append(tp)
                stats["conf"].append(pred_conf.detach().cpu().numpy())
                stats["pred_cls"].append(pred_cls.detach().cpu().numpy())
                stats["target_cls"].append(target_cls.detach().cpu().numpy())
        except Exception as e:
            LOGGER.warning(f"Error in validation metrics batch: {e}")

    metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
    }
    if stats["tp"]:
        tp = np.concatenate(stats["tp"], 0)
        conf = np.concatenate(stats["conf"], 0)
        pred_cls = np.concatenate(stats["pred_cls"], 0)
        target_cls = np.concatenate(stats["target_cls"], 0)
        if tp.shape[0] and target_cls.shape[0]:
            _, _, p, r, _, ap, *_ = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
            metrics = {
                "precision": float(p.mean()) if p.size else 0.0,
                "recall": float(r.mean()) if r.size else 0.0,
                "mAP50": float(ap[:, 0].mean()) if ap.size else 0.0,
                "mAP50-95": float(ap.mean()) if ap.size else 0.0,
            }

    return val_loss / max(loss_batches, 1), metrics


def train(args):
    LOGGER.info(f"\n{colorstr('green', 'Starting dual-stream Network training')}")
    LOGGER.info(f"Dataset: {args.data}")
    LOGGER.info(f"Model: {args.model}")
    LOGGER.info(f"Classes: {args.nc}")
    LOGGER.info(f"Device: {args.device}")
    LOGGER.info(f"Optimizer: {args.optimizer}, lr0={args.lr0}, lrf={args.lrf}")
    
    device = args.device if args.device else "0"
    if torch.cuda.is_available():
        device = f"cuda:{device}" if device != "cpu" else "cpu"
    else:
        device = "cpu"
    
    LOGGER.info(f"Using device: {device}")
    
    if args.weights:
        LOGGER.info(f"Loading weights from {args.weights}")
        model = DualStreamYOLO(args.weights, verbose=args.verbose)
    else:
        LOGGER.info(f"Creating model from config: {args.model}")
        model = DualStreamYOLO(args.model, verbose=args.verbose)
    
    model.to(device)
    
    LOGGER.info("\nLoading datasets...")
    
    train_loader = DualStreamOBBDataLoader(
        img_path=Path(args.data),
        mode="train",
        batch_size=args.batch,
        imgsz=args.imgsz,
        augment=True,
        num_workers=args.workers,
        cache=args.cache,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )
    
    val_loader = DualStreamOBBDataLoader(
        img_path=Path(args.data),
        mode="val",
        batch_size=args.batch,
        imgsz=args.imgsz,
        augment=False,
        num_workers=args.workers,
        cache=args.cache,
    )
    
    LOGGER.info(f"Train dataset: {len(train_loader.dataset)} images")
    LOGGER.info(f"Val dataset: {len(val_loader.dataset)} images")
    
    total_batches = len(train_loader)
    total_steps = args.epochs * total_batches
    warmup_steps = args.warmup_epochs * total_batches
    
    # Create optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer_map = {
        "SGD": lambda: torch.optim.SGD(trainable_params, lr=args.lr0, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True),
        "Adam": lambda: torch.optim.Adam(trainable_params, lr=args.lr0, betas=(args.momentum, 0.999),
                                          weight_decay=args.weight_decay),
        "AdamW": lambda: torch.optim.AdamW(trainable_params, lr=args.lr0, betas=(args.momentum, 0.999),
                                             weight_decay=args.weight_decay),
    }
    
    if args.optimizer not in optimizer_map:
        LOGGER.warning(f"Unknown optimizer '{args.optimizer}', falling back to AdamW")
        args.optimizer = "AdamW"
    
    optimizer = optimizer_map[args.optimizer]()
    LOGGER.info(f"Created {args.optimizer} optimizer with {sum(p.numel() for p in trainable_params):,} parameters")
    
    # Cosine learning rate scheduler with warmup
    # LambdaLR multiplies base_lr by the return value of lr_lambda.
    # lr_lambda returns a MULTIPLIER, not an absolute LR value.
    # Warmup: multiplier ramps from 0 → 1.0
    # Cosine: multiplier decays from 1.0 → lrf
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps       # 0 → 1.0 (effective LR: 0 → lr0)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(torch.pi) * progress).item())
        return args.lrf + (1 - args.lrf) * cosine_decay  # 1.0 → lrf (effective LR: lr0 → lrf*lr0)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    
    model_yaml = args.model
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "device": device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "verbose": args.verbose,
        "nc": args.nc,
        "fusion_mode": args.fusion_mode,
        "warmup_epochs": args.warmup_epochs,
        "clip_grad": args.clip_grad,
        "model": str(Path(model_yaml).resolve()),  # save YAML path for loading
    }
    
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "train_config.yaml", "w") as f:
        yaml.dump(train_config, f)
    
    LOGGER.info(f"\nTraining configuration saved to {save_dir}")
    
    train_logger = create_logger(save_dir, name="training_log.jsonl")
    LOGGER.info(f"Training log: {train_logger.log_path}")
    
    LOGGER.info(f"\n{colorstr('blue', 'Starting training for')} {args.epochs} epochs...")
    LOGGER.info(f"Total batches/epoch: {total_batches}, Warmup: {warmup_steps} steps")
    
    best_fitness = float("inf")
    best_epoch = -1
    
    for epoch in range(args.epochs):
        LOGGER.info(f"\n{'─' * 60}")
        LOGGER.info(f"Epoch {epoch + 1}/{args.epochs}  lr: {optimizer.param_groups[0]['lr']:.6f}")
        LOGGER.info(f"{'─' * 60}")
        
        model.train()
        train_loss = 0.0
        nan_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch["vis"] = batch["vis"].to(device)
            batch["ir"] = batch["ir"].to(device)
            batch["labels"] = batch["labels"].to(device)
            
            try:
                loss = model(batch)
                loss_items = model.get_loss_items()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count <= 3:
                        LOGGER.warning(
                            f"  [{batch_idx + 1:3d}/{total_batches}] NaN/Inf loss detected — skipping batch"
                        )
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                
                grad_norm = None
                if args.clip_grad > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.clip_grad
                    )
                    if grad_norm is not None and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        optimizer.zero_grad()
                        nan_count += 1
                        if nan_count <= 3:
                            LOGGER.warning(
                                f"  [{batch_idx + 1:3d}/{total_batches}] NaN/Inf gradient — skipping batch"
                            )
                        continue
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                log_batch(
                    train_logger, epoch + 1, batch_idx + 1,
                    loss_value=loss.item(),
                    loss_items=loss_items,
                    lr_value=optimizer.param_groups[0]["lr"],
                    grad_norm_value=grad_norm.item() if grad_norm is not None else None,
                )
                
                if (batch_idx + 1) % 10 == 0:
                    if loss_items is not None and len(loss_items) >= 4:
                        box_loss = loss_items[0].item()
                        cls_loss = loss_items[1].item()
                        dfl_loss = loss_items[2].item()
                        angle_loss = loss_items[3].item()
                        
                        msg = (f"  [{batch_idx + 1:3d}/{total_batches}] "
                               f"Loss: {loss.item():.4f} | "
                               f"Box: {box_loss:.4f} | Cls: {cls_loss:.4f} | "
                               f"DFL: {dfl_loss:.4f} | Angle: {angle_loss:.4f}")
                        if grad_norm is not None:
                            msg += f" | Grad: {grad_norm.item():.2f}"
                        LOGGER.info(msg)
                    else:
                        LOGGER.info(f"  [{batch_idx + 1:3d}/{total_batches}] Loss: {loss.item():.4f}")
            
            except Exception as e:
                import traceback
                LOGGER.warning(f"Error in batch {batch_idx}: {e}")
                LOGGER.warning(f"Traceback: {traceback.format_exc()}")
                continue
        
        train_loss /= total_batches
        LOGGER.info(f"  Avg training loss: {train_loss:.4f}")
        if nan_count > 0:
            LOGGER.warning(f"  Skipped {nan_count} batches due to NaN/Inf — consider reducing lr or box weight")
        
        train_logger.flush()
        
        val_loss, val_metrics = validate_epoch(
            model,
            val_loader,
            device,
            args.imgsz,
            args.nc,
            conf_thres=args.val_conf,
            iou_thres=args.val_iou,
            max_det=args.max_det,
        )
        LOGGER.info(
            f"  Validation loss: {val_loss:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
            f"mAP50: {val_metrics['mAP50']:.4f} | mAP50-95: {val_metrics['mAP50-95']:.4f}"
        )
        train_logger.log_epoch(epoch + 1, train_loss, val_loss, val_metrics)
        train_logger.flush()
        
        if val_loss < best_fitness:
            best_fitness = val_loss
            best_epoch = epoch
            best_model_path = save_dir / "best.pt"
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": val_loss,
                "train_args": train_config,
            }, best_model_path)
            
            LOGGER.info(f"  ✓ New best model saved ({best_fitness:.4f}) → {best_model_path}")
        else:
            LOGGER.info(f"  (best so far: {best_fitness:.4f} at epoch {best_epoch + 1})")
        
        last_model_path = save_dir / "last.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": val_loss,
            "train_args": train_config,
        }, last_model_path)
    
    LOGGER.info(f"\n{colorstr('green', 'Training completed!')}")
    LOGGER.info(f"Best epoch: {best_epoch + 1} with val loss: {best_fitness:.4f}")
    LOGGER.info(f"Best model: {save_dir / 'best.pt'}")
    
    train_logger.close()
    LOGGER.info(f"Training log saved to: {save_dir / 'training_log.jsonl'}")
    LOGGER.info(f"Visualize with: python visualize_training.py --log {save_dir / 'training_log.jsonl'}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
