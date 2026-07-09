#!/usr/bin/env python
# Ultralytics YOLOv8 Dual-Stream Training Script
# For Visible + Infrared (RGB-T) Object Detection

"""
Training script for dual-stream YOLOv8 models.

This script demonstrates how to train a dual-stream YOLOv8 model for
visible + infrared object detection.

Usage:
    python train_dualstream.py --config train_dualstream.yaml
    python train_dualstream.py --data dataset_root --epochs 100 --batch 16

Requirements:
    - Paired visible and infrared images
    - YOLO-format annotations
"""

import argparse
import math
import sys
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training_logger import create_logger, log_batch

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("train_dualstream")


def colorstr(*args):
    return str(args[-1])


def parse_scalar(value):
    value = value.strip()
    if value == "":
        return None
    if value[0:1] == value[-1:] and value[:1] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def yaml_safe_load(stream):
    if yaml is not None:
        return yaml.safe_load(stream)

    data = {}
    for raw_line in stream:
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def yaml_safe_dump(data, stream):
    if yaml is not None:
        yaml.safe_dump(data, stream, sort_keys=False)
        return

    for key, value in data.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif value is None:
            value = ""
        stream.write(f"{key}: {value}\n")

# 数据类型
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

# 加载 YAML 配置文件，返回配置字典和配置文件路径
def load_config(config_path):
    """Load YAML training config. Raises FileNotFoundError if config is missing."""
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml_safe_load(f) or {}
    return config, path


def increment_path(path):
    path = Path(path)
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.name}_{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find available save directory for {path}")


def parse_args():
    # 从 sys.argv 中提取 --config 的值，确定加载哪个 YAML
    config_arg = str(ROOT / "argument.yaml")
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--config" and i + 2 < len(sys.argv):
            config_arg = sys.argv[i + 2]
            break
        elif arg.startswith("--config="):
            config_arg = arg.split("=", 1)[1]
            break

    config, config_path = load_config(config_arg)

    # 单一 parser：YAML 值作为 default，CLI 自然覆盖
    parser = argparse.ArgumentParser(description="Train a dual-stream YOLOv8 OBB model.")
    parser.add_argument("--config", default=str(config_path), help="Path to YAML training config.")

    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=str_to_bool, nargs="?", const=True, default=value)
        elif value is None:
            parser.add_argument(f"--{key}", type=str, default=None)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    # argparse 的 default 机制天然实现了"YAML 是底座，CLI 是覆盖层"的优先级链。
    args = parser.parse_args()
    return SimpleNamespace(**vars(args))


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
                # 用于获取张量或数组第一维度大小的属性
                if det.shape[0]:
                    #-1:最后一列，dim=1:按列拼接
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
    global LOGGER, DualStreamOBBDataLoader, DualStreamYOLO, ap_per_class, batch_probiou, colorstr, non_max_suppression

    from ultralytics.data.dataset_obb import DualStreamOBBDataLoader
    from ultralytics.nn.modules.dualstream_model import DualStreamYOLO
    from ultralytics.utils import LOGGER as ultralytics_logger
    from ultralytics.utils import colorstr as ultralytics_colorstr
    from ultralytics.utils.metrics import ap_per_class, batch_probiou
    from ultralytics.utils.nms import non_max_suppression

    LOGGER = ultralytics_logger
    colorstr = ultralytics_colorstr

    LOGGER.info(f"\n{colorstr('green', 'Starting dual-stream Network training')}")
    LOGGER.info(f"Dataset: {args.data}")
    LOGGER.info(f"Model: {args.model}")
    LOGGER.info(f"Config: {args.config}")
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
    
    # 将损失权重从 argument.yaml 注入到 model.args（覆盖硬编码默认值）
    for loss_key in ("box", "cls", "dfl", "angle"):
        val = getattr(args, loss_key, None)
        if val is not None:
            setattr(model.model.args, loss_key, val)
    
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
    if total_batches == 0:
        raise RuntimeError("Training dataloader is empty. Check dataset path and split structure.")
    total_steps = args.epochs * total_batches
    warmup_steps = args.warmup_epochs * total_batches
    
    # Create optimizer 确定可训练参数数量
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    
    optimizer_map = {
        "SGD": lambda: torch.optim.SGD(trainable_params, lr=args.lr0, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=True),
        "Adam": lambda: torch.optim.Adam(trainable_params, lr=args.lr0, betas=(args.momentum, 0.999),
                                          weight_decay=args.weight_decay),
        "AdamW": lambda: torch.optim.AdamW(trainable_params, lr=args.lr0, betas=(args.momentum, 0.999),
                                             weight_decay=args.weight_decay),
    }
    # 如果yaml文件中出现的优化器不在map中，则使用AdamW优化器，并给出警告
    if args.optimizer not in optimizer_map:
        LOGGER.warning(f"Unknown optimizer '{args.optimizer}', falling back to AdamW")
        args.optimizer = "AdamW"
    
    optimizer = optimizer_map[args.optimizer]()
    LOGGER.info(f"Created {args.optimizer} optimizer with {trainable_param_count:,} trainable parameters")
    
    # 学习率调度器，使用预热结合余弦退火算法
    def lr_lambda(step):
        # 预热阶段，学习率线性增加
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        # 预热阶段后，使用余弦退火算法
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # 余弦退火算法，学习率线性增加到最大值，然后线性减小到最小值，cosine_decay从1降至0.
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return args.lrf + (1 - args.lrf) * cosine_decay

    # 学习率调度器，便于动态调节学习率
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    
    model_yaml = args.model
    train_config = vars(args).copy()
    train_config["device"] = device
    train_config["model"] = str(Path(model_yaml).resolve())
    
    save_dir = Path(args.project) / args.name
    # 若已存在相同路径，则自动递增路径。eg. exp1->exp2
    save_dir = save_dir if args.exist_ok else increment_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "train_config.yaml", "w", encoding="utf-8") as f:
        yaml_safe_dump(train_config, f)
    
    LOGGER.info(f"\nTraining configuration saved to {save_dir}")
    
    train_logger = create_logger(save_dir, name="training_log.jsonl")
    LOGGER.info(f"Training log: {train_logger.log_path}")
    
    LOGGER.info(f"\n{colorstr('blue', 'Starting training for')} {args.epochs} epochs...")
    LOGGER.info(f"Total batches/epoch: {total_batches}, Warmup: {warmup_steps} steps")
    
    save_metric = args.save_metric
    metric_mode = "min" if save_metric in {"loss", "val_loss"} else "max"
    best_fitness = float("inf") if metric_mode == "min" else -float("inf")
    best_epoch = -1
    
    for epoch in range(args.epochs):
        LOGGER.info(f"\n{'-' * 60}")
        LOGGER.info(f"Epoch {epoch + 1}/{args.epochs}  lr: {optimizer.param_groups[0]['lr']:.6f}")
        LOGGER.info(f"{'-' * 60}")
        
        model.train()
        train_loss = 0.0
        valid_batches = 0
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
                            f"  [{batch_idx + 1:3d}/{total_batches}] NaN/Inf loss detected - skipping batch"
                        )
                    continue
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                grad_norm = None
                if args.clip_grad > 0:
                    # 计算梯度的L2范数，如果范数大于max_norm，则进行裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.clip_grad
                    )
                    if grad_norm is not None and (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        optimizer.zero_grad(set_to_none=True)
                        nan_count += 1
                        if nan_count <= 3:
                            LOGGER.warning(
                                f"  [{batch_idx + 1:3d}/{total_batches}] NaN/Inf gradient detected - skipping batch"
                            )
                        continue
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                # 实际产生loss次数valid_batches
                valid_batches += 1
                # Batch级别日志
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
        
        train_loss /= max(valid_batches, 1)
        LOGGER.info(f"  Avg training loss: {train_loss:.4f}")
        if nan_count > 0:
            LOGGER.warning(f"  Skipped {nan_count} batches due to NaN/Inf - consider reducing lr or box weight")
        # train_logger.flush()作用是强制将日志缓冲区中的数据立即写入磁盘，确保训练过程中的所有记录被安全持久化，而不会因为程序意外崩溃而丢失。
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
        # .4f表示保留4位小数, f""表示格式化字符串
        LOGGER.info(
            f"  Validation loss: {val_loss:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
            f"mAP50: {val_metrics['mAP50']:.4f} | mAP50-95: {val_metrics['mAP50-95']:.4f}"
        )
        train_logger.log_epoch(epoch + 1, train_loss, val_loss, val_metrics)
        train_logger.flush()
        
        fitness = val_loss if metric_mode == "min" else val_metrics.get(save_metric, 0.0)
        improved = fitness < best_fitness if metric_mode == "min" else fitness > best_fitness

        if improved:
            best_fitness = fitness
            best_epoch = epoch
            best_model_path = save_dir / "best.pt"
            # 保存最佳模型检查点的标准操作
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": val_loss,
                "metrics": val_metrics,
                "fitness": best_fitness,
                "save_metric": save_metric,
                "train_args": train_config,
            }, best_model_path)
            
            LOGGER.info(f"  New best model saved ({save_metric}={best_fitness:.4f}) -> {best_model_path}")
        else:
            LOGGER.info(f"  (best {save_metric}: {best_fitness:.4f} at epoch {best_epoch + 1})")
        
        last_model_path = save_dir / "last.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": val_loss,
            "metrics": val_metrics,
            "fitness": fitness,
            "save_metric": save_metric,
            "train_args": train_config,
        }, last_model_path)
    
    LOGGER.info(f"\n{colorstr('green', 'Training completed!')}")
    LOGGER.info(f"Best epoch: {best_epoch + 1} with {save_metric}: {best_fitness:.4f}")
    LOGGER.info(f"Best model: {save_dir / 'best.pt'}")
    
    train_logger.close()
    LOGGER.info(f"Training log saved to: {save_dir / 'training_log.jsonl'}")
    LOGGER.info(f"Visualize with: python visualize_training.py --log {save_dir / 'training_log.jsonl'}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
