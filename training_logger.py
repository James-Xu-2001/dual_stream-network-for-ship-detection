"""
Training Logger for Dual-Stream YOLOv8.

Records training metrics in JSONL format (one JSON object per line) for:
  - Easy parsing and visualization
  - Low memory overhead
  - Real-time tail-follow capability

Each line contains:
  {
    "step": int,           # Global step counter
    "epoch": int,          # Current epoch (0-indexed or 1-indexed)
    "batch": int,          # Batch index within epoch
    "loss": {              # Loss components
      "total": float,
      "box": float,
      "cls": float,
      "dfl": float,
      "angle": float
    },
    "lr": float,           # Current learning rate
    "grad_norm": float,    # Gradient norm (after clipping, if applied)
    "timestamp": str       # ISO 8601 timestamp
  }
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


class TrainingLogger:
    def __init__(self, log_path: str | Path, flush_interval: int = 1):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval

        self._file = None
        self._step_counter = 0
        self._write_count = 0

    def open(self):
        self._file = open(self.log_path, "w", encoding="utf-8")
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def log(
        self,
        epoch: int,
        batch: int,
        total_loss: float,
        box_loss: float = 0.0,
        cls_loss: float = 0.0,
        dfl_loss: float = 0.0,
        angle_loss: float = 0.0,
        lr: float = 0.0,
        grad_norm: Optional[float] = None,
    ):
        entry = {
            "step": self._step_counter,
            "epoch": epoch,
            "batch": batch,
            "loss": {
                "total": total_loss,
                "box": box_loss,
                "cls": cls_loss,
                "dfl": dfl_loss,
                "angle": angle_loss,
            },
            "lr": lr,
            "grad_norm": grad_norm if grad_norm is not None else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._step_counter += 1
        self._write_count += 1

        if self._write_count % self.flush_interval == 0:
            self._file.flush()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[dict] = None,
    ):
        entry = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "metrics": metrics or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._write_count += 1

        if self._write_count % self.flush_interval == 0:
            self._file.flush()

    def flush(self):
        if self._file:
            self._file.flush()


def create_logger(save_dir: str | Path, name: str = "training_log.jsonl") -> TrainingLogger:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / name
    logger = TrainingLogger(log_path)
    logger.open()
    return logger


def log_batch(
    logger: TrainingLogger,
    epoch: int,
    batch_idx: int,
    loss_value: float,
    loss_items,   # torch.Tensor of shape [4]  or [4,] 
    lr_value: float,
    grad_norm_value: Optional[float] = None,
):
    if loss_items is not None and hasattr(loss_items, "__len__") and len(loss_items) >= 4:
        box = float(loss_items[0])
        cls = float(loss_items[1])
        dfl = float(loss_items[2])
        angle = float(loss_items[3])
    elif loss_items is not None and hasattr(loss_items, "__len__") and len(loss_items) >= 3:
        box = float(loss_items[0])
        cls = float(loss_items[1])
        dfl = float(loss_items[2])
        angle = 0.0
    else:
        box = cls = dfl = angle = 0.0

    logger.log(
        epoch=epoch,
        batch=batch_idx,
        total_loss=float(loss_value),
        box_loss=box,
        cls_loss=cls,
        dfl_loss=dfl,
        angle_loss=angle,
        lr=float(lr_value),
        grad_norm=float(grad_norm_value) if grad_norm_value is not None else None,
    )
