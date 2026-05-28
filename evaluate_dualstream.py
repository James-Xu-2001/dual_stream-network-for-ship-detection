#!/usr/bin/env python
"""Comprehensive evaluation script for Dual-Stream YOLOv8 OBB models.

This script evaluates a trained dual-stream YOLOv8 model on the validation dataset,
computing standard detection metrics (mAP, Precision, Recall, F1-score), generating
a confusion matrix, per-class classification reports, and visualization plots.

Usage:
    python evaluate_dualstream.py --weights runs/dualstream-train/tship_exp/best.pt --data ./Tship
    python evaluate_dualstream.py --weights best.pt --data ./Tship --conf 0.25 --iou 0.5 --visualize
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.nn.modules.dualstream_model import DualStreamYOLO
from ultralytics.data.dataset_obb import DualStreamOBBDataLoader
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import batch_probiou, ap_per_class

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Dual-Stream YOLOv8 OBB Model")

    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="./Tship", help="Path to dataset root directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="0", help="Device: '0', '1', or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for NMS")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")

    parser.add_argument("--project", type=str, default="runs/dualstream-eval", help="Project save directory")
    parser.add_argument("--name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate visualization plots")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="Skip visualization")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save results to JSON")
    parser.add_argument("--save-txt", action="store_true", default=False, help="Save per-image detection txt files")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Overwrite existing results")

    return parser.parse_args()


class DualStreamOBBEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = self._resolve_device(args.device)
        self.save_dir = Path(args.project) / args.name
        self.save_dir.mkdir(parents=True, exist_ok=args.exist_ok)

        self.names: dict[int, str] = self._load_class_names(args.data)
        self.nc: int = len(self.names)

        self.model: DualStreamYOLO | None = None
        self.dataloader = None

        self.all_preds: list[dict[str, torch.Tensor]] = []
        self.all_targets: list[dict[str, torch.Tensor]] = []
        self.inference_times: list[float] = []
        self.nms_times: list[float] = []
        self.total_images: int = 0
        self.total_gt_objects: int = 0
        self.total_detections: int = 0

        logger.info("Evaluator initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Classes: {self.nc} -> {self.names}")
        logger.info(f"  Save dir: {self.save_dir}")

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str.lower() == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_str}")
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")

    @staticmethod
    def _load_class_names(data_root: str) -> dict[int, str]:
        classes_file = Path(data_root) / "label" / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r", encoding="utf-8") as f:
                names = {i: line.strip() for i, line in enumerate(f) if line.strip()}
            return names
        return {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}

    def load_model(self) -> None:
        logger.info(f"Loading model from: {self.args.weights}")
        self.model = DualStreamYOLO(self.args.weights, verbose=False)
        self.model.to(self.device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {param_count:,} total, {trainable_count:,} trainable")

    def prepare_data(self) -> None:
        logger.info(f"Loading validation dataset from: {self.args.data}")
        self.dataloader = DualStreamOBBDataLoader(
            img_path=Path(self.args.data),
            mode="val",
            batch_size=self.args.batch,
            imgsz=self.args.imgsz,
            augment=False,
            num_workers=self.args.workers,
        )
        self.total_images = len(self.dataloader.dataset)
        logger.info(f"  Validation images: {self.total_images}")
        logger.info(f"  Batches: {len(self.dataloader)}")

    @torch.no_grad()
    def run_inference(self) -> None:
        logger.info(f"\n{'=' * 60}")
        logger.info("Running inference on validation set...")
        logger.info(f"{'=' * 60}")

        total_batches = len(self.dataloader)
        for batch_idx, batch_data in enumerate(self.dataloader):
            vis = batch_data["vis"].to(self.device)
            ir = batch_data["ir"].to(self.device)
            labels = batch_data["labels"]
            batch_size = vis.shape[0]

            t0 = time.perf_counter()
            with torch.amp.autocast(device_type=self.device.type):
                output = self.model.model._predict_dual({"vis": vis, "ir": ir})
            t1 = time.perf_counter()
            self.inference_times.append((t1 - t0) / batch_size)

            detections = self._process_predictions(output, batch_size)
            t2 = time.perf_counter()
            self.nms_times.append((t2 - t1) / batch_size)

            ground_truths = self._process_labels(labels, batch_size)

            self.total_detections += sum(d["bboxes"].shape[0] for d in detections)
            self.total_gt_objects += sum(g["bboxes"].shape[0] for g in ground_truths)

            self.all_preds.extend(detections)
            self.all_targets.extend(ground_truths)

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                logger.info(
                    f"  [{batch_idx + 1:4d}/{total_batches}] "
                    f"inf={self.inference_times[-1] * 1000:5.1f}ms/img  "
                    f"nms={self.nms_times[-1] * 1000:5.1f}ms/img  "
                    f"total_dets={self.total_detections}"
                )

        avg_inf = sum(self.inference_times) / len(self.inference_times) * 1000
        avg_nms = sum(self.nms_times) / len(self.nms_times) * 1000
        logger.info(f"\n  Average inference: {avg_inf:.1f} ms/image")
        logger.info(f"  Average NMS:       {avg_nms:.1f} ms/image")
        logger.info(f"  Total detections:  {self.total_detections}")
        logger.info(f"  Total GT objects:  {self.total_gt_objects}")

    def _process_predictions(
        self, raw_output: torch.Tensor, batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        if isinstance(raw_output, (tuple, list)):
            raw_output = raw_output[0]

        if raw_output.dim() == 3 and raw_output.shape[-1] >= 6 + self.nc:
            decoded = raw_output
        else:
            decoded = raw_output.permute(0, 2, 1) if raw_output.dim() == 3 else raw_output

        nc_dim = self.nc
        results = []
        for b in range(batch_size):
            img_pred = decoded[b]
            boxes = img_pred[:, :4]
            scores = img_pred[:, 4 : 4 + nc_dim]
            angle = img_pred[:, 4 + nc_dim : 5 + nc_dim]

            scores_max, cls_pred = scores.max(dim=1, keepdim=True)
            mask = scores_max.squeeze(-1) > self.args.conf
            if not mask.any():
                results.append({
                    "bboxes": torch.zeros(0, 5, device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "cls": torch.zeros(0, dtype=torch.long, device=self.device),
                })
                continue

            boxes = boxes[mask]
            scores_val = scores_max[mask]
            cls_val = cls_pred[mask]
            angle_val = angle[mask]

            keep = self._obb_nms(boxes, scores_val.squeeze(-1), angle_val.squeeze(-1))
            if keep.shape[0] == 0:
                results.append({
                    "bboxes": torch.zeros(0, 5, device=self.device),
                    "scores": torch.zeros(0, device=self.device),
                    "cls": torch.zeros(0, dtype=torch.long, device=self.device),
                })
                continue

            boxes_kept = boxes[keep]
            scores_kept = scores_val[keep].squeeze(-1)
            cls_kept = cls_val[keep].squeeze(-1)
            angle_kept = angle_val[keep].squeeze(-1)

            bboxes_5d = torch.cat([boxes_kept, angle_kept.unsqueeze(-1)], dim=-1)
            results.append({"bboxes": bboxes_5d, "scores": scores_kept, "cls": cls_kept})

        return results

    def _obb_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        order = scores.argsort(descending=True)
        keep = []

        xywhr_boxes = torch.cat([boxes, angles.unsqueeze(-1)], dim=-1)

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i.item())

            iou = batch_probiou(
                xywhr_boxes[i].unsqueeze(0),
                xywhr_boxes[order[1:]],
            )
            mask = iou <= self.args.iou
            order = order[1:][mask.view(-1)]

        return torch.tensor(keep, dtype=torch.long, device=self.device)

    def _process_labels(
        self, labels: torch.Tensor, batch_size: int
    ) -> list[dict[str, torch.Tensor]]:
        results = []
        for b in range(batch_size):
            mask = labels[:, 0].long() == b
            img_labels = labels[mask]
            if img_labels.shape[0] == 0:
                results.append({
                    "bboxes": torch.zeros(0, 5, device=labels.device),
                    "cls": torch.zeros(0, dtype=torch.long, device=labels.device),
                })
            else:
                results.append({
                    "bboxes": img_labels[:, 2:7],
                    "cls": img_labels[:, 1].long(),
                })
        return results

    def compute_metrics(self) -> dict[str, Any]:
        logger.info(f"\n{'=' * 60}")
        logger.info("Computing evaluation metrics...")
        logger.info(f"{'=' * 60}")

        niou = 10
        iouv = torch.linspace(0.5, 0.95, niou)

        stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}

        for img_idx, (pred, target) in enumerate(zip(self.all_preds, self.all_targets)):
            pred_cls = pred["cls"].cpu().numpy()
            pred_bboxes = pred["bboxes"].cpu()
            pred_scores = pred["scores"].cpu().numpy()

            gt_cls = target["cls"].cpu().numpy()
            gt_bboxes = target["bboxes"].cpu()

            nl = pred_cls.shape[0]
            nr = gt_cls.shape[0]

            if nl == 0:
                if nr > 0:
                    for gt_c in gt_cls:
                        stats["target_cls"].append(int(gt_c))
                        stats["target_img"].append(img_idx)
                continue

            if nr == 0:
                tp = np.zeros((nl, niou), dtype=bool)
            else:
                iou = batch_probiou(gt_bboxes, pred_bboxes).cpu().numpy()
                tp = self._match_predictions(pred_cls, gt_cls, iou, iouv.numpy())

            for i in range(nl):
                stats["tp"].append(tp[i])
                stats["conf"].append(pred_scores[i])
                stats["pred_cls"].append(float(pred_cls[i]))
            for gt_c in gt_cls:
                stats["target_cls"].append(int(gt_c))
                stats["target_img"].append(img_idx)

        if not stats["tp"]:
            logger.warning("No predictions to evaluate!")
            return {"overall": {}, "per_class": {}, "confusion_matrix": {}}

        tp_array = np.array(stats["tp"], dtype=bool)
        conf_array = np.array(stats["conf"], dtype=np.float64)
        pred_cls_array = np.array(stats["pred_cls"], dtype=np.float64)
        target_cls_array = np.array(stats["target_cls"], dtype=np.int64)

        results = ap_per_class(
            tp_array, conf_array, pred_cls_array, target_cls_array,
            plot=False, save_dir=self.save_dir, names=self.names, prefix="OBB",
        )

        overall = {}
        per_class = {}
        if results:
            tp_out, fp, p_all, r_all, f1_all, ap_all, ap_class, p_curve, r_curve, f1_curve, x, prec_values = results
            ap50_all = ap_all[:, 0] if ap_all is not None and ap_all.size > 0 else np.array([])
            overall["metrics/mAP50(B)"] = float(np.mean(ap50_all)) if ap50_all.size > 0 else 0.0
            overall["metrics/mAP50-95(B)"] = float(np.mean(ap_all)) if ap_all.size > 0 else 0.0
            overall["metrics/precision(B)"] = float(np.mean(p_all)) if p_all.size > 0 else 0.0
            overall["metrics/recall(B)"] = float(np.mean(r_all)) if r_all.size > 0 else 0.0

            self._log_metrics(overall)

            per_class = self._build_per_class(
                ap50_all, ap_all, p_all, r_all, f1_all, ap_class
            )
        else:
            overall = {}
            per_class = {}
            self._log_metrics(overall)

        confusion = self._build_confusion_matrix()

        return {
            "overall": overall,
            "per_class": per_class,
            "confusion_matrix": confusion,
            "inference_ms": sum(self.inference_times) / max(len(self.inference_times), 1) * 1000,
            "nms_ms": sum(self.nms_times) / max(len(self.nms_times), 1) * 1000,
            "total_images": self.total_images,
            "total_gt_objects": self.total_gt_objects,
            "total_detections": self.total_detections,
        }

    @staticmethod
    def _match_predictions(
        pred_cls: np.ndarray,
        gt_cls: np.ndarray,
        iou: np.ndarray,
        iouv: np.ndarray,
    ) -> np.ndarray:
        nl = pred_cls.shape[0]
        correct = np.zeros((nl, len(iouv)), dtype=bool)
        if gt_cls.shape[0] == 0:
            return correct

        detected = np.zeros(gt_cls.shape[0], dtype=bool)
        for i, cls_val in enumerate(pred_cls):
            cls_mask = gt_cls == cls_val
            if not cls_mask.any():
                continue
            overlaps = iou[:, i]
            cls_overlaps = overlaps.copy()
            cls_overlaps[~cls_mask] = 0.0
            best_j = cls_overlaps.argmax()
            if cls_overlaps[best_j] > 0 and not detected[best_j]:
                for k, thresh in enumerate(iouv):
                    if overlaps[best_j] >= thresh:
                        correct[i, k] = True
                detected[best_j] = True
        return correct

    @staticmethod
    def _log_metrics(results: dict) -> None:
        key_map = {
            "metrics/mAP50(B)": "mAP@50       ",
            "metrics/mAP50-95(B)": "mAP@50:95    ",
            "metrics/precision(B)": "Precision    ",
            "metrics/recall(B)": "Recall       ",
        }
        logger.info(f"\n{'─' * 50}")
        logger.info(f"{'Metric':<20s} {'Value':>10s}")
        logger.info(f"{'─' * 50}")
        for key, label in key_map.items():
            if key in results:
                logger.info(f"{label:<20s} {results[key]:10.4f}")
        logger.info(f"{'─' * 50}")

    def _build_per_class(
        self,
        ap50_all: np.ndarray | None,
        ap_all: np.ndarray | None,
        p_all: np.ndarray | None,
        r_all: np.ndarray | None,
        f1_all: np.ndarray | None,
        ap_class: np.ndarray | None,
    ) -> dict[int, dict[str, float]]:
        per_class = {}
        if ap_all is None or ap_class is None or len(ap_class) == 0:
            for cls_idx in self.names:
                per_class[cls_idx] = {"mAP@50": 0.0, "mAP@50:95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            return per_class

        logger.info(f"\n{'=' * 80}")
        logger.info("Per-Class Performance")
        logger.info(f"{'=' * 80}")
        header = f"{'Class':>16s}  {'mAP@50':>8s}  {'mAP@50:95':>10s}  {'Prec':>8s}  {'Recall':>8s}  {'F1':>8s}"
        logger.info(header)
        logger.info(f"{'─' * 80}")

        seen_classes = set()
        for i in range(len(ap_class)):
            cls_idx = int(ap_class[i])
            seen_classes.add(cls_idx)
            cls_name = self.names.get(cls_idx, f"class_{cls_idx}")
            vals = {
                "mAP@50": float(ap50_all[i]),
                "mAP@50:95": float(np.mean(ap_all[i])) if ap_all is not None else 0.0,
                "precision": float(p_all[i]) if p_all is not None else 0.0,
                "recall": float(r_all[i]) if r_all is not None else 0.0,
                "f1": float(f1_all[i]) if f1_all is not None else 0.0,
            }
            per_class[cls_idx] = vals
            logger.info(
                f"{cls_name:>16s}  {vals['mAP@50']:8.4f}  {vals['mAP@50:95']:10.4f}  "
                f"{vals['precision']:8.4f}  {vals['recall']:8.4f}  {vals['f1']:8.4f}"
            )

        for cls_idx in self.names:
            if cls_idx not in seen_classes:
                vals = {"mAP@50": 0.0, "mAP@50:95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
                per_class[cls_idx] = vals
                cls_name = self.names[cls_idx]
                logger.info(
                    f"{cls_name:>16s}  {vals['mAP@50']:8.4f}  {vals['mAP@50:95']:10.4f}  "
                    f"{vals['precision']:8.4f}  {vals['recall']:8.4f}  {vals['f1']:8.4f}"
                )

        logger.info(f"{'─' * 80}")
        return per_class

    def _build_confusion_matrix(self) -> dict:
        matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)

        for pred, target in zip(self.all_preds, self.all_targets):
            gt_cls = target["cls"].cpu().numpy()
            gt_bboxes = target["bboxes"].cpu().numpy()
            pred_cls = pred["cls"].cpu().numpy()
            pred_bboxes = pred["bboxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()

            if pred_cls.shape[0] == 0:
                for gt_c in gt_cls:
                    matrix[int(gt_c), self.nc] += 1
                continue

            iou = batch_probiou(
                torch.from_numpy(gt_bboxes),
                torch.from_numpy(pred_bboxes),
            ).cpu().numpy()
            iou_shape = iou.shape

            matched_gt = set()
            for i in np.argsort(-pred_scores):
                if iou_shape[0] == 0 or iou_shape[1] == 0:
                    matrix[int(pred_cls[i]), self.nc] += 1
                    continue
                cls_mask = gt_cls == pred_cls[i]
                if not cls_mask.any():
                    matrix[int(pred_cls[i]), self.nc] += 1
                    continue
                overlaps = iou[:, i].copy()
                overlaps[~cls_mask] = 0.0
                best_j = overlaps.argmax()
                if overlaps[best_j] > self.args.iou and best_j not in matched_gt:
                    matrix[int(pred_cls[i]), int(gt_cls[best_j])] += 1
                    matched_gt.add(best_j)
                else:
                    matrix[int(pred_cls[i]), self.nc] += 1

            for j, gt_c in enumerate(gt_cls):
                if j not in matched_gt:
                    matrix[self.nc, int(gt_c)] += 1

        return {
            "matrix": matrix.tolist(),
            "class_names": [self.names.get(i, f"class_{i}") for i in range(self.nc)] + ["background"],
        }

    def save_results(self, metrics_data: dict[str, Any]) -> dict[str, Path]:
        saved_files: dict[str, Path] = {}

        if self.args.save_json:
            json_path = self.save_dir / "evaluation_results.json"
            serializable = self._make_serializable(metrics_data)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
            saved_files["json"] = json_path
            logger.info(f"  JSON results saved to: {json_path}")

        return saved_files

    @staticmethod
    def _make_serializable(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: DualStreamOBBEvaluator._make_serializable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [DualStreamOBBEvaluator._make_serializable(v) for v in data]
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return data.tolist()
        if isinstance(data, (np.floating,)):
            return float(data)
        if isinstance(data, (np.integer,)):
            return int(data)
        return data

    def visualize(self, metrics_data: dict[str, Any]) -> dict[str, Path]:
        if not self.args.visualize:
            return {}
        if not HAS_MPL:
            logger.warning("matplotlib not available, skipping visualization")
            return {}
        saved_files: dict[str, Path] = {}

        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["axes.labelsize"] = 11

        fig_confusion = self._plot_confusion_matrix(metrics_data.get("confusion_matrix", {}))
        if fig_confusion:
            path = self.save_dir / "confusion_matrix.png"
            fig_confusion.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig_confusion)
            saved_files["confusion_matrix"] = path
            logger.info(f"  Confusion matrix saved to: {path}")

        fig_per_class = self._plot_per_class_metrics(metrics_data.get("per_class", {}))
        if fig_per_class:
            path = self.save_dir / "per_class_metrics.png"
            fig_per_class.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig_per_class)
            saved_files["per_class_metrics"] = path
            logger.info(f"  Per-class metrics saved to: {path}")

        fig_summary = self._plot_summary(metrics_data.get("overall", {}))
        if fig_summary:
            path = self.save_dir / "summary_metrics.png"
            fig_summary.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig_summary)
            saved_files["summary_metrics"] = path
            logger.info(f"  Summary metrics saved to: {path}")

        return saved_files

    def _plot_confusion_matrix(self, cm_data: dict) -> Any:
        matrix = np.array(cm_data.get("matrix", []))
        if matrix.size == 0:
            return None
        class_names = cm_data.get("class_names", [str(i) for i in range(self.nc)] + ["BG"])

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_matrix = matrix.astype(np.float64) / row_sums

        fig, ax = plt.subplots(figsize=(max(8, self.nc * 1.5), max(7, self.nc * 1.3)))
        im = ax.imshow(norm_matrix, cmap="Blues", aspect="auto")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                count = matrix[i, j]
                pct = norm_matrix[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{count}", ha="center", va="center", fontsize=8, color=color)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Ground Truth", fontweight="bold")
        ax.set_title("Confusion Matrix (Normalized)", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        return fig

    def _plot_per_class_metrics(self, per_class: dict[int, dict[str, float]]) -> Any:
        if not per_class:
            return None

        class_indices = sorted(per_class.keys())
        class_labels = [self.names.get(i, f"cls_{i}") for i in class_indices]

        metrics_list = ["mAP@50", "mAP@50:95", "precision", "recall", "f1"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for ax_idx, metric_key in enumerate(metrics_list):
            ax = axes[ax_idx]
            values = [per_class[i].get(metric_key, 0.0) for i in class_indices]
            colors = plt.colormaps.get_cmap("viridis")(
                np.linspace(0.15, 0.85, len(class_indices))
            )
            bars = ax.bar(range(len(class_indices)), values, color=colors, edgecolor="gray", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )
            ax.set_xticks(range(len(class_indices)))
            ax.set_xticklabels(class_labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(metric_key)
            ax.set_title(f"Per-Class {metric_key}", fontweight="bold")
            ax.set_ylim(0, max(1.0, max(values) * 1.2))
            ax.grid(axis="y", alpha=0.3)

        axes[-1].axis("off")
        fig.tight_layout()
        return fig

    def _plot_summary(self, overall: dict) -> Any:
        if not overall:
            return None

        key_map = {
            "metrics/mAP50(B)": "mAP@50",
            "metrics/mAP50-95(B)": "mAP@50:95",
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)": "Recall",
        }
        display = {label: overall.get(key, 0.0) for key, label in key_map.items() if key in overall}

        if not display:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(display.keys())
        values = list(display.values())
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][: len(labels)]
        bars = ax.barh(labels, values, color=colors, edgecolor="gray", linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

        ax.set_xlim(0, max(1.0, max(values) * 1.2))
        ax.set_xlabel("Value", fontweight="bold")
        ax.set_title("Overall Evaluation Metrics", fontweight="bold", fontsize=14)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Dual-Stream YOLOv8 OBB Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Dataset:  {args.data}")
    logger.info(f"Conf: {args.conf}, IoU: {args.iou}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    evaluator = DualStreamOBBEvaluator(args)

    try:
        evaluator.load_model()
        evaluator.prepare_data()
        evaluator.run_inference()
        metrics_data = evaluator.compute_metrics()
        saved_files = evaluator.save_results(metrics_data)
        viz_files = evaluator.visualize(metrics_data)

        logger.info(f"\n{'=' * 60}")
        logger.info("Evaluation complete!")
        logger.info(f"{'=' * 60}")
        logger.info(f"Results saved to: {evaluator.save_dir}")
        all_files = {**saved_files, **viz_files}
        for label, path in all_files.items():
            logger.info(f"  {label}: {path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()