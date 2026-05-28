#!/usr/bin/env python
# Ultralytics YOLOv8 Dual-Stream Inference Script
# For Visible + Infrared (RGB-T) Object Detection

"""
Inference script for dual-stream YOLOv8 models.

This script demonstrates how to run inference with a dual-stream YOLOv8 model
on paired visible and infrared images.

Usage:
    python predict_dualstream.py --weights best.pt --source-vis vis.jpg --source-ir ir.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics.nn.modules.dualstream_model import DualStreamYOLO
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.plotting import Annotator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dual-stream YOLOv8 inference")
    
    # Model arguments
    parser.add_argument("--weights", type=str, default="yolov8-dualstream.pt",
                       help="Path to model weights")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size")
    parser.add_argument("--device", type=str, default="0",
                       help="Device (e.g., '0', 'cpu')")
    
    # Source arguments
    parser.add_argument("--source-vis", type=str, required=True,
                       help="Path to visible image or directory")
    parser.add_argument("--source-ir", type=str, required=True,
                       help="Path to infrared image or directory")
    
    # Inference arguments
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300,
                       help="Maximum detections per image")
    parser.add_argument("--agnostic-nms", action="store_true",
                       help="Class-agnostic NMS")
    
    # Output arguments
    parser.add_argument("--show", action="store_true",
                       help="Display results")
    parser.add_argument("--save-txt", action="store_true",
                       help="Save results to text file")
    parser.add_argument("--save-img", action="store_true",
                       help="Save result images")
    parser.add_argument("--project", type=str, default="runs/dualstream-predict",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="exp",
                       help="Save results to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                       help="Overwrite existing results")
    
    return parser.parse_args()


def preprocess_image(img_path: str, imgsz: int):
    """Preprocess image for inference.
    
    Args:
        img_path: Path to image.
        imgsz: Image size.
    
    Returns:
        Preprocessed tensor and original image.
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Store original shape
    orig_shape = img.shape[:2]
    
    # Resize with letterbox
    ratio = min(imgsz / img.shape[0], imgsz / img.shape[1])
    new_unpad = int(img.shape[1] * ratio), int(img.shape[0] * ratio)
    
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    dh = (imgsz - new_unpad[1]) / 2
    dw = (imgsz - new_unpad[0]) / 2
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    
    return img, orig_shape, (ratio, (left, top))


def draw_detections(img, boxes, names, conf_thresh=0.25):
    """Draw detections on image.
    
    Args:
        img: Image to draw on.
        boxes: Detection boxes (xyxy, conf, cls).
        names: Class names.
        conf_thresh: Confidence threshold.
    
    Returns:
        Image with detections.
    """
    annotator = Annotator(img, line_width=2, example=str(names))
    
    for *xyxy, conf, cls in boxes:
        if conf >= conf_thresh:
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=(0, 255, 0))
    
    return annotator.result()


@torch.no_grad()
def predict(args):
    """Run prediction on image pair(s).
    
    Args:
        args: Command-line arguments.
    """
    LOGGER.info(f"\n{colorstr('green', 'Starting dual-stream inference')}")
    LOGGER.info(f"Model: {args.weights}")
    LOGGER.info(f"Device: {args.device}")
    
    # Set device
    device = args.device if args.device else "0"
    if torch.cuda.is_available():
        device = f"cuda:{device}" if device != "cpu" else "cpu"
    else:
        device = "cpu"
    
    LOGGER.info(f"Using device: {device}")
    
    # Load model
    LOGGER.info(f"Loading model from {args.weights}")
    model = DualStreamYOLO(args.weights, verbose=False)
    model.to(device)
    model.eval()
    
    # Prepare sources
    source_vis = Path(args.source_vis)
    source_ir = Path(args.source_ir)
    
    if source_vis.is_dir():
        # Directory of images
        vis_files = sorted(source_vis.glob("*.jpg")) + sorted(source_vis.glob("*.png"))
        ir_files = sorted(source_ir.glob("*.jpg")) + sorted(source_ir.glob("*.png"))
    else:
        # Single image
        vis_files = [source_vis]
        ir_files = [source_ir]
    
    if len(vis_files) != len(ir_files):
        raise ValueError(f"Mismatched files: {len(vis_files)} visible vs {len(ir_files)} infrared")
    
    LOGGER.info(f"Found {len(vis_files)} image pairs")
    
    # Create output directory
    save_dir = Path(args.project) / args.name
    if args.save_img:
        save_dir.mkdir(parents=True, exist_ok=args.exist_ok)
    
    # Run inference
    for idx, (vis_path, ir_path) in enumerate(zip(vis_files, ir_files)):
        LOGGER.info(f"\nProcessing pair {idx + 1}/{len(vis_files)}")
        LOGGER.info(f"  Visible: {vis_path}")
        LOGGER.info(f"  Infrared: {ir_path}")
        
        # Preprocess
        try:
            img_vis, orig_shape_vis, pad_vis = preprocess_image(str(vis_path), args.imgsz)
            img_ir, orig_shape_ir, pad_ir = preprocess_image(str(ir_path), args.imgsz)
        except Exception as e:
            LOGGER.warning(f"Failed to load images: {e}")
            continue
        
        # Move to device
        img_vis = img_vis.to(device)
        img_ir = img_ir.to(device)
        
        # Inference
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
            preds = model.predict(img_vis, img_ir)
        
        # Get predictions
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Apply NMS
        from ultralytics.utils.ops import non_max_suppression
        preds = non_max_suppression(
            preds,
            args.conf,
            args.iou,
            agnostic=args.agnostic_nms,
            max_det=args.max_det,
        )
        
        # Process results
        pred = preds[0] if len(preds) > 0 else torch.zeros(0, 6)
        
        # Scale boxes back to original size
        ratio, (left, top) = pad_vis
        if len(pred) > 0:
            # Remove padding
            pred[:, 0] -= left
            pred[:, 1] -= top
            pred[:, 2] -= left
            pred[:, 3] -= top
            
            # Scale to original size
            pred[:, :4] /= ratio
            
            # Clip to image bounds
            pred[:, [0, 2]] = pred[:, [0, 2]].clamp(0, orig_shape_vis[1])
            pred[:, [1, 3]] = pred[:, [1, 3]].clamp(0, orig_shape_vis[0])
        
        LOGGER.info(f"  Detected {len(pred)} objects")
        
        # Load original image for visualization
        orig_img_vis = cv2.imread(str(vis_path))
        orig_img_ir = cv2.imread(str(ir_path))
        
        # Draw detections on visible image
        names = model.model.names if hasattr(model.model, "names") else {}
        result_img = draw_detections(orig_img_vis.copy(), pred, names, args.conf)
        
        # Save or display results
        if args.show:
            cv2.imshow("Visible - Detections", result_img)
            cv2.imshow("Infrared", orig_img_ir)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if args.save_img:
            save_path = save_dir / f"{vis_path.stem}_result.jpg"
            cv2.imwrite(str(save_path), result_img)
            LOGGER.info(f"  Saved result to {save_path}")
        
        # Save detections to text file
        if args.save_txt:
            txt_path = save_dir / f"{vis_path.stem}_detections.txt"
            with open(txt_path, "w") as f:
                for *xyxy, conf, cls in pred:
                    f.write(f"{int(cls)} {conf:.4f} {xyxy[0]:.2f} {xyxy[1]:.2f} {xyxy[2]:.2f} {xyxy[3]:.2f}\n")
            LOGGER.info(f"  Saved detections to {txt_path}")
    
    LOGGER.info(f"\n{colorstr('green', 'Inference completed!')}")


def main():
    """Main function."""
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    main()
