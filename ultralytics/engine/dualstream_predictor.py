# Ultralytics YOLOv8 Dual-Stream Predictor
# For Visible + Infrared (RGB-T) Object Detection

"""Dual-stream predictor for multispectral object detection."""

from __future__ import annotations

import cv2
import torch
from pathlib import Path
from typing import List, Optional, Union

from ultralytics.engine.predictor import BasePredictor, DetectionPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.torch_utils import select_device


class DualStreamPredictor(DetectionPredictor):
    """Dual-stream predictor for RGB-T (Visible-Infrared) object detection.
    
    This predictor handles inference with paired visible and infrared images,
    processing them through the dual-stream YOLOv8 model.
    
    Example:
        ```python
        from ultralytics.nn.modules.dualstream_model import DualStreamYOLO
        
        model = DualStreamYOLO("yolov8-dualstream.pt")
        predictor = DualStreamPredictor(overrides={"model": "yolov8-dualstream.pt"})
        
        # Inference with image pairs
        results = predictor.predict(
            source_vis="visible_image.jpg",
            source_ir="infrared_image.jpg"
        )
        ```
    """
    
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialize DualStreamPredictor.
        
        Args:
            cfg: Configuration dictionary.
            overrides: Configuration overrides.
            _callbacks: Callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        
        # Override args for dual-stream
        self.args.task = "detect"
    
    def preprocess(self, im: Union[dict, torch.Tensor]) -> dict:
        """Preprocess dual-stream input.
        
        Args:
            im: Input dictionary with 'vis' and 'ir' keys or tensor.
        
        Returns:
            Preprocessed input dictionary.
        """
        if isinstance(im, dict):
            # Dual-stream input
            im_vis = im.get("vis")
            im_ir = im.get("ir")
            
            if im_vis is None or im_ir is None:
                raise ValueError("Dual-stream input requires both 'vis' and 'ir' keys")
            
            # Preprocess visible image
            if not isinstance(im_vis, torch.Tensor):
                im_vis = self._preprocess_single(im_vis)
            
            # Preprocess infrared image
            if not isinstance(im_ir, torch.Tensor):
                im_ir = self._preprocess_single(im_ir)
            
            # Move to device
            im_vis = im_vis.to(self.device)
            im_ir = im_ir.to(self.device)
            
            # Convert to fp16 if needed
            if self.model.fp16:
                im_vis = im_vis.half()
                im_ir = im_ir.half()
            else:
                im_vis = im_vis.float()
                im_ir = im_ir.float()
            
            return {"vis": im_vis, "ir": im_ir}
        
        elif isinstance(im, torch.Tensor):
            # Single tensor input (for compatibility)
            if not isinstance(im, torch.Tensor):
                im = self._preprocess_single(im)
            im = im.to(self.device)
            if self.model.fp16:
                im = im.half()
            else:
                im = im.float()
            return im
        
        else:
            raise ValueError(f"Unsupported input type: {type(im)}")
    
    def _preprocess_single(self, im: Union[str, Path, List, torch.Tensor]) -> torch.Tensor:
        """Preprocess a single image stream.
        
        Args:
            im: Input image (path, list, or tensor).
        
        Returns:
            Preprocessed tensor.
        """
        if isinstance(im, (str, Path)):
            im = cv2.imread(str(im))
            if im is None:
                raise FileNotFoundError(f"Image not found: {im}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.transforms(im)  # Apply transforms
            im = torch.from_numpy(im).permute(2, 0, 1)
        elif isinstance(im, list):
            im = [self._preprocess_single(x) for x in im]
            im = torch.stack(im)
        elif isinstance(im, torch.Tensor):
            pass  # Already tensor
        else:
            raise ValueError(f"Unsupported input type: {type(im)}")
        
        # Normalize
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        
        # Add batch dimension
        if im.ndim == 3:
            im = im.unsqueeze(0)
        
        return im
    
    def preprocess_image(self, im_path_vis: Union[str, Path], im_path_ir: Union[str, Path]) -> dict:
        """Preprocess a pair of visible and infrared images.
        
        Args:
            im_path_vis: Path to visible image.
            im_path_ir: Path to infrared image.
        
        Returns:
            Preprocessed dictionary with 'vis' and 'ir' tensors.
        """
        # Load images
        im_vis = cv2.imread(str(im_path_vis))
        im_ir = cv2.imread(str(im_path_ir))
        
        if im_vis is None:
            raise FileNotFoundError(f"Visible image not found: {im_path_vis}")
        if im_ir is None:
            raise FileNotFoundError(f"Infrared image not found: {im_path_ir}")
        
        # Convert BGR to RGB
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
        im_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        im_vis = self.transforms(im_vis)
        im_ir = self.transforms(im_ir)
        
        # Convert to tensors
        im_vis = torch.from_numpy(im_vis).permute(2, 0, 1).unsqueeze(0)
        im_ir = torch.from_numpy(im_ir).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        im_vis = im_vis.float() / 255.0
        im_ir = im_ir.float() / 255.0
        
        return {"vis": im_vis, "ir": im_ir}
    
    def pre_transform(self, im: Union[str, Path, List]) -> torch.Tensor:
        """Pre-transform input images.
        
        Args:
            im: Input images.
        
        Returns:
            Transformed tensor.
        """
        # For dual-stream, this is handled in preprocess
        return super().pre_transform(im)
    
    def inference(self, im: dict, *args, **kwargs) -> torch.Tensor:
        """Run inference on dual-stream input.
        
        Args:
            im: Preprocessed input dictionary with 'vis' and 'ir' tensors.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Detection predictions.
        """
        # Visualize features if requested
        if self.args.visualize:
            self.model.visualize = True
        
        # Run forward pass
        with torch.amp.autocast("cuda", enabled=self.model.fp16):
            preds = self.model(im, augment=self.args.augment)
        
        return preds
    
    def postprocess(self, preds: torch.Tensor, img: torch.Tensor, orig_imgs: list) -> list:
        """Post-process predictions.
        
        Args:
            preds: Raw predictions from model.
            img: Processed input images.
            orig_imgs: Original input images.
        
        Returns:
            Post-processed results.
        """
        # Apply NMS
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )
        
        # Process predictions for each image
        results = []
        for i, pred in enumerate(preds):
            # Get original image
            orig_img = orig_imgs[i] if i < len(orig_imgs) else None
            
            # Scale boxes to original image size
            if orig_img is not None:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            
            # Create Results object
            result = Results(orig_img, path="", names=self.model.names)
            result.boxes = pred
            results.append(result)
        
        return results
    
    def predict(self, source_vis: Union[str, Path, List], source_ir: Union[str, Path, List], **kwargs):
        """Predict with dual-stream inputs.
        
        Args:
            source_vis: Visible image source (path or list of paths).
            source_ir: Infrared image source (path or list of paths).
            **kwargs: Additional keyword arguments.
        
        Returns:
            List of Results objects.
        """
        # Handle single image pair
        if isinstance(source_vis, (str, Path)):
            source_vis = [source_vis]
            source_ir = [source_ir]
        
        # Ensure equal lengths
        if len(source_vis) != len(source_ir):
            raise ValueError("Visible and infrared sources must have equal length")
        
        results = []
        for vis_path, ir_path in zip(source_vis, source_ir):
            # Preprocess
            im = self.preprocess_image(vis_path, ir_path)
            
            # Run inference
            preds = self.inference(im)
            
            # Load original images for postprocessing
            orig_img_vis = cv2.imread(str(vis_path))
            orig_img_ir = cv2.imread(str(ir_path))
            
            # Postprocess
            result = self.postprocess(preds, im["vis"], [orig_img_vis])
            results.extend(result)
        
        return results


class DualStreamInference:
    """Simple dual-stream inference class for easy usage.
    
    This class provides a simplified interface for running inference
    with dual-stream YOLOv8 models.
    
    Example:
        ```python
        from ultralytics.data.dualstream_dataset import DualStreamInference
        
        # Initialize
        inference = DualStreamInference("yolov8-dualstream.pt", device="cuda")
        
        # Run inference
        results = inference.predict(
            vis_image="visible.jpg",
            ir_image="infrared.jpg"
        )
        
        # Display results
        inference.show_results(results)
        ```
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ):
        """Initialize DualStreamInference.
        
        Args:
            model_path: Path to model weights.
            device: Device to run inference on.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Image size.
        """
        # Import here to avoid circular imports
        from ultralytics.nn.modules.dualstream_model import DualStreamYOLO
        
        # Load model
        self.model = DualStreamYOLO(model_path)
        
        # Set device
        if device == "auto":
            self.device = select_device()
        else:
            self.device = select_device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set parameters
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        
        # Build transforms
        from ultralytics.data.augment import LetterBox
        self.letterbox = LetterBox(imgsz)
    
    def preprocess(self, img_vis: str, img_ir: str) -> dict:
        """Preprocess image pair.
        
        Args:
            img_vis: Path to visible image.
            img_ir: Path to infrared image.
        
        Returns:
            Preprocessed tensors.
        """
        # Load images
        im_vis = cv2.imread(img_vis)
        im_ir = cv2.imread(img_ir)
        
        # Resize with letterbox
        im_vis = self.letterbox(im_vis)
        im_ir = self.letterbox(im_ir)
        
        # BGR to RGB
        im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
        im_ir = cv2.cvtColor(im_ir, cv2.COLOR_BGR2RGB)
        
        # Convert to tensors
        im_vis = torch.from_numpy(im_vis).permute(2, 0, 1).unsqueeze(0)
        im_ir = torch.from_numpy(im_ir).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        im_vis = im_vis.float() / 255.0
        im_ir = im_ir.float() / 255.0
        
        # Move to device
        im_vis = im_vis.to(self.device)
        im_ir = im_ir.to(self.device)
        
        return {"vis": im_vis, "ir": im_ir}
    
    @torch.no_grad()
    def predict(self, vis_image: str, ir_image: str) -> list:
        """Run prediction on image pair.
        
        Args:
            vis_image: Path to visible image.
            ir_image: Path to infrared image.
        
        Returns:
            Detection results.
        """
        # Preprocess
        im = self.preprocess(vis_image, ir_image)
        
        # Inference
        with torch.amp.autocast("cuda", enabled=self.model.model.fp16):
            preds = self.model.predict(im["vis"], im["ir"])
        
        # Postprocess
        preds = ops.non_max_suppression(
            preds[0] if isinstance(preds, (list, tuple)) else preds,
            self.conf,
            self.iou,
        )
        
        # Scale boxes
        orig_shape = cv2.imread(vis_image).shape[:2]
        preds[0][:, :4] = ops.scale_boxes(self.imgsz, preds[0][:, :4], orig_shape)
        
        return preds
    
    def show_results(self, preds: list, image_path: str):
        """Display detection results on image.
        
        Args:
            preds: Detection predictions.
            image_path: Path to image for visualization.
        """
        # Load image
        img = cv2.imread(image_path)
        
        # Draw boxes
        for *xyxy, conf, cls in preds[0]:
            label = f"{self.model.model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
