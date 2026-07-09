# Ultralytics YOLOv8 Dual-Stream Detection Model
# For Visible + Infrared (RGB-T) Object Detection

"""Dual-Stream YOLOv8 model for multispectral object detection."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch
import torch.nn as nn

from ultralytics.nn.tasks import BaseModel, DetectionModel, parse_model
from ultralytics.utils import LOGGER, YAML


class DualStreamDetectionModel(DetectionModel):
    
    def __init__(
        self,
        cfg: Union[dict, str] = "yolov8-dualstream.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
    ):
        from ultralytics.nn.tasks import BaseModel, parse_model, yaml_model_load
        from ultralytics.utils import LOGGER
        from ultralytics.utils.torch_utils import initialize_weights
        import torch
        
        # Initialize nn.Module
        torch.nn.Module.__init__(self)
        
        # Load model configuration (use yaml_model_load to handle scale inference)
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        
        # Override channels and classes
        if nc is not None and nc != self.yaml.get("nc"):
            LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml["nc"] = nc
        self.yaml["channels"] = ch
        
        # Build model with IR stream starting index
        # IR stream starts at layer 10 in the YAML configuration
        self.model, self.save = parse_model(self.yaml, ch=ch, verbose=verbose, ir_start_idx=10)
        
        self.inplace = self.yaml.get("inplace", True)
        
        # Set stride manually for dual-stream (skip the problematic stride check)
        # Standard YOLO strides: P3=8, P4=16, P5=32
        self.stride = torch.tensor([8, 16, 32])
        
        # Also set stride on the OBB detection head layer.
        # v8OBBLoss and Detect.bias_init() read stride from model.model[-1].stride.
        if hasattr(self.model[-1], "stride"):
            self.model[-1].stride = torch.tensor([8, 16, 32])
        
        # Initialize weights
        initialize_weights(self)
        # 用于根据数据集的类别数或锚框尺寸，对偏置项进行特殊初始化
        if hasattr(self.model[-1], "bias_init"):
            self.model[-1].bias_init()
        
        # Store input channels for each stream
        self.ch_vis = ch
        self.ch_ir = ch
        
        # Add args attribute for loss functions compatibility
        # This is required by v8DetectionLoss and v8OBBLoss
        self.args = SimpleNamespace(
            overlap_mask=True,  # Default value for OBB loss
            reg_max=16,  # DFL regression max channels
            box=1.5,  # box loss gain (ProbIoU for OBB) — moderate for from-scratch
            cls=0.5,  # cls loss gain (BCE)
            dfl=1.5,  # dfl loss gain (Distribution Focal Loss)
            kobj=1.0,  # keypoint objectness loss
            label_smoothing=0.0,  # label smoothing
            angle=0.1,  # angle loss gain for OBB (cosine similarity)
        )
        
        if verbose:
            # 调用模型的 info() 方法，该方法通常会打印模型的网络结构摘要，用于调试或查看模型架构。
            self.info()
            LOGGER.info("")
        
        LOGGER.info(
            f"DualStreamDetectionModel initialized with "
            f"RGB channels={self.ch_vis}, IR channels={self.ch_ir}, "
            f"classes={self.yaml['nc']}, stride={self.stride.tolist()}"
        )
        self.criterion = None
    
    def forward(self, x: dict):
        return self._forward_dual(x)
    
    def _forward_dual(
        self,
        x: dict,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
    ):
        
        x_vis = x.get("vis")
        x_ir = x.get("ir")
        
        if x_vis is None or x_ir is None:
            raise ValueError(
                "Dual-stream input requires both 'vis' and 'ir' keys in the input dictionary"
            )
        
        # Check if we're in training mode by looking for labels in the input dict
        # Training input format: {"vis": tensor, "ir": tensor, "labels": labels}
        if "labels" in x:
            # Training mode - compute loss
            return self.loss(x)
        
        # Inference mode - forward pass
        return self._predict_dual(
            {"vis": x_vis, "ir": x_ir},
            profile=profile,
            visualize=visualize,
            augment=augment,
        )
    
    def _predict_dual(
        self,
        x: dict,
        profile: bool = False,
        visualize: bool = False,
        augment: bool = False,
    ):
        x_vis = x["vis"]
        x_ir = x["ir"]
        
        # 统一存储所有特征（索引唯一，不会冲突）
        y = {}
        current_output = None  # 跟踪上一层的输出
        
        # 定义各流的第一层索引（基于 YAML 配置）
        VIS_FIRST_LAYER = 0
        IR_FIRST_LAYER = 10
        
        for i, m in enumerate(self.model):
            # 获取输入
            if m.f == -1:
                # 特殊处理：第一层需要区分使用哪个输入流
                if i == VIS_FIRST_LAYER:
                    x_in = x_vis
                elif i == IR_FIRST_LAYER:
                    x_in = x_ir
                else:
                    # 其他层的 -1 表示使用上一层的输出
                    x_in = current_output
            else:
                # 从统一存储中获取指定层的输出
                if isinstance(m.f, int):
                    x_in = y.get(m.f)
                else:
                    # 多输入
                    x_in = [current_output if j == -1 else y.get(j) for j in m.f]
                    # 即使只有一个输入，也保持列表格式（Concat 等模块需要列表）
                    if not x_in:
                        x_in = None
            
            # 前向传播
            if x_in is not None:
                out = m(x_in)
                y[m.i] = out
                current_output = out  # 更新当前输出
            
            
            # 可视化
                if visualize:
                    from ultralytics.utils.plotting import feature_visualization
                    feature_visualization(out, m.type, m.i, save_dir=visualize)
            else:
                out = None
        
        return out
    
    def loss(
        self, 
        x: dict,
    ):
        
        
        x_vis = x.get("vis")
        x_ir = x.get("ir")
        labels = x.get("labels")  # Shape: (total_objects, 10)
        
        # Forward pass to get predictions
        preds = self._predict_dual({"vis": x_vis, "ir": x_ir})
        
        # Convert labels tensor to batch dict format expected by loss functions
        # labels format: [batch_idx, cls, cx, cy, w, h, angle]
        if labels.shape[0] > 0:
            # For OBB detection, labels are already in xywhr format from dataset
            batch = {
                "batch_idx": labels[:, 0].long(),  # Shape: (N,)
                "cls": labels[:, 1].long(),        # Shape: (N,)
                "bboxes": labels[:, 2:7].float(),  # OBB format: (N, 5) - xywhr normalized
            }
            
            # Debug: Print bbox statistics to verify normalization
            if not hasattr(self, '_debug_printed'):
                bboxes_np = batch["bboxes"].cpu().numpy()
                LOGGER.info(
                    f"DEBUG - Bbox stats: "
                    f"cx=[{bboxes_np[:, 0].min():.3f}, {bboxes_np[:, 0].max():.3f}], "
                    f"cy=[{bboxes_np[:, 1].min():.3f}, {bboxes_np[:, 1].max():.3f}], "
                    f"w=[{bboxes_np[:, 2].min():.3f}, {bboxes_np[:, 2].max():.3f}], "
                    f"h=[{bboxes_np[:, 3].min():.3f}, {bboxes_np[:, 3].max():.3f}]"
                    f"angle=[{bboxes_np[:, 4].min():.3f}, {bboxes_np[:, 4].max():.3f}]"
                )
                self._debug_printed = True
        else:
            batch = {
                "batch_idx": torch.zeros(0, dtype=torch.long, device=labels.device),
                "cls": torch.zeros(0, dtype=torch.long, device=labels.device),
                "bboxes": torch.zeros(0, 5, dtype=torch.float, device=labels.device),
            }
        
        # Compute loss using OBB loss function
        # v8OBBLoss returns a tuple: (loss_components * batch_size, loss_components_detached)
        # loss_components shape: [4] containing [box_loss, cls_loss, dfl_loss, angle_loss]
        from ultralytics.utils.loss import v8OBBLoss
        raw_preds = preds[1] if isinstance(preds, tuple) else preds
        if self.criterion is None or self.criterion.device != raw_preds["boxes"].device:
            self.criterion = v8OBBLoss(self)
        loss_output = self.criterion(preds, batch)
        loss_components_scaled, loss_components = loss_output  # Shape: [4] - [box, cls, dfl, angle]
        
        # Extract individual loss components for logging
        # These are detached tensors for monitoring (no gradients)
        box_loss = loss_components[0].detach()
        cls_loss = loss_components[1].detach()
        dfl_loss = loss_components[2].detach()
        angle_loss = loss_components[3].detach()
        
        self._loss_items = torch.stack([box_loss, cls_loss, dfl_loss, angle_loss])
        loss = loss_components_scaled.sum()
        
        return loss
    
    def get_loss_items(self):
        
        return self._loss_items


class DualStreamYOLO(nn.Module):
    """YOLO wrapper for dual-stream models.
    
    This class provides a high-level interface for dual-stream YOLO models,
    handling model loading, configuration, and inference.
    模型加载，配置，推理
    """
    
    def __init__(
        self,
        model: str = "yolov8-dualstream.yaml",
        verbose: bool = False,
    ):
       
        super().__init__()
        # Load configuration
        model_path = Path(model)
        if model_path.suffix == ".pt":
            # PyTorch加载训练好的模型权重文件
            self.model = self._load_weights(model_path)
        elif model_path.suffix in (".yaml", ".yml"):
            # Load from configuration
            self.model = DualStreamDetectionModel(cfg=model_path, verbose=verbose)
        else:
            raise ValueError(f"Unsupported model file type: {model_path.suffix}")
        
        self.verbose = verbose
    
    # 加载模型权重的pt文件
    def _load_weights(self, weights_path: Path) -> DualStreamDetectionModel:
       
        import torch

        # Load checkpoint
        ckpt = torch.load(weights_path, map_location="cpu")
        
        # Create model from configuration
        cfg = ckpt.get("train_args", {}).get("model")
        if cfg is None:
            from pathlib import Path as _Path
            fallback = _Path(__file__).resolve().parents[2] / "cfg" / "models" / "v8" / "yolov8-dualstream.yaml"
            cfg = str(fallback) if fallback.exists() else None
        if cfg is None:
            raise ValueError(
                "Cannot determine model architecture: checkpoint has no 'model' config path. "
                "Please re-save with 'model' key in train_config or provide a .yaml file."
            )
        model = DualStreamDetectionModel(cfg=cfg)
        
        # Load weights
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        
        return model
    
    def forward(self, x: dict | torch.Tensor):
        """Forward pass through the model."""
        return self.model(x)
    
    def predict(self, x_vis: torch.Tensor, x_ir: torch.Tensor):
        """Perform prediction on visible and infrared inputs.
        
        Args:
            x_vis: Visible image tensor (B, C, H, W).
            x_ir: Infrared image tensor (B, C, H, W).
        
        Returns:
            Detection predictions.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model({"vis": x_vis, "ir": x_ir})
    
    def train(self, mode: bool = True):
        """Set model to training/evaluation mode."""
        self.model.train(mode)
        return self
    
    def get_loss_items(self):
        """Get detached loss components for logging.
        
        Returns:
            torch.Tensor: Stack of [box_loss, cls_loss, dfl_loss, angle_loss]
        """
        if hasattr(self.model, 'get_loss_items'):
            return self.model.get_loss_items()
        return None
