# Ultralytics YOLOv8 Dual-Stream Dataset Loader for Rotated Bounding Boxes
# For Visible + Infrared (RGB-T) Object Detection with OBB

"""Dual-stream dataset loader for multispectral object detection with rotated bounding boxes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.augment import Compose, v8_transforms
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xyxyxyxy2xywhr


class DualStreamOBBDataset(Dataset):
    """Dual-stream dataset for RGB-T (Visible-Infrared) object detection with rotated bounding boxes.
    
    This dataset loads paired visible and infrared images along with their
    rotated bounding box annotations for training dual-stream YOLOv8 models.
    
    Expected directory structure:
    
Attributes:
    img_path_vis: List of paths to visible images.
    img_path_ir: List of paths to infrared images.
    labels: List of label dictionaries.
    transforms: Image transformations.
"""

    def __init__(
        self,
        img_path: Union[str, Path],
        mode: str = "train",
        imgsz: int = 640,
        augment: bool = False,
        hyp: Optional[dict] = None,
        rect: bool = False,
        cache: bool = False,
        single_cls: bool = False,
        stride: int = 32,
        pad: tuple = (0.5, 0.5),
        prefix: str = "",
        use_segments: bool = False,
        use_keypoints: bool = False,
        overlap_mask: bool = True,
        mask_ratio: int = 4,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        flipud: float = 0.0,
        fliplr: float = 0.5,
        bgr: float = 0.0,
        mosaic: float = 1.0,
        mixup: float = 0.0,
        copy_paste: float = 0.0,
        auto_augment: str = "randaugment",
        erasing: float = 0.4,
        crop_fraction: float = 1.0,
        copy_paste_mode: str = "flip",
    ):
        """Initialize DualStreamOBDataset.
        
        Args:
            img_path: Root directory path containing visible/ and infrared/ subdirectories.
            imgsz: Target image size.
            augment: Whether to apply data augmentation.
            hyp: Dictionary of hyperparameters.
            rect: Whether to use rectangular training.
            cache: Whether to cache images in memory.
            single_cls: Whether to train with single class.
            stride: Model stride.
            pad: Padding for batched images.
            prefix: Prefix for logging.
            use_segments: Whether to use segmentation masks.
            use_keypoints: Whether to use keypoints.
            overlap_mask: Whether to overlap masks.
            mask_ratio: Mask downsample ratio.
            hsv_h: HSV hue augmentation.
            hsv_s: HSV saturation augmentation.
            hsv_v: HSV value augmentation.
            degrees: Rotation augmentation range.
            translate: Translation augmentation range.
            scale: Scale augmentation range.
            shear: Shear augmentation range.
            perspective: Perspective augmentation range.
            flipud: Vertical flip probability.
            fliplr: Horizontal flip probability.
            bgr: BGR conversion probability.
            mosaic: Mosaic augmentation probability.
            mixup: Mixup augmentation probability.
            copy_paste: Copy-paste augmentation probability.
            auto_augment: Auto augmentation policy.
            erasing: Random erasing probability.
            crop_fraction: Crop fraction for classification.
        """
        self.img_path = Path(img_path)
        self.mode = mode
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.rect = rect
        self.stride = stride
        self.pad = pad
        

        # Load image paths and labels
        # 1. 加载可见图像路径和红外图像路径
        # 2. 加载旋转边界框标签
        self.img_paths_vis, self.img_paths_ir, self.labels = self._load_dual_streams()
        
        # Cache images if requested
        self.cache = cache
        if self.cache:
            self._cache_images()
        
        # Build transforms
        # LetterBoxOBB now properly handles OBB 8-point coordinates
        self.transforms = self._build_transforms(
            hsv_h=hyp.hsv_h if hasattr(hyp, 'hsv_h') else 0.015,
            hsv_s=hyp.hsv_s if hasattr(hyp, 'hsv_s') else 0.7,
            hsv_v=hyp.hsv_v if hasattr(hyp, 'hsv_v') else 0.4,
        )
        
        LOGGER.info(
            f"{prefix}DualStreamOBDataset loaded {len(self)} images from {self.img_path}"
        )
    # 1. 加载可见光图像路径和红外光图像路径
    # 2. 加载旋转边界框标签
    def _load_dual_streams(self) -> Tuple[List[Path], List[Path], List[dict]]:
        """Load paired visible and infrared image paths and labels for OBB.
        
        Returns:
            Tuple of (visible_paths, infrared_paths, labels).
        """
        img_paths_vis = []
        img_paths_ir = []
        labels = []
        
        split_dir = self.mode
        
        # Visible images directory
        vis_dir = self.img_path / "visible" / split_dir
        ir_dir = self.img_path / "infrared" / split_dir
        labels_dir = self.img_path / "label" / split_dir
        
        if not vis_dir.exists():
            raise FileNotFoundError(f"Visible images directory not found: {vis_dir}")
        if not ir_dir.exists():
            raise FileNotFoundError(f"Infrared images directory not found: {ir_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Get all visible image files
        vis_files = sorted(
            [f for f in vis_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        
        # Match with infrared and labels
        for vis_file in vis_files:
            img_name = vis_file.stem
            
            # Find corresponding infrared image
            ir_file = ir_dir / vis_file.name
            
            if not ir_file.exists():
                LOGGER.warning(f"Infrared image not found for {vis_file}, skipping")
                continue
            
            # Find corresponding label file
            label_file = labels_dir / f"{img_name}.txt"
            if not label_file.exists():
                LOGGER.warning(f"Label file not found for {vis_file}, skipping")
                continue
            
            # Load OBB labels
            label_dict = self._load_obb_label(label_file)
            
            img_paths_vis.append(vis_file)
            img_paths_ir.append(ir_file)
            labels.append(label_dict)
        
        return img_paths_vis, img_paths_ir, labels
    # 通过文件路径，加载旋转边界框标签
    def _load_obb_label(self, label_path: Path) -> dict:
        """Load YOLO OBB format label file for rotated bounding boxes.
        
        OBB format: class_idx x1 y1 x2 y2 x3 y3 x4 y4
        Coordinates are normalized to [0, 1] range.
        
        Args:
            label_path: Path to label text file.
        
        Returns:
            Dictionary with class IDs and rotated bounding boxes.
        """
        labels = {
            "cls": [],           # Class IDs
            "obboxes": [],       # Rotated bounding boxes (8 coordinates per box)
        }
        
        try:
            with open(label_path, "r") as f:
                for line in f:
                    # 移除行首和行尾的空格和换行符
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # 按空格字符分割字符串
                    parts = line.split()
                    if len(parts) >= 9:  # Need at least 9 values: class + 8 coordinates
                        cls_id = int(parts[0])
                        # Parse 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
                        corners = [float(x) for x in parts[1:9]]
                        
                        # Validate coordinate range (should be normalized [0, 1])
                        if not all(0 <= c <= 1 for c in corners):
                            LOGGER.warning(f"OBB coordinates out of range [0,1] in {label_path}: {corners}")
                            continue
                        
                        labels["cls"].append(cls_id)
                        labels["obboxes"].append(corners)
                    else:
                        LOGGER.warning(f"Invalid OBB format in {label_path}: expected 9 values, got {len(parts)}")
        except Exception as e:
            LOGGER.warning(f"Error loading label {label_path}: {e}")
        
        # Convert to numpy arrays
        if labels["cls"]:
            labels["cls"] = np.array(labels["cls"], dtype=np.int64)
            labels["obboxes"] = np.array(labels["obboxes"], dtype=np.float32)  # Shape: (N, 8)
        else:
            labels["cls"] = np.array([], dtype=np.int64)
            labels["obboxes"] = np.array([], dtype=np.float32).reshape(0, 8)  # Shape: (0, 8)
        
        return labels
    # 1.构建数据增强变换
    # 2.颜色空间增强（HSV）
    # 3.LetterBox 进行resize（训练和验证都需要）
    def _build_transforms(self, hyp: Optional[dict] = None, **kwargs) -> Compose:

        if hyp is None:
            from types import SimpleNamespace
            #命名空间对象，用于存储超参数
            # **kwargs: 所有额外参数
            hyp = SimpleNamespace(**kwargs)
         # 从ultralytics导入所需的增强类

        from ultralytics.data.augment import (
            RandomHSV, LetterBoxOBB, Compose
        )
        
        # 构建基础变换管道
        transforms = []
        
        # 训练时添加数据增强
        if self.augment:
            # 颜色空间增强（HSV）
            if any([hyp.hsv_h, hyp.hsv_s, hyp.hsv_v]):
                transforms.append(RandomHSV(
                    hgain=hyp.hsv_h,
                    sgain=hyp.hsv_s,
                    vgain=hyp.hsv_v
                ))

        # 添加 LetterBoxOBB 进行 resize（专门为 OBB 设计的 resize）
        transforms.append(LetterBoxOBB(new_shape=(self.imgsz, self.imgsz)))

        # Compose 就像一个管道，数据依次流过每个变换函数，每个变换对数据进行修改后传给下一个。
        return Compose(transforms)
    # 通过文件路径，加载可见图像和红外图像到内存 分别是images_vis和images_ir
    def _cache_images(self):
        """Cache images in memory for faster training."""
        LOGGER.info(f"{self.prefix}Caching images...")
        
        self.images_vis = []
        self.images_ir = []
        
        for i, (vis_path, ir_path) in enumerate(zip(self.img_paths_vis, self.img_paths_ir)):
            try:
                img_vis = cv2.imread(str(vis_path))
                img_ir = cv2.imread(str(ir_path))
                
                if img_vis is not None and img_ir is not None:
                    self.images_vis.append(img_vis)
                    self.images_ir.append(img_ir)
                else:
                    LOGGER.warning(f"Failed to read images at index {i}")
                    self.images_vis.append(None)
                    self.images_ir.append(None)
            except Exception as e:
                LOGGER.warning(f"Error caching images at index {i}: {e}")
                self.images_vis.append(None)
                self.images_ir.append(None)
    # 获取数据集大小
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.img_paths_vis)
    # 获取数据集中的一个样本
        # 1. 从可见图像路径和红外图像路径中获取可见图像和红外图像或者从cache中获取
        # 2. 从标签中获取可见图像和红外图像的旋转边界框标签
        # 3. 数据增强
        # 4. 结果以字典形式返回{'vis': visible_image, 'ir': ir_image, 'instances': instances}
    def __getitem__(self, index: int) -> dict:
        """Get item from dataset.
        
        Args:
            index: Item index.
        
        Returns:
            Dictionary with 'vis', 'ir', and 'instances' keys.
        """
        # Get cached or load image
        if self.cache:
            img_vis = self.images_vis[index]
            img_ir = self.images_ir[index]
        else:
            img_vis = cv2.imread(str(self.img_paths_vis[index]))
            img_ir = cv2.imread(str(self.img_paths_ir[index]))
        
        # Handle missing images
        if img_vis is None or img_ir is None:
            LOGGER.warning(f"Image loading failed at index {index}")
            if img_vis is None:
                LOGGER.warning(f"  - Visible image is None: {self.img_paths_vis[index]}")
            if img_ir is None:
                LOGGER.warning(f"  - IR image is None: {self.img_paths_ir[index]}")
            return self._get_dummy_item()
        
        # Get labels (deep copy to avoid modifying original data)
        import copy
        
        # OBB coordinates are stored as normalized [0,1], convert to pixel coordinates for transforms
        img_h, img_w = img_vis.shape[:2]
        instances = {
            "cls": self.labels[index]["cls"].copy(),
            "obboxes": self.labels[index]["obboxes"].copy()  # Normalized [0,1], Shape: (N, 8)
        }
        
        # Convert to pixel coordinates (only if there are boxes)
        if len(instances["obboxes"]) > 0:
            instances["obboxes"][:, 0::2] *= img_w  # x coordinates
            instances["obboxes"][:, 1::2] *= img_h  # y coordinates
        
        # Apply transforms to visible image and OBB labels
        if self.transforms:
            # LetterBoxOBB expects "img" key and "instances" with "obboxes" in pixel coordinates
            transformed = self.transforms(
                {
                    "img": img_vis,
                    "instances": instances,
                }
            )
            img_vis = transformed["img"]
            instances = transformed["instances"]  # Updated obboxes (pixel coords in new image)
            
            # Apply same transforms to IR image with same parameters
            transformed_ir = self.transforms(
                {
                    "img": img_ir,
                    "instances": instances.copy(),  # Use same obboxes
                }
            )
            img_ir = transformed_ir["img"]
        
        # 1.Convert to tensors 2.HWC → CHW 维度置换 3.归一化
        img_vis = torch.from_numpy(img_vis).permute(2, 0, 1).float() / 255.0
        img_ir = torch.from_numpy(img_ir).permute(2, 0, 1).float() / 255.0
        
        # Handle single class 创建一个与给定数组形状和数据类型完全相同的新数组，但所有元素都填充为 0。
        if self.single_cls:
            instances["cls"] = np.zeros_like(instances["cls"])
        
        # Convert OBB format from 8-point (xyxyxyxy) to xywhr format for loss computation
        # This conversion is required because YOLOv8 OBB loss expects xywhr format
        if len(instances["obboxes"]) > 0:
            # instances["obboxes"] is in 8-point format: [x1,y1,x2,y2,x3,y3,x4,y4] (pixel coordinates after LetterBox)
            # Convert to xywhr format: [cx, cy, w, h, angle] (pixel coordinates after LetterBox)
            obboxes_8pt = torch.from_numpy(instances["obboxes"])
            obboxes_xywhr = xyxyxyxy2xywhr(obboxes_8pt).numpy()  # Shape: (N, 5) in pixel coords
            
            # Normalize xywhr to [0, 1] for loss computation
            # Use imgsz (LetterBox output size) for normalization, not original image size
            # After LetterBox, image size is always (imgsz, imgsz)
            obboxes_xywhr[:, [0, 2]] /= self.imgsz  # cx, w
            obboxes_xywhr[:, [1, 3]] /= self.imgsz  # cy, h
            # angle is already in radians, no normalization needed
            
            instances["obboxes"] = obboxes_xywhr  # Shape: (N, 5) normalized
        else:
            instances["obboxes"] = np.zeros((0, 5), dtype=np.float32)  # Empty array with correct shape
        
        return {
            "vis": img_vis,
            "ir": img_ir,
            "instances": instances,
            # "img_path_vis": str(self.img_paths_vis[index]),
            # "img_path_ir": str(self.img_paths_ir[index]),
        }
    # 创建一个虚拟样本，用于错误处理
    def _get_dummy_item(self) -> dict:
        """Get dummy item for error handling."""
        dummy_img = torch.zeros(3, self.imgsz, self.imgsz)
        return {
            "vis": dummy_img,
            "ir": dummy_img,
            "instances": {"cls": np.array([]), "obboxes": np.array([])},
            # "img_path_vis": "",
            # "img_path_ir": "",
        }


class DualStreamOBBDataLoader:
    """DataLoader wrapper for dual-stream OBB datasets.
    This class provides a convenient interface for creating DataLoaders
    for dual-stream training and validation with rotated bounding boxes.
    """
    # 允许 DualStreamOBBDataLoader 接受任意额外的参数并将这些参数原封不动地传递给 DualStreamOBBDataset
    def __init__(
        self,
        img_path: Union[str, Path],
        mode: str = "train",
        batch_size: int = 16,
        imgsz: int = 640,
        augment: bool = False,
        num_workers: int = 8,
        **kwargs,
    ):
        """Initialize DualStreamOBDataLoader.
        Args:
            img_path: Dataset root directory.
            batch_size: Batch size.
            imgsz: Image size.
            augment: Whether to augment.
            num_workers: Number of data loading workers.
            **kwargs: Additional arguments for DualStreamOBBDataset.
        """
        self.dataset = DualStreamOBBDataset(
            img_path=img_path,
            mode=mode,
            imgsz=imgsz,
            augment=augment,
            **kwargs,
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=augment,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # 负责将一个 batch 中所有样本的标签合并成一个统一的张量。
    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Custom collate function for dual-stream OBB batches.
        
        Args:
            batch: List of sample dictionaries.
        
        Returns:
            Collated batch dictionary.
        """
        vis_imgs = torch.stack([item["vis"] for item in batch])
        ir_imgs = torch.stack([item["ir"] for item in batch])
        
        # Collate OBB labels
        labels = []
        for i, item in enumerate(batch):
            instances = item["instances"]
            cls = instances["cls"]
            obboxes = instances["obboxes"]
            
            if len(cls) > 0 and len(obboxes) > 0:
                # Get class IDs and OBB coordinates
                # Shape: (N,) and (N, 5) for xywhr format
                # Add batch index to each target
                batch_idx = np.full(len(cls), i, dtype=np.int64)
                # Combine: [batch_idx, cls, cx, cy, w, h, angle]
                combined = np.column_stack([batch_idx, cls, obboxes])
                labels.append(combined)
        
        if labels:
            # Concatenate all labels
            labels = torch.from_numpy(np.concatenate(labels, axis=0))
        else:
            # Empty batch: [batch_idx, cls, 5 coordinates (xywhr)] = 7 columns
            labels = torch.zeros(0, 7)
        
        return {
            "vis": vis_imgs,
            "ir": ir_imgs,
            "labels": labels,  # Shape: (total_objects, 7)
        }

    def __iter__(self):
        """Iterate over dataloader."""
        return iter(self.dataloader)

    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)