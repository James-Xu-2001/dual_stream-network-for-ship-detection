# Ultralytics YOLOv8 Dual-Stream Modules
# Modules for Visible + Infrared (RGB-T) Fusion

"""Dual-Stream fusion modules for multispectral object detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, autopad

__all__ = (
    "DualStreamFusion",
    "CrossModalityAttention",
    "FeatureFusionBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
)


class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration.
    
    This module learns channel-wise relationships to enhance important features
    and suppress less useful ones.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """Initialize channel attention module.
        
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck layer.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input features."""
        b, c, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important spatial regions.
    
    This module learns spatial relationships to highlight important regions
    in the feature map.
    """
    
    def __init__(self, kernel_size: int = 7):
        """Initialize spatial attention module.
        
        Args:
            kernel_size: Kernel size for the convolution layer.
        """
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to input features."""
        # Average pool along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pool along channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve to produce attention map
        attn_input = torch.cat([avg_out, max_out], dim=1)
        attn_map = self.sigmoid(self.conv1(attn_input))
        
        # Apply attention map to original features
        return x * attn_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Combines channel and spatial attention to refine features.
    Proposed in: https://arxiv.org/abs/1807.06521
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """Initialize CBAM module.
        
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for channel attention.
            kernel_size: Kernel size for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM to input features."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DualStreamFusion(nn.Module):
    """Dual-stream feature fusion module for RGB-T (Visible-Infrared) data.
    
    This module fuses features from visible and infrared streams using
    concatenation followed by convolution and optional attention mechanisms.
    
    Attributes:
        mode: Fusion mode ('concat', 'add', 'max', 'attention')
        attention: Whether to apply CBAM attention after fusion
        channels: Output channels after fusion
    """
    
    def __init__(
        self,
        channels: int,
        mode: str = "concat",
        attention: bool = False,
        reduction: int = 16,
    ):
        """Initialize dual-stream fusion module.
        
        Args:
            channels: Number of input channels per stream.
            mode: Fusion mode - 'concat' (default), 'add', 'max', or 'attention'.
            attention: Whether to apply CBAM attention after fusion.
            reduction: Reduction ratio for attention modules.
        """
        super().__init__()
        self.mode = mode
        
        if mode == "concat":
            # Concat doubles the channels, so we need to reduce back
            self.fusion_conv = Conv(channels * 2, channels, k=3, p=1)
        elif mode == "add":
            # Addition keeps same channels
            self.fusion_conv = Conv(channels, channels, k=3, p=1)
        elif mode == "max":
            self.fusion_conv = Conv(channels, channels, k=3, p=1)
        elif mode == "attention":
            # Learnable weighted fusion
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.fusion_conv = Conv(channels, channels, k=3, p=1)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")
        
        self.attention = CBAM(channels, reduction) if attention else nn.Identity()
    
    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        """Fuse visible and infrared features.
        
        Args:
            x_vis: Visible stream features (B, C, H, W).
            x_ir: Infrared stream features (B, C, H, W).
        
        Returns:
            Fused features (B, C, H, W).
        """
        if self.mode == "concat":
            x = torch.cat([x_vis, x_ir], dim=1)
            x = self.fusion_conv(x)
        elif self.mode == "add":
            x = x_vis + x_ir
            x = self.fusion_conv(x)
        elif self.mode == "max":
            x = torch.max(x_vis, x_ir)
            x = self.fusion_conv(x)
        elif self.mode == "attention":
            # Learnable weighted sum
            w = F.softmax(self.weights, dim=0)
            x = w[0] * x_vis + w[1] * x_ir
            x = self.fusion_conv(x)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        
        # Apply attention if enabled
        x = self.attention(x)
        
        return x


class FeatureFusionBlock(nn.Module):
 
    def __init__(
        self,
        c1: int,
        c2: int,
        fusion_mode: str = "concat",
        attention: bool = True,
    ):

        super().__init__()
        
        # Fusion module
        self.fusion = DualStreamFusion(c1, mode=fusion_mode, attention=attention)
        
        # Feature refinement
        self.conv = C2f(c1, c2, n=1, shortcut=True) if c1 == c2 else Conv(c1, c2, k=3, p=1)
    
    def forward(self, x):
        """Forward pass accepting either a list [vis, ir] or two separate tensors.

        List format is used during manual traversal in _predict_dual.
        """
        if isinstance(x, list):
            x_vis, x_ir = x[0], x[1]
        else:
            x_vis = x_ir = x  # fallback for single-input usage

        x = self.fusion(x_vis, x_ir)
        x = self.conv(x)
        return x


class CrossModalityAttention(nn.Module):
    
    def __init__(self, channels: int, num_heads: int = 8):
        """Initialize cross-modality attention.
        
        Args:
            channels: Number of input channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Query, Key, Value projections for visible stream
        self.q_vis = nn.Linear(channels, channels)
        self.k_vis = nn.Linear(channels, channels)
        self.v_vis = nn.Linear(channels, channels)
        
        # Query, Key, Value projections for infrared stream
        self.q_ir = nn.Linear(channels, channels)
        self.k_ir = nn.Linear(channels, channels)
        self.v_ir = nn.Linear(channels, channels)
        
        # Output projection
        self.proj = nn.Linear(channels, channels)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x_vis: torch.Tensor, x_ir: torch.Tensor) -> torch.Tensor:
        """Apply cross-modality attention.
        
        Args:
            x_vis: Visible features (B, C, H, W).
            x_ir: Infrared features (B, C, H, W).
        
        Returns:
            Fused features (B, C, H, W).
        """
        B, C, H, W = x_vis.shape
        
        # Flatten(2)代表从第二个维度开始拉平flatten.transpose(1,2)代表交换维度1和维度2。即(B, C, H, W) -> (B, H*W, C)
        x_vis_flat = x_vis.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_ir_flat = x_ir.flatten(2).transpose(1, 2)    # (B, H*W, C)
        
        # Project to Q, K, V
        q_vis = self.q_vis(x_vis_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, H*W, C) -> (B, num_heads, H*W, head_dim)
        k_ir = self.k_ir(x_ir_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_ir = self.v_ir(x_ir_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算query和key的点积相似度 | transpose(-2, -1)是将倒数第一个维度和倒数第二个维度的顺序调换
        attn = (q_vis @ k_ir.transpose(-2, -1)) * self.scale
        # 得到（B, num_heads, H*W_vis, H*W_ir）
        attn = attn.softmax(dim=-1)
        
        # 得到（B, num_heads, H*W_vis, head_dim）再经过 transpose(1, 2)和reshape(B, -1, C) 得到了（B, H*W_vis, C）
        out = (attn @ v_ir).transpose(1, 2).reshape(B, -1, C)
        
        # Project back
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection
        out = out + x_vis
        
        return out


# Import C2f for FeatureFusionBlock
# Avoid circular import by importing here
from .block import C2f
