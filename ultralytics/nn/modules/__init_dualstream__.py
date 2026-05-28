# Ultralytics YOLOv8 Dual-Stream Modules
# 双流模块导出文件

"""
Dual-stream modules for RGB-T (Visible-Infrared) object detection.

This module provides all necessary components for dual-stream object detection
using YOLOv8 architecture.

Example usage:
    ```python
    from ultralytics.nn.modules.dualstream import (
        DualStreamFusion,
        CrossModalityAttention,
        CBAM,
    )
    
    from ultralytics.nn.modules.dualstream_model import (
        DualStreamYOLO,
        DualStreamDetectionModel,
    )
    
    from ultralytics.data.dualstream_dataset import (
        DualStreamDataset,
        DualStreamDataLoader,
    )
    
    from ultralytics.engine.dualstream_predictor import (
        DualStreamPredictor,
        DualStreamInference,
    )
    ```
"""

from .dualstream import (
    DualStreamFusion,
    CrossModalityAttention,
    FeatureFusionBlock,
    ChannelAttention,
    SpatialAttention,
    CBAM,
)

from .dualstream_model import (
    DualStreamYOLO,
    DualStreamDetectionModel,
)
# 列表明确指定可以被外部访问的类和函数
__all__ = [
    # Fusion modules
    "DualStreamFusion",
    "CrossModalityAttention",
    "FeatureFusionBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    
    # Models
    "DualStreamYOLO",
    "DualStreamDetectionModel",
]
