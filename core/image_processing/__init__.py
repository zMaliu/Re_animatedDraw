# -*- coding: utf-8 -*-
"""
图像处理模块

提供图像预处理、分析和增强功能，包括：
1. 图像预处理和增强
2. 傅里叶描述子形状分析
3. 基于三分法则的显著性计算
"""

from .image_processor import ImageProcessor
from .fourier_descriptor import FourierDescriptor
from .saliency_calculator import SaliencyCalculator

__all__ = [
    'ImageProcessor',
    'FourierDescriptor', 
    'SaliencyCalculator'
]