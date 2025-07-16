# -*- coding: utf-8 -*-
"""
笔触建模模块

实现论文中的笔触特征提取和建模功能
包括几何特征、墨色特征、位置特征的提取
"""

from .feature_extractor import StrokeFeatureExtractor
from .edge_detector import CannyEdgeDetector
from .corner_detector import HarrisCornerDetector
from .interactive_tool import InteractiveStrokeTool

__all__ = [
    'StrokeFeatureExtractor',
    'CannyEdgeDetector', 
    'HarrisCornerDetector',
    'InteractiveStrokeTool'
]