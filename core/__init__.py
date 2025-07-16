# -*- coding: utf-8 -*-
"""
核心算法模块

包含中国画动画构建的核心算法实现
"""

__version__ = '1.0.0'
__author__ = 'AI Assistant'

# 导入主要模块
from .stroke_extraction import StrokeDetector
from .stroke_database import StrokeDatabase
from .stroke_ordering import StrokeOrganizer
from .animation import PaintingAnimator

__all__ = [
    'StrokeDetector',
    'StrokeDatabase', 
    'StrokeOrganizer',
    'PaintingAnimator'
]