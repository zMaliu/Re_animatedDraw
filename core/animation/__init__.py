# -*- coding: utf-8 -*-
"""
动画生成模块

用于生成中国画笔刷绘制动画
包括：
- 动画生成器：PaintingAnimator
- 笔刷模拟器：BrushSimulator
- 动画渲染器：AnimationRenderer
- 时间轴管理：TimelineManager
"""

from .painting_animator import PaintingAnimator
from .brush_simulator import BrushSimulator
from .animation_renderer import AnimationRenderer
from .timeline_manager import TimelineManager

__all__ = [
    'PaintingAnimator',
    'BrushSimulator', 
    'AnimationRenderer',
    'TimelineManager'
]