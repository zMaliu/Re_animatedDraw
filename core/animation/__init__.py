# -*- coding: utf-8 -*-
"""
动态绘制与动画生成模块

实现论文中的动态绘制与动画生成：
1. 笔触方向确定
2. 动画渲染
3. 绘制速度控制
4. 视觉效果优化
"""

from .stroke_animator import StrokeAnimator
from .direction_detector import DirectionDetector
from .flood_renderer import FloodRenderer
from .animation_controller import AnimationController
from .painting_animator import PaintingAnimator, AnimationStyle, PaintingSpeed, AnimationConfig, AnimationFrame, AnimationSequence
from .brush_simulator import BrushSimulator, BrushType, BlendMode, BrushProperties, BrushStroke, BrushState
from .animation_renderer import AnimationRenderer, RenderFormat, RenderQuality, RenderConfig, RenderProgress, RenderResult
from .timeline_manager import TimelineManager, TimelineEvent, TimelineEventType, PlaybackState, TimelineMarker, TimelineLayer, PlaybackConfig, TimelineState

__all__ = [
    'StrokeAnimator',
    'DirectionDetector', 
    'FloodRenderer',
    'AnimationController',
    'PaintingAnimator',
    'AnimationStyle',
    'PaintingSpeed',
    'AnimationConfig',
    'AnimationFrame',
    'AnimationSequence',
    'BrushSimulator',
    'BrushType',
    'BlendMode',
    'BrushProperties',
    'BrushStroke',
    'BrushState',
    'AnimationRenderer',
    'RenderFormat',
    'RenderQuality',
    'RenderConfig',
    'RenderProgress',
    'RenderResult',
    'TimelineManager',
    'TimelineEvent',
    'TimelineEventType',
    'PlaybackState',
    'TimelineMarker',
    'TimelineLayer',
    'PlaybackConfig',
    'TimelineState'
]