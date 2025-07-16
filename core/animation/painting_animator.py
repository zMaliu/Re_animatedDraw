# -*- coding: utf-8 -*-
"""
绘画动画器模块

提供绘画动画的生成和控制功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time
from enum import Enum
from ..stroke_extraction.stroke_detector import Stroke
from .stroke_animator import StrokeAnimator, AnimationMode
from .direction_detector import DirectionDetector
from .animation_controller import AnimationController


class AnimationStyle(Enum):
    """动画风格"""
    REALISTIC = "realistic"  # 写实风格
    ARTISTIC = "artistic"  # 艺术风格
    SKETCH = "sketch"  # 素描风格
    CALLIGRAPHY = "calligraphy"  # 书法风格
    WATERCOLOR = "watercolor"  # 水彩风格


class PaintingSpeed(Enum):
    """绘画速度"""
    VERY_SLOW = 0.2
    SLOW = 0.5
    NORMAL = 1.0
    FAST = 2.0
    VERY_FAST = 5.0


@dataclass
class AnimationConfig:
    """动画配置"""
    fps: int = 30
    duration: float = 10.0
    resolution: Tuple[int, int] = (1920, 1080)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    style: AnimationStyle = AnimationStyle.REALISTIC
    speed: PaintingSpeed = PaintingSpeed.NORMAL
    show_brush: bool = True
    show_progress: bool = True
    enable_effects: bool = True
    smooth_transitions: bool = True


@dataclass
class AnimationFrame:
    """动画帧"""
    frame_number: int
    timestamp: float
    canvas_state: np.ndarray
    active_strokes: List[int]
    brush_position: Optional[Tuple[float, float]]
    brush_pressure: float
    metadata: Dict[str, Any]


@dataclass
class AnimationSequence:
    """动画序列"""
    frames: List[AnimationFrame]
    total_duration: float
    fps: int
    resolution: Tuple[int, int]
    metadata: Dict[str, Any]


class PaintingAnimator:
    """绘画动画器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化绘画动画器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 动画配置
        self.animation_config = AnimationConfig(
            fps=config.get('fps', 30),
            duration=config.get('duration', 10.0),
            resolution=tuple(config.get('resolution', (1920, 1080))),
            background_color=tuple(config.get('background_color', (255, 255, 255))),
            style=AnimationStyle(config.get('style', 'realistic')),
            speed=PaintingSpeed(config.get('speed', 1.0)),
            show_brush=config.get('show_brush', True),
            show_progress=config.get('show_progress', True),
            enable_effects=config.get('enable_effects', True),
            smooth_transitions=config.get('smooth_transitions', True)
        )
        
        # 初始化组件
        self.stroke_animator = StrokeAnimator(config)
        self.direction_detector = DirectionDetector(config)
        self.animation_controller = AnimationController(config)
        
        # 动画状态
        self.current_sequence = None
        self.is_animating = False
        self.animation_progress = 0.0
        
        # 缓存
        self.frame_cache = {}
        self.stroke_cache = {}
        
    def create_animation(self, strokes: List[Stroke], 
                        reference_image: np.ndarray = None,
                        custom_config: AnimationConfig = None) -> AnimationSequence:
        """
        创建绘画动画
        
        Args:
            strokes: 笔触列表
            reference_image: 参考图像
            custom_config: 自定义动画配置
            
        Returns:
            动画序列
        """
        if not strokes:
            return AnimationSequence(
                frames=[],
                total_duration=0.0,
                fps=self.animation_config.fps,
                resolution=self.animation_config.resolution,
                metadata={'error': 'No strokes provided'}
            )
        
        # 使用自定义配置或默认配置
        config = custom_config or self.animation_config
        
        try:
            self.logger.info(f"Creating animation with {len(strokes)} strokes")
            
            # 计算动画时间线
            timeline = self._calculate_timeline(strokes, config)
            
            # 初始化画布
            canvas = self._initialize_canvas(config, reference_image)
            
            # 生成动画帧
            frames = self._generate_frames(strokes, timeline, canvas, config)
            
            # 创建动画序列
            sequence = AnimationSequence(
                frames=frames,
                total_duration=config.duration,
                fps=config.fps,
                resolution=config.resolution,
                metadata={
                    'num_strokes': len(strokes),
                    'num_frames': len(frames),
                    'style': config.style.value,
                    'speed': config.speed.value,
                    'creation_time': time.time()
                }
            )
            
            self.current_sequence = sequence
            self.logger.info(f"Animation created: {len(frames)} frames, {config.duration:.2f}s")
            
            return sequence
            
        except Exception as e:
            self.logger.error(f"Error creating animation: {e}")
            return AnimationSequence(
                frames=[],
                total_duration=0.0,
                fps=config.fps,
                resolution=config.resolution,
                metadata={'error': str(e)}
            )
    
    def animate_stroke_sequence(self, strokes: List[Stroke], 
                               timing_info: Dict[str, Any] = None) -> AnimationSequence:
        """
        为笔触序列创建动画
        
        Args:
            strokes: 笔触列表
            timing_info: 时间信息
            
        Returns:
            动画序列
        """
        # 计算每个笔触的动画时间
        stroke_timings = self._calculate_stroke_timings(strokes, timing_info)
        
        # 为每个笔触生成动画
        all_frames = []
        current_time = 0.0
        canvas = self._initialize_canvas(self.animation_config)
        
        for i, stroke in enumerate(strokes):
            stroke_duration = stroke_timings[i]['duration']
            stroke_start_time = stroke_timings[i]['start_time']
            
            # 为单个笔触创建动画
            stroke_dict = {
                'points': stroke.points if hasattr(stroke, 'points') else [],
                'features': {
                    'length': getattr(stroke, 'length', 50),
                    'complexity': getattr(stroke, 'complexity', 0.5),
                    'thickness': getattr(stroke, 'thickness', 0.5),
                    'wetness': getattr(stroke, 'wetness', 0.5)
                },
                'color': getattr(stroke, 'color', (0, 0, 0))
            }
            
            # 使用stroke_animator创建动画
            stroke_frames_data = self.stroke_animator.create_animation(
                [stroke_dict], 
                [0],  # 单个笔触的顺序
                AnimationMode.SEQUENTIAL
            )
            
            # 转换为AnimationFrame格式
            stroke_frames = []
            for frame_data in stroke_frames_data:
                frame = AnimationFrame(
                    frame_number=frame_data.frame_id,
                    timestamp=frame_data.timestamp,
                    canvas_state=frame_data.canvas,
                    active_strokes=frame_data.active_strokes,
                    brush_position=None,
                    brush_pressure=0.0,
                    metadata=frame_data.metadata
                )
                stroke_frames.append(frame)
            
            # 调整帧时间戳
            for frame in stroke_frames:
                frame.timestamp += stroke_start_time
                all_frames.append(frame)
            
            # 更新画布状态
            if stroke_frames:
                canvas = stroke_frames[-1].canvas_state.copy()
        
        return AnimationSequence(
            frames=all_frames,
            total_duration=max(f.timestamp for f in all_frames) if all_frames else 0.0,
            fps=self.animation_config.fps,
            resolution=self.animation_config.resolution,
            metadata={
                'num_strokes': len(strokes),
                'stroke_timings': stroke_timings
            }
        )
    
    def create_preview_animation(self, strokes: List[Stroke], 
                               max_duration: float = 5.0) -> AnimationSequence:
        """
        创建预览动画（快速版本）
        
        Args:
            strokes: 笔触列表
            max_duration: 最大持续时间
            
        Returns:
            预览动画序列
        """
        # 创建预览配置
        preview_config = AnimationConfig(
            fps=15,  # 降低帧率
            duration=min(max_duration, len(strokes) * 0.1),
            resolution=(640, 480),  # 降低分辨率
            style=self.animation_config.style,
            speed=PaintingSpeed.FAST,
            show_brush=False,
            enable_effects=False,
            smooth_transitions=False
        )
        
        # 简化笔触（如果太多）
        if len(strokes) > 50:
            # 选择代表性笔触
            step = len(strokes) // 50
            preview_strokes = strokes[::step]
        else:
            preview_strokes = strokes
        
        return self.create_animation(preview_strokes, custom_config=preview_config)
    
    def _calculate_timeline(self, strokes: List[Stroke], 
                           config: AnimationConfig) -> Dict[str, Any]:
        """
        计算动画时间线
        
        Args:
            strokes: 笔触列表
            config: 动画配置
            
        Returns:
            时间线信息
        """
        total_frames = int(config.duration * config.fps)
        
        # 计算每个笔触的时间分配
        stroke_weights = []
        for stroke in strokes:
            # 基于笔触复杂度计算权重
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                complexity = len(stroke.points)
            else:
                complexity = 1
            if hasattr(stroke, 'length'):
                complexity *= stroke.length
            stroke_weights.append(complexity)
        
        total_weight = sum(stroke_weights)
        
        # 分配时间
        timeline = {
            'total_frames': total_frames,
            'total_duration': config.duration,
            'stroke_schedule': []
        }
        
        current_frame = 0
        for i, weight in enumerate(stroke_weights):
            # 计算笔触持续时间
            stroke_frames = max(1, int((weight / total_weight) * total_frames * 0.8))  # 80%用于绘制
            
            # 添加间隔
            pause_frames = max(1, int(total_frames * 0.2 / len(strokes)))  # 20%用于间隔
            
            timeline['stroke_schedule'].append({
                'stroke_index': i,
                'start_frame': current_frame,
                'duration_frames': stroke_frames,
                'end_frame': current_frame + stroke_frames,
                'pause_frames': pause_frames
            })
            
            current_frame += stroke_frames + pause_frames
        
        return timeline
    
    def _initialize_canvas(self, config: AnimationConfig, 
                          reference_image: np.ndarray = None) -> np.ndarray:
        """
        初始化画布
        
        Args:
            config: 动画配置
            reference_image: 参考图像
            
        Returns:
            初始画布
        """
        height, width = config.resolution[1], config.resolution[0]
        
        if reference_image is not None:
            # 调整参考图像大小
            import cv2
            canvas = cv2.resize(reference_image, (width, height))
            # 转换为背景色调
            canvas = canvas * 0.1 + np.array(config.background_color) * 0.9
            canvas = canvas.astype(np.uint8)
        else:
            # 创建纯色背景
            canvas = np.full((height, width, 3), config.background_color, dtype=np.uint8)
        
        return canvas
    
    def _generate_frames(self, strokes: List[Stroke], timeline: Dict[str, Any], 
                        initial_canvas: np.ndarray, config: AnimationConfig) -> List[AnimationFrame]:
        """
        生成动画帧
        
        Args:
            strokes: 笔触列表
            timeline: 时间线
            initial_canvas: 初始画布
            config: 动画配置
            
        Returns:
            动画帧列表
        """
        frames = []
        canvas = initial_canvas.copy()
        
        total_frames = timeline['total_frames']
        
        for frame_num in range(total_frames):
            timestamp = frame_num / config.fps
            
            # 确定当前帧应该绘制的笔触
            active_strokes = []
            brush_position = None
            brush_pressure = 0.0
            
            for stroke_info in timeline['stroke_schedule']:
                if (stroke_info['start_frame'] <= frame_num <= 
                    stroke_info['start_frame'] + stroke_info['duration_frames']):
                    
                    stroke_index = stroke_info['stroke_index']
                    active_strokes.append(stroke_index)
                    
                    # 计算笔触内的进度
                    stroke_progress = ((frame_num - stroke_info['start_frame']) / 
                                     stroke_info['duration_frames'])
                    
                    # 绘制笔触的部分
                    stroke = strokes[stroke_index]
                    canvas = self._draw_stroke_partial(canvas, stroke, stroke_progress, config)
                    
                    # 更新笔刷位置
                    # 检查笔触点数据和笔刷显示设置
                    has_valid_points = (hasattr(stroke, 'points') and 
                                       stroke.points is not None and 
                                       len(stroke.points) > 0)
                    # 确保config.show_brush是布尔值
                    show_brush = bool(getattr(config, 'show_brush', False))
                    if has_valid_points and show_brush:
                        point_index = int(stroke_progress * (len(stroke.points) - 1))
                        point_index = min(point_index, len(stroke.points) - 1)
                        brush_position = stroke.points[point_index]
                        brush_pressure = stroke_progress  # 简化的压力模拟
            
            # 创建动画帧
            frame = AnimationFrame(
                frame_number=frame_num,
                timestamp=timestamp,
                canvas_state=canvas.copy(),
                active_strokes=active_strokes,
                brush_position=brush_position,
                brush_pressure=brush_pressure,
                metadata={
                    'progress': frame_num / total_frames,
                    'active_stroke_count': len(active_strokes)
                }
            )
            
            frames.append(frame)
        
        return frames
    
    def _draw_stroke_partial(self, canvas: np.ndarray, stroke: Stroke, 
                           progress: float, config: AnimationConfig) -> np.ndarray:
        """
        部分绘制笔触
        
        Args:
            canvas: 画布
            stroke: 笔触
            progress: 绘制进度 (0-1)
            config: 动画配置
            
        Returns:
            更新后的画布
        """
        # 检查笔触是否有有效的点数据
        if not hasattr(stroke, 'points') or stroke.points is None:
            return canvas
        if len(stroke.points) == 0 or progress <= 0:
            return canvas
        
        # 计算要绘制的点数
        total_points = len(stroke.points)
        points_to_draw = int(progress * total_points)
        points_to_draw = max(1, min(points_to_draw, total_points))
        
        # 获取要绘制的点
        points = stroke.points[:points_to_draw]
        
        if len(points) < 2:
            return canvas
        
        # 绘制线条
        import cv2
        
        # 确定颜色
        if hasattr(stroke, 'color') and stroke.color is not None:
            color = stroke.color
        else:
            color = (0, 0, 0)  # 默认黑色
        
        # 确定线条粗细
        if hasattr(stroke, 'thickness') and stroke.thickness is not None:
            thickness = max(1, int(stroke.thickness))
        else:
            thickness = 2
        
        # 绘制路径
        for i in range(len(points) - 1):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[i+1][0]), int(points[i+1][1]))
            
            # 根据动画风格调整绘制方式
            if config.style == AnimationStyle.WATERCOLOR:
                # 水彩效果：半透明，边缘模糊
                overlay = canvas.copy()
                cv2.line(overlay, pt1, pt2, color, thickness + 2)
                canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)
            
            elif config.style == AnimationStyle.CALLIGRAPHY:
                # 书法效果：变化的线条粗细
                dynamic_thickness = max(1, int(thickness * (0.5 + 0.5 * np.sin(i * 0.5))))
                cv2.line(canvas, pt1, pt2, color, dynamic_thickness)
            
            else:
                # 默认绘制
                cv2.line(canvas, pt1, pt2, color, thickness)
        
        return canvas
    
    def _calculate_stroke_timings(self, strokes: List[Stroke], 
                                 timing_info: Dict[str, Any] = None) -> List[Dict[str, float]]:
        """
        计算笔触时间安排
        
        Args:
            strokes: 笔触列表
            timing_info: 时间信息
            
        Returns:
            每个笔触的时间信息
        """
        timings = []
        current_time = 0.0
        
        base_duration = timing_info.get('base_duration', 0.5) if timing_info else 0.5
        pause_duration = timing_info.get('pause_duration', 0.1) if timing_info else 0.1
        
        for stroke in strokes:
            # 基于笔触复杂度计算持续时间
            complexity_factor = 1.0
            # 检查笔触点数据
            has_valid_points = (hasattr(stroke, 'points') and 
                               stroke.points is not None and 
                               len(stroke.points) > 0)
            if has_valid_points:
                complexity_factor = min(3.0, len(stroke.points) / 10.0)
            
            duration = base_duration * complexity_factor
            
            timings.append({
                'start_time': current_time,
                'duration': duration,
                'end_time': current_time + duration
            })
            
            current_time += duration + pause_duration
        
        return timings
    
    def get_animation_info(self) -> Dict[str, Any]:
        """
        获取当前动画信息
        
        Returns:
            动画信息
        """
        if self.current_sequence is None:
            return {'status': 'no_animation'}
        
        return {
            'status': 'ready',
            'total_frames': len(self.current_sequence.frames),
            'duration': self.current_sequence.total_duration,
            'fps': self.current_sequence.fps,
            'resolution': self.current_sequence.resolution,
            'metadata': self.current_sequence.metadata
        }
    
    def export_frames(self, output_dir: str, format: str = 'png') -> List[str]:
        """
        导出动画帧
        
        Args:
            output_dir: 输出目录
            format: 图像格式
            
        Returns:
            导出的文件路径列表
        """
        if self.current_sequence is None:
            return []
        
        import os
        import cv2
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        
        for i, frame in enumerate(self.current_sequence.frames):
            filename = f"frame_{i:06d}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # 保存帧
            cv2.imwrite(filepath, frame.canvas_state)
            exported_files.append(filepath)
        
        self.logger.info(f"Exported {len(exported_files)} frames to {output_dir}")
        return exported_files
    
    def clear_cache(self):
        """清除缓存"""
        self.frame_cache.clear()
        self.stroke_cache.clear()
        self.logger.info("Animation cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计信息
        """
        return {
            'frame_cache_size': len(self.frame_cache),
            'stroke_cache_size': len(self.stroke_cache),
            'current_sequence_frames': len(self.current_sequence.frames) if self.current_sequence else 0,
            'animation_config': {
                'fps': self.animation_config.fps,
                'resolution': self.animation_config.resolution,
                'style': self.animation_config.style.value,
                'effects_enabled': self.animation_config.enable_effects
            }
        }