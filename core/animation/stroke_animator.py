# -*- coding: utf-8 -*-
"""
笔触动画器

实现论文中的动态绘制功能：
1. 笔触动画生成
2. 绘制速度控制
3. 视觉效果优化
4. 动画序列管理
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
from .direction_detector import DirectionDetector, StrokeDirection


class AnimationMode(Enum):
    """
    动画模式枚举
    """
    SEQUENTIAL = "sequential"  # 顺序绘制
    PARALLEL = "parallel"      # 并行绘制
    STAGED = "staged"          # 分阶段绘制
    INTERACTIVE = "interactive" # 交互式绘制


class SpeedProfile(Enum):
    """
    速度配置枚举
    """
    CONSTANT = "constant"      # 恒定速度
    ADAPTIVE = "adaptive"      # 自适应速度
    ARTISTIC = "artistic"      # 艺术化速度
    REALISTIC = "realistic"    # 真实绘画速度


@dataclass
class AnimationFrame:
    """
    动画帧数据结构
    """
    frame_id: int
    timestamp: float
    canvas: np.ndarray
    active_strokes: List[int] = field(default_factory=list)
    completed_strokes: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrokeAnimation:
    """
    笔触动画数据结构
    """
    stroke_id: int
    direction: StrokeDirection
    start_frame: int
    end_frame: int
    speed_profile: List[float]  # 每帧的绘制速度
    opacity_profile: List[float]  # 每帧的透明度
    brush_size_profile: List[float]  # 每帧的笔刷大小
    current_progress: float = 0.0
    is_completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnimationConfig:
    """
    动画配置
    """
    fps: int = 30
    total_duration: float = 10.0  # 总时长（秒）
    canvas_size: Tuple[int, int] = (800, 600)
    background_color: Tuple[int, int, int] = (255, 255, 255)
    
    # 速度参数
    base_speed: float = 1.0
    speed_variation: float = 0.3
    thickness_speed_factor: float = 0.5  # 厚度对速度的影响
    wetness_speed_factor: float = 0.3    # 湿度对速度的影响
    
    # 视觉效果参数
    fade_in_duration: float = 0.1
    fade_out_duration: float = 0.1
    brush_opacity: float = 0.8
    trail_effect: bool = True
    trail_length: int = 5
    
    # 优化参数
    parallel_rendering: bool = True
    max_workers: int = 4
    memory_optimization: bool = True
    quality_level: str = "high"  # low, medium, high


class StrokeAnimator:
    """
    笔触动画器
    
    负责生成笔触的动态绘制动画
    """
    
    def __init__(self, config: AnimationConfig):
        """
        初始化动画器
        
        Args:
            config: 动画配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.direction_detector = DirectionDetector({
            'gaussian_sigma': 1.0,
            'gradient_threshold': 0.1,
            'skeleton_threshold': 0.5
        })
        
        # 动画状态
        self.animations: List[StrokeAnimation] = []
        self.frames: List[AnimationFrame] = []
        self.current_frame = 0
        self.is_playing = False
        self.is_paused = False
        
        # 渲染缓存
        self._canvas_cache = {}
        self._stroke_cache = {}
        
        # 回调函数
        self.frame_callbacks: List[Callable[[AnimationFrame], None]] = []
        self.completion_callbacks: List[Callable[[], None]] = []
    
    def create_animation(self, strokes: List[Dict[str, Any]], 
                        stroke_order: List[int],
                        mode: AnimationMode = AnimationMode.SEQUENTIAL) -> List[AnimationFrame]:
        """
        创建笔触动画
        
        Args:
            strokes: 笔触列表
            stroke_order: 笔触绘制顺序
            mode: 动画模式
            
        Returns:
            List[AnimationFrame]: 动画帧列表
        """
        try:
            self.logger.info(f"Creating animation with {len(strokes)} strokes in {mode.value} mode")
            
            # 清空之前的动画
            self.animations.clear()
            self.frames.clear()
            
            # 分析笔触并创建动画
            self._analyze_strokes(strokes, stroke_order)
            
            # 根据模式生成动画时间线
            if mode == AnimationMode.SEQUENTIAL:
                self._create_sequential_timeline()
            elif mode == AnimationMode.PARALLEL:
                self._create_parallel_timeline()
            elif mode == AnimationMode.STAGED:
                self._create_staged_timeline(strokes)
            elif mode == AnimationMode.INTERACTIVE:
                self._create_interactive_timeline()
            
            # 生成动画帧
            self._generate_frames()
            
            self.logger.info(f"Animation created with {len(self.frames)} frames")
            return self.frames
            
        except Exception as e:
            self.logger.error(f"Error creating animation: {str(e)}")
            return []
    
    def _analyze_strokes(self, strokes: List[Dict[str, Any]], stroke_order: List[int]):
        """
        分析笔触并创建动画对象
        
        Args:
            strokes: 笔触列表
            stroke_order: 笔触绘制顺序
        """
        for i, stroke_id in enumerate(stroke_order):
            if stroke_id >= len(strokes):
                continue
                
            stroke = strokes[stroke_id]
            
            # 检测笔触方向
            direction = self._detect_stroke_direction(stroke)
            
            # 计算动画参数
            duration = self._calculate_stroke_duration(stroke)
            start_time = self._calculate_start_time(i, duration)
            
            # 创建动画对象
            animation = self._create_stroke_animation(
                stroke_id, stroke, direction, start_time, duration
            )
            
            self.animations.append(animation)
    
    def _detect_stroke_direction(self, stroke: Dict[str, Any]) -> StrokeDirection:
        """
        检测笔触方向
        
        Args:
            stroke: 笔触数据
            
        Returns:
            StrokeDirection: 方向信息
        """
        # 获取笔触掩码
        mask = stroke.get('mask')
        if mask is None:
            # 从轮廓创建掩码
            contour = stroke.get('contour')
            if contour is not None:
                mask = np.zeros(self.config.canvas_size[::-1], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
            else:
                return self._create_default_direction()
        
        # 使用方向检测器
        return self.direction_detector.detect_direction(mask, stroke)
    
    def _calculate_stroke_duration(self, stroke: Dict[str, Any]) -> float:
        """
        计算笔触绘制时长
        
        Args:
            stroke: 笔触数据
            
        Returns:
            float: 绘制时长（秒）
        """
        # 基础时长基于笔触复杂度
        base_duration = 1.0
        
        # 根据笔触特征调整时长
        features = stroke.get('features', {})
        
        # 长度因子
        length = features.get('length', 50)
        length_factor = min(3.0, length / 100.0)
        
        # 复杂度因子
        complexity = features.get('complexity', 0.5)
        complexity_factor = 1.0 + complexity
        
        # 厚度因子（厚笔触绘制更慢）
        thickness = features.get('thickness', 0.5)
        thickness_factor = 1.0 + thickness * self.config.thickness_speed_factor
        
        # 湿度因子（湿笔触绘制更慢）
        wetness = features.get('wetness', 0.5)
        wetness_factor = 1.0 + wetness * self.config.wetness_speed_factor
        
        total_duration = base_duration * length_factor * complexity_factor * thickness_factor * wetness_factor
        
        return max(0.1, min(5.0, total_duration))  # 限制在0.1-5秒之间
    
    def _calculate_start_time(self, order_index: int, duration: float) -> float:
        """
        计算笔触开始时间
        
        Args:
            order_index: 顺序索引
            duration: 笔触时长
            
        Returns:
            float: 开始时间（秒）
        """
        if order_index == 0:
            return 0.0
        
        # 计算前面所有笔触的累计时间
        total_time = 0.0
        for i in range(order_index):
            if i < len(self.animations):
                prev_duration = self.animations[i].end_frame - self.animations[i].start_frame
                total_time += prev_duration / self.config.fps
        
        return total_time
    
    def _create_stroke_animation(self, stroke_id: int, stroke: Dict[str, Any], 
                               direction: StrokeDirection, start_time: float, 
                               duration: float) -> StrokeAnimation:
        """
        创建笔触动画对象
        
        Args:
            stroke_id: 笔触ID
            stroke: 笔触数据
            direction: 方向信息
            start_time: 开始时间
            duration: 持续时间
            
        Returns:
            StrokeAnimation: 动画对象
        """
        start_frame = int(start_time * self.config.fps)
        end_frame = int((start_time + duration) * self.config.fps)
        frame_count = end_frame - start_frame
        
        # 生成速度配置
        speed_profile = self._generate_speed_profile(stroke, frame_count)
        
        # 生成透明度配置
        opacity_profile = self._generate_opacity_profile(frame_count)
        
        # 生成笔刷大小配置
        brush_size_profile = self._generate_brush_size_profile(stroke, frame_count)
        
        return StrokeAnimation(
            stroke_id=stroke_id,
            direction=direction,
            start_frame=start_frame,
            end_frame=end_frame,
            speed_profile=speed_profile,
            opacity_profile=opacity_profile,
            brush_size_profile=brush_size_profile,
            metadata={
                'stroke_data': stroke,
                'duration': duration,
                'start_time': start_time
            }
        )
    
    def _generate_speed_profile(self, stroke: Dict[str, Any], frame_count: int) -> List[float]:
        """
        生成速度配置
        
        Args:
            stroke: 笔触数据
            frame_count: 帧数
            
        Returns:
            List[float]: 速度配置
        """
        features = stroke.get('features', {})
        base_speed = self.config.base_speed
        
        # 根据笔触特征调整速度
        thickness = features.get('thickness', 0.5)
        wetness = features.get('wetness', 0.5)
        
        # 生成速度曲线
        speeds = []
        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)
            
            # 基础速度曲线（开始慢，中间快，结束慢）
            speed_curve = 0.5 + 0.5 * np.sin(progress * np.pi)
            
            # 应用特征调整
            thickness_factor = 1.0 - thickness * 0.3  # 厚笔触慢一些
            wetness_factor = 1.0 - wetness * 0.2      # 湿笔触慢一些
            
            # 添加随机变化
            variation = 1.0 + (np.random.random() - 0.5) * self.config.speed_variation
            
            final_speed = base_speed * speed_curve * thickness_factor * wetness_factor * variation
            speeds.append(max(0.1, min(2.0, final_speed)))
        
        return speeds
    
    def _generate_opacity_profile(self, frame_count: int) -> List[float]:
        """
        生成透明度配置
        
        Args:
            frame_count: 帧数
            
        Returns:
            List[float]: 透明度配置
        """
        fade_in_frames = int(self.config.fade_in_duration * self.config.fps)
        fade_out_frames = int(self.config.fade_out_duration * self.config.fps)
        
        opacities = []
        for i in range(frame_count):
            if i < fade_in_frames:
                # 淡入
                opacity = (i / fade_in_frames) * self.config.brush_opacity
            elif i >= frame_count - fade_out_frames:
                # 淡出
                fade_progress = (frame_count - 1 - i) / fade_out_frames
                opacity = fade_progress * self.config.brush_opacity
            else:
                # 正常透明度
                opacity = self.config.brush_opacity
            
            opacities.append(max(0.0, min(1.0, opacity)))
        
        return opacities
    
    def _generate_brush_size_profile(self, stroke: Dict[str, Any], frame_count: int) -> List[float]:
        """
        生成笔刷大小配置
        
        Args:
            stroke: 笔触数据
            frame_count: 帧数
            
        Returns:
            List[float]: 笔刷大小配置
        """
        features = stroke.get('features', {})
        base_size = features.get('thickness', 0.5) * 20 + 5  # 基础大小5-25像素
        
        sizes = []
        for i in range(frame_count):
            progress = i / max(1, frame_count - 1)
            
            # 笔刷大小变化（模拟压力变化）
            pressure_curve = 0.8 + 0.2 * np.sin(progress * np.pi * 2)
            size = base_size * pressure_curve
            
            sizes.append(max(1.0, size))
        
        return sizes
    
    def _create_sequential_timeline(self):
        """
        创建顺序时间线
        """
        # 笔触按顺序绘制，每个笔触完成后开始下一个
        pass  # 已在_analyze_strokes中处理
    
    def _create_parallel_timeline(self):
        """
        创建并行时间线
        """
        # 所有笔触同时开始
        for animation in self.animations:
            animation.start_frame = 0
    
    def _create_staged_timeline(self, strokes: List[Dict[str, Any]]):
        """
        创建分阶段时间线
        
        Args:
            strokes: 笔触列表
        """
        # 根据笔触阶段分组
        stage_groups = self._group_strokes_by_stage(strokes)
        
        current_time = 0.0
        for stage, stroke_ids in stage_groups.items():
            stage_duration = 0.0
            
            # 该阶段内的笔触并行绘制
            for animation in self.animations:
                if animation.stroke_id in stroke_ids:
                    animation.start_frame = int(current_time * self.config.fps)
                    duration = (animation.end_frame - animation.start_frame) / self.config.fps
                    stage_duration = max(stage_duration, duration)
            
            current_time += stage_duration + 0.5  # 阶段间间隔
    
    def _create_interactive_timeline(self):
        """
        创建交互式时间线
        """
        # 交互式模式下，时间线由用户控制
        for animation in self.animations:
            animation.start_frame = -1  # 标记为待触发
    
    def _group_strokes_by_stage(self, strokes: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        按阶段分组笔触
        
        Args:
            strokes: 笔触列表
            
        Returns:
            Dict[str, List[int]]: 阶段分组
        """
        groups = {'main': [], 'detail': [], 'decoration': []}
        
        for i, stroke in enumerate(strokes):
            stage = stroke.get('stage', 'main')
            if stage in groups:
                groups[stage].append(i)
            else:
                groups['main'].append(i)
        
        return groups
    
    def _generate_frames(self):
        """
        生成动画帧
        """
        # 计算总帧数
        max_frame = max([anim.end_frame for anim in self.animations] + [0])
        total_frames = max_frame + 10  # 添加一些缓冲帧
        
        self.logger.info(f"Generating {total_frames} animation frames")
        
        # 初始化画布
        canvas = np.full(
            (self.config.canvas_size[1], self.config.canvas_size[0], 3),
            self.config.background_color,
            dtype=np.uint8
        )
        
        # 生成每一帧
        for frame_id in range(total_frames):
            frame_canvas = canvas.copy()
            active_strokes = []
            completed_strokes = []
            
            # 渲染当前帧的所有活跃笔触
            for animation in self.animations:
                if animation.start_frame <= frame_id < animation.end_frame:
                    # 笔触正在绘制
                    self._render_stroke_frame(frame_canvas, animation, frame_id)
                    active_strokes.append(animation.stroke_id)
                elif frame_id >= animation.end_frame:
                    # 笔触已完成
                    self._render_completed_stroke(frame_canvas, animation)
                    completed_strokes.append(animation.stroke_id)
            
            # 创建动画帧
            frame = AnimationFrame(
                frame_id=frame_id,
                timestamp=frame_id / self.config.fps,
                canvas=frame_canvas,
                active_strokes=active_strokes,
                completed_strokes=completed_strokes
            )
            
            self.frames.append(frame)
    
    def _render_stroke_frame(self, canvas: np.ndarray, animation: StrokeAnimation, frame_id: int):
        """
        渲染笔触的当前帧
        
        Args:
            canvas: 画布
            animation: 动画对象
            frame_id: 帧ID
        """
        # 计算当前进度
        local_frame = frame_id - animation.start_frame
        total_frames = animation.end_frame - animation.start_frame
        progress = local_frame / max(1, total_frames - 1)
        
        # 获取当前帧的参数
        if local_frame < len(animation.speed_profile):
            speed = animation.speed_profile[local_frame]
            opacity = animation.opacity_profile[local_frame]
            brush_size = animation.brush_size_profile[local_frame]
        else:
            speed = 1.0
            opacity = self.config.brush_opacity
            brush_size = 10.0
        
        # 计算当前绘制位置
        path_points = animation.direction.path_points
        if not path_points:
            return
        
        # 根据进度确定绘制到哪个点
        target_index = int(progress * len(path_points))
        target_index = min(target_index, len(path_points) - 1)
        
        # 绘制路径
        stroke_data = animation.metadata['stroke_data']
        color = stroke_data.get('color', (0, 0, 0))
        
        for i in range(target_index):
            if i + 1 < len(path_points):
                pt1 = path_points[i]
                pt2 = path_points[i + 1]
                
                # 绘制线段
                self._draw_brush_stroke(canvas, pt1, pt2, color, brush_size, opacity)
    
    def _render_completed_stroke(self, canvas: np.ndarray, animation: StrokeAnimation):
        """
        渲染已完成的笔触
        
        Args:
            canvas: 画布
            animation: 动画对象
        """
        # 绘制完整笔触
        stroke_data = animation.metadata['stroke_data']
        mask = stroke_data.get('mask')
        color = stroke_data.get('color', (0, 0, 0))
        
        if mask is not None:
            # 使用掩码绘制
            colored_mask = np.zeros_like(canvas)
            mask_indices = mask > 0
            colored_mask[mask_indices] = color
            
            # 混合到画布
            alpha = self.config.brush_opacity
            canvas[:] = cv2.addWeighted(canvas, 1 - alpha, colored_mask, alpha, 0)
    
    def _draw_brush_stroke(self, canvas: np.ndarray, pt1: Tuple[int, int], 
                          pt2: Tuple[int, int], color: Tuple[int, int, int], 
                          brush_size: float, opacity: float):
        """
        绘制笔刷笔触
        
        Args:
            canvas: 画布
            pt1: 起始点
            pt2: 结束点
            color: 颜色
            brush_size: 笔刷大小
            opacity: 透明度
        """
        # 创建临时图层
        overlay = canvas.copy()
        
        # 绘制线条
        cv2.line(overlay, pt1, pt2, color, int(brush_size))
        
        # 混合到画布
        cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0, canvas)
    
    def play_animation(self, start_frame: int = 0, end_frame: Optional[int] = None, 
                      playback_speed: float = 1.0) -> bool:
        """
        播放动画
        
        Args:
            start_frame: 开始帧
            end_frame: 结束帧
            playback_speed: 播放速度
            
        Returns:
            bool: 是否成功开始播放
        """
        if not self.frames:
            self.logger.warning("No frames to play")
            return False
        
        if end_frame is None:
            end_frame = len(self.frames)
        
        self.is_playing = True
        self.is_paused = False
        self.current_frame = start_frame
        
        try:
            frame_duration = 1.0 / (self.config.fps * playback_speed)
            
            while self.is_playing and self.current_frame < end_frame:
                if not self.is_paused:
                    # 获取当前帧
                    frame = self.frames[self.current_frame]
                    
                    # 调用帧回调
                    for callback in self.frame_callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            self.logger.error(f"Frame callback error: {str(e)}")
                    
                    self.current_frame += 1
                
                # 控制播放速度
                time.sleep(frame_duration)
            
            # 播放完成
            self.is_playing = False
            for callback in self.completion_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Completion callback error: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error playing animation: {str(e)}")
            self.is_playing = False
            return False
    
    def pause_animation(self):
        """
        暂停动画
        """
        self.is_paused = True
    
    def resume_animation(self):
        """
        恢复动画
        """
        self.is_paused = False
    
    def stop_animation(self):
        """
        停止动画
        """
        self.is_playing = False
        self.is_paused = False
    
    def seek_to_frame(self, frame_id: int):
        """
        跳转到指定帧
        
        Args:
            frame_id: 目标帧ID
        """
        if 0 <= frame_id < len(self.frames):
            self.current_frame = frame_id
    
    def export_video(self, output_path: str, codec: str = 'mp4v') -> bool:
        """
        导出视频
        
        Args:
            output_path: 输出路径
            codec: 视频编码
            
        Returns:
            bool: 是否成功导出
        """
        if not self.frames:
            self.logger.warning("No frames to export")
            return False
        
        try:
            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                self.config.canvas_size
            )
            
            # 写入所有帧
            for frame in self.frames:
                # 转换颜色空间（OpenCV使用BGR）
                bgr_frame = cv2.cvtColor(frame.canvas, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
            
            writer.release()
            self.logger.info(f"Video exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting video: {str(e)}")
            return False
    
    def export_frames(self, output_dir: str, format: str = 'png') -> bool:
        """
        导出帧图像
        
        Args:
            output_dir: 输出目录
            format: 图像格式
            
        Returns:
            bool: 是否成功导出
        """
        if not self.frames:
            self.logger.warning("No frames to export")
            return False
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            for i, frame in enumerate(self.frames):
                filename = f"frame_{i:06d}.{format}"
                filepath = os.path.join(output_dir, filename)
                
                # 转换颜色空间
                bgr_frame = cv2.cvtColor(frame.canvas, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr_frame)
            
            self.logger.info(f"Frames exported to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting frames: {str(e)}")
            return False
    
    def add_frame_callback(self, callback: Callable[[AnimationFrame], None]):
        """
        添加帧回调函数
        
        Args:
            callback: 回调函数
        """
        self.frame_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[], None]):
        """
        添加完成回调函数
        
        Args:
            callback: 回调函数
        """
        self.completion_callbacks.append(callback)
    
    def get_animation_info(self) -> Dict[str, Any]:
        """
        获取动画信息
        
        Returns:
            Dict[str, Any]: 动画信息
        """
        if not self.frames:
            return {}
        
        total_duration = len(self.frames) / self.config.fps
        
        return {
            'total_frames': len(self.frames),
            'total_duration': total_duration,
            'fps': self.config.fps,
            'canvas_size': self.config.canvas_size,
            'stroke_count': len(self.animations),
            'current_frame': self.current_frame,
            'is_playing': self.is_playing,
            'is_paused': self.is_paused
        }
    
    def _create_default_direction(self) -> StrokeDirection:
        """
        创建默认方向
        
        Returns:
            StrokeDirection: 默认方向
        """
        from .direction_detector import StrokeDirection, DirectionMethod
        
        return StrokeDirection(
            start_point=(0, 0),
            end_point=(10, 0),
            direction_vector=(1.0, 0.0),
            confidence=0.5,
            method=DirectionMethod.COMBINED,
            path_points=[(0, 0), (10, 0)]
        )