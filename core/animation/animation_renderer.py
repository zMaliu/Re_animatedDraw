# -*- coding: utf-8 -*-
"""
动画渲染器

实现绘画动画的渲染和输出
包括视频生成、帧序列导出、实时预览等功能
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import os
import time
import math
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import imageio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


@dataclass
class RenderSettings:
    """
    渲染设置数据结构
    
    Attributes:
        output_format (str): 输出格式
        resolution (Tuple[int, int]): 分辨率
        fps (int): 帧率
        quality (str): 质量设置
        background_color (Tuple[int, int, int]): 背景颜色
        show_brush (bool): 显示笔刷
        show_trajectory (bool): 显示轨迹
        show_progress (bool): 显示进度
        enable_effects (bool): 启用特效
        codec (str): 编码器
    """
    output_format: str
    resolution: Tuple[int, int]
    fps: int
    quality: str
    background_color: Tuple[int, int, int]
    show_brush: bool
    show_trajectory: bool
    show_progress: bool
    enable_effects: bool
    codec: str


@dataclass
class RenderProgress:
    """
    渲染进度数据结构
    
    Attributes:
        current_frame (int): 当前帧
        total_frames (int): 总帧数
        current_stroke (int): 当前笔画
        total_strokes (int): 总笔画数
        elapsed_time (float): 已用时间
        estimated_time (float): 预计总时间
        progress_percentage (float): 进度百分比
        status (str): 状态
    """
    current_frame: int
    total_frames: int
    current_stroke: int
    total_strokes: int
    elapsed_time: float
    estimated_time: float
    progress_percentage: float
    status: str


class AnimationRenderer:
    """
    动画渲染器
    
    渲染绘画动画为视频或图像序列
    """
    
    def __init__(self, config):
        """
        初始化动画渲染器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 渲染设置
        self.default_resolution = tuple(config['rendering'].get('resolution', (1920, 1080)))
        self.default_fps = config['rendering'].get('fps', 30)
        self.default_quality = config['rendering'].get('quality', 'high')
        self.default_format = config['rendering'].get('format', 'mp4')
        self.default_codec = config['rendering'].get('codec', 'h264')
        
        # 视觉效果设置
        self.show_brush = config['rendering'].get('show_brush', True)
        self.show_trajectory = config['rendering'].get('show_trajectory', False)
        self.show_progress = config['rendering'].get('show_progress', True)
        self.enable_effects = config['rendering'].get('enable_effects', True)
        
        # 颜色设置
        self.background_color = tuple(config['rendering'].get('background_color', (255, 255, 255)))
        self.brush_color = tuple(config['rendering'].get('brush_color', (0, 0, 0)))
        self.trajectory_color = tuple(config['rendering'].get('trajectory_color', (128, 128, 128)))
        self.progress_color = tuple(config['rendering'].get('progress_color', (255, 0, 0)))
        
        # 字体设置
        self.font_size = config['rendering'].get('font_size', 24)
        self.font_path = config['rendering'].get('font_path', None)
        
        # 性能设置
        self.parallel_rendering = config['rendering'].get('parallel_rendering', False)
        self.memory_limit = config['rendering'].get('memory_limit', 2048)  # MB
        self.temp_dir = config['rendering'].get('temp_dir', './temp')
        
        # 初始化
        self._ensure_temp_dir()
        self.current_canvas = None
        self.render_progress = None
    
    def _ensure_temp_dir(self):
        """
        确保临时目录存在
        """
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating temp directory: {str(e)}")
    
    def render_animation(self, painting_animation, output_path: str,
                        render_settings: Optional[RenderSettings] = None,
                        progress_callback: Optional[Callable] = None) -> bool:
        """
        渲染完整动画
        
        Args:
            painting_animation: 绘画动画对象
            output_path (str): 输出路径
            render_settings (Optional[RenderSettings]): 渲染设置
            progress_callback (Optional[Callable]): 进度回调函数
            
        Returns:
            bool: 是否成功
        """
        try:
            self.logger.info(f"Starting animation rendering to {output_path}")
            
            # 使用默认设置或提供的设置
            if render_settings is None:
                render_settings = self._create_default_render_settings()
            
            # 初始化渲染进度
            self.render_progress = RenderProgress(
                current_frame=0,
                total_frames=len(painting_animation.all_frames),
                current_stroke=0,
                total_strokes=len(painting_animation.stroke_animations),
                elapsed_time=0.0,
                estimated_time=0.0,
                progress_percentage=0.0,
                status="initializing"
            )
            
            start_time = time.time()
            
            # 根据输出格式选择渲染方法
            if render_settings.output_format.lower() in ['mp4', 'avi', 'mov']:
                success = self._render_video(
                    painting_animation, output_path, render_settings, progress_callback
                )
            elif render_settings.output_format.lower() in ['gif']:
                success = self._render_gif(
                    painting_animation, output_path, render_settings, progress_callback
                )
            elif render_settings.output_format.lower() in ['frames', 'png_sequence']:
                success = self._render_frame_sequence(
                    painting_animation, output_path, render_settings, progress_callback
                )
            else:
                self.logger.error(f"Unsupported output format: {render_settings.output_format}")
                return False
            
            # 更新最终进度
            if success:
                self.render_progress.status = "completed"
                self.render_progress.progress_percentage = 100.0
                self.render_progress.elapsed_time = time.time() - start_time
                
                if progress_callback:
                    progress_callback(self.render_progress)
                
                self.logger.info(f"Animation rendering completed in {self.render_progress.elapsed_time:.2f}s")
            else:
                self.render_progress.status = "failed"
                if progress_callback:
                    progress_callback(self.render_progress)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rendering animation: {str(e)}")
            if self.render_progress:
                self.render_progress.status = "error"
                if progress_callback:
                    progress_callback(self.render_progress)
            return False
    
    def _create_default_render_settings(self) -> RenderSettings:
        """
        创建默认渲染设置
        
        Returns:
            RenderSettings: 默认渲染设置
        """
        return RenderSettings(
            output_format=self.default_format,
            resolution=self.default_resolution,
            fps=self.default_fps,
            quality=self.default_quality,
            background_color=self.background_color,
            show_brush=self.show_brush,
            show_trajectory=self.show_trajectory,
            show_progress=self.show_progress,
            enable_effects=self.enable_effects,
            codec=self.default_codec
        )
    
    def _render_video(self, painting_animation, output_path: str,
                     render_settings: RenderSettings,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        渲染视频
        
        Args:
            painting_animation: 绘画动画对象
            output_path (str): 输出路径
            render_settings (RenderSettings): 渲染设置
            progress_callback (Optional[Callable]): 进度回调
            
        Returns:
            bool: 是否成功
        """
        try:
            # 设置视频编码器
            fourcc = self._get_fourcc(render_settings.codec)
            
            # 创建视频写入器
            video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                render_settings.fps,
                render_settings.resolution
            )
            
            if not video_writer.isOpened():
                self.logger.error("Failed to open video writer")
                return False
            
            # 初始化画布
            self.current_canvas = self._create_canvas(render_settings)
            
            # 渲染每一帧
            self.render_progress.status = "rendering"
            
            for frame_idx, frame in enumerate(tqdm(painting_animation.all_frames, desc="Rendering frames")):
                # 更新进度
                self.render_progress.current_frame = frame_idx
                self.render_progress.progress_percentage = (frame_idx / len(painting_animation.all_frames)) * 100
                
                if progress_callback and frame_idx % 10 == 0:  # 每10帧更新一次进度
                    progress_callback(self.render_progress)
                
                # 渲染当前帧
                frame_image = self._render_frame(
                    frame, painting_animation, render_settings
                )
                
                # 写入视频
                video_writer.write(frame_image)
            
            # 释放资源
            video_writer.release()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rendering video: {str(e)}")
            return False
    
    def _render_gif(self, painting_animation, output_path: str,
                   render_settings: RenderSettings,
                   progress_callback: Optional[Callable] = None) -> bool:
        """
        渲染GIF动画
        
        Args:
            painting_animation: 绘画动画对象
            output_path (str): 输出路径
            render_settings (RenderSettings): 渲染设置
            progress_callback (Optional[Callable]): 进度回调
            
        Returns:
            bool: 是否成功
        """
        try:
            frames = []
            
            # 初始化画布
            self.current_canvas = self._create_canvas(render_settings)
            
            # 渲染每一帧
            self.render_progress.status = "rendering"
            
            for frame_idx, frame in enumerate(tqdm(painting_animation.all_frames, desc="Rendering GIF frames")):
                # 更新进度
                self.render_progress.current_frame = frame_idx
                self.render_progress.progress_percentage = (frame_idx / len(painting_animation.all_frames)) * 100
                
                if progress_callback and frame_idx % 10 == 0:
                    progress_callback(self.render_progress)
                
                # 渲染当前帧
                frame_image = self._render_frame(
                    frame, painting_animation, render_settings
                )
                
                # 转换为PIL图像
                pil_image = Image.fromarray(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
                frames.append(pil_image)
            
            # 保存GIF
            if frames:
                duration = int(1000 / render_settings.fps)  # 毫秒
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration,
                    loop=0
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rendering GIF: {str(e)}")
            return False
    
    def _render_frame_sequence(self, painting_animation, output_path: str,
                              render_settings: RenderSettings,
                              progress_callback: Optional[Callable] = None) -> bool:
        """
        渲染帧序列
        
        Args:
            painting_animation: 绘画动画对象
            output_path (str): 输出目录路径
            render_settings (RenderSettings): 渲染设置
            progress_callback (Optional[Callable]): 进度回调
            
        Returns:
            bool: 是否成功
        """
        try:
            # 创建输出目录
            os.makedirs(output_path, exist_ok=True)
            
            # 初始化画布
            self.current_canvas = self._create_canvas(render_settings)
            
            # 渲染每一帧
            self.render_progress.status = "rendering"
            
            for frame_idx, frame in enumerate(tqdm(painting_animation.all_frames, desc="Rendering frame sequence")):
                # 更新进度
                self.render_progress.current_frame = frame_idx
                self.render_progress.progress_percentage = (frame_idx / len(painting_animation.all_frames)) * 100
                
                if progress_callback and frame_idx % 10 == 0:
                    progress_callback(self.render_progress)
                
                # 渲染当前帧
                frame_image = self._render_frame(
                    frame, painting_animation, render_settings
                )
                
                # 保存帧图像
                frame_filename = f"frame_{frame_idx:06d}.png"
                frame_path = os.path.join(output_path, frame_filename)
                cv2.imwrite(frame_path, frame_image)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rendering frame sequence: {str(e)}")
            return False
    
    def _render_frame(self, frame, painting_animation, render_settings: RenderSettings) -> np.ndarray:
        """
        渲染单个帧
        
        Args:
            frame: 动画帧对象
            painting_animation: 绘画动画对象
            render_settings (RenderSettings): 渲染设置
            
        Returns:
            np.ndarray: 渲染后的帧图像
        """
        try:
            # 复制当前画布
            frame_canvas = self.current_canvas.copy()
            
            # 获取当前笔画
            current_stroke_id = frame.metadata.get('stroke_id', -1)
            current_stroke = None
            
            for stroke_anim in painting_animation.stroke_animations:
                if stroke_anim.stroke_id == current_stroke_id:
                    current_stroke = stroke_anim
                    break
            
            # 渲染已完成的笔画
            self._render_completed_strokes(
                frame_canvas, painting_animation, frame.timestamp, render_settings
            )
            
            # 渲染当前笔画进度
            if current_stroke:
                self._render_current_stroke_progress(
                    frame_canvas, current_stroke, frame, render_settings
                )
            
            # 渲染笔刷
            if render_settings.show_brush:
                self._render_brush(
                    frame_canvas, frame, render_settings
                )
            
            # 渲染轨迹
            if render_settings.show_trajectory and current_stroke:
                self._render_trajectory(
                    frame_canvas, current_stroke, frame, render_settings
                )
            
            # 渲染进度信息
            if render_settings.show_progress:
                self._render_progress_info(
                    frame_canvas, frame, painting_animation, render_settings
                )
            
            # 应用特效
            if render_settings.enable_effects:
                frame_canvas = self._apply_effects(frame_canvas, frame, render_settings)
            
            return frame_canvas
            
        except Exception as e:
            self.logger.error(f"Error rendering frame: {str(e)}")
            return self.current_canvas.copy()
    
    def _create_canvas(self, render_settings: RenderSettings) -> np.ndarray:
        """
        创建画布
        
        Args:
            render_settings (RenderSettings): 渲染设置
            
        Returns:
            np.ndarray: 画布图像
        """
        try:
            width, height = render_settings.resolution
            canvas = np.full((height, width, 3), render_settings.background_color, dtype=np.uint8)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error creating canvas: {str(e)}")
            return np.full((600, 800, 3), (255, 255, 255), dtype=np.uint8)
    
    def _render_completed_strokes(self, canvas: np.ndarray, painting_animation,
                                 current_time: float, render_settings: RenderSettings):
        """
        渲染已完成的笔画
        
        Args:
            canvas (np.ndarray): 画布
            painting_animation: 绘画动画对象
            current_time (float): 当前时间
            render_settings (RenderSettings): 渲染设置
        """
        try:
            for stroke_anim in painting_animation.stroke_animations:
                stroke_start_time = stroke_anim.metadata.get('start_time', 0.0)
                stroke_end_time = stroke_start_time + stroke_anim.duration
                
                if current_time >= stroke_end_time:
                    # 渲染完整笔画
                    self._render_complete_stroke(canvas, stroke_anim, render_settings)
                elif current_time > stroke_start_time:
                    # 渲染部分笔画
                    progress = (current_time - stroke_start_time) / stroke_anim.duration
                    self._render_partial_stroke(canvas, stroke_anim, progress, render_settings)
            
        except Exception as e:
            self.logger.error(f"Error rendering completed strokes: {str(e)}")
    
    def _render_complete_stroke(self, canvas: np.ndarray, stroke_anim, render_settings: RenderSettings):
        """
        渲染完整笔画
        
        Args:
            canvas (np.ndarray): 画布
            stroke_anim: 笔画动画对象
            render_settings (RenderSettings): 渲染设置
        """
        try:
            trajectory = stroke_anim.trajectory
            brush_sizes = stroke_anim.brush_size_curve
            
            # 渲染笔画路径
            for i in range(len(trajectory) - 1):
                start_pos = trajectory[i]
                end_pos = trajectory[i + 1]
                brush_size = brush_sizes[i] if i < len(brush_sizes) else 5
                
                # 绘制线段
                cv2.line(
                    canvas,
                    (int(start_pos[0]), int(start_pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    self.brush_color,
                    max(1, int(brush_size))
                )
            
        except Exception as e:
            self.logger.error(f"Error rendering complete stroke: {str(e)}")
    
    def _render_partial_stroke(self, canvas: np.ndarray, stroke_anim, progress: float, render_settings: RenderSettings):
        """
        渲染部分笔画
        
        Args:
            canvas (np.ndarray): 画布
            stroke_anim: 笔画动画对象
            progress (float): 进度（0-1）
            render_settings (RenderSettings): 渲染设置
        """
        try:
            trajectory = stroke_anim.trajectory
            brush_sizes = stroke_anim.brush_size_curve
            
            # 计算要渲染的点数
            num_points = int(len(trajectory) * progress)
            
            # 渲染部分路径
            for i in range(min(num_points - 1, len(trajectory) - 1)):
                start_pos = trajectory[i]
                end_pos = trajectory[i + 1]
                brush_size = brush_sizes[i] if i < len(brush_sizes) else 5
                
                # 绘制线段
                cv2.line(
                    canvas,
                    (int(start_pos[0]), int(start_pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    self.brush_color,
                    max(1, int(brush_size))
                )
            
        except Exception as e:
            self.logger.error(f"Error rendering partial stroke: {str(e)}")
    
    def _render_current_stroke_progress(self, canvas: np.ndarray, stroke_anim, frame, render_settings: RenderSettings):
        """
        渲染当前笔画进度
        
        Args:
            canvas (np.ndarray): 画布
            stroke_anim: 笔画动画对象
            frame: 动画帧对象
            render_settings (RenderSettings): 渲染设置
        """
        try:
            # 基于帧的进度渲染当前笔画
            progress = frame.stroke_progress
            self._render_partial_stroke(canvas, stroke_anim, progress, render_settings)
            
        except Exception as e:
            self.logger.error(f"Error rendering current stroke progress: {str(e)}")
    
    def _render_brush(self, canvas: np.ndarray, frame, render_settings: RenderSettings):
        """
        渲染笔刷
        
        Args:
            canvas (np.ndarray): 画布
            frame: 动画帧对象
            render_settings (RenderSettings): 渲染设置
        """
        try:
            x, y = int(frame.brush_position[0]), int(frame.brush_position[1])
            size = int(frame.brush_size)
            
            # 绘制笔刷圆圈
            cv2.circle(canvas, (x, y), size, self.brush_color, 2)
            
            # 绘制笔刷中心点
            cv2.circle(canvas, (x, y), 2, self.progress_color, -1)
            
        except Exception as e:
            self.logger.error(f"Error rendering brush: {str(e)}")
    
    def _render_trajectory(self, canvas: np.ndarray, stroke_anim, frame, render_settings: RenderSettings):
        """
        渲染轨迹
        
        Args:
            canvas (np.ndarray): 画布
            stroke_anim: 笔画动画对象
            frame: 动画帧对象
            render_settings (RenderSettings): 渲染设置
        """
        try:
            trajectory = stroke_anim.trajectory
            
            # 绘制轨迹线
            points = np.array([(int(p[0]), int(p[1])) for p in trajectory], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(canvas, [points], False, self.trajectory_color, 1)
            
            # 标记当前位置
            current_pos = frame.brush_position
            cv2.circle(canvas, (int(current_pos[0]), int(current_pos[1])), 3, self.progress_color, -1)
            
        except Exception as e:
            self.logger.error(f"Error rendering trajectory: {str(e)}")
    
    def _render_progress_info(self, canvas: np.ndarray, frame, painting_animation, render_settings: RenderSettings):
        """
        渲染进度信息
        
        Args:
            canvas (np.ndarray): 画布
            frame: 动画帧对象
            painting_animation: 绘画动画对象
            render_settings (RenderSettings): 渲染设置
        """
        try:
            # 计算进度
            total_duration = painting_animation.total_duration
            current_time = frame.timestamp
            progress_percentage = (current_time / total_duration) * 100 if total_duration > 0 else 0
            
            # 当前笔画信息
            current_stroke_id = frame.metadata.get('stroke_id', 0)
            total_strokes = len(painting_animation.stroke_animations)
            
            # 渲染进度条
            bar_width = 300
            bar_height = 20
            bar_x = 50
            bar_y = 50
            
            # 背景
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
            
            # 进度
            progress_width = int(bar_width * progress_percentage / 100)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), self.progress_color, -1)
            
            # 边框
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
            
            # 文字信息
            text = f"Progress: {progress_percentage:.1f}% | Stroke: {current_stroke_id + 1}/{total_strokes}"
            cv2.putText(canvas, text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 时间信息
            time_text = f"Time: {current_time:.1f}s / {total_duration:.1f}s"
            cv2.putText(canvas, time_text, (bar_x, bar_y + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error rendering progress info: {str(e)}")
    
    def _apply_effects(self, canvas: np.ndarray, frame, render_settings: RenderSettings) -> np.ndarray:
        """
        应用视觉特效
        
        Args:
            canvas (np.ndarray): 画布
            frame: 动画帧对象
            render_settings (RenderSettings): 渲染设置
            
        Returns:
            np.ndarray: 应用特效后的画布
        """
        try:
            # 添加轻微的模糊效果模拟墨水扩散
            if frame.ink_flow > 0.7:
                kernel_size = max(1, int(frame.ink_flow * 3))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), 0.5)
            
            # 添加纸张纹理效果
            if hasattr(self, 'paper_texture') and self.paper_texture is not None:
                # 混合纸张纹理
                alpha = 0.1
                canvas = cv2.addWeighted(canvas, 1 - alpha, self.paper_texture, alpha, 0)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error applying effects: {str(e)}")
            return canvas
    
    def _get_fourcc(self, codec: str) -> int:
        """
        获取视频编码器
        
        Args:
            codec (str): 编码器名称
            
        Returns:
            int: OpenCV fourcc代码
        """
        codec_map = {
            'h264': cv2.VideoWriter_fourcc(*'H264'),
            'xvid': cv2.VideoWriter_fourcc(*'XVID'),
            'mjpg': cv2.VideoWriter_fourcc(*'MJPG'),
            'mp4v': cv2.VideoWriter_fourcc(*'mp4v')
        }
        
        return codec_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'mp4v'))
    
    def create_preview_frame(self, painting_animation, frame_index: int,
                           render_settings: Optional[RenderSettings] = None) -> np.ndarray:
        """
        创建预览帧
        
        Args:
            painting_animation: 绘画动画对象
            frame_index (int): 帧索引
            render_settings (Optional[RenderSettings]): 渲染设置
            
        Returns:
            np.ndarray: 预览帧图像
        """
        try:
            if render_settings is None:
                render_settings = self._create_default_render_settings()
            
            if frame_index < 0 or frame_index >= len(painting_animation.all_frames):
                return self._create_canvas(render_settings)
            
            # 初始化画布
            self.current_canvas = self._create_canvas(render_settings)
            
            # 渲染指定帧
            frame = painting_animation.all_frames[frame_index]
            preview_frame = self._render_frame(frame, painting_animation, render_settings)
            
            return preview_frame
            
        except Exception as e:
            self.logger.error(f"Error creating preview frame: {str(e)}")
            return self._create_canvas(render_settings or self._create_default_render_settings())
    
    def export_thumbnail(self, painting_animation, output_path: str,
                        size: Tuple[int, int] = (300, 200)) -> bool:
        """
        导出缩略图
        
        Args:
            painting_animation: 绘画动画对象
            output_path (str): 输出路径
            size (Tuple[int, int]): 缩略图大小
            
        Returns:
            bool: 是否成功
        """
        try:
            # 选择中间帧作为缩略图
            frame_index = len(painting_animation.all_frames) // 2
            
            # 创建渲染设置
            render_settings = RenderSettings(
                output_format='png',
                resolution=size,
                fps=30,
                quality='medium',
                background_color=self.background_color,
                show_brush=False,
                show_trajectory=False,
                show_progress=False,
                enable_effects=True,
                codec='png'
            )
            
            # 渲染缩略图
            thumbnail = self.create_preview_frame(painting_animation, frame_index, render_settings)
            
            # 保存缩略图
            cv2.imwrite(output_path, thumbnail)
            
            self.logger.info(f"Thumbnail exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting thumbnail: {str(e)}")
            return False
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """
        获取渲染统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            if self.render_progress is None:
                return {'status': 'not_started'}
            
            stats = {
                'progress': {
                    'current_frame': self.render_progress.current_frame,
                    'total_frames': self.render_progress.total_frames,
                    'current_stroke': self.render_progress.current_stroke,
                    'total_strokes': self.render_progress.total_strokes,
                    'percentage': self.render_progress.progress_percentage,
                    'status': self.render_progress.status
                },
                'timing': {
                    'elapsed_time': self.render_progress.elapsed_time,
                    'estimated_time': self.render_progress.estimated_time
                },
                'settings': {
                    'resolution': self.default_resolution,
                    'fps': self.default_fps,
                    'format': self.default_format,
                    'codec': self.default_codec
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting render statistics: {str(e)}")
            return {'error': str(e)}