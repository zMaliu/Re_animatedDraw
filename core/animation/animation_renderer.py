# -*- coding: utf-8 -*-
"""
动画渲染器模块

提供高质量的动画渲染功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from enum import Enum
import cv2
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .painting_animator import AnimationFrame, AnimationSequence
from ..stroke_extraction.stroke_detector import Stroke


class RenderFormat(Enum):
    """渲染格式"""
    MP4 = "mp4"
    AVI = "avi"
    GIF = "gif"
    WEBM = "webm"
    MOV = "mov"
    FRAMES = "frames"  # 输出单独帧


class RenderQuality(Enum):
    """渲染质量"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class RenderConfig:
    """渲染配置"""
    output_path: str
    format: RenderFormat = RenderFormat.MP4
    quality: RenderQuality = RenderQuality.HIGH
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    bitrate: str = "5000k"
    codec: str = "h264"
    background_color: Tuple[int, int, int] = (255, 255, 255)
    enable_antialiasing: bool = True
    enable_motion_blur: bool = False
    motion_blur_strength: float = 0.5
    enable_progressive_jpeg: bool = True
    compression_level: int = 6
    enable_multithreading: bool = True
    max_threads: int = 4


@dataclass
class RenderProgress:
    """渲染进度"""
    current_frame: int
    total_frames: int
    elapsed_time: float
    estimated_remaining: float
    fps_actual: float
    memory_usage: float
    status: str


@dataclass
class RenderResult:
    """渲染结果"""
    success: bool
    output_path: str
    total_frames: int
    duration: float
    file_size: int
    average_fps: float
    error_message: Optional[str] = None
    warnings: List[str] = None


class AnimationRenderer:
    """动画渲染器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动画渲染器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 默认渲染配置
        self.default_config = RenderConfig(
            output_path=config.get('output_path', 'output.mp4'),
            format=RenderFormat(config.get('format', 'mp4')),
            quality=RenderQuality(config.get('quality', 'high')),
            fps=config.get('fps', 30),
            resolution=tuple(config.get('resolution', [1920, 1080])),
            bitrate=config.get('bitrate', '5000k'),
            codec=config.get('codec', 'h264'),
            background_color=tuple(config.get('background_color', [255, 255, 255])),
            enable_antialiasing=config.get('enable_antialiasing', True),
            enable_motion_blur=config.get('enable_motion_blur', False),
            motion_blur_strength=config.get('motion_blur_strength', 0.5),
            enable_multithreading=config.get('enable_multithreading', True),
            max_threads=config.get('max_threads', 4)
        )
        
        # 渲染状态
        self.is_rendering = False
        self.current_progress = None
        self.render_lock = threading.Lock()
        
        # 性能优化
        self.frame_cache = {}
        self.enable_frame_caching = config.get('enable_frame_caching', True)
        self.max_cache_size = config.get('max_cache_size', 100)
        
        # 质量设置映射
        self.quality_settings = {
            RenderQuality.LOW: {
                'crf': 28,
                'preset': 'ultrafast',
                'scale_factor': 0.5
            },
            RenderQuality.MEDIUM: {
                'crf': 23,
                'preset': 'fast',
                'scale_factor': 0.75
            },
            RenderQuality.HIGH: {
                'crf': 18,
                'preset': 'medium',
                'scale_factor': 1.0
            },
            RenderQuality.ULTRA: {
                'crf': 15,
                'preset': 'slow',
                'scale_factor': 1.0
            }
        }
        
    def render_animation(self, animation_sequence: AnimationSequence, 
                        render_config: RenderConfig = None,
                        progress_callback: callable = None) -> RenderResult:
        """
        渲染动画序列
        
        Args:
            animation_sequence: 动画序列
            render_config: 渲染配置
            progress_callback: 进度回调函数
            
        Returns:
            渲染结果
        """
        with self.render_lock:
            if self.is_rendering:
                return RenderResult(
                    success=False,
                    output_path="",
                    total_frames=0,
                    duration=0,
                    file_size=0,
                    average_fps=0,
                    error_message="Another rendering is in progress"
                )
            
            self.is_rendering = True
        
        try:
            config = render_config or self.default_config
            
            # 验证配置
            validation_result = self._validate_config(config)
            if not validation_result[0]:
                return RenderResult(
                    success=False,
                    output_path="",
                    total_frames=0,
                    duration=0,
                    file_size=0,
                    average_fps=0,
                    error_message=validation_result[1]
                )
            
            # 准备输出目录
            output_dir = os.path.dirname(config.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 根据格式选择渲染方法
            if config.format == RenderFormat.FRAMES:
                return self._render_frames(animation_sequence, config, progress_callback)
            elif config.format == RenderFormat.GIF:
                return self._render_gif(animation_sequence, config, progress_callback)
            else:
                return self._render_video(animation_sequence, config, progress_callback)
                
        except Exception as e:
            self.logger.error(f"Error rendering animation: {e}")
            return RenderResult(
                success=False,
                output_path="",
                total_frames=0,
                duration=0,
                file_size=0,
                average_fps=0,
                error_message=str(e)
            )
        finally:
            self.is_rendering = False
    
    def render_frame(self, frame: AnimationFrame, config: RenderConfig = None) -> np.ndarray:
        """
        渲染单个帧
        
        Args:
            frame: 动画帧
            config: 渲染配置
            
        Returns:
            渲染后的图像
        """
        config = config or self.default_config
        
        try:
            # 检查缓存
            cache_key = self._get_frame_cache_key(frame, config)
            if self.enable_frame_caching and cache_key in self.frame_cache:
                return self.frame_cache[cache_key].copy()
            
            # 如果frame有canvas_state属性，直接使用它
            if hasattr(frame, 'canvas_state') and frame.canvas_state is not None:
                canvas = frame.canvas_state.copy()
                # 应用后处理效果
                canvas = self._apply_post_processing(canvas, config)
            else:
                # 创建画布
                canvas = self._create_canvas(config)
                
                # 如果有active_strokes，尝试渲染（这需要额外的笔触数据）
                # 由于AnimationFrame不包含完整的笔触数据，这里只返回基础画布
                if hasattr(frame, 'active_strokes') and frame.active_strokes:
                    # 这里可以添加基于active_strokes的渲染逻辑
                    # 但需要额外的笔触数据源
                    pass
                
                # 应用后处理效果
                canvas = self._apply_post_processing(canvas, config)
            
            # 缓存结果
            if self.enable_frame_caching:
                self._cache_frame(cache_key, canvas)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error rendering frame: {e}")
            return self._create_canvas(config)
    
    def _render_video(self, animation_sequence: AnimationSequence, 
                     config: RenderConfig, progress_callback: callable) -> RenderResult:
        """
        渲染视频
        
        Args:
            animation_sequence: 动画序列
            config: 渲染配置
            progress_callback: 进度回调
            
        Returns:
            渲染结果
        """
        import time
        start_time = time.time()
        
        # 设置视频编写器
        fourcc = self._get_fourcc(config.codec)
        quality_settings = self.quality_settings[config.quality]
        
        # 调整分辨率
        actual_resolution = (
            int(config.resolution[0] * quality_settings['scale_factor']),
            int(config.resolution[1] * quality_settings['scale_factor'])
        )
        
        video_writer = cv2.VideoWriter(
            config.output_path,
            fourcc,
            config.fps,
            actual_resolution
        )
        
        if not video_writer.isOpened():
            return RenderResult(
                success=False,
                output_path="",
                total_frames=0,
                duration=0,
                file_size=0,
                average_fps=0,
                error_message="Failed to open video writer"
            )
        
        total_frames = len(animation_sequence.frames)
        warnings = []
        
        try:
            # 渲染帧
            if config.enable_multithreading and total_frames > 10:
                # 多线程渲染
                rendered_frames = self._render_frames_parallel(
                    animation_sequence.frames, config, progress_callback
                )
            else:
                # 单线程渲染
                rendered_frames = self._render_frames_sequential(
                    animation_sequence.frames, config, progress_callback
                )
            
            # 写入视频
            for i, frame_image in enumerate(rendered_frames):
                if frame_image is None:
                    warnings.append(f"Frame {i} failed to render")
                    frame_image = self._create_canvas(config)
                
                # 调整帧大小
                current_resolution = (frame_image.shape[1], frame_image.shape[0])  # (width, height)
                if current_resolution != tuple(actual_resolution):
                    frame_image = cv2.resize(frame_image, actual_resolution)
                
                # 应用运动模糊
                if config.enable_motion_blur and i > 0:
                    frame_image = self._apply_motion_blur(
                        frame_image, rendered_frames[i-1], config.motion_blur_strength
                    )
                
                video_writer.write(frame_image)
                
                # 更新进度
                if progress_callback:
                    progress = RenderProgress(
                        current_frame=i + 1,
                        total_frames=total_frames,
                        elapsed_time=time.time() - start_time,
                        estimated_remaining=0,  # 计算剩余时间
                        fps_actual=0,  # 计算实际FPS
                        memory_usage=0,  # 计算内存使用
                        status="Writing video"
                    )
                    progress_callback(progress)
            
            video_writer.release()
            
            # 计算结果统计
            end_time = time.time()
            duration = end_time - start_time
            file_size = os.path.getsize(config.output_path) if os.path.exists(config.output_path) else 0
            average_fps = total_frames / duration if duration > 0 else 0
            
            return RenderResult(
                success=True,
                output_path=config.output_path,
                total_frames=total_frames,
                duration=duration,
                file_size=file_size,
                average_fps=average_fps,
                warnings=warnings if warnings else None
            )
            
        except Exception as e:
            video_writer.release()
            raise e
    
    def _render_gif(self, animation_sequence: AnimationSequence, 
                   config: RenderConfig, progress_callback: callable) -> RenderResult:
        """
        渲染GIF
        
        Args:
            animation_sequence: 动画序列
            config: 渲染配置
            progress_callback: 进度回调
            
        Returns:
            渲染结果
        """
        try:
            from PIL import Image
        except ImportError:
            return RenderResult(
                success=False,
                output_path="",
                total_frames=0,
                duration=0,
                file_size=0,
                average_fps=0,
                error_message="PIL is required for GIF rendering"
            )
        
        import time
        start_time = time.time()
        
        total_frames = len(animation_sequence.frames)
        pil_images = []
        
        # 渲染所有帧
        for i, frame in enumerate(animation_sequence.frames):
            # 渲染帧
            frame_image = self.render_frame(frame, config)
            
            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 优化GIF大小
            if config.quality != RenderQuality.ULTRA:
                # 减少颜色数量
                pil_image = pil_image.quantize(colors=256)
            
            pil_images.append(pil_image)
            
            # 更新进度
            if progress_callback:
                progress = RenderProgress(
                    current_frame=i + 1,
                    total_frames=total_frames,
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=0,
                    fps_actual=0,
                    memory_usage=0,
                    status="Rendering GIF frames"
                )
                progress_callback(progress)
        
        # 保存GIF
        if pil_images:
            duration_per_frame = int(1000 / config.fps)  # 毫秒
            
            pil_images[0].save(
                config.output_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration_per_frame,
                loop=0,
                optimize=True
            )
        
        # 计算结果统计
        end_time = time.time()
        duration = end_time - start_time
        file_size = os.path.getsize(config.output_path) if os.path.exists(config.output_path) else 0
        average_fps = total_frames / duration if duration > 0 else 0
        
        return RenderResult(
            success=True,
            output_path=config.output_path,
            total_frames=total_frames,
            duration=duration,
            file_size=file_size,
            average_fps=average_fps
        )
    
    def _render_frames(self, animation_sequence: AnimationSequence, 
                      config: RenderConfig, progress_callback: callable) -> RenderResult:
        """
        渲染单独帧
        
        Args:
            animation_sequence: 动画序列
            config: 渲染配置
            progress_callback: 进度回调
            
        Returns:
            渲染结果
        """
        import time
        start_time = time.time()
        
        # 创建输出目录
        output_dir = config.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        total_frames = len(animation_sequence.frames)
        
        # 渲染每一帧
        for i, frame in enumerate(animation_sequence.frames):
            # 渲染帧
            frame_image = self.render_frame(frame, config)
            
            # 保存帧
            frame_filename = f"frame_{i:06d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame_image)
            
            # 更新进度
            if progress_callback:
                progress = RenderProgress(
                    current_frame=i + 1,
                    total_frames=total_frames,
                    elapsed_time=time.time() - start_time,
                    estimated_remaining=0,
                    fps_actual=0,
                    memory_usage=0,
                    status="Saving frames"
                )
                progress_callback(progress)
        
        # 计算结果统计
        end_time = time.time()
        duration = end_time - start_time
        
        # 计算总文件大小
        total_size = 0
        for filename in os.listdir(output_dir):
            if filename.startswith("frame_") and filename.endswith(".png"):
                total_size += os.path.getsize(os.path.join(output_dir, filename))
        
        average_fps = total_frames / duration if duration > 0 else 0
        
        return RenderResult(
            success=True,
            output_path=output_dir,
            total_frames=total_frames,
            duration=duration,
            file_size=total_size,
            average_fps=average_fps
        )
    
    def _render_frames_parallel(self, frames: List[AnimationFrame], 
                               config: RenderConfig, progress_callback: callable) -> List[np.ndarray]:
        """
        并行渲染帧
        
        Args:
            frames: 帧列表
            config: 渲染配置
            progress_callback: 进度回调
            
        Returns:
            渲染后的图像列表
        """
        rendered_frames = [None] * len(frames)
        
        with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
            # 提交渲染任务
            future_to_index = {
                executor.submit(self.render_frame, frame, config): i 
                for i, frame in enumerate(frames)
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    rendered_frames[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Error rendering frame {index}: {e}")
                    rendered_frames[index] = None
                
                completed += 1
                
                # 更新进度
                if progress_callback:
                    progress = RenderProgress(
                        current_frame=completed,
                        total_frames=len(frames),
                        elapsed_time=0,
                        estimated_remaining=0,
                        fps_actual=0,
                        memory_usage=0,
                        status="Rendering frames"
                    )
                    progress_callback(progress)
        
        return rendered_frames
    
    def _render_frames_sequential(self, frames: List[AnimationFrame], 
                                 config: RenderConfig, progress_callback: callable) -> List[np.ndarray]:
        """
        顺序渲染帧
        
        Args:
            frames: 帧列表
            config: 渲染配置
            progress_callback: 进度回调
            
        Returns:
            渲染后的图像列表
        """
        rendered_frames = []
        
        for i, frame in enumerate(frames):
            try:
                rendered_frame = self.render_frame(frame, config)
                rendered_frames.append(rendered_frame)
            except Exception as e:
                self.logger.error(f"Error rendering frame {i}: {e}")
                rendered_frames.append(None)
            
            # 更新进度
            if progress_callback:
                progress = RenderProgress(
                    current_frame=i + 1,
                    total_frames=len(frames),
                    elapsed_time=0,
                    estimated_remaining=0,
                    fps_actual=0,
                    memory_usage=0,
                    status="Rendering frames"
                )
                progress_callback(progress)
        
        return rendered_frames
    
    def _create_canvas(self, config: RenderConfig) -> np.ndarray:
        """
        创建画布
        
        Args:
            config: 渲染配置
            
        Returns:
            画布图像
        """
        quality_settings = self.quality_settings[config.quality]
        actual_resolution = (
            int(config.resolution[1] * quality_settings['scale_factor']),  # height
            int(config.resolution[0] * quality_settings['scale_factor']),  # width
            3  # channels
        )
        
        canvas = np.full(actual_resolution, config.background_color, dtype=np.uint8)
        return canvas
    
    def _render_stroke_on_canvas(self, canvas: np.ndarray, stroke_data: Dict[str, Any], 
                                config: RenderConfig) -> np.ndarray:
        """
        在画布上渲染笔触
        
        Args:
            canvas: 画布
            stroke_data: 笔触数据
            config: 渲染配置
            
        Returns:
            更新后的画布
        """
        # 这里应该调用笔刷模拟器来渲染笔触
        # 为了简化，这里只是绘制简单的线条
        
        if 'points' in stroke_data and len(stroke_data['points']) > 1:
            points = stroke_data['points']
            color = stroke_data.get('color', (0, 0, 0))
            thickness = stroke_data.get('thickness', 2)
            
            # 绘制线条
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                
                if config.enable_antialiasing:
                    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
                else:
                    cv2.line(canvas, pt1, pt2, color, thickness)
        
        return canvas
    
    def _apply_post_processing(self, image: np.ndarray, config: RenderConfig) -> np.ndarray:
        """
        应用后处理效果
        
        Args:
            image: 输入图像
            config: 渲染配置
            
        Returns:
            处理后的图像
        """
        result = image.copy()
        
        # 抗锯齿
        if config.enable_antialiasing and config.quality in [RenderQuality.HIGH, RenderQuality.ULTRA]:
            result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        return result
    
    def _apply_motion_blur(self, current_frame: np.ndarray, previous_frame: np.ndarray, 
                          strength: float) -> np.ndarray:
        """
        应用运动模糊
        
        Args:
            current_frame: 当前帧
            previous_frame: 前一帧
            strength: 模糊强度
            
        Returns:
            应用运动模糊后的帧
        """
        if previous_frame is None:
            return current_frame
        
        # 简单的运动模糊实现
        alpha = 1.0 - strength
        blurred = cv2.addWeighted(current_frame, alpha, previous_frame, strength, 0)
        return blurred
    
    def _get_fourcc(self, codec: str) -> int:
        """
        获取视频编码器
        
        Args:
            codec: 编码器名称
            
        Returns:
            FourCC代码
        """
        codec_map = {
            'h264': cv2.VideoWriter_fourcc(*'H264'),
            'xvid': cv2.VideoWriter_fourcc(*'XVID'),
            'mjpg': cv2.VideoWriter_fourcc(*'MJPG'),
            'mp4v': cv2.VideoWriter_fourcc(*'MP4V')
        }
        
        return codec_map.get(codec.lower(), cv2.VideoWriter_fourcc(*'H264'))
    
    def _validate_config(self, config: RenderConfig) -> Tuple[bool, str]:
        """
        验证渲染配置
        
        Args:
            config: 渲染配置
            
        Returns:
            (是否有效, 错误信息)
        """
        if config.fps <= 0:
            return False, "FPS must be positive"
        
        if config.resolution[0] <= 0 or config.resolution[1] <= 0:
            return False, "Resolution must be positive"
        
        if not config.output_path:
            return False, "Output path is required"
        
        # 检查输出目录是否可写
        output_dir = os.path.dirname(config.output_path)
        if output_dir and not os.access(output_dir, os.W_OK):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                return False, f"Cannot write to output directory: {output_dir}"
        
        return True, ""
    
    def _get_frame_cache_key(self, frame: AnimationFrame, config: RenderConfig) -> str:
        """
        生成帧缓存键
        
        Args:
            frame: 动画帧
            config: 渲染配置
            
        Returns:
            缓存键
        """
        # 简化的缓存键生成
        key_parts = [
            str(frame.timestamp),
            str(len(frame.active_strokes) if hasattr(frame, 'active_strokes') and frame.active_strokes else 0),
            str(config.resolution),
            str(config.quality.value)
        ]
        return "_".join(key_parts)
    
    def _cache_frame(self, cache_key: str, frame_image: np.ndarray):
        """
        缓存帧
        
        Args:
            cache_key: 缓存键
            frame_image: 帧图像
        """
        if len(self.frame_cache) >= self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
        
        self.frame_cache[cache_key] = frame_image.copy()
    
    def get_render_progress(self) -> Optional[RenderProgress]:
        """
        获取当前渲染进度
        
        Returns:
            渲染进度
        """
        return self.current_progress
    
    def is_rendering_active(self) -> bool:
        """
        检查是否正在渲染
        
        Returns:
            是否正在渲染
        """
        return self.is_rendering
    
    def cancel_rendering(self):
        """取消当前渲染"""
        # 这里应该实现渲染取消逻辑
        self.logger.info("Rendering cancellation requested")
    
    def clear_frame_cache(self):
        """清除帧缓存"""
        self.frame_cache.clear()
        self.logger.info("Frame cache cleared")
    
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的输出格式
        
        Returns:
            支持的格式列表
        """
        return [format.value for format in RenderFormat]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计信息
        """
        return {
            'is_rendering': self.is_rendering,
            'frame_cache_size': len(self.frame_cache),
            'max_cache_size': self.max_cache_size,
            'enable_frame_caching': self.enable_frame_caching,
            'multithreading_enabled': self.default_config.enable_multithreading,
            'max_threads': self.default_config.max_threads
        }