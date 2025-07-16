# -*- coding: utf-8 -*-
"""
动画控制器

实现论文中的动画控制功能：
1. 动画播放控制
2. 速度和时间控制
3. 交互式控制
4. 动画状态管理
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from queue import Queue, Empty
from .stroke_animator import StrokeAnimator, AnimationFrame, AnimationConfig
from .flood_renderer import FloodRenderer, FloodParameters
from .direction_detector import DirectionDetector


class PlaybackState(Enum):
    """
    播放状态枚举
    """
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    LOADING = "loading"


class ControlMode(Enum):
    """
    控制模式枚举
    """
    AUTOMATIC = "automatic"      # 自动播放
    MANUAL = "manual"            # 手动控制
    INTERACTIVE = "interactive"  # 交互式
    STEP_BY_STEP = "step_by_step" # 逐步播放


@dataclass
class PlaybackSettings:
    """
    播放设置
    """
    speed: float = 1.0           # 播放速度倍率
    loop: bool = False           # 是否循环播放
    auto_pause: bool = False     # 自动暂停
    frame_skip: int = 0          # 跳帧数
    quality: str = "high"        # 播放质量
    
    # 交互设置
    enable_scrubbing: bool = True    # 启用拖拽
    enable_keyboard: bool = True     # 启用键盘控制
    enable_mouse: bool = True        # 启用鼠标控制
    
    # 性能设置
    max_fps: int = 60               # 最大帧率
    buffer_size: int = 100          # 缓冲区大小
    preload_frames: int = 10        # 预加载帧数


@dataclass
class AnimationState:
    """
    动画状态
    """
    current_frame: int = 0
    total_frames: int = 0
    playback_state: PlaybackState = PlaybackState.STOPPED
    current_time: float = 0.0
    total_duration: float = 0.0
    fps: float = 30.0
    
    # 播放统计
    frames_rendered: int = 0
    dropped_frames: int = 0
    average_fps: float = 0.0
    
    # 缓冲状态
    buffer_level: float = 0.0
    is_buffering: bool = False


@dataclass
class ControlEvent:
    """
    控制事件
    """
    event_type: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class AnimationController:
    """
    动画控制器
    
    负责管理和控制笔触动画的播放
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化控制器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.animator = StrokeAnimator(AnimationConfig(**config.get('animation', {})))
        self.flood_renderer = FloodRenderer(config.get('flood_renderer', {}))
        self.direction_detector = DirectionDetector(config.get('direction_detector', {}))
        
        # 状态管理
        self.state = AnimationState()
        self.settings = PlaybackSettings(**config.get('playback', {}))
        self.control_mode = ControlMode.AUTOMATIC
        
        # 动画数据
        self.frames: List[AnimationFrame] = []
        self.strokes: List[Dict[str, Any]] = []
        self.stroke_order: List[int] = []
        
        # 控制线程
        self.playback_thread: Optional[threading.Thread] = None
        self.control_queue = Queue()
        self.frame_buffer = Queue(maxsize=self.settings.buffer_size)
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            'frame_update': [],
            'playback_start': [],
            'playback_stop': [],
            'playback_pause': [],
            'playback_resume': [],
            'seek_complete': [],
            'animation_complete': [],
            'error': []
        }
        
        # 性能监控
        self.performance_stats = {
            'frame_times': [],
            'render_times': [],
            'total_render_time': 0.0,
            'memory_usage': 0.0
        }
        
        # 交互状态
        self.is_scrubbing = False
        self.last_interaction_time = 0.0
        
        self.logger.info("Animation controller initialized")
    
    def load_animation(self, strokes: List[Dict[str, Any]], stroke_order: List[int]) -> bool:
        """
        加载动画数据
        
        Args:
            strokes: 笔触列表
            stroke_order: 笔触顺序
            
        Returns:
            bool: 是否成功加载
        """
        try:
            self.logger.info(f"Loading animation with {len(strokes)} strokes")
            
            # 更新状态
            self.state.playback_state = PlaybackState.LOADING
            
            # 保存数据
            self.strokes = strokes
            self.stroke_order = stroke_order
            
            # 生成动画帧
            self.frames = self.animator.create_animation(strokes, stroke_order)
            
            # 更新状态信息
            self.state.total_frames = len(self.frames)
            self.state.total_duration = self.state.total_frames / self.animator.config.fps
            self.state.fps = self.animator.config.fps
            self.state.current_frame = 0
            self.state.current_time = 0.0
            self.state.playback_state = PlaybackState.STOPPED
            
            self.logger.info(f"Animation loaded: {self.state.total_frames} frames, {self.state.total_duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading animation: {str(e)}")
            self.state.playback_state = PlaybackState.STOPPED
            self._trigger_event('error', {'message': str(e)})
            return False
    
    def play(self, start_frame: Optional[int] = None) -> bool:
        """
        开始播放动画
        
        Args:
            start_frame: 起始帧（可选）
            
        Returns:
            bool: 是否成功开始播放
        """
        if not self.frames:
            self.logger.warning("No animation loaded")
            return False
        
        if self.state.playback_state == PlaybackState.PLAYING:
            self.logger.info("Animation is already playing")
            return True
        
        try:
            # 设置起始帧
            if start_frame is not None:
                self.seek_to_frame(start_frame)
            
            # 更新状态
            self.state.playback_state = PlaybackState.PLAYING
            
            # 启动播放线程
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
                self.playback_thread.start()
            
            self._trigger_event('playback_start', {'frame': self.state.current_frame})
            self.logger.info(f"Animation playback started from frame {self.state.current_frame}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting playback: {str(e)}")
            self.state.playback_state = PlaybackState.STOPPED
            self._trigger_event('error', {'message': str(e)})
            return False
    
    def pause(self) -> bool:
        """
        暂停播放
        
        Returns:
            bool: 是否成功暂停
        """
        if self.state.playback_state != PlaybackState.PLAYING:
            return False
        
        self.state.playback_state = PlaybackState.PAUSED
        self._trigger_event('playback_pause', {'frame': self.state.current_frame})
        self.logger.info("Animation playback paused")
        return True
    
    def resume(self) -> bool:
        """
        恢复播放
        
        Returns:
            bool: 是否成功恢复
        """
        if self.state.playback_state != PlaybackState.PAUSED:
            return False
        
        self.state.playback_state = PlaybackState.PLAYING
        self._trigger_event('playback_resume', {'frame': self.state.current_frame})
        self.logger.info("Animation playback resumed")
        return True
    
    def stop(self) -> bool:
        """
        停止播放
        
        Returns:
            bool: 是否成功停止
        """
        if self.state.playback_state == PlaybackState.STOPPED:
            return True
        
        self.state.playback_state = PlaybackState.STOPPED
        self.state.current_frame = 0
        self.state.current_time = 0.0
        
        self._trigger_event('playback_stop', {'frame': self.state.current_frame})
        self.logger.info("Animation playback stopped")
        return True
    
    def seek_to_frame(self, frame_id: int) -> bool:
        """
        跳转到指定帧
        
        Args:
            frame_id: 目标帧ID
            
        Returns:
            bool: 是否成功跳转
        """
        if not self.frames or frame_id < 0 or frame_id >= len(self.frames):
            return False
        
        try:
            self.state.playback_state = PlaybackState.SEEKING
            self.state.current_frame = frame_id
            self.state.current_time = frame_id / self.state.fps
            
            # 触发帧更新事件
            current_frame = self.frames[frame_id]
            self._trigger_event('frame_update', {
                'frame': current_frame,
                'frame_id': frame_id,
                'timestamp': self.state.current_time
            })
            
            self._trigger_event('seek_complete', {'frame': frame_id})
            
            # 恢复之前的播放状态
            if hasattr(self, '_previous_state'):
                self.state.playback_state = self._previous_state
            else:
                self.state.playback_state = PlaybackState.STOPPED
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error seeking to frame {frame_id}: {str(e)}")
            return False
    
    def seek_to_time(self, timestamp: float) -> bool:
        """
        跳转到指定时间
        
        Args:
            timestamp: 目标时间（秒）
            
        Returns:
            bool: 是否成功跳转
        """
        if timestamp < 0 or timestamp > self.state.total_duration:
            return False
        
        frame_id = int(timestamp * self.state.fps)
        return self.seek_to_frame(frame_id)
    
    def set_playback_speed(self, speed: float) -> bool:
        """
        设置播放速度
        
        Args:
            speed: 播放速度倍率
            
        Returns:
            bool: 是否成功设置
        """
        if speed <= 0:
            return False
        
        self.settings.speed = speed
        self.logger.info(f"Playback speed set to {speed}x")
        return True
    
    def set_control_mode(self, mode: ControlMode) -> bool:
        """
        设置控制模式
        
        Args:
            mode: 控制模式
            
        Returns:
            bool: 是否成功设置
        """
        self.control_mode = mode
        self.logger.info(f"Control mode set to {mode.value}")
        return True
    
    def step_forward(self) -> bool:
        """
        前进一帧
        
        Returns:
            bool: 是否成功前进
        """
        if self.state.current_frame < self.state.total_frames - 1:
            return self.seek_to_frame(self.state.current_frame + 1)
        return False
    
    def step_backward(self) -> bool:
        """
        后退一帧
        
        Returns:
            bool: 是否成功后退
        """
        if self.state.current_frame > 0:
            return self.seek_to_frame(self.state.current_frame - 1)
        return False
    
    def _playback_loop(self):
        """
        播放循环（在单独线程中运行）
        """
        last_frame_time = time.time()
        frame_duration = 1.0 / (self.state.fps * self.settings.speed)
        
        while self.state.playback_state in [PlaybackState.PLAYING, PlaybackState.PAUSED]:
            current_time = time.time()
            
            # 处理控制事件
            self._process_control_events()
            
            # 如果暂停，等待
            if self.state.playback_state == PlaybackState.PAUSED:
                time.sleep(0.01)
                last_frame_time = current_time
                continue
            
            # 检查是否到了下一帧的时间
            if current_time - last_frame_time >= frame_duration:
                # 渲染当前帧
                if self._render_current_frame():
                    self.state.current_frame += 1
                    self.state.current_time = self.state.current_frame / self.state.fps
                    
                    # 更新性能统计
                    self.state.frames_rendered += 1
                    frame_time = current_time - last_frame_time
                    self.performance_stats['frame_times'].append(frame_time)
                    
                    # 保持最近100帧的统计
                    if len(self.performance_stats['frame_times']) > 100:
                        self.performance_stats['frame_times'].pop(0)
                    
                    # 计算平均FPS
                    if self.performance_stats['frame_times']:
                        avg_frame_time = np.mean(self.performance_stats['frame_times'])
                        self.state.average_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    
                    last_frame_time = current_time
                else:
                    # 渲染失败，记录丢帧
                    self.state.dropped_frames += 1
                
                # 检查是否播放完成
                if self.state.current_frame >= self.state.total_frames:
                    if self.settings.loop:
                        self.state.current_frame = 0
                        self.state.current_time = 0.0
                    else:
                        self.state.playback_state = PlaybackState.STOPPED
                        self._trigger_event('animation_complete', {})
                        break
            else:
                # 等待下一帧时间
                sleep_time = frame_duration - (current_time - last_frame_time)
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.001))  # 最多等待1ms
    
    def _render_current_frame(self) -> bool:
        """
        渲染当前帧
        
        Returns:
            bool: 是否成功渲染
        """
        try:
            if self.state.current_frame >= len(self.frames):
                return False
            
            start_time = time.time()
            
            # 获取当前帧
            current_frame = self.frames[self.state.current_frame]
            
            # 触发帧更新事件
            self._trigger_event('frame_update', {
                'frame': current_frame,
                'frame_id': self.state.current_frame,
                'timestamp': self.state.current_time
            })
            
            # 记录渲染时间
            render_time = time.time() - start_time
            self.performance_stats['render_times'].append(render_time)
            self.performance_stats['total_render_time'] += render_time
            
            # 保持最近100帧的渲染时间统计
            if len(self.performance_stats['render_times']) > 100:
                self.performance_stats['render_times'].pop(0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rendering frame {self.state.current_frame}: {str(e)}")
            return False
    
    def _process_control_events(self):
        """
        处理控制事件
        """
        try:
            while True:
                event = self.control_queue.get_nowait()
                self._handle_control_event(event)
        except Empty:
            pass
    
    def _handle_control_event(self, event: ControlEvent):
        """
        处理单个控制事件
        
        Args:
            event: 控制事件
        """
        try:
            if event.event_type == 'seek':
                frame_id = event.data.get('frame_id')
                if frame_id is not None:
                    self.seek_to_frame(frame_id)
            
            elif event.event_type == 'speed_change':
                speed = event.data.get('speed')
                if speed is not None:
                    self.set_playback_speed(speed)
            
            elif event.event_type == 'pause':
                self.pause()
            
            elif event.event_type == 'resume':
                self.resume()
            
            elif event.event_type == 'stop':
                self.stop()
            
        except Exception as e:
            self.logger.error(f"Error handling control event: {str(e)}")
    
    def send_control_event(self, event_type: str, data: Dict[str, Any] = None):
        """
        发送控制事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if data is None:
            data = {}
        
        event = ControlEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        
        try:
            self.control_queue.put_nowait(event)
        except:
            self.logger.warning(f"Control queue full, dropping event: {event_type}")
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """
        添加事件回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def remove_event_callback(self, event_type: str, callback: Callable):
        """
        移除事件回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_callbacks:
            try:
                self.event_callbacks[event_type].remove(callback)
            except ValueError:
                pass
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """
        触发事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event callback for {event_type}: {str(e)}")
    
    def get_current_frame(self) -> Optional[AnimationFrame]:
        """
        获取当前帧
        
        Returns:
            Optional[AnimationFrame]: 当前帧
        """
        if 0 <= self.state.current_frame < len(self.frames):
            return self.frames[self.state.current_frame]
        return None
    
    def get_animation_state(self) -> AnimationState:
        """
        获取动画状态
        
        Returns:
            AnimationState: 动画状态
        """
        return self.state
    
    def get_playback_settings(self) -> PlaybackSettings:
        """
        获取播放设置
        
        Returns:
            PlaybackSettings: 播放设置
        """
        return self.settings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            Dict[str, Any]: 性能统计
        """
        stats = self.performance_stats.copy()
        
        # 计算统计值
        if stats['frame_times']:
            stats['average_frame_time'] = np.mean(stats['frame_times'])
            stats['min_frame_time'] = np.min(stats['frame_times'])
            stats['max_frame_time'] = np.max(stats['frame_times'])
        
        if stats['render_times']:
            stats['average_render_time'] = np.mean(stats['render_times'])
            stats['min_render_time'] = np.min(stats['render_times'])
            stats['max_render_time'] = np.max(stats['render_times'])
        
        # 添加状态信息
        stats['current_fps'] = self.state.average_fps
        stats['frames_rendered'] = self.state.frames_rendered
        stats['dropped_frames'] = self.state.dropped_frames
        stats['drop_rate'] = (self.state.dropped_frames / 
                             max(1, self.state.frames_rendered + self.state.dropped_frames))
        
        return stats
    
    def export_animation_data(self) -> Dict[str, Any]:
        """
        导出动画数据
        
        Returns:
            Dict[str, Any]: 动画数据
        """
        return {
            'strokes': self.strokes,
            'stroke_order': self.stroke_order,
            'total_frames': self.state.total_frames,
            'total_duration': self.state.total_duration,
            'fps': self.state.fps,
            'config': self.config,
            'performance_stats': self.get_performance_stats()
        }
    
    def cleanup(self):
        """
        清理资源
        """
        # 停止播放
        self.stop()
        
        # 等待播放线程结束
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)
        
        # 清空队列
        while not self.control_queue.empty():
            try:
                self.control_queue.get_nowait()
            except Empty:
                break
        
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except Empty:
                break
        
        # 清空回调
        for callbacks in self.event_callbacks.values():
            callbacks.clear()
        
        self.logger.info("Animation controller cleaned up")
    
    def __del__(self):
        """
        析构函数
        """
        try:
            self.cleanup()
        except:
            pass