# -*- coding: utf-8 -*-
"""
时间轴管理器

管理绘画动画的时间轴、同步和播放控制
包括时间插值、事件调度、播放控制等功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
import math
from enum import Enum
import bisect
from scipy.interpolate import interp1d
import json


class PlaybackState(Enum):
    """
    播放状态枚举
    """
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    BUFFERING = "buffering"


class TimelineEventType(Enum):
    """
    时间轴事件类型枚举
    """
    STROKE_START = "stroke_start"
    STROKE_END = "stroke_end"
    PAUSE = "pause"
    SPEED_CHANGE = "speed_change"
    MARKER = "marker"
    CUSTOM = "custom"


@dataclass
class TimelineEvent:
    """
    时间轴事件数据结构
    
    Attributes:
        event_id (str): 事件ID
        timestamp (float): 时间戳
        event_type (TimelineEventType): 事件类型
        data (Dict): 事件数据
        callback (Optional[Callable]): 回调函数
        enabled (bool): 是否启用
    """
    event_id: str
    timestamp: float
    event_type: TimelineEventType
    data: Dict[str, Any]
    callback: Optional[Callable]
    enabled: bool = True


@dataclass
class TimelineMarker:
    """
    时间轴标记数据结构
    
    Attributes:
        marker_id (str): 标记ID
        timestamp (float): 时间戳
        name (str): 标记名称
        description (str): 描述
        color (Tuple[int, int, int]): 颜色
        metadata (Dict): 元数据
    """
    marker_id: str
    timestamp: float
    name: str
    description: str
    color: Tuple[int, int, int]
    metadata: Dict[str, Any]


@dataclass
class PlaybackSettings:
    """
    播放设置数据结构
    
    Attributes:
        speed (float): 播放速度
        loop (bool): 是否循环
        auto_pause_on_stroke (bool): 笔画结束时自动暂停
        smooth_seeking (bool): 平滑跳转
        buffer_size (float): 缓冲大小（秒）
        sync_tolerance (float): 同步容差（秒）
    """
    speed: float = 1.0
    loop: bool = False
    auto_pause_on_stroke: bool = False
    smooth_seeking: bool = True
    buffer_size: float = 1.0
    sync_tolerance: float = 0.1


@dataclass
class TimelineState:
    """
    时间轴状态数据结构
    
    Attributes:
        current_time (float): 当前时间
        total_duration (float): 总时长
        playback_state (PlaybackState): 播放状态
        playback_speed (float): 播放速度
        current_frame (int): 当前帧
        total_frames (int): 总帧数
        current_stroke (int): 当前笔画
        total_strokes (int): 总笔画数
        is_seeking (bool): 是否在跳转
        last_update_time (float): 上次更新时间
    """
    current_time: float
    total_duration: float
    playback_state: PlaybackState
    playback_speed: float
    current_frame: int
    total_frames: int
    current_stroke: int
    total_strokes: int
    is_seeking: bool
    last_update_time: float


class TimelineManager:
    """
    时间轴管理器
    
    管理绘画动画的时间轴和播放控制
    """
    
    def __init__(self, config):
        """
        初始化时间轴管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 时间轴参数
        self.fps = config['animation'].get('fps', 30)
        self.frame_duration = 1.0 / self.fps
        
        # 播放设置
        self.playback_settings = PlaybackSettings(
            speed=config['timeline'].get('default_speed', 1.0),
            loop=config['timeline'].get('loop', False),
            auto_pause_on_stroke=config['timeline'].get('auto_pause_on_stroke', False),
            smooth_seeking=config['timeline'].get('smooth_seeking', True),
            buffer_size=config['timeline'].get('buffer_size', 1.0),
            sync_tolerance=config['timeline'].get('sync_tolerance', 0.1)
        )
        
        # 时间轴状态
        self.timeline_state = TimelineState(
            current_time=0.0,
            total_duration=0.0,
            playback_state=PlaybackState.STOPPED,
            playback_speed=1.0,
            current_frame=0,
            total_frames=0,
            current_stroke=0,
            total_strokes=0,
            is_seeking=False,
            last_update_time=time.time()
        )
        
        # 事件和标记
        self.events: List[TimelineEvent] = []
        self.markers: List[TimelineMarker] = []
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 动画数据
        self.painting_animation = None
        self.frame_timestamps: List[float] = []
        self.stroke_timestamps: List[Tuple[float, float]] = []  # (start, end)
        
        # 插值器
        self.time_interpolator = None
        self.frame_interpolator = None
        
        # 同步控制
        self.sync_enabled = config['timeline'].get('sync_enabled', True)
        self.real_time_start = None
        self.animation_time_start = 0.0
    
    def load_animation(self, painting_animation) -> bool:
        """
        加载绘画动画
        
        Args:
            painting_animation: 绘画动画对象
            
        Returns:
            bool: 是否成功
        """
        try:
            self.painting_animation = painting_animation
            
            # 更新时间轴状态
            self.timeline_state.total_duration = painting_animation.total_duration
            self.timeline_state.total_frames = len(painting_animation.all_frames)
            self.timeline_state.total_strokes = len(painting_animation.stroke_animations)
            
            # 构建帧时间戳列表
            self.frame_timestamps = [frame.timestamp for frame in painting_animation.all_frames]
            
            # 构建笔画时间戳列表
            self.stroke_timestamps = []
            for stroke_anim in painting_animation.stroke_animations:
                start_time = stroke_anim.metadata.get('start_time', 0.0)
                end_time = start_time + stroke_anim.duration
                self.stroke_timestamps.append((start_time, end_time))
            
            # 创建插值器
            self._create_interpolators()
            
            # 生成自动事件
            self._generate_auto_events()
            
            self.logger.info(f"Animation loaded: {self.timeline_state.total_duration:.2f}s, {self.timeline_state.total_frames} frames")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading animation: {str(e)}")
            return False
    
    def _create_interpolators(self):
        """
        创建插值器
        """
        try:
            if not self.frame_timestamps:
                return
            
            # 时间到帧的插值器
            frame_indices = list(range(len(self.frame_timestamps)))
            self.time_interpolator = interp1d(
                self.frame_timestamps, frame_indices,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            
            # 帧到时间的插值器
            self.frame_interpolator = interp1d(
                frame_indices, self.frame_timestamps,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            
        except Exception as e:
            self.logger.error(f"Error creating interpolators: {str(e)}")
    
    def _generate_auto_events(self):
        """
        生成自动事件
        """
        try:
            # 清除现有的自动事件
            self.events = [event for event in self.events if not event.event_id.startswith('auto_')]
            
            # 生成笔画开始和结束事件
            for i, (start_time, end_time) in enumerate(self.stroke_timestamps):
                # 笔画开始事件
                start_event = TimelineEvent(
                    event_id=f"auto_stroke_start_{i}",
                    timestamp=start_time,
                    event_type=TimelineEventType.STROKE_START,
                    data={'stroke_id': i, 'stroke_index': i},
                    callback=None
                )
                self.events.append(start_event)
                
                # 笔画结束事件
                end_event = TimelineEvent(
                    event_id=f"auto_stroke_end_{i}",
                    timestamp=end_time,
                    event_type=TimelineEventType.STROKE_END,
                    data={'stroke_id': i, 'stroke_index': i},
                    callback=None
                )
                self.events.append(end_event)
            
            # 按时间戳排序事件
            self.events.sort(key=lambda e: e.timestamp)
            
        except Exception as e:
            self.logger.error(f"Error generating auto events: {str(e)}")
    
    def play(self) -> bool:
        """
        开始播放
        
        Returns:
            bool: 是否成功
        """
        try:
            if self.painting_animation is None:
                self.logger.warning("No animation loaded")
                return False
            
            self.timeline_state.playback_state = PlaybackState.PLAYING
            self.timeline_state.playback_speed = self.playback_settings.speed
            
            # 记录实时开始时间
            self.real_time_start = time.time()
            self.animation_time_start = self.timeline_state.current_time
            
            self.logger.info("Playback started")
            self._trigger_event_callbacks('playback_started', {'time': self.timeline_state.current_time})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting playback: {str(e)}")
            return False
    
    def pause(self) -> bool:
        """
        暂停播放
        
        Returns:
            bool: 是否成功
        """
        try:
            if self.timeline_state.playback_state == PlaybackState.PLAYING:
                self.timeline_state.playback_state = PlaybackState.PAUSED
                
                self.logger.info("Playback paused")
                self._trigger_event_callbacks('playback_paused', {'time': self.timeline_state.current_time})
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error pausing playback: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        停止播放
        
        Returns:
            bool: 是否成功
        """
        try:
            self.timeline_state.playback_state = PlaybackState.STOPPED
            self.timeline_state.current_time = 0.0
            self.timeline_state.current_frame = 0
            self.timeline_state.current_stroke = 0
            
            self.real_time_start = None
            self.animation_time_start = 0.0
            
            self.logger.info("Playback stopped")
            self._trigger_event_callbacks('playback_stopped', {'time': self.timeline_state.current_time})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping playback: {str(e)}")
            return False
    
    def seek(self, target_time: float) -> bool:
        """
        跳转到指定时间
        
        Args:
            target_time (float): 目标时间
            
        Returns:
            bool: 是否成功
        """
        try:
            # 限制时间范围
            target_time = max(0.0, min(target_time, self.timeline_state.total_duration))
            
            self.timeline_state.is_seeking = True
            
            if self.playback_settings.smooth_seeking:
                # 平滑跳转
                self._smooth_seek(target_time)
            else:
                # 直接跳转
                self._direct_seek(target_time)
            
            self.timeline_state.is_seeking = False
            
            # 更新实时同步基准
            if self.timeline_state.playback_state == PlaybackState.PLAYING:
                self.real_time_start = time.time()
                self.animation_time_start = self.timeline_state.current_time
            
            self.logger.info(f"Seeked to time: {target_time:.2f}s")
            self._trigger_event_callbacks('seeked', {'time': target_time})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error seeking: {str(e)}")
            return False
    
    def _smooth_seek(self, target_time: float):
        """
        平滑跳转
        
        Args:
            target_time (float): 目标时间
        """
        try:
            # 简单的线性插值实现平滑跳转
            start_time = self.timeline_state.current_time
            steps = 10
            
            for i in range(steps + 1):
                t = i / steps
                current_time = start_time + t * (target_time - start_time)
                self._direct_seek(current_time)
                
                # 在实际应用中，这里应该有适当的延迟
                # time.sleep(0.01)
            
        except Exception as e:
            self.logger.error(f"Error in smooth seek: {str(e)}")
            self._direct_seek(target_time)
    
    def _direct_seek(self, target_time: float):
        """
        直接跳转
        
        Args:
            target_time (float): 目标时间
        """
        try:
            self.timeline_state.current_time = target_time
            
            # 更新当前帧
            if self.time_interpolator is not None:
                frame_index = self.time_interpolator(target_time)
                self.timeline_state.current_frame = int(np.clip(frame_index, 0, self.timeline_state.total_frames - 1))
            
            # 更新当前笔画
            self.timeline_state.current_stroke = self._get_current_stroke(target_time)
            
        except Exception as e:
            self.logger.error(f"Error in direct seek: {str(e)}")
    
    def _get_current_stroke(self, current_time: float) -> int:
        """
        获取当前时间对应的笔画索引
        
        Args:
            current_time (float): 当前时间
            
        Returns:
            int: 笔画索引
        """
        try:
            for i, (start_time, end_time) in enumerate(self.stroke_timestamps):
                if start_time <= current_time <= end_time:
                    return i
            
            # 如果不在任何笔画时间范围内，返回最近的笔画
            for i, (start_time, end_time) in enumerate(self.stroke_timestamps):
                if current_time < start_time:
                    return max(0, i - 1)
            
            return len(self.stroke_timestamps) - 1
            
        except Exception as e:
            self.logger.error(f"Error getting current stroke: {str(e)}")
            return 0
    
    def update(self) -> bool:
        """
        更新时间轴状态
        
        Returns:
            bool: 是否有更新
        """
        try:
            current_real_time = time.time()
            self.timeline_state.last_update_time = current_real_time
            
            if self.timeline_state.playback_state != PlaybackState.PLAYING:
                return False
            
            # 计算新的动画时间
            if self.real_time_start is not None and self.sync_enabled:
                elapsed_real_time = current_real_time - self.real_time_start
                new_animation_time = self.animation_time_start + elapsed_real_time * self.timeline_state.playback_speed
            else:
                # 基于帧率的时间更新
                new_animation_time = self.timeline_state.current_time + self.frame_duration * self.timeline_state.playback_speed
            
            # 检查是否到达结尾
            if new_animation_time >= self.timeline_state.total_duration:
                if self.playback_settings.loop:
                    # 循环播放
                    new_animation_time = 0.0
                    self.real_time_start = current_real_time
                    self.animation_time_start = 0.0
                else:
                    # 停止播放
                    new_animation_time = self.timeline_state.total_duration
                    self.stop()
                    return True
            
            # 更新时间
            old_time = self.timeline_state.current_time
            self.timeline_state.current_time = new_animation_time
            
            # 更新帧和笔画索引
            self._update_frame_and_stroke_indices()
            
            # 处理时间轴事件
            self._process_timeline_events(old_time, new_animation_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating timeline: {str(e)}")
            return False
    
    def _update_frame_and_stroke_indices(self):
        """
        更新帧和笔画索引
        """
        try:
            # 更新当前帧
            if self.time_interpolator is not None:
                frame_index = self.time_interpolator(self.timeline_state.current_time)
                self.timeline_state.current_frame = int(np.clip(frame_index, 0, self.timeline_state.total_frames - 1))
            
            # 更新当前笔画
            self.timeline_state.current_stroke = self._get_current_stroke(self.timeline_state.current_time)
            
        except Exception as e:
            self.logger.error(f"Error updating frame and stroke indices: {str(e)}")
    
    def _process_timeline_events(self, old_time: float, new_time: float):
        """
        处理时间轴事件
        
        Args:
            old_time (float): 旧时间
            new_time (float): 新时间
        """
        try:
            for event in self.events:
                if not event.enabled:
                    continue
                
                # 检查事件是否在时间范围内触发
                if old_time < event.timestamp <= new_time:
                    self._trigger_event(event)
            
        except Exception as e:
            self.logger.error(f"Error processing timeline events: {str(e)}")
    
    def _trigger_event(self, event: TimelineEvent):
        """
        触发时间轴事件
        
        Args:
            event (TimelineEvent): 事件对象
        """
        try:
            self.logger.debug(f"Triggering event: {event.event_id} at {event.timestamp:.2f}s")
            
            # 执行事件回调
            if event.callback:
                event.callback(event)
            
            # 触发事件类型回调
            event_type_name = event.event_type.value
            self._trigger_event_callbacks(event_type_name, event.data)
            
            # 处理特殊事件
            if event.event_type == TimelineEventType.STROKE_END and self.playback_settings.auto_pause_on_stroke:
                self.pause()
            
        except Exception as e:
            self.logger.error(f"Error triggering event {event.event_id}: {str(e)}")
    
    def add_event(self, event: TimelineEvent) -> bool:
        """
        添加时间轴事件
        
        Args:
            event (TimelineEvent): 事件对象
            
        Returns:
            bool: 是否成功
        """
        try:
            # 检查事件ID是否唯一
            if any(e.event_id == event.event_id for e in self.events):
                self.logger.warning(f"Event ID {event.event_id} already exists")
                return False
            
            self.events.append(event)
            
            # 按时间戳排序
            self.events.sort(key=lambda e: e.timestamp)
            
            self.logger.info(f"Event added: {event.event_id} at {event.timestamp:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding event: {str(e)}")
            return False
    
    def remove_event(self, event_id: str) -> bool:
        """
        移除时间轴事件
        
        Args:
            event_id (str): 事件ID
            
        Returns:
            bool: 是否成功
        """
        try:
            for i, event in enumerate(self.events):
                if event.event_id == event_id:
                    del self.events[i]
                    self.logger.info(f"Event removed: {event_id}")
                    return True
            
            self.logger.warning(f"Event not found: {event_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing event: {str(e)}")
            return False
    
    def add_marker(self, marker: TimelineMarker) -> bool:
        """
        添加时间轴标记
        
        Args:
            marker (TimelineMarker): 标记对象
            
        Returns:
            bool: 是否成功
        """
        try:
            # 检查标记ID是否唯一
            if any(m.marker_id == marker.marker_id for m in self.markers):
                self.logger.warning(f"Marker ID {marker.marker_id} already exists")
                return False
            
            self.markers.append(marker)
            
            # 按时间戳排序
            self.markers.sort(key=lambda m: m.timestamp)
            
            self.logger.info(f"Marker added: {marker.marker_id} at {marker.timestamp:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding marker: {str(e)}")
            return False
    
    def remove_marker(self, marker_id: str) -> bool:
        """
        移除时间轴标记
        
        Args:
            marker_id (str): 标记ID
            
        Returns:
            bool: 是否成功
        """
        try:
            for i, marker in enumerate(self.markers):
                if marker.marker_id == marker_id:
                    del self.markers[i]
                    self.logger.info(f"Marker removed: {marker_id}")
                    return True
            
            self.logger.warning(f"Marker not found: {marker_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing marker: {str(e)}")
            return False
    
    def set_playback_speed(self, speed: float) -> bool:
        """
        设置播放速度
        
        Args:
            speed (float): 播放速度
            
        Returns:
            bool: 是否成功
        """
        try:
            speed = max(0.1, min(10.0, speed))  # 限制速度范围
            
            self.playback_settings.speed = speed
            self.timeline_state.playback_speed = speed
            
            # 更新实时同步基准
            if self.timeline_state.playback_state == PlaybackState.PLAYING:
                self.real_time_start = time.time()
                self.animation_time_start = self.timeline_state.current_time
            
            self.logger.info(f"Playback speed set to: {speed}x")
            self._trigger_event_callbacks('speed_changed', {'speed': speed})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting playback speed: {str(e)}")
            return False
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """
        注册事件回调
        
        Args:
            event_type (str): 事件类型
            callback (Callable): 回调函数
        """
        try:
            if event_type not in self.event_callbacks:
                self.event_callbacks[event_type] = []
            
            self.event_callbacks[event_type].append(callback)
            self.logger.info(f"Event callback registered for: {event_type}")
            
        except Exception as e:
            self.logger.error(f"Error registering event callback: {str(e)}")
    
    def _trigger_event_callbacks(self, event_type: str, data: Dict[str, Any]):
        """
        触发事件回调
        
        Args:
            event_type (str): 事件类型
            data (Dict[str, Any]): 事件数据
        """
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in event callback: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error triggering event callbacks: {str(e)}")
    
    def get_current_frame(self):
        """
        获取当前帧对象
        
        Returns:
            当前帧对象或None
        """
        try:
            if (self.painting_animation and 
                0 <= self.timeline_state.current_frame < len(self.painting_animation.all_frames)):
                return self.painting_animation.all_frames[self.timeline_state.current_frame]
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current frame: {str(e)}")
            return None
    
    def get_timeline_statistics(self) -> Dict[str, Any]:
        """
        获取时间轴统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'timeline_state': {
                    'current_time': self.timeline_state.current_time,
                    'total_duration': self.timeline_state.total_duration,
                    'playback_state': self.timeline_state.playback_state.value,
                    'playback_speed': self.timeline_state.playback_speed,
                    'current_frame': self.timeline_state.current_frame,
                    'total_frames': self.timeline_state.total_frames,
                    'current_stroke': self.timeline_state.current_stroke,
                    'total_strokes': self.timeline_state.total_strokes,
                    'progress_percentage': (self.timeline_state.current_time / self.timeline_state.total_duration * 100) if self.timeline_state.total_duration > 0 else 0
                },
                'playback_settings': {
                    'speed': self.playback_settings.speed,
                    'loop': self.playback_settings.loop,
                    'auto_pause_on_stroke': self.playback_settings.auto_pause_on_stroke,
                    'smooth_seeking': self.playback_settings.smooth_seeking
                },
                'events_and_markers': {
                    'total_events': len(self.events),
                    'enabled_events': len([e for e in self.events if e.enabled]),
                    'total_markers': len(self.markers)
                },
                'sync_info': {
                    'sync_enabled': self.sync_enabled,
                    'fps': self.fps,
                    'frame_duration': self.frame_duration
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting timeline statistics: {str(e)}")
            return {'error': str(e)}
    
    def export_timeline_data(self, output_path: str) -> bool:
        """
        导出时间轴数据
        
        Args:
            output_path (str): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            timeline_data = {
                'timeline_state': {
                    'total_duration': self.timeline_state.total_duration,
                    'total_frames': self.timeline_state.total_frames,
                    'total_strokes': self.timeline_state.total_strokes
                },
                'playback_settings': {
                    'speed': self.playback_settings.speed,
                    'loop': self.playback_settings.loop,
                    'auto_pause_on_stroke': self.playback_settings.auto_pause_on_stroke,
                    'smooth_seeking': self.playback_settings.smooth_seeking
                },
                'events': [
                    {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'event_type': event.event_type.value,
                        'data': event.data,
                        'enabled': event.enabled
                    }
                    for event in self.events
                ],
                'markers': [
                    {
                        'marker_id': marker.marker_id,
                        'timestamp': marker.timestamp,
                        'name': marker.name,
                        'description': marker.description,
                        'color': marker.color,
                        'metadata': marker.metadata
                    }
                    for marker in self.markers
                ],
                'frame_timestamps': self.frame_timestamps,
                'stroke_timestamps': self.stroke_timestamps
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Timeline data exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting timeline data: {str(e)}")
            return False