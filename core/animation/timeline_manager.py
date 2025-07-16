# -*- coding: utf-8 -*-
"""
时间线管理器模块

提供动画时间线的管理和控制功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
from pathlib import Path
from ..stroke_extraction.stroke_detector import Stroke
from .painting_animator import AnimationFrame, AnimationSequence


class TimelineEventType(Enum):
    """时间线事件类型"""
    STROKE_START = "stroke_start"
    STROKE_END = "stroke_end"
    LAYER_CHANGE = "layer_change"
    COLOR_CHANGE = "color_change"
    BRUSH_CHANGE = "brush_change"
    PAUSE = "pause"
    SPEED_CHANGE = "speed_change"
    MARKER = "marker"


class PlaybackState(Enum):
    """播放状态"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"


@dataclass
class TimelineMarker:
    """时间线标记"""
    timestamp: float
    name: str
    description: str = ""
    color: Tuple[int, int, int] = (255, 0, 0)
    is_keyframe: bool = False


@dataclass
class TimelineEvent:
    """时间线事件"""
    timestamp: float
    event_type: TimelineEventType
    data: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    layer_id: Optional[str] = None


@dataclass
class TimelineLayer:
    """时间线图层"""
    layer_id: str
    name: str
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend_mode: str = "normal"
    events: List[TimelineEvent] = field(default_factory=list)
    color: Tuple[int, int, int] = (128, 128, 128)


@dataclass
class PlaybackConfig:
    """播放配置"""
    fps: float = 30.0
    speed_multiplier: float = 1.0
    loop: bool = False
    auto_pause_at_markers: bool = False
    smooth_playback: bool = True
    preload_frames: int = 10
    enable_audio: bool = False
    audio_file: Optional[str] = None


@dataclass
class TimelineState:
    """时间线状态"""
    current_time: float = 0.0
    total_duration: float = 0.0
    playback_state: PlaybackState = PlaybackState.STOPPED
    current_frame: int = 0
    total_frames: int = 0
    speed_multiplier: float = 1.0
    is_looping: bool = False


class TimelineManager:
    """时间线管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化时间线管理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 播放配置
        self.playback_config = PlaybackConfig(
            fps=config.get('fps', 30.0),
            speed_multiplier=config.get('speed_multiplier', 1.0),
            loop=config.get('loop', False),
            auto_pause_at_markers=config.get('auto_pause_at_markers', False),
            smooth_playback=config.get('smooth_playback', True),
            preload_frames=config.get('preload_frames', 10)
        )
        
        # 时间线状态
        self.state = TimelineState()
        
        # 时间线数据
        self.layers: List[TimelineLayer] = []
        self.markers: List[TimelineMarker] = []
        self.global_events: List[TimelineEvent] = []
        
        # 动画序列
        self.animation_sequence: Optional[AnimationSequence] = None
        
        # 回调函数
        self.event_callbacks: Dict[str, List[callable]] = {
            'time_changed': [],
            'playback_state_changed': [],
            'marker_reached': [],
            'event_triggered': [],
            'layer_changed': []
        }
        
        # 性能优化
        self.frame_cache: Dict[int, AnimationFrame] = {}
        self.enable_caching = config.get('enable_caching', True)
        self.max_cache_size = config.get('max_cache_size', 100)
        
        # 创建默认图层
        self._create_default_layer()
        
    def create_timeline_from_strokes(self, strokes: List[Stroke], 
                                   duration: float = None) -> AnimationSequence:
        """
        从笔触创建时间线
        
        Args:
            strokes: 笔触列表
            duration: 总持续时间（秒）
            
        Returns:
            动画序列
        """
        if not strokes:
            return AnimationSequence(frames=[], duration=0.0, fps=self.playback_config.fps)
        
        try:
            # 计算总持续时间
            if duration is None:
                duration = len(strokes) * 0.5  # 每个笔触0.5秒
            
            # 计算帧数
            total_frames = int(duration * self.playback_config.fps)
            
            # 分配笔触到时间线
            stroke_timeline = self._distribute_strokes_on_timeline(strokes, duration)
            
            # 生成动画帧
            frames = self._generate_animation_frames(stroke_timeline, total_frames, duration)
            
            # 创建动画序列
            self.animation_sequence = AnimationSequence(
                frames=frames,
                duration=duration,
                fps=self.playback_config.fps
            )
            
            # 更新状态
            self.state.total_duration = duration
            self.state.total_frames = total_frames
            
            # 创建时间线事件
            self._create_timeline_events_from_strokes(strokes, duration)
            
            self.logger.info(f"Timeline created with {len(frames)} frames, duration: {duration}s")
            
            return self.animation_sequence
            
        except Exception as e:
            self.logger.error(f"Error creating timeline from strokes: {e}")
            return AnimationSequence(frames=[], duration=0.0, fps=self.playback_config.fps)
    
    def add_layer(self, name: str, layer_id: str = None) -> TimelineLayer:
        """
        添加图层
        
        Args:
            name: 图层名称
            layer_id: 图层ID
            
        Returns:
            创建的图层
        """
        if layer_id is None:
            layer_id = f"layer_{len(self.layers)}"
        
        layer = TimelineLayer(
            layer_id=layer_id,
            name=name,
            color=(int(np.random.randint(100, 255)), int(np.random.randint(100, 255)), int(np.random.randint(100, 255)))
        )
        
        self.layers.append(layer)
        self._trigger_callback('layer_changed', {'action': 'added', 'layer': layer})
        
        self.logger.info(f"Layer added: {name} (ID: {layer_id})")
        return layer
    
    def remove_layer(self, layer_id: str) -> bool:
        """
        移除图层
        
        Args:
            layer_id: 图层ID
            
        Returns:
            是否成功移除
        """
        for i, layer in enumerate(self.layers):
            if layer.layer_id == layer_id:
                removed_layer = self.layers.pop(i)
                self._trigger_callback('layer_changed', {'action': 'removed', 'layer': removed_layer})
                self.logger.info(f"Layer removed: {layer_id}")
                return True
        
        return False
    
    def add_marker(self, timestamp: float, name: str, description: str = "", 
                   is_keyframe: bool = False) -> TimelineMarker:
        """
        添加时间线标记
        
        Args:
            timestamp: 时间戳
            name: 标记名称
            description: 描述
            is_keyframe: 是否为关键帧
            
        Returns:
            创建的标记
        """
        marker = TimelineMarker(
            timestamp=timestamp,
            name=name,
            description=description,
            is_keyframe=is_keyframe
        )
        
        # 按时间戳排序插入
        insert_index = 0
        for i, existing_marker in enumerate(self.markers):
            if existing_marker.timestamp > timestamp:
                break
            insert_index = i + 1
        
        self.markers.insert(insert_index, marker)
        self.logger.info(f"Marker added: {name} at {timestamp}s")
        
        return marker
    
    def remove_marker(self, timestamp: float, name: str = None) -> bool:
        """
        移除时间线标记
        
        Args:
            timestamp: 时间戳
            name: 标记名称（可选）
            
        Returns:
            是否成功移除
        """
        for i, marker in enumerate(self.markers):
            if abs(marker.timestamp - timestamp) < 0.01:  # 允许小误差
                if name is None or marker.name == name:
                    removed_marker = self.markers.pop(i)
                    self.logger.info(f"Marker removed: {removed_marker.name}")
                    return True
        
        return False
    
    def play(self, from_time: float = None):
        """
        开始播放
        
        Args:
            from_time: 起始时间
        """
        if from_time is not None:
            self.seek_to_time(from_time)
        
        self.state.playback_state = PlaybackState.PLAYING
        self._trigger_callback('playback_state_changed', {'state': PlaybackState.PLAYING})
        
        self.logger.info(f"Playback started from {self.state.current_time}s")
    
    def pause(self):
        """暂停播放"""
        self.state.playback_state = PlaybackState.PAUSED
        self._trigger_callback('playback_state_changed', {'state': PlaybackState.PAUSED})
        
        self.logger.info("Playback paused")
    
    def stop(self):
        """停止播放"""
        self.state.playback_state = PlaybackState.STOPPED
        self.state.current_time = 0.0
        self.state.current_frame = 0
        
        self._trigger_callback('playback_state_changed', {'state': PlaybackState.STOPPED})
        self._trigger_callback('time_changed', {'time': 0.0, 'frame': 0})
        
        self.logger.info("Playback stopped")
    
    def seek_to_time(self, time: float):
        """
        跳转到指定时间
        
        Args:
            time: 目标时间
        """
        time = max(0.0, min(time, self.state.total_duration))
        
        old_state = self.state.playback_state
        self.state.playback_state = PlaybackState.SEEKING
        
        self.state.current_time = time
        self.state.current_frame = int(time * self.playback_config.fps)
        
        # 检查是否到达标记
        self._check_markers_at_time(time)
        
        self.state.playback_state = old_state
        
        self._trigger_callback('time_changed', {'time': time, 'frame': self.state.current_frame})
        
        self.logger.debug(f"Seeked to time: {time}s (frame: {self.state.current_frame})")
    
    def seek_to_frame(self, frame: int):
        """
        跳转到指定帧
        
        Args:
            frame: 目标帧
        """
        frame = max(0, min(frame, self.state.total_frames - 1))
        time = frame / self.playback_config.fps
        self.seek_to_time(time)
    
    def step_forward(self, frames: int = 1):
        """
        向前步进
        
        Args:
            frames: 步进帧数
        """
        new_frame = self.state.current_frame + frames
        self.seek_to_frame(new_frame)
    
    def step_backward(self, frames: int = 1):
        """
        向后步进
        
        Args:
            frames: 步进帧数
        """
        new_frame = self.state.current_frame - frames
        self.seek_to_frame(new_frame)
    
    def set_speed(self, speed_multiplier: float):
        """
        设置播放速度
        
        Args:
            speed_multiplier: 速度倍数
        """
        self.playback_config.speed_multiplier = max(0.1, min(speed_multiplier, 10.0))
        self.state.speed_multiplier = self.playback_config.speed_multiplier
        
        self.logger.info(f"Playback speed set to {speed_multiplier}x")
    
    def set_loop(self, loop: bool):
        """
        设置循环播放
        
        Args:
            loop: 是否循环
        """
        self.playback_config.loop = loop
        self.state.is_looping = loop
        
        self.logger.info(f"Loop playback: {loop}")
    
    def update(self, delta_time: float):
        """
        更新时间线状态
        
        Args:
            delta_time: 时间增量（秒）
        """
        if self.state.playback_state != PlaybackState.PLAYING:
            return
        
        # 计算新时间
        time_increment = delta_time * self.playback_config.speed_multiplier
        new_time = self.state.current_time + time_increment
        
        # 检查是否到达结尾
        if new_time >= self.state.total_duration:
            if self.playback_config.loop:
                new_time = 0.0
            else:
                new_time = self.state.total_duration
                self.pause()
        
        # 更新时间
        self.seek_to_time(new_time)
    
    def get_current_frame(self) -> Optional[AnimationFrame]:
        """
        获取当前帧
        
        Returns:
            当前动画帧
        """
        if not self.animation_sequence or self.state.current_frame >= len(self.animation_sequence.frames):
            return None
        
        frame_index = self.state.current_frame
        
        # 检查缓存
        if self.enable_caching and frame_index in self.frame_cache:
            return self.frame_cache[frame_index]
        
        # 获取帧
        frame = self.animation_sequence.frames[frame_index]
        
        # 缓存帧
        if self.enable_caching:
            self._cache_frame(frame_index, frame)
        
        return frame
    
    def get_frame_at_time(self, time: float) -> Optional[AnimationFrame]:
        """
        获取指定时间的帧
        
        Args:
            time: 时间
            
        Returns:
            动画帧
        """
        if not self.animation_sequence:
            return None
        
        frame_index = int(time * self.playback_config.fps)
        frame_index = max(0, min(frame_index, len(self.animation_sequence.frames) - 1))
        
        return self.animation_sequence.frames[frame_index]
    
    def get_visible_layers(self) -> List[TimelineLayer]:
        """
        获取可见图层
        
        Returns:
            可见图层列表
        """
        return [layer for layer in self.layers if layer.visible]
    
    def get_markers_in_range(self, start_time: float, end_time: float) -> List[TimelineMarker]:
        """
        获取时间范围内的标记
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            标记列表
        """
        return [
            marker for marker in self.markers
            if start_time <= marker.timestamp <= end_time
        ]
    
    def export_timeline(self, file_path: str) -> bool:
        """
        导出时间线
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            timeline_data = {
                'version': '1.0',
                'playback_config': {
                    'fps': self.playback_config.fps,
                    'speed_multiplier': self.playback_config.speed_multiplier,
                    'loop': self.playback_config.loop,
                    'auto_pause_at_markers': self.playback_config.auto_pause_at_markers
                },
                'state': {
                    'total_duration': self.state.total_duration,
                    'total_frames': self.state.total_frames
                },
                'layers': [
                    {
                        'layer_id': layer.layer_id,
                        'name': layer.name,
                        'visible': layer.visible,
                        'locked': layer.locked,
                        'opacity': layer.opacity,
                        'blend_mode': layer.blend_mode,
                        'color': layer.color
                    }
                    for layer in self.layers
                ],
                'markers': [
                    {
                        'timestamp': marker.timestamp,
                        'name': marker.name,
                        'description': marker.description,
                        'color': marker.color,
                        'is_keyframe': marker.is_keyframe
                    }
                    for marker in self.markers
                ]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Timeline exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting timeline: {e}")
            return False
    
    def import_timeline(self, file_path: str) -> bool:
        """
        导入时间线
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                timeline_data = json.load(f)
            
            # 恢复播放配置
            if 'playback_config' in timeline_data:
                config = timeline_data['playback_config']
                self.playback_config.fps = config.get('fps', 30.0)
                self.playback_config.speed_multiplier = config.get('speed_multiplier', 1.0)
                self.playback_config.loop = config.get('loop', False)
                self.playback_config.auto_pause_at_markers = config.get('auto_pause_at_markers', False)
            
            # 恢复状态
            if 'state' in timeline_data:
                state = timeline_data['state']
                self.state.total_duration = state.get('total_duration', 0.0)
                self.state.total_frames = state.get('total_frames', 0)
            
            # 恢复图层
            self.layers.clear()
            if 'layers' in timeline_data:
                for layer_data in timeline_data['layers']:
                    layer = TimelineLayer(
                        layer_id=layer_data['layer_id'],
                        name=layer_data['name'],
                        visible=layer_data.get('visible', True),
                        locked=layer_data.get('locked', False),
                        opacity=layer_data.get('opacity', 1.0),
                        blend_mode=layer_data.get('blend_mode', 'normal'),
                        color=tuple(layer_data.get('color', [128, 128, 128]))
                    )
                    self.layers.append(layer)
            
            # 恢复标记
            self.markers.clear()
            if 'markers' in timeline_data:
                for marker_data in timeline_data['markers']:
                    marker = TimelineMarker(
                        timestamp=marker_data['timestamp'],
                        name=marker_data['name'],
                        description=marker_data.get('description', ''),
                        color=tuple(marker_data.get('color', [255, 0, 0])),
                        is_keyframe=marker_data.get('is_keyframe', False)
                    )
                    self.markers.append(marker)
            
            self.logger.info(f"Timeline imported from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing timeline: {e}")
            return False
    
    def add_event_callback(self, event_type: str, callback: callable):
        """
        添加事件回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def remove_event_callback(self, event_type: str, callback: callable):
        """
        移除事件回调
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
    
    def _create_default_layer(self):
        """创建默认图层"""
        default_layer = TimelineLayer(
            layer_id="default",
            name="Default Layer",
            color=(100, 150, 200)
        )
        self.layers.append(default_layer)
    
    def _distribute_strokes_on_timeline(self, strokes: List[Stroke], 
                                      duration: float) -> Dict[float, List[Stroke]]:
        """
        在时间线上分布笔触
        
        Args:
            strokes: 笔触列表
            duration: 总持续时间
            
        Returns:
            时间戳到笔触的映射
        """
        stroke_timeline = {}
        
        if not strokes:
            return stroke_timeline
        
        # 简单的均匀分布
        time_per_stroke = duration / len(strokes)
        
        for i, stroke in enumerate(strokes):
            timestamp = i * time_per_stroke
            if timestamp not in stroke_timeline:
                stroke_timeline[timestamp] = []
            stroke_timeline[timestamp].append(stroke)
        
        return stroke_timeline
    
    def _generate_animation_frames(self, stroke_timeline: Dict[float, List[Stroke]], 
                                 total_frames: int, duration: float) -> List[AnimationFrame]:
        """
        生成动画帧
        
        Args:
            stroke_timeline: 笔触时间线
            total_frames: 总帧数
            duration: 总持续时间
            
        Returns:
            动画帧列表
        """
        frames = []
        accumulated_strokes = []
        
        for frame_index in range(total_frames):
            timestamp = frame_index / self.playback_config.fps
            
            # 检查是否有新笔触
            for stroke_time, strokes in stroke_timeline.items():
                if abs(stroke_time - timestamp) < (1.0 / self.playback_config.fps / 2):
                    accumulated_strokes.extend(strokes)
            
            # 创建帧数据
            frame_strokes = []
            for stroke in accumulated_strokes:
                stroke_data = {
                    'points': stroke.points,
                    'color': getattr(stroke, 'color', (0, 0, 0)),
                    'thickness': getattr(stroke, 'thickness', 2),
                    'opacity': getattr(stroke, 'opacity', 1.0)
                }
                frame_strokes.append(stroke_data)
            
            frame = AnimationFrame(
                timestamp=timestamp,
                strokes=frame_strokes,
                canvas_state=None  # 将在渲染时生成
            )
            
            frames.append(frame)
        
        return frames
    
    def _create_timeline_events_from_strokes(self, strokes: List[Stroke], duration: float):
        """
        从笔触创建时间线事件
        
        Args:
            strokes: 笔触列表
            duration: 总持续时间
        """
        self.global_events.clear()
        
        time_per_stroke = duration / len(strokes) if strokes else 0
        
        for i, stroke in enumerate(strokes):
            start_time = i * time_per_stroke
            end_time = start_time + time_per_stroke * 0.8  # 80%的时间用于绘制
            
            # 笔触开始事件
            start_event = TimelineEvent(
                timestamp=start_time,
                event_type=TimelineEventType.STROKE_START,
                data={'stroke_index': i, 'stroke': stroke},
                layer_id="default"
            )
            self.global_events.append(start_event)
            
            # 笔触结束事件
            end_event = TimelineEvent(
                timestamp=end_time,
                event_type=TimelineEventType.STROKE_END,
                data={'stroke_index': i, 'stroke': stroke},
                layer_id="default"
            )
            self.global_events.append(end_event)
    
    def _check_markers_at_time(self, time: float):
        """
        检查指定时间的标记
        
        Args:
            time: 时间
        """
        tolerance = 1.0 / self.playback_config.fps / 2  # 半帧的容差
        
        for marker in self.markers:
            if abs(marker.timestamp - time) <= tolerance:
                self._trigger_callback('marker_reached', {'marker': marker, 'time': time})
                
                if self.playback_config.auto_pause_at_markers:
                    self.pause()
    
    def _trigger_callback(self, event_type: str, data: Dict[str, Any]):
        """
        触发回调
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback for {event_type}: {e}")
    
    def _cache_frame(self, frame_index: int, frame: AnimationFrame):
        """
        缓存帧
        
        Args:
            frame_index: 帧索引
            frame: 动画帧
        """
        if len(self.frame_cache) >= self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]
        
        self.frame_cache[frame_index] = frame
    
    def clear_cache(self):
        """清除缓存"""
        self.frame_cache.clear()
        self.logger.info("Timeline cache cleared")
    
    def get_timeline_info(self) -> Dict[str, Any]:
        """
        获取时间线信息
        
        Returns:
            时间线信息
        """
        return {
            'duration': self.state.total_duration,
            'total_frames': self.state.total_frames,
            'fps': self.playback_config.fps,
            'current_time': self.state.current_time,
            'current_frame': self.state.current_frame,
            'playback_state': self.state.playback_state.value,
            'speed_multiplier': self.state.speed_multiplier,
            'is_looping': self.state.is_looping,
            'layer_count': len(self.layers),
            'marker_count': len(self.markers),
            'event_count': len(self.global_events)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计信息
        """
        return {
            'frame_cache_size': len(self.frame_cache),
            'max_cache_size': self.max_cache_size,
            'enable_caching': self.enable_caching,
            'smooth_playback': self.playback_config.smooth_playback,
            'preload_frames': self.playback_config.preload_frames
        }