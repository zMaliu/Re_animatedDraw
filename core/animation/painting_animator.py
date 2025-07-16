# -*- coding: utf-8 -*-
"""
绘画动画器

实现中国画笔刷绘制动画生成
包括笔画轨迹生成、时间控制、动画渲染等功能
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import time
import math
from scipy.interpolate import interp1d, splprep, splev
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import json
import os


@dataclass
class AnimationFrame:
    """
    动画帧数据结构
    
    Attributes:
        frame_id (int): 帧ID
        timestamp (float): 时间戳
        brush_position (Tuple[float, float]): 笔刷位置
        brush_pressure (float): 笔刷压力
        brush_angle (float): 笔刷角度
        brush_size (float): 笔刷大小
        ink_flow (float): 墨水流量
        stroke_progress (float): 笔画进度
        canvas_state (np.ndarray): 画布状态
        metadata (Dict): 元数据
    """
    frame_id: int
    timestamp: float
    brush_position: Tuple[float, float]
    brush_pressure: float
    brush_angle: float
    brush_size: float
    ink_flow: float
    stroke_progress: float
    canvas_state: Optional[np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class StrokeAnimation:
    """
    笔画动画数据结构
    
    Attributes:
        stroke_id (int): 笔画ID
        trajectory (List[Tuple[float, float]]): 轨迹点列表
        timing (List[float]): 时间点列表
        pressure_curve (List[float]): 压力曲线
        velocity_curve (List[float]): 速度曲线
        brush_size_curve (List[float]): 笔刷大小曲线
        ink_flow_curve (List[float]): 墨水流量曲线
        duration (float): 持续时间
        frames (List[AnimationFrame]): 动画帧列表
        metadata (Dict): 元数据
    """
    stroke_id: int
    trajectory: List[Tuple[float, float]]
    timing: List[float]
    pressure_curve: List[float]
    velocity_curve: List[float]
    brush_size_curve: List[float]
    ink_flow_curve: List[float]
    duration: float
    frames: List[AnimationFrame]
    metadata: Dict[str, Any]


@dataclass
class PaintingAnimation:
    """
    绘画动画数据结构
    
    Attributes:
        animation_id (str): 动画ID
        stroke_animations (List[StrokeAnimation]): 笔画动画列表
        total_duration (float): 总持续时间
        fps (int): 帧率
        canvas_size (Tuple[int, int]): 画布大小
        background_color (Tuple[int, int, int]): 背景颜色
        all_frames (List[AnimationFrame]): 所有动画帧
        metadata (Dict): 元数据
    """
    animation_id: str
    stroke_animations: List[StrokeAnimation]
    total_duration: float
    fps: int
    canvas_size: Tuple[int, int]
    background_color: Tuple[int, int, int]
    all_frames: List[AnimationFrame]
    metadata: Dict[str, Any]


class PaintingAnimator:
    """
    绘画动画器
    
    生成中国画笔刷绘制动画
    """
    
    def __init__(self, config):
        """
        初始化绘画动画器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 动画参数
        self.fps = config['animation'].get('fps', 30)
        self.canvas_size = tuple(config['animation'].get('canvas_size', (800, 600)))
        self.background_color = tuple(config['animation'].get('background_color', (255, 255, 255)))
        
        # 笔刷参数
        self.default_brush_size = config['animation'].get('default_brush_size', 5)
        self.min_brush_size = config['animation'].get('min_brush_size', 1)
        self.max_brush_size = config['animation'].get('max_brush_size', 20)
        
        # 时间参数
        self.stroke_duration_base = config['animation'].get('stroke_duration_base', 2.0)
        self.pause_between_strokes = config['animation'].get('pause_between_strokes', 0.5)
        self.speed_variation = config['animation'].get('speed_variation', 0.3)
        
        # 物理参数
        self.pressure_sensitivity = config['animation'].get('pressure_sensitivity', 0.8)
        self.ink_flow_rate = config['animation'].get('ink_flow_rate', 0.7)
        self.brush_elasticity = config['animation'].get('brush_elasticity', 0.5)
        
        # 插值参数
        self.trajectory_smoothness = config['animation'].get('trajectory_smoothness', 0.1)
        self.curve_resolution = config['animation'].get('curve_resolution', 100)
        
        # 渲染参数
        self.enable_brush_texture = config['animation'].get('enable_brush_texture', True)
        self.enable_ink_diffusion = config['animation'].get('enable_ink_diffusion', True)
        self.enable_paper_texture = config['animation'].get('enable_paper_texture', False)
    
    def create_animation(self, strokes: List[Dict[str, Any]], 
                        stroke_order: List[int]) -> PaintingAnimation:
        """
        创建绘画动画
        
        Args:
            strokes (List[Dict]): 笔画列表
            stroke_order (List[int]): 笔画顺序
            
        Returns:
            PaintingAnimation: 绘画动画
        """
        try:
            self.logger.info(f"Creating animation for {len(strokes)} strokes")
            
            # 生成笔画动画
            stroke_animations = []
            current_time = 0.0
            
            for i, stroke_idx in enumerate(stroke_order):
                if stroke_idx < len(strokes):
                    stroke = strokes[stroke_idx]
                    
                    # 生成单个笔画动画
                    stroke_anim = self._create_stroke_animation(
                        stroke, i, current_time
                    )
                    
                    stroke_animations.append(stroke_anim)
                    current_time += stroke_anim.duration + self.pause_between_strokes
            
            # 生成完整动画
            total_duration = current_time - self.pause_between_strokes
            all_frames = self._combine_stroke_animations(stroke_animations)
            
            animation_id = f"painting_animation_{int(time.time())}"
            
            painting_animation = PaintingAnimation(
                animation_id=animation_id,
                stroke_animations=stroke_animations,
                total_duration=total_duration,
                fps=self.fps,
                canvas_size=self.canvas_size,
                background_color=self.background_color,
                all_frames=all_frames,
                metadata={
                    'num_strokes': len(stroke_animations),
                    'total_frames': len(all_frames),
                    'creation_time': time.time(),
                    'config': {
                        'fps': self.fps,
                        'canvas_size': self.canvas_size,
                        'stroke_duration_base': self.stroke_duration_base
                    }
                }
            )
            
            self.logger.info(f"Animation created: {total_duration:.2f}s, {len(all_frames)} frames")
            
            return painting_animation
            
        except Exception as e:
            self.logger.error(f"Error creating animation: {str(e)}")
            return self._create_default_animation(strokes)
    
    def _create_stroke_animation(self, stroke: Dict[str, Any], 
                               stroke_index: int, start_time: float) -> StrokeAnimation:
        """
        创建单个笔画动画
        
        Args:
            stroke (Dict): 笔画数据
            stroke_index (int): 笔画索引
            start_time (float): 开始时间
            
        Returns:
            StrokeAnimation: 笔画动画
        """
        try:
            # 提取笔画轨迹
            trajectory = self._extract_stroke_trajectory(stroke)
            
            # 计算动画持续时间
            duration = self._calculate_stroke_duration(stroke)
            
            # 生成时间轴
            timing = self._generate_timing_curve(trajectory, duration)
            
            # 生成动画曲线
            pressure_curve = self._generate_pressure_curve(trajectory, timing)
            velocity_curve = self._generate_velocity_curve(trajectory, timing)
            brush_size_curve = self._generate_brush_size_curve(pressure_curve)
            ink_flow_curve = self._generate_ink_flow_curve(velocity_curve, pressure_curve)
            
            # 生成动画帧
            frames = self._generate_stroke_frames(
                trajectory, timing, pressure_curve, velocity_curve,
                brush_size_curve, ink_flow_curve, start_time, stroke_index
            )
            
            return StrokeAnimation(
                stroke_id=stroke_index,
                trajectory=trajectory,
                timing=timing,
                pressure_curve=pressure_curve,
                velocity_curve=velocity_curve,
                brush_size_curve=brush_size_curve,
                ink_flow_curve=ink_flow_curve,
                duration=duration,
                frames=frames,
                metadata={
                    'stroke_type': stroke.get('stroke_class', 'unknown'),
                    'complexity': stroke.get('complexity', 1.0),
                    'length': stroke.get('arc_length', 0),
                    'start_time': start_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error creating stroke animation: {str(e)}")
            return self._create_default_stroke_animation(stroke_index, start_time)
    
    def _extract_stroke_trajectory(self, stroke: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        提取笔画轨迹
        
        Args:
            stroke (Dict): 笔画数据
            
        Returns:
            List[Tuple[float, float]]: 轨迹点列表
        """
        try:
            # 优先使用骨架作为轨迹
            skeleton = stroke.get('skeleton', [])
            # 安全检查skeleton是否有效
            skeleton_valid = False
            if isinstance(skeleton, np.ndarray):
                skeleton_valid = skeleton.size > 0 and len(skeleton) > 1
            elif isinstance(skeleton, list):
                skeleton_valid = len(skeleton) > 1
            else:
                skeleton_valid = bool(skeleton) and len(skeleton) > 1 if hasattr(skeleton, '__len__') else False
                
            if skeleton_valid:
                trajectory = [(float(p[0]), float(p[1])) for p in skeleton]
            else:
                # 使用轮廓中心线
                contour = stroke.get('contour', [])
                # 安全检查contour是否有效
                contour_valid = False
                if isinstance(contour, np.ndarray):
                    contour_valid = contour.size > 0 and len(contour) > 1
                elif isinstance(contour, list):
                    contour_valid = len(contour) > 1
                else:
                    contour_valid = bool(contour) and len(contour) > 1 if hasattr(contour, '__len__') else False
                    
                if contour_valid:
                    # 简化轮廓为中心线
                    trajectory = self._contour_to_centerline(contour)
                else:
                    # 使用边界框生成简单轨迹
                    bbox = stroke.get('bounding_rect', (0, 0, 100, 20))
                    trajectory = self._bbox_to_trajectory(bbox, stroke.get('stroke_class', 'horizontal'))
            
            # 平滑轨迹
            if len(trajectory) > 2:
                trajectory = self._smooth_trajectory(trajectory)
            
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Error extracting stroke trajectory: {str(e)}")
            # 返回默认轨迹
            return [(0, 0), (100, 0)]
    
    def _contour_to_centerline(self, contour: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        将轮廓转换为中心线
        
        Args:
            contour (List[Tuple[int, int]]): 轮廓点
            
        Returns:
            List[Tuple[float, float]]: 中心线点
        """
        try:
            if len(contour) < 3:
                return [(float(p[0]), float(p[1])) for p in contour]
            
            # 转换为numpy数组
            contour_array = np.array(contour)
            
            # 计算轮廓的主轴方向
            centroid = np.mean(contour_array, axis=0)
            
            # 使用PCA找到主方向
            centered = contour_array - centroid
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 主方向向量
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            
            # 投影到主方向
            projections = np.dot(centered, main_direction)
            
            # 排序并生成中心线
            sorted_indices = np.argsort(projections)
            
            # 采样点
            num_points = min(20, len(contour) // 2)
            step = len(sorted_indices) // num_points if num_points > 0 else 1
            
            centerline = []
            for i in range(0, len(sorted_indices), step):
                idx = sorted_indices[i]
                centerline.append((float(contour_array[idx][0]), float(contour_array[idx][1])))
            
            return centerline
            
        except Exception as e:
            self.logger.error(f"Error converting contour to centerline: {str(e)}")
            return [(float(contour[0][0]), float(contour[0][1])), 
                   (float(contour[-1][0]), float(contour[-1][1]))]
    
    def _bbox_to_trajectory(self, bbox: Tuple[int, int, int, int], 
                          stroke_type: str) -> List[Tuple[float, float]]:
        """
        根据边界框生成轨迹
        
        Args:
            bbox (Tuple[int, int, int, int]): 边界框 (x, y, w, h)
            stroke_type (str): 笔画类型
            
        Returns:
            List[Tuple[float, float]]: 轨迹点
        """
        try:
            x, y, w, h = bbox
            
            if stroke_type == 'horizontal':
                # 水平笔画：从左到右
                return [(float(x), float(y + h // 2)), (float(x + w), float(y + h // 2))]
            elif stroke_type == 'vertical':
                # 垂直笔画：从上到下
                return [(float(x + w // 2), float(y)), (float(x + w // 2), float(y + h))]
            elif stroke_type == 'left_falling':
                # 撇：从右上到左下
                return [(float(x + w), float(y)), (float(x), float(y + h))]
            elif stroke_type == 'right_falling':
                # 捺：从左上到右下
                return [(float(x), float(y)), (float(x + w), float(y + h))]
            elif stroke_type == 'dot':
                # 点：中心点
                center = (float(x + w // 2), float(y + h // 2))
                return [center, center]
            else:
                # 默认：对角线
                return [(float(x), float(y)), (float(x + w), float(y + h))]
                
        except Exception as e:
            self.logger.error(f"Error generating trajectory from bbox: {str(e)}")
            return [(0.0, 0.0), (100.0, 0.0)]
    
    def _smooth_trajectory(self, trajectory: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        平滑轨迹
        
        Args:
            trajectory (List[Tuple[float, float]]): 原始轨迹
            
        Returns:
            List[Tuple[float, float]]: 平滑后的轨迹
        """
        try:
            if len(trajectory) < 3:
                return trajectory
            
            # 转换为numpy数组
            points = np.array(trajectory)
            
            # 使用样条插值平滑
            # 计算累积距离作为参数
            distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)
            
            # 归一化距离
            # 安全比较，避免数组真值歧义
            last_distance = distances[-1]
            if isinstance(last_distance, np.ndarray):
                last_distance_val = last_distance.item() if last_distance.size == 1 else last_distance
            else:
                last_distance_val = last_distance
                
            if last_distance_val > 0:
                t = distances / distances[-1]
            else:
                t = np.linspace(0, 1, len(points))
            
            # 样条插值
            try:
                tck, u = splprep([points[:, 0], points[:, 1]], s=self.trajectory_smoothness, k=min(3, len(points)-1))
                
                # 生成平滑轨迹
                u_new = np.linspace(0, 1, self.curve_resolution)
                smooth_points = splev(u_new, tck)
                
                smoothed_trajectory = [(float(x), float(y)) for x, y in zip(smooth_points[0], smooth_points[1])]
                
                return smoothed_trajectory
                
            except Exception:
                # 如果样条插值失败，使用线性插值
                f_x = interp1d(t, points[:, 0], kind='linear')
                f_y = interp1d(t, points[:, 1], kind='linear')
                
                t_new = np.linspace(0, 1, self.curve_resolution)
                smooth_x = f_x(t_new)
                smooth_y = f_y(t_new)
                
                return [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]
            
        except Exception as e:
            self.logger.error(f"Error smoothing trajectory: {str(e)}")
            return trajectory
    
    def _calculate_stroke_duration(self, stroke: Dict[str, Any]) -> float:
        """
        计算笔画动画持续时间
        
        Args:
            stroke (Dict): 笔画数据
            
        Returns:
            float: 持续时间（秒）
        """
        try:
            # 基础持续时间
            base_duration = self.stroke_duration_base
            
            # 根据笔画复杂度调整
            complexity = stroke.get('complexity', 1.0)
            complexity_factor = 0.5 + complexity * 0.5  # 0.5 到 1.0
            
            # 根据笔画长度调整
            arc_length = stroke.get('arc_length', 100)
            length_factor = max(0.3, min(2.0, arc_length / 100.0))  # 0.3 到 2.0
            
            # 根据笔画类型调整
            stroke_type = stroke.get('stroke_class', 'horizontal')
            type_factors = {
                'dot': 0.3,
                'horizontal': 0.8,
                'vertical': 0.9,
                'left_falling': 1.0,
                'right_falling': 1.0,
                'hook': 1.2,
                'turning': 1.3,
                'curve': 1.4,
                'complex': 1.5
            }
            type_factor = type_factors.get(stroke_type, 1.0)
            
            # 添加随机变化
            variation = 1.0 + (np.random.random() - 0.5) * self.speed_variation
            
            # 计算最终持续时间
            duration = base_duration * complexity_factor * length_factor * type_factor * variation
            
            # 限制在合理范围内
            duration = max(0.5, min(10.0, duration))
            
            return duration
            
        except Exception as e:
            self.logger.error(f"Error calculating stroke duration: {str(e)}")
            return self.stroke_duration_base
    
    def _generate_timing_curve(self, trajectory: List[Tuple[float, float]], 
                             duration: float) -> List[float]:
        """
        生成时间曲线
        
        Args:
            trajectory (List[Tuple[float, float]]): 轨迹点
            duration (float): 持续时间
            
        Returns:
            List[float]: 时间点列表
        """
        try:
            if len(trajectory) <= 1:
                return [0.0]
            
            # 计算累积距离
            points = np.array(trajectory)
            distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
            distances = np.insert(distances, 0, 0)
            
            # 归一化距离
            # 安全比较，避免数组真值歧义
            last_distance = distances[-1]
            if isinstance(last_distance, np.ndarray):
                last_distance_val = last_distance.item() if last_distance.size == 1 else last_distance
            else:
                last_distance_val = last_distance
                
            if last_distance_val > 0:
                normalized_distances = distances / distances[-1]
            else:
                normalized_distances = np.linspace(0, 1, len(trajectory))
            
            # 应用速度变化曲线（起笔慢，中间快，收笔慢）
            # 使用贝塞尔曲线形状
            t = normalized_distances
            speed_curve = 3 * t**2 - 2 * t**3  # S型曲线
            
            # 转换为时间
            timing = speed_curve * duration
            
            return timing.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating timing curve: {str(e)}")
            return np.linspace(0, duration, len(trajectory)).tolist()
    
    def _generate_pressure_curve(self, trajectory: List[Tuple[float, float]], 
                               timing: List[float]) -> List[float]:
        """
        生成压力曲线
        
        Args:
            trajectory (List[Tuple[float, float]]): 轨迹点
            timing (List[float]): 时间点
            
        Returns:
            List[float]: 压力值列表
        """
        try:
            if len(timing) <= 1:
                return [0.5]
            
            # 归一化时间
            max_time = max(timing)
            if max_time > 0:
                t = np.array(timing) / max_time
            else:
                t = np.linspace(0, 1, len(timing))
            
            # 生成压力曲线（起笔轻，中间重，收笔轻）
            # 使用高斯函数的变形
            pressure = np.exp(-2 * (t - 0.5)**2) * 0.8 + 0.2  # 0.2 到 1.0
            
            # 添加细微变化
            noise = np.random.normal(0, 0.05, len(pressure))
            pressure = np.clip(pressure + noise, 0.1, 1.0)
            
            return pressure.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating pressure curve: {str(e)}")
            return [0.5] * len(timing)
    
    def _generate_velocity_curve(self, trajectory: List[Tuple[float, float]], 
                               timing: List[float]) -> List[float]:
        """
        生成速度曲线
        
        Args:
            trajectory (List[Tuple[float, float]]): 轨迹点
            timing (List[float]): 时间点
            
        Returns:
            List[float]: 速度值列表
        """
        try:
            if len(trajectory) <= 1 or len(timing) <= 1:
                return [1.0]
            
            # 计算瞬时速度
            velocities = [0.0]
            
            for i in range(1, len(trajectory)):
                # 计算距离
                dx = trajectory[i][0] - trajectory[i-1][0]
                dy = trajectory[i][1] - trajectory[i-1][1]
                distance = math.sqrt(dx**2 + dy**2)
                
                # 计算时间差
                dt = timing[i] - timing[i-1]
                
                # 计算速度
                if dt > 0:
                    velocity = distance / dt
                else:
                    velocity = 0.0
                
                velocities.append(velocity)
            
            # 归一化速度
            if max(velocities) > 0:
                velocities = np.array(velocities) / max(velocities)
            else:
                velocities = np.ones(len(velocities))
            
            # 平滑速度曲线
            if len(velocities) > 3:
                from scipy.ndimage import gaussian_filter1d
                velocities = gaussian_filter1d(velocities, sigma=1.0)
            
            return velocities.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating velocity curve: {str(e)}")
            return [1.0] * len(timing)
    
    def _generate_brush_size_curve(self, pressure_curve: List[float]) -> List[float]:
        """
        生成笔刷大小曲线
        
        Args:
            pressure_curve (List[float]): 压力曲线
            
        Returns:
            List[float]: 笔刷大小列表
        """
        try:
            # 基于压力计算笔刷大小
            brush_sizes = []
            
            for pressure in pressure_curve:
                # 压力越大，笔刷越大
                size = self.min_brush_size + (self.max_brush_size - self.min_brush_size) * pressure
                brush_sizes.append(size)
            
            return brush_sizes
            
        except Exception as e:
            self.logger.error(f"Error generating brush size curve: {str(e)}")
            return [self.default_brush_size] * len(pressure_curve)
    
    def _generate_ink_flow_curve(self, velocity_curve: List[float], 
                               pressure_curve: List[float]) -> List[float]:
        """
        生成墨水流量曲线
        
        Args:
            velocity_curve (List[float]): 速度曲线
            pressure_curve (List[float]): 压力曲线
            
        Returns:
            List[float]: 墨水流量列表
        """
        try:
            ink_flows = []
            
            for velocity, pressure in zip(velocity_curve, pressure_curve):
                # 墨水流量与压力正相关，与速度负相关
                flow = self.ink_flow_rate * pressure * (1.0 - 0.3 * velocity)
                flow = max(0.1, min(1.0, flow))
                ink_flows.append(flow)
            
            return ink_flows
            
        except Exception as e:
            self.logger.error(f"Error generating ink flow curve: {str(e)}")
            return [self.ink_flow_rate] * len(velocity_curve)
    
    def _generate_stroke_frames(self, trajectory: List[Tuple[float, float]], 
                              timing: List[float], pressure_curve: List[float],
                              velocity_curve: List[float], brush_size_curve: List[float],
                              ink_flow_curve: List[float], start_time: float,
                              stroke_id: int) -> List[AnimationFrame]:
        """
        生成笔画动画帧
        
        Args:
            trajectory (List[Tuple[float, float]]): 轨迹点
            timing (List[float]): 时间点
            pressure_curve (List[float]): 压力曲线
            velocity_curve (List[float]): 速度曲线
            brush_size_curve (List[float]): 笔刷大小曲线
            ink_flow_curve (List[float]): 墨水流量曲线
            start_time (float): 开始时间
            stroke_id (int): 笔画ID
            
        Returns:
            List[AnimationFrame]: 动画帧列表
        """
        try:
            frames = []
            
            for i, (pos, time_offset, pressure, velocity, brush_size, ink_flow) in enumerate(
                zip(trajectory, timing, pressure_curve, velocity_curve, brush_size_curve, ink_flow_curve)
            ):
                
                # 计算笔刷角度（基于运动方向）
                if i > 0:
                    dx = pos[0] - trajectory[i-1][0]
                    dy = pos[1] - trajectory[i-1][1]
                    angle = math.atan2(dy, dx)
                else:
                    angle = 0.0
                
                # 计算笔画进度
                progress = i / (len(trajectory) - 1) if len(trajectory) > 1 else 1.0
                
                frame = AnimationFrame(
                    frame_id=len(frames),
                    timestamp=start_time + time_offset,
                    brush_position=pos,
                    brush_pressure=pressure,
                    brush_angle=angle,
                    brush_size=brush_size,
                    ink_flow=ink_flow,
                    stroke_progress=progress,
                    canvas_state=None,  # 将在渲染时填充
                    metadata={
                        'stroke_id': stroke_id,
                        'point_index': i,
                        'velocity': velocity
                    }
                )
                
                frames.append(frame)
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Error generating stroke frames: {str(e)}")
            return []
    
    def _combine_stroke_animations(self, stroke_animations: List[StrokeAnimation]) -> List[AnimationFrame]:
        """
        合并所有笔画动画帧
        
        Args:
            stroke_animations (List[StrokeAnimation]): 笔画动画列表
            
        Returns:
            List[AnimationFrame]: 所有动画帧
        """
        try:
            all_frames = []
            
            # 收集所有帧
            for stroke_anim in stroke_animations:
                all_frames.extend(stroke_anim.frames)
            
            # 按时间戳排序
            all_frames.sort(key=lambda f: f.timestamp)
            
            # 重新分配帧ID
            for i, frame in enumerate(all_frames):
                frame.frame_id = i
            
            return all_frames
            
        except Exception as e:
            self.logger.error(f"Error combining stroke animations: {str(e)}")
            return []
    
    def save_animation(self, animation: PaintingAnimation, output_path: str) -> bool:
        """
        保存动画数据
        
        Args:
            animation (PaintingAnimation): 绘画动画
            output_path (str): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 准备序列化数据
            animation_data = {
                'animation_id': animation.animation_id,
                'total_duration': animation.total_duration,
                'fps': animation.fps,
                'canvas_size': animation.canvas_size,
                'background_color': animation.background_color,
                'metadata': animation.metadata,
                'stroke_animations': []
            }
            
            # 序列化笔画动画
            for stroke_anim in animation.stroke_animations:
                stroke_data = {
                    'stroke_id': stroke_anim.stroke_id,
                    'trajectory': stroke_anim.trajectory,
                    'timing': stroke_anim.timing,
                    'pressure_curve': stroke_anim.pressure_curve,
                    'velocity_curve': stroke_anim.velocity_curve,
                    'brush_size_curve': stroke_anim.brush_size_curve,
                    'ink_flow_curve': stroke_anim.ink_flow_curve,
                    'duration': stroke_anim.duration,
                    'metadata': stroke_anim.metadata
                }
                animation_data['stroke_animations'].append(stroke_data)
            
            # 保存到文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(animation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Animation saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving animation: {str(e)}")
            return False
    
    def load_animation(self, input_path: str) -> Optional[PaintingAnimation]:
        """
        加载动画数据
        
        Args:
            input_path (str): 输入路径
            
        Returns:
            Optional[PaintingAnimation]: 绘画动画
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                animation_data = json.load(f)
            
            # 重建笔画动画
            stroke_animations = []
            for stroke_data in animation_data['stroke_animations']:
                # 重建动画帧（简化版本）
                frames = []
                trajectory = stroke_data['trajectory']
                timing = stroke_data['timing']
                pressure_curve = stroke_data['pressure_curve']
                
                for i, (pos, time_offset, pressure) in enumerate(zip(trajectory, timing, pressure_curve)):
                    frame = AnimationFrame(
                        frame_id=i,
                        timestamp=time_offset,
                        brush_position=tuple(pos),
                        brush_pressure=pressure,
                        brush_angle=0.0,
                        brush_size=stroke_data['brush_size_curve'][i],
                        ink_flow=stroke_data['ink_flow_curve'][i],
                        stroke_progress=i / (len(trajectory) - 1) if len(trajectory) > 1 else 1.0,
                        canvas_state=None,
                        metadata={'stroke_id': stroke_data['stroke_id']}
                    )
                    frames.append(frame)
                
                stroke_anim = StrokeAnimation(
                    stroke_id=stroke_data['stroke_id'],
                    trajectory=stroke_data['trajectory'],
                    timing=stroke_data['timing'],
                    pressure_curve=stroke_data['pressure_curve'],
                    velocity_curve=stroke_data['velocity_curve'],
                    brush_size_curve=stroke_data['brush_size_curve'],
                    ink_flow_curve=stroke_data['ink_flow_curve'],
                    duration=stroke_data['duration'],
                    frames=frames,
                    metadata=stroke_data['metadata']
                )
                stroke_animations.append(stroke_anim)
            
            # 重建完整动画
            all_frames = self._combine_stroke_animations(stroke_animations)
            
            animation = PaintingAnimation(
                animation_id=animation_data['animation_id'],
                stroke_animations=stroke_animations,
                total_duration=animation_data['total_duration'],
                fps=animation_data['fps'],
                canvas_size=tuple(animation_data['canvas_size']),
                background_color=tuple(animation_data['background_color']),
                all_frames=all_frames,
                metadata=animation_data['metadata']
            )
            
            self.logger.info(f"Animation loaded from {input_path}")
            return animation
            
        except Exception as e:
            self.logger.error(f"Error loading animation: {str(e)}")
            return None
    
    def _create_default_animation(self, strokes: List[Dict[str, Any]]) -> PaintingAnimation:
        """
        创建默认动画
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            PaintingAnimation: 默认动画
        """
        return PaintingAnimation(
            animation_id="default_animation",
            stroke_animations=[],
            total_duration=1.0,
            fps=self.fps,
            canvas_size=self.canvas_size,
            background_color=self.background_color,
            all_frames=[],
            metadata={'error': 'Failed to create animation'}
        )
    
    def _create_default_stroke_animation(self, stroke_id: int, start_time: float) -> StrokeAnimation:
        """
        创建默认笔画动画
        
        Args:
            stroke_id (int): 笔画ID
            start_time (float): 开始时间
            
        Returns:
            StrokeAnimation: 默认笔画动画
        """
        return StrokeAnimation(
            stroke_id=stroke_id,
            trajectory=[(0, 0), (100, 0)],
            timing=[0.0, 1.0],
            pressure_curve=[0.5, 0.5],
            velocity_curve=[1.0, 1.0],
            brush_size_curve=[5.0, 5.0],
            ink_flow_curve=[0.7, 0.7],
            duration=1.0,
            frames=[],
            metadata={'error': 'Default stroke animation'}
        )
    
    def get_animation_statistics(self, animation: PaintingAnimation) -> Dict[str, Any]:
        """
        获取动画统计信息
        
        Args:
            animation (PaintingAnimation): 绘画动画
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'animation_id': animation.animation_id,
                'total_duration': animation.total_duration,
                'fps': animation.fps,
                'total_frames': len(animation.all_frames),
                'num_strokes': len(animation.stroke_animations),
                'canvas_size': animation.canvas_size,
                'stroke_statistics': []
            }
            
            for stroke_anim in animation.stroke_animations:
                stroke_stats = {
                    'stroke_id': stroke_anim.stroke_id,
                    'duration': stroke_anim.duration,
                    'num_frames': len(stroke_anim.frames),
                    'trajectory_length': len(stroke_anim.trajectory),
                    'avg_pressure': np.mean(stroke_anim.pressure_curve),
                    'avg_velocity': np.mean(stroke_anim.velocity_curve),
                    'avg_brush_size': np.mean(stroke_anim.brush_size_curve),
                    'avg_ink_flow': np.mean(stroke_anim.ink_flow_curve)
                }
                stats['stroke_statistics'].append(stroke_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting animation statistics: {str(e)}")
            return {'error': str(e)}