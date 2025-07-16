# -*- coding: utf-8 -*-
"""
Flood渲染器

实现论文中的flood-filling算法：
1. 模拟笔触绘制过程
2. 8邻域和12邻域种子扩散
3. 椭圆足迹模型
4. 动态绘制速度调整
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, ellipse


class NeighborhoodType(Enum):
    """
    邻域类型枚举
    """
    FOUR_CONNECTED = "4_connected"    # 4邻域
    EIGHT_CONNECTED = "8_connected"   # 8邻域
    TWELVE_CONNECTED = "12_connected" # 12邻域
    CUSTOM = "custom"                 # 自定义邻域


class FloodMode(Enum):
    """
    填充模式枚举
    """
    UNIFORM = "uniform"           # 均匀填充
    GRADIENT = "gradient"         # 梯度填充
    TEXTURED = "textured"         # 纹理填充
    ARTISTIC = "artistic"         # 艺术化填充


@dataclass
class FloodSeed:
    """
    填充种子数据结构
    """
    position: Tuple[int, int]
    color: Tuple[int, int, int]
    intensity: float = 1.0
    radius: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EllipticalFootprint:
    """
    椭圆足迹数据结构
    """
    center: Tuple[int, int]
    major_axis: float
    minor_axis: float
    angle: float  # 旋转角度（弧度）
    intensity: float = 1.0
    color: Tuple[int, int, int] = (0, 0, 0)


@dataclass
class FloodParameters:
    """
    填充参数
    """
    neighborhood_type: NeighborhoodType = NeighborhoodType.EIGHT_CONNECTED
    flood_mode: FloodMode = FloodMode.ARTISTIC
    
    # 速度参数
    base_speed: float = 1.0
    speed_variation: float = 0.2
    thickness_speed_factor: float = 0.5
    
    # 扩散参数
    max_iterations: int = 1000
    convergence_threshold: float = 0.01
    boundary_smoothing: bool = True
    
    # 视觉效果参数
    opacity_decay: float = 0.95
    color_blending: bool = True
    edge_softening: bool = True
    
    # 椭圆足迹参数
    use_elliptical_footprint: bool = True
    footprint_aspect_ratio: float = 2.0
    footprint_orientation_adaptive: bool = True


class FloodRenderer:
    """
    Flood渲染器
    
    使用flood-filling算法模拟笔触绘制过程
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化渲染器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 渲染参数
        self.canvas_size = config.get('canvas_size', (800, 600))
        self.background_color = config.get('background_color', (255, 255, 255))
        
        # 邻域定义
        self._init_neighborhoods()
        
        # 渲染缓存
        self._distance_cache = {}
        self._footprint_cache = {}
        
        # 性能统计
        self.render_stats = {
            'total_pixels_filled': 0,
            'total_iterations': 0,
            'average_speed': 0.0,
            'render_time': 0.0
        }
    
    def _init_neighborhoods(self):
        """
        初始化邻域定义
        """
        # 4邻域
        self.neighborhoods = {
            NeighborhoodType.FOUR_CONNECTED: [
                (0, 1), (1, 0), (0, -1), (-1, 0)
            ],
            # 8邻域
            NeighborhoodType.EIGHT_CONNECTED: [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ],
            # 12邻域（包含更远的邻居）
            NeighborhoodType.TWELVE_CONNECTED: [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
                (-2, 0),  (2, 0),  (0, -2), (0, 2)
            ]
        }
    
    def render_stroke(self, stroke_mask: np.ndarray, stroke_features: Dict[str, Any], 
                     parameters: FloodParameters) -> List[np.ndarray]:
        """
        渲染笔触
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            parameters: 填充参数
            
        Returns:
            List[np.ndarray]: 渲染帧序列
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info(f"Rendering stroke with {parameters.flood_mode.value} mode")
            
            # 初始化画布
            canvas = np.full(
                (self.canvas_size[1], self.canvas_size[0], 3),
                self.background_color,
                dtype=np.uint8
            )
            
            # 生成填充种子
            seeds = self._generate_flood_seeds(stroke_mask, stroke_features, parameters)
            
            # 执行flood filling
            frames = self._perform_flood_filling(canvas, stroke_mask, seeds, parameters)
            
            # 更新统计信息
            self.render_stats['render_time'] = time.time() - start_time
            self.render_stats['total_iterations'] = len(frames)
            
            self.logger.info(f"Stroke rendered in {self.render_stats['render_time']:.3f}s with {len(frames)} frames")
            
            return frames
            
        except Exception as e:
            self.logger.error(f"Error rendering stroke: {str(e)}")
            return []
    
    def _generate_flood_seeds(self, stroke_mask: np.ndarray, stroke_features: Dict[str, Any], 
                            parameters: FloodParameters) -> List[FloodSeed]:
        """
        生成填充种子
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            parameters: 填充参数
            
        Returns:
            List[FloodSeed]: 种子列表
        """
        seeds = []
        
        # 获取笔触的骨架点作为种子起始位置
        skeleton_points = stroke_features.get('skeleton_points', [])
        if not skeleton_points:
            # 如果没有骨架点，使用质心作为起始点
            y_coords, x_coords = np.where(stroke_mask > 0)
            if len(x_coords) > 0:
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                skeleton_points = [(centroid_x, centroid_y)]
        
        # 获取笔触颜色
        stroke_color = stroke_features.get('color', (0, 0, 0))
        
        # 根据笔触方向确定种子顺序
        direction = stroke_features.get('direction')
        if direction and hasattr(direction, 'path_points') and direction.path_points:
            # 使用路径点作为种子
            for i, point in enumerate(direction.path_points):
                if self._is_point_in_mask(point, stroke_mask):
                    # 计算种子参数
                    intensity = self._calculate_seed_intensity(point, stroke_features, i, len(direction.path_points))
                    radius = self._calculate_seed_radius(point, stroke_features)
                    timestamp = i / max(1, len(direction.path_points) - 1)
                    
                    seed = FloodSeed(
                        position=point,
                        color=stroke_color,
                        intensity=intensity,
                        radius=radius,
                        timestamp=timestamp,
                        metadata={'path_index': i}
                    )
                    seeds.append(seed)
        else:
            # 使用骨架点作为种子
            for i, point in enumerate(skeleton_points):
                if self._is_point_in_mask(point, stroke_mask):
                    intensity = 1.0 - (i / max(1, len(skeleton_points) - 1)) * 0.3  # 逐渐减弱
                    radius = stroke_features.get('thickness', 0.5) * 10 + 2
                    
                    seed = FloodSeed(
                        position=point,
                        color=stroke_color,
                        intensity=intensity,
                        radius=radius,
                        timestamp=i / max(1, len(skeleton_points) - 1),
                        metadata={'skeleton_index': i}
                    )
                    seeds.append(seed)
        
        # 如果没有有效种子，创建默认种子
        if not seeds:
            y_coords, x_coords = np.where(stroke_mask > 0)
            if len(x_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                seed = FloodSeed(
                    position=(center_x, center_y),
                    color=stroke_color,
                    intensity=1.0,
                    radius=5.0,
                    timestamp=0.0
                )
                seeds.append(seed)
        
        return seeds
    
    def _perform_flood_filling(self, canvas: np.ndarray, stroke_mask: np.ndarray, 
                             seeds: List[FloodSeed], parameters: FloodParameters) -> List[np.ndarray]:
        """
        执行flood filling
        
        Args:
            canvas: 画布
            stroke_mask: 笔触掩码
            seeds: 种子列表
            parameters: 填充参数
            
        Returns:
            List[np.ndarray]: 帧序列
        """
        frames = []
        current_canvas = canvas.copy()
        
        # 初始化填充状态
        filled_mask = np.zeros(stroke_mask.shape, dtype=bool)
        active_seeds = deque()
        
        # 计算每个种子的激活时间
        total_seeds = len(seeds)
        seed_activation_frames = []
        
        for i, seed in enumerate(seeds):
            # 根据时间戳计算激活帧
            activation_frame = int(seed.timestamp * parameters.max_iterations)
            seed_activation_frames.append(activation_frame)
        
        # 执行迭代填充
        for iteration in range(parameters.max_iterations):
            # 激活新种子
            for i, seed in enumerate(seeds):
                if seed_activation_frames[i] == iteration:
                    active_seeds.append(seed)
            
            if not active_seeds:
                continue
            
            # 当前帧的变化
            frame_changed = False
            new_active_seeds = deque()
            
            # 处理所有活跃种子
            while active_seeds:
                seed = active_seeds.popleft()
                
                # 执行单步扩散
                expanded_seeds = self._expand_seed(
                    seed, current_canvas, stroke_mask, filled_mask, parameters
                )
                
                if expanded_seeds:
                    frame_changed = True
                    new_active_seeds.extend(expanded_seeds)
            
            # 更新活跃种子
            active_seeds = new_active_seeds
            
            # 如果有变化，保存当前帧
            if frame_changed:
                frames.append(current_canvas.copy())
            
            # 检查收敛条件
            if not active_seeds:
                break
            
            # 检查填充完成度
            filled_count = float(np.sum(filled_mask))
            stroke_count = float(np.sum(stroke_mask > 0))
            filled_ratio = filled_count / max(1, stroke_count)
            if filled_ratio >= 1.0 - parameters.convergence_threshold:
                break
        
        # 确保至少有一帧
        if not frames:
            frames.append(current_canvas)
        
        return frames
    
    def _expand_seed(self, seed: FloodSeed, canvas: np.ndarray, stroke_mask: np.ndarray, 
                   filled_mask: np.ndarray, parameters: FloodParameters) -> List[FloodSeed]:
        """
        扩展单个种子
        
        Args:
            seed: 当前种子
            canvas: 画布
            stroke_mask: 笔触掩码
            filled_mask: 已填充掩码
            parameters: 填充参数
            
        Returns:
            List[FloodSeed]: 新生成的种子
        """
        new_seeds = []
        x, y = seed.position
        
        # 检查当前位置是否有效
        if (not self._is_valid_position(x, y, canvas.shape) or
            filled_mask[y, x] or stroke_mask[y, x] == 0):
            return new_seeds
        
        # 绘制当前种子
        if parameters.use_elliptical_footprint:
            self._draw_elliptical_footprint(canvas, seed, parameters)
        else:
            self._draw_circular_footprint(canvas, seed, parameters)
        
        # 标记为已填充
        filled_mask[y, x] = True
        
        # 获取邻域
        neighborhood = self.neighborhoods[parameters.neighborhood_type]
        
        # 扩展到邻居
        for dx, dy in neighborhood:
            nx, ny = x + dx, y + dy
            
            if (self._is_valid_position(nx, ny, canvas.shape) and
                not filled_mask[ny, nx] and stroke_mask[ny, nx] > 0):
                
                # 计算新种子的参数
                new_intensity = seed.intensity * parameters.opacity_decay
                new_radius = self._calculate_propagated_radius(seed, (nx, ny), parameters)
                
                # 只有强度足够的种子才继续传播
                if new_intensity > 0.1:
                    new_seed = FloodSeed(
                        position=(nx, ny),
                        color=seed.color,
                        intensity=new_intensity,
                        radius=new_radius,
                        timestamp=seed.timestamp,
                        metadata=seed.metadata.copy()
                    )
                    new_seeds.append(new_seed)
        
        return new_seeds
    
    def _draw_elliptical_footprint(self, canvas: np.ndarray, seed: FloodSeed, 
                                 parameters: FloodParameters):
        """
        绘制椭圆足迹
        
        Args:
            canvas: 画布
            seed: 种子
            parameters: 参数
        """
        x, y = seed.position
        
        # 计算椭圆参数
        major_axis = seed.radius
        minor_axis = seed.radius / parameters.footprint_aspect_ratio
        
        # 计算方向角（如果启用自适应方向）
        angle = 0.0
        if parameters.footprint_orientation_adaptive:
            # 根据种子的元数据或周围梯度确定方向
            angle = self._calculate_footprint_orientation(seed, canvas)
        
        # 创建椭圆足迹
        footprint = EllipticalFootprint(
            center=(x, y),
            major_axis=major_axis,
            minor_axis=minor_axis,
            angle=angle,
            intensity=seed.intensity,
            color=seed.color
        )
        
        # 绘制椭圆
        self._render_elliptical_footprint(canvas, footprint, parameters)
    
    def _draw_circular_footprint(self, canvas: np.ndarray, seed: FloodSeed, 
                               parameters: FloodParameters):
        """
        绘制圆形足迹
        
        Args:
            canvas: 画布
            seed: 种子
            parameters: 参数
        """
        x, y = seed.position
        radius = int(seed.radius)
        
        # 创建圆形掩码
        y_min = max(0, y - radius)
        y_max = min(canvas.shape[0], y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(canvas.shape[1], x + radius + 1)
        
        # 绘制圆形区域
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                distance = float(np.sqrt((px - x)**2 + (py - y)**2))
                if distance <= radius:
                    # 计算混合权重
                    if parameters.edge_softening:
                        weight = seed.intensity * (1.0 - distance / radius)
                    else:
                        weight = seed.intensity
                    
                    # 混合颜色
                    if parameters.color_blending:
                        self._blend_pixel(canvas, px, py, seed.color, weight)
                    else:
                        canvas[py, px] = seed.color
    
    def _render_elliptical_footprint(self, canvas: np.ndarray, footprint: EllipticalFootprint, 
                                   parameters: FloodParameters):
        """
        渲染椭圆足迹
        
        Args:
            canvas: 画布
            footprint: 椭圆足迹
            parameters: 参数
        """
        x, y = footprint.center
        a = footprint.major_axis
        b = footprint.minor_axis
        angle = footprint.angle
        
        # 计算椭圆的边界框
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 椭圆的外接矩形
        extent_x = int(np.ceil(a * abs(cos_angle) + b * abs(sin_angle)))
        extent_y = int(np.ceil(a * abs(sin_angle) + b * abs(cos_angle)))
        
        x_min = max(0, x - extent_x)
        x_max = min(canvas.shape[1], x + extent_x + 1)
        y_min = max(0, y - extent_y)
        y_max = min(canvas.shape[0], y + extent_y + 1)
        
        # 绘制椭圆区域
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                # 转换到椭圆坐标系
                dx = px - x
                dy = py - y
                
                # 旋转坐标
                rx = dx * cos_angle + dy * sin_angle
                ry = -dx * sin_angle + dy * cos_angle
                
                # 检查是否在椭圆内
                ellipse_value = (rx / a)**2 + (ry / b)**2
                
                if ellipse_value <= 1.0:
                    # 计算混合权重
                    if parameters.edge_softening:
                        weight = footprint.intensity * (1.0 - ellipse_value)
                    else:
                        weight = footprint.intensity
                    
                    # 混合颜色
                    if parameters.color_blending:
                        self._blend_pixel(canvas, px, py, footprint.color, weight)
                    else:
                        canvas[py, px] = footprint.color
    
    def _blend_pixel(self, canvas: np.ndarray, x: int, y: int, 
                    color: Tuple[int, int, int], weight: float):
        """
        混合像素颜色
        
        Args:
            canvas: 画布
            x: X坐标
            y: Y坐标
            color: 新颜色
            weight: 混合权重
        """
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            current_color = canvas[y, x].astype(np.float32)
            new_color = np.array(color, dtype=np.float32)
            
            # Alpha混合
            blended_color = current_color * (1 - weight) + new_color * weight
            canvas[y, x] = blended_color.astype(np.uint8)
    
    def _calculate_seed_intensity(self, point: Tuple[int, int], stroke_features: Dict[str, Any], 
                                index: int, total_points: int) -> float:
        """
        计算种子强度
        
        Args:
            point: 种子位置
            stroke_features: 笔触特征
            index: 路径索引
            total_points: 总点数
            
        Returns:
            float: 强度值
        """
        # 基础强度
        base_intensity = 1.0
        
        # 根据路径位置调整（开始和结束较弱）
        progress = index / max(1, total_points - 1)
        position_factor = 0.5 + 0.5 * np.sin(progress * np.pi)  # 中间强，两端弱
        
        # 根据笔触特征调整
        wetness = stroke_features.get('wetness', 0.5)
        thickness = stroke_features.get('thickness', 0.5)
        
        # 湿度和厚度影响强度
        feature_factor = 0.5 + 0.3 * wetness + 0.2 * thickness
        
        return base_intensity * position_factor * feature_factor
    
    def _calculate_seed_radius(self, point: Tuple[int, int], stroke_features: Dict[str, Any]) -> float:
        """
        计算种子半径
        
        Args:
            point: 种子位置
            stroke_features: 笔触特征
            
        Returns:
            float: 半径值
        """
        # 基础半径
        base_radius = 3.0
        
        # 根据厚度调整
        thickness = stroke_features.get('thickness', 0.5)
        thickness_factor = 0.5 + thickness * 1.5
        
        # 根据尺度调整
        scale = stroke_features.get('scale', 0.5)
        scale_factor = 0.8 + scale * 0.4
        
        return base_radius * thickness_factor * scale_factor
    
    def _calculate_propagated_radius(self, parent_seed: FloodSeed, new_position: Tuple[int, int], 
                                   parameters: FloodParameters) -> float:
        """
        计算传播后的半径
        
        Args:
            parent_seed: 父种子
            new_position: 新位置
            parameters: 参数
            
        Returns:
            float: 新半径
        """
        # 半径逐渐衰减
        decay_factor = 0.95
        return parent_seed.radius * decay_factor
    
    def _calculate_footprint_orientation(self, seed: FloodSeed, canvas: np.ndarray) -> float:
        """
        计算足迹方向
        
        Args:
            seed: 种子
            canvas: 画布
            
        Returns:
            float: 方向角（弧度）
        """
        x, y = seed.position
        
        # 计算局部梯度
        if (x > 0 and x < canvas.shape[1] - 1 and 
            y > 0 and y < canvas.shape[0] - 1):
            
            # 转换为灰度
            gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
            
            # 计算梯度
            gx = float(gray[y, x + 1]) - float(gray[y, x - 1])
            gy = float(gray[y + 1, x]) - float(gray[y - 1, x])
            
            # 计算角度
            if gx != 0 or gy != 0:
                angle = np.arctan2(gy, gx)
                return angle
        
        return 0.0  # 默认水平方向
    
    def _is_point_in_mask(self, point: Tuple[int, int], mask: np.ndarray) -> bool:
        """
        检查点是否在掩码内
        
        Args:
            point: 点坐标
            mask: 掩码
            
        Returns:
            bool: 是否在掩码内
        """
        x, y = point
        return (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0)
    
    def _is_valid_position(self, x: int, y: int, shape: Tuple[int, ...]) -> bool:
        """
        检查位置是否有效
        
        Args:
            x: X坐标
            y: Y坐标
            shape: 画布形状
            
        Returns:
            bool: 是否有效
        """
        return 0 <= x < shape[1] and 0 <= y < shape[0]
    
    def create_custom_neighborhood(self, pattern: List[Tuple[int, int]]) -> NeighborhoodType:
        """
        创建自定义邻域
        
        Args:
            pattern: 邻域模式
            
        Returns:
            NeighborhoodType: 邻域类型
        """
        self.neighborhoods[NeighborhoodType.CUSTOM] = pattern
        return NeighborhoodType.CUSTOM
    
    def optimize_rendering_speed(self, stroke_features: Dict[str, Any]) -> FloodParameters:
        """
        优化渲染速度
        
        Args:
            stroke_features: 笔触特征
            
        Returns:
            FloodParameters: 优化后的参数
        """
        # 根据笔触复杂度选择合适的参数
        complexity = stroke_features.get('complexity', 0.5)
        area = stroke_features.get('area', 100)
        
        if complexity < 0.3 and area < 500:
            # 简单小笔触：使用快速模式
            return FloodParameters(
                neighborhood_type=NeighborhoodType.FOUR_CONNECTED,
                flood_mode=FloodMode.UNIFORM,
                max_iterations=200,
                use_elliptical_footprint=False,
                edge_softening=False
            )
        elif complexity > 0.7 or area > 2000:
            # 复杂大笔触：使用高质量模式
            return FloodParameters(
                neighborhood_type=NeighborhoodType.TWELVE_CONNECTED,
                flood_mode=FloodMode.ARTISTIC,
                max_iterations=1000,
                use_elliptical_footprint=True,
                edge_softening=True,
                boundary_smoothing=True
            )
        else:
            # 中等复杂度：使用平衡模式
            return FloodParameters(
                neighborhood_type=NeighborhoodType.EIGHT_CONNECTED,
                flood_mode=FloodMode.GRADIENT,
                max_iterations=500,
                use_elliptical_footprint=True,
                edge_softening=True
            )
    
    def get_rendering_statistics(self) -> Dict[str, Any]:
        """
        获取渲染统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.render_stats.copy()
    
    def visualize_flood_process(self, frames: List[np.ndarray], output_path: str) -> bool:
        """
        可视化填充过程
        
        Args:
            frames: 帧序列
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if not frames:
                return False
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            # 写入所有帧
            for frame in frames:
                # 转换颜色空间
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)
            
            writer.release()
            self.logger.info(f"Flood process visualization saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error visualizing flood process: {str(e)}")
            return False