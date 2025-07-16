# -*- coding: utf-8 -*-
"""
笔刷模拟器模块

提供真实的笔刷绘画效果模拟
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import cv2
from ..stroke_extraction.stroke_detector import Stroke


class BrushType(Enum):
    """笔刷类型"""
    ROUND = "round"  # 圆形笔刷
    FLAT = "flat"  # 扁平笔刷
    DETAIL = "detail"  # 细节笔刷
    TEXTURE = "texture"  # 纹理笔刷
    CALLIGRAPHY = "calligraphy"  # 书法笔刷
    WATERCOLOR = "watercolor"  # 水彩笔刷


class BlendMode(Enum):
    """混合模式"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    COLOR_BURN = "color_burn"
    LINEAR_BURN = "linear_burn"


@dataclass
class BrushProperties:
    """笔刷属性"""
    brush_type: BrushType = BrushType.ROUND
    size: float = 10.0
    opacity: float = 1.0
    hardness: float = 0.8
    spacing: float = 0.1
    angle: float = 0.0
    roundness: float = 1.0
    flow: float = 1.0
    texture_strength: float = 0.0
    pressure_sensitivity: float = 0.5
    velocity_sensitivity: float = 0.3


@dataclass
class BrushStroke:
    """笔刷笔触"""
    points: List[Tuple[float, float]]
    pressures: List[float]
    velocities: List[float]
    timestamps: List[float]
    properties: BrushProperties
    color: Tuple[int, int, int]
    blend_mode: BlendMode = BlendMode.NORMAL


@dataclass
class BrushState:
    """笔刷状态"""
    position: Tuple[float, float]
    pressure: float
    velocity: float
    angle: float
    size: float
    opacity: float
    color: Tuple[int, int, int]
    is_painting: bool


class BrushSimulator:
    """笔刷模拟器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化笔刷模拟器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 默认笔刷属性
        self.default_properties = BrushProperties(
            brush_type=BrushType(config.get('brush_type', 'round')),
            size=config.get('brush_size', 10.0),
            opacity=config.get('brush_opacity', 1.0),
            hardness=config.get('brush_hardness', 0.8),
            spacing=config.get('brush_spacing', 0.1),
            pressure_sensitivity=config.get('pressure_sensitivity', 0.5),
            velocity_sensitivity=config.get('velocity_sensitivity', 0.3)
        )
        
        # 笔刷纹理缓存
        self.texture_cache = {}
        
        # 当前笔刷状态
        self.current_state = BrushState(
            position=(0, 0),
            pressure=0.0,
            velocity=0.0,
            angle=0.0,
            size=self.default_properties.size,
            opacity=self.default_properties.opacity,
            color=(0, 0, 0),
            is_painting=False
        )
        
        # 性能优化设置
        self.enable_antialiasing = config.get('enable_antialiasing', True)
        self.enable_pressure_dynamics = config.get('enable_pressure_dynamics', True)
        self.enable_velocity_dynamics = config.get('enable_velocity_dynamics', True)
        self.enable_texture_effects = config.get('enable_texture_effects', True)
        
    def simulate_stroke(self, stroke: Stroke, canvas: np.ndarray, 
                       properties: BrushProperties = None) -> np.ndarray:
        """
        模拟笔触绘制
        
        Args:
            stroke: 笔触对象
            canvas: 画布
            properties: 笔刷属性
            
        Returns:
            绘制后的画布
        """
        if not (hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) >= 2):
            return canvas
        
        properties = properties or self.default_properties
        
        try:
            # 转换笔触为笔刷笔触
            brush_stroke = self._convert_stroke_to_brush_stroke(stroke, properties)
            
            # 绘制笔触
            result_canvas = self._render_brush_stroke(canvas, brush_stroke)
            
            return result_canvas
            
        except Exception as e:
            self.logger.error(f"Error simulating stroke: {e}")
            return canvas
    
    def simulate_stroke_sequence(self, strokes: List[Stroke], canvas: np.ndarray,
                               properties_list: List[BrushProperties] = None) -> np.ndarray:
        """
        模拟笔触序列绘制
        
        Args:
            strokes: 笔触列表
            canvas: 画布
            properties_list: 笔刷属性列表
            
        Returns:
            绘制后的画布
        """
        result_canvas = canvas.copy()
        
        for i, stroke in enumerate(strokes):
            # 获取对应的笔刷属性
            if properties_list and i < len(properties_list):
                properties = properties_list[i]
            else:
                properties = self.default_properties
            
            # 模拟笔触
            result_canvas = self.simulate_stroke(stroke, result_canvas, properties)
        
        return result_canvas
    
    def create_brush_texture(self, brush_type: BrushType, size: int) -> np.ndarray:
        """
        创建笔刷纹理
        
        Args:
            brush_type: 笔刷类型
            size: 纹理大小
            
        Returns:
            笔刷纹理
        """
        cache_key = f"{brush_type.value}_{size}"
        
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]
        
        # 创建基础纹理
        texture = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        
        if brush_type == BrushType.ROUND:
            # 圆形笔刷
            y, x = np.ogrid[:size, :size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 2) ** 2
            texture[mask] = 1.0
            
            # 添加软边缘
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                    if dist <= size // 2:
                        texture[i, j] = max(0, 1.0 - (dist / (size // 2)) ** 2)
        
        elif brush_type == BrushType.FLAT:
            # 扁平笔刷
            texture[center - size//4:center + size//4, :] = 1.0
            
            # 添加边缘渐变
            for i in range(size):
                for j in range(size):
                    if center - size//4 <= i <= center + size//4:
                        edge_dist = min(abs(i - (center - size//4)), abs(i - (center + size//4)))
                        texture[i, j] = min(1.0, edge_dist / (size//8) + 0.5)
        
        elif brush_type == BrushType.DETAIL:
            # 细节笔刷（小而硬）
            y, x = np.ogrid[:size, :size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 4) ** 2
            texture[mask] = 1.0
        
        elif brush_type == BrushType.TEXTURE:
            # 纹理笔刷
            np.random.seed(42)  # 确保可重复性
            noise = np.random.random((size, size))
            
            # 创建圆形遮罩
            y, x = np.ogrid[:size, :size]
            mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 2) ** 2
            
            texture = noise * mask.astype(float)
            texture = cv2.GaussianBlur(texture, (3, 3), 1.0)
        
        elif brush_type == BrushType.CALLIGRAPHY:
            # 书法笔刷（椭圆形）
            y, x = np.ogrid[:size, :size]
            ellipse_mask = ((x - center) / (size // 2)) ** 2 + ((y - center) / (size // 4)) ** 2 <= 1
            texture[ellipse_mask] = 1.0
            
            # 添加角度变化
            texture = self._rotate_texture(texture, 45)
        
        elif brush_type == BrushType.WATERCOLOR:
            # 水彩笔刷（不规则边缘）
            y, x = np.ogrid[:size, :size]
            base_mask = (x - center) ** 2 + (y - center) ** 2 <= (size // 2) ** 2
            
            # 添加随机变化
            np.random.seed(42)
            noise = np.random.random((size, size)) * 0.3
            
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                    if dist <= size // 2:
                        # 不规则边缘
                        edge_variation = noise[i, j]
                        texture[i, j] = max(0, 1.0 - (dist / (size // 2)) ** 1.5 + edge_variation)
        
        # 归一化
        if texture.max() > 0:
            texture = texture / texture.max()
        
        # 缓存纹理
        self.texture_cache[cache_key] = texture
        
        return texture
    
    def _convert_stroke_to_brush_stroke(self, stroke: Stroke, 
                                       properties: BrushProperties) -> BrushStroke:
        """
        将笔触转换为笔刷笔触
        
        Args:
            stroke: 原始笔触
            properties: 笔刷属性
            
        Returns:
            笔刷笔触
        """
        points = stroke.points
        
        # 计算压力值
        pressures = self._calculate_pressures(points, properties)
        
        # 计算速度值
        velocities = self._calculate_velocities(points)
        
        # 计算时间戳
        timestamps = self._calculate_timestamps(points, velocities)
        
        # 确定颜色
        if hasattr(stroke, 'color') and stroke.color is not None:
            color = stroke.color
        else:
            color = (0, 0, 0)  # 默认黑色
        
        return BrushStroke(
            points=points,
            pressures=pressures,
            velocities=velocities,
            timestamps=timestamps,
            properties=properties,
            color=color
        )
    
    def _calculate_pressures(self, points: List[Tuple[float, float]], 
                           properties: BrushProperties) -> List[float]:
        """
        计算压力值
        
        Args:
            points: 点列表
            properties: 笔刷属性
            
        Returns:
            压力值列表
        """
        if not self.enable_pressure_dynamics:
            return [1.0] * len(points)
        
        pressures = []
        
        for i, point in enumerate(points):
            # 基于位置计算压力（简化模型）
            progress = i / (len(points) - 1) if len(points) > 1 else 0
            
            # 笔触开始和结束时压力较小
            if progress < 0.1:
                pressure = progress * 10  # 0 到 1
            elif progress > 0.9:
                pressure = (1.0 - progress) * 10  # 1 到 0
            else:
                pressure = 1.0
            
            # 添加随机变化
            pressure += np.random.normal(0, 0.1) * properties.pressure_sensitivity
            pressure = np.clip(pressure, 0.1, 1.0)
            
            pressures.append(pressure)
        
        return pressures
    
    def _calculate_velocities(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        计算速度值
        
        Args:
            points: 点列表
            
        Returns:
            速度值列表
        """
        if len(points) < 2:
            return [0.0] * len(points)
        
        velocities = [0.0]  # 第一个点速度为0
        
        for i in range(1, len(points)):
            # 计算距离
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # 假设时间间隔恒定
            velocity = distance
            velocities.append(velocity)
        
        # 归一化速度
        if velocities:
            max_velocity = max(velocities)
            if max_velocity > 0:
                velocities = [v / max_velocity for v in velocities]
        
        return velocities
    
    def _calculate_timestamps(self, points: List[Tuple[float, float]], 
                            velocities: List[float]) -> List[float]:
        """
        计算时间戳
        
        Args:
            points: 点列表
            velocities: 速度列表
            
        Returns:
            时间戳列表
        """
        timestamps = [0.0]
        
        for i in range(1, len(points)):
            # 基于速度计算时间间隔
            velocity = velocities[i] if i < len(velocities) else 1.0
            time_interval = 1.0 / (velocity + 0.1)  # 避免除零
            
            timestamps.append(timestamps[-1] + time_interval)
        
        return timestamps
    
    def _render_brush_stroke(self, canvas: np.ndarray, brush_stroke: BrushStroke) -> np.ndarray:
        """
        渲染笔刷笔触
        
        Args:
            canvas: 画布
            brush_stroke: 笔刷笔触
            
        Returns:
            渲染后的画布
        """
        result = canvas.copy()
        properties = brush_stroke.properties
        
        # 创建笔刷纹理
        base_size = int(properties.size)
        brush_texture = self.create_brush_texture(properties.brush_type, base_size * 2)
        
        # 沿路径绘制
        for i in range(len(brush_stroke.points)):
            point = brush_stroke.points[i]
            pressure = brush_stroke.pressures[i] if i < len(brush_stroke.pressures) else 1.0
            velocity = brush_stroke.velocities[i] if i < len(brush_stroke.velocities) else 1.0
            
            # 计算动态属性
            dynamic_size = self._calculate_dynamic_size(properties, pressure, velocity)
            dynamic_opacity = self._calculate_dynamic_opacity(properties, pressure, velocity)
            
            # 应用笔刷印记
            result = self._apply_brush_dab(result, point, brush_texture, 
                                         dynamic_size, dynamic_opacity, 
                                         brush_stroke.color, properties)
        
        return result
    
    def _calculate_dynamic_size(self, properties: BrushProperties, 
                               pressure: float, velocity: float) -> float:
        """
        计算动态大小
        
        Args:
            properties: 笔刷属性
            pressure: 压力
            velocity: 速度
            
        Returns:
            动态大小
        """
        base_size = properties.size
        
        # 压力影响
        pressure_factor = 1.0
        if self.enable_pressure_dynamics:
            pressure_factor = 0.5 + 0.5 * pressure * properties.pressure_sensitivity
        
        # 速度影响
        velocity_factor = 1.0
        if self.enable_velocity_dynamics:
            velocity_factor = 1.0 - velocity * properties.velocity_sensitivity * 0.3
        
        dynamic_size = base_size * pressure_factor * velocity_factor
        return max(1.0, dynamic_size)
    
    def _calculate_dynamic_opacity(self, properties: BrushProperties, 
                                  pressure: float, velocity: float) -> float:
        """
        计算动态透明度
        
        Args:
            properties: 笔刷属性
            pressure: 压力
            velocity: 速度
            
        Returns:
            动态透明度
        """
        base_opacity = properties.opacity
        
        # 压力影响
        pressure_factor = 1.0
        if self.enable_pressure_dynamics:
            pressure_factor = pressure * properties.pressure_sensitivity + (1 - properties.pressure_sensitivity)
        
        # 速度影响（高速时透明度降低）
        velocity_factor = 1.0
        if self.enable_velocity_dynamics:
            velocity_factor = 1.0 - velocity * properties.velocity_sensitivity * 0.2
        
        dynamic_opacity = base_opacity * pressure_factor * velocity_factor
        return np.clip(dynamic_opacity, 0.0, 1.0)
    
    def _apply_brush_dab(self, canvas: np.ndarray, position: Tuple[float, float],
                        brush_texture: np.ndarray, size: float, opacity: float,
                        color: Tuple[int, int, int], properties: BrushProperties) -> np.ndarray:
        """
        应用笔刷印记
        
        Args:
            canvas: 画布
            position: 位置
            brush_texture: 笔刷纹理
            size: 大小
            opacity: 透明度
            color: 颜色
            properties: 笔刷属性
            
        Returns:
            更新后的画布
        """
        x, y = int(position[0]), int(position[1])
        
        # 调整纹理大小
        texture_size = int(size)
        if texture_size != brush_texture.shape[0]:
            brush_texture = cv2.resize(brush_texture, (texture_size, texture_size))
        
        # 计算绘制区域
        half_size = texture_size // 2
        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(canvas.shape[1], x + half_size), min(canvas.shape[0], y + half_size)
        
        # 计算纹理区域
        tx1, ty1 = max(0, half_size - x), max(0, half_size - y)
        tx2, ty2 = tx1 + (x2 - x1), ty1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1 or tx2 <= tx1 or ty2 <= ty1:
            return canvas
        
        # 获取纹理片段
        texture_patch = brush_texture[ty1:ty2, tx1:tx2]
        
        if texture_patch.size == 0:
            return canvas
        
        # 应用硬度
        texture_patch = texture_patch ** (2.0 - properties.hardness)
        
        # 应用透明度
        alpha = texture_patch * opacity
        
        # 混合颜色
        canvas_patch = canvas[y1:y2, x1:x2]
        
        for c in range(3):
            # 使用numpy.array_equal避免数组布尔值歧义
            if np.array_equal(canvas_patch.shape[:2], alpha.shape):
                canvas[y1:y2, x1:x2, c] = (canvas_patch[:, :, c] * (1 - alpha) + 
                                          color[c] * alpha).astype(np.uint8)
        
        return canvas
    
    def _rotate_texture(self, texture: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转纹理
        
        Args:
            texture: 纹理
            angle: 角度（度）
            
        Returns:
            旋转后的纹理
        """
        center = (texture.shape[1] // 2, texture.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(texture, rotation_matrix, texture.shape[::-1])
        return rotated
    
    def set_brush_properties(self, properties: BrushProperties):
        """
        设置笔刷属性
        
        Args:
            properties: 笔刷属性
        """
        self.default_properties = properties
        self.logger.info(f"Brush properties updated: {properties.brush_type.value}")
    
    def get_brush_preview(self, size: int = 64) -> np.ndarray:
        """
        获取笔刷预览
        
        Args:
            size: 预览大小
            
        Returns:
            笔刷预览图像
        """
        texture = self.create_brush_texture(self.default_properties.brush_type, size)
        
        # 转换为RGB图像
        preview = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            preview[:, :, c] = (texture * 255).astype(np.uint8)
        
        return preview
    
    def clear_texture_cache(self):
        """清除纹理缓存"""
        self.texture_cache.clear()
        self.logger.info("Brush texture cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计信息
        """
        return {
            'texture_cache_size': len(self.texture_cache),
            'current_brush_type': self.default_properties.brush_type.value,
            'current_brush_size': self.default_properties.size,
            'dynamics_enabled': {
                'pressure': self.enable_pressure_dynamics,
                'velocity': self.enable_velocity_dynamics,
                'texture': self.enable_texture_effects
            },
            'antialiasing_enabled': self.enable_antialiasing
        }