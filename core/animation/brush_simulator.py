# -*- coding: utf-8 -*-
"""
笔刷模拟器

实现中国画笔刷的物理模拟
包括笔毛形变、墨水扩散、纸张纹理等效果
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import random


@dataclass
class BrushHair:
    """
    笔毛数据结构
    
    Attributes:
        hair_id (int): 笔毛ID
        position (Tuple[float, float]): 位置
        length (float): 长度
        stiffness (float): 硬度
        ink_capacity (float): 墨水容量
        current_ink (float): 当前墨水量
        bend_angle (float): 弯曲角度
        tip_position (Tuple[float, float]): 笔尖位置
    """
    hair_id: int
    position: Tuple[float, float]
    length: float
    stiffness: float
    ink_capacity: float
    current_ink: float
    bend_angle: float
    tip_position: Tuple[float, float]


@dataclass
class BrushState:
    """
    笔刷状态数据结构
    
    Attributes:
        position (Tuple[float, float]): 笔刷位置
        angle (float): 笔刷角度
        pressure (float): 压力
        velocity (Tuple[float, float]): 速度
        ink_level (float): 墨水水平
        hairs (List[BrushHair]): 笔毛列表
        contact_area (float): 接触面积
        deformation (float): 形变程度
    """
    position: Tuple[float, float]
    angle: float
    pressure: float
    velocity: Tuple[float, float]
    ink_level: float
    hairs: List[BrushHair]
    contact_area: float
    deformation: float


@dataclass
class InkDrop:
    """
    墨滴数据结构
    
    Attributes:
        position (Tuple[float, float]): 位置
        size (float): 大小
        concentration (float): 浓度
        age (float): 年龄
        velocity (Tuple[float, float]): 速度
        absorbed (bool): 是否被吸收
    """
    position: Tuple[float, float]
    size: float
    concentration: float
    age: float
    velocity: Tuple[float, float]
    absorbed: bool


class BrushSimulator:
    """
    笔刷模拟器
    
    模拟中国画笔刷的物理行为
    """
    
    def __init__(self, config):
        """
        初始化笔刷模拟器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 笔刷物理参数
        self.brush_radius = config['brush'].get('radius', 10.0)
        self.hair_count = config['brush'].get('hair_count', 50)
        self.hair_length = config['brush'].get('hair_length', 15.0)
        self.hair_stiffness = config['brush'].get('hair_stiffness', 0.3)
        self.hair_variation = config['brush'].get('hair_variation', 0.2)
        
        # 墨水参数
        self.ink_viscosity = config['ink'].get('viscosity', 0.8)
        self.ink_absorption_rate = config['ink'].get('absorption_rate', 0.1)
        self.ink_diffusion_rate = config['ink'].get('diffusion_rate', 0.05)
        self.ink_capacity = config['ink'].get('capacity', 1.0)
        
        # 纸张参数
        self.paper_roughness = config['paper'].get('roughness', 0.1)
        self.paper_absorption = config['paper'].get('absorption', 0.3)
        self.paper_fiber_density = config['paper'].get('fiber_density', 0.5)
        
        # 物理参数
        self.gravity = config['physics'].get('gravity', 9.8)
        self.air_resistance = config['physics'].get('air_resistance', 0.1)
        self.surface_tension = config['physics'].get('surface_tension', 0.07)
        
        # 渲染参数
        self.enable_hair_rendering = config['rendering'].get('enable_hair_rendering', True)
        self.enable_ink_diffusion = config['rendering'].get('enable_ink_diffusion', True)
        self.enable_paper_texture = config['rendering'].get('enable_paper_texture', True)
        
        # 初始化笔刷
        self.brush_state = self._initialize_brush()
        self.ink_drops = []
        self.canvas = None
        self.paper_texture = None
    
    def _initialize_brush(self) -> BrushState:
        """
        初始化笔刷状态
        
        Returns:
            BrushState: 初始笔刷状态
        """
        try:
            # 生成笔毛
            hairs = []
            for i in range(self.hair_count):
                # 随机分布在笔刷圆形区域内
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, self.brush_radius) * math.sqrt(random.random())
                
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                
                # 笔毛属性随机变化
                length = self.hair_length * (1.0 + random.uniform(-self.hair_variation, self.hair_variation))
                stiffness = self.hair_stiffness * (1.0 + random.uniform(-self.hair_variation, self.hair_variation))
                ink_capacity = self.ink_capacity / self.hair_count
                
                hair = BrushHair(
                    hair_id=i,
                    position=(x, y),
                    length=length,
                    stiffness=stiffness,
                    ink_capacity=ink_capacity,
                    current_ink=ink_capacity,  # 初始满墨
                    bend_angle=0.0,
                    tip_position=(x, y)
                )
                hairs.append(hair)
            
            # 初始笔刷状态
            brush_state = BrushState(
                position=(0.0, 0.0),
                angle=0.0,
                pressure=0.0,
                velocity=(0.0, 0.0),
                ink_level=1.0,
                hairs=hairs,
                contact_area=0.0,
                deformation=0.0
            )
            
            return brush_state
            
        except Exception as e:
            self.logger.error(f"Error initializing brush: {str(e)}")
            return BrushState(
                position=(0.0, 0.0),
                angle=0.0,
                pressure=0.0,
                velocity=(0.0, 0.0),
                ink_level=1.0,
                hairs=[],
                contact_area=0.0,
                deformation=0.0
            )
    
    def update_brush_state(self, position: Tuple[float, float], 
                          pressure: float, velocity: Tuple[float, float],
                          angle: float = 0.0) -> BrushState:
        """
        更新笔刷状态
        
        Args:
            position (Tuple[float, float]): 新位置
            pressure (float): 压力
            velocity (Tuple[float, float]): 速度
            angle (float): 角度
            
        Returns:
            BrushState: 更新后的笔刷状态
        """
        try:
            # 更新基本状态
            self.brush_state.position = position
            self.brush_state.pressure = pressure
            self.brush_state.velocity = velocity
            self.brush_state.angle = angle
            
            # 计算形变
            self.brush_state.deformation = self._calculate_deformation(pressure)
            
            # 计算接触面积
            self.brush_state.contact_area = self._calculate_contact_area(pressure, self.brush_state.deformation)
            
            # 更新笔毛状态
            self._update_hair_states(pressure, velocity, angle)
            
            # 更新墨水水平
            self._update_ink_level()
            
            return self.brush_state
            
        except Exception as e:
            self.logger.error(f"Error updating brush state: {str(e)}")
            return self.brush_state
    
    def _calculate_deformation(self, pressure: float) -> float:
        """
        计算笔刷形变
        
        Args:
            pressure (float): 压力
            
        Returns:
            float: 形变程度
        """
        try:
            # 基于压力和笔毛硬度计算形变
            avg_stiffness = np.mean([hair.stiffness for hair in self.brush_state.hairs])
            
            # 形变与压力成正比，与硬度成反比
            deformation = pressure / (avg_stiffness + 0.1)
            
            # 限制形变范围
            deformation = max(0.0, min(1.0, deformation))
            
            return deformation
            
        except Exception as e:
            self.logger.error(f"Error calculating deformation: {str(e)}")
            return 0.0
    
    def _calculate_contact_area(self, pressure: float, deformation: float) -> float:
        """
        计算接触面积
        
        Args:
            pressure (float): 压力
            deformation (float): 形变程度
            
        Returns:
            float: 接触面积
        """
        try:
            # 基础接触面积
            base_area = math.pi * self.brush_radius ** 2
            
            # 形变增加接触面积
            deformation_factor = 1.0 + deformation * 2.0
            
            # 压力影响接触面积
            pressure_factor = 0.5 + pressure * 0.5
            
            contact_area = base_area * deformation_factor * pressure_factor
            
            return contact_area
            
        except Exception as e:
            self.logger.error(f"Error calculating contact area: {str(e)}")
            return math.pi * self.brush_radius ** 2
    
    def _update_hair_states(self, pressure: float, velocity: Tuple[float, float], angle: float):
        """
        更新笔毛状态
        
        Args:
            pressure (float): 压力
            velocity (Tuple[float, float]): 速度
            angle (float): 角度
        """
        try:
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            
            for hair in self.brush_state.hairs:
                # 计算笔毛弯曲
                bend_factor = pressure / (hair.stiffness + 0.1)
                
                # 速度影响弯曲方向
                if speed > 0.1:
                    velocity_angle = math.atan2(velocity[1], velocity[0])
                    bend_direction = velocity_angle + math.pi  # 向后弯曲
                else:
                    bend_direction = angle
                
                hair.bend_angle = bend_direction
                
                # 计算笔尖位置
                bend_amount = bend_factor * hair.length * 0.5
                tip_x = hair.position[0] + bend_amount * math.cos(bend_direction)
                tip_y = hair.position[1] + bend_amount * math.sin(bend_direction)
                
                hair.tip_position = (tip_x, tip_y)
                
                # 更新墨水量（基于接触和流动）
                if pressure > 0.1:
                    ink_flow = self._calculate_hair_ink_flow(hair, pressure, speed)
                    hair.current_ink = max(0.0, hair.current_ink - ink_flow)
            
        except Exception as e:
            self.logger.error(f"Error updating hair states: {str(e)}")
    
    def _calculate_hair_ink_flow(self, hair: BrushHair, pressure: float, speed: float) -> float:
        """
        计算笔毛墨水流量
        
        Args:
            hair (BrushHair): 笔毛
            pressure (float): 压力
            speed (float): 速度
            
        Returns:
            float: 墨水流量
        """
        try:
            # 基础流量与压力和墨水量成正比
            base_flow = pressure * hair.current_ink * 0.01
            
            # 速度影响流量
            speed_factor = 1.0 + speed * 0.1
            
            # 粘度影响流量
            viscosity_factor = 1.0 / (self.ink_viscosity + 0.1)
            
            flow = base_flow * speed_factor * viscosity_factor
            
            return min(flow, hair.current_ink)
            
        except Exception as e:
            self.logger.error(f"Error calculating hair ink flow: {str(e)}")
            return 0.0
    
    def _update_ink_level(self):
        """
        更新整体墨水水平
        """
        try:
            total_ink = sum(hair.current_ink for hair in self.brush_state.hairs)
            total_capacity = sum(hair.ink_capacity for hair in self.brush_state.hairs)
            
            if total_capacity > 0:
                self.brush_state.ink_level = total_ink / total_capacity
            else:
                self.brush_state.ink_level = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating ink level: {str(e)}")
            self.brush_state.ink_level = 0.0
    
    def render_brush_stroke(self, canvas: np.ndarray, 
                           start_pos: Tuple[float, float],
                           end_pos: Tuple[float, float],
                           pressure: float,
                           brush_size: float,
                           ink_flow: float) -> np.ndarray:
        """
        渲染笔刷笔画
        
        Args:
            canvas (np.ndarray): 画布
            start_pos (Tuple[float, float]): 起始位置
            end_pos (Tuple[float, float]): 结束位置
            pressure (float): 压力
            brush_size (float): 笔刷大小
            ink_flow (float): 墨水流量
            
        Returns:
            np.ndarray: 渲染后的画布
        """
        try:
            # 计算笔画参数
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 0.1:
                return canvas
            
            # 计算步数
            steps = max(1, int(distance))
            
            # 逐步渲染
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                
                # 插值位置
                x = start_pos[0] + t * dx
                y = start_pos[1] + t * dy
                
                # 计算当前参数
                current_pressure = pressure
                current_size = brush_size * (0.5 + 0.5 * current_pressure)
                current_ink = ink_flow * (0.8 + 0.2 * current_pressure)
                
                # 渲染单点
                canvas = self._render_brush_point(
                    canvas, (x, y), current_size, current_ink, current_pressure
                )
            
            # 添加墨滴效果
            if self.enable_ink_diffusion:
                canvas = self._add_ink_diffusion(canvas, start_pos, end_pos, ink_flow)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error rendering brush stroke: {str(e)}")
            return canvas
    
    def _render_brush_point(self, canvas: np.ndarray, 
                           position: Tuple[float, float],
                           size: float, ink_flow: float, pressure: float) -> np.ndarray:
        """
        渲染单个笔刷点
        
        Args:
            canvas (np.ndarray): 画布
            position (Tuple[float, float]): 位置
            size (float): 大小
            ink_flow (float): 墨水流量
            pressure (float): 压力
            
        Returns:
            np.ndarray: 渲染后的画布
        """
        try:
            x, y = int(position[0]), int(position[1])
            h, w = canvas.shape[:2]
            
            # 检查边界
            if x < 0 or x >= w or y < 0 or y >= h:
                return canvas
            
            # 计算笔刷半径
            radius = int(size / 2)
            
            # 创建笔刷掩码
            mask = self._create_brush_mask(radius, pressure)
            
            # 计算墨水颜色强度
            ink_intensity = ink_flow * 255
            
            # 应用笔刷效果
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px, py = x + dx, y + dy
                    
                    if 0 <= px < w and 0 <= py < h:
                        mask_x, mask_y = dx + radius, dy + radius
                        
                        if (mask_x < mask.shape[1] and mask_y < mask.shape[0] and 
                            mask[mask_y, mask_x] > 0):
                            
                            # 计算混合强度
                            blend_factor = mask[mask_y, mask_x] * ink_intensity / 255.0
                            
                            if len(canvas.shape) == 3:  # 彩色图像
                                for c in range(canvas.shape[2]):
                                    current_value = canvas[py, px, c]
                                    new_value = current_value * (1 - blend_factor)
                                    canvas[py, px, c] = int(new_value)
                            else:  # 灰度图像
                                current_value = canvas[py, px]
                                new_value = current_value * (1 - blend_factor)
                                canvas[py, px] = int(new_value)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error rendering brush point: {str(e)}")
            return canvas
    
    def _create_brush_mask(self, radius: int, pressure: float) -> np.ndarray:
        """
        创建笔刷掩码
        
        Args:
            radius (int): 半径
            pressure (float): 压力
            
        Returns:
            np.ndarray: 笔刷掩码
        """
        try:
            size = 2 * radius + 1
            mask = np.zeros((size, size), dtype=np.float32)
            
            center = radius
            
            for y in range(size):
                for x in range(size):
                    # 计算到中心的距离
                    distance = math.sqrt((x - center)**2 + (y - center)**2)
                    
                    if distance <= radius:
                        # 基于距离的衰减
                        falloff = 1.0 - (distance / radius)
                        
                        # 压力影响强度分布
                        pressure_factor = 0.3 + 0.7 * pressure
                        
                        # 添加纹理效果
                        if self.enable_paper_texture:
                            texture_noise = random.uniform(0.8, 1.2)
                            falloff *= texture_noise
                        
                        mask[y, x] = falloff * pressure_factor
            
            # 高斯模糊软化边缘
            mask = gaussian_filter(mask, sigma=0.5)
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Error creating brush mask: {str(e)}")
            return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    
    def _add_ink_diffusion(self, canvas: np.ndarray,
                          start_pos: Tuple[float, float],
                          end_pos: Tuple[float, float],
                          ink_flow: float) -> np.ndarray:
        """
        添加墨水扩散效果
        
        Args:
            canvas (np.ndarray): 画布
            start_pos (Tuple[float, float]): 起始位置
            end_pos (Tuple[float, float]): 结束位置
            ink_flow (float): 墨水流量
            
        Returns:
            np.ndarray: 添加扩散效果后的画布
        """
        try:
            # 生成墨滴
            num_drops = int(ink_flow * 10)
            
            for _ in range(num_drops):
                # 随机位置在笔画路径附近
                t = random.random()
                base_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                base_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                
                # 添加随机偏移
                offset_x = random.uniform(-5, 5)
                offset_y = random.uniform(-5, 5)
                
                drop_pos = (base_x + offset_x, base_y + offset_y)
                drop_size = random.uniform(0.5, 2.0)
                drop_concentration = random.uniform(0.3, 0.8)
                
                ink_drop = InkDrop(
                    position=drop_pos,
                    size=drop_size,
                    concentration=drop_concentration,
                    age=0.0,
                    velocity=(0.0, 0.0),
                    absorbed=False
                )
                
                self.ink_drops.append(ink_drop)
            
            # 更新和渲染墨滴
            canvas = self._update_and_render_ink_drops(canvas)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error adding ink diffusion: {str(e)}")
            return canvas
    
    def _update_and_render_ink_drops(self, canvas: np.ndarray) -> np.ndarray:
        """
        更新和渲染墨滴
        
        Args:
            canvas (np.ndarray): 画布
            
        Returns:
            np.ndarray: 渲染后的画布
        """
        try:
            drops_to_remove = []
            
            for i, drop in enumerate(self.ink_drops):
                # 更新墨滴年龄
                drop.age += 0.1
                
                # 墨滴扩散
                if not drop.absorbed:
                    # 扩散速度与浓度和纸张吸收性相关
                    diffusion_rate = self.ink_diffusion_rate * drop.concentration * self.paper_absorption
                    drop.size += diffusion_rate
                    
                    # 浓度随扩散降低
                    drop.concentration *= 0.98
                    
                    # 渲染墨滴
                    canvas = self._render_ink_drop(canvas, drop)
                    
                    # 检查是否被吸收或消散
                    if drop.concentration < 0.1 or drop.age > 10.0:
                        drop.absorbed = True
                        drops_to_remove.append(i)
            
            # 移除已吸收的墨滴
            for i in reversed(drops_to_remove):
                del self.ink_drops[i]
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error updating ink drops: {str(e)}")
            return canvas
    
    def _render_ink_drop(self, canvas: np.ndarray, drop: InkDrop) -> np.ndarray:
        """
        渲染单个墨滴
        
        Args:
            canvas (np.ndarray): 画布
            drop (InkDrop): 墨滴
            
        Returns:
            np.ndarray: 渲染后的画布
        """
        try:
            x, y = int(drop.position[0]), int(drop.position[1])
            h, w = canvas.shape[:2]
            
            if x < 0 or x >= w or y < 0 or y >= h:
                return canvas
            
            radius = int(drop.size)
            intensity = drop.concentration * 50  # 调整强度
            
            # 渲染圆形墨滴
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px, py = x + dx, y + dy
                    
                    if 0 <= px < w and 0 <= py < h:
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance <= radius:
                            # 计算衰减
                            falloff = 1.0 - (distance / radius) if radius > 0 else 1.0
                            effect = intensity * falloff
                            
                            if len(canvas.shape) == 3:  # 彩色图像
                                for c in range(canvas.shape[2]):
                                    current_value = canvas[py, px, c]
                                    new_value = max(0, current_value - effect)
                                    canvas[py, px, c] = int(new_value)
                            else:  # 灰度图像
                                current_value = canvas[py, px]
                                new_value = max(0, current_value - effect)
                                canvas[py, px] = int(new_value)
            
            return canvas
            
        except Exception as e:
            self.logger.error(f"Error rendering ink drop: {str(e)}")
            return canvas
    
    def create_paper_texture(self, width: int, height: int) -> np.ndarray:
        """
        创建纸张纹理
        
        Args:
            width (int): 宽度
            height (int): 高度
            
        Returns:
            np.ndarray: 纸张纹理
        """
        try:
            # 基础白色背景
            texture = np.full((height, width, 3), 255, dtype=np.uint8)
            
            if not self.enable_paper_texture:
                return texture
            
            # 添加纤维纹理
            fiber_noise = np.random.normal(0, self.paper_roughness * 10, (height, width))
            fiber_noise = gaussian_filter(fiber_noise, sigma=1.0)
            
            # 添加纸张颗粒
            grain_noise = np.random.normal(0, self.paper_roughness * 5, (height, width))
            
            # 合并纹理
            combined_noise = fiber_noise + grain_noise
            
            # 应用到纹理
            for c in range(3):
                channel = texture[:, :, c].astype(np.float32)
                channel += combined_noise
                channel = np.clip(channel, 0, 255)
                texture[:, :, c] = channel.astype(np.uint8)
            
            return texture
            
        except Exception as e:
            self.logger.error(f"Error creating paper texture: {str(e)}")
            return np.full((height, width, 3), 255, dtype=np.uint8)
    
    def reset_brush(self):
        """
        重置笔刷状态
        """
        try:
            self.brush_state = self._initialize_brush()
            self.ink_drops.clear()
            self.logger.info("Brush reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting brush: {str(e)}")
    
    def refill_ink(self, amount: float = 1.0):
        """
        补充墨水
        
        Args:
            amount (float): 补充量（0-1）
        """
        try:
            for hair in self.brush_state.hairs:
                hair.current_ink = min(hair.ink_capacity, hair.current_ink + hair.ink_capacity * amount)
            
            self._update_ink_level()
            self.logger.info(f"Ink refilled: {amount * 100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error refilling ink: {str(e)}")
    
    def get_brush_statistics(self) -> Dict[str, Any]:
        """
        获取笔刷统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'brush_state': {
                    'position': self.brush_state.position,
                    'pressure': self.brush_state.pressure,
                    'ink_level': self.brush_state.ink_level,
                    'contact_area': self.brush_state.contact_area,
                    'deformation': self.brush_state.deformation
                },
                'hair_statistics': {
                    'count': len(self.brush_state.hairs),
                    'avg_ink_level': np.mean([hair.current_ink / hair.ink_capacity for hair in self.brush_state.hairs]),
                    'avg_bend_angle': np.mean([hair.bend_angle for hair in self.brush_state.hairs]),
                    'avg_stiffness': np.mean([hair.stiffness for hair in self.brush_state.hairs])
                },
                'ink_drops': {
                    'count': len(self.ink_drops),
                    'avg_size': np.mean([drop.size for drop in self.ink_drops]) if self.ink_drops else 0,
                    'avg_concentration': np.mean([drop.concentration for drop in self.ink_drops]) if self.ink_drops else 0
                },
                'parameters': {
                    'brush_radius': self.brush_radius,
                    'hair_count': self.hair_count,
                    'ink_viscosity': self.ink_viscosity,
                    'paper_roughness': self.paper_roughness
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting brush statistics: {str(e)}")
            return {'error': str(e)}