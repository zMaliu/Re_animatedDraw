#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画布管理模块
提供绘画画布的创建、管理和渲染功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from .stroke_model import Stroke, StrokeCollection

class Canvas:
    """绘画画布类"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 background_color: Tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.background_color = background_color
        
        # 创建画布
        self.canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
        self.stroke_layer = np.zeros((height, width), dtype=np.uint8)
        
        # 笔触管理
        self.stroke_collection = StrokeCollection()
        self.render_history = []  # 渲染历史
        
        # 画布属性
        self.paper_texture = None
        self.ink_absorption = 0.8  # 墨水吸收率
        
    def clear(self):
        """清空画布"""
        self.canvas = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)
        self.stroke_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        self.stroke_collection = StrokeCollection()
        self.render_history = []
    
    def add_stroke(self, stroke: Stroke, render_immediately: bool = True):
        """添加笔触到画布"""
        self.stroke_collection.add_stroke(stroke)
        
        if render_immediately:
            self.render_stroke(stroke)
    
    def render_stroke(self, stroke: Stroke, ink_color: Tuple[int, int, int] = (0, 0, 0)):
        """渲染单个笔触"""
        if stroke.mask is None:
            return
        
        # 确保掩码尺寸匹配画布
        if stroke.mask.shape != (self.height, self.width):
            # 如果掩码尺寸不匹配，创建一个匹配的掩码
            full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            x, y, w, h = stroke.bbox
            
            # 确保边界框在画布范围内
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            w = min(w, self.width - x)
            h = min(h, self.height - y)
            
            if w > 0 and h > 0:
                # 调整掩码大小到边界框
                resized_mask = cv2.resize(stroke.mask, (w, h))
                full_mask[y:y+h, x:x+w] = resized_mask
            
            mask = full_mask
        else:
            mask = stroke.mask
        
        # 模拟墨水渗透效果
        ink_intensity = mask.astype(np.float32) / 255.0
        ink_intensity *= stroke.properties.wetness * self.ink_absorption
        
        # 应用笔触到画布
        for c in range(3):
            # 混合墨水颜色
            self.canvas[:, :, c] = self.canvas[:, :, c] * (1 - ink_intensity) + \
                                  ink_color[c] * ink_intensity
        
        # 更新笔触层
        self.stroke_layer = np.maximum(self.stroke_layer, mask)
        
        # 记录渲染历史
        self.render_history.append({
            'stroke_id': stroke.id,
            'ink_color': ink_color,
            'timestamp': len(self.render_history)
        })
    
    def render_all_strokes(self, ink_color: Tuple[int, int, int] = (0, 0, 0)):
        """渲染所有笔触"""
        self.clear()
        for stroke in self.stroke_collection:
            self.render_stroke(stroke, ink_color)
    
    def render_strokes_sequence(self, strokes: List[Stroke], 
                              ink_colors: List[Tuple[int, int, int]] = None) -> List[np.ndarray]:
        """按序列渲染笔触，返回每一步的画布状态"""
        if ink_colors is None:
            ink_colors = [(0, 0, 0)] * len(strokes)
        
        frames = []
        self.clear()
        
        for i, stroke in enumerate(strokes):
            color = ink_colors[i % len(ink_colors)]
            self.render_stroke(stroke, color)
            frames.append(self.canvas.copy())
        
        return frames
    
    def add_paper_texture(self, texture_path: str = None, intensity: float = 0.3):
        """添加纸张纹理"""
        if texture_path and Path(texture_path).exists():
            texture = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
            texture = cv2.resize(texture, (self.width, self.height))
        else:
            # 生成简单的纸张纹理
            texture = self._generate_paper_texture()
        
        self.paper_texture = texture
        
        # 应用纹理到背景
        texture_effect = texture.astype(np.float32) / 255.0 * intensity
        for c in range(3):
            self.canvas[:, :, c] = self.canvas[:, :, c] * (1 - texture_effect) + \
                                  self.background_color[c] * texture_effect
    
    def _generate_paper_texture(self) -> np.ndarray:
        """生成简单的纸张纹理"""
        # 使用噪声生成纸张纹理
        noise = np.random.normal(0, 10, (self.height, self.width))
        texture = np.clip(128 + noise, 0, 255).astype(np.uint8)
        
        # 添加纤维纹理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        texture = cv2.morphologyEx(texture, cv2.MORPH_OPEN, kernel)
        
        return texture
    
    def apply_ink_bleeding(self, bleeding_radius: int = 2, bleeding_strength: float = 0.3):
        """应用墨水渗透效果"""
        if bleeding_radius <= 0:
            return
        
        # 创建膨胀核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (bleeding_radius*2+1, bleeding_radius*2+1))
        
        # 对笔触层进行膨胀
        bleeding_mask = cv2.dilate(self.stroke_layer, kernel, iterations=1)
        
        # 计算渗透区域
        bleeding_area = bleeding_mask - self.stroke_layer
        bleeding_intensity = bleeding_area.astype(np.float32) / 255.0 * bleeding_strength
        
        # 应用渗透效果
        for c in range(3):
            self.canvas[:, :, c] = self.canvas[:, :, c] * (1 - bleeding_intensity) + \
                                  (self.canvas[:, :, c] * 0.7) * bleeding_intensity
    
    def get_canvas_image(self) -> np.ndarray:
        """获取画布图像"""
        return self.canvas.copy()
    
    def save_canvas(self, filepath: str):
        """保存画布图像"""
        cv2.imwrite(filepath, self.canvas)
    
    def save_stroke_data(self, filepath: str):
        """保存笔触数据"""
        self.stroke_collection.save_to_json(filepath)
    
    def load_stroke_data(self, filepath: str):
        """加载笔触数据"""
        self.stroke_collection.load_from_json(filepath)
    
    def get_canvas_statistics(self) -> Dict:
        """获取画布统计信息"""
        stats = {
            'canvas_size': (self.width, self.height),
            'background_color': self.background_color,
            'total_strokes': len(self.stroke_collection),
            'render_history_length': len(self.render_history)
        }
        
        # 添加笔触统计
        stroke_stats = self.stroke_collection.get_statistics()
        stats.update(stroke_stats)
        
        # 计算画布覆盖率
        coverage = np.sum(self.stroke_layer > 0) / (self.width * self.height)
        stats['canvas_coverage'] = coverage
        
        return stats
    
    def create_animation_frames(self, strokes: List[Stroke], 
                              frame_duration: int = 5,
                              ink_color: Tuple[int, int, int] = (0, 0, 0)) -> List[np.ndarray]:
        """创建动画帧序列"""
        frames = []
        self.clear()
        
        for stroke in strokes:
            # 为每个笔触创建多帧动画
            stroke_frames = self._create_stroke_animation(stroke, frame_duration, ink_color)
            frames.extend(stroke_frames)
        
        return frames
    
    def _create_stroke_animation(self, stroke: Stroke, frame_duration: int,
                               ink_color: Tuple[int, int, int]) -> List[np.ndarray]:
        """为单个笔触创建动画帧"""
        frames = []
        
        if not stroke.path_points:
            # 如果没有路径点，直接渲染整个笔触
            self.render_stroke(stroke, ink_color)
            for _ in range(frame_duration):
                frames.append(self.canvas.copy())
            return frames
        
        # 按路径点逐步绘制
        path_length = len(stroke.path_points)
        points_per_frame = max(1, path_length // frame_duration)
        
        for frame_idx in range(frame_duration):
            # 计算当前帧应该绘制到哪个点
            end_point_idx = min((frame_idx + 1) * points_per_frame, path_length)
            
            # 创建部分笔触
            partial_stroke = self._create_partial_stroke(stroke, end_point_idx)
            if partial_stroke:
                self.render_stroke(partial_stroke, ink_color)
            
            frames.append(self.canvas.copy())
        
        return frames
    
    def _create_partial_stroke(self, stroke: Stroke, end_point_idx: int) -> Optional[Stroke]:
        """创建部分笔触（用于动画）"""
        if not stroke.path_points or end_point_idx <= 0:
            return None
        
        # 获取部分路径点
        partial_points = stroke.path_points[:end_point_idx]
        
        if len(partial_points) < 2:
            return None
        
        # 创建部分掩码
        partial_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # 绘制部分路径
        for i, point in enumerate(partial_points):
            x, y = point
            if 0 <= x < self.width and 0 <= y < self.height:
                # 根据笔触属性计算宽度
                width = max(1, int(5 * stroke.properties.thickness))
                cv2.circle(partial_mask, (x, y), width//2, 255, -1)
                
                # 连接到下一个点
                if i < len(partial_points) - 1:
                    next_point = partial_points[i + 1]
                    cv2.line(partial_mask, point, next_point, 255, width)
        
        # 创建部分笔触对象
        contours, _ = cv2.findContours(partial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        contour = max(contours, key=cv2.contourArea)
        
        # 计算边界框
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]
        bbox = (int(np.min(x_coords)), int(np.min(y_coords)),
               int(np.max(x_coords) - np.min(x_coords)),
               int(np.max(y_coords) - np.min(y_coords)))
        
        partial_stroke = Stroke(stroke.id, partial_mask, contour, bbox, stroke.stroke_type)
        partial_stroke.properties = stroke.properties
        
        return partial_stroke

class CanvasManager:
    """画布管理器"""
    
    def __init__(self):
        self.canvases: Dict[str, Canvas] = {}
        self.active_canvas: Optional[str] = None
    
    def create_canvas(self, name: str, width: int = 800, height: int = 600,
                     background_color: Tuple[int, int, int] = (255, 255, 255)) -> Canvas:
        """创建新画布"""
        canvas = Canvas(width, height, background_color)
        self.canvases[name] = canvas
        
        if self.active_canvas is None:
            self.active_canvas = name
        
        return canvas
    
    def get_canvas(self, name: str) -> Optional[Canvas]:
        """获取指定画布"""
        return self.canvases.get(name)
    
    def set_active_canvas(self, name: str) -> bool:
        """设置活动画布"""
        if name in self.canvases:
            self.active_canvas = name
            return True
        return False
    
    def get_active_canvas(self) -> Optional[Canvas]:
        """获取活动画布"""
        if self.active_canvas:
            return self.canvases.get(self.active_canvas)
        return None
    
    def delete_canvas(self, name: str) -> bool:
        """删除画布"""
        if name in self.canvases:
            del self.canvases[name]
            if self.active_canvas == name:
                self.active_canvas = next(iter(self.canvases.keys()), None)
            return True
        return False
    
    def list_canvases(self) -> List[str]:
        """列出所有画布名称"""
        return list(self.canvases.keys())
    
    def save_all_canvases(self, output_dir: str):
        """保存所有画布"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, canvas in self.canvases.items():
            # 保存画布图像
            canvas.save_canvas(str(output_path / f"{name}_canvas.jpg"))
            
            # 保存笔触数据
            canvas.save_stroke_data(str(output_path / f"{name}_strokes.json"))
            
            # 保存统计信息
            stats = canvas.get_canvas_statistics()
            with open(output_path / f"{name}_stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)