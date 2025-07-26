#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
笔触建模核心模块
定义笔触的基础数据结构和属性
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json

class StrokeType(Enum):
    """笔触类型枚举"""
    HORIZONTAL = "horizontal"  # 横笔
    VERTICAL = "vertical"      # 竖笔
    DOT = "dot"               # 点
    HOOK = "hook"             # 钩
    CURVE = "curve"           # 弯
    DIAGONAL = "diagonal"     # 撇捺
    COMPLEX = "complex"       # 复合笔触

# 移除了毛笔类型枚举，专注于基础笔触建模

@dataclass
class StrokeProperties:
    """笔触属性"""
    # 基础几何属性
    area: float = 0.0
    perimeter: float = 0.0
    length: float = 0.0
    width: float = 0.0
    aspect_ratio: float = 0.0
    
    # 形状特征
    circularity: float = 0.0
    convexity: float = 0.0
    solidity: float = 0.0
    
    # 笔触特征
    pressure: float = 0.5      # 笔压 [0,1]
    wetness: float = 0.5       # 墨湿度 [0,1]
    thickness: float = 0.5     # 笔触厚度 [0,1]
    speed: float = 0.5         # 书写速度 [0,1]
    
    # 方向特征
    orientation: float = 0.0   # 主方向角度
    curvature: float = 0.0     # 曲率
    
    # 位置特征
    position_x: float = 0.0
    position_y: float = 0.0
    saliency: float = 0.0      # 位置显著性

class Stroke:
    """增强的笔触类"""
    
    def __init__(self, stroke_id: int, mask: np.ndarray = None, 
                 contour: np.ndarray = None, bbox: Tuple[int, int, int, int] = None,
                 stroke_type: StrokeType = StrokeType.COMPLEX):
        self.id = stroke_id
        self.mask = mask
        self.contour = contour
        self.bbox = bbox or (0, 0, 0, 0)
        self.stroke_type = stroke_type
        
        # 计算基础属性
        if contour is not None:
            self.area = cv2.contourArea(contour)
            self.perimeter = cv2.arcLength(contour, True)
            
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                self.centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            else:
                self.centroid = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        else:
            self.area = 0
            self.perimeter = 0
            self.centroid = (0, 0)
        
        # 笔触属性
        self.properties = StrokeProperties()
        self._compute_properties()
        
        # 骨架点
        self._skeleton_points = None
        
        # 笔触路径（用于动画）
        self.path_points = []
        self.control_points = []
        
    def _compute_properties(self):
        """计算笔触属性"""
        if self.contour is None:
            return
            
        # 基础几何属性
        self.properties.area = self.area
        self.properties.perimeter = self.perimeter
        
        # 长宽比
        if self.bbox[3] > 0:
            self.properties.aspect_ratio = self.bbox[2] / self.bbox[3]
        
        # 圆形度
        if self.perimeter > 0:
            self.properties.circularity = 4 * np.pi * self.area / (self.perimeter ** 2)
        
        # 凸包特征
        if len(self.contour) >= 3:
            hull = cv2.convexHull(self.contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                self.properties.convexity = self.area / hull_area
        
        # 位置特征
        self.properties.position_x = self.centroid[0]
        self.properties.position_y = self.centroid[1]
        
        # 方向特征
        self._compute_orientation()
    
    def _compute_orientation(self):
        """计算主方向"""
        if self.contour is None or len(self.contour) < 5:
            return
            
        # 使用椭圆拟合计算主方向
        try:
            ellipse = cv2.fitEllipse(self.contour)
            self.properties.orientation = ellipse[2]  # 角度
        except:
            self.properties.orientation = 0.0
    
    def get_skeleton_points(self) -> List[Tuple[int, int]]:
        """获取骨架点"""
        if self._skeleton_points is not None:
            return self._skeleton_points
            
        if self.mask is None:
            return [self.centroid]
        
        try:
            from skimage import morphology
            # 使用skimage的骨架化
            binary_mask = self.mask > 0
            skeleton = morphology.skeletonize(binary_mask)
            skeleton = skeleton.astype(np.uint8) * 255
            
            # Harris角点检测找端点
            corners = cv2.cornerHarris(skeleton.astype(np.float32), 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            # 提取角点坐标
            corner_coords = np.where(corners > 0.01 * corners.max())
            skeleton_points = list(zip(corner_coords[1], corner_coords[0]))
            
            self._skeleton_points = skeleton_points if skeleton_points else [self.centroid]
        except:
            self._skeleton_points = [self.centroid]
            
        return self._skeleton_points
    
    def set_stroke_properties(self, pressure: float = 0.5, wetness: float = 0.5, 
                            thickness: float = 0.5, speed: float = 0.5):
        """设置笔触属性"""
        self.properties.pressure = np.clip(pressure, 0.0, 1.0)
        self.properties.wetness = np.clip(wetness, 0.0, 1.0)
        self.properties.thickness = np.clip(thickness, 0.0, 1.0)
        self.properties.speed = np.clip(speed, 0.0, 1.0)
    
    def generate_path(self, num_points: int = 50) -> List[Tuple[int, int]]:
        """生成笔触路径点"""
        if self.contour is None:
            return [self.centroid]
        
        # 简化轮廓
        epsilon = 0.02 * cv2.arcLength(self.contour, True)
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        
        # 生成平滑路径
        if len(approx) >= 2:
            path_points = []
            for i in range(len(approx)):
                point = approx[i][0]
                path_points.append((int(point[0]), int(point[1])))
            
            self.path_points = path_points
        else:
            self.path_points = [self.centroid]
        
        return self.path_points
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'id': self.id,
            'stroke_type': self.stroke_type.value,
            'area': float(self.area),
            'perimeter': float(self.perimeter),
            'centroid': self.centroid,
            'bbox': self.bbox,
            'properties': {
                'pressure': self.properties.pressure,
                'wetness': self.properties.wetness,
                'thickness': self.properties.thickness,
                'speed': self.properties.speed,
                'orientation': self.properties.orientation,
                'circularity': self.properties.circularity,
                'aspect_ratio': self.properties.aspect_ratio,
                'convexity': self.properties.convexity
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Stroke':
        """从字典创建笔触对象"""
        stroke = cls(
            stroke_id=data['id'],
            stroke_type=StrokeType(data.get('stroke_type', 'complex'))
        )
        
        stroke.area = data['area']
        stroke.perimeter = data['perimeter']
        stroke.centroid = tuple(data['centroid'])
        stroke.bbox = tuple(data['bbox'])
        
        # 设置属性
        props = data.get('properties', {})
        stroke.set_stroke_properties(
            pressure=props.get('pressure', 0.5),
            wetness=props.get('wetness', 0.5),
            thickness=props.get('thickness', 0.5),
            speed=props.get('speed', 0.5)
        )
        
        return stroke

class StrokeCollection:
    """笔触集合管理类"""
    
    def __init__(self):
        self.strokes: List[Stroke] = []
        self.metadata = {}
    
    def add_stroke(self, stroke: Stroke):
        """添加笔触"""
        self.strokes.append(stroke)
    
    def remove_stroke(self, stroke_id: int):
        """移除笔触"""
        self.strokes = [s for s in self.strokes if s.id != stroke_id]
    
    def get_stroke(self, stroke_id: int) -> Optional[Stroke]:
        """获取指定笔触"""
        for stroke in self.strokes:
            if stroke.id == stroke_id:
                return stroke
        return None
    
    def filter_by_type(self, stroke_type: StrokeType) -> List[Stroke]:
        """按类型过滤笔触"""
        return [s for s in self.strokes if s.stroke_type == stroke_type]
    
    def filter_by_area(self, min_area: float = 0, max_area: float = float('inf')) -> List[Stroke]:
        """按面积过滤笔触"""
        return [s for s in self.strokes if min_area <= s.area <= max_area]
    
    def sort_by_position(self, reverse: bool = False) -> List[Stroke]:
        """按位置排序（从左到右，从上到下）"""
        return sorted(self.strokes, 
                     key=lambda s: (s.centroid[1], s.centroid[0]), 
                     reverse=reverse)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.strokes:
            return {}
        
        areas = [s.area for s in self.strokes]
        perimeters = [s.perimeter for s in self.strokes]
        
        return {
            'total_strokes': len(self.strokes),
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas)
            },
            'perimeter_stats': {
                'mean': np.mean(perimeters),
                'std': np.std(perimeters),
                'min': np.min(perimeters),
                'max': np.max(perimeters)
            },
            'stroke_types': {t.value: len(self.filter_by_type(t)) for t in StrokeType}
        }
    
    def save_to_json(self, filepath: str):
        """保存到JSON文件"""
        data = {
            'metadata': self.metadata,
            'statistics': self.get_statistics(),
            'strokes': [stroke.to_dict() for stroke in self.strokes]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filepath: str):
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.strokes = [Stroke.from_dict(stroke_data) for stroke_data in data.get('strokes', [])]
    
    def __len__(self):
        return len(self.strokes)
    
    def __iter__(self):
        return iter(self.strokes)
    
    def __getitem__(self, index):
        return self.strokes[index]