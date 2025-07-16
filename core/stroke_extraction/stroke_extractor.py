# -*- coding: utf-8 -*-
"""
笔画特征提取器

从检测到的笔画中提取详细的特征信息
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import ndimage
from skimage import measure, morphology

from .stroke_detector import Stroke


@dataclass
class StrokeFeatures:
    """
    笔画特征数据结构
    
    Attributes:
        geometric_features (Dict): 几何特征
        texture_features (Dict): 纹理特征
        shape_features (Dict): 形状特征
        statistical_features (Dict): 统计特征
    """
    geometric_features: Dict
    texture_features: Dict
    shape_features: Dict
    statistical_features: Dict


class StrokeExtractor:
    """
    笔画特征提取器
    
    从笔画中提取各种特征用于分类和匹配
    """
    
    def __init__(self, config=None):
        """
        初始化特征提取器
        
        Args:
            config: 配置对象
        """
        self.config = config or {}
        
    def extract_stroke_features(self, stroke: Stroke) -> StrokeFeatures:
        """
        提取笔画的完整特征
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            StrokeFeatures: 提取的特征
        """
        # 提取几何特征
        geometric_features = self._extract_geometric_features(stroke)
        
        # 提取纹理特征
        texture_features = self._extract_texture_features(stroke)
        
        # 提取形状特征
        shape_features = self._extract_shape_features(stroke)
        
        # 提取统计特征
        statistical_features = self._extract_statistical_features(stroke)
        
        return StrokeFeatures(
            geometric_features=geometric_features,
            texture_features=texture_features,
            shape_features=shape_features,
            statistical_features=statistical_features
        )
    
    def _extract_geometric_features(self, stroke: Stroke) -> Dict:
        """
        提取几何特征
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            Dict: 几何特征字典
        """
        features = {}
        
        # 基本几何属性
        features['length'] = stroke.length
        features['area'] = stroke.area
        features['orientation'] = stroke.orientation
        features['bbox'] = stroke.bbox
        
        # 长宽比
        bbox_width = stroke.bbox[2]
        bbox_height = stroke.bbox[3]
        features['aspect_ratio'] = bbox_width / max(bbox_height, 1)
        
        # 紧凑度
        features['compactness'] = (4 * np.pi * stroke.area) / (stroke.length ** 2) if stroke.length > 0 else 0
        
        # 骨架曲率
        if len(stroke.skeleton) > 2:
            features['curvature'] = self._calculate_curvature(stroke.skeleton)
        else:
            features['curvature'] = 0
        
        # 端点信息
        if len(stroke.skeleton) > 0:
            features['start_point'] = tuple(stroke.skeleton[0])
            features['end_point'] = tuple(stroke.skeleton[-1])
            features['endpoint_distance'] = np.linalg.norm(stroke.skeleton[-1] - stroke.skeleton[0])
        
        return features
    
    def _extract_texture_features(self, stroke: Stroke) -> Dict:
        """
        提取纹理特征
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            Dict: 纹理特征字典
        """
        features = {}
        
        # 宽度变化特征
        if len(stroke.width_profile) > 0:
            features['width_mean'] = np.mean(stroke.width_profile)
            features['width_std'] = np.std(stroke.width_profile)
            features['width_max'] = np.max(stroke.width_profile)
            features['width_min'] = np.min(stroke.width_profile)
            features['width_range'] = features['width_max'] - features['width_min']
        
        # 纹理变化特征
        if len(stroke.texture_profile) > 0:
            features['texture_mean'] = np.mean(stroke.texture_profile)
            features['texture_std'] = np.std(stroke.texture_profile)
            features['texture_entropy'] = self._calculate_entropy(stroke.texture_profile)
        
        return features
    
    def _extract_shape_features(self, stroke: Stroke) -> Dict:
        """
        提取形状特征
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            Dict: 形状特征字典
        """
        features = {}
        
        # 轮廓特征
        if len(stroke.contour) > 0:
            # 周长
            features['perimeter'] = cv2.arcLength(stroke.contour, True)
            
            # 凸包面积
            hull = cv2.convexHull(stroke.contour)
            features['convex_area'] = cv2.contourArea(hull)
            
            # 凸性缺陷
            if len(hull) > 3:
                defects = cv2.convexityDefects(stroke.contour, cv2.convexHull(stroke.contour, returnPoints=False))
                features['convexity_defects'] = len(defects) if defects is not None else 0
            else:
                features['convexity_defects'] = 0
            
            # 形状矩
            moments = cv2.moments(stroke.contour)
            if moments['m00'] != 0:
                features['hu_moments'] = cv2.HuMoments(moments).flatten()
            else:
                features['hu_moments'] = np.zeros(7)
        
        return features
    
    def _extract_statistical_features(self, stroke: Stroke) -> Dict:
        """
        提取统计特征
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            Dict: 统计特征字典
        """
        features = {}
        
        # 基本统计信息
        features['confidence'] = stroke.confidence
        features['stroke_type'] = stroke.stroke_type
        
        # 骨架点密度
        if stroke.length > 0:
            features['skeleton_density'] = len(stroke.skeleton) / stroke.length
        else:
            features['skeleton_density'] = 0
        
        # 轮廓点密度
        if stroke.area > 0:
            features['contour_density'] = len(stroke.contour) / stroke.area
        else:
            features['contour_density'] = 0
        
        return features
    
    def _calculate_curvature(self, skeleton: np.ndarray) -> float:
        """
        计算骨架的平均曲率
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            
        Returns:
            float: 平均曲率
        """
        if len(skeleton) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(skeleton) - 1):
            p1 = skeleton[i-1]
            p2 = skeleton[i]
            p3 = skeleton[i+1]
            
            # 计算三点间的角度变化
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 避免除零错误
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)  # 确保在有效范围内
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """
        计算数据的熵
        
        Args:
            data (np.ndarray): 输入数据
            
        Returns:
            float: 熵值
        """
        if len(data) == 0:
            return 0.0
        
        # 计算直方图
        hist, _ = np.histogram(data, bins=50, density=True)
        
        # 计算概率
        hist = hist[hist > 0]  # 移除零值
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def extract_features_batch(self, strokes: List[Stroke]) -> List[StrokeFeatures]:
        """
        批量提取笔画特征
        
        Args:
            strokes (List[Stroke]): 笔画列表
            
        Returns:
            List[StrokeFeatures]: 特征列表
        """
        features_list = []
        
        for stroke in strokes:
            features = self.extract_stroke_features(stroke)
            features_list.append(features)
        
        return features_list
    
    def get_feature_vector(self, features: StrokeFeatures) -> np.ndarray:
        """
        将特征转换为向量形式
        
        Args:
            features (StrokeFeatures): 笔画特征
            
        Returns:
            np.ndarray: 特征向量
        """
        vector_parts = []
        
        # 几何特征
        geom = features.geometric_features
        vector_parts.extend([
            geom.get('length', 0),
            geom.get('area', 0),
            geom.get('orientation', 0),
            geom.get('aspect_ratio', 0),
            geom.get('compactness', 0),
            geom.get('curvature', 0),
            geom.get('endpoint_distance', 0)
        ])
        
        # 纹理特征
        texture = features.texture_features
        vector_parts.extend([
            texture.get('width_mean', 0),
            texture.get('width_std', 0),
            texture.get('width_range', 0),
            texture.get('texture_mean', 0),
            texture.get('texture_std', 0),
            texture.get('texture_entropy', 0)
        ])
        
        # 形状特征
        shape = features.shape_features
        vector_parts.extend([
            shape.get('perimeter', 0),
            shape.get('convex_area', 0),
            shape.get('convexity_defects', 0)
        ])
        
        # Hu矩
        hu_moments = shape.get('hu_moments', np.zeros(7))
        vector_parts.extend(hu_moments)
        
        # 统计特征
        stats = features.statistical_features
        vector_parts.extend([
            stats.get('confidence', 0),
            stats.get('skeleton_density', 0),
            stats.get('contour_density', 0)
        ])
        
        return np.array(vector_parts, dtype=np.float32)