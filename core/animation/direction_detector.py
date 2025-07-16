# -*- coding: utf-8 -*-
"""
笔触方向检测器

实现论文中的笔触方向确定：
1. 根据墨湿度梯度确定方向
2. 根据厚度梯度确定方向
3. 从骨架点中选择起始点
4. 计算绘制路径
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


class DirectionMethod(Enum):
    """
    方向检测方法枚举
    """
    WETNESS_GRADIENT = "wetness_gradient"  # 湿度梯度
    THICKNESS_GRADIENT = "thickness_gradient"  # 厚度梯度
    SKELETON_ANALYSIS = "skeleton_analysis"  # 骨架分析
    COMBINED = "combined"  # 组合方法


@dataclass
class StrokeDirection:
    """
    笔触方向数据结构
    """
    start_point: Tuple[int, int]  # 起始点
    end_point: Tuple[int, int]  # 结束点
    direction_vector: Tuple[float, float]  # 方向向量
    confidence: float  # 置信度
    method: DirectionMethod  # 检测方法
    path_points: List[Tuple[int, int]] = None  # 路径点
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.path_points is None:
            self.path_points = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GradientInfo:
    """
    梯度信息数据结构
    """
    magnitude: np.ndarray  # 梯度幅值
    direction: np.ndarray  # 梯度方向
    gradient_x: np.ndarray  # X方向梯度
    gradient_y: np.ndarray  # Y方向梯度
    max_gradient_point: Tuple[int, int]  # 最大梯度点
    min_gradient_point: Tuple[int, int]  # 最小梯度点


class DirectionDetector:
    """
    笔触方向检测器
    
    根据墨湿度和厚度梯度确定笔触绘制方向
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化方向检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 检测参数
        self.gaussian_sigma = config.get('gaussian_sigma', 1.0)
        self.gradient_threshold = config.get('gradient_threshold', 0.1)
        self.skeleton_threshold = config.get('skeleton_threshold', 0.5)
        self.min_confidence = config.get('min_confidence', 0.3)
        
        # 权重参数
        self.wetness_weight = config.get('wetness_weight', 0.6)
        self.thickness_weight = config.get('thickness_weight', 0.4)
        self.skeleton_weight = config.get('skeleton_weight', 0.8)
        
        # 路径参数
        self.path_smoothing = config.get('path_smoothing', True)
        self.path_simplification = config.get('path_simplification', True)
        self.max_path_points = config.get('max_path_points', 100)
        
        # 缓存
        self._gradient_cache = {}
        self._skeleton_cache = {}
    
    def detect_direction(self, stroke_mask: np.ndarray, 
                        stroke_features: Dict[str, Any],
                        method: DirectionMethod = DirectionMethod.COMBINED) -> StrokeDirection:
        """
        检测笔触方向
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            method: 检测方法
            
        Returns:
            StrokeDirection: 笔触方向信息
        """
        if stroke_mask is None or stroke_mask.size == 0:
            return self._create_default_direction()
        
        try:
            if method == DirectionMethod.WETNESS_GRADIENT:
                return self._detect_by_wetness_gradient(stroke_mask, stroke_features)
            elif method == DirectionMethod.THICKNESS_GRADIENT:
                return self._detect_by_thickness_gradient(stroke_mask, stroke_features)
            elif method == DirectionMethod.SKELETON_ANALYSIS:
                return self._detect_by_skeleton_analysis(stroke_mask, stroke_features)
            elif method == DirectionMethod.COMBINED:
                return self._detect_by_combined_method(stroke_mask, stroke_features)
            else:
                self.logger.warning(f"Unknown detection method: {method}")
                return self._create_default_direction()
                
        except Exception as e:
            self.logger.error(f"Error detecting direction: {str(e)}")
            return self._create_default_direction()
    
    def _detect_by_wetness_gradient(self, stroke_mask: np.ndarray, 
                                   stroke_features: Dict[str, Any]) -> StrokeDirection:
        """
        基于湿度梯度检测方向
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            StrokeDirection: 方向信息
        """
        # 获取湿度信息
        wetness_map = self._extract_wetness_map(stroke_mask, stroke_features)
        if wetness_map is None:
            return self._create_default_direction()
        
        # 计算湿度梯度
        gradient_info = self._calculate_gradient(wetness_map)
        
        # 确定起始点（湿度最高点）和结束点（湿度最低点）
        wet_points = self._find_extreme_points(wetness_map, stroke_mask, find_max=True)
        dry_points = self._find_extreme_points(wetness_map, stroke_mask, find_max=False)
        
        if not wet_points or not dry_points:
            return self._create_default_direction()
        
        start_point = wet_points[0]  # 最湿的点作为起始点
        end_point = dry_points[0]    # 最干的点作为结束点
        
        # 计算方向向量
        direction_vector = self._calculate_direction_vector(start_point, end_point)
        
        # 计算置信度
        confidence = self._calculate_wetness_confidence(gradient_info, start_point, end_point)
        
        # 生成路径
        path_points = self._generate_path(start_point, end_point, stroke_mask, wetness_map)
        
        return StrokeDirection(
            start_point=start_point,
            end_point=end_point,
            direction_vector=direction_vector,
            confidence=confidence,
            method=DirectionMethod.WETNESS_GRADIENT,
            path_points=path_points,
            metadata={
                'wetness_range': (np.min(wetness_map), np.max(wetness_map)),
                'gradient_magnitude': np.mean(gradient_info.magnitude)
            }
        )
    
    def _detect_by_thickness_gradient(self, stroke_mask: np.ndarray, 
                                     stroke_features: Dict[str, Any]) -> StrokeDirection:
        """
        基于厚度梯度检测方向
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            StrokeDirection: 方向信息
        """
        # 获取厚度信息
        thickness_map = self._extract_thickness_map(stroke_mask, stroke_features)
        if thickness_map is None:
            return self._create_default_direction()
        
        # 计算厚度梯度
        gradient_info = self._calculate_gradient(thickness_map)
        
        # 确定起始点（厚度最大点）和结束点（厚度最小点）
        thick_points = self._find_extreme_points(thickness_map, stroke_mask, find_max=True)
        thin_points = self._find_extreme_points(thickness_map, stroke_mask, find_max=False)
        
        if not thick_points or not thin_points:
            return self._create_default_direction()
        
        start_point = thick_points[0]  # 最粗的点作为起始点
        end_point = thin_points[0]     # 最细的点作为结束点
        
        # 计算方向向量
        direction_vector = self._calculate_direction_vector(start_point, end_point)
        
        # 计算置信度
        confidence = self._calculate_thickness_confidence(gradient_info, start_point, end_point)
        
        # 生成路径
        path_points = self._generate_path(start_point, end_point, stroke_mask, thickness_map)
        
        return StrokeDirection(
            start_point=start_point,
            end_point=end_point,
            direction_vector=direction_vector,
            confidence=confidence,
            method=DirectionMethod.THICKNESS_GRADIENT,
            path_points=path_points,
            metadata={
                'thickness_range': (np.min(thickness_map), np.max(thickness_map)),
                'gradient_magnitude': np.mean(gradient_info.magnitude)
            }
        )
    
    def _detect_by_skeleton_analysis(self, stroke_mask: np.ndarray, 
                                   stroke_features: Dict[str, Any]) -> StrokeDirection:
        """
        基于骨架分析检测方向
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            StrokeDirection: 方向信息
        """
        # 提取骨架
        skeleton = self._extract_skeleton(stroke_mask)
        if skeleton is None or np.sum(skeleton) == 0:
            return self._create_default_direction()
        
        # 找到骨架端点
        endpoints = self._find_skeleton_endpoints(skeleton)
        if len(endpoints) < 2:
            return self._create_default_direction()
        
        # 选择最佳的起始点和结束点
        start_point, end_point = self._select_best_endpoints(endpoints, stroke_mask, stroke_features)
        
        # 计算方向向量
        direction_vector = self._calculate_direction_vector(start_point, end_point)
        
        # 计算置信度
        confidence = self._calculate_skeleton_confidence(skeleton, start_point, end_point)
        
        # 沿骨架生成路径
        path_points = self._trace_skeleton_path(skeleton, start_point, end_point)
        
        return StrokeDirection(
            start_point=start_point,
            end_point=end_point,
            direction_vector=direction_vector,
            confidence=confidence,
            method=DirectionMethod.SKELETON_ANALYSIS,
            path_points=path_points,
            metadata={
                'skeleton_length': np.sum(skeleton),
                'num_endpoints': len(endpoints),
                'skeleton_complexity': self._calculate_skeleton_complexity(skeleton)
            }
        )
    
    def _detect_by_combined_method(self, stroke_mask: np.ndarray, 
                                 stroke_features: Dict[str, Any]) -> StrokeDirection:
        """
        基于组合方法检测方向
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            StrokeDirection: 方向信息
        """
        # 分别使用不同方法检测
        wetness_direction = self._detect_by_wetness_gradient(stroke_mask, stroke_features)
        thickness_direction = self._detect_by_thickness_gradient(stroke_mask, stroke_features)
        skeleton_direction = self._detect_by_skeleton_analysis(stroke_mask, stroke_features)
        
        # 计算加权平均
        directions = [
            (wetness_direction, self.wetness_weight),
            (thickness_direction, self.thickness_weight),
            (skeleton_direction, self.skeleton_weight)
        ]
        
        # 选择置信度最高的方向作为基础
        best_direction = max(directions, key=lambda x: x[0].confidence * x[1])[0]
        
        # 融合其他方向的信息
        combined_confidence = self._calculate_combined_confidence(directions)
        combined_path = self._combine_paths([d[0].path_points for d in directions])
        
        return StrokeDirection(
            start_point=best_direction.start_point,
            end_point=best_direction.end_point,
            direction_vector=best_direction.direction_vector,
            confidence=combined_confidence,
            method=DirectionMethod.COMBINED,
            path_points=combined_path,
            metadata={
                'wetness_confidence': wetness_direction.confidence,
                'thickness_confidence': thickness_direction.confidence,
                'skeleton_confidence': skeleton_direction.confidence,
                'best_method': best_direction.method.value
            }
        )
    
    def _extract_wetness_map(self, stroke_mask: np.ndarray, 
                           stroke_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        提取湿度图
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            Optional[np.ndarray]: 湿度图
        """
        # 尝试从特征中获取湿度信息
        if 'wetness_map' in stroke_features:
            return stroke_features['wetness_map']
        
        # 如果没有湿度图，基于像素强度估算
        if 'image' in stroke_features:
            image = stroke_features['image']
            if len(image.shape) == 3:
                # 转换为灰度图
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 应用掩码
            wetness_map = gray.astype(np.float32) / 255.0
            mask_indices = stroke_mask == 0
            wetness_map[mask_indices] = 0
            
            # 高斯平滑
            wetness_map = gaussian_filter(wetness_map, sigma=self.gaussian_sigma)
            
            return wetness_map
        
        # 如果都没有，基于掩码距离变换估算
        distance_transform = cv2.distanceTransform(
            stroke_mask.astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        # 归一化距离变换作为湿度近似
        max_distance = np.max(distance_transform)
        wetness_map = distance_transform / max_distance if max_distance > 0 else distance_transform
        
        return wetness_map
    
    def _extract_thickness_map(self, stroke_mask: np.ndarray, 
                             stroke_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        提取厚度图
        
        Args:
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            Optional[np.ndarray]: 厚度图
        """
        # 尝试从特征中获取厚度信息
        if 'thickness_map' in stroke_features:
            return stroke_features['thickness_map']
        
        # 基于距离变换计算厚度
        distance_transform = cv2.distanceTransform(
            stroke_mask.astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        
        # 厚度与到边界的距离成正比
        thickness_map = distance_transform
        
        # 如果有像素强度信息，结合使用
        if 'image' in stroke_features:
            image = stroke_features['image']
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 反转亮度作为厚度指示（暗像素=厚笔触）
            intensity_thickness = (255 - gray.astype(np.float32)) / 255.0
            mask_indices = stroke_mask == 0
            intensity_thickness[mask_indices] = 0
            
            # 结合距离变换和强度信息
            thickness_map = 0.7 * distance_transform + 0.3 * intensity_thickness * np.max(distance_transform)
        
        # 高斯平滑
        thickness_map = gaussian_filter(thickness_map, sigma=self.gaussian_sigma)
        
        return thickness_map
    
    def _extract_skeleton(self, stroke_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        提取骨架
        
        Args:
            stroke_mask: 笔触掩码
            
        Returns:
            Optional[np.ndarray]: 骨架
        """
        try:
            # 使用skimage的骨架化算法
            skeleton = skeletonize(stroke_mask > 0)
            return skeleton.astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error extracting skeleton: {str(e)}")
            return None
    
    def _calculate_gradient(self, value_map: np.ndarray) -> GradientInfo:
        """
        计算梯度信息
        
        Args:
            value_map: 值图（湿度或厚度）
            
        Returns:
            GradientInfo: 梯度信息
        """
        # 计算梯度
        gradient_y, gradient_x = np.gradient(value_map)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # 找到极值点
        valid_mask = magnitude > self.gradient_threshold
        has_valid_points = bool(np.any(valid_mask))
        if has_valid_points:
            max_idx = np.unravel_index(np.argmax(magnitude * valid_mask), magnitude.shape)
            min_idx = np.unravel_index(np.argmin(value_map + (1 - valid_mask) * np.max(value_map)), value_map.shape)
        else:
            max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            min_idx = np.unravel_index(np.argmin(value_map), value_map.shape)
        
        return GradientInfo(
            magnitude=magnitude,
            direction=direction,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            max_gradient_point=(max_idx[1], max_idx[0]),  # (x, y)
            min_gradient_point=(min_idx[1], min_idx[0])   # (x, y)
        )
    
    def _find_extreme_points(self, value_map: np.ndarray, mask: np.ndarray, 
                           find_max: bool = True, num_points: int = 3) -> List[Tuple[int, int]]:
        """
        找到极值点
        
        Args:
            value_map: 值图
            mask: 掩码
            find_max: 是否找最大值点
            num_points: 返回点的数量
            
        Returns:
            List[Tuple[int, int]]: 极值点列表
        """
        # 应用掩码
        masked_values = value_map.copy()
        fill_value = np.min(value_map) if find_max else np.max(value_map)
        mask_indices = mask == 0
        masked_values[mask_indices] = fill_value
        
        # 找到极值点
        points = []
        temp_map = masked_values.copy()
        
        for _ in range(num_points):
            if find_max:
                idx = np.unravel_index(np.argmax(temp_map), temp_map.shape)
            else:
                idx = np.unravel_index(np.argmin(temp_map), temp_map.shape)
            
            point = (idx[1], idx[0])  # (x, y)
            points.append(point)
            
            # 在该点周围设置无效值，避免重复选择
            y, x = idx
            radius = 5
            y_min, y_max = max(0, y-radius), min(temp_map.shape[0], y+radius+1)
            x_min, x_max = max(0, x-radius), min(temp_map.shape[1], x+radius+1)
            
            if find_max:
                temp_map[y_min:y_max, x_min:x_max] = np.min(temp_map)
            else:
                temp_map[y_min:y_max, x_min:x_max] = np.max(temp_map)
        
        return points
    
    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        找到骨架端点
        
        Args:
            skeleton: 骨架图
            
        Returns:
            List[Tuple[int, int]]: 端点列表
        """
        # 使用3x3卷积核计算邻居数量
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # 端点：骨架点且只有一个邻居
        endpoints_mask = (skeleton > 0) & (neighbor_count == 1)
        
        # 获取端点坐标
        y_coords, x_coords = np.where(endpoints_mask)
        endpoints = [(x, y) for x, y in zip(x_coords, y_coords)]
        
        return endpoints
    
    def _select_best_endpoints(self, endpoints: List[Tuple[int, int]], 
                             stroke_mask: np.ndarray, 
                             stroke_features: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        选择最佳的起始点和结束点
        
        Args:
            endpoints: 端点列表
            stroke_mask: 笔触掩码
            stroke_features: 笔触特征
            
        Returns:
            Tuple: (起始点, 结束点)
        """
        if len(endpoints) < 2:
            # 如果端点不足，使用质心和最远点
            centroid = self._calculate_centroid(stroke_mask)
            farthest = self._find_farthest_point(centroid, endpoints + [centroid])
            return centroid, farthest
        
        # 获取湿度和厚度信息
        wetness_map = self._extract_wetness_map(stroke_mask, stroke_features)
        thickness_map = self._extract_thickness_map(stroke_mask, stroke_features)
        
        # 为每个端点计算得分
        endpoint_scores = []
        for point in endpoints:
            x, y = point
            
            # 湿度得分（湿度高的点作为起始点）
            wetness_score = wetness_map[y, x] if wetness_map is not None else 0.5
            
            # 厚度得分（厚度大的点作为起始点）
            thickness_score = thickness_map[y, x] if thickness_map is not None else 0.5
            
            # 综合得分
            total_score = self.wetness_weight * wetness_score + self.thickness_weight * thickness_score
            endpoint_scores.append((point, total_score))
        
        # 排序并选择最佳起始点和结束点
        endpoint_scores.sort(key=lambda x: x[1], reverse=True)
        
        start_point = endpoint_scores[0][0]  # 得分最高的作为起始点
        end_point = endpoint_scores[-1][0]   # 得分最低的作为结束点
        
        return start_point, end_point
    
    def _calculate_direction_vector(self, start_point: Tuple[int, int], 
                                  end_point: Tuple[int, int]) -> Tuple[float, float]:
        """
        计算方向向量
        
        Args:
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            Tuple[float, float]: 归一化方向向量
        """
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        # 归一化
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            return (dx / length, dy / length)
        else:
            return (1.0, 0.0)  # 默认水平方向
    
    def _calculate_wetness_confidence(self, gradient_info: GradientInfo, 
                                    start_point: Tuple[int, int], 
                                    end_point: Tuple[int, int]) -> float:
        """
        计算湿度方向的置信度
        
        Args:
            gradient_info: 梯度信息
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            float: 置信度
        """
        # 基于梯度幅值的置信度
        avg_magnitude = np.mean(gradient_info.magnitude)
        magnitude_confidence = min(1.0, avg_magnitude / 0.5)  # 归一化
        
        # 基于起始点和结束点距离的置信度
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        distance_confidence = min(1.0, distance / 50.0)  # 归一化
        
        # 综合置信度
        confidence = 0.7 * magnitude_confidence + 0.3 * distance_confidence
        return max(self.min_confidence, confidence)
    
    def _calculate_thickness_confidence(self, gradient_info: GradientInfo, 
                                      start_point: Tuple[int, int], 
                                      end_point: Tuple[int, int]) -> float:
        """
        计算厚度方向的置信度
        
        Args:
            gradient_info: 梯度信息
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            float: 置信度
        """
        # 类似于湿度置信度计算
        avg_magnitude = np.mean(gradient_info.magnitude)
        magnitude_confidence = min(1.0, avg_magnitude / 0.3)  # 厚度梯度通常较小
        
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        distance_confidence = min(1.0, distance / 40.0)
        
        confidence = 0.6 * magnitude_confidence + 0.4 * distance_confidence
        return max(self.min_confidence, confidence)
    
    def _calculate_skeleton_confidence(self, skeleton: np.ndarray, 
                                     start_point: Tuple[int, int], 
                                     end_point: Tuple[int, int]) -> float:
        """
        计算骨架方向的置信度
        
        Args:
            skeleton: 骨架图
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            float: 置信度
        """
        # 基于骨架连通性的置信度
        skeleton_length = np.sum(skeleton)
        length_confidence = min(1.0, skeleton_length / 100.0)
        
        # 基于骨架复杂度的置信度
        complexity = self._calculate_skeleton_complexity(skeleton)
        complexity_confidence = max(0.3, 1.0 - complexity / 5.0)  # 复杂度越低置信度越高
        
        # 基于端点距离的置信度
        distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        distance_confidence = min(1.0, distance / 60.0)
        
        confidence = 0.4 * length_confidence + 0.3 * complexity_confidence + 0.3 * distance_confidence
        return max(self.min_confidence, confidence)
    
    def _calculate_combined_confidence(self, directions: List[Tuple[StrokeDirection, float]]) -> float:
        """
        计算组合方法的置信度
        
        Args:
            directions: 方向和权重列表
            
        Returns:
            float: 组合置信度
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for direction, weight in directions:
            weighted_sum += direction.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self.min_confidence
    
    def _generate_path(self, start_point: Tuple[int, int], end_point: Tuple[int, int], 
                      stroke_mask: np.ndarray, value_map: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        生成绘制路径
        
        Args:
            start_point: 起始点
            end_point: 结束点
            stroke_mask: 笔触掩码
            value_map: 值图（用于路径优化）
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        # 简单的直线路径
        path_points = self._generate_line_path(start_point, end_point)
        
        # 如果有值图，优化路径
        if value_map is not None:
            path_points = self._optimize_path(path_points, stroke_mask, value_map)
        
        # 路径平滑
        if self.path_smoothing:
            path_points = self._smooth_path(path_points)
        
        # 路径简化
        if self.path_simplification:
            path_points = self._simplify_path(path_points)
        
        # 限制路径点数量
        if len(path_points) > self.max_path_points:
            step = len(path_points) // self.max_path_points
            path_points = path_points[::step]
        
        return path_points
    
    def _trace_skeleton_path(self, skeleton: np.ndarray, 
                           start_point: Tuple[int, int], 
                           end_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        沿骨架追踪路径
        
        Args:
            skeleton: 骨架图
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        # 使用A*算法或简单的贪心搜索沿骨架找路径
        path = [start_point]
        current = start_point
        visited = set([current])
        
        while current != end_point and len(path) < self.max_path_points:
            # 找到下一个骨架点
            next_point = self._find_next_skeleton_point(current, end_point, skeleton, visited)
            if next_point is None:
                break
            
            path.append(next_point)
            visited.add(next_point)
            current = next_point
        
        # 确保包含结束点
        if not np.array_equal(path[-1], end_point):
            path.append(end_point)
        
        return path
    
    def _find_next_skeleton_point(self, current: Tuple[int, int], target: Tuple[int, int], 
                                skeleton: np.ndarray, visited: set) -> Optional[Tuple[int, int]]:
        """
        找到下一个骨架点
        
        Args:
            current: 当前点
            target: 目标点
            skeleton: 骨架图
            visited: 已访问点集合
            
        Returns:
            Optional[Tuple[int, int]]: 下一个点
        """
        x, y = current
        target_x, target_y = target
        
        # 搜索8邻域
        candidates = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                    skeleton[ny, nx] > 0 and (nx, ny) not in visited):
                    
                    # 计算到目标的距离
                    distance = np.sqrt((nx - target_x)**2 + (ny - target_y)**2)
                    candidates.append(((nx, ny), distance))
        
        if candidates:
            # 选择距离目标最近的点
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return None
    
    def _generate_line_path(self, start_point: Tuple[int, int], 
                          end_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        生成直线路径
        
        Args:
            start_point: 起始点
            end_point: 结束点
            
        Returns:
            List[Tuple[int, int]]: 路径点列表
        """
        x1, y1 = start_point
        x2, y2 = end_point
        
        # 使用Bresenham算法生成直线
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        
        if dx > dy:
            error = dx / 2
            y = y1
            for x in range(x1, x2 + x_step, x_step):
                points.append((x, y))
                error -= dy
                if error < 0:
                    y += y_step
                    error += dx
        else:
            error = dy / 2
            x = x1
            for y in range(y1, y2 + y_step, y_step):
                points.append((x, y))
                error -= dx
                if error < 0:
                    x += x_step
                    error += dy
        
        return points
    
    def _optimize_path(self, path_points: List[Tuple[int, int]], 
                      stroke_mask: np.ndarray, value_map: np.ndarray) -> List[Tuple[int, int]]:
        """
        优化路径
        
        Args:
            path_points: 原始路径点
            stroke_mask: 笔触掩码
            value_map: 值图
            
        Returns:
            List[Tuple[int, int]]: 优化后的路径点
        """
        # 简单的路径优化：确保路径点在笔触内
        optimized_points = []
        
        for point in path_points:
            x, y = point
            if (0 <= x < stroke_mask.shape[1] and 0 <= y < stroke_mask.shape[0] and
                stroke_mask[y, x] > 0):
                optimized_points.append(point)
            else:
                # 找到最近的有效点
                nearest_point = self._find_nearest_valid_point(point, stroke_mask)
                if nearest_point:
                    optimized_points.append(nearest_point)
        
        return optimized_points if optimized_points else path_points
    
    def _smooth_path(self, path_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        平滑路径
        
        Args:
            path_points: 原始路径点
            
        Returns:
            List[Tuple[int, int]]: 平滑后的路径点
        """
        if len(path_points) < 3:
            return path_points
        
        # 使用移动平均平滑
        smoothed_points = [path_points[0]]  # 保持起始点
        
        for i in range(1, len(path_points) - 1):
            prev_point = path_points[i - 1]
            curr_point = path_points[i]
            next_point = path_points[i + 1]
            
            # 计算平均位置
            avg_x = (prev_point[0] + curr_point[0] + next_point[0]) // 3
            avg_y = (prev_point[1] + curr_point[1] + next_point[1]) // 3
            
            smoothed_points.append((avg_x, avg_y))
        
        smoothed_points.append(path_points[-1])  # 保持结束点
        
        return smoothed_points
    
    def _simplify_path(self, path_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        简化路径（移除冗余点）
        
        Args:
            path_points: 原始路径点
            
        Returns:
            List[Tuple[int, int]]: 简化后的路径点
        """
        if len(path_points) < 3:
            return path_points
        
        # 使用Douglas-Peucker算法的简化版本
        simplified_points = [path_points[0]]
        
        i = 0
        while i < len(path_points) - 1:
            current = path_points[i]
            
            # 找到下一个显著不同的点
            j = i + 1
            while j < len(path_points):
                next_point = path_points[j]
                distance = np.sqrt((next_point[0] - current[0])**2 + (next_point[1] - current[1])**2)
                
                if distance > 3:  # 阈值
                    break
                j += 1
            
            if j < len(path_points):
                simplified_points.append(path_points[j])
                i = j
            else:
                break
        
        # 确保包含最后一个点
        if not np.array_equal(simplified_points[-1], path_points[-1]):
            simplified_points.append(path_points[-1])
        
        return simplified_points
    
    def _combine_paths(self, path_lists: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """
        组合多个路径
        
        Args:
            path_lists: 路径列表
            
        Returns:
            List[Tuple[int, int]]: 组合后的路径
        """
        # 选择最长的路径作为基础
        if not path_lists or not any(path_lists):
            return []
        
        valid_paths = [path for path in path_lists if path]
        if not valid_paths:
            return []
        
        # 返回最长的路径
        return max(valid_paths, key=len)
    
    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        计算质心
        
        Args:
            mask: 掩码
            
        Returns:
            Tuple[int, int]: 质心坐标
        """
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            return (centroid_x, centroid_y)
        else:
            return (mask.shape[1] // 2, mask.shape[0] // 2)
    
    def _find_farthest_point(self, reference: Tuple[int, int], 
                           points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        找到距离参考点最远的点
        
        Args:
            reference: 参考点
            points: 候选点列表
            
        Returns:
            Tuple[int, int]: 最远点
        """
        if not points:
            return reference
        
        max_distance = -1
        farthest_point = points[0]
        
        for point in points:
            distance = np.sqrt((point[0] - reference[0])**2 + (point[1] - reference[1])**2)
            if distance > max_distance:
                max_distance = distance
                farthest_point = point
        
        return farthest_point
    
    def _find_nearest_valid_point(self, point: Tuple[int, int], 
                                mask: np.ndarray, max_radius: int = 10) -> Optional[Tuple[int, int]]:
        """
        找到最近的有效点
        
        Args:
            point: 目标点
            mask: 掩码
            max_radius: 最大搜索半径
            
        Returns:
            Optional[Tuple[int, int]]: 最近的有效点
        """
        x, y = point
        
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and
                            mask[ny, nx] > 0):
                            return (nx, ny)
        
        return None
    
    def _calculate_skeleton_complexity(self, skeleton: np.ndarray) -> float:
        """
        计算骨架复杂度
        
        Args:
            skeleton: 骨架图
            
        Returns:
            float: 复杂度值
        """
        # 计算分支点数量
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        branch_points = np.sum((skeleton > 0) & (neighbor_count > 2))
        
        # 复杂度 = 分支点数量 / 骨架长度
        skeleton_length = np.sum(skeleton)
        if skeleton_length > 0:
            complexity = branch_points / skeleton_length
        else:
            complexity = 0.0
        
        return complexity
    
    def _create_default_direction(self) -> StrokeDirection:
        """
        创建默认方向
        
        Returns:
            StrokeDirection: 默认方向
        """
        return StrokeDirection(
            start_point=(0, 0),
            end_point=(1, 0),
            direction_vector=(1.0, 0.0),
            confidence=self.min_confidence,
            method=DirectionMethod.COMBINED,
            path_points=[(0, 0), (1, 0)],
            metadata={'error': 'default_direction'}
        )
    
    def visualize_direction(self, stroke_mask: np.ndarray, direction: StrokeDirection, 
                          output_path: Optional[str] = None) -> np.ndarray:
        """
        可视化方向检测结果
        
        Args:
            stroke_mask: 笔触掩码
            direction: 方向信息
            output_path: 输出路径
            
        Returns:
            np.ndarray: 可视化图像
        """
        # 创建彩色图像
        vis_image = cv2.cvtColor((stroke_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # 绘制起始点（绿色）
        cv2.circle(vis_image, direction.start_point, 5, (0, 255, 0), -1)
        
        # 绘制结束点（红色）
        cv2.circle(vis_image, direction.end_point, 5, (0, 0, 255), -1)
        
        # 绘制方向箭头
        cv2.arrowedLine(vis_image, direction.start_point, direction.end_point, (255, 0, 0), 2)
        
        # 绘制路径（蓝色）
        if direction.path_points and len(direction.path_points) > 1:
            for i in range(len(direction.path_points) - 1):
                pt1 = direction.path_points[i]
                pt2 = direction.path_points[i + 1]
                cv2.line(vis_image, pt1, pt2, (255, 255, 0), 1)
        
        # 添加文本信息
        text = f"Method: {direction.method.value}, Confidence: {direction.confidence:.3f}"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存图像
        if output_path:
            cv2.imwrite(output_path, vis_image)
            self.logger.info(f"Direction visualization saved to {output_path}")
        
        return vis_image