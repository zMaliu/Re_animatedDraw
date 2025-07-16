# -*- coding: utf-8 -*-
"""
笔画分割器

将复杂的笔画分割成更简单的子笔画
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import ndimage
from skimage import measure, morphology

from .stroke_detector import Stroke


@dataclass
class SegmentationResult:
    """
    分割结果数据结构
    
    Attributes:
        segments (List[Stroke]): 分割后的笔画段
        junction_points (List[Tuple]): 连接点位置
        segmentation_confidence (float): 分割置信度
    """
    segments: List[Stroke]
    junction_points: List[Tuple[int, int]]
    segmentation_confidence: float


class StrokeSegmenter:
    """
    笔画分割器
    
    将复杂笔画分割成简单的基本笔画单元
    """
    
    def __init__(self, config=None):
        """
        初始化分割器
        
        Args:
            config: 配置对象
        """
        self.config = config or {}
        
        # 分割参数
        self.min_segment_length = self.config.get('min_segment_length', 10)
        self.curvature_threshold = self.config.get('curvature_threshold', 0.5)
        self.width_change_threshold = self.config.get('width_change_threshold', 0.3)
        self.junction_detection_radius = self.config.get('junction_detection_radius', 5)
        
    def segment_stroke(self, stroke: Stroke) -> SegmentationResult:
        """
        分割单个笔画
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            SegmentationResult: 分割结果
        """
        # 检测分割点
        split_points = self._detect_split_points(stroke)
        
        # 如果没有分割点，返回原笔画
        if not split_points:
            return SegmentationResult(
                segments=[stroke],
                junction_points=[],
                segmentation_confidence=1.0
            )
        
        # 执行分割
        segments = self._split_stroke(stroke, split_points)
        
        # 计算分割置信度
        confidence = self._calculate_segmentation_confidence(stroke, segments)
        
        return SegmentationResult(
            segments=segments,
            junction_points=split_points,
            segmentation_confidence=confidence
        )
    
    def _detect_split_points(self, stroke: Stroke) -> List[Tuple[int, int]]:
        """
        检测笔画的分割点
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            List[Tuple[int, int]]: 分割点列表
        """
        split_points = []
        
        # 基于曲率的分割点检测
        curvature_points = self._detect_curvature_split_points(stroke)
        split_points.extend(curvature_points)
        
        # 基于宽度变化的分割点检测
        width_points = self._detect_width_split_points(stroke)
        split_points.extend(width_points)
        
        # 基于连接点的分割点检测
        junction_points = self._detect_junction_points(stroke)
        split_points.extend(junction_points)
        
        # 去重和排序
        split_points = self._merge_nearby_points(split_points)
        
        return split_points
    
    def _detect_curvature_split_points(self, stroke: Stroke) -> List[Tuple[int, int]]:
        """
        基于曲率变化检测分割点
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            List[Tuple[int, int]]: 曲率分割点
        """
        if len(stroke.skeleton) < 5:
            return []
        
        split_points = []
        curvatures = self._calculate_local_curvatures(stroke.skeleton)
        
        # 寻找曲率峰值
        for i in range(2, len(curvatures) - 2):
            # 安全获取曲率值，避免数组真值歧义
            from utils.math_utils import ensure_scalar
            
            # 使用通用的标量转换函数
            curr_val = ensure_scalar(curvatures[i])
            prev_val = ensure_scalar(curvatures[i-1])
            next_val = ensure_scalar(curvatures[i+1])
            
            if (curr_val > self.curvature_threshold and
                curr_val > prev_val and
                curr_val > next_val):
                
                point = tuple(stroke.skeleton[i])
                split_points.append(point)
        
        return split_points
    
    def _detect_width_split_points(self, stroke: Stroke) -> List[Tuple[int, int]]:
        """
        基于宽度变化检测分割点
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            List[Tuple[int, int]]: 宽度分割点
        """
        if len(stroke.width_profile) < 5:
            return []
        
        split_points = []
        width_changes = np.diff(stroke.width_profile)
        
        # 寻找宽度急剧变化的点
        threshold = self.width_change_threshold * np.std(width_changes)
        
        for i in range(1, len(width_changes) - 1):
            # 安全比较，避免数组真值歧义
            width_change_val = width_changes[i]
            if isinstance(width_change_val, np.ndarray):
                width_change_scalar = width_change_val.item() if width_change_val.size == 1 else width_change_val
            else:
                width_change_scalar = width_change_val
                
            if abs(width_change_scalar) > threshold:
                if i < len(stroke.skeleton):
                    point = tuple(stroke.skeleton[i])
                    split_points.append(point)
        
        return split_points
    
    def _detect_junction_points(self, stroke: Stroke) -> List[Tuple[int, int]]:
        """
        检测笔画连接点
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            List[Tuple[int, int]]: 连接点
        """
        # 这里可以实现更复杂的连接点检测算法
        # 目前返回空列表，表示没有检测到连接点
        return []
    
    def _calculate_local_curvatures(self, skeleton: np.ndarray) -> np.ndarray:
        """
        计算骨架上每点的局部曲率
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            
        Returns:
            np.ndarray: 曲率数组
        """
        if len(skeleton) < 3:
            return np.array([])
        
        curvatures = np.zeros(len(skeleton))
        
        for i in range(1, len(skeleton) - 1):
            p1 = skeleton[i-1]
            p2 = skeleton[i]
            p3 = skeleton[i+1]
            
            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算角度变化
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures[i] = angle
        
        return curvatures
    
    def _merge_nearby_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        合并距离很近的分割点
        
        Args:
            points (List[Tuple[int, int]]): 原始分割点
            
        Returns:
            List[Tuple[int, int]]: 合并后的分割点
        """
        if not points:
            return []
        
        merged_points = []
        points = sorted(points, key=lambda p: (p[0], p[1]))
        
        current_group = [points[0]]
        
        for i in range(1, len(points)):
            # 计算与当前组中心的距离
            group_center = np.mean(current_group, axis=0)
            points_i_array = np.array(points[i])
            distance = np.linalg.norm(points_i_array - group_center)
            
            if distance < self.junction_detection_radius:
                current_group.append(points[i])
            else:
                # 添加当前组的中心点
                center = tuple(np.mean(current_group, axis=0).astype(int))
                merged_points.append(center)
                current_group = [points[i]]
        
        # 添加最后一组
        if current_group:
            center = tuple(np.mean(current_group, axis=0).astype(int))
            merged_points.append(center)
        
        return merged_points
    
    def _split_stroke(self, stroke: Stroke, split_points: List[Tuple[int, int]]) -> List[Stroke]:
        """
        根据分割点分割笔画
        
        Args:
            stroke (Stroke): 原始笔画
            split_points (List[Tuple[int, int]]): 分割点
            
        Returns:
            List[Stroke]: 分割后的笔画段
        """
        if not split_points:
            return [stroke]
        
        segments = []
        
        # 找到分割点在骨架中的索引
        split_indices = []
        for split_point in split_points:
            # 找到最近的骨架点
            split_point_array = np.array(split_point)
            distances = [np.linalg.norm(split_point_array - skeleton_point) 
                        for skeleton_point in stroke.skeleton]
            closest_index = np.argmin(distances)
            split_indices.append(closest_index)
        
        split_indices = sorted(set(split_indices))  # 去重并排序
        
        # 分割骨架
        start_idx = 0
        for split_idx in split_indices:
            if split_idx > start_idx + self.min_segment_length:
                segment = self._create_segment(stroke, start_idx, split_idx)
                if segment:
                    segments.append(segment)
                start_idx = split_idx
        
        # 添加最后一段
        if start_idx < len(stroke.skeleton) - self.min_segment_length:
            segment = self._create_segment(stroke, start_idx, len(stroke.skeleton))
            if segment:
                segments.append(segment)
        
        return segments if segments else [stroke]
    
    def _create_segment(self, original_stroke: Stroke, start_idx: int, end_idx: int) -> Optional[Stroke]:
        """
        创建笔画段
        
        Args:
            original_stroke (Stroke): 原始笔画
            start_idx (int): 起始索引
            end_idx (int): 结束索引
            
        Returns:
            Optional[Stroke]: 笔画段，如果无效则返回None
        """
        if end_idx <= start_idx or end_idx - start_idx < self.min_segment_length:
            return None
        
        # 提取骨架段
        segment_skeleton = original_stroke.skeleton[start_idx:end_idx]
        
        # 提取对应的宽度和纹理信息
        segment_width = original_stroke.width_profile[start_idx:end_idx] if len(original_stroke.width_profile) > end_idx else []
        segment_texture = original_stroke.texture_profile[start_idx:end_idx] if len(original_stroke.texture_profile) > end_idx else []
        
        # 计算新的边界框
        if len(segment_skeleton) > 0:
            x_coords = segment_skeleton[:, 0]
            y_coords = segment_skeleton[:, 1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            bbox = (0, 0, 0, 0)
        
        # 计算长度
        length = 0
        for i in range(1, len(segment_skeleton)):
            length += np.linalg.norm(segment_skeleton[i] - segment_skeleton[i-1])
        
        # 计算面积（简化估算）
        area = length * np.mean(segment_width) if segment_width else length
        
        # 计算方向
        if len(segment_skeleton) > 1:
            direction_vector = segment_skeleton[-1] - segment_skeleton[0]
            orientation = np.arctan2(direction_vector[1], direction_vector[0])
        else:
            orientation = 0
        
        # 创建新的笔画对象
        segment_stroke = Stroke(
            id=original_stroke.id * 1000 + start_idx,  # 生成唯一ID
            skeleton=segment_skeleton,
            contour=np.array([]),  # 轮廓需要重新计算
            width_profile=np.array(segment_width),
            texture_profile=np.array(segment_texture),
            bbox=bbox,
            area=area,
            length=length,
            orientation=orientation,
            stroke_type=original_stroke.stroke_type,
            confidence=original_stroke.confidence * 0.8  # 分割后置信度略降
        )
        
        return segment_stroke
    
    def _calculate_segmentation_confidence(self, original_stroke: Stroke, segments: List[Stroke]) -> float:
        """
        计算分割置信度
        
        Args:
            original_stroke (Stroke): 原始笔画
            segments (List[Stroke]): 分割后的笔画段
            
        Returns:
            float: 分割置信度
        """
        if len(segments) <= 1:
            return 1.0
        
        # 基于长度保持性的置信度
        total_segment_length = sum(segment.length for segment in segments)
        length_ratio = total_segment_length / max(original_stroke.length, 1)
        
        # 基于段数的置信度（太多段可能过度分割）
        segment_count_penalty = max(0, 1 - (len(segments) - 2) * 0.1)
        
        # 基于最小段长度的置信度
        min_segment_length = min(segment.length for segment in segments)
        min_length_ratio = min_segment_length / max(original_stroke.length / len(segments), 1)
        
        confidence = length_ratio * segment_count_penalty * min_length_ratio
        return np.clip(confidence, 0.0, 1.0)
    
    def segment_strokes_batch(self, strokes: List[Stroke]) -> List[SegmentationResult]:
        """
        批量分割笔画
        
        Args:
            strokes (List[Stroke]): 笔画列表
            
        Returns:
            List[SegmentationResult]: 分割结果列表
        """
        results = []
        
        for stroke in strokes:
            result = self.segment_stroke(stroke)
            results.append(result)
        
        return results
    
    def get_all_segments(self, segmentation_results: List[SegmentationResult]) -> List[Stroke]:
        """
        从分割结果中提取所有笔画段
        
        Args:
            segmentation_results (List[SegmentationResult]): 分割结果列表
            
        Returns:
            List[Stroke]: 所有笔画段
        """
        all_segments = []
        
        for result in segmentation_results:
            all_segments.extend(result.segments)
        
        return all_segments