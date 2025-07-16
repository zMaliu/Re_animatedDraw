# -*- coding: utf-8 -*-
"""
笔触分割模块

实现笔触的分割和细化算法
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from .stroke_detector import Stroke


class SegmentationMethod(Enum):
    """分割方法枚举"""
    SKELETON_BASED = "skeleton_based"
    CURVATURE_BASED = "curvature_based"
    WATERSHED = "watershed"
    ADAPTIVE = "adaptive"


@dataclass
class StrokeSegment:
    """笔触片段数据结构"""
    id: int
    parent_stroke_id: int
    contour: np.ndarray
    skeleton_points: np.ndarray
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    length: float
    width: float
    curvature: float
    confidence: float
    features: Dict[str, Any]
    area: float = 0.0  # 添加area属性
    perimeter: float = 0.0  # 添加perimeter属性
    center: Tuple[float, float] = (0.0, 0.0)  # 添加center属性
    points: np.ndarray = None  # 添加points属性
    angle: float = 0.0  # 添加angle属性
    
    def __post_init__(self):
        """后处理初始化"""
        if self.features is None:
            self.features = {}
        # 如果area未设置，从轮廓计算
        if self.area == 0.0 and len(self.contour) > 0:
            self.area = cv2.contourArea(self.contour)
        # 如果perimeter未设置，从轮廓计算
        if self.perimeter == 0.0 and len(self.contour) > 0:
            self.perimeter = cv2.arcLength(self.contour, True)
        # 如果center未设置，从轮廓计算质心
        if self.center == (0.0, 0.0) and len(self.contour) > 0:
            M = cv2.moments(self.contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.center = (float(cx), float(cy))
        # 如果points未设置，使用skeleton_points
        if self.points is None:
            self.points = self.skeleton_points if self.skeleton_points is not None else np.array([])
        # 如果angle未设置，从起点和终点计算角度
        if self.angle == 0.0:
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            if dx != 0 or dy != 0:
                import math
                self.angle = math.atan2(dy, dx)


@dataclass
class SegmentationResult:
    """分割结果数据结构"""
    original_stroke: Stroke
    segments: List[StrokeSegment]
    junction_points: List[Tuple[float, float]]
    segmentation_quality: float
    method_used: SegmentationMethod
    processing_time: float


class StrokeSegmenter:
    """笔触分割器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化笔触分割器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.method = SegmentationMethod(self.config.get('method', 'skeleton_based'))
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'method': 'skeleton_based',
            'min_segment_length': 10,
            'max_curvature_threshold': 0.5,
            'skeleton_pruning_threshold': 10,
            'junction_detection_radius': 5,
            'smoothing_kernel_size': 5,
            'curvature_window_size': 7,
            'watershed_markers_distance': 10,
            'adaptive_threshold_factor': 0.8
        }
    
    def segment_stroke(self, stroke: Stroke) -> SegmentationResult:
        """
        分割单个笔触
        
        Args:
            stroke: 输入笔触
            
        Returns:
            分割结果
        """
        import time
        start_time = time.time()
        
        if self.method == SegmentationMethod.SKELETON_BASED:
            segments, junctions = self._segment_by_skeleton(stroke)
        elif self.method == SegmentationMethod.CURVATURE_BASED:
            segments, junctions = self._segment_by_curvature(stroke)
        elif self.method == SegmentationMethod.WATERSHED:
            segments, junctions = self._segment_by_watershed(stroke)
        elif self.method == SegmentationMethod.ADAPTIVE:
            segments, junctions = self._segment_by_adaptive_method(stroke)
        else:
            raise ValueError(f"Unsupported segmentation method: {self.method}")
        
        processing_time = time.time() - start_time
        quality = self._calculate_segmentation_quality(stroke, segments)
        
        return SegmentationResult(
            original_stroke=stroke,
            segments=segments,
            junction_points=junctions,
            segmentation_quality=quality,
            method_used=self.method,
            processing_time=processing_time
        )
    
    def segment_strokes(self, strokes: List[Stroke]) -> List[SegmentationResult]:
        """
        分割多个笔触
        
        Args:
            strokes: 输入笔触列表
            
        Returns:
            分割结果列表
        """
        results = []
        for stroke in strokes:
            try:
                result = self.segment_stroke(stroke)
                results.append(result)
            except Exception as e:
                print(f"Error segmenting stroke {stroke.id}: {e}")
                # 创建空的分割结果
                empty_result = SegmentationResult(
                    original_stroke=stroke,
                    segments=[],
                    junction_points=[],
                    segmentation_quality=0.0,
                    method_used=self.method,
                    processing_time=0.0
                )
                results.append(empty_result)
        
        return results
    
    def _segment_by_skeleton(self, stroke: Stroke) -> Tuple[List[StrokeSegment], List[Tuple[float, float]]]:
        """基于骨架的分割"""
        # 提取骨架
        skeleton = self._extract_skeleton(stroke.mask)
        
        # 骨架修剪
        pruned_skeleton = self._prune_skeleton(skeleton)
        
        # 检测分支点
        junction_points = self._detect_junction_points(pruned_skeleton)
        
        # 基于分支点分割
        segments = self._split_skeleton_at_junctions(stroke, pruned_skeleton, junction_points)
        
        return segments, junction_points
    
    def _segment_by_curvature(self, stroke: Stroke) -> Tuple[List[StrokeSegment], List[Tuple[float, float]]]:
        """基于曲率的分割"""
        # 计算轮廓曲率
        curvatures = self._calculate_contour_curvature(stroke.contour)
        
        # 检测高曲率点
        high_curvature_points = self._detect_high_curvature_points(
            stroke.contour, curvatures
        )
        
        # 基于高曲率点分割
        segments = self._split_contour_at_points(stroke, high_curvature_points)
        
        return segments, high_curvature_points
    
    def _segment_by_watershed(self, stroke: Stroke) -> Tuple[List[StrokeSegment], List[Tuple[float, float]]]:
        """基于分水岭的分割"""
        # 距离变换
        dist_transform = cv2.distanceTransform(stroke.mask, cv2.DIST_L2, 5)
        
        # 查找局部最大值作为标记
        markers = self._find_watershed_markers(dist_transform)
        
        # 分水岭分割
        watershed_result = cv2.watershed(cv2.cvtColor(stroke.mask, cv2.COLOR_GRAY2BGR), markers)
        
        # 提取分割区域
        segments = self._extract_watershed_segments(stroke, watershed_result)
        
        # 检测边界点作为连接点
        junction_points = self._detect_watershed_boundaries(watershed_result)
        
        return segments, junction_points
    
    def _segment_by_adaptive_method(self, stroke: Stroke) -> Tuple[List[StrokeSegment], List[Tuple[float, float]]]:
        """自适应分割方法"""
        # 结合多种方法
        skeleton_segments, skeleton_junctions = self._segment_by_skeleton(stroke)
        curvature_segments, curvature_junctions = self._segment_by_curvature(stroke)
        
        # 选择最佳分割结果
        if len(skeleton_segments) > 0 and len(curvature_segments) > 0:
            # 比较分割质量
            skeleton_quality = self._calculate_segmentation_quality(stroke, skeleton_segments)
            curvature_quality = self._calculate_segmentation_quality(stroke, curvature_segments)
            
            if skeleton_quality > curvature_quality:
                return skeleton_segments, skeleton_junctions
            else:
                return curvature_segments, curvature_junctions
        elif len(skeleton_segments) > 0:
            return skeleton_segments, skeleton_junctions
        elif len(curvature_segments) > 0:
            return curvature_segments, curvature_junctions
        else:
            # 创建单个片段
            single_segment = self._create_single_segment(stroke)
            return [single_segment], []
    
    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """提取骨架"""
        # 使用形态学骨架化
        skeleton = np.zeros(mask.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()
            
            if cv2.countNonZero(mask) == 0:
                break
        
        return skeleton
    
    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """修剪骨架"""
        pruned = skeleton.copy()
        threshold = self.config['skeleton_pruning_threshold']
        
        # 迭代修剪短分支
        changed = True
        while changed:
            changed = False
            
            # 查找端点
            endpoints = self._find_skeleton_endpoints(pruned)
            
            for point in endpoints:
                # 计算从端点开始的分支长度
                branch_length = self._calculate_branch_length(pruned, point)
                
                if branch_length < threshold:
                    # 移除短分支
                    self._remove_branch(pruned, point, branch_length)
                    changed = True
        
        return pruned
    
    def _detect_junction_points(self, skeleton: np.ndarray) -> List[Tuple[float, float]]:
        """检测分支点"""
        junctions = []
        height, width = skeleton.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] > 0:
                    # 检查8邻域
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            if skeleton[y + dy, x + dx] > 0:
                                neighbors.append((dx, dy))
                    
                    # 如果有3个或更多邻居，则为分支点
                    if len(neighbors) >= 3:
                        junctions.append((float(x), float(y)))
        
        return junctions
    
    def _split_skeleton_at_junctions(self, stroke: Stroke, skeleton: np.ndarray, 
                                   junctions: List[Tuple[float, float]]) -> List[StrokeSegment]:
        """在分支点处分割骨架"""
        if not junctions:
            # 没有分支点，创建单个片段
            return [self._create_single_segment(stroke)]
        
        segments = []
        
        # 在分支点处断开骨架
        modified_skeleton = skeleton.copy()
        for jx, jy in junctions:
            cv2.circle(modified_skeleton, (int(jx), int(jy)), 1, 0, -1)
        
        # 查找连通组件
        num_labels, labels = cv2.connectedComponents(modified_skeleton)
        
        segment_id = 0
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # 提取骨架点
            skeleton_points = np.column_stack(np.where(component_mask > 0))
            
            if len(skeleton_points) >= self.config['min_segment_length']:
                segment = self._create_segment_from_skeleton(
                    segment_id, stroke.id, skeleton_points, stroke
                )
                if segment:
                    segments.append(segment)
                    segment_id += 1
        
        return segments
    
    def _calculate_contour_curvature(self, contour: np.ndarray) -> np.ndarray:
        """计算轮廓曲率"""
        # 平滑轮廓
        smoothed = cv2.GaussianBlur(
            contour.astype(np.float32), 
            (self.config['smoothing_kernel_size'], 1), 0
        )
        
        curvatures = []
        window_size = self.config['curvature_window_size']
        
        for i in range(len(smoothed)):
            # 获取窗口内的点
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(smoothed), i + window_size // 2 + 1)
            
            if end_idx - start_idx >= 3:
                points = smoothed[start_idx:end_idx].reshape(-1, 2)
                curvature = self._calculate_point_curvature(points, i - start_idx)
                curvatures.append(curvature)
            else:
                curvatures.append(0.0)
        
        return np.array(curvatures)
    
    def _calculate_point_curvature(self, points: np.ndarray, center_idx: int) -> float:
        """计算点的曲率"""
        if len(points) < 3 or center_idx < 1 or center_idx >= len(points) - 1:
            return 0.0
        
        try:
            # 使用三点法计算曲率
            p1 = points[center_idx - 1]
            p2 = points[center_idx]
            p3 = points[center_idx + 1]
            
            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 计算角度变化
            dot_product = float(np.dot(v1, v2))
            norms = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            
            if norms > 0:
                cos_angle = float(np.clip(dot_product / norms, -1.0, 1.0))
                angle = float(np.arccos(cos_angle))
                return angle
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _detect_high_curvature_points(self, contour: np.ndarray, 
                                     curvatures: np.ndarray) -> List[Tuple[float, float]]:
        """检测高曲率点"""
        threshold = self.config['max_curvature_threshold']
        high_curvature_points = []
        
        for i, curvature in enumerate(curvatures):
            if curvature > threshold:
                point = contour[i][0]  # 轮廓点格式为 [[[x, y]]]
                high_curvature_points.append((float(point[0]), float(point[1])))
        
        return high_curvature_points
    
    def _split_contour_at_points(self, stroke: Stroke, 
                               split_points: List[Tuple[float, float]]) -> List[StrokeSegment]:
        """在指定点处分割轮廓"""
        if not split_points:
            return [self._create_single_segment(stroke)]
        
        # 找到分割点在轮廓上的索引
        split_indices = []
        for point in split_points:
            closest_idx = self._find_closest_contour_point(stroke.contour, point)
            split_indices.append(closest_idx)
        
        split_indices.sort()
        
        # 分割轮廓
        segments = []
        segment_id = 0
        
        for i in range(len(split_indices)):
            start_idx = split_indices[i]
            end_idx = split_indices[(i + 1) % len(split_indices)]
            
            if end_idx > start_idx:
                segment_contour = stroke.contour[start_idx:end_idx]
            else:
                # 跨越轮廓末尾
                segment_contour = np.vstack([
                    stroke.contour[start_idx:],
                    stroke.contour[:end_idx]
                ])
            
            if len(segment_contour) >= self.config['min_segment_length']:
                segment = self._create_segment_from_contour(
                    segment_id, stroke.id, segment_contour
                )
                if segment:
                    segments.append(segment)
                    segment_id += 1
        
        return segments
    
    def _find_closest_contour_point(self, contour: np.ndarray, 
                                   point: Tuple[float, float]) -> int:
        """找到轮廓上最接近指定点的索引"""
        distances = []
        for i, contour_point in enumerate(contour):
            cp = contour_point[0]  # 轮廓点格式
            dist = float(np.sqrt((cp[0] - point[0])**2 + (cp[1] - point[1])**2))
            distances.append(dist)
        
        return int(np.argmin(distances))
    
    def _create_single_segment(self, stroke: Stroke) -> StrokeSegment:
        """创建单个片段（不分割）"""
        # 提取骨架点
        skeleton = self._extract_skeleton(stroke.mask)
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        return StrokeSegment(
            id=0,
            parent_stroke_id=stroke.id,
            contour=stroke.contour,
            skeleton_points=skeleton_points,
            start_point=stroke.center,
            end_point=stroke.center,
            length=stroke.length,
            width=stroke.width,
            curvature=0.0,
            confidence=stroke.confidence,
            features=stroke.features.copy()
        )
    
    def _create_segment_from_skeleton(self, segment_id: int, parent_stroke_id: int,
                                    skeleton_points: np.ndarray, 
                                    original_stroke: Stroke) -> Optional[StrokeSegment]:
        """从骨架点创建片段"""
        if len(skeleton_points) < 2:
            return None
        
        try:
            # 排序骨架点
            ordered_points = self._order_skeleton_points(skeleton_points)
            
            # 计算起始和结束点
            start_point = (float(ordered_points[0][1]), float(ordered_points[0][0]))
            end_point = (float(ordered_points[-1][1]), float(ordered_points[-1][0]))
            
            # 计算长度
            length = self._calculate_skeleton_length(ordered_points)
            
            # 估算宽度
            width = self._estimate_segment_width(skeleton_points, original_stroke.mask)
            
            # 计算曲率
            curvature = self._calculate_skeleton_curvature(ordered_points)
            
            # 创建轮廓（简化版本）
            contour = self._create_contour_from_skeleton(ordered_points, width)
            
            return StrokeSegment(
                id=segment_id,
                parent_stroke_id=parent_stroke_id,
                contour=contour,
                skeleton_points=ordered_points,
                start_point=start_point,
                end_point=end_point,
                length=length,
                width=width,
                curvature=curvature,
                confidence=0.8,  # 默认置信度
                features={}
            )
            
        except Exception as e:
            print(f"Error creating segment from skeleton: {e}")
            return None
    
    def _create_segment_from_contour(self, segment_id: int, parent_stroke_id: int,
                                   contour: np.ndarray) -> Optional[StrokeSegment]:
        """从轮廓创建片段"""
        if len(contour) < 3:
            return None
        
        try:
            # 计算基本属性
            start_point = (float(contour[0][0][0]), float(contour[0][0][1]))
            end_point = (float(contour[-1][0][0]), float(contour[-1][0][1]))
            
            length = cv2.arcLength(contour, False)
            
            # 估算宽度
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                width = min(ellipse[1])
            else:
                width = 5.0  # 默认宽度
            
            # 计算曲率
            curvatures = self._calculate_contour_curvature(contour)
            avg_curvature = np.mean(curvatures) if len(curvatures) > 0 else 0.0
            
            # 创建骨架点（简化）
            skeleton_points = self._approximate_skeleton_from_contour(contour)
            
            return StrokeSegment(
                id=segment_id,
                parent_stroke_id=parent_stroke_id,
                contour=contour,
                skeleton_points=skeleton_points,
                start_point=start_point,
                end_point=end_point,
                length=length,
                width=width,
                curvature=avg_curvature,
                confidence=0.7,  # 默认置信度
                features={}
            )
            
        except Exception as e:
            print(f"Error creating segment from contour: {e}")
            return None
    
    def _calculate_segmentation_quality(self, stroke: Stroke, 
                                      segments: List[StrokeSegment]) -> float:
        """计算分割质量"""
        if not segments:
            return 0.0
        
        try:
            # 基于片段数量的质量评估
            num_segments = len(segments)
            if num_segments == 1:
                return 0.5  # 没有分割
            
            # 基于长度保持的质量评估
            total_segment_length = sum(seg.length for seg in segments)
            length_ratio = min(total_segment_length / stroke.length, 1.0)
            
            # 基于片段均匀性的质量评估
            segment_lengths = [seg.length for seg in segments]
            length_std = np.std(segment_lengths)
            length_mean = np.mean(segment_lengths)
            uniformity = 1.0 - (length_std / (length_mean + 1e-6))
            uniformity = max(0.0, min(1.0, uniformity))
            
            # 综合质量评分
            quality = (length_ratio * 0.6 + uniformity * 0.4)
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
    
    # 辅助方法的简化实现
    def _find_skeleton_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """查找骨架端点"""
        endpoints = []
        height, width = skeleton.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if skeleton[y, x] > 0:
                    neighbor_count = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbor_count == 1:  # 只有一个邻居
                        endpoints.append((x, y))
        
        return endpoints
    
    def _calculate_branch_length(self, skeleton: np.ndarray, 
                               start_point: Tuple[int, int]) -> int:
        """计算分支长度"""
        # 简化实现：沿着骨架追踪直到分支点或端点
        visited = set()
        current = start_point
        length = 0
        
        while current and current not in visited:
            visited.add(current)
            length += 1
            
            # 查找下一个点
            x, y = current
            next_point = None
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0 and (nx, ny) not in visited):
                        next_point = (nx, ny)
                        break
                
                if next_point:
                    break
            
            current = next_point
        
        return length
    
    def _remove_branch(self, skeleton: np.ndarray, start_point: Tuple[int, int], 
                      length: int):
        """移除分支"""
        # 简化实现：从起点开始移除指定长度的像素
        current = start_point
        removed = 0
        
        while current and removed < length:
            x, y = current
            skeleton[y, x] = 0
            removed += 1
            
            # 查找下一个点
            next_point = None
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0):
                        next_point = (nx, ny)
                        break
                
                if next_point:
                    break
            
            current = next_point
    
    def _order_skeleton_points(self, skeleton_points: np.ndarray) -> np.ndarray:
        """排序骨架点"""
        if len(skeleton_points) <= 2:
            return skeleton_points
        
        # 简化实现：基于距离的贪心排序
        ordered = [skeleton_points[0]]
        remaining = list(skeleton_points[1:])
        
        while remaining:
            current = ordered[-1]
            distances = [np.linalg.norm(point - current) for point in remaining]
            closest_idx = int(np.argmin(distances))
            ordered.append(remaining.pop(closest_idx))
        
        return np.array(ordered)
    
    def _calculate_skeleton_length(self, ordered_points: np.ndarray) -> float:
        """计算骨架长度"""
        if len(ordered_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(ordered_points)):
            dist = np.linalg.norm(ordered_points[i] - ordered_points[i-1])
            total_length += dist
        
        return total_length
    
    def _estimate_segment_width(self, skeleton_points: np.ndarray, 
                              mask: np.ndarray) -> float:
        """估算片段宽度"""
        # 简化实现：使用距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        widths = []
        for point in skeleton_points:
            y, x = int(point[0]), int(point[1])
            if 0 <= y < dist_transform.shape[0] and 0 <= x < dist_transform.shape[1]:
                width = dist_transform[y, x] * 2  # 直径
                widths.append(width)
        
        return np.mean(widths) if widths else 5.0
    
    def _calculate_skeleton_curvature(self, ordered_points: np.ndarray) -> float:
        """计算骨架曲率"""
        if len(ordered_points) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(ordered_points) - 1):
            curvature = self._calculate_point_curvature(
                ordered_points[i-1:i+2], 1
            )
            curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _create_contour_from_skeleton(self, skeleton_points: np.ndarray, 
                                    width: float) -> np.ndarray:
        """从骨架创建轮廓"""
        # 简化实现：创建矩形轮廓
        if len(skeleton_points) < 2:
            return np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
        
        # 使用第一个和最后一个点创建简单矩形
        start = skeleton_points[0]
        end = skeleton_points[-1]
        
        # 计算方向向量
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            perpendicular = np.array([-direction[1], direction[0]]) * width / 2
            
            # 创建矩形的四个角点
            p1 = start + perpendicular
            p2 = start - perpendicular
            p3 = end - perpendicular
            p4 = end + perpendicular
            
            contour = np.array([[p1], [p2], [p3], [p4]], dtype=np.int32)
        else:
            # 创建小正方形
            x, y = int(start[1]), int(start[0])
            hw = int(width / 2)
            contour = np.array([[[x-hw, y-hw]], [[x+hw, y-hw]], 
                              [[x+hw, y+hw]], [[x-hw, y+hw]]], dtype=np.int32)
        
        return contour
    
    def _approximate_skeleton_from_contour(self, contour: np.ndarray) -> np.ndarray:
        """从轮廓近似骨架"""
        # 简化实现：使用轮廓的中心线
        if len(contour) < 2:
            return np.array([[0, 0]])
        
        # 计算轮廓的重心作为骨架点
        points = contour.reshape(-1, 2)
        center = np.mean(points, axis=0)
        
        return np.array([[center[1], center[0]]])  # 转换为 (y, x) 格式
    
    # 分水岭相关方法的简化实现
    def _find_watershed_markers(self, dist_transform: np.ndarray) -> np.ndarray:
        """查找分水岭标记"""
        # 简化实现
        markers = np.zeros(dist_transform.shape, dtype=np.int32)
        return markers
    
    def _extract_watershed_segments(self, stroke: Stroke, 
                                  watershed_result: np.ndarray) -> List[StrokeSegment]:
        """提取分水岭分割片段"""
        # 简化实现：返回单个片段
        return [self._create_single_segment(stroke)]
    
    def _detect_watershed_boundaries(self, watershed_result: np.ndarray) -> List[Tuple[float, float]]:
        """检测分水岭边界"""
        # 简化实现
        return []