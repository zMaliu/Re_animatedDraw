# -*- coding: utf-8 -*-
"""
笔触检测模块

实现基于区域增长和自适应阈值的笔触检测算法
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class DetectionMethod(Enum):
    """检测方法枚举"""
    REGION_GROWING = "region_growing"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    CONTOUR_BASED = "contour_based"
    HYBRID = "hybrid"


@dataclass
class Stroke:
    """笔触数据结构"""
    id: int
    contour: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]
    area: float
    perimeter: float
    length: float
    width: float
    angle: float
    confidence: float
    features: Dict[str, Any]
    
    def __post_init__(self):
        """后处理初始化"""
        if self.features is None:
            self.features = {}


class StrokeDetector:
    """笔触检测器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化笔触检测器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.method = DetectionMethod(self.config.get('method', 'region_growing'))
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'method': 'region_growing',
            'adaptive_block_size': 25,
            'adaptive_c': 10,
            'region_growing_threshold': 2.0,
            'min_region_size': 100,
            'min_stroke_length': 10,
            'max_stroke_width': 50,
            'gaussian_blur_kernel': 5,
            'morphology_kernel_size': 3,
            'contour_min_area': 50,
            'contour_approximation_epsilon': 0.02
        }
    
    def detect_strokes(self, image: np.ndarray) -> List[Stroke]:
        """
        检测图像中的笔触
        
        Args:
            image: 输入图像
            
        Returns:
            检测到的笔触列表
        """
        if self.method == DetectionMethod.REGION_GROWING:
            return self._detect_by_region_growing(image)
        elif self.method == DetectionMethod.ADAPTIVE_THRESHOLD:
            return self._detect_by_adaptive_threshold(image)
        elif self.method == DetectionMethod.CONTOUR_BASED:
            return self._detect_by_contours(image)
        elif self.method == DetectionMethod.HYBRID:
            return self._detect_by_hybrid_method(image)
        else:
            raise ValueError(f"Unsupported detection method: {self.method}")
    
    def _detect_by_region_growing(self, image: np.ndarray) -> List[Stroke]:
        """基于区域增长的笔触检测"""
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (self.config['gaussian_blur_kernel'], 
                                         self.config['gaussian_blur_kernel']), 0)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.config['adaptive_block_size'], self.config['adaptive_c']
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.config['morphology_kernel_size'],
                                          self.config['morphology_kernel_size']))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 区域增长
        labeled_image = self._region_growing(binary)
        
        # 提取笔触
        return self._extract_strokes_from_labeled_image(labeled_image, binary)
    
    def _detect_by_adaptive_threshold(self, image: np.ndarray) -> List[Stroke]:
        """基于自适应阈值的笔触检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            self.config['adaptive_block_size'], self.config['adaptive_c']
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._contours_to_strokes(contours, binary)
    
    def _detect_by_contours(self, image: np.ndarray) -> List[Stroke]:
        """基于轮廓的笔触检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._contours_to_strokes(contours, edges)
    
    def _detect_by_hybrid_method(self, image: np.ndarray) -> List[Stroke]:
        """混合方法检测"""
        # 结合多种方法的结果
        strokes_rg = self._detect_by_region_growing(image)
        strokes_at = self._detect_by_adaptive_threshold(image)
        strokes_ct = self._detect_by_contours(image)
        
        # 合并和去重
        all_strokes = strokes_rg + strokes_at + strokes_ct
        return self._merge_duplicate_strokes(all_strokes)
    
    def _region_growing(self, binary_image: np.ndarray) -> np.ndarray:
        """区域增长算法"""
        height, width = binary_image.shape
        labeled = np.zeros((height, width), dtype=np.int32)
        label = 1
        
        for y in range(height):
            for x in range(width):
                if binary_image[y, x] > 0 and labeled[y, x] == 0:
                    # 开始新的区域增长
                    region_size = self._grow_region(binary_image, labeled, x, y, label)
                    
                    if region_size >= self.config['min_region_size']:
                        label += 1
                    else:
                        # 移除太小的区域
                        label_indices = labeled == label
                        labeled[label_indices] = 0
        
        return labeled
    
    def _grow_region(self, binary_image: np.ndarray, labeled: np.ndarray, 
                    start_x: int, start_y: int, label: int) -> int:
        """从种子点开始增长区域"""
        height, width = binary_image.shape
        stack = [(start_x, start_y)]
        region_size = 0
        threshold = self.config['region_growing_threshold']
        
        while stack:
            x, y = stack.pop()
            
            if (x < 0 or x >= width or y < 0 or y >= height or 
                labeled[y, x] != 0 or binary_image[y, x] == 0):
                continue
            
            labeled[y, x] = label
            region_size += 1
            
            # 检查8邻域
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and 
                        labeled[ny, nx] == 0 and binary_image[ny, nx] > 0):
                        
                        # 检查相似性
                        if abs(int(binary_image[y, x]) - int(binary_image[ny, nx])) <= threshold:
                            stack.append((nx, ny))
        
        return region_size
    
    def _extract_strokes_from_labeled_image(self, labeled_image: np.ndarray, 
                                           binary_image: np.ndarray) -> List[Stroke]:
        """从标记图像中提取笔触"""
        strokes = []
        unique_labels = np.unique(labeled_image)
        
        stroke_id = 0
        for label in unique_labels:
            if label == 0:  # 跳过背景
                continue
            
            # 创建掩码
            mask = (labeled_image == label).astype(np.uint8) * 255
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 选择最大的轮廓
                contour = max(contours, key=cv2.contourArea)
                
                # 创建笔触对象
                stroke = self._create_stroke_from_contour(stroke_id, contour, mask)
                if stroke:
                    strokes.append(stroke)
                    stroke_id += 1
        
        return strokes
    
    def _contours_to_strokes(self, contours: List[np.ndarray], 
                            binary_image: np.ndarray) -> List[Stroke]:
        """将轮廓转换为笔触"""
        strokes = []
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) >= self.config['contour_min_area']:
                # 创建掩码
                mask = np.zeros(binary_image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # 创建笔触对象
                stroke = self._create_stroke_from_contour(i, contour, mask)
                if stroke:
                    strokes.append(stroke)
        
        return strokes
    
    def _create_stroke_from_contour(self, stroke_id: int, contour: np.ndarray, 
                                   mask: np.ndarray) -> Optional[Stroke]:
        """从轮廓创建笔触对象"""
        try:
            # 基本几何属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < self.config['contour_min_area']:
                return None
            
            # 边界框
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            
            # 中心点
            M = cv2.moments(contour)
            if M['m00'] != 0:
                center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
            else:
                center = (x + w/2, y + h/2)
            
            # 拟合椭圆获取长度、宽度和角度
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                length = max(ellipse[1])
                width = min(ellipse[1])
                angle = ellipse[2]
            else:
                length = max(w, h)
                width = min(w, h)
                angle = 0
            
            # 检查笔触尺寸
            if (length < self.config['min_stroke_length'] or 
                width > self.config['max_stroke_width']):
                return None
            
            # 计算置信度
            confidence = self._calculate_stroke_confidence(contour, area, perimeter)
            
            # 提取特征
            features = self._extract_stroke_features(contour, mask)
            
            return Stroke(
                id=stroke_id,
                contour=contour,
                mask=mask,
                bbox=bbox,
                center=center,
                area=area,
                perimeter=perimeter,
                length=length,
                width=width,
                angle=angle,
                confidence=confidence,
                features=features
            )
            
        except Exception as e:
            print(f"Error creating stroke from contour: {e}")
            return None
    
    def _calculate_stroke_confidence(self, contour: np.ndarray, 
                                   area: float, perimeter: float) -> float:
        """计算笔触置信度"""
        try:
            # 基于形状规律性的置信度
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            
            # 基于长宽比的置信度
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                aspect_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                aspect_confidence = min(aspect_ratio / 10.0, 1.0)  # 偏好细长形状
            else:
                aspect_confidence = 0.5
            
            # 综合置信度
            confidence = (solidity * 0.5 + aspect_confidence * 0.5)
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    def _extract_stroke_features(self, contour: np.ndarray, 
                               mask: np.ndarray) -> Dict[str, Any]:
        """提取笔触特征"""
        features = {}
        
        try:
            # 形状特征
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
            else:
                features['circularity'] = 0
            
            # 凸包特征
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                features['solidity'] = area / hull_area
            else:
                features['solidity'] = 0
            
            # 椭圆拟合特征
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                features['ellipse_center'] = ellipse[0]
                features['ellipse_axes'] = ellipse[1]
                features['ellipse_angle'] = ellipse[2]
                features['aspect_ratio'] = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
            
            # 边界框特征
            x, y, w, h = cv2.boundingRect(contour)
            features['bbox_aspect_ratio'] = w / (h + 1e-6)
            features['extent'] = area / (w * h)
            
        except Exception as e:
            print(f"Error extracting stroke features: {e}")
        
        return features
    
    def _merge_duplicate_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """合并重复的笔触"""
        if not strokes:
            return strokes
        
        merged_strokes = []
        used_indices = set()
        
        for i, stroke1 in enumerate(strokes):
            if i in used_indices:
                continue
            
            # 查找重叠的笔触
            overlapping_strokes = [stroke1]
            used_indices.add(i)
            
            for j, stroke2 in enumerate(strokes[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._strokes_overlap(stroke1, stroke2):
                    overlapping_strokes.append(stroke2)
                    used_indices.add(j)
            
            # 合并重叠的笔触
            if len(overlapping_strokes) > 1:
                merged_stroke = self._merge_strokes(overlapping_strokes)
                if merged_stroke:
                    merged_strokes.append(merged_stroke)
            else:
                merged_strokes.append(stroke1)
        
        return merged_strokes
    
    def _strokes_overlap(self, stroke1: Stroke, stroke2: Stroke, 
                        threshold: float = 0.3) -> bool:
        """检查两个笔触是否重叠"""
        # 检查边界框重叠
        x1, y1, w1, h1 = stroke1.bbox
        x2, y2, w2, h2 = stroke2.bbox
        
        # 计算重叠区域
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        # 计算重叠比例
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)
        
        if min_area > 0:
            overlap_ratio = overlap_area / min_area
            return overlap_ratio > threshold
        
        return False
    
    def _merge_strokes(self, strokes: List[Stroke]) -> Optional[Stroke]:
        """合并多个笔触"""
        if not strokes:
            return None
        
        if len(strokes) == 1:
            return strokes[0]
        
        try:
            # 合并轮廓
            all_points = np.vstack([stroke.contour for stroke in strokes])
            merged_contour = cv2.convexHull(all_points)
            
            # 创建合并的掩码
            mask_shape = strokes[0].mask.shape
            merged_mask = np.zeros(mask_shape, dtype=np.uint8)
            
            for stroke in strokes:
                merged_mask = cv2.bitwise_or(merged_mask, stroke.mask)
            
            # 创建新的笔触对象
            return self._create_stroke_from_contour(
                strokes[0].id, merged_contour, merged_mask
            )
            
        except Exception as e:
            print(f"Error merging strokes: {e}")
            return strokes[0]  # 返回第一个笔触作为备选