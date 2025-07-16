# -*- coding: utf-8 -*-
"""
笔画检测器

实现从中国画图像中检测和提取笔画的核心算法
基于论文《Animated Construction of Chinese Brush Paintings》的方法
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from skimage import morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt

from .skeleton_extractor import SkeletonExtractor
from .contour_analyzer import ContourAnalyzer


@dataclass
class Stroke:
    """
    笔画数据结构
    
    Attributes:
        id (int): 笔画唯一标识
        skeleton (np.ndarray): 笔画骨架点序列
        contour (np.ndarray): 笔画轮廓点序列
        width_profile (np.ndarray): 沿骨架的宽度变化
        texture_profile (np.ndarray): 沿骨架的纹理变化
        bbox (tuple): 边界框 (x, y, w, h)
        area (float): 笔画面积
        length (float): 笔画长度
        orientation (float): 主方向角度
        stroke_type (str): 笔画类型
        confidence (float): 检测置信度
    """
    id: int
    skeleton: np.ndarray
    contour: np.ndarray
    width_profile: np.ndarray
    texture_profile: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    length: float
    orientation: float
    stroke_type: str = "unknown"
    confidence: float = 0.0


class StrokeDetector:
    """
    笔画检测器
    
    使用多种图像处理技术检测和提取中国画中的笔画
    """
    
    def __init__(self, config):
        """
        初始化笔画检测器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.skeleton_extractor = SkeletonExtractor(config)
        self.contour_analyzer = ContourAnalyzer(config)
        
        # 检测参数
        self.canny_low = config['stroke_detection']['canny_low_threshold']
        self.canny_high = config['stroke_detection']['canny_high_threshold']
        self.min_stroke_length = config['stroke_detection']['min_stroke_length']
        self.max_stroke_width = config['stroke_detection']['max_stroke_width']
        self.morph_kernel_size = config['stroke_detection']['morphology_kernel_size']
        
    def detect_strokes(self, image: np.ndarray) -> List[Stroke]:
        """
        检测图像中的所有笔画
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            List[Stroke]: 检测到的笔画列表
        """
        print("开始笔画检测...")
        
        # 1. 图像预处理
        processed_image = self._preprocess_image(image)
        
        # 2. 边缘检测
        edges = self._detect_edges(processed_image)
        
        # 3. 连通组件分析
        components = self._find_connected_components(edges)
        
        # 4. 笔画候选区域提取
        stroke_candidates = self._extract_stroke_candidates(components, processed_image)
        
        # 5. 笔画验证和特征提取
        strokes = self._validate_and_extract_features(stroke_candidates, processed_image)
        
        # 6. 笔画分类
        classified_strokes = self._classify_strokes(strokes)
        
        print(f"检测完成，共找到 {len(classified_strokes)} 个有效笔画")
        return classified_strokes
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image (np.ndarray): 原始图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊去噪
        kernel_size = self.config['image']['gaussian_blur_kernel']
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # 双边滤波保持边缘
        bilateral = cv2.bilateralFilter(
            blurred,
            self.config['image']['bilateral_filter_d'],
            self.config['image']['bilateral_filter_sigma_color'],
            self.config['image']['bilateral_filter_sigma_space']
        )
        
        # 直方图均衡化增强对比度
        equalized = cv2.equalizeHist(bilateral)
        
        return equalized
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘检测
        
        Args:
            image (np.ndarray): 预处理后的图像
            
        Returns:
            np.ndarray: 边缘图像
        """
        # Canny边缘检测
        edges = cv2.Canny(image, self.canny_low, self.canny_high)
        
        # 形态学操作连接断裂的边缘
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _find_connected_components(self, edges: np.ndarray) -> List[np.ndarray]:
        """
        连通组件分析
        
        Args:
            edges (np.ndarray): 边缘图像
            
        Returns:
            List[np.ndarray]: 连通组件列表
        """
        # 标记连通组件
        num_labels, labels = cv2.connectedComponents(edges)
        
        components = []
        for label in range(1, num_labels):  # 跳过背景(标签0)
            component_mask = (labels == label).astype(np.uint8) * 255
            
            # 过滤太小的组件
            area = np.sum(component_mask > 0)
            if area >= self.min_stroke_length:
                components.append(component_mask)
        
        return components
    
    def _extract_stroke_candidates(self, components: List[np.ndarray], 
                                 original_image: np.ndarray) -> List[Dict]:
        """
        提取笔画候选区域
        
        Args:
            components (List[np.ndarray]): 连通组件列表
            original_image (np.ndarray): 原始图像
            
        Returns:
            List[Dict]: 笔画候选区域信息
        """
        candidates = []
        
        for i, component in enumerate(components):
            # 计算边界框
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
                
            # 选择最大的轮廓
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 过滤不合理的候选区域
            if w < 5 or h < 5 or max(w, h) > self.max_stroke_width * 10:
                continue
            
            # 提取候选区域
            roi = component[y:y+h, x:x+w]
            original_roi = original_image[y:y+h, x:x+w]
            
            candidate = {
                'id': i,
                'bbox': (x, y, w, h),
                'mask': roi,
                'original_roi': original_roi,
                'contour': main_contour,
                'area': cv2.contourArea(main_contour)
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _validate_and_extract_features(self, candidates: List[Dict], 
                                     original_image: np.ndarray) -> List[Stroke]:
        """
        验证候选区域并提取特征
        
        Args:
            candidates (List[Dict]): 笔画候选区域
            original_image (np.ndarray): 原始图像
            
        Returns:
            List[Stroke]: 验证后的笔画列表
        """
        strokes = []
        
        for candidate in candidates:
            try:
                # 提取骨架
                skeleton = self.skeleton_extractor.extract_skeleton(candidate['mask'])
                if skeleton is None or len(skeleton) < self.min_stroke_length:
                    continue
                
                # 分析轮廓
                contour_info = self.contour_analyzer.analyze_contour(
                    candidate['contour'], candidate['original_roi']
                )
                
                # 计算宽度变化
                width_profile = self._compute_width_profile(
                    skeleton, candidate['mask']
                )
                
                # 计算纹理变化
                texture_profile = self._compute_texture_profile(
                    skeleton, candidate['original_roi']
                )
                
                # 计算几何特征
                length = self._compute_stroke_length(skeleton)
                orientation = self._compute_orientation(skeleton)
                
                # 创建笔画对象
                stroke = Stroke(
                    id=candidate['id'],
                    skeleton=skeleton,
                    contour=candidate['contour'],
                    width_profile=width_profile,
                    texture_profile=texture_profile,
                    bbox=candidate['bbox'],
                    area=candidate['area'],
                    length=length,
                    orientation=orientation,
                    confidence=self._compute_confidence(candidate, skeleton)
                )
                
                strokes.append(stroke)
                
            except Exception as e:
                print(f"处理候选笔画 {candidate['id']} 时发生错误: {str(e)}")
                continue
        
        return strokes
    
    def _compute_width_profile(self, skeleton: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        计算沿骨架的宽度变化
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            mask (np.ndarray): 笔画掩码
            
        Returns:
            np.ndarray: 宽度变化曲线
        """
        width_profile = []
        
        for i in range(len(skeleton)):
            point = skeleton[i]
            
            # 计算垂直于骨架方向的宽度
            if i == 0:
                direction = skeleton[i+1] - skeleton[i]
            elif i == len(skeleton) - 1:
                direction = skeleton[i] - skeleton[i-1]
            else:
                direction = skeleton[i+1] - skeleton[i-1]
            
            # 垂直方向
            perpendicular = np.array([-direction[1], direction[0]])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            # 沿垂直方向测量宽度
            width = self._measure_width_at_point(point, perpendicular, mask)
            width_profile.append(width)
        
        return np.array(width_profile)
    
    def _measure_width_at_point(self, point: np.ndarray, direction: np.ndarray, 
                               mask: np.ndarray) -> float:
        """
        在指定点沿指定方向测量宽度
        
        Args:
            point (np.ndarray): 测量点
            direction (np.ndarray): 测量方向
            mask (np.ndarray): 笔画掩码
            
        Returns:
            float: 测量的宽度
        """
        max_distance = min(mask.shape) // 2
        width = 0
        
        # 向两个方向搜索边界
        for sign in [-1, 1]:
            for distance in range(1, max_distance):
                test_point = point + sign * distance * direction
                x, y = int(test_point[0]), int(test_point[1])
                
                if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                    if mask[y, x] == 0:  # 到达边界
                        width += distance
                        break
                else:
                    width += distance
                    break
        
        return width
    
    def _compute_texture_profile(self, skeleton: np.ndarray, 
                               original_roi: np.ndarray) -> np.ndarray:
        """
        计算沿骨架的纹理变化
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            original_roi (np.ndarray): 原始图像区域
            
        Returns:
            np.ndarray: 纹理变化曲线
        """
        texture_profile = []
        window_size = 5
        
        for point in skeleton:
            x, y = int(point[0]), int(point[1])
            
            # 提取局部窗口
            x1 = max(0, x - window_size // 2)
            y1 = max(0, y - window_size // 2)
            x2 = min(original_roi.shape[1], x + window_size // 2 + 1)
            y2 = min(original_roi.shape[0], y + window_size // 2 + 1)
            
            window = original_roi[y1:y2, x1:x2]
            
            if window.size > 0:
                # 计算局部纹理特征（标准差）
                texture_value = np.std(window.astype(np.float32))
            else:
                texture_value = 0
            
            texture_profile.append(texture_value)
        
        return np.array(texture_profile)
    
    def _compute_stroke_length(self, skeleton: np.ndarray) -> float:
        """
        计算笔画长度
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            
        Returns:
            float: 笔画长度
        """
        if len(skeleton) < 2:
            return 0
        
        total_length = 0
        for i in range(1, len(skeleton)):
            distance = np.linalg.norm(skeleton[i] - skeleton[i-1])
            total_length += distance
        
        return total_length
    
    def _compute_orientation(self, skeleton: np.ndarray) -> float:
        """
        计算笔画主方向
        
        Args:
            skeleton (np.ndarray): 骨架点序列
            
        Returns:
            float: 主方向角度（弧度）
        """
        if len(skeleton) < 2:
            return 0
        
        # 使用起点和终点计算主方向
        start_point = skeleton[0]
        end_point = skeleton[-1]
        direction = end_point - start_point
        
        angle = np.arctan2(direction[1], direction[0])
        return angle
    
    def _compute_confidence(self, candidate: Dict, skeleton: np.ndarray) -> float:
        """
        计算笔画检测置信度
        
        Args:
            candidate (Dict): 候选笔画信息
            skeleton (np.ndarray): 骨架点序列
            
        Returns:
            float: 置信度分数 (0-1)
        """
        # 基于多个因素计算置信度
        factors = []
        
        # 1. 长度因子
        length = len(skeleton)
        length_factor = min(1.0, length / (self.min_stroke_length * 2))
        factors.append(length_factor)
        
        # 2. 面积因子
        area = candidate['area']
        area_factor = min(1.0, area / 100)  # 假设合理面积为100像素
        factors.append(area_factor)
        
        # 3. 形状因子（长宽比）
        x, y, w, h = candidate['bbox']
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        shape_factor = min(1.0, aspect_ratio / 10)  # 笔画通常是细长的
        factors.append(shape_factor)
        
        # 计算综合置信度
        confidence = np.mean(factors)
        return confidence
    
    def _classify_strokes(self, strokes: List[Stroke]) -> List[Stroke]:
        """
        对笔画进行分类
        
        Args:
            strokes (List[Stroke]): 输入笔画列表
            
        Returns:
            List[Stroke]: 分类后的笔画列表
        """
        for stroke in strokes:
            # 基于几何特征进行简单分类
            stroke.stroke_type = self._classify_single_stroke(stroke)
        
        return strokes
    
    def _classify_single_stroke(self, stroke: Stroke) -> str:
        """
        对单个笔画进行分类
        
        Args:
            stroke (Stroke): 输入笔画
            
        Returns:
            str: 笔画类型
        """
        # 简单的基于角度和长度的分类
        angle = abs(stroke.orientation)
        length = stroke.length
        x, y, w, h = stroke.bbox
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        # 基于角度判断方向
        if angle < np.pi / 8 or angle > 7 * np.pi / 8:  # 接近水平
            if aspect_ratio > 3:
                return "horizontal"
        elif np.pi * 3 / 8 < angle < np.pi * 5 / 8:  # 接近垂直
            if aspect_ratio > 3:
                return "vertical"
        
        # 基于长度判断类型
        if length < 20:
            return "dot"
        elif length > 100:
            return "long_stroke"
        else:
            return "medium_stroke"
    
    def visualize_detection_results(self, image: np.ndarray, strokes: List[Stroke]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image (np.ndarray): 原始图像
            strokes (List[Stroke]): 检测到的笔画
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        
        for i, stroke in enumerate(strokes):
            color = colors[i % len(colors)]
            
            # 绘制边界框
            x, y, w, h = stroke.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # 绘制骨架
            if len(stroke.skeleton) > 1:
                for j in range(1, len(stroke.skeleton)):
                    pt1 = tuple(stroke.skeleton[j-1].astype(int))
                    pt2 = tuple(stroke.skeleton[j].astype(int))
                    cv2.line(vis_image, pt1, pt2, color, 2)
            
            # 添加标签
            cv2.putText(vis_image, f"{i}:{stroke.stroke_type}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image