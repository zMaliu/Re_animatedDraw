# -*- coding: utf-8 -*-
"""
笔触特征提取器

实现论文中要求的多维度特征提取：
1. 几何特征：骨架点、长度、面积、尺度、形状
2. 墨色特征：湿度、厚度、颜色
3. 位置特征：基于三分法则的显著性得分
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
import math

# 导入新的模块
from ..image_processing.fourier_descriptor import FourierDescriptor
from ..image_processing.saliency_calculator import SaliencyCalculator


@dataclass
class StrokeFeatures:
    """
    笔触特征数据结构
    """
    # 几何特征
    skeleton_points: List[Tuple[float, float]]  # 骨架点
    geodesic_length: float                      # 测地距离长度
    area: float                                 # 面积（有效像素数量）
    scale: float                                # 尺度（最大内接正方形大小）
    circularity: float                          # 圆度（基于傅里叶描述子）
    elongation: float                           # 细长比
    
    # 墨色特征
    wetness: float                              # 湿度（轮廓内有效像素占比）
    thickness: float                            # 厚度（基于像素亮度反转值）
    color_rgb: Tuple[float, float, float]       # RGB平均值
    
    # 位置特征
    saliency_score: float                       # 显著性得分（基于三分法则）
    
    # 辅助信息
    bounding_rect: Tuple[int, int, int, int]    # 边界框
    centroid: Tuple[float, float]               # 质心
    orientation: float                          # 主方向角度


class StrokeFeatureExtractor:
    """
    笔触特征提取器
    
    按照论文要求提取笔触的多维度特征
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 特征提取参数
        self.min_skeleton_points = config.get('min_skeleton_points', 3)
        self.fourier_descriptors = config.get('fourier_descriptors', 10)
        self.saliency_template_size = config.get('saliency_template_size', (3, 3))
        
        # 初始化新的模块
        self.fourier_descriptor = FourierDescriptor()
        self.saliency_calculator = SaliencyCalculator()
        
        # 预计算三分法则模板
        self._rule_of_thirds_template = self._create_rule_of_thirds_template()
    
    def extract_features(self, stroke_mask: np.ndarray, 
                        original_image: np.ndarray,
                        canvas_size: Tuple[int, int]) -> StrokeFeatures:
        """
        提取笔触特征
        
        Args:
            stroke_mask: 笔触掩码
            original_image: 原始图像
            canvas_size: 画布尺寸
            
        Returns:
            StrokeFeatures: 提取的特征
        """
        try:
            # 提取几何特征
            geometric_features = self._extract_geometric_features(stroke_mask)
            
            # 提取墨色特征
            ink_features = self._extract_ink_features(stroke_mask, original_image)
            
            # 提取位置特征
            position_features = self._extract_position_features(stroke_mask, canvas_size)
            
            return StrokeFeatures(
                # 几何特征
                skeleton_points=geometric_features['skeleton_points'],
                geodesic_length=geometric_features['geodesic_length'],
                area=geometric_features['area'],
                scale=geometric_features['scale'],
                circularity=geometric_features['circularity'],
                elongation=geometric_features['elongation'],
                
                # 墨色特征
                wetness=ink_features['wetness'],
                thickness=ink_features['thickness'],
                color_rgb=ink_features['color_rgb'],
                
                # 位置特征
                saliency_score=position_features['saliency_score'],
                
                # 辅助信息
                bounding_rect=geometric_features['bounding_rect'],
                centroid=geometric_features['centroid'],
                orientation=geometric_features['orientation']
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting stroke features: {str(e)}")
            return self._create_default_features()
    
    def _extract_geometric_features(self, stroke_mask: np.ndarray) -> Dict[str, Any]:
        """
        提取几何特征
        
        Args:
            stroke_mask: 笔触掩码
            
        Returns:
            Dict: 几何特征字典
        """
        features = {}
        
        # 查找轮廓
        contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._create_default_geometric_features()
        
        # 选择最大轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 边界框
        x, y, w, h = cv2.boundingRect(main_contour)
        features['bounding_rect'] = (x, y, w, h)
        
        # 质心
        M = cv2.moments(main_contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            features['centroid'] = (cx, cy)
        else:
            features['centroid'] = (x + w/2, y + h/2)
        
        # 面积（有效像素数量）
        features['area'] = float(np.sum(stroke_mask > 0))
        
        # 骨架提取和骨架点
        skeleton = self._extract_skeleton(stroke_mask)
        skeleton_points = self._extract_skeleton_points(skeleton)
        features['skeleton_points'] = skeleton_points
        
        # 测地距离长度
        features['geodesic_length'] = self._calculate_geodesic_length(skeleton_points)
        
        # 尺度（最大内接正方形大小）
        features['scale'] = self._calculate_max_inscribed_square(stroke_mask)
        
        # 主方向
        features['orientation'] = self._calculate_orientation(main_contour)
        
        # 形状特征（基于傅里叶描述子）
        circularity, elongation = self._calculate_shape_descriptors(main_contour)
        features['circularity'] = circularity
        features['elongation'] = elongation
        
        return features
    
    def _extract_ink_features(self, stroke_mask: np.ndarray, 
                             original_image: np.ndarray) -> Dict[str, Any]:
        """
        提取墨色特征
        
        Args:
            stroke_mask: 笔触掩码
            original_image: 原始图像
            
        Returns:
            Dict: 墨色特征字典
        """
        features = {}
        
        # 获取笔触区域的像素
        mask_indices = stroke_mask > 0
        stroke_pixels = original_image[mask_indices]
        
        if len(stroke_pixels) == 0:
            return {'wetness': 0.0, 'thickness': 0.0, 'color_rgb': (0.0, 0.0, 0.0)}
        
        # 湿度：轮廓内有效像素占比
        contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(main_contour)
            effective_pixels = np.sum(stroke_mask > 0)
            features['wetness'] = float(effective_pixels / contour_area) if contour_area > 0 else 0.0
        else:
            features['wetness'] = 0.0
        
        # 厚度：基于像素亮度的反转值
        if len(original_image.shape) == 3:
            # 彩色图像，转换为灰度
            gray_pixels = cv2.cvtColor(stroke_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
        else:
            gray_pixels = stroke_pixels.flatten()
        
        # 厚度 = 1 - 归一化亮度（越暗越厚）
        normalized_brightness = gray_pixels / 255.0
        features['thickness'] = float(1.0 - np.mean(normalized_brightness))
        
        # 颜色：RGB平均值
        if len(original_image.shape) == 3:
            color_mean = np.mean(stroke_pixels, axis=0)
            features['color_rgb'] = tuple(float(c) for c in color_mean)
        else:
            gray_mean = np.mean(stroke_pixels)
            features['color_rgb'] = (float(gray_mean), float(gray_mean), float(gray_mean))
        
        return features
    
    def _extract_position_features(self, stroke_mask: np.ndarray, 
                                  canvas_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        提取位置特征
        
        Args:
            stroke_mask: 笔触掩码
            canvas_size: 画布尺寸
            
        Returns:
            Dict: 位置特征字典
        """
        features = {}
        
        # 基于三分法则的显著性得分
        saliency_score = self._calculate_saliency_score(stroke_mask, canvas_size)
        features['saliency_score'] = saliency_score
        
        return features
    
    def _extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        提取骨架
        
        Args:
            binary_mask: 二值掩码
            
        Returns:
            np.ndarray: 骨架图像
        """
        # 使用形态学骨架化
        skeleton = np.zeros_like(binary_mask, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(binary_mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary_mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary_mask = eroded.copy()
            
            if cv2.countNonZero(binary_mask) == 0:
                break
        
        return skeleton
    
    def _extract_skeleton_points(self, skeleton: np.ndarray) -> List[Tuple[float, float]]:
        """
        从骨架图像中提取关键点
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            List[Tuple[float, float]]: 骨架点列表
        """
        # 查找骨架像素
        skeleton_pixels = np.where(skeleton > 0)
        
        if len(skeleton_pixels[0]) == 0:
            return []
        
        # 转换为点列表
        points = list(zip(skeleton_pixels[1].astype(float), skeleton_pixels[0].astype(float)))
        
        # 如果点太多，进行采样
        if len(points) > 50:
            step = len(points) // 50
            points = points[::step]
        
        return points
    
    def _calculate_geodesic_length(self, skeleton_points: List[Tuple[float, float]]) -> float:
        """
        计算骨架点间的测地距离
        
        Args:
            skeleton_points: 骨架点列表
            
        Returns:
            float: 测地距离长度
        """
        if len(skeleton_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(skeleton_points)):
            p1 = skeleton_points[i-1]
            p2 = skeleton_points[i]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_length += distance
        
        return total_length
    
    def _calculate_max_inscribed_square(self, binary_mask: np.ndarray) -> float:
        """
        计算最大内接正方形大小
        
        Args:
            binary_mask: 二值掩码
            
        Returns:
            float: 最大内接正方形的边长
        """
        # 使用距离变换
        dist_transform = distance_transform_edt(binary_mask)
        
        # 最大值对应最大内接圆的半径
        max_radius = np.max(dist_transform)
        
        # 内接正方形边长约为内接圆直径的 1/√2
        square_side = max_radius * 2 / math.sqrt(2)
        
        return float(square_side)
    
    def _calculate_orientation(self, contour: np.ndarray) -> float:
        """
        计算主方向角度
        
        Args:
            contour: 轮廓点
            
        Returns:
            float: 主方向角度（弧度）
        """
        # 使用PCA计算主方向
        points = contour.reshape(-1, 2).astype(np.float32)
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # 协方差矩阵
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 主方向向量
        main_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        
        # 计算角度
        angle = math.atan2(main_direction[1], main_direction[0])
        
        return float(angle)
    
    def _calculate_shape_descriptors(self, contour: np.ndarray) -> Tuple[float, float]:
        """
        基于傅里叶描述子计算形状特征
        
        Args:
            contour: 轮廓点
            
        Returns:
            Tuple[float, float]: (圆度, 细长比)
        """
        try:
            # 使用傅里叶描述子计算形状特征
            shape_features = self.fourier_descriptor.extract_shape_features(contour)
            
            circularity = shape_features.get('circularity', 0.0)
            elongation = shape_features.get('elongation', 0.0)
            
            return float(circularity), float(elongation)
            
        except Exception as e:
            self.logger.warning(f"Error calculating shape descriptors: {str(e)}")
            # 回退到简单计算
            try:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if area == 0 or perimeter == 0:
                    return 0.0, 0.0
                
                # 圆度 = 4π * 面积 / 周长²
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                
                # 细长比：基于边界框
                x, y, w, h = cv2.boundingRect(contour)
                if min(w, h) == 0:
                    elongation = 0.0
                else:
                    elongation = max(w, h) / min(w, h)
                
                return float(circularity), float(elongation)
            except:
                return 0.0, 0.0
    
    def _calculate_saliency_score(self, stroke_mask: np.ndarray, 
                                 canvas_size: Tuple[int, int]) -> float:
        """
        基于三分法则计算显著性得分
        
        Args:
            stroke_mask: 笔触掩码
            canvas_size: 画布尺寸
            
        Returns:
            float: 显著性得分
        """
        try:
            # 使用显著性计算器
            saliency_score = self.saliency_calculator.calculate_stroke_saliency(
                stroke_mask, canvas_size
            )
            return float(saliency_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating saliency score: {str(e)}")
            # 回退到简单计算
            try:
                canvas_width, canvas_height = canvas_size
                
                # 创建简单的三分法则权重图
                weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
                
                # 三分线位置
                h_lines = [canvas_height // 3, 2 * canvas_height // 3]
                v_lines = [canvas_width // 3, 2 * canvas_width // 3]
                
                # 交点权重最高
                for h_line in h_lines:
                    for v_line in v_lines:
                        cv2.circle(weight_map, (v_line, h_line), 20, 1.0, -1)
                
                # 调整掩码尺寸以匹配权重图
                if not np.array_equal(stroke_mask.shape, weight_map.shape):
                    stroke_mask_resized = cv2.resize(stroke_mask, (canvas_width, canvas_height))
                else:
                    stroke_mask_resized = stroke_mask
                
                # 计算加权得分
                weighted_sum = np.sum(weight_map * (stroke_mask_resized > 0))
                total_pixels = np.sum(stroke_mask_resized > 0)
                
                if total_pixels == 0:
                    return 0.0
                
                return float(weighted_sum / total_pixels)
            except:
                return 0.0
    
    def _create_rule_of_thirds_template(self) -> np.ndarray:
        """
        创建三分法则模板
        
        Returns:
            np.ndarray: 三分法则权重模板
        """
        template_size = self.saliency_template_size
        template = np.zeros(template_size, dtype=np.float32)
        
        # 简单的三分法则模板
        h, w = template_size
        template[h//3, :] = 0.5
        template[2*h//3, :] = 0.5
        template[:, w//3] = 0.5
        template[:, 2*w//3] = 0.5
        template[h//3, w//3] = 1.0
        template[h//3, 2*w//3] = 1.0
        template[2*h//3, w//3] = 1.0
        template[2*h//3, 2*w//3] = 1.0
        
        return template
    
    def _create_default_features(self) -> StrokeFeatures:
        """
        创建默认特征
        
        Returns:
            StrokeFeatures: 默认特征对象
        """
        return StrokeFeatures(
            skeleton_points=[],
            geodesic_length=0.0,
            area=0.0,
            scale=0.0,
            circularity=0.0,
            elongation=0.0,
            wetness=0.0,
            thickness=0.0,
            color_rgb=(0.0, 0.0, 0.0),
            saliency_score=0.0,
            bounding_rect=(0, 0, 0, 0),
            centroid=(0.0, 0.0),
            orientation=0.0
        )
    
    def _create_default_geometric_features(self) -> Dict[str, Any]:
        """
        创建默认几何特征
        
        Returns:
            Dict: 默认几何特征字典
        """
        return {
            'skeleton_points': [],
            'geodesic_length': 0.0,
            'area': 0.0,
            'scale': 0.0,
            'circularity': 0.0,
            'elongation': 0.0,
            'bounding_rect': (0, 0, 0, 0),
            'centroid': (0.0, 0.0),
            'orientation': 0.0
        }