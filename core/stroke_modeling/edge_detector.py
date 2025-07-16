# -*- coding: utf-8 -*-
"""
Canny边缘检测器

实现论文中要求的Canny边缘检测算法
用于提取笔触的边缘信息
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging


class CannyEdgeDetector:
    """
    Canny边缘检测器
    
    实现论文中的边缘检测功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化边缘检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Canny参数
        self.low_threshold = config.get('canny_low_threshold', 50)
        self.high_threshold = config.get('canny_high_threshold', 150)
        self.aperture_size = config.get('canny_aperture_size', 3)
        self.l2_gradient = config.get('canny_l2_gradient', False)
        
        # 预处理参数
        self.gaussian_kernel_size = config.get('gaussian_kernel_size', 5)
        self.gaussian_sigma = config.get('gaussian_sigma', 1.0)
        
    def detect_edges(self, image: np.ndarray, 
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        检测图像边缘
        
        Args:
            image: 输入图像
            mask: 可选的掩码
            
        Returns:
            np.ndarray: 边缘图像
        """
        try:
            # 预处理
            processed_image = self._preprocess_image(image)
            
            # Canny边缘检测
            edges = cv2.Canny(
                processed_image,
                self.low_threshold,
                self.high_threshold,
                apertureSize=self.aperture_size,
                L2gradient=self.l2_gradient
            )
            
            # 应用掩码
            if mask is not None:
                edges = cv2.bitwise_and(edges, mask)
            
            return edges
            
        except Exception as e:
            self.logger.error(f"Error in edge detection: {str(e)}")
            return np.zeros_like(image[:, :, 0] if len(image.shape) == 3 else image, dtype=np.uint8)
    
    def detect_stroke_edges(self, stroke_mask: np.ndarray, 
                           original_image: np.ndarray) -> Dict[str, Any]:
        """
        检测笔触的边缘信息
        
        Args:
            stroke_mask: 笔触掩码
            original_image: 原始图像
            
        Returns:
            Dict: 边缘检测结果
        """
        try:
            # 在笔触区域内检测边缘
            edges = self.detect_edges(original_image, stroke_mask)
            
            # 查找边缘轮廓
            contours, hierarchy = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 分析边缘特征
            edge_features = self._analyze_edge_features(edges, contours)
            
            return {
                'edges': edges,
                'contours': contours,
                'hierarchy': hierarchy,
                'features': edge_features
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting stroke edges: {str(e)}")
            return {
                'edges': np.zeros_like(stroke_mask),
                'contours': [],
                'hierarchy': None,
                'features': {}
            }
    
    def extract_edge_skeleton(self, edges: np.ndarray) -> np.ndarray:
        """
        从边缘提取骨架
        
        Args:
            edges: 边缘图像
            
        Returns:
            np.ndarray: 骨架图像
        """
        try:
            # 形态学细化
            skeleton = self._morphological_thinning(edges)
            
            # 去除噪声
            skeleton = self._remove_noise(skeleton)
            
            return skeleton
            
        except Exception as e:
            self.logger.error(f"Error extracting edge skeleton: {str(e)}")
            return np.zeros_like(edges)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(
            gray, 
            (self.gaussian_kernel_size, self.gaussian_kernel_size),
            self.gaussian_sigma
        )
        
        return blurred
    
    def _analyze_edge_features(self, edges: np.ndarray, 
                              contours: List[np.ndarray]) -> Dict[str, Any]:
        """
        分析边缘特征
        
        Args:
            edges: 边缘图像
            contours: 轮廓列表
            
        Returns:
            Dict: 边缘特征
        """
        features = {}
        
        # 边缘像素数量
        features['edge_pixel_count'] = int(np.sum(edges > 0))
        
        # 轮廓数量
        features['contour_count'] = len(contours)
        
        if contours:
            # 主轮廓（最大轮廓）
            main_contour = max(contours, key=cv2.contourArea)
            
            # 轮廓长度
            features['main_contour_length'] = float(cv2.arcLength(main_contour, True))
            
            # 轮廓面积
            features['main_contour_area'] = float(cv2.contourArea(main_contour))
            
            # 轮廓复杂度（周长²/面积）
            if features['main_contour_area'] > 0:
                features['contour_complexity'] = (
                    features['main_contour_length'] ** 2 / features['main_contour_area']
                )
            else:
                features['contour_complexity'] = 0.0
            
            # 轮廓凸包
            hull = cv2.convexHull(main_contour)
            features['convex_hull_area'] = float(cv2.contourArea(hull))
            
            # 凸性
            if features['convex_hull_area'] > 0:
                features['convexity'] = features['main_contour_area'] / features['convex_hull_area']
            else:
                features['convexity'] = 0.0
        else:
            features.update({
                'main_contour_length': 0.0,
                'main_contour_area': 0.0,
                'contour_complexity': 0.0,
                'convex_hull_area': 0.0,
                'convexity': 0.0
            })
        
        return features
    
    def _morphological_thinning(self, binary_image: np.ndarray) -> np.ndarray:
        """
        形态学细化
        
        Args:
            binary_image: 二值图像
            
        Returns:
            np.ndarray: 细化后的图像
        """
        # Zhang-Suen细化算法
        skeleton = binary_image.copy()
        skeleton = skeleton // 255  # 确保是0和1
        
        changing1 = changing2 = 1
        while changing1 or changing2:
            # 第一次迭代
            changing1 = self._zhang_suen_iteration(skeleton, 0)
            # 第二次迭代
            changing2 = self._zhang_suen_iteration(skeleton, 1)
        
        return (skeleton * 255).astype(np.uint8)
    
    def _zhang_suen_iteration(self, image: np.ndarray, iteration: int) -> bool:
        """
        Zhang-Suen算法的一次迭代
        
        Args:
            image: 输入图像
            iteration: 迭代类型（0或1）
            
        Returns:
            bool: 是否有像素被改变
        """
        changing = False
        rows, cols = image.shape
        
        # 标记要删除的像素
        to_delete = []
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if image[i, j] == 1:
                    # 获取8邻域
                    p2 = image[i-1, j]
                    p3 = image[i-1, j+1]
                    p4 = image[i, j+1]
                    p5 = image[i+1, j+1]
                    p6 = image[i+1, j]
                    p7 = image[i+1, j-1]
                    p8 = image[i, j-1]
                    p9 = image[i-1, j-1]
                    
                    # 计算邻域特征
                    A = self._count_transitions([p2, p3, p4, p5, p6, p7, p8, p9])
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    
                    # Zhang-Suen条件
                    if (2 <= B <= 6 and A == 1):
                        if iteration == 0:
                            if (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                                to_delete.append((i, j))
                        else:
                            if (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                                to_delete.append((i, j))
        
        # 删除标记的像素
        for i, j in to_delete:
            image[i, j] = 0
            changing = True
        
        return changing
    
    def _count_transitions(self, neighbors: List[int]) -> int:
        """
        计算邻域中0到1的转换次数
        
        Args:
            neighbors: 8邻域像素值列表
            
        Returns:
            int: 转换次数
        """
        transitions = 0
        for i in range(len(neighbors)):
            if neighbors[i] == 0 and neighbors[(i + 1) % len(neighbors)] == 1:
                transitions += 1
        return transitions
    
    def _remove_noise(self, skeleton: np.ndarray) -> np.ndarray:
        """
        去除骨架中的噪声
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            np.ndarray: 去噪后的骨架
        """
        # 形态学开运算去除小的噪声点
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        cleaned = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel)
        
        # 连接断开的线段
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        return cleaned
    
    def get_adaptive_thresholds(self, image: np.ndarray) -> Tuple[float, float]:
        """
        自适应计算Canny阈值
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple[float, float]: (低阈值, 高阈值)
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # 计算图像梯度的统计信息
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 基于梯度统计计算阈值
            median_gradient = np.median(gradient_magnitude)
            low_threshold = 0.5 * median_gradient
            high_threshold = 1.5 * median_gradient
            
            # 确保阈值在合理范围内
            low_threshold = max(10, min(100, low_threshold))
            high_threshold = max(50, min(200, high_threshold))
            
            return float(low_threshold), float(high_threshold)
            
        except Exception as e:
            self.logger.warning(f"Error calculating adaptive thresholds: {str(e)}")
            return float(self.low_threshold), float(self.high_threshold)