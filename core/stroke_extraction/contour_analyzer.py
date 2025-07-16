# -*- coding: utf-8 -*-
"""
轮廓分析器

分析笔画轮廓的几何特征和形状属性
提供轮廓简化、特征提取和形状描述功能
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ContourFeatures:
    """
    轮廓特征数据结构
    
    Attributes:
        area (float): 轮廓面积
        perimeter (float): 轮廓周长
        centroid (Tuple[float, float]): 质心坐标
        bounding_rect (Tuple[int, int, int, int]): 边界矩形
        convex_hull (np.ndarray): 凸包
        convexity (float): 凸性度量
        aspect_ratio (float): 长宽比
        extent (float): 范围（面积/边界矩形面积）
        solidity (float): 实心度（面积/凸包面积）
        orientation (float): 主方向角度
        major_axis_length (float): 长轴长度
        minor_axis_length (float): 短轴长度
        eccentricity (float): 离心率
        circularity (float): 圆形度
        rectangularity (float): 矩形度
    """
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    bounding_rect: Tuple[int, int, int, int]
    convex_hull: np.ndarray
    convexity: float
    aspect_ratio: float
    extent: float
    solidity: float
    orientation: float
    major_axis_length: float
    minor_axis_length: float
    eccentricity: float
    circularity: float
    rectangularity: float


class ContourAnalyzer:
    """
    轮廓分析器
    
    提供笔画轮廓的几何分析和特征提取功能
    """
    
    def __init__(self, config):
        """
        初始化轮廓分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.approximation_epsilon = config['stroke_detection'].get(
            'contour_approximation_epsilon', 0.02
        )
    
    def analyze_contour(self, contour: np.ndarray, 
                       original_roi: Optional[np.ndarray] = None) -> ContourFeatures:
        """
        分析轮廓特征
        
        Args:
            contour (np.ndarray): 输入轮廓
            original_roi (np.ndarray, optional): 原始图像区域
            
        Returns:
            ContourFeatures: 轮廓特征
        """
        # 基本几何特征
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 质心
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            centroid = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        else:
            centroid = (0, 0)
        
        # 边界矩形
        bounding_rect = cv2.boundingRect(contour)
        x, y, w, h = bounding_rect
        
        # 凸包
        convex_hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(convex_hull)
        
        # 计算各种特征
        convexity = self._compute_convexity(contour, convex_hull)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        extent = area / (w * h) if w * h > 0 else 0
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 椭圆拟合特征
        if len(contour) >= 5:  # 椭圆拟合至少需要5个点
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            orientation = math.radians(angle)
            major_axis_length = max(major_axis, minor_axis)
            minor_axis_length = min(major_axis, minor_axis)
            
            # 离心率
            if major_axis_length > 0:
                eccentricity = math.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
            else:
                eccentricity = 0
        else:
            orientation = 0
            major_axis_length = max(w, h)
            minor_axis_length = min(w, h)
            eccentricity = 0
        
        # 形状度量
        circularity = self._compute_circularity(area, perimeter)
        rectangularity = self._compute_rectangularity(area, w, h)
        
        return ContourFeatures(
            area=area,
            perimeter=perimeter,
            centroid=centroid,
            bounding_rect=bounding_rect,
            convex_hull=convex_hull,
            convexity=convexity,
            aspect_ratio=aspect_ratio,
            extent=extent,
            solidity=solidity,
            orientation=orientation,
            major_axis_length=major_axis_length,
            minor_axis_length=minor_axis_length,
            eccentricity=eccentricity,
            circularity=circularity,
            rectangularity=rectangularity
        )
    
    def simplify_contour(self, contour: np.ndarray, epsilon_factor: float = None) -> np.ndarray:
        """
        简化轮廓
        
        Args:
            contour (np.ndarray): 输入轮廓
            epsilon_factor (float, optional): 简化因子
            
        Returns:
            np.ndarray: 简化后的轮廓
        """
        if epsilon_factor is None:
            epsilon_factor = self.approximation_epsilon
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        
        # Douglas-Peucker算法简化轮廓
        epsilon = epsilon_factor * perimeter
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        return simplified_contour
    
    def extract_curvature(self, contour: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        提取轮廓曲率
        
        Args:
            contour (np.ndarray): 输入轮廓
            window_size (int): 计算窗口大小
            
        Returns:
            np.ndarray: 曲率数组
        """
        if len(contour) < window_size * 2 + 1:
            return np.zeros(len(contour))
        
        # 将轮廓转换为点序列
        points = contour.reshape(-1, 2)
        curvatures = []
        
        for i in range(len(points)):
            # 获取邻域点
            prev_idx = (i - window_size) % len(points)
            next_idx = (i + window_size) % len(points)
            
            prev_point = points[prev_idx]
            curr_point = points[i]
            next_point = points[next_idx]
            
            # 计算曲率
            curvature = self._compute_curvature_at_point(
                prev_point, curr_point, next_point
            )
            curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def find_corner_points(self, contour: np.ndarray, 
                          curvature_threshold: float = 0.1) -> List[int]:
        """
        查找轮廓的角点
        
        Args:
            contour (np.ndarray): 输入轮廓
            curvature_threshold (float): 曲率阈值
            
        Returns:
            List[int]: 角点索引列表
        """
        # 提取曲率
        curvatures = self.extract_curvature(contour)
        
        # 查找曲率峰值
        corner_indices = []
        for i in range(1, len(curvatures) - 1):
            # 安全获取曲率值，避免数组真值歧义
            from utils.math_utils import ensure_scalar
            
            # 使用通用的标量转换函数
            curr_val = ensure_scalar(curvatures[i])
            prev_val = ensure_scalar(curvatures[i-1])
            next_val = ensure_scalar(curvatures[i+1])
            
            if (curr_val > curvature_threshold and
                curr_val > prev_val and
                curr_val > next_val):
                corner_indices.append(i)
        
        return corner_indices
    
    def compute_shape_context(self, contour: np.ndarray, 
                             num_points: int = 100) -> np.ndarray:
        """
        计算形状上下文描述符
        
        Args:
            contour (np.ndarray): 输入轮廓
            num_points (int): 采样点数
            
        Returns:
            np.ndarray: 形状上下文描述符
        """
        # 重采样轮廓到固定点数
        resampled_contour = self._resample_contour(contour, num_points)
        points = resampled_contour.reshape(-1, 2)
        
        # 计算每个点的形状上下文
        shape_contexts = []
        
        for i, point in enumerate(points):
            # 计算相对于当前点的其他点的极坐标
            relative_points = points - point
            
            # 转换为极坐标
            distances = np.linalg.norm(relative_points, axis=1)
            angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
            
            # 创建直方图
            hist = self._create_shape_context_histogram(distances, angles)
            shape_contexts.append(hist)
        
        return np.array(shape_contexts)
    
    def _compute_convexity(self, contour: np.ndarray, convex_hull: np.ndarray) -> float:
        """
        计算凸性度量
        
        Args:
            contour (np.ndarray): 原始轮廓
            convex_hull (np.ndarray): 凸包
            
        Returns:
            float: 凸性度量
        """
        hull_perimeter = cv2.arcLength(convex_hull, True)
        contour_perimeter = cv2.arcLength(contour, True)
        
        if hull_perimeter > 0:
            return contour_perimeter / hull_perimeter
        else:
            return 0
    
    def _compute_circularity(self, area: float, perimeter: float) -> float:
        """
        计算圆形度
        
        Args:
            area (float): 面积
            perimeter (float): 周长
            
        Returns:
            float: 圆形度
        """
        if perimeter > 0:
            return 4 * math.pi * area / (perimeter ** 2)
        else:
            return 0
    
    def _compute_rectangularity(self, area: float, width: float, height: float) -> float:
        """
        计算矩形度
        
        Args:
            area (float): 面积
            width (float): 宽度
            height (float): 高度
            
        Returns:
            float: 矩形度
        """
        rect_area = width * height
        if rect_area > 0:
            return area / rect_area
        else:
            return 0
    
    def _compute_curvature_at_point(self, prev_point: np.ndarray, 
                                   curr_point: np.ndarray, 
                                   next_point: np.ndarray) -> float:
        """
        计算指定点的曲率
        
        Args:
            prev_point (np.ndarray): 前一个点
            curr_point (np.ndarray): 当前点
            next_point (np.ndarray): 下一个点
            
        Returns:
            float: 曲率值
        """
        # 计算向量
        v1 = curr_point - prev_point
        v2 = next_point - curr_point
        
        # 计算向量长度
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        
        if len1 == 0 or len2 == 0:
            return 0
        
        # 归一化向量
        v1_norm = v1 / len1
        v2_norm = v2 / len2
        
        # 计算角度变化
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1, 1)  # 防止数值误差
        angle_change = math.acos(dot_product)
        
        # 计算曲率（角度变化除以弧长）
        arc_length = (len1 + len2) / 2
        if arc_length > 0:
            curvature = angle_change / arc_length
        else:
            curvature = 0
        
        return curvature
    
    def _resample_contour(self, contour: np.ndarray, num_points: int) -> np.ndarray:
        """
        重采样轮廓到指定点数
        
        Args:
            contour (np.ndarray): 输入轮廓
            num_points (int): 目标点数
            
        Returns:
            np.ndarray: 重采样后的轮廓
        """
        points = contour.reshape(-1, 2)
        
        # 计算累积弧长
        distances = [0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            distances.append(distances[-1] + dist)
        
        total_length = distances[-1]
        if total_length == 0:
            return contour
        
        # 等间距采样
        sample_distances = np.linspace(0, total_length, num_points)
        resampled_points = []
        
        for sample_dist in sample_distances:
            # 找到对应的线段
            for i in range(len(distances) - 1):
                # 安全比较，避免数组真值歧义
                dist_i = distances[i]
                dist_i_plus_1 = distances[i + 1]
                
                # 确保比较的是标量值
                if isinstance(dist_i, np.ndarray):
                    dist_i_val = dist_i.item() if dist_i.size == 1 else dist_i
                else:
                    dist_i_val = dist_i
                    
                if isinstance(dist_i_plus_1, np.ndarray):
                    dist_i_plus_1_val = dist_i_plus_1.item() if dist_i_plus_1.size == 1 else dist_i_plus_1
                else:
                    dist_i_plus_1_val = dist_i_plus_1
                
                if dist_i_val <= sample_dist <= dist_i_plus_1_val:
                    # 线性插值
                    if dist_i_plus_1_val - dist_i_val > 0:
                        t = (sample_dist - dist_i_val) / (dist_i_plus_1_val - dist_i_val)
                        point = points[i] + t * (points[i + 1] - points[i])
                    else:
                        point = points[i]
                    resampled_points.append(point)
                    break
            else:
                resampled_points.append(points[-1])
        
        return np.array(resampled_points).reshape(-1, 1, 2).astype(np.int32)
    
    def _create_shape_context_histogram(self, distances: np.ndarray, 
                                       angles: np.ndarray) -> np.ndarray:
        """
        创建形状上下文直方图
        
        Args:
            distances (np.ndarray): 距离数组
            angles (np.ndarray): 角度数组
            
        Returns:
            np.ndarray: 形状上下文直方图
        """
        # 设置直方图参数
        num_distance_bins = 5
        num_angle_bins = 12
        
        # 距离分箱（对数尺度）
        has_positive_distances = bool(np.any(distances > 0))
        max_distance = np.max(distances[distances > 0]) if has_positive_distances else 1
        distance_bins = np.logspace(0, np.log10(max_distance), num_distance_bins + 1)
        
        # 角度分箱
        angle_bins = np.linspace(-np.pi, np.pi, num_angle_bins + 1)
        
        # 创建2D直方图
        hist, _, _ = np.histogram2d(
            distances, angles, 
            bins=[distance_bins, angle_bins]
        )
        
        # 归一化
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum
        
        return hist.flatten()
    
    def visualize_contour_analysis(self, image: np.ndarray, contour: np.ndarray, 
                                  features: ContourFeatures) -> np.ndarray:
        """
        可视化轮廓分析结果
        
        Args:
            image (np.ndarray): 原始图像
            contour (np.ndarray): 轮廓
            features (ContourFeatures): 轮廓特征
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # 绘制轮廓
        cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
        
        # 绘制凸包
        cv2.drawContours(vis_image, [features.convex_hull], -1, (255, 0, 0), 1)
        
        # 绘制边界矩形
        x, y, w, h = features.bounding_rect
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        
        # 绘制质心
        centroid = (int(features.centroid[0]), int(features.centroid[1]))
        cv2.circle(vis_image, centroid, 3, (255, 255, 0), -1)
        
        # 添加文本信息
        info_text = [
            f"Area: {features.area:.1f}",
            f"Perimeter: {features.perimeter:.1f}",
            f"Aspect Ratio: {features.aspect_ratio:.2f}",
            f"Circularity: {features.circularity:.3f}",
            f"Solidity: {features.solidity:.3f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(vis_image, text, (10, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image