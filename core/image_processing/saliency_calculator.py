# -*- coding: utf-8 -*-
"""
显著性计算器

实现论文中要求的基于三分法则的显著性得分计算
用于评估笔触的位置特征
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging


class SaliencyCalculator:
    """
    显著性计算器
    
    基于三分法则计算笔触的显著性得分
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化显著性计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 三分法则参数
        self.grid_size = config.get('rule_of_thirds_grid', (3, 3))
        self.intersection_weight = config.get('intersection_weight', 1.0)
        self.line_weight = config.get('line_weight', 0.6)
        self.center_weight = config.get('center_weight', 0.3)
        
        # 距离衰减参数
        self.distance_decay = config.get('distance_decay', 0.1)
        self.max_distance_ratio = config.get('max_distance_ratio', 0.5)
        
        # 预计算三分法则模板
        self._rule_of_thirds_template = None
        
    def calculate_saliency_score(self, stroke_mask: np.ndarray, 
                                canvas_size: Tuple[int, int]) -> float:
        """
        计算笔触的显著性得分
        
        Args:
            stroke_mask: 笔触掩码
            canvas_size: 画布尺寸
            
        Returns:
            float: 显著性得分
        """
        try:
            # 获取笔触质心
            centroid = self._calculate_centroid(stroke_mask)
            if centroid is None:
                return 0.0
            
            # 计算基于三分法则的得分
            rule_score = self._calculate_rule_of_thirds_score(centroid, canvas_size)
            
            # 计算基于位置的得分
            position_score = self._calculate_position_score(centroid, canvas_size)
            
            # 计算基于面积分布的得分
            area_score = self._calculate_area_distribution_score(stroke_mask, canvas_size)
            
            # 组合得分
            total_score = (
                0.5 * rule_score +
                0.3 * position_score +
                0.2 * area_score
            )
            
            return float(np.clip(total_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating saliency score: {str(e)}")
            return 0.0
    
    def _calculate_centroid(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        计算掩码的质心
        
        Args:
            mask: 二值掩码
            
        Returns:
            Tuple: 质心坐标，如果失败则返回None
        """
        try:
            moments = cv2.moments(mask)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                return (cx, cy)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating centroid: {str(e)}")
            return None
    
    def _calculate_rule_of_thirds_score(self, centroid: Tuple[float, float], 
                                      canvas_size: Tuple[int, int]) -> float:
        """
        计算基于三分法则的得分
        
        Args:
            centroid: 质心坐标
            canvas_size: 画布尺寸
            
        Returns:
            float: 三分法则得分
        """
        try:
            width, height = canvas_size
            cx, cy = centroid
            
            # 归一化坐标
            norm_x = cx / width
            norm_y = cy / height
            
            # 三分法则交点
            intersection_points = [
                (1/3, 1/3), (2/3, 1/3),
                (1/3, 2/3), (2/3, 2/3)
            ]
            
            # 计算到最近交点的距离
            min_distance = float('inf')
            for ix, iy in intersection_points:
                distance = np.sqrt((norm_x - ix)**2 + (norm_y - iy)**2)
                min_distance = min(min_distance, distance)
            
            # 距离衰减得分
            intersection_score = np.exp(-min_distance / self.distance_decay)
            
            # 三分法则线得分
            line_distances = [
                abs(norm_x - 1/3),  # 左三分线
                abs(norm_x - 2/3),  # 右三分线
                abs(norm_y - 1/3),  # 上三分线
                abs(norm_y - 2/3)   # 下三分线
            ]
            
            min_line_distance = min(line_distances)
            line_score = np.exp(-min_line_distance / self.distance_decay)
            
            # 组合得分
            total_score = (
                self.intersection_weight * intersection_score +
                self.line_weight * line_score
            )
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error calculating rule of thirds score: {str(e)}")
            return 0.0
    
    def _calculate_position_score(self, centroid: Tuple[float, float], 
                                canvas_size: Tuple[int, int]) -> float:
        """
        计算基于位置的得分
        
        Args:
            centroid: 质心坐标
            canvas_size: 画布尺寸
            
        Returns:
            float: 位置得分
        """
        try:
            width, height = canvas_size
            cx, cy = centroid
            
            # 归一化坐标
            norm_x = cx / width
            norm_y = cy / height
            
            # 中心得分（距离画面中心的距离）
            center_distance = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
            center_score = 1.0 - center_distance / np.sqrt(0.5)
            
            # 边缘得分（避免过于靠近边缘）
            edge_distances = [norm_x, 1-norm_x, norm_y, 1-norm_y]
            min_edge_distance = min(edge_distances)
            edge_score = min_edge_distance * 4  # 归一化到0-1
            
            # 黄金比例得分
            golden_ratio = 0.618
            golden_points = [
                (golden_ratio, 0.5), (1-golden_ratio, 0.5),
                (0.5, golden_ratio), (0.5, 1-golden_ratio)
            ]
            
            min_golden_distance = float('inf')
            for gx, gy in golden_points:
                distance = np.sqrt((norm_x - gx)**2 + (norm_y - gy)**2)
                min_golden_distance = min(min_golden_distance, distance)
            
            golden_score = np.exp(-min_golden_distance / self.distance_decay)
            
            # 组合得分
            total_score = (
                0.3 * center_score +
                0.3 * edge_score +
                0.4 * golden_score
            )
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error calculating position score: {str(e)}")
            return 0.0
    
    def _calculate_area_distribution_score(self, stroke_mask: np.ndarray, 
                                         canvas_size: Tuple[int, int]) -> float:
        """
        计算基于面积分布的得分
        
        Args:
            stroke_mask: 笔触掩码
            canvas_size: 画布尺寸
            
        Returns:
            float: 面积分布得分
        """
        try:
            height, width = stroke_mask.shape
            canvas_width, canvas_height = canvas_size
            
            # 调整掩码尺寸以匹配画布
            if (width, height) != tuple(canvas_size):
                resized_mask = cv2.resize(stroke_mask, canvas_size)
            else:
                resized_mask = stroke_mask
            
            # 将画布分为9个区域（3x3网格）
            grid_height = canvas_height // 3
            grid_width = canvas_width // 3
            
            region_scores = []
            
            for i in range(3):
                for j in range(3):
                    # 定义区域边界
                    y1 = i * grid_height
                    y2 = (i + 1) * grid_height if i < 2 else canvas_height
                    x1 = j * grid_width
                    x2 = (j + 1) * grid_width if j < 2 else canvas_width
                    
                    # 提取区域
                    region = resized_mask[y1:y2, x1:x2]
                    
                    # 计算区域内的笔触密度
                    region_area = region.shape[0] * region.shape[1]
                    stroke_pixels = np.sum(region > 0)
                    
                    if region_area > 0:
                        density = stroke_pixels / region_area
                    else:
                        density = 0.0
                    
                    # 根据区域位置给予不同权重
                    # 中心区域和三分法则交点区域权重较高
                    if (i == 1 and j == 1):  # 中心区域
                        weight = 0.8
                    elif (i in [0, 2] and j in [0, 2]):  # 四个角
                        weight = 1.2
                    else:  # 边缘中点
                        weight = 1.0
                    
                    region_scores.append(density * weight)
            
            # 计算分布的均匀性和集中度
            if region_scores:
                mean_score = np.mean(region_scores)
                std_score = np.std(region_scores)
                
                # 平衡均匀性和集中度
                uniformity = 1.0 / (1.0 + std_score)
                concentration = mean_score
                
                distribution_score = 0.6 * concentration + 0.4 * uniformity
            else:
                distribution_score = 0.0
            
            return distribution_score
            
        except Exception as e:
            self.logger.error(f"Error calculating area distribution score: {str(e)}")
            return 0.0
    
    def create_saliency_map(self, canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        创建显著性地图
        
        Args:
            canvas_size: 画布尺寸
            
        Returns:
            np.ndarray: 显著性地图
        """
        try:
            width, height = canvas_size
            saliency_map = np.zeros((height, width), dtype=np.float32)
            
            # 为每个像素计算显著性得分
            for y in range(height):
                for x in range(width):
                    # 创建单点掩码
                    point_mask = np.zeros((height, width), dtype=np.uint8)
                    point_mask[y, x] = 255
                    
                    # 计算该点的显著性得分
                    score = self.calculate_saliency_score(point_mask, canvas_size)
                    saliency_map[y, x] = score
            
            # 归一化
            max_saliency = np.max(saliency_map)
            if max_saliency > 0:
                saliency_map = saliency_map / max_saliency
            
            return saliency_map
            
        except Exception as e:
            self.logger.error(f"Error creating saliency map: {str(e)}")
            return np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)
    
    def visualize_rule_of_thirds(self, canvas_size: Tuple[int, int]) -> np.ndarray:
        """
        可视化三分法则网格
        
        Args:
            canvas_size: 画布尺寸
            
        Returns:
            np.ndarray: 三分法则可视化图像
        """
        try:
            width, height = canvas_size
            visualization = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 绘制三分法则线
            line_color = (255, 255, 255)
            line_thickness = 2
            
            # 垂直线
            cv2.line(visualization, (width//3, 0), (width//3, height), line_color, line_thickness)
            cv2.line(visualization, (2*width//3, 0), (2*width//3, height), line_color, line_thickness)
            
            # 水平线
            cv2.line(visualization, (0, height//3), (width, height//3), line_color, line_thickness)
            cv2.line(visualization, (0, 2*height//3), (width, 2*height//3), line_color, line_thickness)
            
            # 标记交点
            intersection_color = (0, 255, 0)
            radius = 8
            
            intersections = [
                (width//3, height//3), (2*width//3, height//3),
                (width//3, 2*height//3), (2*width//3, 2*height//3)
            ]
            
            for x, y in intersections:
                cv2.circle(visualization, (x, y), radius, intersection_color, -1)
            
            return visualization
            
        except Exception as e:
            self.logger.error(f"Error visualizing rule of thirds: {str(e)}")
            return np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)