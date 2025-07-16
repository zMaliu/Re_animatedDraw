# -*- coding: utf-8 -*-
"""
Harris角点检测器

实现论文中要求的Harris角点检测算法
用于提取笔触骨架的关键点
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging


class HarrisCornerDetector:
    """
    Harris角点检测器
    
    实现论文中的角点检测功能，用于提取骨架关键点
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化角点检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Harris角点检测参数
        self.block_size = config.get('harris_block_size', 2)
        self.ksize = config.get('harris_ksize', 3)
        self.k = config.get('harris_k', 0.04)
        self.threshold = config.get('harris_threshold', 0.01)
        
        # 非极大值抑制参数
        self.nms_radius = config.get('nms_radius', 5)
        
        # 角点质量参数
        self.quality_level = config.get('corner_quality_level', 0.01)
        self.min_distance = config.get('corner_min_distance', 10)
        self.max_corners = config.get('max_corners', 100)
        
    def detect_corners(self, image: np.ndarray, 
                      mask: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        """
        检测图像中的角点
        
        Args:
            image: 输入图像
            mask: 可选的掩码
            
        Returns:
            List[Tuple[float, float]]: 角点坐标列表
        """
        try:
            # 预处理
            gray = self._preprocess_image(image)
            
            # Harris角点检测
            harris_response = cv2.cornerHarris(
                gray, 
                self.block_size, 
                self.ksize, 
                self.k
            )
            
            # 应用掩码
            if mask is not None:
                harris_response = harris_response * (mask > 0)
            
            # 提取角点
            corners = self._extract_corners_from_response(harris_response)
            
            return corners
            
        except Exception as e:
            self.logger.error(f"Error in corner detection: {str(e)}")
            return []
    
    def detect_skeleton_keypoints(self, skeleton: np.ndarray) -> List[Tuple[float, float]]:
        """
        检测骨架的关键点
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            List[Tuple[float, float]]: 关键点坐标列表
        """
        try:
            # 在骨架上检测角点
            corners = self.detect_corners(skeleton, skeleton)
            
            # 添加端点检测
            endpoints = self._detect_endpoints(skeleton)
            
            # 添加分叉点检测
            junctions = self._detect_junctions(skeleton)
            
            # 合并所有关键点
            all_keypoints = corners + endpoints + junctions
            
            # 去重和过滤
            filtered_keypoints = self._filter_keypoints(all_keypoints)
            
            return filtered_keypoints
            
        except Exception as e:
            self.logger.error(f"Error detecting skeleton keypoints: {str(e)}")
            return []
    
    def detect_stroke_corners(self, stroke_mask: np.ndarray, 
                             original_image: np.ndarray) -> Dict[str, Any]:
        """
        检测笔触的角点信息
        
        Args:
            stroke_mask: 笔触掩码
            original_image: 原始图像
            
        Returns:
            Dict: 角点检测结果
        """
        try:
            # 在笔触区域内检测角点
            corners = self.detect_corners(original_image, stroke_mask)
            
            # 分析角点特征
            corner_features = self._analyze_corner_features(corners, stroke_mask)
            
            # 计算角点质量
            corner_qualities = self._calculate_corner_qualities(
                original_image, corners, stroke_mask
            )
            
            return {
                'corners': corners,
                'features': corner_features,
                'qualities': corner_qualities,
                'count': len(corners)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting stroke corners: {str(e)}")
            return {
                'corners': [],
                'features': {},
                'qualities': [],
                'count': 0
            }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的灰度图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 确保数据类型正确
        gray = gray.astype(np.float32)
        
        return gray
    
    def _extract_corners_from_response(self, harris_response: np.ndarray) -> List[Tuple[float, float]]:
        """
        从Harris响应中提取角点
        
        Args:
            harris_response: Harris角点响应图
            
        Returns:
            List[Tuple[float, float]]: 角点坐标列表
        """
        # 阈值化
        threshold_value = self.threshold * harris_response.max()
        corner_mask = harris_response > threshold_value
        
        # 非极大值抑制
        corners = self._non_maximum_suppression(harris_response, corner_mask)
        
        return corners
    
    def _non_maximum_suppression(self, response: np.ndarray, 
                                mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        非极大值抑制
        
        Args:
            response: 响应图
            mask: 候选点掩码
            
        Returns:
            List[Tuple[float, float]]: 抑制后的角点列表
        """
        corners = []
        
        # 获取候选点坐标
        candidate_coords = np.where(mask)
        candidate_points = list(zip(candidate_coords[1], candidate_coords[0]))  # (x, y)
        
        # 按响应值排序
        candidate_points.sort(
            key=lambda p: response[p[1], p[0]], reverse=True
        )
        
        # 非极大值抑制
        for point in candidate_points:
            x, y = point
            
            # 检查是否与已选择的角点距离过近
            too_close = False
            for existing_corner in corners:
                ex, ey = existing_corner
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)
                if distance < self.nms_radius:
                    too_close = True
                    break
            
            if not too_close:
                corners.append((float(x), float(y)))
                
                # 限制角点数量
                if len(corners) >= self.max_corners:
                    break
        
        return corners
    
    def _detect_endpoints(self, skeleton: np.ndarray) -> List[Tuple[float, float]]:
        """
        检测骨架端点
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            List[Tuple[float, float]]: 端点坐标列表
        """
        endpoints = []
        
        # 端点检测核
        endpoint_kernels = [
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),  # 右上
            np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),  # 上
            np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]),  # 左上
            np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),  # 右
            np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),  # 左
            np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),  # 右下
            np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]),  # 下
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])   # 左下
        ]
        
        skeleton_binary = (skeleton > 0).astype(np.uint8)
        
        for kernel in endpoint_kernels:
            # 模板匹配
            result = cv2.filter2D(skeleton_binary, -1, kernel)
            
            # 找到匹配的端点
            endpoint_mask = (result == 2) & (skeleton_binary == 1)
            endpoint_coords = np.where(endpoint_mask)
            
            for y, x in zip(endpoint_coords[0], endpoint_coords[1]):
                endpoints.append((float(x), float(y)))
        
        return endpoints
    
    def _detect_junctions(self, skeleton: np.ndarray) -> List[Tuple[float, float]]:
        """
        检测骨架分叉点
        
        Args:
            skeleton: 骨架图像
            
        Returns:
            List[Tuple[float, float]]: 分叉点坐标列表
        """
        junctions = []
        
        skeleton_binary = (skeleton > 0).astype(np.uint8)
        rows, cols = skeleton_binary.shape
        
        # 遍历每个骨架像素
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if skeleton_binary[y, x] == 1:
                    # 检查8邻域
                    neighbors = [
                        skeleton_binary[y-1, x-1], skeleton_binary[y-1, x], skeleton_binary[y-1, x+1],
                        skeleton_binary[y, x+1], skeleton_binary[y+1, x+1], skeleton_binary[y+1, x],
                        skeleton_binary[y+1, x-1], skeleton_binary[y, x-1]
                    ]
                    
                    # 计算连通分量数量
                    transitions = 0
                    for i in range(len(neighbors)):
                        if neighbors[i] == 0 and neighbors[(i + 1) % len(neighbors)] == 1:
                            transitions += 1
                    
                    # 分叉点通常有3个或更多连通分量
                    if transitions >= 3:
                        junctions.append((float(x), float(y)))
        
        return junctions
    
    def _filter_keypoints(self, keypoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        过滤和去重关键点
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            List[Tuple[float, float]]: 过滤后的关键点列表
        """
        if not keypoints:
            return []
        
        # 去重
        unique_keypoints = []
        for point in keypoints:
            is_duplicate = False
            for existing_point in unique_keypoints:
                distance = np.sqrt(
                    (point[0] - existing_point[0])**2 + 
                    (point[1] - existing_point[1])**2
                )
                if distance < self.min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_keypoints.append(point)
        
        # 限制数量
        if len(unique_keypoints) > self.max_corners:
            unique_keypoints = unique_keypoints[:self.max_corners]
        
        return unique_keypoints
    
    def _analyze_corner_features(self, corners: List[Tuple[float, float]], 
                                stroke_mask: np.ndarray) -> Dict[str, Any]:
        """
        分析角点特征
        
        Args:
            corners: 角点列表
            stroke_mask: 笔触掩码
            
        Returns:
            Dict: 角点特征
        """
        features = {
            'corner_count': len(corners),
            'corner_density': 0.0,
            'corner_distribution': 'uniform'
        }
        
        if not corners:
            return features
        
        # 计算角点密度
        stroke_area = np.sum(stroke_mask > 0)
        if stroke_area > 0:
            features['corner_density'] = len(corners) / stroke_area
        
        # 分析角点分布
        if len(corners) > 1:
            corner_array = np.array(corners)
            
            # 计算角点间距离的统计信息
            distances = []
            for i in range(len(corners)):
                for j in range(i + 1, len(corners)):
                    dist = np.sqrt(
                        (corners[i][0] - corners[j][0])**2 + 
                        (corners[i][1] - corners[j][1])**2
                    )
                    distances.append(dist)
            
            if distances:
                features['mean_corner_distance'] = float(np.mean(distances))
                features['std_corner_distance'] = float(np.std(distances))
                
                # 判断分布类型
                cv = features['std_corner_distance'] / features['mean_corner_distance'] if features['mean_corner_distance'] > 0 else 0
                if cv < 0.3:
                    features['corner_distribution'] = 'uniform'
                elif cv > 0.7:
                    features['corner_distribution'] = 'clustered'
                else:
                    features['corner_distribution'] = 'random'
        
        return features
    
    def _calculate_corner_qualities(self, image: np.ndarray, 
                                   corners: List[Tuple[float, float]],
                                   mask: np.ndarray) -> List[float]:
        """
        计算角点质量
        
        Args:
            image: 原始图像
            corners: 角点列表
            mask: 掩码
            
        Returns:
            List[float]: 角点质量列表
        """
        qualities = []
        
        if not corners:
            return qualities
        
        # 转换为灰度图
        gray = self._preprocess_image(image)
        
        # 计算Harris响应
        harris_response = cv2.cornerHarris(
            gray, self.block_size, self.ksize, self.k
        )
        
        # 应用掩码
        harris_response = harris_response * (mask > 0)
        
        # 获取每个角点的质量
        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            if 0 <= x < harris_response.shape[1] and 0 <= y < harris_response.shape[0]:
                quality = float(harris_response[y, x])
            else:
                quality = 0.0
            qualities.append(quality)
        
        return qualities
    
    def get_adaptive_parameters(self, image: np.ndarray) -> Dict[str, float]:
        """
        自适应计算检测参数
        
        Args:
            image: 输入图像
            
        Returns:
            Dict[str, float]: 自适应参数
        """
        try:
            # 转换为灰度图
            gray = self._preprocess_image(image)
            
            # 计算图像统计信息
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # 自适应阈值
            adaptive_threshold = max(0.001, min(0.1, std_intensity / mean_intensity * 0.01))
            
            # 自适应最小距离
            image_size = min(gray.shape)
            adaptive_min_distance = max(5, image_size // 50)
            
            return {
                'threshold': adaptive_threshold,
                'min_distance': float(adaptive_min_distance),
                'quality_level': adaptive_threshold * 0.1
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating adaptive parameters: {str(e)}")
            return {
                'threshold': self.threshold,
                'min_distance': float(self.min_distance),
                'quality_level': self.quality_level
            }