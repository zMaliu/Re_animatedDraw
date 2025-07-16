# -*- coding: utf-8 -*-
"""
骨架提取器

实现从笔画图像中提取中心线骨架的算法
支持多种骨架化方法，包括Zhang-Suen算法和形态学骨架化
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from skimage import morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt


class SkeletonExtractor:
    """
    骨架提取器
    
    提供多种骨架化算法，用于从二值化笔画图像中提取中心线
    """
    
    def __init__(self, config):
        """
        初始化骨架提取器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.method = config['stroke_detection'].get('skeleton_method', 'zhang_suen')
        
    def extract_skeleton(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """
        从二值图像中提取骨架
        
        Args:
            binary_image (np.ndarray): 二值化笔画图像
            
        Returns:
            Optional[np.ndarray]: 骨架点序列，形状为 (N, 2)
        """
        if binary_image is None or binary_image.size == 0:
            return None
        
        # 确保输入是二值图像
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
        
        # 根据配置选择骨架化方法
        if self.method == 'zhang_suen':
            skeleton_image = self._zhang_suen_thinning(binary_image)
        elif self.method == 'morphological':
            skeleton_image = self._morphological_skeleton(binary_image)
        elif self.method == 'medial_axis':
            skeleton_image = self._medial_axis_skeleton(binary_image)
        else:
            # 默认使用Zhang-Suen算法
            skeleton_image = self._zhang_suen_thinning(binary_image)
        
        if skeleton_image is None:
            return None
        
        # 将骨架图像转换为点序列
        skeleton_points = self._skeleton_to_points(skeleton_image)
        
        # 排序骨架点
        if skeleton_points is not None and len(skeleton_points) > 1:
            skeleton_points = self._order_skeleton_points(skeleton_points)
        
        return skeleton_points
    
    def _zhang_suen_thinning(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Zhang-Suen细化算法
        
        Args:
            binary_image (np.ndarray): 二值图像
            
        Returns:
            Optional[np.ndarray]: 骨架图像
        """
        # 将图像转换为0和1
        img = (binary_image > 0).astype(np.uint8)
        
        # 迭代细化
        changing = True
        while changing:
            changing = False
            
            # 第一次迭代
            to_remove = []
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    if img[i, j] == 1:
                        if self._zhang_suen_condition(img, i, j, step=1):
                            to_remove.append((i, j))
            
            for i, j in to_remove:
                img[i, j] = 0
                changing = True
            
            # 第二次迭代
            to_remove = []
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    if img[i, j] == 1:
                        if self._zhang_suen_condition(img, i, j, step=2):
                            to_remove.append((i, j))
            
            for i, j in to_remove:
                img[i, j] = 0
                changing = True
        
        return (img * 255).astype(np.uint8)
    
    def _zhang_suen_condition(self, img: np.ndarray, i: int, j: int, step: int) -> bool:
        """
        Zhang-Suen算法的条件检查
        
        Args:
            img (np.ndarray): 图像
            i, j (int): 像素坐标
            step (int): 迭代步骤 (1 或 2)
            
        Returns:
            bool: 是否满足删除条件
        """
        # 获取8邻域
        p2 = img[i-1, j]
        p3 = img[i-1, j+1]
        p4 = img[i, j+1]
        p5 = img[i+1, j+1]
        p6 = img[i+1, j]
        p7 = img[i+1, j-1]
        p8 = img[i, j-1]
        p9 = img[i-1, j-1]
        
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
        
        # 条件1: 2 <= B(P1) <= 6
        B = sum(neighbors)
        if not (2 <= B <= 6):
            return False
        
        # 条件2: A(P1) = 1
        A = 0
        for k in range(8):
            # 安全比较，避免数组真值歧义
            curr_neighbor = neighbors[k]
            next_neighbor = neighbors[(k+1) % 8]
            
            # 确保比较的是标量值
            if isinstance(curr_neighbor, np.ndarray):
                curr_val = curr_neighbor.item() if curr_neighbor.size == 1 else curr_neighbor
            else:
                curr_val = curr_neighbor
                
            if isinstance(next_neighbor, np.ndarray):
                next_val = next_neighbor.item() if next_neighbor.size == 1 else next_neighbor
            else:
                next_val = next_neighbor
                
            if curr_val == 0 and next_val == 1:
                A += 1
        if A != 1:
            return False
        
        # 条件3和4根据步骤不同
        if step == 1:
            # P2 * P4 * P6 = 0
            # 安全计算乘积，避免数组真值歧义
            product1 = p2 * p4 * p6
            if isinstance(product1, np.ndarray):
                product1_val = product1.item() if product1.size == 1 else product1
            else:
                product1_val = product1
            if product1_val != 0:
                return False
                
            # P4 * P6 * P8 = 0
            product2 = p4 * p6 * p8
            if isinstance(product2, np.ndarray):
                product2_val = product2.item() if product2.size == 1 else product2
            else:
                product2_val = product2
            if product2_val != 0:
                return False
        else:  # step == 2
            # P2 * P4 * P8 = 0
            product3 = p2 * p4 * p8
            if isinstance(product3, np.ndarray):
                product3_val = product3.item() if product3.size == 1 else product3
            else:
                product3_val = product3
            if product3_val != 0:
                return False
                
            # P2 * P6 * P8 = 0
            product4 = p2 * p6 * p8
            if isinstance(product4, np.ndarray):
                product4_val = product4.item() if product4.size == 1 else product4
            else:
                product4_val = product4
            if product4_val != 0:
                return False
        
        return True
    
    def _morphological_skeleton(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """
        形态学骨架化
        
        Args:
            binary_image (np.ndarray): 二值图像
            
        Returns:
            Optional[np.ndarray]: 骨架图像
        """
        try:
            # 转换为布尔图像
            binary = binary_image > 0
            
            # 使用skimage的骨架化
            skeleton = morphology.skeletonize(binary)
            
            return (skeleton * 255).astype(np.uint8)
        except Exception as e:
            print(f"形态学骨架化失败: {str(e)}")
            return None
    
    def _medial_axis_skeleton(self, binary_image: np.ndarray) -> Optional[np.ndarray]:
        """
        中轴变换骨架化
        
        Args:
            binary_image (np.ndarray): 二值图像
            
        Returns:
            Optional[np.ndarray]: 骨架图像
        """
        try:
            # 转换为布尔图像
            binary = binary_image > 0
            
            # 使用skimage的中轴变换
            skeleton = morphology.medial_axis(binary)
            
            return (skeleton * 255).astype(np.uint8)
        except Exception as e:
            print(f"中轴变换骨架化失败: {str(e)}")
            return None
    
    def _skeleton_to_points(self, skeleton_image: np.ndarray) -> Optional[np.ndarray]:
        """
        将骨架图像转换为点序列
        
        Args:
            skeleton_image (np.ndarray): 骨架图像
            
        Returns:
            Optional[np.ndarray]: 骨架点序列
        """
        if skeleton_image is None:
            return None
        
        # 找到所有骨架像素
        skeleton_pixels = np.where(skeleton_image > 0)
        
        if len(skeleton_pixels[0]) == 0:
            return None
        
        # 转换为点列表 (x, y)
        points = np.column_stack((skeleton_pixels[1], skeleton_pixels[0]))
        
        return points
    
    def _order_skeleton_points(self, points: np.ndarray) -> np.ndarray:
        """
        对骨架点进行排序，形成连续的路径
        
        Args:
            points (np.ndarray): 无序的骨架点
            
        Returns:
            np.ndarray: 有序的骨架点
        """
        if len(points) <= 2:
            return points
        
        # 找到端点（只有一个邻居的点）
        endpoints = self._find_endpoints(points)
        
        if len(endpoints) == 0:
            # 如果没有端点，可能是闭合曲线，从任意点开始
            start_point = points[0]
        else:
            # 从第一个端点开始
            start_point = endpoints[0]
        
        # 使用贪心算法构建路径
        ordered_points = [start_point]
        remaining_points = []
        for p in points:
            if not np.array_equal(p, start_point):
                remaining_points.append(p)
        
        current_point = start_point
        while remaining_points:
            # 找到距离当前点最近的点
            distances = [np.linalg.norm(p - current_point) for p in remaining_points]
            min_idx = np.argmin(distances)
            
            # 只考虑相邻的点（距离小于等于sqrt(2)）
            # 安全比较，避免数组真值歧义
            min_distance = distances[min_idx]
            if isinstance(min_distance, np.ndarray):
                min_distance_val = min_distance.item() if min_distance.size == 1 else min_distance
            else:
                min_distance_val = min_distance
            
            if min_distance_val <= np.sqrt(2) + 0.1:
                next_point = remaining_points[min_idx]
                ordered_points.append(next_point)
                current_point = next_point
                remaining_points.pop(min_idx)
            else:
                # 如果没有相邻点，可能有断裂，从剩余点中选择一个新的起点
                if remaining_points:
                    next_point = remaining_points[0]
                    ordered_points.append(next_point)
                    current_point = next_point
                    remaining_points.pop(0)
                else:
                    break
        
        return np.array(ordered_points)
    
    def _find_endpoints(self, points: np.ndarray) -> List[np.ndarray]:
        """
        找到骨架的端点
        
        Args:
            points (np.ndarray): 骨架点
            
        Returns:
            List[np.ndarray]: 端点列表
        """
        endpoints = []
        
        for point in points:
            # 计算邻居数量
            neighbors = 0
            for other_point in points:
                if not np.array_equal(point, other_point):
                    distance = np.linalg.norm(point - other_point)
                    if distance <= np.sqrt(2) + 0.1:  # 8连通邻域
                        neighbors += 1
            
            # 端点只有一个邻居
            if neighbors == 1:
                endpoints.append(point)
        
        return endpoints
    
    def smooth_skeleton(self, skeleton_points: np.ndarray, 
                       smoothing_factor: float = 0.1) -> np.ndarray:
        """
        平滑骨架点序列
        
        Args:
            skeleton_points (np.ndarray): 原始骨架点
            smoothing_factor (float): 平滑因子
            
        Returns:
            np.ndarray: 平滑后的骨架点
        """
        if len(skeleton_points) <= 2:
            return skeleton_points
        
        smoothed_points = skeleton_points.copy().astype(np.float32)
        
        # 简单的移动平均平滑
        for i in range(1, len(smoothed_points) - 1):
            prev_point = smoothed_points[i - 1]
            curr_point = smoothed_points[i]
            next_point = smoothed_points[i + 1]
            
            # 计算平均位置
            avg_point = (prev_point + curr_point + next_point) / 3
            
            # 应用平滑
            smoothed_points[i] = curr_point + smoothing_factor * (avg_point - curr_point)
        
        return smoothed_points.astype(np.int32)
    
    def resample_skeleton(self, skeleton_points: np.ndarray, 
                         target_length: int) -> np.ndarray:
        """
        重采样骨架点序列到指定长度
        
        Args:
            skeleton_points (np.ndarray): 原始骨架点
            target_length (int): 目标长度
            
        Returns:
            np.ndarray: 重采样后的骨架点
        """
        if len(skeleton_points) <= 1:
            return skeleton_points
        
        # 计算累积距离
        distances = [0]
        for i in range(1, len(skeleton_points)):
            dist = np.linalg.norm(skeleton_points[i] - skeleton_points[i-1])
            distances.append(distances[-1] + dist)
        
        total_length = distances[-1]
        if total_length == 0:
            return skeleton_points
        
        # 生成等间距的采样点
        sample_distances = np.linspace(0, total_length, target_length)
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
                    t = (sample_dist - dist_i_val) / (dist_i_plus_1_val - dist_i_val)
                    point = skeleton_points[i] + t * (skeleton_points[i + 1] - skeleton_points[i])
                    resampled_points.append(point)
                    break
            else:
                # 如果没找到，使用最后一个点
                resampled_points.append(skeleton_points[-1])
        
        return np.array(resampled_points)
    
    def visualize_skeleton(self, original_image: np.ndarray, 
                          skeleton_points: np.ndarray) -> np.ndarray:
        """
        可视化骨架提取结果
        
        Args:
            original_image (np.ndarray): 原始图像
            skeleton_points (np.ndarray): 骨架点
            
        Returns:
            np.ndarray: 可视化图像
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        if skeleton_points is not None and len(skeleton_points) > 0:
            # 绘制骨架点
            for point in skeleton_points:
                cv2.circle(vis_image, tuple(point.astype(int)), 1, (0, 255, 0), -1)
            
            # 绘制骨架线
            if len(skeleton_points) > 1:
                for i in range(1, len(skeleton_points)):
                    pt1 = tuple(skeleton_points[i-1].astype(int))
                    pt2 = tuple(skeleton_points[i].astype(int))
                    cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)
            
            # 标记起点和终点
            if len(skeleton_points) > 0:
                start_point = tuple(skeleton_points[0].astype(int))
                end_point = tuple(skeleton_points[-1].astype(int))
                cv2.circle(vis_image, start_point, 5, (0, 0, 255), -1)  # 红色起点
                cv2.circle(vis_image, end_point, 5, (255, 0, 255), -1)   # 品红终点
        
        return vis_image