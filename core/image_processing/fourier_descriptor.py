# -*- coding: utf-8 -*-
"""
傅里叶描述子

实现论文中要求的傅里叶描述子算法
用于计算笔触的圆度和细长比等形状特征
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.fft import fft, ifft


class FourierDescriptor:
    """
    傅里叶描述子
    
    实现论文中的形状特征提取功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化傅里叶描述子
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 傅里叶描述子参数
        self.num_descriptors = config.get('fourier_descriptors', 10)
        self.normalize = config.get('normalize_descriptors', True)
        self.invariant_to_rotation = config.get('rotation_invariant', True)
        self.invariant_to_scale = config.get('scale_invariant', True)
        self.invariant_to_translation = config.get('translation_invariant', True)
        
        # 轮廓处理参数
        self.contour_approximation = config.get('contour_approximation', 0.02)
        self.min_contour_points = config.get('min_contour_points', 10)
        
    def extract_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """
        从轮廓提取形状特征
        
        Args:
            contour: 轮廓点
            
        Returns:
            Dict: 形状特征字典
        """
        try:
            if len(contour) < self.min_contour_points:
                return self._create_default_features()
            
            # 计算傅里叶描述子
            descriptors = self.compute_fourier_descriptors(contour)
            
            # 计算形状特征
            circularity = self._calculate_circularity(contour, descriptors)
            elongation = self._calculate_elongation(contour, descriptors)
            compactness = self._calculate_compactness(contour)
            convexity = self._calculate_convexity(contour)
            
            # 计算几何特征
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 计算边界框特征
            rect = cv2.boundingRect(contour)
            aspect_ratio = rect[2] / rect[3] if rect[3] > 0 else 1.0
            
            # 计算最小外接矩形
            min_rect = cv2.minAreaRect(contour)
            min_width, min_height = min_rect[1]
            min_aspect_ratio = max(min_width, min_height) / min(min_width, min_height) if min(min_width, min_height) > 0 else 1.0
            
            return {
                'circularity': circularity,
                'elongation': elongation,
                'compactness': compactness,
                'convexity': convexity,
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'min_aspect_ratio': min_aspect_ratio,
                'fourier_descriptors': descriptors[:self.num_descriptors].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting shape features: {str(e)}")
            return self._create_default_features()
    
    def compute_fourier_descriptors(self, contour: np.ndarray) -> np.ndarray:
        """
        计算轮廓的傅里叶描述子
        
        Args:
            contour: 轮廓点
            
        Returns:
            np.ndarray: 傅里叶描述子
        """
        try:
            # 预处理轮廓
            processed_contour = self._preprocess_contour(contour)
            
            # 转换为复数表示
            complex_contour = processed_contour[:, 0] + 1j * processed_contour[:, 1]
            
            # 计算傅里叶变换
            fourier_coeffs = fft(complex_contour)
            
            # 应用不变性
            if self.invariant_to_translation:
                fourier_coeffs[0] = 0  # 去除直流分量
            
            if self.invariant_to_scale and len(fourier_coeffs) > 1:
                # 归一化到第一个非零系数
                first_nonzero = 1 if fourier_coeffs[1] != 0 else 2
                if first_nonzero < len(fourier_coeffs) and fourier_coeffs[first_nonzero] != 0:
                    fourier_coeffs = fourier_coeffs / abs(fourier_coeffs[first_nonzero])
            
            if self.invariant_to_rotation and len(fourier_coeffs) > 1:
                # 取模长，去除相位信息
                fourier_coeffs = np.abs(fourier_coeffs)
            
            # 归一化
            if self.normalize and len(fourier_coeffs) > 0:
                max_coeff = np.max(np.abs(fourier_coeffs))
                if max_coeff > 0:
                    fourier_coeffs = fourier_coeffs / max_coeff
            
            return fourier_coeffs
            
        except Exception as e:
            self.logger.error(f"Error computing Fourier descriptors: {str(e)}")
            return np.zeros(self.num_descriptors, dtype=complex)
    
    def _preprocess_contour(self, contour: np.ndarray) -> np.ndarray:
        """
        预处理轮廓
        
        Args:
            contour: 原始轮廓
            
        Returns:
            np.ndarray: 预处理后的轮廓
        """
        # 简化轮廓
        epsilon = self.contour_approximation * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # 重新采样到固定点数
        target_points = max(self.min_contour_points, len(approx_contour))
        resampled = self._resample_contour(approx_contour, target_points)
        
        return resampled
    
    def _resample_contour(self, contour: np.ndarray, num_points: int) -> np.ndarray:
        """
        重新采样轮廓到指定点数
        
        Args:
            contour: 输入轮廓
            num_points: 目标点数
            
        Returns:
            np.ndarray: 重采样后的轮廓
        """
        # 计算轮廓的累积弧长
        contour_2d = contour.reshape(-1, 2)
        distances = np.sqrt(np.sum(np.diff(contour_2d, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distances[-1]
        
        # 等间距采样
        target_distances = np.linspace(0, total_length, num_points, endpoint=False)
        
        # 插值得到新的点
        resampled_x = np.interp(target_distances, cumulative_distances, contour_2d[:, 0])
        resampled_y = np.interp(target_distances, cumulative_distances, contour_2d[:, 1])
        
        return np.column_stack([resampled_x, resampled_y])
    
    def _calculate_circularity(self, contour: np.ndarray, descriptors: np.ndarray) -> float:
        """
        计算圆度
        
        Args:
            contour: 轮廓
            descriptors: 傅里叶描述子
            
        Returns:
            float: 圆度值
        """
        try:
            # 方法1：基于面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity_1 = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity_1 = 0.0
            
            # 方法2：基于傅里叶描述子
            if len(descriptors) > 2:
                # 第一个谐波与其他谐波的比值
                first_harmonic = abs(descriptors[1]) if len(descriptors) > 1 else 0
                higher_harmonics = np.sum(np.abs(descriptors[2:]))
                
                if higher_harmonics > 0:
                    circularity_2 = first_harmonic / (first_harmonic + higher_harmonics)
                else:
                    circularity_2 = 1.0
            else:
                circularity_2 = 0.0
            
            # 组合两种方法
            return 0.7 * circularity_1 + 0.3 * circularity_2
            
        except Exception as e:
            self.logger.error(f"Error calculating circularity: {str(e)}")
            return 0.0
    
    def _calculate_elongation(self, contour: np.ndarray, descriptors: np.ndarray) -> float:
        """
        计算细长比
        
        Args:
            contour: 轮廓
            descriptors: 傅里叶描述子
            
        Returns:
            float: 细长比值
        """
        try:
            # 方法1：基于最小外接矩形
            min_rect = cv2.minAreaRect(contour)
            width, height = min_rect[1]
            
            if min(width, height) > 0:
                elongation_1 = max(width, height) / min(width, height)
            else:
                elongation_1 = 1.0
            
            # 方法2：基于主轴分析
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                # 计算二阶矩
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                
                # 计算主轴长度
                lambda1 = 0.5 * (mu20 + mu02 + np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
                lambda2 = 0.5 * (mu20 + mu02 - np.sqrt(4 * mu11**2 + (mu20 - mu02)**2))
                
                if lambda2 > 0:
                    elongation_2 = np.sqrt(lambda1 / lambda2)
                else:
                    elongation_2 = 1.0
            else:
                elongation_2 = 1.0
            
            # 组合两种方法
            return 0.6 * elongation_1 + 0.4 * elongation_2
            
        except Exception as e:
            self.logger.error(f"Error calculating elongation: {str(e)}")
            return 1.0
    
    def _calculate_compactness(self, contour: np.ndarray) -> float:
        """
        计算紧凑度
        
        Args:
            contour: 轮廓
            
        Returns:
            float: 紧凑度值
        """
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 0:
                return perimeter**2 / (4 * np.pi * area)
            else:
                return float('inf')
                
        except Exception as e:
            self.logger.error(f"Error calculating compactness: {str(e)}")
            return float('inf')
    
    def _calculate_convexity(self, contour: np.ndarray) -> float:
        """
        计算凸性
        
        Args:
            contour: 轮廓
            
        Returns:
            float: 凸性值
        """
        try:
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                return area / hull_area
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating convexity: {str(e)}")
            return 0.0
    
    def _create_default_features(self) -> Dict[str, float]:
        """
        创建默认特征
        
        Returns:
            Dict: 默认特征字典
        """
        return {
            'circularity': 0.0,
            'elongation': 1.0,
            'compactness': float('inf'),
            'convexity': 0.0,
            'area': 0.0,
            'perimeter': 0.0,
            'aspect_ratio': 1.0,
            'min_aspect_ratio': 1.0,
            'fourier_descriptors': [0.0] * self.num_descriptors
        }
    
    def compare_shapes(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> float:
        """
        比较两个形状的相似性
        
        Args:
            descriptors1: 第一个形状的傅里叶描述子
            descriptors2: 第二个形状的傅里叶描述子
            
        Returns:
            float: 相似性得分（0-1，1表示完全相似）
        """
        try:
            # 确保描述子长度一致
            min_len = min(len(descriptors1), len(descriptors2))
            desc1 = descriptors1[:min_len]
            desc2 = descriptors2[:min_len]
            
            # 计算欧氏距离
            distance = np.linalg.norm(desc1 - desc2)
            
            # 转换为相似性得分
            similarity = np.exp(-distance)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error comparing shapes: {str(e)}")
            return 0.0