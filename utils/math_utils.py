# -*- coding: utf-8 -*-
"""
数学工具

提供几何计算、统计分析、数值计算等数学功能
包括向量运算、矩阵变换、插值、优化等
以及安全的数组操作和比较函数
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from scipy import interpolate, optimize, spatial, stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class MathUtils:
    """
    数学工具类
    
    提供基础数学计算功能
    """
    
    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """
        向量归一化
        
        Args:
            vector (np.ndarray): 输入向量
            
        Returns:
            np.ndarray: 归一化后的向量
        """
        try:
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return vector / norm
            
        except Exception as e:
            logging.error(f"Error normalizing vector: {str(e)}")
            return vector
    
    @staticmethod
    def calculate_angle(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
        """
        计算两个向量之间的夹角
        
        Args:
            v1 (np.ndarray): 向量1
            v2 (np.ndarray): 向量2
            degrees (bool): 是否返回角度制
            
        Returns:
            float: 夹角
        """
        try:
            # 归一化向量
            v1_norm = MathUtils.normalize_vector(v1)
            v2_norm = MathUtils.normalize_vector(v2)
            
            # 计算点积
            dot_product = np.dot(v1_norm, v2_norm)
            
            # 限制点积范围以避免数值误差
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # 计算角度
            angle = np.arccos(dot_product)
            
            if degrees:
                angle = np.degrees(angle)
            
            return float(angle)
            
        except Exception as e:
            logging.error(f"Error calculating angle: {str(e)}")
            return 0.0
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], center: Tuple[float, float], 
                    angle: float) -> Tuple[float, float]:
        """
        绕中心点旋转点
        
        Args:
            point (Tuple[float, float]): 待旋转的点
            center (Tuple[float, float]): 旋转中心
            angle (float): 旋转角度（弧度）
            
        Returns:
            Tuple[float, float]: 旋转后的点
        """
        try:
            # 平移到原点
            x = point[0] - center[0]
            y = point[1] - center[1]
            
            # 旋转
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            
            # 平移回原位置
            new_x += center[0]
            new_y += center[1]
            
            return (new_x, new_y)
            
        except Exception as e:
            logging.error(f"Error rotating point: {str(e)}")
            return point
    
    @staticmethod
    def interpolate_points(points: List[Tuple[float, float]], num_points: int, 
                          method: str = 'linear') -> List[Tuple[float, float]]:
        """
        点插值
        
        Args:
            points (List[Tuple[float, float]]): 输入点列表
            num_points (int): 插值后的点数
            method (str): 插值方法
            
        Returns:
            List[Tuple[float, float]]: 插值后的点列表
        """
        try:
            if len(points) < 2:
                return points
            
            # 提取x和y坐标
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 计算累积距离作为参数
            distances = [0]
            for i in range(1, len(points)):
                dist = math.sqrt((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)
                distances.append(distances[-1] + dist)
            
            # 归一化距离
            total_distance = distances[-1]
            if total_distance == 0:
                return points
            
            normalized_distances = [d / total_distance for d in distances]
            
            # 创建插值函数
            if method == 'linear':
                f_x = interpolate.interp1d(normalized_distances, x_coords, kind='linear')
                f_y = interpolate.interp1d(normalized_distances, y_coords, kind='linear')
            elif method == 'cubic':
                if len(points) >= 4:
                    f_x = interpolate.interp1d(normalized_distances, x_coords, kind='cubic')
                    f_y = interpolate.interp1d(normalized_distances, y_coords, kind='cubic')
                else:
                    f_x = interpolate.interp1d(normalized_distances, x_coords, kind='linear')
                    f_y = interpolate.interp1d(normalized_distances, y_coords, kind='linear')
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            # 生成新的参数值
            new_params = np.linspace(0, 1, num_points)
            
            # 插值
            new_x = f_x(new_params)
            new_y = f_y(new_params)
            
            # 返回点列表
            interpolated_points = [(float(x), float(y)) for x, y in zip(new_x, new_y)]
            
            return interpolated_points
            
        except Exception as e:
            logging.error(f"Error interpolating points: {str(e)}")
            return points
    
    @staticmethod
    def smooth_curve(points: List[Tuple[float, float]], window_size: int = 5, 
                    method: str = 'moving_average') -> List[Tuple[float, float]]:
        """
        曲线平滑
        
        Args:
            points (List[Tuple[float, float]]): 输入点列表
            window_size (int): 窗口大小
            method (str): 平滑方法
            
        Returns:
            List[Tuple[float, float]]: 平滑后的点列表
        """
        try:
            if len(points) < window_size:
                return points
            
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            if method == 'moving_average':
                # 移动平均
                smoothed_x = np.convolve(x_coords, np.ones(window_size)/window_size, mode='same')
                smoothed_y = np.convolve(y_coords, np.ones(window_size)/window_size, mode='same')
            
            elif method == 'gaussian':
                # 高斯平滑
                from scipy.ndimage import gaussian_filter1d
                sigma = window_size / 3.0
                smoothed_x = gaussian_filter1d(x_coords, sigma)
                smoothed_y = gaussian_filter1d(y_coords, sigma)
            
            elif method == 'savgol':
                # Savitzky-Golay滤波
                from scipy.signal import savgol_filter
                if window_size % 2 == 0:
                    window_size += 1  # 确保窗口大小为奇数
                poly_order = min(3, window_size - 1)
                smoothed_x = savgol_filter(x_coords, window_size, poly_order)
                smoothed_y = savgol_filter(y_coords, window_size, poly_order)
            
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
            
            # 返回平滑后的点列表
            smoothed_points = [(float(x), float(y)) for x, y in zip(smoothed_x, smoothed_y)]
            
            return smoothed_points
            
        except Exception as e:
            logging.error(f"Error smoothing curve: {str(e)}")
            return points
    
    @staticmethod
    def calculate_curvature(points: List[Tuple[float, float]]) -> List[float]:
        """
        计算曲线曲率
        
        Args:
            points (List[Tuple[float, float]]): 输入点列表
            
        Returns:
            List[float]: 曲率列表
        """
        try:
            if len(points) < 3:
                return [0.0] * len(points)
            
            curvatures = []
            
            for i in range(len(points)):
                if i == 0 or i == len(points) - 1:
                    curvatures.append(0.0)
                    continue
                
                # 取三个连续点
                p1 = points[i-1]
                p2 = points[i]
                p3 = points[i+1]
                
                # 计算向量
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # 计算叉积和点积
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                
                # 计算向量长度
                len_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
                len_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if len_v1 == 0 or len_v2 == 0:
                    curvatures.append(0.0)
                    continue
                
                # 计算曲率
                curvature = abs(cross_product) / (len_v1 * len_v2 * (len_v1 + len_v2))
                curvatures.append(curvature)
            
            return curvatures
            
        except Exception as e:
            logging.error(f"Error calculating curvature: {str(e)}")
            return [0.0] * len(points)
    
    @staticmethod
    def fit_polynomial(points: List[Tuple[float, float]], degree: int = 3) -> np.ndarray:
        """
        多项式拟合
        
        Args:
            points (List[Tuple[float, float]]): 输入点列表
            degree (int): 多项式次数
            
        Returns:
            np.ndarray: 多项式系数
        """
        try:
            if len(points) < degree + 1:
                return np.array([])
            
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # 多项式拟合
            coefficients = np.polyfit(x_coords, y_coords, degree)
            
            return coefficients
            
        except Exception as e:
            logging.error(f"Error fitting polynomial: {str(e)}")
            return np.array([])


class GeometryUtils:
    """
    几何工具类
    
    提供几何计算功能
    """
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """
        计算点到直线的距离
        
        Args:
            point (Tuple[float, float]): 点坐标
            line_start (Tuple[float, float]): 直线起点
            line_end (Tuple[float, float]): 直线终点
            
        Returns:
            float: 距离
        """
        try:
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end
            
            # 直线长度
            line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if line_length == 0:
                # 如果直线退化为点，返回点到点的距离
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # 计算点到直线的距离
            distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_length
            
            return distance
            
        except Exception as e:
            logging.error(f"Error calculating point to line distance: {str(e)}")
            return 0.0
    
    @staticmethod
    def point_to_segment_distance(point: Tuple[float, float], 
                                 segment_start: Tuple[float, float], 
                                 segment_end: Tuple[float, float]) -> float:
        """
        计算点到线段的距离
        
        Args:
            point (Tuple[float, float]): 点坐标
            segment_start (Tuple[float, float]): 线段起点
            segment_end (Tuple[float, float]): 线段终点
            
        Returns:
            float: 距离
        """
        try:
            x0, y0 = point
            x1, y1 = segment_start
            x2, y2 = segment_end
            
            # 线段长度的平方
            segment_length_sq = (x2 - x1)**2 + (y2 - y1)**2
            
            if segment_length_sq == 0:
                # 如果线段退化为点，返回点到点的距离
                return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # 计算投影参数
            t = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / segment_length_sq
            
            # 限制t在[0, 1]范围内
            t = max(0, min(1, t))
            
            # 计算投影点
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            # 计算距离
            distance = math.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)
            
            return distance
            
        except Exception as e:
            logging.error(f"Error calculating point to segment distance: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
        """
        计算多边形面积（使用鞋带公式）
        
        Args:
            points (List[Tuple[float, float]]): 多边形顶点
            
        Returns:
            float: 面积
        """
        try:
            if len(points) < 3:
                return 0.0
            
            area = 0.0
            n = len(points)
            
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            
            return abs(area) / 2.0
            
        except Exception as e:
            logging.error(f"Error calculating polygon area: {str(e)}")
            return 0.0
    
    @staticmethod
    def calculate_polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        计算多边形质心
        
        Args:
            points (List[Tuple[float, float]]): 多边形顶点
            
        Returns:
            Tuple[float, float]: 质心坐标
        """
        try:
            if len(points) < 3:
                if len(points) == 1:
                    return points[0]
                elif len(points) == 2:
                    return ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
                else:
                    return (0.0, 0.0)
            
            area = GeometryUtils.calculate_polygon_area(points)
            if area == 0:
                # 如果面积为0，返回顶点的平均值
                avg_x = sum(p[0] for p in points) / len(points)
                avg_y = sum(p[1] for p in points) / len(points)
                return (avg_x, avg_y)
            
            cx = 0.0
            cy = 0.0
            n = len(points)
            
            for i in range(n):
                j = (i + 1) % n
                factor = points[i][0] * points[j][1] - points[j][0] * points[i][1]
                cx += (points[i][0] + points[j][0]) * factor
                cy += (points[i][1] + points[j][1]) * factor
            
            cx /= (6.0 * area)
            cy /= (6.0 * area)
            
            return (cx, cy)
            
        except Exception as e:
            logging.error(f"Error calculating polygon centroid: {str(e)}")
            return (0.0, 0.0)
    
    @staticmethod
    def point_in_polygon(point: Tuple[float, float], 
                        polygon: List[Tuple[float, float]]) -> bool:
        """
        判断点是否在多边形内（射线法）
        
        Args:
            point (Tuple[float, float]): 点坐标
            polygon (List[Tuple[float, float]]): 多边形顶点
            
        Returns:
            bool: 是否在多边形内
        """
        try:
            x, y = point
            n = len(polygon)
            inside = False
            
            p1x, p1y = polygon[0]
            for i in range(1, n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
            
        except Exception as e:
            logging.error(f"Error checking point in polygon: {str(e)}")
            return False
    
    @staticmethod
    def calculate_bounding_box(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        计算点集的边界框
        
        Args:
            points (List[Tuple[float, float]]): 点列表
            
        Returns:
            Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y)
        """
        try:
            if not points:
                return (0.0, 0.0, 0.0, 0.0)
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            return (min_x, min_y, max_x, max_y)
            
        except Exception as e:
            logging.error(f"Error calculating bounding box: {str(e)}")
            return (0.0, 0.0, 0.0, 0.0)
    
    @staticmethod
    def line_intersection(line1_start: Tuple[float, float], line1_end: Tuple[float, float],
                         line2_start: Tuple[float, float], line2_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        计算两条直线的交点
        
        Args:
            line1_start (Tuple[float, float]): 直线1起点
            line1_end (Tuple[float, float]): 直线1终点
            line2_start (Tuple[float, float]): 直线2起点
            line2_end (Tuple[float, float]): 直线2终点
            
        Returns:
            Optional[Tuple[float, float]]: 交点坐标，如果平行则返回None
        """
        try:
            x1, y1 = line1_start
            x2, y2 = line1_end
            x3, y3 = line2_start
            x4, y4 = line2_end
            
            # 计算分母
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if abs(denom) < 1e-10:
                # 直线平行
                return None
            
            # 计算交点
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            
            return (intersection_x, intersection_y)
            
        except Exception as e:
            logging.error(f"Error calculating line intersection: {str(e)}")
            return None


class StatisticsUtils:
    """
    统计工具类
    
    提供统计分析功能
    """
    
    @staticmethod
    def calculate_basic_statistics(data: List[float]) -> Dict[str, float]:
        """
        计算基础统计量
        
        Args:
            data (List[float]): 数据列表
            
        Returns:
            Dict[str, float]: 统计量字典
        """
        try:
            if not data:
                return {}
            
            data_array = np.array(data)
            
            statistics = {
                'count': len(data),
                'mean': float(np.mean(data_array)),
                'median': float(np.median(data_array)),
                'std': float(np.std(data_array)),
                'var': float(np.var(data_array)),
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'range': float(np.max(data_array) - np.min(data_array)),
                'q25': float(np.percentile(data_array, 25)),
                'q75': float(np.percentile(data_array, 75)),
                'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25))
            }
            
            # 偏度和峰度
            if len(data) > 2:
                statistics['skewness'] = float(stats.skew(data_array))
                statistics['kurtosis'] = float(stats.kurtosis(data_array))
            
            return statistics
            
        except Exception as e:
            logging.error(f"Error calculating basic statistics: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
        """
        计算相关性
        
        Args:
            x (List[float]): 数据x
            y (List[float]): 数据y
            
        Returns:
            Dict[str, float]: 相关性结果
        """
        try:
            if len(x) != len(y) or len(x) < 2:
                return {}
            
            x_array = np.array(x)
            y_array = np.array(y)
            
            # Pearson相关系数
            pearson_corr, pearson_p = stats.pearsonr(x_array, y_array)
            
            # Spearman相关系数
            spearman_corr, spearman_p = stats.spearmanr(x_array, y_array)
            
            correlation = {
                'pearson_correlation': float(pearson_corr),
                'pearson_p_value': float(pearson_p),
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p)
            }
            
            return correlation
            
        except Exception as e:
            logging.error(f"Error calculating correlation: {str(e)}")
            return {}
    
    @staticmethod
    def detect_outliers(data: List[float], method: str = 'iqr', 
                       threshold: float = 1.5) -> List[int]:
        """
        异常值检测
        
        Args:
            data (List[float]): 数据列表
            method (str): 检测方法 ('iqr', 'zscore')
            threshold (float): 阈值
            
        Returns:
            List[int]: 异常值索引列表
        """
        try:
            if not data:
                return []
            
            data_array = np.array(data)
            outlier_indices = []
            
            if method == 'iqr':
                q25 = np.percentile(data_array, 25)
                q75 = np.percentile(data_array, 75)
                iqr = q75 - q25
                
                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr
                
                outlier_indices = [i for i, value in enumerate(data) 
                                 if value < lower_bound or value > upper_bound]
            
            elif method == 'zscore':
                mean = np.mean(data_array)
                std = np.std(data_array)
                
                if std > 0:
                    z_scores = np.abs((data_array - mean) / std)
                    outlier_indices = [i for i, z_score in enumerate(z_scores) 
                                     if z_score > threshold]
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            return outlier_indices
            
        except Exception as e:
            logging.error(f"Error detecting outliers: {str(e)}")
            return []
    
    @staticmethod
    def cluster_data(data: np.ndarray, method: str = 'kmeans', 
                    n_clusters: int = 3, **kwargs) -> Dict[str, Any]:
        """
        数据聚类
        
        Args:
            data (np.ndarray): 数据矩阵
            method (str): 聚类方法
            n_clusters (int): 聚类数量
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 聚类结果
        """
        try:
            if len(data) == 0:
                return {}
            
            if method == 'kmeans':
                random_state = kwargs.get('random_state', 42)
                kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(data)
                centers = kmeans.cluster_centers_
                
                result = {
                    'labels': labels.tolist(),
                    'centers': centers.tolist(),
                    'inertia': float(kmeans.inertia_)
                }
            
            elif method == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)
                
                result = {
                    'labels': labels.tolist(),
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'n_noise': list(labels).count(-1)
                }
            
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error clustering data: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_distance_matrix(points: List[Tuple[float, float]], 
                                 metric: str = 'euclidean') -> np.ndarray:
        """
        计算距离矩阵
        
        Args:
            points (List[Tuple[float, float]]): 点列表
            metric (str): 距离度量
            
        Returns:
            np.ndarray: 距离矩阵
        """
        try:
            if not points:
                return np.array([])
            
            points_array = np.array(points)
            distance_matrix = cdist(points_array, points_array, metric=metric)
            
            return distance_matrix
            
        except Exception as e:
            logging.error(f"Error calculating distance matrix: {str(e)}")
            return np.array([])
    
    @staticmethod
    def perform_pca(data: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
        """
        主成分分析
        
        Args:
            data (np.ndarray): 数据矩阵
            n_components (int): 主成分数量
            
        Returns:
            Dict[str, Any]: PCA结果
        """
        try:
            if len(data) == 0:
                return {}
            
            # 标准化数据
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # PCA
            pca = PCA(n_components=n_components)
            data_pca = pca.fit_transform(data_scaled)
            
            result = {
                'transformed_data': data_pca.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca.components_.tolist(),
                'n_components': n_components
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error performing PCA: {str(e)}")
            return {}


def ensure_scalar(value, default=0.0):
    """
    将值转换为标量，处理numpy数组和其他类型
    
    Args:
        value: 输入值，可能是numpy数组或其他类型
        default (float): 默认值，当输入为None或空数组时返回
        
    Returns:
        float: 转换后的标量值
    """
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        elif value.size == 1:
            return value.item()
        else:
            return float(value.mean())
    return float(value)


def safe_compare(a, b):
    """
    安全比较两个值，避免数组真值歧义
    
    Args:
        a: 第一个值，可能是numpy数组或其他类型
        b: 第二个值，可能是numpy数组或其他类型
        
    Returns:
        bool: 是否相等
    """
    try:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return False
        else:
            return a == b
    except (ValueError, TypeError):
        return False