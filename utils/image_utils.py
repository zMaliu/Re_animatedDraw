# -*- coding: utf-8 -*-
"""
图像处理工具

提供图像预处理、增强和分析功能
包括滤波、变换、特征提取等操作
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import logging
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    # 兼容不同版本的scikit-image
    try:
        from scipy.ndimage import maximum_filter
        from scipy.ndimage import generate_binary_structure
        def peak_local_maxima(image, min_distance=1, threshold_abs=None, threshold_rel=None):
            """简化的局部最大值检测实现"""
            if threshold_abs is None:
                threshold_abs = 0
            if threshold_rel is None:
                threshold_rel = 0
            
            # 使用形态学操作检测局部最大值
            neighborhood = generate_binary_structure(2, 2)
            local_maxima = maximum_filter(image, footprint=neighborhood) == image
            
            # 应用阈值
            if threshold_abs > 0:
                local_maxima = local_maxima & (image > threshold_abs)
            if threshold_rel > 0:
                threshold = threshold_rel * image.max()
                local_maxima = local_maxima & (image > threshold)
            
            # 返回坐标
            coords = np.where(local_maxima)
            return np.column_stack(coords)
    except ImportError:
        def peak_local_maxima(*args, **kwargs):
            """占位函数"""
            return np.array([])
            
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter


class ImageProcessor:
    """
    图像处理器
    
    提供基础的图像处理功能
    """
    
    def __init__(self, config=None):
        """
        初始化图像处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def load_image(image_path: str, color_mode: str = 'RGB') -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path (str): 图像路径
            color_mode (str): 颜色模式 ('RGB', 'BGR', 'GRAY')
            
        Returns:
            Optional[np.ndarray]: 图像数组
        """
        try:
            if color_mode == 'GRAY':
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if color_mode == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
        """
        保存图像
        
        Args:
            image (np.ndarray): 图像数组
            output_path (str): 输出路径
            quality (int): 图像质量
            
        Returns:
            bool: 是否成功
        """
        try:
            # 确保图像数据类型正确
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # 转换颜色空间（如果需要）
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 保存图像
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return True
            
        except Exception as e:
            logging.error(f"Error saving image to {output_path}: {str(e)}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: str = 'bilinear') -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image (np.ndarray): 输入图像
            target_size (Tuple[int, int]): 目标大小 (width, height)
            interpolation (str): 插值方法
            
        Returns:
            np.ndarray: 调整后的图像
        """
        try:
            height, width = target_size[1], target_size[0]
            
            if interpolation == 'nearest':
                method = cv2.INTER_NEAREST
            elif interpolation == 'bilinear':
                method = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                method = cv2.INTER_CUBIC
            else:
                method = cv2.INTER_LINEAR
            
            resized = cv2.resize(image, (width, height), interpolation=method)
            return resized
            
        except Exception as e:
            logging.error(f"Error resizing image: {str(e)}")
            return image
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        转换为灰度图像
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 灰度图像
        """
        try:
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 4:
                    # RGBA to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            return gray
            
        except Exception as e:
            logging.error(f"Error converting to grayscale: {str(e)}")
            return image
    
    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, 
                           sigma: float = 1.0) -> np.ndarray:
        """
        应用高斯模糊
        
        Args:
            image (np.ndarray): 输入图像
            kernel_size (int): 核大小
            sigma (float): 标准差
            
        Returns:
            np.ndarray: 模糊后的图像
        """
        try:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            return blurred
            
        except Exception as e:
            logging.error(f"Error applying Gaussian blur: {str(e)}")
            return image
    
    @staticmethod
    def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        应用中值滤波
        
        Args:
            image (np.ndarray): 输入图像
            kernel_size (int): 核大小
            
        Returns:
            np.ndarray: 滤波后的图像
        """
        try:
            filtered = cv2.medianBlur(image, kernel_size)
            return filtered
            
        except Exception as e:
            logging.error(f"Error applying median filter: {str(e)}")
            return image
    
    @staticmethod
    def detect_edges(image: np.ndarray, method: str = 'canny', 
                    **kwargs) -> np.ndarray:
        """
        边缘检测
        
        Args:
            image (np.ndarray): 输入图像
            method (str): 检测方法 ('canny', 'sobel', 'laplacian')
            **kwargs: 方法参数
            
        Returns:
            np.ndarray: 边缘图像
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = ImageProcessor.convert_to_grayscale(image)
            else:
                gray = image
            
            if method == 'canny':
                low_threshold = kwargs.get('low_threshold', 50)
                high_threshold = kwargs.get('high_threshold', 150)
                edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            elif method == 'sobel':
                ksize = kwargs.get('ksize', 3)
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                edges = np.sqrt(sobel_x**2 + sobel_y**2)
                edges = (edges / edges.max() * 255).astype(np.uint8)
            
            elif method == 'laplacian':
                ksize = kwargs.get('ksize', 3)
                edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                edges = np.abs(edges)
                edges = (edges / edges.max() * 255).astype(np.uint8)
            
            else:
                raise ValueError(f"Unknown edge detection method: {method}")
            
            return edges
            
        except Exception as e:
            logging.error(f"Error detecting edges: {str(e)}")
            return np.zeros_like(image)
    
    @staticmethod
    def threshold_image(image: np.ndarray, threshold_value: float = 127, 
                       method: str = 'binary') -> np.ndarray:
        """
        图像阈值化
        
        Args:
            image (np.ndarray): 输入图像
            threshold_value (float): 阈值
            method (str): 阈值化方法
            
        Returns:
            np.ndarray: 阈值化后的图像
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = ImageProcessor.convert_to_grayscale(image)
            else:
                gray = image
            
            if method == 'binary':
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            elif method == 'binary_inv':
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            elif method == 'otsu':
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'adaptive':
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            else:
                raise ValueError(f"Unknown threshold method: {method}")
            
            return thresh
            
        except Exception as e:
            logging.error(f"Error thresholding image: {str(e)}")
            return image
    
    @staticmethod
    def morphological_operation(image: np.ndarray, operation: str, 
                               kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        形态学操作
        
        Args:
            image (np.ndarray): 输入图像
            operation (str): 操作类型 ('erosion', 'dilation', 'opening', 'closing')
            kernel_size (int): 核大小
            iterations (int): 迭代次数
            
        Returns:
            np.ndarray: 处理后的图像
        """
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            if operation == 'erosion':
                result = cv2.erode(image, kernel, iterations=iterations)
            elif operation == 'dilation':
                result = cv2.dilate(image, kernel, iterations=iterations)
            elif operation == 'opening':
                result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif operation == 'closing':
                result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            else:
                raise ValueError(f"Unknown morphological operation: {operation}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in morphological operation: {str(e)}")
            return image


class ImageEnhancer:
    """
    图像增强器
    
    提供图像增强和改善功能
    """
    
    def __init__(self, config=None):
        """
        初始化图像增强器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整亮度
        
        Args:
            image (np.ndarray): 输入图像
            factor (float): 亮度因子
            
        Returns:
            np.ndarray: 调整后的图像
        """
        try:
            enhanced = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            return enhanced
            
        except Exception as e:
            logging.error(f"Error adjusting brightness: {str(e)}")
            return image
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        调整对比度
        
        Args:
            image (np.ndarray): 输入图像
            factor (float): 对比度因子
            
        Returns:
            np.ndarray: 调整后的图像
        """
        try:
            enhanced = cv2.convertScaleAbs(image, alpha=factor, beta=0)
            return enhanced
            
        except Exception as e:
            logging.error(f"Error adjusting contrast: {str(e)}")
            return image
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        直方图均衡化
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 均衡化后的图像
        """
        try:
            if len(image.shape) == 3:
                # 彩色图像
                yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                # 灰度图像
                enhanced = cv2.equalizeHist(image)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in histogram equalization: {str(e)}")
            return image
    
    @staticmethod
    def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, 
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        CLAHE（对比度限制自适应直方图均衡化）增强
        
        Args:
            image (np.ndarray): 输入图像
            clip_limit (float): 对比度限制
            tile_grid_size (Tuple[int, int]): 瓦片网格大小
            
        Returns:
            np.ndarray: 增强后的图像
        """
        try:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            if len(image.shape) == 3:
                # 彩色图像
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # 灰度图像
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error in CLAHE enhancement: {str(e)}")
            return image
    
    @staticmethod
    def unsharp_masking(image: np.ndarray, sigma: float = 1.0, 
                       strength: float = 1.0) -> np.ndarray:
        """
        锐化掩模
        
        Args:
            image (np.ndarray): 输入图像
            sigma (float): 高斯模糊标准差
            strength (float): 锐化强度
            
        Returns:
            np.ndarray: 锐化后的图像
        """
        try:
            # 高斯模糊
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # 锐化掩模
            sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            
            return sharpened
            
        except Exception as e:
            logging.error(f"Error in unsharp masking: {str(e)}")
            return image
    
    @staticmethod
    def denoise_image(image: np.ndarray, method: str = 'bilateral', 
                     **kwargs) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image (np.ndarray): 输入图像
            method (str): 去噪方法
            **kwargs: 方法参数
            
        Returns:
            np.ndarray: 去噪后的图像
        """
        try:
            if method == 'bilateral':
                d = kwargs.get('d', 9)
                sigma_color = kwargs.get('sigma_color', 75)
                sigma_space = kwargs.get('sigma_space', 75)
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            elif method == 'non_local_means':
                h = kwargs.get('h', 10)
                template_window_size = kwargs.get('template_window_size', 7)
                search_window_size = kwargs.get('search_window_size', 21)
                
                if len(image.shape) == 3:
                    denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 
                                                             template_window_size, search_window_size)
                else:
                    denoised = cv2.fastNlMeansDenoising(image, None, h, 
                                                       template_window_size, search_window_size)
            
            elif method == 'gaussian':
                kernel_size = kwargs.get('kernel_size', 5)
                sigma = kwargs.get('sigma', 1.0)
                denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            else:
                raise ValueError(f"Unknown denoising method: {method}")
            
            return denoised
            
        except Exception as e:
            logging.error(f"Error denoising image: {str(e)}")
            return image


class ImageAnalyzer:
    """
    图像分析器
    
    提供图像分析和特征提取功能
    """
    
    def __init__(self, config=None):
        """
        初始化图像分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def calculate_histogram(image: np.ndarray, bins: int = 256) -> Dict[str, np.ndarray]:
        """
        计算图像直方图
        
        Args:
            image (np.ndarray): 输入图像
            bins (int): 直方图箱数
            
        Returns:
            Dict[str, np.ndarray]: 直方图数据
        """
        try:
            histograms = {}
            
            if len(image.shape) == 3:
                # 彩色图像
                colors = ['red', 'green', 'blue']
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
                    histograms[color] = hist.flatten()
            else:
                # 灰度图像
                hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
                histograms['gray'] = hist.flatten()
            
            return histograms
            
        except Exception as e:
            logging.error(f"Error calculating histogram: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_image_statistics(image: np.ndarray) -> Dict[str, Any]:
        """
        计算图像统计信息
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            stats = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min': float(np.min(image)),
                'max': float(np.max(image)),
                'mean': float(np.mean(image)),
                'std': float(np.std(image)),
                'median': float(np.median(image))
            }
            
            if len(image.shape) == 3:
                # 彩色图像的通道统计
                for i, channel in enumerate(['red', 'green', 'blue']):
                    channel_data = image[:, :, i]
                    stats[f'{channel}_mean'] = float(np.mean(channel_data))
                    stats[f'{channel}_std'] = float(np.std(channel_data))
            
            return stats
            
        except Exception as e:
            logging.error(f"Error calculating image statistics: {str(e)}")
            return {}
    
    @staticmethod
    def find_connected_components(binary_image: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        查找连通组件
        
        Args:
            binary_image (np.ndarray): 二值图像
            
        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray]: 组件数量、标签、统计信息、质心
        """
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
            return num_labels, labels, stats, centroids
            
        except Exception as e:
            logging.error(f"Error finding connected components: {str(e)}")
            return 0, np.array([]), np.array([]), np.array([])
    
    @staticmethod
    def extract_contours(binary_image: np.ndarray) -> List[np.ndarray]:
        """
        提取轮廓
        
        Args:
            binary_image (np.ndarray): 二值图像
            
        Returns:
            List[np.ndarray]: 轮廓列表
        """
        try:
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
            
        except Exception as e:
            logging.error(f"Error extracting contours: {str(e)}")
            return []
    
    @staticmethod
    def calculate_contour_properties(contour: np.ndarray) -> Dict[str, Any]:
        """
        计算轮廓属性
        
        Args:
            contour (np.ndarray): 轮廓
            
        Returns:
            Dict[str, Any]: 轮廓属性
        """
        try:
            properties = {}
            
            # 基本属性
            properties['area'] = cv2.contourArea(contour)
            properties['perimeter'] = cv2.arcLength(contour, True)
            
            # 边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            properties['bounding_rect'] = (x, y, w, h)
            properties['aspect_ratio'] = float(w) / h if h > 0 else 0
            
            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            properties['min_area_rect'] = rect
            
            # 最小外接圆
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            properties['min_enclosing_circle'] = ((cx, cy), radius)
            
            # 质心
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                properties['centroid'] = (cx, cy)
            else:
                properties['centroid'] = (0, 0)
            
            # 凸包
            hull = cv2.convexHull(contour)
            properties['convex_hull'] = hull
            properties['hull_area'] = cv2.contourArea(hull)
            properties['solidity'] = properties['area'] / properties['hull_area'] if properties['hull_area'] > 0 else 0
            
            # 椭圆拟合
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                properties['fitted_ellipse'] = ellipse
            
            return properties
            
        except Exception as e:
            logging.error(f"Error calculating contour properties: {str(e)}")
            return {}
    
    @staticmethod
    def detect_corners(image: np.ndarray, method: str = 'harris', 
                      **kwargs) -> np.ndarray:
        """
        角点检测
        
        Args:
            image (np.ndarray): 输入图像
            method (str): 检测方法
            **kwargs: 方法参数
            
        Returns:
            np.ndarray: 角点坐标
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = ImageProcessor.convert_to_grayscale(image)
            else:
                gray = image
            
            if method == 'harris':
                block_size = kwargs.get('block_size', 2)
                ksize = kwargs.get('ksize', 3)
                k = kwargs.get('k', 0.04)
                threshold = kwargs.get('threshold', 0.01)
                
                harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
                corners = np.argwhere(harris_response > threshold * harris_response.max())
                corners = corners[:, [1, 0]]  # 转换为 (x, y) 格式
            
            elif method == 'shi_tomasi':
                max_corners = kwargs.get('max_corners', 100)
                quality_level = kwargs.get('quality_level', 0.01)
                min_distance = kwargs.get('min_distance', 10)
                
                corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
                if corners is not None:
                    corners = corners.reshape(-1, 2)
                else:
                    corners = np.array([])
            
            else:
                raise ValueError(f"Unknown corner detection method: {method}")
            
            return corners
            
        except Exception as e:
            logging.error(f"Error detecting corners: {str(e)}")
            return np.array([])
    
    @staticmethod
    def calculate_texture_features(image: np.ndarray, window_size: int = 5) -> Dict[str, float]:
        """
        计算纹理特征
        
        Args:
            image (np.ndarray): 输入图像
            window_size (int): 窗口大小
            
        Returns:
            Dict[str, float]: 纹理特征
        """
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = ImageProcessor.convert_to_grayscale(image)
            else:
                gray = image.astype(np.float32)
            
            features = {}
            
            # 局部二值模式 (LBP)
            from skimage.feature import local_binary_pattern
            radius = window_size // 2
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            features['lbp_mean'] = float(np.mean(lbp))
            features['lbp_std'] = float(np.std(lbp))
            
            # 梯度特征
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = float(np.mean(gradient_magnitude))
            features['gradient_std'] = float(np.std(gradient_magnitude))
            
            # 方差特征
            kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
            mean_filtered = cv2.filter2D(gray, -1, kernel)
            variance = cv2.filter2D((gray - mean_filtered)**2, -1, kernel)
            
            features['variance_mean'] = float(np.mean(variance))
            features['variance_std'] = float(np.std(variance))
            
            return features
            
        except Exception as e:
            logging.error(f"Error calculating texture features: {str(e)}")
            return {}