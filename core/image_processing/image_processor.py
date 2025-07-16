# -*- coding: utf-8 -*-
"""
图像处理模块

实现图像预处理、分析和增强功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path
from scipy import ndimage
import os
from PIL import Image
import io

class ImageProcessor:
    """
    图像处理器
    
    提供图像预处理、分析和增强功能
    """
    
    def __init__(self, config):
        """
        初始化图像处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 预处理参数
        preprocess_config = config['image_processor']['preprocess']
        self.target_size = preprocess_config.get('target_size', (800, 800))
        self.blur_kernel_size = preprocess_config.get('blur_kernel_size', 3)
        self.blur_sigma = preprocess_config.get('blur_sigma', 1.0)
        
        # 增强参数
        enhance_config = config['image_processor']['enhance']
        self.contrast_limit = enhance_config.get('contrast_limit', 3.0)
        self.tile_grid_size = enhance_config.get('tile_grid_size', (8, 8))
        
        # 分析参数
        analysis_config = config['image_processor']['analysis']
        self.canny_low = analysis_config.get('canny_low_threshold', 50)
        self.canny_high = analysis_config.get('canny_high_threshold', 150)
        self.hough_threshold = analysis_config.get('hough_threshold', 50)
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            np.ndarray: 图像数据，如果失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # 获取文件扩展名
            file_ext = os.path.splitext(image_path)[1].lower()
            
            # 处理SVG文件
            if file_ext == '.svg':
                return self._load_svg_image(image_path)
            
            # 处理其他格式
            image = cv2.imread(image_path)
            if image is None:
                # 尝试使用PIL加载
                try:
                    pil_image = Image.open(image_path)
                    # 转换为RGB格式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    # 转换为numpy数组并调整颜色通道顺序（PIL使用RGB，OpenCV使用BGR）
                    image = np.array(pil_image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                except Exception as pil_e:
                    self.logger.error(f"Failed to load image with both OpenCV and PIL: {image_path}, PIL error: {str(pil_e)}")
                    return None
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def _load_svg_image(self, svg_path: str) -> Optional[np.ndarray]:
        """
        加载SVG图像
        
        Args:
            svg_path (str): SVG文件路径
            
        Returns:
            np.ndarray: 图像数据，如果失败则返回None
        """
        try:
            # 尝试导入cairosvg
            try:
                import cairosvg
            except ImportError:
                self.logger.error("cairosvg not installed. Please install it with: pip install cairosvg")
                return None
            
            # 将SVG转换为PNG字节流
            png_data = cairosvg.svg2png(url=svg_path)
            
            # 使用PIL从字节流加载图像
            pil_image = Image.open(io.BytesIO(png_data))
            
            # 转换为RGB格式（如果是RGBA，去除alpha通道）
            if pil_image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])  # 使用alpha通道作为mask
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组并调整颜色通道顺序
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading SVG image {svg_path}: {str(e)}")
            return None
     
     def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
         """
         转换为灰度图像
         
         Args:
             image (np.ndarray): 输入图像
             
         Returns:
             np.ndarray: 灰度图像
         """
         try:
             if len(image.shape) == 3:
                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
             else:
                 gray = image.copy()
             return gray
         except Exception as e:
             self.logger.error(f"Error converting to grayscale: {str(e)}")
             return image
     
     def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        try:
            # 调整大小
            processed = self._resize_image(image)
            
            # 转换为灰度图
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            if self.blur_kernel_size > 0:
                processed = cv2.GaussianBlur(
                    processed,
                    (self.blur_kernel_size, self.blur_kernel_size),
                    self.blur_sigma
                )
            
            # 对比度增强
            processed = self._enhance_contrast(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像特征
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            Dict: 分析结果
        """
        try:
            results = {}
            
            # 边缘检测
            edges = cv2.Canny(image, self.canny_low, self.canny_high)
            results['edges'] = edges
            
            # 计算梯度
            gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_direction = np.arctan2(gradient_y, gradient_x)
            
            results['gradient_magnitude'] = gradient_magnitude
            results['gradient_direction'] = gradient_direction
            
            # 计算纹理特征
            texture_features = self._compute_texture_features(image)
            results.update(texture_features)
            
            # 区域分析
            region_features = self._analyze_regions(image)
            results.update(region_features)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return {}
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像质量
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 增强后的图像
        """
        try:
            # 对比度增强
            enhanced = self._enhance_contrast(image)
            
            # 锐化
            enhanced = self._sharpen_image(enhanced)
            
            # 去噪
            enhanced = self._denoise_image(enhanced)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 调整后的图像
        """
        try:
            height, width = image.shape[:2]
            target_width, target_height = self.target_size
            
            # 计算缩放比例
            scale = min(target_width/width, target_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整大小
            resized = cv2.resize(image, (new_width, new_height))
            
            return resized
            
        except Exception as e:
            self.logger.error(f"Error resizing image: {str(e)}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像对比度
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 增强后的图像
        """
        try:
            # 创建CLAHE对象
            clahe = cv2.createCLAHE(
                clipLimit=self.contrast_limit,
                tileGridSize=self.tile_grid_size
            )
            
            # 应用CLAHE
            if len(image.shape) == 2:
                enhanced = clahe.apply(image)
            else:
                # 对每个通道分别增强
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    enhanced[:,:,i] = clahe.apply(image[:,:,i])
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing contrast: {str(e)}")
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        锐化图像
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 锐化后的图像
        """
        try:
            # 创建锐化核
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            
            # 应用锐化
            sharpened = cv2.filter2D(image, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Error sharpening image: {str(e)}")
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像去噪
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 去噪后的图像
        """
        try:
            # 非局部均值去噪
            if len(image.shape) == 2:
                denoised = cv2.fastNlMeansDenoising(image)
            else:
                denoised = cv2.fastNlMeansDenoisingColored(image)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error denoising image: {str(e)}")
            return image
    
    def _compute_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算纹理特征
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            Dict: 纹理特征
        """
        try:
            features = {}
            
            # 计算灰度共生矩阵
            if len(image.shape) == 2:
                gray = image
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算局部二值模式
            radius = 1
            n_points = 8 * radius
            lbp = self._local_binary_pattern(gray, n_points, radius)
            features['lbp'] = lbp
            
            # 计算Gabor特征
            gabor_features = self._gabor_features(gray)
            features['gabor'] = gabor_features
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error computing texture features: {str(e)}")
            return {}
    
    def _local_binary_pattern(self, image: np.ndarray, n_points: int, radius: int) -> np.ndarray:
        """
        计算局部二值模式
        
        Args:
            image (np.ndarray): 输入图像
            n_points (int): 采样点数
            radius (int): 半径
            
        Returns:
            np.ndarray: LBP特征图
        """
        try:
            # 计算采样点坐标
            angles = 2 * np.pi * np.arange(n_points) / n_points
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            
            # 创建输出图像
            height, width = image.shape
            lbp = np.zeros_like(image, dtype=np.uint8)
            
            # 计算LBP值
            for i in range(radius, height-radius):
                for j in range(radius, width-radius):
                    pattern = 0
                    center = image[i, j]
                    
                    for k in range(n_points):
                        # 计算采样点坐标
                        sample_x = j + int(round(x[k]))
                        sample_y = i + int(round(y[k]))
                        
                        # 比较像素值
                        if image[sample_y, sample_x] >= center:
                            pattern |= (1 << k)
                    
                    lbp[i, j] = pattern
            
            return lbp
            
        except Exception as e:
            self.logger.error(f"Error computing LBP: {str(e)}")
            return np.zeros_like(image)
    
    def _gabor_features(self, image: np.ndarray) -> np.ndarray:
        """
        计算Gabor特征
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: Gabor特征
        """
        try:
            # Gabor滤波器参数
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.2, 0.3, 0.4]
            
            # 创建特征图
            height, width = image.shape
            features = np.zeros((height, width, len(orientations)*len(frequencies)))
            
            # 应用Gabor滤波器
            for i, theta in enumerate(orientations):
                for j, frequency in enumerate(frequencies):
                    # 创建Gabor核
                    kernel = cv2.getGaborKernel(
                        (21, 21),
                        sigma=4.0,
                        theta=theta*np.pi/180.0,
                        lambd=1.0/frequency,
                        gamma=0.5,
                        psi=0
                    )
                    
                    # 应用滤波器
                    filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
                    features[:,:,i*len(frequencies)+j] = filtered
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error computing Gabor features: {str(e)}")
            return np.zeros((image.shape[0], image.shape[1], 1))
    
    def _analyze_regions(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像区域
        
        Args:
            image (np.ndarray): 输入图像
            
        Returns:
            Dict: 区域分析结果
        """
        try:
            results = {}
            
            # 二值化
            if len(image.shape) == 2:
                gray = image
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 连通区域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            # 提取区域特征
            regions = []
            for i in range(1, num_labels):  # 跳过背景
                region = {
                    'id': i,
                    'area': stats[i, cv2.CC_STAT_AREA],
                    'width': stats[i, cv2.CC_STAT_WIDTH],
                    'height': stats[i, cv2.CC_STAT_HEIGHT],
                    'centroid': centroids[i],
                    'bbox': stats[i, :4].tolist()
                }
                regions.append(region)
            
            results['regions'] = regions
            results['num_regions'] = num_labels - 1
            results['labels'] = labels
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing regions: {str(e)}")
            return {}