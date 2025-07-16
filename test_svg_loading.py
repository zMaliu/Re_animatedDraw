#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SVG图像加载功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.image_processing.image_processor import ImageProcessor
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_svg_loading():
    """
    测试SVG文件加载
    """
    # 创建ImageProcessor实例
    config = {
        'image_processor': {
            'preprocess': {
                'target_size': (800, 800),
                'blur_kernel_size': 3,
                'blur_sigma': 1.0
            },
            'enhance': {
                'contrast_limit': 3.0,
                'tile_grid_size': (8, 8)
            },
            'analysis': {
                'canny_low_threshold': 50,
                'canny_high_threshold': 150,
                'hough_threshold': 50
            }
        }
    }
    
    processor = ImageProcessor(config)
    
    # 测试SVG文件路径
    svg_path = "data/test.svg"
    
    if not os.path.exists(svg_path):
        logger.error(f"SVG文件不存在: {svg_path}")
        return False
    
    logger.info(f"尝试加载SVG文件: {svg_path}")
    
    # 加载SVG图像
    image = processor.load_image(svg_path)
    
    if image is None:
        logger.error("SVG图像加载失败")
        return False
    
    logger.info(f"SVG图像加载成功! 图像尺寸: {image.shape}")
    
    # 测试转换为灰度图
    gray_image = processor.convert_to_grayscale(image)
    logger.info(f"灰度图像转换成功! 图像尺寸: {gray_image.shape}")
    
    return True

if __name__ == "__main__":
    success = test_svg_loading()
    if success:
        print("✅ SVG加载测试通过!")
    else:
        print("❌ SVG加载测试失败!")
        sys.exit(1)