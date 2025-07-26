#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块1: 数据准备与笔触提取
实现从输入水墨画图像中分割出独立的笔触（stroke）

主要功能:
1. 图像预处理（去噪、二值化）
2. 笔触分割（边缘检测、连通域分析）
3. 笔触过滤（去除噪点）
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from skimage import morphology
import json

class Stroke:
    """笔触类，存储单个笔触的信息"""
    
    def __init__(self, stroke_id: int, mask: np.ndarray, contour: np.ndarray, 
                 bbox: Tuple[int, int, int, int]):
        self.id = stroke_id
        self.mask = mask  # 二值掩码
        self.contour = contour  # 轮廓点
        self.bbox = bbox  # 边界框 (x, y, w, h)
        self.area = cv2.contourArea(contour)
        self.perimeter = cv2.arcLength(contour, True)
        
        # 计算质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            self.centroid = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
    
    def get_skeleton_points(self) -> List[Tuple[int, int]]:
        """提取笔触骨架端点"""
        # 使用skimage的骨架化
        binary_mask = self.mask > 0
        skeleton = morphology.skeletonize(binary_mask)
        skeleton = skeleton.astype(np.uint8) * 255
        
        # Harris角点检测找端点
        corners = cv2.cornerHarris(skeleton.astype(np.float32), 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # 提取角点坐标
        corner_coords = np.where(corners > 0.01 * corners.max())
        skeleton_points = list(zip(corner_coords[1], corner_coords[0]))
        
        return skeleton_points if skeleton_points else [self.centroid]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'id': self.id,
            'area': float(self.area),
            'perimeter': float(self.perimeter),
            'centroid': self.centroid,
            'bbox': self.bbox
        }

class StrokeExtractor:
    """笔触提取器"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.original_image = None
        self.processed_image = None
        self.binary_image = None
        self.strokes = []
        
        # 参数设置
        self.min_area = 50  # 最小笔触面积
        self.max_area = 50000  # 最大笔触面积
        self.gaussian_kernel = (5, 5)  # 高斯滤波核大小
        self.canny_low = 50  # Canny低阈值
        self.canny_high = 150  # Canny高阈值
    
    def extract_strokes(self, image_path: str) -> List[Stroke]:
        """从图像中提取笔触"""
        # 1. 读取图像
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        print(f"图像尺寸: {self.original_image.shape}")
        
        # 2. 图像预处理
        self._preprocess_image()
        
        # 3. 笔触分割
        self._segment_strokes()
        
        # 4. 后处理和过滤
        self._filter_strokes()
        
        return self.strokes
    
    def _preprocess_image(self):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        
        # 自适应阈值二值化
        # 水墨画通常是深色笔触在浅色背景上
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学操作去除小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.processed_image = blurred
        self.binary_image = binary
        
        if self.debug:
            print("图像预处理完成")
    
    def _segment_strokes(self):
        """笔触分割"""
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.binary_image, connectivity=8
        )
        
        self.strokes = []
        
        # 遍历每个连通域（跳过背景标签0）
        for i in range(1, num_labels):
            # 获取连通域统计信息
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 面积过滤
            if area < self.min_area or area > self.max_area:
                continue
            
            # 创建笔触掩码
            mask = (labels == i).astype(np.uint8) * 255
            
            # 提取轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # 选择最大轮廓
            contour = max(contours, key=cv2.contourArea)
            
            # 创建笔触对象
            stroke = Stroke(len(self.strokes), mask, contour, (x, y, w, h))
            self.strokes.append(stroke)
        
        if self.debug:
            print(f"分割得到 {len(self.strokes)} 个候选笔触")
    
    def _filter_strokes(self):
        """笔触过滤和优化"""
        filtered_strokes = []
        
        for stroke in self.strokes:
            # 形状过滤：去除过于圆形的区域（可能是噪点）
            circularity = 4 * np.pi * stroke.area / (stroke.perimeter ** 2)
            if circularity > 0.9:  # 过于圆形
                continue
            
            # 长宽比过滤：去除过于方形的区域
            aspect_ratio = stroke.bbox[2] / stroke.bbox[3]
            if 0.2 < aspect_ratio < 5.0:  # 合理的长宽比范围
                filtered_strokes.append(stroke)
        
        self.strokes = filtered_strokes
        
        # 重新分配ID
        for i, stroke in enumerate(self.strokes):
            stroke.id = i
        
        if self.debug:
            print(f"过滤后剩余 {len(self.strokes)} 个笔触")
    
    def save_debug_images(self, output_dir: Path):
        """保存调试图像"""
        if not self.debug:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原图
        cv2.imwrite(str(output_dir / "01_original.jpg"), self.original_image)
        
        # 保存预处理结果
        cv2.imwrite(str(output_dir / "02_processed.jpg"), self.processed_image)
        cv2.imwrite(str(output_dir / "03_binary.jpg"), self.binary_image)
        
        # 保存笔触分割结果
        stroke_vis = self.original_image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 128, 128)]
        
        for i, stroke in enumerate(self.strokes):
            color = colors[i % len(colors)]
            cv2.drawContours(stroke_vis, [stroke.contour], -1, color, 2)
            cv2.putText(stroke_vis, str(stroke.id), stroke.centroid, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(output_dir / "04_strokes.jpg"), stroke_vis)
        
        # 保存笔触信息
        stroke_info = {
            'total_strokes': len(self.strokes),
            'strokes': [stroke.to_dict() for stroke in self.strokes]
        }
        
        with open(output_dir / "stroke_info.json", 'w', encoding='utf-8') as f:
            json.dump(stroke_info, f, indent=2, ensure_ascii=False)
        
        print(f"调试图像已保存至: {output_dir}")

# 测试代码
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python stroke_extraction.py <image_path>")
        sys.exit(1)
    
    extractor = StrokeExtractor(debug=True)
    strokes = extractor.extract_strokes(sys.argv[1])
    
    print(f"\n提取结果:")
    print(f"总笔触数: {len(strokes)}")
    for stroke in strokes[:5]:  # 显示前5个笔触信息
        print(f"笔触 {stroke.id}: 面积={stroke.area:.1f}, 周长={stroke.perimeter:.1f}")