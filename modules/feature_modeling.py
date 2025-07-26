#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块2: 笔触特征建模
量化每个笔触的几何、颜色、纹理特征，为后续排序提供依据

主要功能:
1. 骨架点与距离特征
2. 面积与尺度特征
3. 形状特征（傅里叶描述子、圆形度等）
4. 墨湿度与厚度特征
5. 颜色与位置显著性特征
"""

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt

from .stroke_extraction import Stroke

class FeatureModeler:
    """笔触特征建模器"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.features_df = None
        self.saliency_map = self._create_saliency_map()
    
    def _create_saliency_map(self, size: int = 300) -> np.ndarray:
        """创建位置显著性查找表（300x300）"""
        # 创建中心高、边缘低的高斯分布显著性图
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # 高斯分布，中心显著性最高
        saliency = np.exp(-(X**2 + Y**2) / 0.5)
        return saliency
    
    def extract_features(self, strokes: List[Stroke]) -> pd.DataFrame:
        """提取所有笔触的特征"""
        features_list = []
        
        for stroke in strokes:
            features = self._extract_single_stroke_features(stroke)
            features_list.append(features)
        
        # 创建DataFrame
        self.features_df = pd.DataFrame(features_list)
        
        # 特征归一化
        self._normalize_features()
        
        if self.debug:
            print(f"特征提取完成，共 {len(self.features_df)} 个笔触")
            print(f"特征维度: {self.features_df.shape[1]}")
            print(f"特征列: {list(self.features_df.columns)}")
        
        return self.features_df
    
    def _extract_single_stroke_features(self, stroke: Stroke) -> Dict:
        """提取单个笔触的所有特征"""
        features = {'stroke_id': stroke.id}
        
        # 1. 骨架点与距离特征
        skeleton_features = self._extract_skeleton_features(stroke)
        features.update(skeleton_features)
        
        # 2. 面积与尺度特征
        scale_features = self._extract_scale_features(stroke)
        features.update(scale_features)
        
        # 3. 形状特征
        shape_features = self._extract_shape_features(stroke)
        features.update(shape_features)
        
        # 4. 墨湿度与厚度特征
        ink_features = self._extract_ink_features(stroke)
        features.update(ink_features)
        
        # 5. 颜色特征
        color_features = self._extract_color_features(stroke)
        features.update(color_features)
        
        # 6. 位置显著性特征
        saliency_features = self._extract_saliency_features(stroke)
        features.update(saliency_features)
        
        return features
    
    def _extract_skeleton_features(self, stroke: Stroke) -> Dict:
        """提取骨架点与距离特征"""
        try:
            skeleton_points = stroke.get_skeleton_points()
            
            # 骨架长度（端点间距离）
            if len(skeleton_points) >= 2:
                distances = pdist(skeleton_points)
                skeleton_length = np.max(distances) if len(distances) > 0 else 0
            else:
                skeleton_length = 0
            
            # 骨架点数量
            num_skeleton_points = len(skeleton_points)
            
            return {
                'skeleton_length': skeleton_length,
                'num_skeleton_points': num_skeleton_points
            }
        except Exception as e:
            print(f"提取骨架特征时出错: {e}")
            return {
                'skeleton_length': 0,
                'num_skeleton_points': 0
            }
    
    def _extract_scale_features(self, stroke: Stroke) -> Dict:
        """提取面积与尺度特征"""
        # 面积
        area = stroke.area
        
        # 周长
        perimeter = stroke.perimeter
        
        # 最小外接矩形
        rect = cv2.minAreaRect(stroke.contour)
        box_area = rect[1][0] * rect[1][1]
        
        # 尺度（最小外接正方形边长）
        scale = max(rect[1][0], rect[1][1])
        
        # 紧凑度
        compactness = area / box_area if box_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'scale': scale,
            'compactness': compactness
        }
    
    def _extract_shape_features(self, stroke: Stroke) -> Dict:
        """提取形状特征"""
        # 圆形度
        circularity = 4 * np.pi * stroke.area / (stroke.perimeter ** 2) if stroke.perimeter > 0 else 0
        
        # 长宽比
        rect = cv2.minAreaRect(stroke.contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
        
        # 轮廓傅里叶描述子（简化版）
        contour_complex = stroke.contour[:, 0, 0] + 1j * stroke.contour[:, 0, 1]
        if len(contour_complex) > 4:
            fourier_desc = np.abs(fft(contour_complex))
            # 取前4个低频分量作为形状特征
            fourier_features = fourier_desc[1:5] / fourier_desc[1] if fourier_desc[1] > 0 else np.zeros(4)
        else:
            fourier_features = np.zeros(4)
        
        # 凸包比率
        hull = cv2.convexHull(stroke.contour)
        hull_area = cv2.contourArea(hull)
        convexity = stroke.area / hull_area if hull_area > 0 else 0
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'fourier_1': fourier_features[0],
            'fourier_2': fourier_features[1],
            'fourier_3': fourier_features[2],
            'fourier_4': fourier_features[3],
            'convexity': convexity
        }
    
    def _extract_ink_features(self, stroke: Stroke) -> Dict:
        """提取墨湿度与厚度特征"""
        # 获取笔触区域的像素值
        mask_coords = np.where(stroke.mask > 0)
        if len(mask_coords[0]) == 0:
            return {'wetness': 0, 'thickness': 0}
        
        # 模拟墨湿度：有效像素占外接矩形的比例
        bbox_area = stroke.bbox[2] * stroke.bbox[3]
        wetness = stroke.area / bbox_area if bbox_area > 0 else 0
        
        # 模拟厚度：255 - 平均灰度值
        # 这里使用掩码区域的平均值作为近似
        avg_intensity = np.mean(stroke.mask[mask_coords])
        thickness = (255 - avg_intensity) / 255.0
        
        return {
            'wetness': wetness,
            'thickness': thickness
        }
    
    def _extract_color_features(self, stroke: Stroke) -> Dict:
        """提取颜色特征（简化版）"""
        # 由于我们主要处理二值化后的图像，这里提供基础的颜色特征
        # 在实际应用中，应该从原始彩色图像中提取
        
        # 平均灰度值（归一化）
        mask_coords = np.where(stroke.mask > 0)
        if len(mask_coords[0]) == 0:
            avg_gray = 0
        else:
            avg_gray = np.mean(stroke.mask[mask_coords]) / 255.0
        
        return {
            'avg_gray': avg_gray,
            'color_variance': 0.1  # 占位符，实际应从原图计算
        }
    
    def _extract_saliency_features(self, stroke: Stroke) -> Dict:
        """提取位置显著性特征"""
        # 将笔触质心坐标映射到显著性图
        centroid_x, centroid_y = stroke.centroid
        
        # 假设图像尺寸，实际应该从原图获取
        img_height, img_width = 600, 800  # 默认尺寸
        
        # 归一化坐标到[0, 299]范围
        norm_x = int((centroid_x / img_width) * 299)
        norm_y = int((centroid_y / img_height) * 299)
        
        # 确保坐标在有效范围内
        norm_x = np.clip(norm_x, 0, 299)
        norm_y = np.clip(norm_y, 0, 299)
        
        # 查询显著性值
        saliency = self.saliency_map[norm_y, norm_x]
        
        return {
            'position_saliency': saliency,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y
        }
    
    def _normalize_features(self):
        """特征归一化"""
        # 选择需要归一化的数值特征
        numeric_features = self.features_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['stroke_id']  # 不需要归一化的列
        
        features_to_normalize = [col for col in numeric_features if col not in exclude_cols]
        
        # MinMax归一化到[0,1]区间
        scaler = MinMaxScaler()
        self.features_df[features_to_normalize] = scaler.fit_transform(
            self.features_df[features_to_normalize]
        )
        
        if self.debug:
            print(f"归一化特征: {features_to_normalize}")
    
    def get_feature_weights(self) -> Dict[str, float]:
        """获取特征权重（论文中所有特征等权重）"""
        # 根据论文Section 5.2，所有特征等权重
        feature_names = [
            'skeleton_length', 'area', 'scale', 'wetness', 
            'thickness', 'position_saliency'
        ]
        
        return {name: 1.0 for name in feature_names}
    
    def compute_comprehensive_score(self, stroke_id: int) -> float:
        """计算笔触的综合特征得分"""
        if self.features_df is None:
            raise ValueError("请先调用extract_features()")
        
        stroke_features = self.features_df[self.features_df['stroke_id'] == stroke_id]
        if stroke_features.empty:
            return 0.0
        
        weights = self.get_feature_weights()
        score = 0.0
        
        for feature_name, weight in weights.items():
            if feature_name in stroke_features.columns:
                score += weight * stroke_features[feature_name].iloc[0]
        
        return score
    
    def save_feature_analysis(self, features_df: pd.DataFrame, output_dir: Path):
        """保存特征分析结果"""
        if not self.debug:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存特征数据
        features_df.to_csv(output_dir / "features.csv", index=False)
        
        # 保存特征统计
        feature_stats = features_df.describe()
        feature_stats.to_csv(output_dir / "feature_statistics.csv")
        
        # 生成特征分布图
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'stroke_id']
        
        if len(numeric_features) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature in enumerate(numeric_features[:6]):
                if i < len(axes):
                    axes[i].hist(features_df[feature], bins=20, alpha=0.7)
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Frequency')
            
            # 隐藏多余的子图
            for i in range(len(numeric_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 保存特征相关性矩阵
        if len(numeric_features) > 1:
            correlation_matrix = features_df[numeric_features].corr()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(numeric_features)), numeric_features, rotation=45)
            plt.yticks(range(len(numeric_features)), numeric_features)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_dir / "feature_correlation.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"特征分析结果已保存至: {output_dir}")

# 测试代码
if __name__ == "__main__":
    from stroke_extraction import StrokeExtractor
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python feature_modeling.py <image_path>")
        sys.exit(1)
    
    # 先提取笔触
    extractor = StrokeExtractor(debug=True)
    strokes = extractor.extract_strokes(sys.argv[1])
    
    # 提取特征
    modeler = FeatureModeler(debug=True)
    features = modeler.extract_features(strokes)
    
    print(f"\n特征提取结果:")
    print(features.head())
    print(f"\n特征统计:")
    print(features.describe())