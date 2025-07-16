# -*- coding: utf-8 -*-
"""
阶段分类器

实现论文中的三阶段笔触分类：
1. 主要笔触：决定构图的大块深色笔触
2. 局部细节笔触：细化主体的中等复杂度笔触
3. 装饰笔触：干燥或细点缀的修饰性笔触
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum


class StrokeStage(Enum):
    """
    笔触阶段枚举
    """
    MAIN = "main"  # 主要笔触
    DETAIL = "detail"  # 局部细节笔触
    DECORATION = "decoration"  # 装饰笔触


@dataclass
class StageFeatures:
    """
    阶段特征数据结构
    """
    # 几何特征
    area_ratio: float  # 面积比例
    size_score: float  # 尺寸得分
    complexity_score: float  # 复杂度得分
    
    # 墨色特征
    darkness_score: float  # 深色得分
    wetness_score: float  # 湿度得分
    thickness_score: float  # 厚度得分
    
    # 位置特征
    centrality_score: float  # 中心性得分
    prominence_score: float  # 显著性得分
    
    # 综合得分
    main_score: float = 0.0
    detail_score: float = 0.0
    decoration_score: float = 0.0


class StageClassifier:
    """
    阶段分类器
    
    根据论文中的艺术原则将笔触分为三个阶段
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化阶段分类器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分类权重
        self.weights = {
            'main': {
                'area_ratio': config.get('main_area_weight', 0.3),
                'darkness': config.get('main_darkness_weight', 0.25),
                'centrality': config.get('main_centrality_weight', 0.2),
                'size': config.get('main_size_weight', 0.15),
                'prominence': config.get('main_prominence_weight', 0.1)
            },
            'detail': {
                'complexity': config.get('detail_complexity_weight', 0.3),
                'wetness': config.get('detail_wetness_weight', 0.2),
                'thickness': config.get('detail_thickness_weight', 0.2),
                'area_ratio': config.get('detail_area_weight', 0.15),
                'centrality': config.get('detail_centrality_weight', 0.15)
            },
            'decoration': {
                'wetness': config.get('decoration_wetness_weight', 0.25),  # 干燥特征
                'size': config.get('decoration_size_weight', 0.25),  # 小尺寸
                'complexity': config.get('decoration_complexity_weight', 0.2),
                'prominence': config.get('decoration_prominence_weight', 0.15),
                'thickness': config.get('decoration_thickness_weight', 0.15)
            }
        }
        
        # 阈值参数
        self.thresholds = {
            'main_min_area_ratio': config.get('main_min_area_ratio', 0.05),
            'main_min_darkness': config.get('main_min_darkness', 0.6),
            'detail_min_complexity': config.get('detail_min_complexity', 0.3),
            'decoration_max_size': config.get('decoration_max_size', 0.02),
            'decoration_max_wetness': config.get('decoration_max_wetness', 0.4)
        }
        
        # 统计信息
        self.image_stats = None
        
    def classify_strokes(self, stroke_features: List[Dict[str, Any]], 
                        image_shape: Tuple[int, int]) -> Dict[int, StrokeStage]:
        """
        对笔触进行阶段分类
        
        Args:
            stroke_features: 笔触特征列表
            image_shape: 图像形状
            
        Returns:
            Dict[int, StrokeStage]: 笔触ID到阶段的映射
        """
        try:
            if not stroke_features:
                return {}
            
            # 计算图像统计信息
            self._calculate_image_statistics(stroke_features, image_shape)
            
            # 计算阶段特征
            stage_features = self._calculate_stage_features(stroke_features)
            
            # 执行分类
            classifications = self._perform_classification(stage_features)
            
            # 后处理和优化
            optimized_classifications = self._optimize_classifications(
                classifications, stage_features
            )
            
            self.logger.info(f"Classified {len(stroke_features)} strokes into stages")
            return optimized_classifications
            
        except Exception as e:
            self.logger.error(f"Error in stroke stage classification: {str(e)}")
            return {}
    
    def get_stage_statistics(self, classifications: Dict[int, StrokeStage]) -> Dict[str, Any]:
        """
        获取阶段分类统计信息
        
        Args:
            classifications: 分类结果
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_strokes': len(classifications),
            'main_count': 0,
            'detail_count': 0,
            'decoration_count': 0
        }
        
        for stage in classifications.values():
            if stage == StrokeStage.MAIN:
                stats['main_count'] += 1
            elif stage == StrokeStage.DETAIL:
                stats['detail_count'] += 1
            elif stage == StrokeStage.DECORATION:
                stats['decoration_count'] += 1
        
        # 计算比例
        if stats['total_strokes'] > 0:
            stats['main_ratio'] = stats['main_count'] / stats['total_strokes']
            stats['detail_ratio'] = stats['detail_count'] / stats['total_strokes']
            stats['decoration_ratio'] = stats['decoration_count'] / stats['total_strokes']
        else:
            stats['main_ratio'] = stats['detail_ratio'] = stats['decoration_ratio'] = 0.0
        
        return stats
    
    def _calculate_image_statistics(self, stroke_features: List[Dict[str, Any]], 
                                   image_shape: Tuple[int, int]):
        """
        计算图像统计信息
        
        Args:
            stroke_features: 笔触特征列表
            image_shape: 图像形状
        """
        total_image_area = image_shape[0] * image_shape[1]
        
        # 提取各种特征值
        areas = [f.get('area', 0) for f in stroke_features]
        darkness_values = [f.get('darkness', 0) for f in stroke_features]
        wetness_values = [f.get('wetness', 0) for f in stroke_features]
        thickness_values = [f.get('thickness', 0) for f in stroke_features]
        
        self.image_stats = {
            'total_area': total_image_area,
            'stroke_count': len(stroke_features),
            'total_stroke_area': sum(areas),
            'mean_area': np.mean(areas) if areas else 0,
            'std_area': np.std(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'mean_darkness': np.mean(darkness_values) if darkness_values else 0,
            'std_darkness': np.std(darkness_values) if darkness_values else 0,
            'mean_wetness': np.mean(wetness_values) if wetness_values else 0,
            'std_wetness': np.std(wetness_values) if wetness_values else 0,
            'mean_thickness': np.mean(thickness_values) if thickness_values else 0,
            'std_thickness': np.std(thickness_values) if thickness_values else 0
        }
    
    def _calculate_stage_features(self, stroke_features: List[Dict[str, Any]]) -> List[StageFeatures]:
        """
        计算阶段特征
        
        Args:
            stroke_features: 笔触特征列表
            
        Returns:
            List[StageFeatures]: 阶段特征列表
        """
        stage_features = []
        
        for features in stroke_features:
            # 几何特征
            area_ratio = self._calculate_area_ratio(features)
            size_score = self._calculate_size_score(features)
            complexity_score = self._calculate_complexity_score(features)
            
            # 墨色特征
            darkness_score = self._calculate_darkness_score(features)
            wetness_score = self._calculate_wetness_score(features)
            thickness_score = self._calculate_thickness_score(features)
            
            # 位置特征
            centrality_score = self._calculate_centrality_score(features)
            prominence_score = self._calculate_prominence_score(features)
            
            stage_feature = StageFeatures(
                area_ratio=area_ratio,
                size_score=size_score,
                complexity_score=complexity_score,
                darkness_score=darkness_score,
                wetness_score=wetness_score,
                thickness_score=thickness_score,
                centrality_score=centrality_score,
                prominence_score=prominence_score
            )
            
            # 计算各阶段得分
            stage_feature.main_score = self._calculate_main_score(stage_feature)
            stage_feature.detail_score = self._calculate_detail_score(stage_feature)
            stage_feature.decoration_score = self._calculate_decoration_score(stage_feature)
            
            stage_features.append(stage_feature)
        
        return stage_features
    
    def _calculate_area_ratio(self, features: Dict[str, Any]) -> float:
        """
        计算面积比例
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 面积比例
        """
        area = features.get('area', 0)
        if self.image_stats and self.image_stats['total_area'] > 0:
            return area / self.image_stats['total_area']
        return 0.0
    
    def _calculate_size_score(self, features: Dict[str, Any]) -> float:
        """
        计算尺寸得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 尺寸得分 (0-1)
        """
        area = features.get('area', 0)
        if self.image_stats and self.image_stats['max_area'] > 0:
            return area / self.image_stats['max_area']
        return 0.0
    
    def _calculate_complexity_score(self, features: Dict[str, Any]) -> float:
        """
        计算复杂度得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 复杂度得分 (0-1)
        """
        # 基于轮廓复杂度、角点数量等
        perimeter = features.get('perimeter', 0)
        area = features.get('area', 1)
        corner_count = features.get('corner_count', 0)
        
        # 形状复杂度 (周长平方/面积)
        shape_complexity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # 角点密度
        corner_density = corner_count / area if area > 0 else 0
        
        # 综合复杂度
        complexity = 0.6 * min(shape_complexity / 5.0, 1.0) + 0.4 * min(corner_density * 1000, 1.0)
        
        return min(complexity, 1.0)
    
    def _calculate_darkness_score(self, features: Dict[str, Any]) -> float:
        """
        计算深色得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 深色得分 (0-1)
        """
        darkness = features.get('darkness', 0)
        # 归一化到0-1范围
        return min(max(darkness, 0.0), 1.0)
    
    def _calculate_wetness_score(self, features: Dict[str, Any]) -> float:
        """
        计算湿度得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 湿度得分 (0-1)
        """
        wetness = features.get('wetness', 0)
        return min(max(wetness, 0.0), 1.0)
    
    def _calculate_thickness_score(self, features: Dict[str, Any]) -> float:
        """
        计算厚度得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 厚度得分 (0-1)
        """
        thickness = features.get('thickness', 0)
        return min(max(thickness, 0.0), 1.0)
    
    def _calculate_centrality_score(self, features: Dict[str, Any]) -> float:
        """
        计算中心性得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 中心性得分 (0-1)
        """
        center = features.get('center', (0, 0))
        if self.image_stats:
            # 假设图像中心
            image_center_x = self.image_stats.get('image_width', 100) / 2
            image_center_y = self.image_stats.get('image_height', 100) / 2
            
            # 计算到中心的距离
            distance = np.sqrt(
                (center[0] - image_center_x) ** 2 + 
                (center[1] - image_center_y) ** 2
            )
            
            # 归一化距离
            max_distance = np.sqrt(image_center_x ** 2 + image_center_y ** 2)
            if max_distance > 0:
                centrality = 1.0 - (distance / max_distance)
                return max(centrality, 0.0)
        
        return 0.5  # 默认中等中心性
    
    def _calculate_prominence_score(self, features: Dict[str, Any]) -> float:
        """
        计算显著性得分
        
        Args:
            features: 笔触特征
            
        Returns:
            float: 显著性得分 (0-1)
        """
        # 基于三分法则的显著性
        prominence = features.get('prominence', 0)
        return min(max(prominence, 0.0), 1.0)
    
    def _calculate_main_score(self, stage_feature: StageFeatures) -> float:
        """
        计算主要笔触得分
        
        Args:
            stage_feature: 阶段特征
            
        Returns:
            float: 主要笔触得分
        """
        weights = self.weights['main']
        
        score = (
            weights['area_ratio'] * stage_feature.area_ratio +
            weights['darkness'] * stage_feature.darkness_score +
            weights['centrality'] * stage_feature.centrality_score +
            weights['size'] * stage_feature.size_score +
            weights['prominence'] * stage_feature.prominence_score
        )
        
        return score
    
    def _calculate_detail_score(self, stage_feature: StageFeatures) -> float:
        """
        计算局部细节笔触得分
        
        Args:
            stage_feature: 阶段特征
            
        Returns:
            float: 局部细节笔触得分
        """
        weights = self.weights['detail']
        
        score = (
            weights['complexity'] * stage_feature.complexity_score +
            weights['wetness'] * stage_feature.wetness_score +
            weights['thickness'] * stage_feature.thickness_score +
            weights['area_ratio'] * stage_feature.area_ratio +
            weights['centrality'] * stage_feature.centrality_score
        )
        
        return score
    
    def _calculate_decoration_score(self, stage_feature: StageFeatures) -> float:
        """
        计算装饰笔触得分
        
        Args:
            stage_feature: 阶段特征
            
        Returns:
            float: 装饰笔触得分
        """
        weights = self.weights['decoration']
        
        # 装饰笔触特征：干燥(低湿度)、小尺寸、细节丰富
        dryness_score = 1.0 - stage_feature.wetness_score  # 干燥度
        small_size_score = 1.0 - stage_feature.size_score  # 小尺寸
        
        score = (
            weights['wetness'] * dryness_score +
            weights['size'] * small_size_score +
            weights['complexity'] * stage_feature.complexity_score +
            weights['prominence'] * stage_feature.prominence_score +
            weights['thickness'] * (1.0 - stage_feature.thickness_score)  # 细笔触
        )
        
        return score
    
    def _perform_classification(self, stage_features: List[StageFeatures]) -> Dict[int, StrokeStage]:
        """
        执行分类
        
        Args:
            stage_features: 阶段特征列表
            
        Returns:
            Dict[int, StrokeStage]: 分类结果
        """
        classifications = {}
        
        for i, features in enumerate(stage_features):
            # 获取各阶段得分
            scores = {
                StrokeStage.MAIN: features.main_score,
                StrokeStage.DETAIL: features.detail_score,
                StrokeStage.DECORATION: features.decoration_score
            }
            
            # 应用硬约束
            if not self._check_main_constraints(features):
                scores[StrokeStage.MAIN] = 0.0
            
            if not self._check_decoration_constraints(features):
                scores[StrokeStage.DECORATION] = 0.0
            
            # 选择得分最高的阶段
            best_stage = max(scores.keys(), key=lambda k: scores[k])
            classifications[i] = best_stage
        
        return classifications
    
    def _check_main_constraints(self, features: StageFeatures) -> bool:
        """
        检查主要笔触约束
        
        Args:
            features: 阶段特征
            
        Returns:
            bool: 是否满足约束
        """
        # 主要笔触必须有足够的面积和深色度
        return (
            features.area_ratio >= self.thresholds['main_min_area_ratio'] and
            features.darkness_score >= self.thresholds['main_min_darkness']
        )
    
    def _check_decoration_constraints(self, features: StageFeatures) -> bool:
        """
        检查装饰笔触约束
        
        Args:
            features: 阶段特征
            
        Returns:
            bool: 是否满足约束
        """
        # 装饰笔触通常是小尺寸和干燥的
        return (
            features.size_score <= self.thresholds['decoration_max_size'] or
            features.wetness_score <= self.thresholds['decoration_max_wetness']
        )
    
    def _optimize_classifications(self, classifications: Dict[int, StrokeStage],
                                 stage_features: List[StageFeatures]) -> Dict[int, StrokeStage]:
        """
        优化分类结果
        
        Args:
            classifications: 初始分类结果
            stage_features: 阶段特征列表
            
        Returns:
            Dict[int, StrokeStage]: 优化后的分类结果
        """
        optimized = classifications.copy()
        
        # 确保至少有一些主要笔触
        main_count = sum(1 for stage in optimized.values() if stage == StrokeStage.MAIN)
        if main_count == 0 and len(optimized) > 0:
            # 找到最适合作为主要笔触的候选
            best_main_candidate = max(
                range(len(stage_features)),
                key=lambda i: stage_features[i].main_score
            )
            optimized[best_main_candidate] = StrokeStage.MAIN
        
        # 平衡各阶段比例
        total_strokes = len(optimized)
        if total_strokes > 0:
            main_ratio = sum(1 for stage in optimized.values() if stage == StrokeStage.MAIN) / total_strokes
            detail_ratio = sum(1 for stage in optimized.values() if stage == StrokeStage.DETAIL) / total_strokes
            decoration_ratio = sum(1 for stage in optimized.values() if stage == StrokeStage.DECORATION) / total_strokes
            
            # 理想比例：主要20-40%，细节40-60%，装饰10-30%
            if main_ratio > 0.5:  # 主要笔触过多
                self._rebalance_stages(optimized, stage_features, 'reduce_main')
            elif detail_ratio < 0.3:  # 细节笔触过少
                self._rebalance_stages(optimized, stage_features, 'increase_detail')
        
        return optimized
    
    def _rebalance_stages(self, classifications: Dict[int, StrokeStage],
                         stage_features: List[StageFeatures],
                         strategy: str):
        """
        重新平衡阶段分布
        
        Args:
            classifications: 分类结果
            stage_features: 阶段特征列表
            strategy: 平衡策略
        """
        if strategy == 'reduce_main':
            # 将一些主要笔触改为细节笔触
            main_strokes = [i for i, stage in classifications.items() if stage == StrokeStage.MAIN]
            # 选择主要得分较低的笔触
            candidates = sorted(main_strokes, key=lambda i: stage_features[i].main_score)
            
            # 改变前一半的分类
            for i in candidates[:len(candidates)//2]:
                if stage_features[i].detail_score > stage_features[i].decoration_score:
                    classifications[i] = StrokeStage.DETAIL
                else:
                    classifications[i] = StrokeStage.DECORATION
        
        elif strategy == 'increase_detail':
            # 将一些装饰笔触改为细节笔触
            decoration_strokes = [i for i, stage in classifications.items() if stage == StrokeStage.DECORATION]
            # 选择细节得分较高的装饰笔触
            candidates = sorted(decoration_strokes, key=lambda i: stage_features[i].detail_score, reverse=True)
            
            # 改变前一半的分类
            for i in candidates[:len(candidates)//2]:
                classifications[i] = StrokeStage.DETAIL
    
    def visualize_stage_distribution(self, classifications: Dict[int, StrokeStage],
                                   stroke_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        可视化阶段分布
        
        Args:
            classifications: 分类结果
            stroke_features: 笔触特征列表
            
        Returns:
            Dict: 可视化数据
        """
        visualization_data = {
            'stage_counts': self.get_stage_statistics(classifications),
            'stage_areas': {'main': 0, 'detail': 0, 'decoration': 0},
            'stage_positions': {'main': [], 'detail': [], 'decoration': []}
        }
        
        for stroke_id, stage in classifications.items():
            if stroke_id < len(stroke_features):
                features = stroke_features[stroke_id]
                area = features.get('area', 0)
                center = features.get('center', (0, 0))
                
                stage_name = stage.value
                visualization_data['stage_areas'][stage_name] += area
                visualization_data['stage_positions'][stage_name].append(center)
        
        return visualization_data