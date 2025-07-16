# -*- coding: utf-8 -*-
"""
笔画匹配器

在笔画库中查找与输入笔画最相似的模板
支持多种相似度度量和匹配策略
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import math


@dataclass
class MatchResult:
    """
    匹配结果数据结构
    
    Attributes:
        template_id (str): 匹配的模板ID
        similarity_score (float): 相似度分数
        geometric_similarity (float): 几何相似度
        texture_similarity (float): 纹理相似度
        dynamic_similarity (float): 动态特征相似度
        confidence (float): 匹配置信度
        transformation (Dict): 变换参数
        match_details (Dict): 详细匹配信息
    """
    template_id: str
    similarity_score: float
    geometric_similarity: float
    texture_similarity: float
    dynamic_similarity: float
    confidence: float
    transformation: Dict[str, Any]
    match_details: Dict[str, Any]


class StrokeMatcher:
    """
    笔画匹配器
    
    提供多种笔画匹配算法和相似度计算方法
    """
    
    def __init__(self, config):
        """
        初始化笔画匹配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 匹配参数
        self.geometric_weight = config['stroke_matching'].get('geometric_weight', 0.4)
        self.texture_weight = config['stroke_matching'].get('texture_weight', 0.3)
        self.dynamic_weight = config['stroke_matching'].get('dynamic_weight', 0.3)
        
        self.similarity_threshold = config['stroke_matching'].get('similarity_threshold', 0.7)
        self.max_candidates = config['stroke_matching'].get('max_candidates', 50)
        
        # 几何匹配参数
        self.skeleton_weight = config['stroke_matching'].get('skeleton_weight', 0.6)
        self.contour_weight = config['stroke_matching'].get('contour_weight', 0.4)
        
        # 动态特征权重
        self.width_weight = config['stroke_matching'].get('width_weight', 0.4)
        self.pressure_weight = config['stroke_matching'].get('pressure_weight', 0.3)
        self.velocity_weight = config['stroke_matching'].get('velocity_weight', 0.3)
    
    def find_best_matches(self, query_stroke: Dict[str, Any], 
                         stroke_database, 
                         category: str = None,
                         top_k: int = 5) -> List[MatchResult]:
        """
        查找最佳匹配的笔画模板
        
        Args:
            query_stroke (Dict): 查询笔画数据
            stroke_database: 笔画数据库
            category (str, optional): 限制搜索的类别
            top_k (int): 返回前k个最佳匹配
            
        Returns:
            List[MatchResult]: 匹配结果列表
        """
        try:
            # 获取候选模板
            candidates = self._get_candidates(stroke_database, category)
            
            if not candidates:
                self.logger.warning("No candidate templates found")
                return []
            
            # 计算匹配结果
            match_results = []
            
            for template in candidates:
                match_result = self._compute_match_score(query_stroke, template)
                
                if match_result.similarity_score >= self.similarity_threshold:
                    match_results.append(match_result)
            
            # 按相似度排序
            match_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            self.logger.info(f"Found {len(match_results)} matches above threshold")
            return match_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding matches: {str(e)}")
            return []
    
    def _get_candidates(self, stroke_database, category: str = None) -> List:
        """
        获取候选模板
        
        Args:
            stroke_database: 笔画数据库
            category (str, optional): 类别限制
            
        Returns:
            List: 候选模板列表
        """
        if category:
            candidates = stroke_database.get_templates_by_category(category)
        else:
            candidates = list(stroke_database.templates.values())
        
        # 限制候选数量
        if len(candidates) > self.max_candidates:
            # 可以添加预筛选逻辑
            candidates = candidates[:self.max_candidates]
        
        return candidates
    
    def _compute_match_score(self, query_stroke: Dict[str, Any], template) -> MatchResult:
        """
        计算匹配分数
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 计算几何相似度
            geometric_sim = self._compute_geometric_similarity(query_stroke, template)
            
            # 计算纹理相似度
            texture_sim = self._compute_texture_similarity(query_stroke, template)
            
            # 计算动态特征相似度
            dynamic_sim = self._compute_dynamic_similarity(query_stroke, template)
            
            # 计算综合相似度
            overall_similarity = (
                self.geometric_weight * geometric_sim +
                self.texture_weight * texture_sim +
                self.dynamic_weight * dynamic_sim
            )
            
            # 计算置信度
            confidence = self._compute_confidence(
                geometric_sim, texture_sim, dynamic_sim
            )
            
            # 计算变换参数
            transformation = self._compute_transformation(query_stroke, template)
            
            # 详细匹配信息
            match_details = {
                'skeleton_similarity': self._compute_skeleton_similarity(query_stroke, template),
                'contour_similarity': self._compute_contour_similarity(query_stroke, template),
                'width_similarity': self._compute_width_similarity(query_stroke, template),
                'aspect_ratio_diff': abs(
                    query_stroke.get('aspect_ratio', 1.0) - 
                    template.features.get('aspect_ratio', 1.0)
                ),
                'length_ratio': self._compute_length_ratio(query_stroke, template)
            }
            
            return MatchResult(
                template_id=template.id,
                similarity_score=overall_similarity,
                geometric_similarity=geometric_sim,
                texture_similarity=texture_sim,
                dynamic_similarity=dynamic_sim,
                confidence=confidence,
                transformation=transformation,
                match_details=match_details
            )
            
        except Exception as e:
            self.logger.error(f"Error computing match score for template {template.id}: {str(e)}")
            return MatchResult(
                template_id=template.id,
                similarity_score=0.0,
                geometric_similarity=0.0,
                texture_similarity=0.0,
                dynamic_similarity=0.0,
                confidence=0.0,
                transformation={},
                match_details={}
            )
    
    def _compute_geometric_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算几何相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 几何相似度
        """
        try:
            # 骨架相似度
            skeleton_sim = self._compute_skeleton_similarity(query_stroke, template)
            
            # 轮廓相似度
            contour_sim = self._compute_contour_similarity(query_stroke, template)
            
            # 综合几何相似度
            geometric_sim = (
                self.skeleton_weight * skeleton_sim +
                self.contour_weight * contour_sim
            )
            
            return geometric_sim
            
        except Exception as e:
            self.logger.error(f"Error computing geometric similarity: {str(e)}")
            return 0.0
    
    def _compute_skeleton_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算骨架相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 骨架相似度
        """
        try:
            query_skeleton = query_stroke.get('skeleton')
            template_skeleton = template.skeleton
            
            if query_skeleton is None or template_skeleton is None:
                return 0.0
            
            # 归一化骨架
            query_norm = self._normalize_skeleton(query_skeleton)
            template_norm = self._normalize_skeleton(template_skeleton)
            
            # 重采样到相同长度
            target_length = min(len(query_norm), len(template_norm), 100)
            query_resampled = self._resample_curve(query_norm, target_length)
            template_resampled = self._resample_curve(template_norm, target_length)
            
            # 计算点对点距离
            distances = np.linalg.norm(query_resampled - template_resampled, axis=1)
            mean_distance = np.mean(distances)
            
            # 转换为相似度（距离越小，相似度越高）
            max_distance = np.sqrt(2)  # 归一化空间中的最大距离
            similarity = max(0, 1 - mean_distance / max_distance)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing skeleton similarity: {str(e)}")
            return 0.0
    
    def _compute_contour_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算轮廓相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 轮廓相似度
        """
        try:
            query_contour = query_stroke.get('contour')
            template_contour = template.contour
            
            if query_contour is None or template_contour is None:
                return 0.0
            
            # 使用Hu矩进行形状匹配
            query_moments = cv2.HuMoments(cv2.moments(query_contour))
            template_moments = cv2.HuMoments(cv2.moments(template_contour))
            
            # 计算Hu矩距离
            hu_distance = cv2.matchShapes(query_contour, template_contour, cv2.CONTOURS_MATCH_I1, 0)
            
            # 转换为相似度
            similarity = max(0, 1 / (1 + hu_distance))
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing contour similarity: {str(e)}")
            return 0.0
    
    def _compute_texture_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算纹理相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 纹理相似度
        """
        try:
            # 提取纹理特征
            query_texture = query_stroke.get('texture_features', {})
            template_texture = template.features.get('texture_features', {})
            
            if not query_texture or not template_texture:
                return 0.5  # 默认中等相似度
            
            # 比较各种纹理特征
            similarities = []
            
            # 对比度
            if 'contrast' in query_texture and 'contrast' in template_texture:
                contrast_sim = 1 - abs(query_texture['contrast'] - template_texture['contrast']) / 255
                similarities.append(contrast_sim)
            
            # 能量
            if 'energy' in query_texture and 'energy' in template_texture:
                energy_sim = 1 - abs(query_texture['energy'] - template_texture['energy'])
                similarities.append(energy_sim)
            
            # 均匀性
            if 'homogeneity' in query_texture and 'homogeneity' in template_texture:
                homogeneity_sim = 1 - abs(query_texture['homogeneity'] - template_texture['homogeneity'])
                similarities.append(homogeneity_sim)
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error computing texture similarity: {str(e)}")
            return 0.5
    
    def _compute_dynamic_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算动态特征相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 动态特征相似度
        """
        try:
            # 宽度变化相似度
            width_sim = self._compute_width_similarity(query_stroke, template)
            
            # 压力变化相似度（如果有的话）
            pressure_sim = self._compute_pressure_similarity(query_stroke, template)
            
            # 速度变化相似度（如果有的话）
            velocity_sim = self._compute_velocity_similarity(query_stroke, template)
            
            # 综合动态相似度
            dynamic_sim = (
                self.width_weight * width_sim +
                self.pressure_weight * pressure_sim +
                self.velocity_weight * velocity_sim
            )
            
            return dynamic_sim
            
        except Exception as e:
            self.logger.error(f"Error computing dynamic similarity: {str(e)}")
            return 0.0
    
    def _compute_width_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算宽度变化相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 宽度相似度
        """
        try:
            query_width = query_stroke.get('width_profile')
            template_width = template.width_profile
            
            if query_width is None or template_width is None:
                return 0.5
            
            # 归一化宽度曲线
            query_width_norm = self._normalize_profile(query_width)
            template_width_norm = self._normalize_profile(template_width)
            
            # 重采样到相同长度
            target_length = min(len(query_width_norm), len(template_width_norm), 50)
            query_resampled = self._resample_profile(query_width_norm, target_length)
            template_resampled = self._resample_profile(template_width_norm, target_length)
            
            # 计算相关系数
            correlation = np.corrcoef(query_resampled, template_resampled)[0, 1]
            
            # 处理NaN值
            if np.isnan(correlation):
                correlation = 0
            
            # 转换为正相似度
            similarity = (correlation + 1) / 2
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing width similarity: {str(e)}")
            return 0.5
    
    def _compute_pressure_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算压力变化相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 压力相似度
        """
        try:
            query_pressure = query_stroke.get('pressure_profile')
            template_pressure = template.pressure_profile
            
            if query_pressure is None or template_pressure is None:
                return 0.5  # 默认中等相似度
            
            # 类似宽度相似度的计算
            query_pressure_norm = self._normalize_profile(query_pressure)
            template_pressure_norm = self._normalize_profile(template_pressure)
            
            target_length = min(len(query_pressure_norm), len(template_pressure_norm), 50)
            query_resampled = self._resample_profile(query_pressure_norm, target_length)
            template_resampled = self._resample_profile(template_pressure_norm, target_length)
            
            correlation = np.corrcoef(query_resampled, template_resampled)[0, 1]
            
            if np.isnan(correlation):
                correlation = 0
            
            similarity = (correlation + 1) / 2
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing pressure similarity: {str(e)}")
            return 0.5
    
    def _compute_velocity_similarity(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算速度变化相似度
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 速度相似度
        """
        try:
            query_velocity = query_stroke.get('velocity_profile')
            template_velocity = template.velocity_profile
            
            if query_velocity is None or template_velocity is None:
                return 0.5  # 默认中等相似度
            
            # 类似宽度相似度的计算
            query_velocity_norm = self._normalize_profile(query_velocity)
            template_velocity_norm = self._normalize_profile(template_velocity)
            
            target_length = min(len(query_velocity_norm), len(template_velocity_norm), 50)
            query_resampled = self._resample_profile(query_velocity_norm, target_length)
            template_resampled = self._resample_profile(template_velocity_norm, target_length)
            
            correlation = np.corrcoef(query_resampled, template_resampled)[0, 1]
            
            if np.isnan(correlation):
                correlation = 0
            
            similarity = (correlation + 1) / 2
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing velocity similarity: {str(e)}")
            return 0.5
    
    def _compute_confidence(self, geometric_sim: float, texture_sim: float, dynamic_sim: float) -> float:
        """
        计算匹配置信度
        
        Args:
            geometric_sim (float): 几何相似度
            texture_sim (float): 纹理相似度
            dynamic_sim (float): 动态相似度
            
        Returns:
            float: 置信度
        """
        # 基于各项相似度的一致性计算置信度
        similarities = [geometric_sim, texture_sim, dynamic_sim]
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # 一致性越高，置信度越高
        consistency = max(0, 1 - std_sim)
        
        # 综合平均相似度和一致性
        confidence = 0.7 * mean_sim + 0.3 * consistency
        
        return confidence
    
    def _compute_transformation(self, query_stroke: Dict[str, Any], template) -> Dict[str, Any]:
        """
        计算从模板到查询笔画的变换参数
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            Dict: 变换参数
        """
        try:
            transformation = {
                'scale': 1.0,
                'rotation': 0.0,
                'translation': [0.0, 0.0],
                'shear': 0.0
            }
            
            # 计算尺度变换
            query_bbox = query_stroke.get('bounding_rect')
            template_bbox = template.features.get('bounding_rect')
            
            if query_bbox and template_bbox:
                query_w, query_h = query_bbox[2], query_bbox[3]
                template_w, template_h = template_bbox[2], template_bbox[3]
                
                if template_w > 0 and template_h > 0:
                    scale_x = query_w / template_w
                    scale_y = query_h / template_h
                    transformation['scale'] = (scale_x + scale_y) / 2
            
            # 计算旋转角度（基于主方向）
            query_orientation = query_stroke.get('orientation', 0)
            template_orientation = template.features.get('orientation', 0)
            transformation['rotation'] = query_orientation - template_orientation
            
            # 计算平移（基于质心）
            query_centroid = query_stroke.get('centroid')
            template_centroid = template.features.get('centroid')
            
            if query_centroid and template_centroid:
                transformation['translation'] = [
                    query_centroid[0] - template_centroid[0],
                    query_centroid[1] - template_centroid[1]
                ]
            
            return transformation
            
        except Exception as e:
            self.logger.error(f"Error computing transformation: {str(e)}")
            return {'scale': 1.0, 'rotation': 0.0, 'translation': [0.0, 0.0], 'shear': 0.0}
    
    def _compute_length_ratio(self, query_stroke: Dict[str, Any], template) -> float:
        """
        计算长度比例
        
        Args:
            query_stroke (Dict): 查询笔画
            template: 模板笔画
            
        Returns:
            float: 长度比例
        """
        try:
            query_length = query_stroke.get('length', 0)
            template_length = template.features.get('length', 0)
            
            if template_length > 0:
                return query_length / template_length
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _normalize_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        归一化骨架坐标
        
        Args:
            skeleton (np.ndarray): 骨架点
            
        Returns:
            np.ndarray: 归一化骨架
        """
        if len(skeleton) == 0:
            return skeleton
        
        # 计算边界框
        min_coords = np.min(skeleton, axis=0)
        max_coords = np.max(skeleton, axis=0)
        
        # 避免除零
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1
        
        # 归一化到[0, 1]
        normalized = (skeleton - min_coords) / range_coords
        
        return normalized
    
    def _normalize_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        归一化轮廓曲线
        
        Args:
            profile (np.ndarray): 轮廓曲线
            
        Returns:
            np.ndarray: 归一化轮廓
        """
        if len(profile) == 0:
            return profile
        
        min_val = np.min(profile)
        max_val = np.max(profile)
        
        if max_val - min_val > 0:
            normalized = (profile - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(profile)
        
        return normalized
    
    def _resample_curve(self, curve: np.ndarray, target_length: int) -> np.ndarray:
        """
        重采样曲线到指定长度
        
        Args:
            curve (np.ndarray): 输入曲线
            target_length (int): 目标长度
            
        Returns:
            np.ndarray: 重采样曲线
        """
        if len(curve) == 0:
            return np.zeros((target_length, curve.shape[1] if curve.ndim > 1 else 1))
        
        if len(curve) == target_length:
            return curve
        
        # 线性插值重采样
        indices = np.linspace(0, len(curve) - 1, target_length)
        
        if curve.ndim == 1:
            resampled = np.interp(indices, np.arange(len(curve)), curve)
        else:
            resampled = np.zeros((target_length, curve.shape[1]))
            for i in range(curve.shape[1]):
                resampled[:, i] = np.interp(indices, np.arange(len(curve)), curve[:, i])
        
        return resampled
    
    def _resample_profile(self, profile: np.ndarray, target_length: int) -> np.ndarray:
        """
        重采样轮廓到指定长度
        
        Args:
            profile (np.ndarray): 输入轮廓
            target_length (int): 目标长度
            
        Returns:
            np.ndarray: 重采样轮廓
        """
        if len(profile) == 0:
            return np.zeros(target_length)
        
        if len(profile) == target_length:
            return profile
        
        # 线性插值重采样
        indices = np.linspace(0, len(profile) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(profile)), profile)
        
        return resampled