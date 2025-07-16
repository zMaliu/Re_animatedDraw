# -*- coding: utf-8 -*-
"""
笔触匹配模块

提供笔触相似度计算和匹配功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import cv2
from scipy.spatial.distance import euclidean, cosine
from scipy.optimize import linear_sum_assignment
import logging
from ..stroke_extraction.stroke_detector import Stroke
from .stroke_database import StrokeTemplate


@dataclass
class MatchResult:
    """匹配结果数据结构"""
    template_id: int
    template_name: str
    similarity_score: float
    feature_similarities: Dict[str, float]
    geometric_similarity: float
    shape_similarity: float
    confidence: float
    match_details: Dict[str, Any]


class StrokeMatcher:
    """笔触匹配器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化笔触匹配器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 匹配参数
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.feature_weights = config.get('feature_weights', {
            'geometric': 0.3,
            'shape': 0.4,
            'texture': 0.2,
            'context': 0.1
        })
        
        # 几何特征权重
        self.geometric_weights = config.get('geometric_weights', {
            'area': 0.2,
            'perimeter': 0.15,
            'length': 0.2,
            'width': 0.15,
            'aspect_ratio': 0.15,
            'angle': 0.15
        })
        
        # 形状特征权重
        self.shape_weights = config.get('shape_weights', {
            'contour_similarity': 0.4,
            'skeleton_similarity': 0.3,
            'fourier_descriptor': 0.3
        })
        
        # 缓存
        self._descriptor_cache = {}
        
    def match_stroke(self, stroke: Stroke, templates: List[StrokeTemplate], 
                    top_k: int = 5) -> List[MatchResult]:
        """
        匹配笔触与模板
        
        Args:
            stroke: 输入笔触
            templates: 候选模板列表
            top_k: 返回前k个最佳匹配
            
        Returns:
            匹配结果列表
        """
        if not templates:
            return []
        
        results = []
        
        # 预计算笔触特征
        stroke_features = self._extract_matching_features(stroke)
        
        for template in templates:
            try:
                # 计算相似度
                similarity_score, details = self._calculate_similarity(
                    stroke, stroke_features, template
                )
                
                # 创建匹配结果
                if similarity_score >= self.similarity_threshold:
                    result = MatchResult(
                        template_id=template.id,
                        template_name=template.name,
                        similarity_score=similarity_score,
                        feature_similarities=details['feature_similarities'],
                        geometric_similarity=details['geometric_similarity'],
                        shape_similarity=details['shape_similarity'],
                        confidence=self._calculate_confidence(similarity_score, details),
                        match_details=details
                    )
                    results.append(result)
                    
            except Exception as e:
                self.logger.warning(f"Error matching template {template.id}: {e}")
                continue
        
        # 按相似度排序并返回前k个
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def find_best_match(self, stroke: Stroke, templates: List[StrokeTemplate]) -> Optional[MatchResult]:
        """
        找到笔触的最佳匹配模板
        
        Args:
            stroke: 输入笔触
            templates: 候选模板列表
            
        Returns:
            最佳匹配结果，如果没有找到合适的匹配则返回None
        """
        matches = self.match_stroke(stroke, templates, top_k=1)
        return matches[0] if matches else None
    
    def batch_match_strokes(self, strokes: List[Stroke], 
                           templates: List[StrokeTemplate],
                           top_k: int = 3) -> Dict[int, List[MatchResult]]:
        """
        批量匹配笔触
        
        Args:
            strokes: 笔触列表
            templates: 模板列表
            top_k: 每个笔触返回的最佳匹配数
            
        Returns:
            笔触ID到匹配结果的映射
        """
        results = {}
        
        for stroke in strokes:
            try:
                matches = self.match_stroke(stroke, templates, top_k)
                results[stroke.id] = matches
            except Exception as e:
                self.logger.error(f"Error matching stroke {stroke.id}: {e}")
                results[stroke.id] = []
        
        return results
    
    def find_best_template_assignment(self, strokes: List[Stroke], 
                                    templates: List[StrokeTemplate]) -> Dict[int, int]:
        """
        找到笔触到模板的最佳分配
        
        Args:
            strokes: 笔触列表
            templates: 模板列表
            
        Returns:
            笔触ID到模板ID的最佳分配
        """
        if not strokes or not templates:
            return {}
        
        # 构建相似度矩阵
        similarity_matrix = np.zeros((len(strokes), len(templates)))
        
        for i, stroke in enumerate(strokes):
            stroke_features = self._extract_matching_features(stroke)
            for j, template in enumerate(templates):
                try:
                    similarity, _ = self._calculate_similarity(
                        stroke, stroke_features, template
                    )
                    similarity_matrix[i, j] = similarity
                except Exception as e:
                    self.logger.warning(f"Error calculating similarity: {e}")
                    similarity_matrix[i, j] = 0.0
        
        # 使用匈牙利算法找到最佳分配
        # 注意：linear_sum_assignment最小化成本，所以我们使用1-similarity
        cost_matrix = 1.0 - similarity_matrix
        stroke_indices, template_indices = linear_sum_assignment(cost_matrix)
        
        # 构建分配结果
        assignment = {}
        for stroke_idx, template_idx in zip(stroke_indices, template_indices):
            similarity = similarity_matrix[stroke_idx, template_idx]
            if similarity >= self.similarity_threshold:
                assignment[strokes[stroke_idx].id] = templates[template_idx].id
        
        return assignment
    
    def calculate_stroke_similarity(self, stroke1: Stroke, stroke2: Stroke) -> float:
        """
        计算两个笔触之间的相似度
        
        Args:
            stroke1: 第一个笔触
            stroke2: 第二个笔触
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 提取特征
            features1 = self._extract_matching_features(stroke1)
            features2 = self._extract_matching_features(stroke2)
            
            # 计算几何相似度
            geometric_sim = self._calculate_geometric_similarity(
                features1['geometric'], features2['geometric']
            )
            
            # 计算形状相似度
            shape_sim = self._calculate_shape_similarity(
                stroke1.contour, stroke2.contour,
                features1.get('skeleton'), features2.get('skeleton')
            )
            
            # 加权平均
            total_similarity = (
                geometric_sim * self.feature_weights['geometric'] +
                shape_sim * self.feature_weights['shape']
            )
            
            return max(0.0, min(1.0, total_similarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating stroke similarity: {e}")
            return 0.0
    
    def _calculate_similarity(self, stroke: Stroke, stroke_features: Dict[str, Any], 
                            template: StrokeTemplate) -> Tuple[float, Dict[str, Any]]:
        """
        计算笔触与模板的相似度
        
        Args:
            stroke: 笔触
            stroke_features: 预计算的笔触特征
            template: 模板
            
        Returns:
            相似度分数和详细信息
        """
        details = {
            'feature_similarities': {},
            'geometric_similarity': 0.0,
            'shape_similarity': 0.0,
            'texture_similarity': 0.0,
            'context_similarity': 0.0
        }
        
        # 几何相似度
        geometric_sim = self._calculate_geometric_similarity(
            stroke_features['geometric'], template.features
        )
        details['geometric_similarity'] = geometric_sim
        
        # 形状相似度
        shape_sim = self._calculate_shape_similarity(
            stroke.contour, template.contour,
            stroke_features.get('skeleton'), template.skeleton
        )
        details['shape_similarity'] = shape_sim
        
        # 纹理相似度（简化实现）
        texture_sim = self._calculate_texture_similarity(
            stroke_features.get('texture', {}), 
            template.features.get('texture', {})
        )
        details['texture_similarity'] = texture_sim
        
        # 上下文相似度（简化实现）
        context_sim = self._calculate_context_similarity(
            stroke_features.get('context', {}),
            template.features.get('context', {})
        )
        details['context_similarity'] = context_sim
        
        # 计算总相似度
        total_similarity = (
            geometric_sim * self.feature_weights['geometric'] +
            shape_sim * self.feature_weights['shape'] +
            texture_sim * self.feature_weights['texture'] +
            context_sim * self.feature_weights['context']
        )
        
        # 记录各特征相似度
        details['feature_similarities'] = {
            'geometric': geometric_sim,
            'shape': shape_sim,
            'texture': texture_sim,
            'context': context_sim
        }
        
        return max(0.0, min(1.0, total_similarity)), details
    
    def _extract_matching_features(self, stroke: Stroke) -> Dict[str, Any]:
        """
        提取用于匹配的特征
        
        Args:
            stroke: 笔触
            
        Returns:
            特征字典
        """
        features = {
            'geometric': {
                'area': stroke.area,
                'perimeter': stroke.perimeter,
                'length': stroke.length,
                'width': stroke.width,
                'aspect_ratio': stroke.length / max(stroke.width, 1e-6),
                'angle': stroke.angle
            },
            'shape': {
                'contour': stroke.contour,
                'fourier_descriptor': self._calculate_fourier_descriptor(stroke.contour)
            }
        }
        
        # 添加骨架特征（如果可用）
        if hasattr(stroke, 'skeleton') and stroke.skeleton is not None:
            features['skeleton'] = stroke.skeleton
        
        # 添加纹理特征（简化）
        features['texture'] = self._extract_texture_features(stroke)
        
        # 添加上下文特征（简化）
        features['context'] = self._extract_context_features(stroke)
        
        return features
    
    def _calculate_geometric_similarity(self, features1: Dict[str, float], 
                                      features2: Dict[str, float]) -> float:
        """
        计算几何特征相似度
        
        Args:
            features1: 第一组几何特征
            features2: 第二组几何特征
            
        Returns:
            几何相似度
        """
        similarities = []
        
        for feature_name, weight in self.geometric_weights.items():
            if feature_name in features1 and feature_name in features2:
                val1 = features1[feature_name]
                val2 = features2[feature_name]
                
                # 归一化相似度计算
                if feature_name == 'angle':
                    # 角度特殊处理（考虑周期性）
                    diff = abs(val1 - val2)
                    diff = min(diff, 360 - diff)
                    similarity = 1.0 - (diff / 180.0)
                else:
                    # 其他特征使用相对差异
                    max_val = max(abs(val1), abs(val2), 1e-6)
                    similarity = 1.0 - abs(val1 - val2) / max_val
                
                similarities.append(similarity * weight)
        
        return sum(similarities) if similarities else 0.0
    
    def _calculate_shape_similarity(self, contour1: np.ndarray, contour2: np.ndarray,
                                  skeleton1: np.ndarray = None, 
                                  skeleton2: np.ndarray = None) -> float:
        """
        计算形状相似度
        
        Args:
            contour1: 第一个轮廓
            contour2: 第二个轮廓
            skeleton1: 第一个骨架
            skeleton2: 第二个骨架
            
        Returns:
            形状相似度
        """
        similarities = []
        
        # 轮廓相似度
        if len(contour1) > 0 and len(contour2) > 0:
            contour_sim = self._calculate_contour_similarity(contour1, contour2)
            similarities.append(contour_sim * self.shape_weights['contour_similarity'])
        
        # 骨架相似度
        if skeleton1 is not None and skeleton2 is not None:
            skeleton_sim = self._calculate_skeleton_similarity(skeleton1, skeleton2)
            similarities.append(skeleton_sim * self.shape_weights['skeleton_similarity'])
        
        # 傅里叶描述子相似度
        fd1 = self._calculate_fourier_descriptor(contour1)
        fd2 = self._calculate_fourier_descriptor(contour2)
        if fd1 is not None and fd2 is not None:
            fd_sim = self._calculate_fourier_similarity(fd1, fd2)
            similarities.append(fd_sim * self.shape_weights['fourier_descriptor'])
        
        return sum(similarities) if similarities else 0.0
    
    def _calculate_contour_similarity(self, contour1: np.ndarray, 
                                    contour2: np.ndarray) -> float:
        """
        计算轮廓相似度
        
        Args:
            contour1: 第一个轮廓
            contour2: 第二个轮廓
            
        Returns:
            轮廓相似度
        """
        try:
            # 使用OpenCV的matchShapes函数
            similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            # 转换为0-1范围的相似度
            return max(0.0, 1.0 - similarity)
        except Exception as e:
            self.logger.warning(f"Error calculating contour similarity: {e}")
            return 0.0
    
    def _calculate_skeleton_similarity(self, skeleton1: np.ndarray, 
                                     skeleton2: np.ndarray) -> float:
        """
        计算骨架相似度
        
        Args:
            skeleton1: 第一个骨架
            skeleton2: 第二个骨架
            
        Returns:
            骨架相似度
        """
        try:
            # 简化实现：使用点集之间的Hausdorff距离
            if len(skeleton1) == 0 or len(skeleton2) == 0:
                return 0.0
            
            # 归一化骨架点
            skel1_norm = self._normalize_points(skeleton1)
            skel2_norm = self._normalize_points(skeleton2)
            
            # 计算双向Hausdorff距离
            dist1 = self._hausdorff_distance(skel1_norm, skel2_norm)
            dist2 = self._hausdorff_distance(skel2_norm, skel1_norm)
            hausdorff_dist = max(dist1, dist2)
            
            # 转换为相似度
            return max(0.0, 1.0 - hausdorff_dist)
            
        except Exception as e:
            self.logger.warning(f"Error calculating skeleton similarity: {e}")
            return 0.0
    
    def _calculate_fourier_descriptor(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        计算傅里叶描述子
        
        Args:
            contour: 轮廓点
            
        Returns:
            傅里叶描述子
        """
        try:
            if len(contour) < 3:
                return None
            
            # 将轮廓转换为复数序列
            contour_points = contour.reshape(-1, 2)
            complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]
            
            # 计算FFT
            fft_result = np.fft.fft(complex_contour)
            
            # 取前几个系数作为描述子（归一化）
            n_descriptors = min(16, len(fft_result) // 2)
            descriptors = fft_result[1:n_descriptors+1]  # 跳过DC分量
            
            # 归一化（平移和缩放不变性）
            if len(descriptors) > 0 and abs(descriptors[0]) > 1e-6:
                descriptors = descriptors / abs(descriptors[0])
            
            return np.abs(descriptors)  # 只保留幅度（旋转不变性）
            
        except Exception as e:
            self.logger.warning(f"Error calculating Fourier descriptor: {e}")
            return None
    
    def _calculate_fourier_similarity(self, fd1: np.ndarray, fd2: np.ndarray) -> float:
        """
        计算傅里叶描述子相似度
        
        Args:
            fd1: 第一个傅里叶描述子
            fd2: 第二个傅里叶描述子
            
        Returns:
            相似度
        """
        try:
            # 确保长度一致
            min_len = min(len(fd1), len(fd2))
            if min_len == 0:
                return 0.0
            
            fd1_truncated = fd1[:min_len]
            fd2_truncated = fd2[:min_len]
            
            # 使用余弦相似度
            similarity = 1.0 - cosine(fd1_truncated, fd2_truncated)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning(f"Error calculating Fourier similarity: {e}")
            return 0.0
    
    def _calculate_texture_similarity(self, texture1: Dict[str, Any], 
                                    texture2: Dict[str, Any]) -> float:
        """
        计算纹理相似度（简化实现）
        
        Args:
            texture1: 第一组纹理特征
            texture2: 第二组纹理特征
            
        Returns:
            纹理相似度
        """
        # 简化实现：返回固定值
        return 0.5
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """
        计算上下文相似度（简化实现）
        
        Args:
            context1: 第一组上下文特征
            context2: 第二组上下文特征
            
        Returns:
            上下文相似度
        """
        # 简化实现：返回固定值
        return 0.5
    
    def _extract_texture_features(self, stroke: Stroke) -> Dict[str, Any]:
        """
        提取纹理特征（简化实现）
        
        Args:
            stroke: 笔触
            
        Returns:
            纹理特征
        """
        return {'placeholder': True}
    
    def _extract_context_features(self, stroke: Stroke) -> Dict[str, Any]:
        """
        提取上下文特征（简化实现）
        
        Args:
            stroke: 笔触
            
        Returns:
            上下文特征
        """
        return {'placeholder': True}
    
    def _calculate_confidence(self, similarity_score: float, 
                            details: Dict[str, Any]) -> float:
        """
        计算匹配置信度
        
        Args:
            similarity_score: 相似度分数
            details: 匹配详细信息
            
        Returns:
            置信度
        """
        # 基于相似度分数和特征一致性计算置信度
        feature_consistency = np.std(list(details['feature_similarities'].values()))
        confidence = similarity_score * (1.0 - feature_consistency)
        return max(0.0, min(1.0, confidence))
    
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        归一化点集
        
        Args:
            points: 点集
            
        Returns:
            归一化后的点集
        """
        if len(points) == 0:
            return points
        
        points = points.reshape(-1, 2)
        
        # 中心化
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # 缩放
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        if max_dist > 1e-6:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized
    
    def _hausdorff_distance(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """
        计算Hausdorff距离
        
        Args:
            points1: 第一组点
            points2: 第二组点
            
        Returns:
            Hausdorff距离
        """
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')
        
        # 计算每个点到另一组点的最小距离
        max_min_dist = 0.0
        for p1 in points1:
            min_dist = float('inf')
            for p2 in points2:
                dist = euclidean(p1, p2)
                min_dist = min(min_dist, dist)
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist