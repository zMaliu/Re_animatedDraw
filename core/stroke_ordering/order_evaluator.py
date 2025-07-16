# -*- coding: utf-8 -*-
"""
排序结果评估器

实现论文中的排序质量评估：
1. 艺术一致性评估
2. 视觉连贯性评估
3. 绘制效率评估
4. 用户偏好评估
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
import cv2


class EvaluationMetric(Enum):
    """
    评估指标枚举
    """
    ARTISTIC_CONSISTENCY = "artistic_consistency"
    VISUAL_COHERENCE = "visual_coherence"
    DRAWING_EFFICIENCY = "drawing_efficiency"
    STAGE_COMPLIANCE = "stage_compliance"
    SPATIAL_LOCALITY = "spatial_locality"
    INK_TRANSITION = "ink_transition"
    USER_PREFERENCE = "user_preference"


@dataclass
class EvaluationResult:
    """
    评估结果数据结构
    """
    metric: EvaluationMetric
    score: float  # 0-1之间的得分
    details: Dict[str, Any]
    description: str = ""
    
    def __post_init__(self):
        # 确保得分在有效范围内
        self.score = max(0.0, min(1.0, self.score))


@dataclass
class OverallEvaluation:
    """
    整体评估结果
    """
    overall_score: float
    metric_scores: Dict[EvaluationMetric, EvaluationResult]
    weighted_score: float
    ranking_quality: str  # "excellent", "good", "fair", "poor"
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # 确定质量等级
        if self.overall_score >= 0.9:
            self.ranking_quality = "excellent"
        elif self.overall_score >= 0.7:
            self.ranking_quality = "good"
        elif self.overall_score >= 0.5:
            self.ranking_quality = "fair"
        else:
            self.ranking_quality = "poor"


class OrderEvaluator:
    """
    排序结果评估器
    
    评估笔触排序的质量和艺术一致性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 评估权重
        self.metric_weights = config.get('metric_weights', {
            EvaluationMetric.ARTISTIC_CONSISTENCY: 0.25,
            EvaluationMetric.VISUAL_COHERENCE: 0.20,
            EvaluationMetric.DRAWING_EFFICIENCY: 0.15,
            EvaluationMetric.STAGE_COMPLIANCE: 0.20,
            EvaluationMetric.SPATIAL_LOCALITY: 0.10,
            EvaluationMetric.INK_TRANSITION: 0.10
        })
        
        # 评估参数
        self.spatial_threshold = config.get('spatial_threshold', 100.0)
        self.ink_transition_weight = config.get('ink_transition_weight', 1.0)
        self.stage_weight = config.get('stage_weight', 2.0)
        
        # 缓存
        self._distance_cache = {}
        self._feature_cache = {}
    
    def evaluate_order(self, order: List[int], stroke_features: Dict[int, Any], 
                      stage_mapping: Optional[Dict[int, int]] = None,
                      reference_order: Optional[List[int]] = None) -> OverallEvaluation:
        """
        评估笔触排序
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征字典
            stage_mapping: 阶段映射
            reference_order: 参考排序（用于对比）
            
        Returns:
            OverallEvaluation: 整体评估结果
        """
        if not order or not stroke_features:
            return self._create_empty_evaluation()
        
        # 计算各项指标
        metric_results = {}
        
        # 艺术一致性评估
        metric_results[EvaluationMetric.ARTISTIC_CONSISTENCY] = \
            self._evaluate_artistic_consistency(order, stroke_features)
        
        # 视觉连贯性评估
        metric_results[EvaluationMetric.VISUAL_COHERENCE] = \
            self._evaluate_visual_coherence(order, stroke_features)
        
        # 绘制效率评估
        metric_results[EvaluationMetric.DRAWING_EFFICIENCY] = \
            self._evaluate_drawing_efficiency(order, stroke_features)
        
        # 阶段遵循性评估
        if stage_mapping:
            metric_results[EvaluationMetric.STAGE_COMPLIANCE] = \
                self._evaluate_stage_compliance(order, stage_mapping)
        
        # 空间局部性评估
        metric_results[EvaluationMetric.SPATIAL_LOCALITY] = \
            self._evaluate_spatial_locality(order, stroke_features)
        
        # 墨色过渡评估
        metric_results[EvaluationMetric.INK_TRANSITION] = \
            self._evaluate_ink_transition(order, stroke_features)
        
        # 用户偏好评估（如果有参考排序）
        if reference_order:
            metric_results[EvaluationMetric.USER_PREFERENCE] = \
                self._evaluate_user_preference(order, reference_order)
        
        # 计算整体得分
        overall_score = self._calculate_overall_score(metric_results)
        weighted_score = self._calculate_weighted_score(metric_results)
        
        # 生成建议
        recommendations = self._generate_recommendations(metric_results)
        
        return OverallEvaluation(
            overall_score=overall_score,
            metric_scores=metric_results,
            weighted_score=weighted_score,
            ranking_quality="",  # 将在__post_init__中设置
            recommendations=recommendations,
            metadata={
                'order_length': len(order),
                'evaluation_timestamp': np.datetime64('now'),
                'config': self.config
            }
        )
    
    def _evaluate_artistic_consistency(self, order: List[int], 
                                     stroke_features: Dict[int, Any]) -> EvaluationResult:
        """
        评估艺术一致性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            EvaluationResult: 评估结果
        """
        if len(order) < 2:
            return EvaluationResult(
                metric=EvaluationMetric.ARTISTIC_CONSISTENCY,
                score=1.0,
                details={'reason': 'insufficient_strokes'},
                description="Insufficient strokes for consistency evaluation"
            )
        
        # 计算相邻笔触的特征相似性
        similarities = []
        transitions = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if stroke1_id in stroke_features and stroke2_id in stroke_features:
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                # 计算特征相似性
                similarity = self._calculate_feature_similarity(features1, features2)
                similarities.append(similarity)
                
                # 计算过渡合理性
                transition = self._calculate_transition_reasonableness(features1, features2)
                transitions.append(transition)
        
        if not similarities:
            return EvaluationResult(
                metric=EvaluationMetric.ARTISTIC_CONSISTENCY,
                score=0.0,
                details={'reason': 'no_valid_features'},
                description="No valid features for consistency evaluation"
            )
        
        # 计算平均相似性和过渡合理性
        avg_similarity = np.mean(similarities)
        avg_transition = np.mean(transitions)
        
        # 综合得分
        consistency_score = 0.6 * avg_similarity + 0.4 * avg_transition
        
        return EvaluationResult(
            metric=EvaluationMetric.ARTISTIC_CONSISTENCY,
            score=consistency_score,
            details={
                'avg_similarity': avg_similarity,
                'avg_transition': avg_transition,
                'similarity_std': np.std(similarities),
                'transition_std': np.std(transitions),
                'num_comparisons': len(similarities)
            },
            description=f"Artistic consistency score: {consistency_score:.3f}"
        )
    
    def _evaluate_visual_coherence(self, order: List[int], 
                                 stroke_features: Dict[int, Any]) -> EvaluationResult:
        """
        评估视觉连贯性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            EvaluationResult: 评估结果
        """
        if len(order) < 3:
            return EvaluationResult(
                metric=EvaluationMetric.VISUAL_COHERENCE,
                score=1.0,
                details={'reason': 'insufficient_strokes'},
                description="Insufficient strokes for coherence evaluation"
            )
        
        # 计算颜色连贯性
        color_coherence = self._calculate_color_coherence(order, stroke_features)
        
        # 计算形状连贯性
        shape_coherence = self._calculate_shape_coherence(order, stroke_features)
        
        # 计算尺寸连贯性
        size_coherence = self._calculate_size_coherence(order, stroke_features)
        
        # 综合连贯性得分
        coherence_score = (color_coherence + shape_coherence + size_coherence) / 3.0
        
        return EvaluationResult(
            metric=EvaluationMetric.VISUAL_COHERENCE,
            score=coherence_score,
            details={
                'color_coherence': color_coherence,
                'shape_coherence': shape_coherence,
                'size_coherence': size_coherence
            },
            description=f"Visual coherence score: {coherence_score:.3f}"
        )
    
    def _evaluate_drawing_efficiency(self, order: List[int], 
                                   stroke_features: Dict[int, Any]) -> EvaluationResult:
        """
        评估绘制效率
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            EvaluationResult: 评估结果
        """
        if len(order) < 2:
            return EvaluationResult(
                metric=EvaluationMetric.DRAWING_EFFICIENCY,
                score=1.0,
                details={'reason': 'insufficient_strokes'},
                description="Insufficient strokes for efficiency evaluation"
            )
        
        # 计算总移动距离
        total_distance = 0.0
        valid_moves = 0
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                # 获取笔触中心位置
                pos1 = self._get_stroke_center(features1)
                pos2 = self._get_stroke_center(features2)
                
                if pos1 is not None and pos2 is not None:
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    total_distance += distance
                    valid_moves += 1
        
        if valid_moves == 0:
            return EvaluationResult(
                metric=EvaluationMetric.DRAWING_EFFICIENCY,
                score=0.0,
                details={'reason': 'no_valid_positions'},
                description="No valid positions for efficiency evaluation"
            )
        
        # 计算平均移动距离
        avg_distance = total_distance / valid_moves
        
        # 计算效率得分（距离越小效率越高）
        # 使用指数衰减函数
        efficiency_score = np.exp(-avg_distance / self.spatial_threshold)
        
        return EvaluationResult(
            metric=EvaluationMetric.DRAWING_EFFICIENCY,
            score=efficiency_score,
            details={
                'total_distance': total_distance,
                'avg_distance': avg_distance,
                'valid_moves': valid_moves,
                'spatial_threshold': self.spatial_threshold
            },
            description=f"Drawing efficiency score: {efficiency_score:.3f}"
        )
    
    def _evaluate_stage_compliance(self, order: List[int], 
                                 stage_mapping: Dict[int, int]) -> EvaluationResult:
        """
        评估阶段遵循性
        
        Args:
            order: 笔触排序
            stage_mapping: 阶段映射
            
        Returns:
            EvaluationResult: 评估结果
        """
        if not stage_mapping:
            return EvaluationResult(
                metric=EvaluationMetric.STAGE_COMPLIANCE,
                score=1.0,
                details={'reason': 'no_stage_mapping'},
                description="No stage mapping provided"
            )
        
        # 检查阶段顺序违反
        violations = 0
        total_transitions = 0
        stage_sequence = []
        
        for stroke_id in order:
            if stroke_id in stage_mapping:
                stage_sequence.append(stage_mapping[stroke_id])
        
        if len(stage_sequence) < 2:
            return EvaluationResult(
                metric=EvaluationMetric.STAGE_COMPLIANCE,
                score=1.0,
                details={'reason': 'insufficient_stages'},
                description="Insufficient stages for compliance evaluation"
            )
        
        # 计算阶段违反次数
        for i in range(len(stage_sequence) - 1):
            current_stage = stage_sequence[i]
            next_stage = stage_sequence[i + 1]
            total_transitions += 1
            
            if next_stage < current_stage:
                violations += 1
        
        # 计算遵循性得分
        if total_transitions > 0:
            compliance_score = 1.0 - (violations / total_transitions)
        else:
            compliance_score = 1.0
        
        return EvaluationResult(
            metric=EvaluationMetric.STAGE_COMPLIANCE,
            score=compliance_score,
            details={
                'violations': violations,
                'total_transitions': total_transitions,
                'stage_sequence': stage_sequence,
                'violation_rate': violations / max(total_transitions, 1)
            },
            description=f"Stage compliance score: {compliance_score:.3f}"
        )
    
    def _evaluate_spatial_locality(self, order: List[int], 
                                 stroke_features: Dict[int, Any]) -> EvaluationResult:
        """
        评估空间局部性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            EvaluationResult: 评估结果
        """
        if len(order) < 2:
            return EvaluationResult(
                metric=EvaluationMetric.SPATIAL_LOCALITY,
                score=1.0,
                details={'reason': 'insufficient_strokes'},
                description="Insufficient strokes for locality evaluation"
            )
        
        # 计算相邻笔触的空间距离
        distances = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                pos1 = self._get_stroke_center(features1)
                pos2 = self._get_stroke_center(features2)
                
                if pos1 is not None and pos2 is not None:
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    distances.append(distance)
        
        if not distances:
            return EvaluationResult(
                metric=EvaluationMetric.SPATIAL_LOCALITY,
                score=0.0,
                details={'reason': 'no_valid_distances'},
                description="No valid distances for locality evaluation"
            )
        
        # 计算局部性得分
        avg_distance = np.mean(distances)
        locality_score = np.exp(-avg_distance / self.spatial_threshold)
        
        return EvaluationResult(
            metric=EvaluationMetric.SPATIAL_LOCALITY,
            score=locality_score,
            details={
                'avg_distance': avg_distance,
                'distance_std': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'num_distances': len(distances)
            },
            description=f"Spatial locality score: {locality_score:.3f}"
        )
    
    def _evaluate_ink_transition(self, order: List[int], 
                               stroke_features: Dict[int, Any]) -> EvaluationResult:
        """
        评估墨色过渡
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            EvaluationResult: 评估结果
        """
        if len(order) < 2:
            return EvaluationResult(
                metric=EvaluationMetric.INK_TRANSITION,
                score=1.0,
                details={'reason': 'insufficient_strokes'},
                description="Insufficient strokes for ink transition evaluation"
            )
        
        # 计算墨色过渡合理性
        wetness_transitions = []
        thickness_transitions = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                # 湿度过渡（湿到干）
                wetness1 = self._get_feature_value(features1, 'wetness', 0.5)
                wetness2 = self._get_feature_value(features2, 'wetness', 0.5)
                wetness_transition = self._evaluate_wetness_transition(wetness1, wetness2)
                wetness_transitions.append(wetness_transition)
                
                # 厚度过渡（粗到细）
                thickness1 = self._get_feature_value(features1, 'thickness', 0.5)
                thickness2 = self._get_feature_value(features2, 'thickness', 0.5)
                thickness_transition = self._evaluate_thickness_transition(thickness1, thickness2)
                thickness_transitions.append(thickness_transition)
        
        if not wetness_transitions or not thickness_transitions:
            return EvaluationResult(
                metric=EvaluationMetric.INK_TRANSITION,
                score=0.0,
                details={'reason': 'no_valid_transitions'},
                description="No valid transitions for ink evaluation"
            )
        
        # 计算平均过渡得分
        avg_wetness_transition = np.mean(wetness_transitions)
        avg_thickness_transition = np.mean(thickness_transitions)
        
        # 综合墨色过渡得分
        ink_transition_score = (avg_wetness_transition + avg_thickness_transition) / 2.0
        
        return EvaluationResult(
            metric=EvaluationMetric.INK_TRANSITION,
            score=ink_transition_score,
            details={
                'avg_wetness_transition': avg_wetness_transition,
                'avg_thickness_transition': avg_thickness_transition,
                'wetness_std': np.std(wetness_transitions),
                'thickness_std': np.std(thickness_transitions),
                'num_transitions': len(wetness_transitions)
            },
            description=f"Ink transition score: {ink_transition_score:.3f}"
        )
    
    def _evaluate_user_preference(self, order: List[int], 
                                reference_order: List[int]) -> EvaluationResult:
        """
        评估用户偏好（与参考排序的相似性）
        
        Args:
            order: 当前排序
            reference_order: 参考排序
            
        Returns:
            EvaluationResult: 评估结果
        """
        if not reference_order or len(order) != len(reference_order):
            return EvaluationResult(
                metric=EvaluationMetric.USER_PREFERENCE,
                score=0.0,
                details={'reason': 'invalid_reference'},
                description="Invalid reference order"
            )
        
        # 计算Spearman秩相关系数
        try:
            spearman_corr, spearman_p = spearmanr(order, reference_order)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except:
            spearman_corr = 0.0
            spearman_p = 1.0
        
        # 计算Kendall's tau
        try:
            kendall_corr, kendall_p = kendalltau(order, reference_order)
            if np.isnan(kendall_corr):
                kendall_corr = 0.0
        except:
            kendall_corr = 0.0
            kendall_p = 1.0
        
        # 计算位置差异
        position_diffs = []
        for i, stroke_id in enumerate(order):
            try:
                ref_pos = reference_order.index(stroke_id)
                position_diffs.append(abs(i - ref_pos))
            except ValueError:
                position_diffs.append(len(order))  # 最大惩罚
        
        avg_position_diff = np.mean(position_diffs)
        normalized_position_diff = avg_position_diff / len(order)
        
        # 综合用户偏好得分
        preference_score = (abs(spearman_corr) + abs(kendall_corr) + (1 - normalized_position_diff)) / 3.0
        
        return EvaluationResult(
            metric=EvaluationMetric.USER_PREFERENCE,
            score=preference_score,
            details={
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'kendall_correlation': kendall_corr,
                'kendall_p_value': kendall_p,
                'avg_position_diff': avg_position_diff,
                'normalized_position_diff': normalized_position_diff
            },
            description=f"User preference score: {preference_score:.3f}"
        )
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """
        计算特征相似性
        
        Args:
            features1: 第一个笔触特征
            features2: 第二个笔触特征
            
        Returns:
            float: 相似性得分
        """
        # 几何特征相似性
        area1 = self._get_feature_value(features1, 'area', 1.0)
        area2 = self._get_feature_value(features2, 'area', 1.0)
        area_similarity = 1.0 - abs(area1 - area2) / max(area1 + area2, 1e-6)
        
        # 形状特征相似性
        roundness1 = self._get_feature_value(features1, 'roundness', 0.5)
        roundness2 = self._get_feature_value(features2, 'roundness', 0.5)
        roundness_similarity = 1.0 - abs(roundness1 - roundness2)
        
        # 颜色特征相似性
        color1 = self._get_feature_value(features1, 'color', [0, 0, 0])
        color2 = self._get_feature_value(features2, 'color', [0, 0, 0])
        color_similarity = self._calculate_color_similarity(color1, color2)
        
        # 综合相似性
        similarity = (area_similarity + roundness_similarity + color_similarity) / 3.0
        return max(0.0, min(1.0, similarity))
    
    def _calculate_transition_reasonableness(self, features1: Dict[str, Any], 
                                           features2: Dict[str, Any]) -> float:
        """
        计算过渡合理性
        
        Args:
            features1: 第一个笔触特征
            features2: 第二个笔触特征
            
        Returns:
            float: 过渡合理性得分
        """
        # 湿度过渡合理性
        wetness1 = self._get_feature_value(features1, 'wetness', 0.5)
        wetness2 = self._get_feature_value(features2, 'wetness', 0.5)
        wetness_reasonableness = self._evaluate_wetness_transition(wetness1, wetness2)
        
        # 厚度过渡合理性
        thickness1 = self._get_feature_value(features1, 'thickness', 0.5)
        thickness2 = self._get_feature_value(features2, 'thickness', 0.5)
        thickness_reasonableness = self._evaluate_thickness_transition(thickness1, thickness2)
        
        # 尺寸过渡合理性
        scale1 = self._get_feature_value(features1, 'scale', 1.0)
        scale2 = self._get_feature_value(features2, 'scale', 1.0)
        scale_reasonableness = 1.0 - abs(scale1 - scale2) / max(scale1 + scale2, 1e-6)
        
        # 综合过渡合理性
        reasonableness = (wetness_reasonableness + thickness_reasonableness + scale_reasonableness) / 3.0
        return max(0.0, min(1.0, reasonableness))
    
    def _calculate_color_coherence(self, order: List[int], 
                                 stroke_features: Dict[int, Any]) -> float:
        """
        计算颜色连贯性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            float: 颜色连贯性得分
        """
        color_similarities = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                color1 = self._get_feature_value(features1, 'color', [0, 0, 0])
                color2 = self._get_feature_value(features2, 'color', [0, 0, 0])
                
                similarity = self._calculate_color_similarity(color1, color2)
                color_similarities.append(similarity)
        
        return np.mean(color_similarities) if color_similarities else 0.0
    
    def _calculate_shape_coherence(self, order: List[int], 
                                 stroke_features: Dict[int, Any]) -> float:
        """
        计算形状连贯性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            float: 形状连贯性得分
        """
        shape_similarities = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                roundness1 = self._get_feature_value(features1, 'roundness', 0.5)
                roundness2 = self._get_feature_value(features2, 'roundness', 0.5)
                
                elongation1 = self._get_feature_value(features1, 'elongation', 1.0)
                elongation2 = self._get_feature_value(features2, 'elongation', 1.0)
                
                roundness_sim = 1.0 - abs(roundness1 - roundness2)
                elongation_sim = 1.0 - abs(elongation1 - elongation2) / max(elongation1 + elongation2, 1e-6)
                
                shape_similarity = (roundness_sim + elongation_sim) / 2.0
                shape_similarities.append(shape_similarity)
        
        return np.mean(shape_similarities) if shape_similarities else 0.0
    
    def _calculate_size_coherence(self, order: List[int], 
                                stroke_features: Dict[int, Any]) -> float:
        """
        计算尺寸连贯性
        
        Args:
            order: 笔触排序
            stroke_features: 笔触特征
            
        Returns:
            float: 尺寸连贯性得分
        """
        size_similarities = []
        
        for i in range(len(order) - 1):
            stroke1_id = int(np.round(order[i]))
            stroke2_id = int(np.round(order[i + 1]))
            
            if (stroke1_id in stroke_features and stroke2_id in stroke_features):
                features1 = stroke_features[stroke1_id]
                features2 = stroke_features[stroke2_id]
                
                area1 = self._get_feature_value(features1, 'area', 1.0)
                area2 = self._get_feature_value(features2, 'area', 1.0)
                
                scale1 = self._get_feature_value(features1, 'scale', 1.0)
                scale2 = self._get_feature_value(features2, 'scale', 1.0)
                
                area_sim = 1.0 - abs(area1 - area2) / max(area1 + area2, 1e-6)
                scale_sim = 1.0 - abs(scale1 - scale2) / max(scale1 + scale2, 1e-6)
                
                size_similarity = (area_sim + scale_sim) / 2.0
                size_similarities.append(size_similarity)
        
        return np.mean(size_similarities) if size_similarities else 0.0
    
    def _evaluate_wetness_transition(self, wetness1: float, wetness2: float) -> float:
        """
        评估湿度过渡合理性
        
        Args:
            wetness1: 第一个笔触湿度
            wetness2: 第二个笔触湿度
            
        Returns:
            float: 过渡合理性得分
        """
        # 湿笔触应该先于干笔触
        if wetness1 >= wetness2:
            return 1.0  # 合理过渡
        else:
            # 惩罚反向过渡
            penalty = (wetness2 - wetness1) * 0.5
            return max(0.0, 1.0 - penalty)
    
    def _evaluate_thickness_transition(self, thickness1: float, thickness2: float) -> float:
        """
        评估厚度过渡合理性
        
        Args:
            thickness1: 第一个笔触厚度
            thickness2: 第二个笔触厚度
            
        Returns:
            float: 过渡合理性得分
        """
        # 粗笔触应该先于细笔触
        if thickness1 >= thickness2:
            return 1.0  # 合理过渡
        else:
            # 惩罚反向过渡
            penalty = (thickness2 - thickness1) * 0.3
            return max(0.0, 1.0 - penalty)
    
    def _calculate_color_similarity(self, color1: List[float], color2: List[float]) -> float:
        """
        计算颜色相似性
        
        Args:
            color1: 第一个颜色
            color2: 第二个颜色
            
        Returns:
            float: 颜色相似性
        """
        if len(color1) != len(color2) or len(color1) != 3:
            return 0.0
        
        # 计算欧几里得距离
        distance = np.linalg.norm(np.array(color1) - np.array(color2))
        
        # 归一化到0-1范围（假设颜色值在0-255范围内）
        max_distance = np.sqrt(3 * 255**2)
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def _get_stroke_center(self, features: Dict[str, Any]) -> Optional[List[float]]:
        """
        获取笔触中心位置
        
        Args:
            features: 笔触特征
            
        Returns:
            Optional[List[float]]: 中心位置
        """
        # 尝试多种可能的位置字段
        for key in ['center', 'centroid', 'position', 'bbox_center']:
            if key in features:
                pos = features[key]
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    return [float(pos[0]), float(pos[1])]
        
        # 如果有边界框，计算中心
        if 'bbox' in features:
            bbox = features['bbox']
            if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2.0
                center_y = (bbox[1] + bbox[3]) / 2.0
                return [center_x, center_y]
        
        return None
    
    def _get_feature_value(self, features: Dict[str, Any], key: str, default: Any) -> Any:
        """
        安全获取特征值
        
        Args:
            features: 特征字典
            key: 特征键
            default: 默认值
            
        Returns:
            Any: 特征值
        """
        return features.get(key, default)
    
    def _calculate_overall_score(self, metric_results: Dict[EvaluationMetric, EvaluationResult]) -> float:
        """
        计算整体得分
        
        Args:
            metric_results: 指标结果
            
        Returns:
            float: 整体得分
        """
        if not metric_results:
            return 0.0
        
        scores = [result.score for result in metric_results.values()]
        return np.mean(scores)
    
    def _calculate_weighted_score(self, metric_results: Dict[EvaluationMetric, EvaluationResult]) -> float:
        """
        计算加权得分
        
        Args:
            metric_results: 指标结果
            
        Returns:
            float: 加权得分
        """
        if not metric_results:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, result in metric_results.items():
            weight = self.metric_weights.get(metric, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / max(total_weight, 1e-6)
    
    def _generate_recommendations(self, metric_results: Dict[EvaluationMetric, EvaluationResult]) -> List[str]:
        """
        生成改进建议
        
        Args:
            metric_results: 指标结果
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        for metric, result in metric_results.items():
            if result.score < 0.5:
                if metric == EvaluationMetric.ARTISTIC_CONSISTENCY:
                    recommendations.append("Consider grouping similar strokes together")
                elif metric == EvaluationMetric.VISUAL_COHERENCE:
                    recommendations.append("Improve color and shape transitions between strokes")
                elif metric == EvaluationMetric.DRAWING_EFFICIENCY:
                    recommendations.append("Reduce spatial jumps between consecutive strokes")
                elif metric == EvaluationMetric.STAGE_COMPLIANCE:
                    recommendations.append("Follow the three-stage painting order more strictly")
                elif metric == EvaluationMetric.SPATIAL_LOCALITY:
                    recommendations.append("Draw spatially close strokes consecutively")
                elif metric == EvaluationMetric.INK_TRANSITION:
                    recommendations.append("Follow wet-to-dry and thick-to-thin ink transitions")
        
        if not recommendations:
            recommendations.append("The stroke order shows good quality across all metrics")
        
        return recommendations
    
    def _create_empty_evaluation(self) -> OverallEvaluation:
        """
        创建空评估结果
        
        Returns:
            OverallEvaluation: 空评估结果
        """
        return OverallEvaluation(
            overall_score=0.0,
            metric_scores={},
            weighted_score=0.0,
            ranking_quality="poor",
            recommendations=["No strokes or features provided for evaluation"],
            metadata={'error': 'empty_input'}
        )
    
    def compare_orders(self, order1: List[int], order2: List[int], 
                      stroke_features: Dict[int, Any],
                      stage_mapping: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
        """
        比较两个排序
        
        Args:
            order1: 第一个排序
            order2: 第二个排序
            stroke_features: 笔触特征
            stage_mapping: 阶段映射
            
        Returns:
            Dict: 比较结果
        """
        eval1 = self.evaluate_order(order1, stroke_features, stage_mapping)
        eval2 = self.evaluate_order(order2, stroke_features, stage_mapping)
        
        return {
            'order1_score': eval1.overall_score,
            'order2_score': eval2.overall_score,
            'better_order': 1 if eval1.overall_score > eval2.overall_score else 2,
            'score_difference': abs(eval1.overall_score - eval2.overall_score),
            'metric_comparison': {
                metric.value: {
                    'order1': eval1.metric_scores.get(metric, EvaluationResult(metric, 0.0, {})).score,
                    'order2': eval2.metric_scores.get(metric, EvaluationResult(metric, 0.0, {})).score
                }
                for metric in EvaluationMetric
            },
            'evaluation1': eval1,
            'evaluation2': eval2
        }
    
    def export_evaluation_report(self, evaluation: OverallEvaluation, 
                               output_path: str):
        """
        导出评估报告
        
        Args:
            evaluation: 评估结果
            output_path: 输出路径
        """
        import json
        
        report = {
            'overall_score': evaluation.overall_score,
            'weighted_score': evaluation.weighted_score,
            'ranking_quality': evaluation.ranking_quality,
            'recommendations': evaluation.recommendations,
            'metric_scores': {
                metric.value: {
                    'score': result.score,
                    'details': result.details,
                    'description': result.description
                }
                for metric, result in evaluation.metric_scores.items()
            },
            'metadata': evaluation.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Evaluation report exported to {output_path}")