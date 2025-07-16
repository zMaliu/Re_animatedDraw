# -*- coding: utf-8 -*-
"""
层级关系构建器

实现论文中的层级关系建模：
1. 定义笔触间的偏序关系
2. 基于几何、墨色、位置特征的加权组合
3. 构建笔触依赖关系
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum


class RelationType(Enum):
    """
    关系类型枚举
    """
    BEFORE = "before"  # 前置关系
    AFTER = "after"  # 后置关系
    INDEPENDENT = "independent"  # 独立关系
    CONCURRENT = "concurrent"  # 并发关系


@dataclass
class StrokeRelation:
    """
    笔触关系数据结构
    """
    source_id: int
    target_id: int
    relation_type: RelationType
    confidence: float
    features: Dict[str, float]
    

@dataclass
class HierarchyFeatures:
    """
    层级特征数据结构
    """
    # 几何关系特征
    spatial_distance: float
    size_ratio: float
    overlap_ratio: float
    containment_score: float
    
    # 墨色关系特征
    darkness_diff: float
    wetness_diff: float
    thickness_diff: float
    color_similarity: float
    
    # 位置关系特征
    centrality_diff: float
    prominence_diff: float
    layer_depth: float
    
    # 艺术规则特征
    stage_priority: float
    composition_order: float
    artistic_flow: float


class HierarchyBuilder:
    """
    层级关系构建器
    
    根据论文中的艺术原则构建笔触间的层级关系
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化层级关系构建器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 关系权重
        self.relation_weights = {
            'geometric': {
                'spatial_distance': config.get('geo_spatial_weight', 0.25),
                'size_ratio': config.get('geo_size_weight', 0.25),
                'overlap_ratio': config.get('geo_overlap_weight', 0.25),
                'containment': config.get('geo_containment_weight', 0.25)
            },
            'ink': {
                'darkness_diff': config.get('ink_darkness_weight', 0.3),
                'wetness_diff': config.get('ink_wetness_weight', 0.3),
                'thickness_diff': config.get('ink_thickness_weight', 0.2),
                'color_similarity': config.get('ink_color_weight', 0.2)
            },
            'position': {
                'centrality_diff': config.get('pos_centrality_weight', 0.4),
                'prominence_diff': config.get('pos_prominence_weight', 0.3),
                'layer_depth': config.get('pos_layer_weight', 0.3)
            },
            'artistic': {
                'stage_priority': config.get('art_stage_weight', 0.4),
                'composition_order': config.get('art_composition_weight', 0.3),
                'artistic_flow': config.get('art_flow_weight', 0.3)
            }
        }
        
        # 全局权重
        self.global_weights = {
            'geometric': config.get('global_geometric_weight', 0.25),
            'ink': config.get('global_ink_weight', 0.3),
            'position': config.get('global_position_weight', 0.2),
            'artistic': config.get('global_artistic_weight', 0.25)
        }
        
        # 关系阈值
        self.thresholds = {
            'strong_relation': config.get('strong_relation_threshold', 0.7),
            'weak_relation': config.get('weak_relation_threshold', 0.3),
            'independence': config.get('independence_threshold', 0.1),
            'max_distance': config.get('max_spatial_distance', 100.0)
        }
        
        # 艺术规则
        self.artistic_rules = {
            'wet_before_dry': config.get('wet_before_dry', True),
            'thick_before_thin': config.get('thick_before_thin', True),
            'dark_before_light': config.get('dark_before_light', True),
            'main_before_detail': config.get('main_before_detail', True),
            'center_before_edge': config.get('center_before_edge', True)
        }
        
    def build_hierarchy(self, stroke_features: List[Dict[str, Any]], 
                       stage_classifications: Dict[int, Any]) -> Dict[str, Any]:
        """
        构建笔触层级关系
        
        Args:
            stroke_features: 笔触特征列表
            stage_classifications: 阶段分类结果
            
        Returns:
            Dict: 层级关系数据
        """
        try:
            if not stroke_features:
                return {'relations': [], 'hierarchy_matrix': np.array([])}
            
            # 计算所有笔触对的关系特征
            hierarchy_features = self._calculate_hierarchy_features(
                stroke_features, stage_classifications
            )
            
            # 构建关系
            relations = self._build_relations(hierarchy_features, stroke_features)
            
            # 创建层级矩阵
            hierarchy_matrix = self._create_hierarchy_matrix(relations, len(stroke_features))
            
            # 验证和优化层级关系
            optimized_relations = self._optimize_hierarchy(relations, stroke_features)
            
            result = {
                'relations': optimized_relations,
                'hierarchy_matrix': hierarchy_matrix,
                'features': hierarchy_features,
                'statistics': self._calculate_hierarchy_statistics(optimized_relations)
            }
            
            self.logger.info(f"Built hierarchy with {len(optimized_relations)} relations")
            return result
            
        except Exception as e:
            self.logger.error(f"Error building hierarchy: {str(e)}")
            return {'relations': [], 'hierarchy_matrix': np.array([])}
    
    def _calculate_hierarchy_features(self, stroke_features: List[Dict[str, Any]], 
                                     stage_classifications: Dict[int, Any]) -> Dict[Tuple[int, int], HierarchyFeatures]:
        """
        计算层级特征
        
        Args:
            stroke_features: 笔触特征列表
            stage_classifications: 阶段分类结果
            
        Returns:
            Dict: 笔触对到层级特征的映射
        """
        hierarchy_features = {}
        n_strokes = len(stroke_features)
        
        for i in range(n_strokes):
            for j in range(i + 1, n_strokes):
                features_i = stroke_features[i]
                features_j = stroke_features[j]
                stage_i = stage_classifications.get(i)
                stage_j = stage_classifications.get(j)
                
                # 计算各类特征
                geometric_features = self._calculate_geometric_features(features_i, features_j)
                ink_features = self._calculate_ink_features(features_i, features_j)
                position_features = self._calculate_position_features(features_i, features_j)
                artistic_features = self._calculate_artistic_features(
                    features_i, features_j, stage_i, stage_j
                )
                
                hierarchy_feature = HierarchyFeatures(
                    # 几何特征
                    spatial_distance=geometric_features['spatial_distance'],
                    size_ratio=geometric_features['size_ratio'],
                    overlap_ratio=geometric_features['overlap_ratio'],
                    containment_score=geometric_features['containment_score'],
                    
                    # 墨色特征
                    darkness_diff=ink_features['darkness_diff'],
                    wetness_diff=ink_features['wetness_diff'],
                    thickness_diff=ink_features['thickness_diff'],
                    color_similarity=ink_features['color_similarity'],
                    
                    # 位置特征
                    centrality_diff=position_features['centrality_diff'],
                    prominence_diff=position_features['prominence_diff'],
                    layer_depth=position_features['layer_depth'],
                    
                    # 艺术特征
                    stage_priority=artistic_features['stage_priority'],
                    composition_order=artistic_features['composition_order'],
                    artistic_flow=artistic_features['artistic_flow']
                )
                
                hierarchy_features[(i, j)] = hierarchy_feature
        
        return hierarchy_features
    
    def _calculate_geometric_features(self, features_i: Dict[str, Any], 
                                     features_j: Dict[str, Any]) -> Dict[str, float]:
        """
        计算几何关系特征
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            Dict: 几何特征
        """
        # 空间距离
        center_i = features_i.get('center', (0, 0))
        center_j = features_j.get('center', (0, 0))
        spatial_distance = np.sqrt(
            (center_i[0] - center_j[0]) ** 2 + (center_i[1] - center_j[1]) ** 2
        )
        
        # 归一化距离
        normalized_distance = min(spatial_distance / self.thresholds['max_distance'], 1.0)
        
        # 尺寸比例
        area_i = features_i.get('area', 1)
        area_j = features_j.get('area', 1)
        size_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0
        
        # 重叠比例（简化计算）
        bbox_i = features_i.get('bbox', (0, 0, 0, 0))
        bbox_j = features_j.get('bbox', (0, 0, 0, 0))
        overlap_ratio = self._calculate_bbox_overlap(bbox_i, bbox_j)
        
        # 包含关系得分
        containment_score = self._calculate_containment_score(bbox_i, bbox_j)
        
        return {
            'spatial_distance': 1.0 - normalized_distance,  # 距离越近关系越强
            'size_ratio': size_ratio,
            'overlap_ratio': overlap_ratio,
            'containment_score': containment_score
        }
    
    def _calculate_ink_features(self, features_i: Dict[str, Any], 
                               features_j: Dict[str, Any]) -> Dict[str, float]:
        """
        计算墨色关系特征
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            Dict: 墨色特征
        """
        # 深色度差异
        darkness_i = features_i.get('darkness', 0)
        darkness_j = features_j.get('darkness', 0)
        darkness_diff = abs(darkness_i - darkness_j)
        
        # 湿度差异
        wetness_i = features_i.get('wetness', 0)
        wetness_j = features_j.get('wetness', 0)
        wetness_diff = abs(wetness_i - wetness_j)
        
        # 厚度差异
        thickness_i = features_i.get('thickness', 0)
        thickness_j = features_j.get('thickness', 0)
        thickness_diff = abs(thickness_i - thickness_j)
        
        # 颜色相似性
        color_i = features_i.get('color', (0, 0, 0))
        color_j = features_j.get('color', (0, 0, 0))
        color_similarity = self._calculate_color_similarity(color_i, color_j)
        
        return {
            'darkness_diff': darkness_diff,
            'wetness_diff': wetness_diff,
            'thickness_diff': thickness_diff,
            'color_similarity': color_similarity
        }
    
    def _calculate_position_features(self, features_i: Dict[str, Any], 
                                    features_j: Dict[str, Any]) -> Dict[str, float]:
        """
        计算位置关系特征
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            Dict: 位置特征
        """
        # 中心性差异
        centrality_i = features_i.get('centrality', 0)
        centrality_j = features_j.get('centrality', 0)
        centrality_diff = abs(centrality_i - centrality_j)
        
        # 显著性差异
        prominence_i = features_i.get('prominence', 0)
        prominence_j = features_j.get('prominence', 0)
        prominence_diff = abs(prominence_i - prominence_j)
        
        # 层次深度（基于z-order或重叠关系）
        layer_depth = self._calculate_layer_depth(features_i, features_j)
        
        return {
            'centrality_diff': centrality_diff,
            'prominence_diff': prominence_diff,
            'layer_depth': layer_depth
        }
    
    def _calculate_artistic_features(self, features_i: Dict[str, Any], 
                                    features_j: Dict[str, Any],
                                    stage_i: Any, stage_j: Any) -> Dict[str, float]:
        """
        计算艺术关系特征
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            stage_i: 笔触i的阶段
            stage_j: 笔触j的阶段
            
        Returns:
            Dict: 艺术特征
        """
        # 阶段优先级
        stage_priority = self._calculate_stage_priority(stage_i, stage_j)
        
        # 构图顺序
        composition_order = self._calculate_composition_order(features_i, features_j)
        
        # 艺术流动性
        artistic_flow = self._calculate_artistic_flow(features_i, features_j)
        
        return {
            'stage_priority': stage_priority,
            'composition_order': composition_order,
            'artistic_flow': artistic_flow
        }
    
    def _calculate_bbox_overlap(self, bbox_i: Tuple[int, int, int, int], 
                               bbox_j: Tuple[int, int, int, int]) -> float:
        """
        计算边界框重叠比例
        
        Args:
            bbox_i: 边界框i (x, y, w, h)
            bbox_j: 边界框j (x, y, w, h)
            
        Returns:
            float: 重叠比例
        """
        x1_i, y1_i, w_i, h_i = bbox_i
        x2_i, y2_i = x1_i + w_i, y1_i + h_i
        
        x1_j, y1_j, w_j, h_j = bbox_j
        x2_j, y2_j = x1_j + w_j, y1_j + h_j
        
        # 计算重叠区域
        overlap_x1 = max(x1_i, x1_j)
        overlap_y1 = max(y1_i, y1_j)
        overlap_x2 = min(x2_i, x2_j)
        overlap_y2 = min(y2_i, y2_j)
        
        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            total_area = w_i * h_i + w_j * h_j - overlap_area
            return overlap_area / total_area if total_area > 0 else 0
        
        return 0.0
    
    def _calculate_containment_score(self, bbox_i: Tuple[int, int, int, int], 
                                    bbox_j: Tuple[int, int, int, int]) -> float:
        """
        计算包含关系得分
        
        Args:
            bbox_i: 边界框i
            bbox_j: 边界框j
            
        Returns:
            float: 包含得分
        """
        x1_i, y1_i, w_i, h_i = bbox_i
        x2_i, y2_i = x1_i + w_i, y1_i + h_i
        
        x1_j, y1_j, w_j, h_j = bbox_j
        x2_j, y2_j = x1_j + w_j, y1_j + h_j
        
        # 检查i是否包含j
        if x1_i <= x1_j and y1_i <= y1_j and x2_i >= x2_j and y2_i >= y2_j:
            return 1.0  # i完全包含j
        
        # 检查j是否包含i
        if x1_j <= x1_i and y1_j <= y1_i and x2_j >= x2_i and y2_j >= y2_i:
            return -1.0  # j完全包含i
        
        return 0.0  # 无包含关系
    
    def _calculate_color_similarity(self, color_i: Tuple[float, float, float], 
                                   color_j: Tuple[float, float, float]) -> float:
        """
        计算颜色相似性
        
        Args:
            color_i: 颜色i (R, G, B)
            color_j: 颜色j (R, G, B)
            
        Returns:
            float: 颜色相似性 (0-1)
        """
        # 欧几里得距离
        distance = np.sqrt(
            (color_i[0] - color_j[0]) ** 2 +
            (color_i[1] - color_j[1]) ** 2 +
            (color_i[2] - color_j[2]) ** 2
        )
        
        # 归一化到0-1范围
        max_distance = np.sqrt(3 * 255 ** 2)  # RGB最大距离
        similarity = 1.0 - (distance / max_distance)
        
        return max(similarity, 0.0)
    
    def _calculate_layer_depth(self, features_i: Dict[str, Any], 
                              features_j: Dict[str, Any]) -> float:
        """
        计算层次深度
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            float: 层次深度差异
        """
        # 基于深色度和厚度推断层次
        darkness_i = features_i.get('darkness', 0)
        darkness_j = features_j.get('darkness', 0)
        thickness_i = features_i.get('thickness', 0)
        thickness_j = features_j.get('thickness', 0)
        
        # 深色且厚的笔触通常在下层
        depth_i = darkness_i * 0.6 + thickness_i * 0.4
        depth_j = darkness_j * 0.6 + thickness_j * 0.4
        
        return depth_i - depth_j
    
    def _calculate_stage_priority(self, stage_i: Any, stage_j: Any) -> float:
        """
        计算阶段优先级
        
        Args:
            stage_i: 阶段i
            stage_j: 阶段j
            
        Returns:
            float: 优先级差异 (-1到1)
        """
        # 阶段优先级：主要 > 细节 > 装饰
        stage_order = {
            'main': 3,
            'detail': 2,
            'decoration': 1
        }
        
        order_i = stage_order.get(str(stage_i).lower(), 2)
        order_j = stage_order.get(str(stage_j).lower(), 2)
        
        # 归一化到-1到1范围
        return (order_i - order_j) / 2.0
    
    def _calculate_composition_order(self, features_i: Dict[str, Any], 
                                    features_j: Dict[str, Any]) -> float:
        """
        计算构图顺序
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            float: 构图顺序得分
        """
        # 基于位置和重要性的构图顺序
        centrality_i = features_i.get('centrality', 0)
        centrality_j = features_j.get('centrality', 0)
        prominence_i = features_i.get('prominence', 0)
        prominence_j = features_j.get('prominence', 0)
        
        # 中心且显著的笔触优先
        importance_i = centrality_i * 0.6 + prominence_i * 0.4
        importance_j = centrality_j * 0.6 + prominence_j * 0.4
        
        return importance_i - importance_j
    
    def _calculate_artistic_flow(self, features_i: Dict[str, Any], 
                                features_j: Dict[str, Any]) -> float:
        """
        计算艺术流动性
        
        Args:
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            float: 艺术流动性得分
        """
        # 基于笔触方向和连续性的流动性
        # 这里简化为基于位置的流动性
        center_i = features_i.get('center', (0, 0))
        center_j = features_j.get('center', (0, 0))
        
        # 计算方向向量
        direction_vector = (center_j[0] - center_i[0], center_j[1] - center_i[1])
        distance = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
        
        if distance > 0:
            # 归一化方向
            normalized_direction = (direction_vector[0] / distance, direction_vector[1] / distance)
            
            # 基于方向的流动性（这里简化处理）
            flow_score = abs(normalized_direction[0]) + abs(normalized_direction[1])
            return min(flow_score, 1.0)
        
        return 0.0
    
    def _build_relations(self, hierarchy_features: Dict[Tuple[int, int], HierarchyFeatures],
                        stroke_features: List[Dict[str, Any]]) -> List[StrokeRelation]:
        """
        构建笔触关系
        
        Args:
            hierarchy_features: 层级特征
            stroke_features: 笔触特征列表
            
        Returns:
            List[StrokeRelation]: 关系列表
        """
        relations = []
        
        for (i, j), features in hierarchy_features.items():
            # 计算关系强度
            relation_strength = self._calculate_relation_strength(features)
            
            # 确定关系类型
            relation_type = self._determine_relation_type(features, stroke_features[i], stroke_features[j])
            
            # 计算置信度
            confidence = self._calculate_relation_confidence(features, relation_strength)
            
            # 创建关系
            if confidence > self.thresholds['weak_relation']:
                relation = StrokeRelation(
                    source_id=i,
                    target_id=j,
                    relation_type=relation_type,
                    confidence=confidence,
                    features={
                        'strength': relation_strength,
                        'geometric_score': self._calculate_geometric_score(features),
                        'ink_score': self._calculate_ink_score(features),
                        'position_score': self._calculate_position_score(features),
                        'artistic_score': self._calculate_artistic_score(features)
                    }
                )
                relations.append(relation)
        
        return relations
    
    def _calculate_relation_strength(self, features: HierarchyFeatures) -> float:
        """
        计算关系强度
        
        Args:
            features: 层级特征
            
        Returns:
            float: 关系强度
        """
        # 计算各类特征得分
        geometric_score = self._calculate_geometric_score(features)
        ink_score = self._calculate_ink_score(features)
        position_score = self._calculate_position_score(features)
        artistic_score = self._calculate_artistic_score(features)
        
        # 加权组合
        total_strength = (
            self.global_weights['geometric'] * geometric_score +
            self.global_weights['ink'] * ink_score +
            self.global_weights['position'] * position_score +
            self.global_weights['artistic'] * artistic_score
        )
        
        return min(max(total_strength, 0.0), 1.0)
    
    def _calculate_geometric_score(self, features: HierarchyFeatures) -> float:
        """
        计算几何得分
        
        Args:
            features: 层级特征
            
        Returns:
            float: 几何得分
        """
        weights = self.relation_weights['geometric']
        
        score = (
            weights['spatial_distance'] * features.spatial_distance +
            weights['size_ratio'] * features.size_ratio +
            weights['overlap_ratio'] * features.overlap_ratio +
            weights['containment'] * abs(features.containment_score)
        )
        
        return score
    
    def _calculate_ink_score(self, features: HierarchyFeatures) -> float:
        """
        计算墨色得分
        
        Args:
            features: 层级特征
            
        Returns:
            float: 墨色得分
        """
        weights = self.relation_weights['ink']
        
        # 相似性得分（差异越小，相似性越高）
        darkness_similarity = 1.0 - features.darkness_diff
        wetness_similarity = 1.0 - features.wetness_diff
        thickness_similarity = 1.0 - features.thickness_diff
        
        score = (
            weights['darkness_diff'] * darkness_similarity +
            weights['wetness_diff'] * wetness_similarity +
            weights['thickness_diff'] * thickness_similarity +
            weights['color_similarity'] * features.color_similarity
        )
        
        return score
    
    def _calculate_position_score(self, features: HierarchyFeatures) -> float:
        """
        计算位置得分
        
        Args:
            features: 层级特征
            
        Returns:
            float: 位置得分
        """
        weights = self.relation_weights['position']
        
        # 位置相关性得分
        centrality_relation = 1.0 - features.centrality_diff
        prominence_relation = 1.0 - features.prominence_diff
        layer_relation = abs(features.layer_depth)  # 层次差异
        
        score = (
            weights['centrality_diff'] * centrality_relation +
            weights['prominence_diff'] * prominence_relation +
            weights['layer_depth'] * layer_relation
        )
        
        return score
    
    def _calculate_artistic_score(self, features: HierarchyFeatures) -> float:
        """
        计算艺术得分
        
        Args:
            features: 层级特征
            
        Returns:
            float: 艺术得分
        """
        weights = self.relation_weights['artistic']
        
        score = (
            weights['stage_priority'] * abs(features.stage_priority) +
            weights['composition_order'] * abs(features.composition_order) +
            weights['artistic_flow'] * features.artistic_flow
        )
        
        return score
    
    def _determine_relation_type(self, features: HierarchyFeatures,
                                features_i: Dict[str, Any],
                                features_j: Dict[str, Any]) -> RelationType:
        """
        确定关系类型
        
        Args:
            features: 层级特征
            features_i: 笔触i的特征
            features_j: 笔触j的特征
            
        Returns:
            RelationType: 关系类型
        """
        # 应用艺术规则
        if self.artistic_rules['wet_before_dry']:
            wetness_i = features_i.get('wetness', 0)
            wetness_j = features_j.get('wetness', 0)
            if wetness_i > wetness_j + 0.1:
                return RelationType.BEFORE
            elif wetness_j > wetness_i + 0.1:
                return RelationType.AFTER
        
        if self.artistic_rules['thick_before_thin']:
            thickness_i = features_i.get('thickness', 0)
            thickness_j = features_j.get('thickness', 0)
            if thickness_i > thickness_j + 0.1:
                return RelationType.BEFORE
            elif thickness_j > thickness_i + 0.1:
                return RelationType.AFTER
        
        if self.artistic_rules['dark_before_light']:
            darkness_i = features_i.get('darkness', 0)
            darkness_j = features_j.get('darkness', 0)
            if darkness_i > darkness_j + 0.1:
                return RelationType.BEFORE
            elif darkness_j > darkness_i + 0.1:
                return RelationType.AFTER
        
        # 基于阶段优先级
        if abs(features.stage_priority) > 0.3:
            if features.stage_priority > 0:
                return RelationType.BEFORE
            else:
                return RelationType.AFTER
        
        # 基于空间关系
        if features.spatial_distance < 0.3:
            return RelationType.CONCURRENT
        
        return RelationType.INDEPENDENT
    
    def _calculate_relation_confidence(self, features: HierarchyFeatures, 
                                      strength: float) -> float:
        """
        计算关系置信度
        
        Args:
            features: 层级特征
            strength: 关系强度
            
        Returns:
            float: 置信度
        """
        # 基于多个因素的置信度
        base_confidence = strength
        
        # 空间距离影响
        spatial_factor = features.spatial_distance
        
        # 艺术规则一致性
        artistic_factor = abs(features.stage_priority) + abs(features.composition_order)
        
        # 综合置信度
        confidence = base_confidence * 0.6 + spatial_factor * 0.2 + artistic_factor * 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_hierarchy_matrix(self, relations: List[StrokeRelation], 
                                n_strokes: int) -> np.ndarray:
        """
        创建层级矩阵
        
        Args:
            relations: 关系列表
            n_strokes: 笔触数量
            
        Returns:
            np.ndarray: 层级矩阵
        """
        matrix = np.zeros((n_strokes, n_strokes))
        
        for relation in relations:
            i, j = relation.source_id, relation.target_id
            
            if relation.relation_type == RelationType.BEFORE:
                matrix[i, j] = relation.confidence
            elif relation.relation_type == RelationType.AFTER:
                matrix[j, i] = relation.confidence
            elif relation.relation_type == RelationType.CONCURRENT:
                matrix[i, j] = matrix[j, i] = relation.confidence * 0.5
        
        return matrix
    
    def _optimize_hierarchy(self, relations: List[StrokeRelation],
                           stroke_features: List[Dict[str, Any]]) -> List[StrokeRelation]:
        """
        优化层级关系
        
        Args:
            relations: 原始关系列表
            stroke_features: 笔触特征列表
            
        Returns:
            List[StrokeRelation]: 优化后的关系列表
        """
        # 移除冲突关系
        optimized_relations = self._remove_conflicting_relations(relations)
        
        # 确保传递性
        optimized_relations = self._ensure_transitivity(optimized_relations)
        
        # 移除冗余关系
        optimized_relations = self._remove_redundant_relations(optimized_relations)
        
        return optimized_relations
    
    def _remove_conflicting_relations(self, relations: List[StrokeRelation]) -> List[StrokeRelation]:
        """
        移除冲突关系
        
        Args:
            relations: 关系列表
            
        Returns:
            List[StrokeRelation]: 无冲突的关系列表
        """
        # 构建关系图
        relation_map = {}
        for relation in relations:
            key = (relation.source_id, relation.target_id)
            if key not in relation_map or relation.confidence > relation_map[key].confidence:
                relation_map[key] = relation
        
        # 检查并解决冲突
        filtered_relations = []
        for relation in relation_map.values():
            # 检查是否存在相反的关系
            reverse_key = (relation.target_id, relation.source_id)
            if reverse_key in relation_map:
                reverse_relation = relation_map[reverse_key]
                # 保留置信度更高的关系
                if relation.confidence >= reverse_relation.confidence:
                    filtered_relations.append(relation)
            else:
                filtered_relations.append(relation)
        
        return filtered_relations
    
    def _ensure_transitivity(self, relations: List[StrokeRelation]) -> List[StrokeRelation]:
        """
        确保传递性
        
        Args:
            relations: 关系列表
            
        Returns:
            List[StrokeRelation]: 具有传递性的关系列表
        """
        # 这里简化处理，实际应该实现完整的传递闭包
        return relations
    
    def _remove_redundant_relations(self, relations: List[StrokeRelation]) -> List[StrokeRelation]:
        """
        移除冗余关系
        
        Args:
            relations: 关系列表
            
        Returns:
            List[StrokeRelation]: 无冗余的关系列表
        """
        # 移除置信度过低的关系
        filtered_relations = [
            relation for relation in relations 
            if relation.confidence >= self.thresholds['weak_relation']
        ]
        
        return filtered_relations
    
    def _calculate_hierarchy_statistics(self, relations: List[StrokeRelation]) -> Dict[str, Any]:
        """
        计算层级统计信息
        
        Args:
            relations: 关系列表
            
        Returns:
            Dict: 统计信息
        """
        if not relations:
            return {
                'total_relations': 0,
                'before_relations': 0,
                'after_relations': 0,
                'concurrent_relations': 0,
                'independent_relations': 0,
                'average_confidence': 0.0
            }
        
        stats = {
            'total_relations': len(relations),
            'before_relations': sum(1 for r in relations if r.relation_type == RelationType.BEFORE),
            'after_relations': sum(1 for r in relations if r.relation_type == RelationType.AFTER),
            'concurrent_relations': sum(1 for r in relations if r.relation_type == RelationType.CONCURRENT),
            'independent_relations': sum(1 for r in relations if r.relation_type == RelationType.INDEPENDENT),
            'average_confidence': np.mean([r.confidence for r in relations])
        }
        
        return stats