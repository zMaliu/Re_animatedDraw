# -*- coding: utf-8 -*-
"""
基于规则的排序模块

提供基于绘画规则和启发式的笔触排序方法
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from enum import Enum
from ..stroke_extraction.stroke_detector import Stroke
from .spearman_correlation import SpearmanCorrelationCalculator


class OrderingRule(Enum):
    """排序规则类型"""
    SPATIAL_PROXIMITY = "spatial_proximity"  # 空间邻近性
    SIZE_BASED = "size_based"  # 基于大小
    BRIGHTNESS_BASED = "brightness_based"  # 基于亮度
    LAYER_BASED = "layer_based"  # 基于层次
    DIRECTION_BASED = "direction_based"  # 基于方向
    COMPLEXITY_BASED = "complexity_based"  # 基于复杂度
    SEMANTIC_BASED = "semantic_based"  # 基于语义
    ARTISTIC_CONVENTION = "artistic_convention"  # 艺术惯例


@dataclass
class RuleWeight:
    """规则权重"""
    rule: OrderingRule
    weight: float
    priority: int = 0
    enabled: bool = True


@dataclass
class OrderingContext:
    """排序上下文"""
    image_size: Tuple[int, int]
    drawing_style: str = "realistic"  # realistic, abstract, sketch, etc.
    complexity_level: str = "medium"  # low, medium, high
    target_audience: str = "general"  # beginner, intermediate, advanced, general
    time_constraint: float = 1.0  # 时间约束因子
    quality_preference: str = "balanced"  # speed, quality, balanced


@dataclass
class RuleBasedResult:
    """基于规则的排序结果"""
    ordered_strokes: List[Stroke]
    ordering_indices: List[int]
    rule_scores: Dict[OrderingRule, float]
    total_score: float
    applied_rules: List[OrderingRule]
    metadata: Dict[str, Any]


class RuleBasedOrdering:
    """基于规则的排序器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基于规则的排序器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化规则权重
        self.rule_weights = self._initialize_rule_weights()
        
        # 初始化Spearman相关系数计算器
        self.spearman_calculator = SpearmanCorrelationCalculator(config)
        
        # 规则函数映射
        self.rule_functions = {
            OrderingRule.SPATIAL_PROXIMITY: self._apply_spatial_proximity_rule,
            OrderingRule.SIZE_BASED: self._apply_size_based_rule,
            OrderingRule.BRIGHTNESS_BASED: self._apply_brightness_based_rule,
            OrderingRule.LAYER_BASED: self._apply_layer_based_rule,
            OrderingRule.DIRECTION_BASED: self._apply_direction_based_rule,
            OrderingRule.COMPLEXITY_BASED: self._apply_complexity_based_rule,
            OrderingRule.SEMANTIC_BASED: self._apply_semantic_based_rule,
            OrderingRule.ARTISTIC_CONVENTION: self._apply_artistic_convention_rule
        }
    
    def order_strokes(self, strokes: List[Stroke], 
                     context: OrderingContext = None,
                     custom_rules: List[RuleWeight] = None) -> RuleBasedResult:
        """
        基于规则对笔触进行排序
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            custom_rules: 自定义规则权重
            
        Returns:
            排序结果
        """
        if not strokes:
            return RuleBasedResult(
                ordered_strokes=[],
                ordering_indices=[],
                rule_scores={},
                total_score=0.0,
                applied_rules=[],
                metadata={}
            )
        
        # 使用默认上下文
        if context is None:
            context = OrderingContext(
                image_size=(800, 600),
                drawing_style=self.config.get('drawing_style', 'realistic'),
                complexity_level=self.config.get('complexity_level', 'medium')
            )
        
        # 使用自定义规则或默认规则
        rules = custom_rules or self._get_context_appropriate_rules(context)
        
        try:
            # 计算每个规则的分数
            rule_scores = {}
            stroke_rankings = {}
            
            for rule_weight in rules:
                if not rule_weight.enabled:
                    continue
                
                rule = rule_weight.rule
                if rule in self.rule_functions:
                    ranking = self.rule_functions[rule](strokes, context)
                    rule_scores[rule] = self._evaluate_ranking_quality(ranking, strokes)
                    stroke_rankings[rule] = ranking
                    
                    self.logger.debug(f"Applied rule {rule.value}: score={rule_scores[rule]:.4f}")
            
            # 组合规则结果
            final_ranking = self._combine_rule_rankings(stroke_rankings, rules)
            
            # 计算总分
            total_score = self._calculate_total_score(rule_scores, rules)
            
            # 生成最终排序
            ordered_indices = final_ranking
            ordered_strokes = [strokes[i] for i in ordered_indices]
            
            # 应用的规则
            applied_rules = [rw.rule for rw in rules if rw.enabled and rw.rule in rule_scores]
            
            result = RuleBasedResult(
                ordered_strokes=ordered_strokes,
                ordering_indices=ordered_indices,
                rule_scores=rule_scores,
                total_score=total_score,
                applied_rules=applied_rules,
                metadata={
                    'context': context,
                    'num_strokes': len(strokes),
                    'rules_applied': len(applied_rules),
                    'stroke_rankings': stroke_rankings
                }
            )
            
            self.logger.info(f"Rule-based ordering completed: {len(strokes)} strokes, "
                           f"{len(applied_rules)} rules, score={total_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in rule-based ordering: {e}")
            
            # 返回原始顺序作为后备
            return RuleBasedResult(
                ordered_strokes=strokes,
                ordering_indices=list(range(len(strokes))),
                rule_scores={},
                total_score=0.0,
                applied_rules=[],
                metadata={'error': str(e)}
            )
    
    def _initialize_rule_weights(self) -> List[RuleWeight]:
        """
        初始化默认规则权重
        
        Returns:
            规则权重列表
        """
        default_weights = {
            OrderingRule.SPATIAL_PROXIMITY: 0.25,
            OrderingRule.SIZE_BASED: 0.15,
            OrderingRule.BRIGHTNESS_BASED: 0.10,
            OrderingRule.LAYER_BASED: 0.20,
            OrderingRule.DIRECTION_BASED: 0.10,
            OrderingRule.COMPLEXITY_BASED: 0.10,
            OrderingRule.SEMANTIC_BASED: 0.05,
            OrderingRule.ARTISTIC_CONVENTION: 0.05
        }
        
        # 从配置中获取权重
        config_weights = self.config.get('rule_weights', {})
        
        rule_weights = []
        for rule, default_weight in default_weights.items():
            weight = config_weights.get(rule.value, default_weight)
            enabled = config_weights.get(f'{rule.value}_enabled', True)
            priority = config_weights.get(f'{rule.value}_priority', 0)
            
            rule_weights.append(RuleWeight(
                rule=rule,
                weight=weight,
                priority=priority,
                enabled=enabled
            ))
        
        # 按优先级排序
        rule_weights.sort(key=lambda x: x.priority, reverse=True)
        
        return rule_weights
    
    def _get_context_appropriate_rules(self, context: OrderingContext) -> List[RuleWeight]:
        """
        根据上下文获取合适的规则
        
        Args:
            context: 排序上下文
            
        Returns:
            适合的规则权重列表
        """
        rules = self.rule_weights.copy()
        
        # 根据绘画风格调整权重
        if context.drawing_style == "sketch":
            # 素描风格更注重空间和方向
            self._adjust_rule_weight(rules, OrderingRule.SPATIAL_PROXIMITY, 1.2)
            self._adjust_rule_weight(rules, OrderingRule.DIRECTION_BASED, 1.3)
            self._adjust_rule_weight(rules, OrderingRule.COMPLEXITY_BASED, 0.8)
        
        elif context.drawing_style == "abstract":
            # 抽象风格更注重语义和艺术惯例
            self._adjust_rule_weight(rules, OrderingRule.SEMANTIC_BASED, 1.5)
            self._adjust_rule_weight(rules, OrderingRule.ARTISTIC_CONVENTION, 1.4)
            self._adjust_rule_weight(rules, OrderingRule.SPATIAL_PROXIMITY, 0.8)
        
        elif context.drawing_style == "realistic":
            # 写实风格更注重层次和亮度
            self._adjust_rule_weight(rules, OrderingRule.LAYER_BASED, 1.3)
            self._adjust_rule_weight(rules, OrderingRule.BRIGHTNESS_BASED, 1.2)
        
        # 根据复杂度调整权重
        if context.complexity_level == "high":
            self._adjust_rule_weight(rules, OrderingRule.COMPLEXITY_BASED, 1.4)
            self._adjust_rule_weight(rules, OrderingRule.SEMANTIC_BASED, 1.2)
        elif context.complexity_level == "low":
            self._adjust_rule_weight(rules, OrderingRule.SPATIAL_PROXIMITY, 1.3)
            self._adjust_rule_weight(rules, OrderingRule.SIZE_BASED, 1.2)
        
        # 根据质量偏好调整
        if context.quality_preference == "speed":
            # 速度优先，使用简单规则
            self._adjust_rule_weight(rules, OrderingRule.SPATIAL_PROXIMITY, 1.5)
            self._adjust_rule_weight(rules, OrderingRule.SIZE_BASED, 1.3)
            self._disable_rule(rules, OrderingRule.SEMANTIC_BASED)
            self._disable_rule(rules, OrderingRule.ARTISTIC_CONVENTION)
        
        elif context.quality_preference == "quality":
            # 质量优先，使用所有规则
            for rule_weight in rules:
                rule_weight.enabled = True
        
        return rules
    
    def _adjust_rule_weight(self, rules: List[RuleWeight], rule: OrderingRule, factor: float):
        """
        调整规则权重
        
        Args:
            rules: 规则列表
            rule: 要调整的规则
            factor: 调整因子
        """
        for rule_weight in rules:
            if rule_weight.rule == rule:
                rule_weight.weight *= factor
                break
    
    def _disable_rule(self, rules: List[RuleWeight], rule: OrderingRule):
        """
        禁用规则
        
        Args:
            rules: 规则列表
            rule: 要禁用的规则
        """
        for rule_weight in rules:
            if rule_weight.rule == rule:
                rule_weight.enabled = False
                break
    
    def _apply_spatial_proximity_rule(self, strokes: List[Stroke], 
                                     context: OrderingContext) -> List[int]:
        """
        应用空间邻近性规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        if not strokes:
            return []
        
        # 计算笔触中心点
        centers = []
        for stroke in strokes:
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                center_x = float(np.mean([p[0] for p in stroke.points]))
                center_y = float(np.mean([p[1] for p in stroke.points]))
                centers.append((center_x, center_y))
            else:
                centers.append((0, 0))
        
        # 使用最近邻算法排序
        visited = [False] * len(strokes)
        ordering = []
        
        # 从左上角开始
        current_idx = 0
        min_distance = float('inf')
        for i, (x, y) in enumerate(centers):
            distance = x + y  # 曼哈顿距离到原点
            if distance < min_distance:
                min_distance = distance
                current_idx = i
        
        while len(ordering) < len(strokes):
            visited[current_idx] = True
            ordering.append(current_idx)
            
            # 找到最近的未访问笔触
            min_distance = float('inf')
            next_idx = -1
            
            current_center = centers[current_idx]
            
            for i, center in enumerate(centers):
                if not visited[i]:
                    distance = float(np.sqrt((current_center[0] - center[0])**2 + 
                                           (current_center[1] - center[1])**2))
                    if distance < min_distance:
                        min_distance = distance
                        next_idx = i
            
            if next_idx != -1:
                current_idx = next_idx
            else:
                # 找到任何未访问的笔触
                for i in range(len(strokes)):
                    if not visited[i]:
                        current_idx = i
                        break
        
        return ordering
    
    def _apply_size_based_rule(self, strokes: List[Stroke], 
                              context: OrderingContext) -> List[int]:
        """
        应用基于大小的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 计算笔触大小（边界框面积）
        sizes = []
        for stroke in strokes:
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                xs = [p[0] for p in stroke.points]
                ys = [p[1] for p in stroke.points]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                size = width * height
            else:
                size = 0
            sizes.append(size)
        
        # 从大到小排序（大笔触先画）
        size_indices = list(range(len(strokes)))
        size_indices.sort(key=lambda i: sizes[i], reverse=True)
        
        return size_indices
    
    def _apply_brightness_based_rule(self, strokes: List[Stroke], 
                                   context: OrderingContext) -> List[int]:
        """
        应用基于亮度的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 计算笔触平均亮度
        brightnesses = []
        for stroke in strokes:
            if hasattr(stroke, 'color') and stroke.color is not None:
                # 假设颜色是RGB格式
                if isinstance(stroke.color, (list, tuple)) and len(stroke.color) >= 3:
                    r, g, b = stroke.color[:3]
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b
                else:
                    brightness = 128  # 默认中等亮度
            else:
                brightness = 128  # 默认中等亮度
            brightnesses.append(brightness)
        
        # 从暗到亮排序（暗色先画）
        brightness_indices = list(range(len(strokes)))
        brightness_indices.sort(key=lambda i: brightnesses[i])
        
        return brightness_indices
    
    def _apply_layer_based_rule(self, strokes: List[Stroke], 
                               context: OrderingContext) -> List[int]:
        """
        应用基于层次的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 计算笔触的层次（基于位置和大小）
        layers = []
        for stroke in strokes:
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                # 计算中心点
                center_x = np.mean([p[0] for p in stroke.points])
                center_y = np.mean([p[1] for p in stroke.points])
                
                # 计算大小
                xs = [p[0] for p in stroke.points]
                ys = [p[1] for p in stroke.points]
                size = (max(xs) - min(xs)) * (max(ys) - min(ys))
                
                # 层次 = 大小权重 + 位置权重（背景元素通常更大且居中）
                layer = size * 0.7 + (context.image_size[0] * context.image_size[1] - 
                                     (center_x - context.image_size[0]/2)**2 - 
                                     (center_y - context.image_size[1]/2)**2) * 0.3
            else:
                layer = 0
            layers.append(layer)
        
        # 从背景到前景排序（大层次值先画）
        layer_indices = list(range(len(strokes)))
        layer_indices.sort(key=lambda i: layers[i], reverse=True)
        
        return layer_indices
    
    def _apply_direction_based_rule(self, strokes: List[Stroke], 
                                   context: OrderingContext) -> List[int]:
        """
        应用基于方向的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 计算笔触的主要方向
        directions = []
        for stroke in strokes:
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) >= 2:
                # 计算主要方向向量
                start_point = stroke.points[0]
                end_point = stroke.points[-1]
                
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                # 计算角度（弧度）
                angle = np.arctan2(dy, dx)
                
                # 转换为0-2π范围
                if angle < 0:
                    angle += 2 * np.pi
                
                directions.append(angle)
            else:
                directions.append(0)
        
        # 按方向排序（从水平到垂直）
        direction_indices = list(range(len(strokes)))
        direction_indices.sort(key=lambda i: directions[i])
        
        return direction_indices
    
    def _apply_complexity_based_rule(self, strokes: List[Stroke], 
                                    context: OrderingContext) -> List[int]:
        """
        应用基于复杂度的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 计算笔触复杂度
        complexities = []
        for stroke in strokes:
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                # 复杂度 = 点数 + 曲率变化
                point_count = len(stroke.points)
                
                # 计算曲率变化
                curvature_changes = 0
                if point_count >= 3:
                    for i in range(1, point_count - 1):
                        p1 = stroke.points[i-1]
                        p2 = stroke.points[i]
                        p3 = stroke.points[i+1]
                        
                        # 计算角度变化
                        v1 = (p2[0] - p1[0], p2[1] - p1[1])
                        v2 = (p3[0] - p2[0], p3[1] - p2[1])
                        
                        # 避免除零
                        len1 = np.sqrt(v1[0]**2 + v1[1]**2)
                        len2 = np.sqrt(v2[0]**2 + v2[1]**2)
                        
                        if len1 > 0 and len2 > 0:
                            cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle_change = np.arccos(cos_angle)
                            curvature_changes += angle_change
                
                complexity = point_count * 0.3 + curvature_changes * 0.7
            else:
                complexity = 0
            
            complexities.append(complexity)
        
        # 从简单到复杂排序（简单笔触先画）
        complexity_indices = list(range(len(strokes)))
        complexity_indices.sort(key=lambda i: complexities[i])
        
        return complexity_indices
    
    def _apply_semantic_based_rule(self, strokes: List[Stroke], 
                                  context: OrderingContext) -> List[int]:
        """
        应用基于语义的规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 简化的语义分析（基于位置和形状特征）
        semantic_scores = []
        
        for stroke in strokes:
            score = 0
            
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                # 计算位置特征
                center_x = np.mean([p[0] for p in stroke.points])
                center_y = np.mean([p[1] for p in stroke.points])
                
                # 背景元素（通常在中心或边缘）
                center_distance = np.sqrt((center_x - context.image_size[0]/2)**2 + 
                                        (center_y - context.image_size[1]/2)**2)
                
                # 计算形状特征
                xs = [p[0] for p in stroke.points]
                ys = [p[1] for p in stroke.points]
                width = max(xs) - min(xs) if xs else 0
                height = max(ys) - min(ys) if ys else 0
                aspect_ratio = width / height if height > 0 else 1
                
                # 语义评分（背景元素先画）
                if aspect_ratio > 2 or aspect_ratio < 0.5:  # 长条形状，可能是背景线条
                    score += 10
                
                if center_distance < min(context.image_size) * 0.3:  # 中心区域
                    score += 5
                
                if width * height > context.image_size[0] * context.image_size[1] * 0.1:  # 大面积
                    score += 8
            
            semantic_scores.append(score)
        
        # 按语义重要性排序（重要元素先画）
        semantic_indices = list(range(len(strokes)))
        semantic_indices.sort(key=lambda i: semantic_scores[i], reverse=True)
        
        return semantic_indices
    
    def _apply_artistic_convention_rule(self, strokes: List[Stroke], 
                                       context: OrderingContext) -> List[int]:
        """
        应用艺术惯例规则
        
        Args:
            strokes: 笔触列表
            context: 排序上下文
            
        Returns:
            排序索引
        """
        # 艺术惯例：从左到右，从上到下，从粗到细
        convention_scores = []
        
        for stroke in strokes:
            score = 0
            
            if hasattr(stroke, 'points') and stroke.points is not None and len(stroke.points) > 0:
                # 位置权重（左上优先）
                center_x = np.mean([p[0] for p in stroke.points])
                center_y = np.mean([p[1] for p in stroke.points])
                
                # 归一化位置
                norm_x = center_x / context.image_size[0]
                norm_y = center_y / context.image_size[1]
                
                # 左上角权重更高
                position_score = (1 - norm_x) * 0.5 + (1 - norm_y) * 0.5
                
                # 笔触粗细（如果有的话）
                thickness_score = 0
                if hasattr(stroke, 'thickness') and stroke.thickness is not None:
                    # 粗笔触先画
                    thickness_score = stroke.thickness / 10.0  # 假设最大粗细为10
                
                score = position_score * 0.7 + thickness_score * 0.3
            
            convention_scores.append(score)
        
        # 按艺术惯例排序
        convention_indices = list(range(len(strokes)))
        convention_indices.sort(key=lambda i: convention_scores[i], reverse=True)
        
        return convention_indices
    
    def _combine_rule_rankings(self, stroke_rankings: Dict[OrderingRule, List[int]], 
                              rules: List[RuleWeight]) -> List[int]:
        """
        组合多个规则的排序结果
        
        Args:
            stroke_rankings: 各规则的排序结果
            rules: 规则权重列表
            
        Returns:
            组合后的排序
        """
        if not stroke_rankings:
            return list(range(len(stroke_rankings.get(list(stroke_rankings.keys())[0], []))))
        
        # 获取笔触数量
        num_strokes = len(next(iter(stroke_rankings.values())))
        
        # 计算加权排名分数
        weighted_scores = np.zeros(num_strokes)
        total_weight = 0
        
        for rule_weight in rules:
            if not rule_weight.enabled or rule_weight.rule not in stroke_rankings:
                continue
            
            ranking = stroke_rankings[rule_weight.rule]
            weight = rule_weight.weight
            
            # 将排序转换为分数（排名越靠前分数越高）
            for rank, stroke_idx in enumerate(ranking):
                score = (num_strokes - rank) / num_strokes  # 归一化分数
                weighted_scores[stroke_idx] += weight * score
            
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            weighted_scores /= total_weight
        
        # 按分数排序
        final_ranking = list(range(num_strokes))
        final_ranking.sort(key=lambda i: weighted_scores[i], reverse=True)
        
        return final_ranking
    
    def _evaluate_ranking_quality(self, ranking: List[int], strokes: List[Stroke]) -> float:
        """
        评估排序质量
        
        Args:
            ranking: 排序结果
            strokes: 笔触列表
            
        Returns:
            质量分数
        """
        if not ranking or len(ranking) != len(strokes):
            return 0.0
        
        # 计算空间连续性
        spatial_continuity = self._calculate_spatial_continuity(ranking, strokes)
        
        # 计算排序的多样性（避免过于单调）
        diversity = self._calculate_ranking_diversity(ranking)
        
        # 组合分数
        quality_score = spatial_continuity * 0.7 + diversity * 0.3
        
        return quality_score
    
    def _calculate_spatial_continuity(self, ranking: List[int], strokes: List[Stroke]) -> float:
        """
        计算空间连续性
        
        Args:
            ranking: 排序结果
            strokes: 笔触列表
            
        Returns:
            连续性分数
        """
        if len(ranking) < 2:
            return 1.0
        
        total_distance = 0
        max_possible_distance = 0
        
        for i in range(len(ranking) - 1):
            stroke1 = strokes[ranking[i]]
            stroke2 = strokes[ranking[i + 1]]
            
            # 计算笔触间距离
            if (hasattr(stroke1, 'points') and stroke1.points is not None and len(stroke1.points) > 0 and
                hasattr(stroke2, 'points') and stroke2.points is not None and len(stroke2.points) > 0):
                center1 = (np.mean([p[0] for p in stroke1.points]), 
                          np.mean([p[1] for p in stroke1.points]))
                center2 = (np.mean([p[0] for p in stroke2.points]), 
                          np.mean([p[1] for p in stroke2.points]))
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                total_distance += distance
                
                # 最大可能距离（对角线）
                max_possible_distance += np.sqrt(800**2 + 600**2)  # 假设图像大小
        
        if max_possible_distance > 0:
            # 连续性 = 1 - (实际距离 / 最大可能距离)
            continuity = 1 - (total_distance / max_possible_distance)
            return max(0, continuity)
        
        return 1.0
    
    def _calculate_ranking_diversity(self, ranking: List[int]) -> float:
        """
        计算排序多样性
        
        Args:
            ranking: 排序结果
            
        Returns:
            多样性分数
        """
        if len(ranking) < 2:
            return 1.0
        
        # 计算排序的熵
        # 这里简化为检查排序是否过于规律
        differences = []
        for i in range(len(ranking) - 1):
            diff = abs(ranking[i+1] - ranking[i])
            differences.append(diff)
        
        if not differences:
            return 1.0
        
        # 计算差异的标准差（多样性指标）
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # 归一化多样性分数
        diversity = min(1.0, std_diff / (mean_diff + 1e-6))
        
        return diversity
    
    def _calculate_total_score(self, rule_scores: Dict[OrderingRule, float], 
                              rules: List[RuleWeight]) -> float:
        """
        计算总分
        
        Args:
            rule_scores: 各规则分数
            rules: 规则权重
            
        Returns:
            总分
        """
        total_score = 0
        total_weight = 0
        
        for rule_weight in rules:
            if rule_weight.enabled and rule_weight.rule in rule_scores:
                total_score += rule_weight.weight * rule_scores[rule_weight.rule]
                total_weight += rule_weight.weight
        
        if total_weight > 0:
            return total_score / total_weight
        
        return 0.0
    
    def update_rule_weights(self, new_weights: Dict[OrderingRule, float]):
        """
        更新规则权重
        
        Args:
            new_weights: 新的权重字典
        """
        for rule_weight in self.rule_weights:
            if rule_weight.rule in new_weights:
                rule_weight.weight = new_weights[rule_weight.rule]
        
        self.logger.info(f"Updated rule weights: {new_weights}")
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        获取规则统计信息
        
        Returns:
            统计信息
        """
        stats = {
            'total_rules': len(self.rule_weights),
            'enabled_rules': sum(1 for rw in self.rule_weights if rw.enabled),
            'rule_weights': {rw.rule.value: rw.weight for rw in self.rule_weights},
            'rule_priorities': {rw.rule.value: rw.priority for rw in self.rule_weights},
            'rule_status': {rw.rule.value: rw.enabled for rw in self.rule_weights}
        }
        
        return stats