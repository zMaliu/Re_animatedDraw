# -*- coding: utf-8 -*-
"""
笔触组织器模块

提供笔触的组织和管理功能，整合各种排序算法
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from ..stroke_extraction.stroke_detector import Stroke
from .energy_function import EnergyFunction
from .nes_optimizer import NESOptimizer
from .constraint_handler import ConstraintHandler
from .order_evaluator import OrderEvaluator
from .spearman_correlation import SpearmanCorrelationCalculator


@dataclass
class OrganizationResult:
    """组织结果数据结构"""
    organized_strokes: List[Stroke]
    organization_method: str
    quality_score: float
    execution_time: float
    metadata: Dict[str, Any]
    stage_results: Dict[str, Any]


class StrokeOrganizer:
    """笔触组织器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化笔触组织器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组织方法配置
        self.organization_method = config.get('organization_method', 'hierarchical')
        self.enable_optimization = config.get('enable_optimization', True)
        self.use_constraints = config.get('use_constraints', True)
        
        # 初始化组件
        self.energy_function = EnergyFunction(config)
        self.nes_optimizer = NESOptimizer(config) if self.enable_optimization else None
        self.constraint_handler = ConstraintHandler(config) if self.use_constraints else None
        self.order_evaluator = OrderEvaluator(config)
        self.spearman_calculator = SpearmanCorrelationCalculator(config)
        
        # 组织参数
        self.max_group_size = config.get('max_group_size', 20)
        self.min_group_size = config.get('min_group_size', 3)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        
    def organize_strokes(self, strokes: List[Stroke], 
                        method: str = None) -> OrganizationResult:
        """
        组织笔触
        
        Args:
            strokes: 笔触列表
            method: 组织方法
            
        Returns:
            组织结果
        """
        if not strokes:
            return OrganizationResult(
                organized_strokes=[],
                organization_method='none',
                quality_score=0.0,
                execution_time=0.0,
                metadata={},
                stage_results={}
            )
        
        import time
        start_time = time.time()
        
        method = method or self.organization_method
        
        try:
            if method == 'hierarchical':
                result = self._hierarchical_organization(strokes)
            elif method == 'spatial':
                result = self._spatial_organization(strokes)
            elif method == 'semantic':
                result = self._semantic_organization(strokes)
            elif method == 'hybrid':
                result = self._hybrid_organization(strokes)
            else:
                raise ValueError(f"Unknown organization method: {method}")
            
            execution_time = time.time() - start_time
            
            # 评估组织质量
            quality_score = self._evaluate_organization_quality(result['organized_strokes'])
            
            return OrganizationResult(
                organized_strokes=result['organized_strokes'],
                organization_method=method,
                quality_score=quality_score,
                execution_time=execution_time,
                metadata=result.get('metadata', {}),
                stage_results=result.get('stage_results', {})
            )
            
        except Exception as e:
            self.logger.error(f"Error organizing strokes: {e}")
            execution_time = time.time() - start_time
            
            return OrganizationResult(
                organized_strokes=strokes,  # 返回原始顺序
                organization_method=f'{method}_failed',
                quality_score=0.0,
                execution_time=execution_time,
                metadata={'error': str(e)},
                stage_results={}
            )
    
    def group_strokes_by_similarity(self, strokes: List[Stroke]) -> List[List[Stroke]]:
        """
        按相似性分组笔触
        
        Args:
            strokes: 笔触列表
            
        Returns:
            笔触分组列表
        """
        if not strokes:
            return []
        
        groups = []
        remaining_strokes = strokes.copy()
        
        while remaining_strokes:
            # 选择第一个笔触作为种子
            seed_stroke = remaining_strokes.pop(0)
            current_group = [seed_stroke]
            
            # 找到相似的笔触
            i = 0
            while i < len(remaining_strokes) and len(current_group) < self.max_group_size:
                stroke = remaining_strokes[i]
                
                # 计算与组内笔触的平均相似度
                similarities = []
                for group_stroke in current_group:
                    similarity = self._calculate_stroke_similarity(stroke, group_stroke)
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                
                if avg_similarity >= self.similarity_threshold:
                    current_group.append(remaining_strokes.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def group_strokes_by_spatial_proximity(self, strokes: List[Stroke]) -> List[List[Stroke]]:
        """
        按空间邻近性分组笔触
        
        Args:
            strokes: 笔触列表
            
        Returns:
            笔触分组列表
        """
        if not strokes:
            return []
        
        # 使用聚类算法进行空间分组
        from sklearn.cluster import DBSCAN
        
        # 提取笔触中心点
        centers = np.array([stroke.center for stroke in strokes])
        
        # 计算适当的eps参数
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        eps = np.percentile(distances, 20) if distances else 50.0
        
        # 执行DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=self.min_group_size)
        cluster_labels = clustering.fit_predict(centers)
        
        # 组织分组结果
        groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(strokes[i])
        
        # 处理噪声点（label = -1）
        result_groups = []
        for label, group in groups.items():
            if label == -1:
                # 噪声点单独成组
                for stroke in group:
                    result_groups.append([stroke])
            else:
                result_groups.append(group)
        
        return result_groups
    
    def _hierarchical_organization(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """
        分层组织方法
        
        Args:
            strokes: 笔触列表
            
        Returns:
            组织结果
        """
        stage_results = {}
        
        # 第一阶段：按类型分组
        type_groups = self._group_by_stroke_type(strokes)
        stage_results['type_grouping'] = {
            'num_groups': len(type_groups),
            'group_sizes': [len(group) for group in type_groups.values()]
        }
        
        # 第二阶段：每个类型内部按空间分组
        spatial_groups = []
        for stroke_type, type_strokes in type_groups.items():
            groups = self.group_strokes_by_spatial_proximity(type_strokes)
            spatial_groups.extend(groups)
        
        stage_results['spatial_grouping'] = {
            'num_groups': len(spatial_groups),
            'group_sizes': [len(group) for group in spatial_groups]
        }
        
        # 第三阶段：组内排序
        organized_strokes = []
        for group in spatial_groups:
            if len(group) > 1:
                ordered_group = self._order_group_strokes(group)
            else:
                ordered_group = group
            organized_strokes.extend(ordered_group)
        
        stage_results['final_ordering'] = {
            'total_strokes': len(organized_strokes)
        }
        
        return {
            'organized_strokes': organized_strokes,
            'metadata': {
                'method': 'hierarchical',
                'num_stages': 3
            },
            'stage_results': stage_results
        }
    
    def _spatial_organization(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """
        空间组织方法
        
        Args:
            strokes: 笔触列表
            
        Returns:
            组织结果
        """
        # 按空间位置排序（从左到右，从上到下）
        sorted_strokes = sorted(strokes, key=lambda s: (s.center[1], s.center[0]))
        
        return {
            'organized_strokes': sorted_strokes,
            'metadata': {
                'method': 'spatial',
                'sorting_key': 'position'
            },
            'stage_results': {
                'spatial_sorting': {
                    'total_strokes': len(sorted_strokes)
                }
            }
        }
    
    def _semantic_organization(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """
        语义组织方法
        
        Args:
            strokes: 笔触列表
            
        Returns:
            组织结果
        """
        # 按语义重要性排序
        # 这里使用面积和位置作为重要性指标
        importance_scores = []
        for stroke in strokes:
            # 计算重要性分数（面积 + 中心性）
            area_score = stroke.area / max([s.area for s in strokes])
            center_score = 1.0 - (abs(stroke.center[0] - 0.5) + abs(stroke.center[1] - 0.5)) / 2.0
            importance = 0.6 * area_score + 0.4 * center_score
            importance_scores.append(importance)
        
        # 按重要性排序
        stroke_importance_pairs = list(zip(strokes, importance_scores))
        stroke_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        organized_strokes = [pair[0] for pair in stroke_importance_pairs]
        
        return {
            'organized_strokes': organized_strokes,
            'metadata': {
                'method': 'semantic',
                'importance_weights': {'area': 0.6, 'centrality': 0.4}
            },
            'stage_results': {
                'importance_calculation': {
                    'max_importance': max(importance_scores),
                    'min_importance': min(importance_scores),
                    'avg_importance': np.mean(importance_scores)
                }
            }
        }
    
    def _hybrid_organization(self, strokes: List[Stroke]) -> Dict[str, Any]:
        """
        混合组织方法
        
        Args:
            strokes: 笔触列表
            
        Returns:
            组织结果
        """
        stage_results = {}
        
        # 第一阶段：语义分组
        semantic_result = self._semantic_organization(strokes)
        semantic_strokes = semantic_result['organized_strokes']
        stage_results['semantic_stage'] = semantic_result['stage_results']
        
        # 第二阶段：空间优化
        spatial_groups = self.group_strokes_by_spatial_proximity(semantic_strokes)
        stage_results['spatial_grouping'] = {
            'num_groups': len(spatial_groups),
            'group_sizes': [len(group) for group in spatial_groups]
        }
        
        # 第三阶段：能量优化（如果启用）
        final_strokes = []
        if self.enable_optimization and self.nes_optimizer:
            for group in spatial_groups:
                if len(group) > 3:  # 只对较大的组进行优化
                    optimized_group = self._optimize_group_order(group)
                    final_strokes.extend(optimized_group)
                else:
                    final_strokes.extend(group)
        else:
            for group in spatial_groups:
                final_strokes.extend(group)
        
        stage_results['optimization_stage'] = {
            'optimized_groups': len([g for g in spatial_groups if len(g) > 3]),
            'total_strokes': len(final_strokes)
        }
        
        return {
            'organized_strokes': final_strokes,
            'metadata': {
                'method': 'hybrid',
                'stages': ['semantic', 'spatial', 'optimization']
            },
            'stage_results': stage_results
        }
    
    def _group_by_stroke_type(self, strokes: List[Stroke]) -> Dict[str, List[Stroke]]:
        """
        按笔触类型分组
        
        Args:
            strokes: 笔触列表
            
        Returns:
            类型分组字典
        """
        groups = {}
        
        for stroke in strokes:
            # 根据笔触特征确定类型
            stroke_type = self._determine_stroke_type(stroke)
            
            if stroke_type not in groups:
                groups[stroke_type] = []
            groups[stroke_type].append(stroke)
        
        return groups
    
    def _determine_stroke_type(self, stroke: Stroke) -> str:
        """
        确定笔触类型
        
        Args:
            stroke: 笔触
            
        Returns:
            笔触类型
        """
        # 简化的类型判断逻辑
        aspect_ratio = stroke.length / max(stroke.width, 1e-6)
        
        if aspect_ratio > 5.0:
            return 'line'  # 线条
        elif stroke.area > 1000:
            return 'fill'  # 填充
        else:
            return 'detail'  # 细节
    
    def _order_group_strokes(self, group: List[Stroke]) -> List[Stroke]:
        """
        对组内笔触进行排序
        
        Args:
            group: 笔触组
            
        Returns:
            排序后的笔触列表
        """
        if len(group) <= 1:
            return group
        
        # 使用能量函数进行排序
        if self.enable_optimization and self.nes_optimizer:
            return self._optimize_group_order(group)
        else:
            # 简单的空间排序
            return sorted(group, key=lambda s: (s.center[1], s.center[0]))
    
    def _optimize_group_order(self, group: List[Stroke]) -> List[Stroke]:
        """
        优化组内笔触顺序
        
        Args:
            group: 笔触组
            
        Returns:
            优化后的笔触列表
        """
        try:
            # 使用NES优化器优化顺序
            initial_order = list(range(len(group)))
            
            def objective_function(order):
                ordered_strokes = [group[i] for i in order]
                # 准备笔触特征用于能量计算
                stroke_features = {}
                for idx, stroke in enumerate(ordered_strokes):
                    stroke_features[idx] = {
                        'id': getattr(stroke, 'id', idx),
                        'area': getattr(stroke, 'area', 0.0),
                        'length': getattr(stroke, 'length', 0.0),
                        'center': getattr(stroke, 'center', (0.0, 0.0)),
                        'color': getattr(stroke, 'color', (0, 0, 0)),
                        'width': getattr(stroke, 'width', 1.0)
                    }
                energy_components = self.energy_function.calculate_energy(order, stroke_features)
                return -energy_components.total_energy
            
            # 准备笔触特征
            stroke_features = []
            for stroke in group:
                features = {
                    'id': getattr(stroke, 'id', 0),
                    'area': getattr(stroke, 'area', 0.0),
                    'length': getattr(stroke, 'length', 0.0),
                    'center': getattr(stroke, 'center', (0.0, 0.0))
                }
                stroke_features.append(features)
            
            result = self.nes_optimizer.optimize(
                energy_function=objective_function,
                stroke_features=stroke_features,
                initial_order=initial_order
            )
            
            if result.success and result.best_order is not None:
                optimized_order = result.best_order
                # 确保顺序有效
                optimized_order = self._fix_order_sequence(optimized_order, len(group))
                return [group[i] for i in optimized_order]
            else:
                return group
                
        except Exception as e:
            self.logger.warning(f"Error optimizing group order: {e}")
            return group
    
    def _fix_order_sequence(self, order: List[int], length: int) -> List[int]:
        """
        修复顺序序列，确保每个索引只出现一次
        
        Args:
            order: 原始顺序
            length: 序列长度
            
        Returns:
            修复后的顺序
        """
        # 检查输入有效性
        if order is None or length <= 0:
            return list(range(length))
        
        # 确保所有值在有效范围内
        try:
            order = [max(0, min(length-1, int(x))) for x in order if x is not None]
        except (ValueError, TypeError):
            return list(range(length))
        
        # 处理重复值
        used = set()
        fixed_order = []
        available = list(range(length))
        
        for val in order:
            if val not in used:
                fixed_order.append(val)
                used.add(val)
                available.remove(val)
            else:
                # 使用第一个可用的值
                if available:
                    replacement = available.pop(0)
                    fixed_order.append(replacement)
                    used.add(replacement)
        
        # 添加任何遗漏的值
        fixed_order.extend(available)
        
        return fixed_order[:length]
    
    def _calculate_stroke_similarity(self, stroke1: Stroke, stroke2: Stroke) -> float:
        """
        计算两个笔触的相似度
        
        Args:
            stroke1: 第一个笔触
            stroke2: 第二个笔触
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 几何相似度
            area_sim = 1.0 - abs(stroke1.area - stroke2.area) / max(stroke1.area + stroke2.area, 1e-6)
            length_sim = 1.0 - abs(stroke1.length - stroke2.length) / max(stroke1.length + stroke2.length, 1e-6)
            width_sim = 1.0 - abs(stroke1.width - stroke2.width) / max(stroke1.width + stroke2.width, 1e-6)
            
            # 位置相似度
            distance = np.linalg.norm(np.array(stroke1.center) - np.array(stroke2.center))
            max_distance = np.sqrt(2)  # 假设归一化坐标
            position_sim = 1.0 - distance / max_distance
            
            # 加权平均
            similarity = 0.3 * area_sim + 0.2 * length_sim + 0.2 * width_sim + 0.3 * position_sim
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning(f"Error calculating stroke similarity: {e}")
            return 0.0
    
    def _evaluate_organization_quality(self, organized_strokes: List[Stroke]) -> float:
        """
        评估组织质量
        
        Args:
            organized_strokes: 组织后的笔触列表
            
        Returns:
            质量分数 (0-1)
        """
        try:
            if len(organized_strokes) <= 1:
                return 1.0
            
            # 准备笔触特征
            stroke_features = {}
            for i, stroke in enumerate(organized_strokes):
                stroke_features[i] = {
                    'id': getattr(stroke, 'id', i),
                    'area': getattr(stroke, 'area', 0.0),
                    'length': getattr(stroke, 'length', 0.0),
                    'center': getattr(stroke, 'center', (0.0, 0.0)),
                    'color': getattr(stroke, 'color', (0, 0, 0)),
                    'width': getattr(stroke, 'width', 1.0)
                }
            
            # 使用订单评估器评估质量
            order = list(range(len(organized_strokes)))
            evaluation_result = self.order_evaluator.evaluate_order(order, stroke_features)
            
            return evaluation_result.overall_score
            
        except Exception as e:
            self.logger.warning(f"Error evaluating organization quality: {e}")
            return 0.5
    
    def get_organization_statistics(self, result: OrganizationResult) -> Dict[str, Any]:
        """
        获取组织统计信息
        
        Args:
            result: 组织结果
            
        Returns:
            统计信息
        """
        strokes = result.organized_strokes
        
        if not strokes:
            return {}
        
        # 基本统计
        stats = {
            'total_strokes': len(strokes),
            'organization_method': result.organization_method,
            'quality_score': result.quality_score,
            'execution_time': result.execution_time,
            'average_stroke_area': np.mean([s.area for s in strokes]),
            'total_area': sum(s.area for s in strokes)
        }
        
        # 空间分布统计
        centers = np.array([s.center for s in strokes])
        stats['spatial_distribution'] = {
            'center_x_mean': np.mean(centers[:, 0]),
            'center_y_mean': np.mean(centers[:, 1]),
            'center_x_std': np.std(centers[:, 0]),
            'center_y_std': np.std(centers[:, 1])
        }
        
        # 类型分布统计
        type_counts = {}
        for stroke in strokes:
            stroke_type = self._determine_stroke_type(stroke)
            type_counts[stroke_type] = type_counts.get(stroke_type, 0) + 1
        
        stats['type_distribution'] = type_counts
        
        return stats