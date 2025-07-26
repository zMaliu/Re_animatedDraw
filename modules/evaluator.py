#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块
用于量化评估复现效果

主要功能:
1. 笔触提取质量评估
2. 特征建模准确性评估
3. 结构构建合理性评估
4. 排序优化效果评估
5. 动画质量评估
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from scipy import stats
from sklearn.metrics import silhouette_score
import networkx as nx
from collections import defaultdict

from .stroke_extraction import Stroke

class Evaluator:
    """评估器类"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.metrics = {}
        
    def evaluate_stroke_extraction(self, strokes: List[Stroke], 
                                 original_image_path: str) -> Dict:
        """评估笔触提取质量"""
        if not strokes:
            return {'error': '没有提取到笔触'}
        
        # 读取原始图像
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            return {'error': '无法读取原始图像'}
        
        metrics = {}
        
        # 1. 覆盖率评估
        coverage = self._calculate_coverage(strokes, original_image)
        metrics['coverage'] = coverage
        
        # 2. 笔触数量合理性
        stroke_count = len(strokes)
        image_area = original_image.shape[0] * original_image.shape[1]
        stroke_density = stroke_count / (image_area / 10000)  # 每万像素的笔触数
        metrics['stroke_count'] = stroke_count
        metrics['stroke_density'] = stroke_density
        
        # 3. 笔触大小分布
        areas = [stroke.area for stroke in strokes]
        metrics['area_stats'] = {
            'mean': np.mean(areas),
            'std': np.std(areas),
            'min': np.min(areas),
            'max': np.max(areas),
            'cv': np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        }
        
        # 4. 形状复杂度
        complexities = []
        for stroke in strokes:
            if stroke.perimeter > 0:
                complexity = stroke.perimeter ** 2 / (4 * np.pi * stroke.area)
                complexities.append(complexity)
        
        if complexities:
            metrics['shape_complexity'] = {
                'mean': np.mean(complexities),
                'std': np.std(complexities)
            }
        
        # 5. 连通性评估
        connectivity_score = self._evaluate_connectivity(strokes)
        metrics['connectivity_score'] = connectivity_score
        
        # 6. 整体质量评分
        quality_score = self._calculate_extraction_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        if self.debug:
            print(f"笔触提取评估完成: 质量得分 {quality_score:.3f}")
        
        return metrics
    
    def evaluate_feature_modeling(self, features: Dict) -> Dict:
        """评估特征建模准确性"""
        if not features:
            return {'error': '没有特征数据'}
        
        metrics = {}
        
        # 1. 特征完整性
        completeness = self._check_feature_completeness(features)
        metrics['completeness'] = completeness
        
        # 2. 特征分布合理性
        distribution_scores = self._evaluate_feature_distributions(features)
        metrics['distribution_scores'] = distribution_scores
        
        # 3. 特征相关性分析
        correlation_analysis = self._analyze_feature_correlations(features)
        metrics['correlation_analysis'] = correlation_analysis
        
        # 4. 特征区分度
        discrimination_score = self._calculate_feature_discrimination(features)
        metrics['discrimination_score'] = discrimination_score
        
        # 5. 综合得分合理性
        score_analysis = self._analyze_comprehensive_scores(features)
        metrics['score_analysis'] = score_analysis
        
        # 6. 整体质量评分
        quality_score = self._calculate_feature_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        if self.debug:
            print(f"特征建模评估完成: 质量得分 {quality_score:.3f}")
        
        return metrics
    
    def evaluate_structure_construction(self, hasse_graph: nx.DiGraph, 
                                      stages: Dict, features: Dict) -> Dict:
        """评估结构构建合理性"""
        if not hasse_graph.nodes():
            return {'error': '没有结构图数据'}
        
        metrics = {}
        
        # 1. 图结构特性
        graph_properties = self._analyze_graph_properties(hasse_graph)
        metrics['graph_properties'] = graph_properties
        
        # 2. 阶段分布合理性
        stage_distribution = self._evaluate_stage_distribution(stages, features)
        metrics['stage_distribution'] = stage_distribution
        
        # 3. 偏序关系一致性
        consistency_score = self._evaluate_partial_order_consistency(hasse_graph, features)
        metrics['consistency_score'] = consistency_score
        
        # 4. 层次结构清晰度
        hierarchy_clarity = self._evaluate_hierarchy_clarity(hasse_graph, stages)
        metrics['hierarchy_clarity'] = hierarchy_clarity
        
        # 5. 拓扑排序有效性
        topological_validity = self._evaluate_topological_validity(hasse_graph)
        metrics['topological_validity'] = topological_validity
        
        # 6. 整体质量评分
        quality_score = self._calculate_structure_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        if self.debug:
            print(f"结构构建评估完成: 质量得分 {quality_score:.3f}")
        
        return metrics
    
    def evaluate_stroke_ordering(self, optimized_order: List[Tuple[int, str]], 
                               hasse_graph: nx.DiGraph, features: Dict, 
                               stages: Dict) -> Dict:
        """评估排序优化效果"""
        if not optimized_order:
            return {'error': '没有排序结果'}
        
        metrics = {}
        
        # 1. 约束满足度
        constraint_satisfaction = self._evaluate_constraint_satisfaction(
            optimized_order, hasse_graph)
        metrics['constraint_satisfaction'] = constraint_satisfaction
        
        # 2. 绘制顺序合理性
        order_rationality = self._evaluate_order_rationality(
            optimized_order, features, stages)
        metrics['order_rationality'] = order_rationality
        
        # 3. 方向选择准确性
        direction_accuracy = self._evaluate_direction_accuracy(
            optimized_order, features)
        metrics['direction_accuracy'] = direction_accuracy
        
        # 4. 能量函数收敛性
        convergence_analysis = self._analyze_convergence(optimized_order, features)
        metrics['convergence_analysis'] = convergence_analysis
        
        # 5. 阶段连续性
        stage_continuity = self._evaluate_stage_continuity(optimized_order, stages)
        metrics['stage_continuity'] = stage_continuity
        
        # 6. 整体质量评分
        quality_score = self._calculate_ordering_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        if self.debug:
            print(f"排序优化评估完成: 质量得分 {quality_score:.3f}")
        
        return metrics
    
    def evaluate_animation_quality(self, animation_path: Path, 
                                 strokes: List[Stroke], 
                                 optimized_order: List[Tuple[int, str]]) -> Dict:
        """评估动画质量"""
        if not animation_path.exists():
            return {'error': '动画文件不存在'}
        
        metrics = {}
        
        # 1. 视频基本信息
        video_info = self._get_video_info(animation_path)
        metrics['video_info'] = video_info
        
        # 2. 绘制流畅性（通过帧间差异评估）
        smoothness_score = self._evaluate_animation_smoothness(animation_path)
        metrics['smoothness_score'] = smoothness_score
        
        # 3. 绘制完整性
        completeness_score = self._evaluate_animation_completeness(
            animation_path, strokes)
        metrics['completeness_score'] = completeness_score
        
        # 4. 时序一致性
        temporal_consistency = self._evaluate_temporal_consistency(
            animation_path, optimized_order)
        metrics['temporal_consistency'] = temporal_consistency
        
        # 5. 视觉质量
        visual_quality = self._evaluate_visual_quality(animation_path)
        metrics['visual_quality'] = visual_quality
        
        # 6. 整体质量评分
        quality_score = self._calculate_animation_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        if self.debug:
            print(f"动画质量评估完成: 质量得分 {quality_score:.3f}")
        
        return metrics
    
    def generate_comprehensive_report(self, stroke_metrics: Dict, 
                                    feature_metrics: Dict, 
                                    structure_metrics: Dict, 
                                    ordering_metrics: Dict, 
                                    animation_metrics: Dict, 
                                    output_path: Path) -> Dict:
        """生成综合评估报告"""
        # 计算各模块权重
        weights = {
            'stroke_extraction': 0.2,
            'feature_modeling': 0.2,
            'structure_construction': 0.2,
            'stroke_ordering': 0.2,
            'animation_generation': 0.2
        }
        
        # 提取各模块质量得分
        module_scores = {
            'stroke_extraction': stroke_metrics.get('quality_score', 0),
            'feature_modeling': feature_metrics.get('quality_score', 0),
            'structure_construction': structure_metrics.get('quality_score', 0),
            'stroke_ordering': ordering_metrics.get('quality_score', 0),
            'animation_generation': animation_metrics.get('quality_score', 0)
        }
        
        # 计算总体质量得分
        overall_score = sum(weights[module] * score 
                          for module, score in module_scores.items())
        
        # 生成评估等级
        if overall_score >= 0.9:
            grade = 'A+'
            description = '优秀'
        elif overall_score >= 0.8:
            grade = 'A'
            description = '良好'
        elif overall_score >= 0.7:
            grade = 'B'
            description = '中等'
        elif overall_score >= 0.6:
            grade = 'C'
            description = '及格'
        else:
            grade = 'D'
            description = '需要改进'
        
        # 构建综合报告
        comprehensive_report = {
            'overall_assessment': {
                'score': overall_score,
                'grade': grade,
                'description': description
            },
            'module_scores': module_scores,
            'weights': weights,
            'detailed_metrics': {
                'stroke_extraction': stroke_metrics,
                'feature_modeling': feature_metrics,
                'structure_construction': structure_metrics,
                'stroke_ordering': ordering_metrics,
                'animation_generation': animation_metrics
            },
            'recommendations': self._generate_recommendations(module_scores)
        }
        
        # 保存报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
        
        if self.debug:
            print(f"综合评估报告已保存至: {output_path}")
            print(f"总体评估: {grade} ({description}) - {overall_score:.3f}")
        
        return comprehensive_report
    
    # 辅助方法
    def _calculate_coverage(self, strokes: List[Stroke], original_image: np.ndarray) -> float:
        """计算笔触覆盖率"""
        # 创建合成掩码
        combined_mask = np.zeros_like(original_image)
        
        for stroke in strokes:
            x, y, w, h = stroke.bbox
            if (x + w <= original_image.shape[1] and y + h <= original_image.shape[0]):
                combined_mask[y:y+h, x:x+w] = np.maximum(
                    combined_mask[y:y+h, x:x+w], stroke.mask)
        
        # 计算原始图像的前景像素
        _, binary_original = cv2.threshold(original_image, 0, 255, 
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 计算覆盖率
        original_foreground = np.sum(binary_original > 0)
        covered_foreground = np.sum((binary_original > 0) & (combined_mask > 0))
        
        if original_foreground > 0:
            return covered_foreground / original_foreground
        else:
            return 0.0
    
    def _evaluate_connectivity(self, strokes: List[Stroke]) -> float:
        """评估笔触连通性"""
        if len(strokes) < 2:
            return 1.0
        
        # 计算笔触间的距离
        distances = []
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                dist = np.sqrt((strokes[i].centroid[0] - strokes[j].centroid[0])**2 + 
                             (strokes[i].centroid[1] - strokes[j].centroid[1])**2)
                distances.append(dist)
        
        # 使用距离分布评估连通性
        if distances:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            cv = std_dist / mean_dist if mean_dist > 0 else 0
            return max(0, 1 - cv)  # 变异系数越小，连通性越好
        else:
            return 1.0
    
    def _calculate_extraction_quality_score(self, metrics: Dict) -> float:
        """计算笔触提取质量得分"""
        score = 0.0
        
        # 覆盖率权重 40%
        coverage = metrics.get('coverage', 0)
        score += 0.4 * coverage
        
        # 笔触密度合理性权重 20%
        density = metrics.get('stroke_density', 0)
        density_score = min(1.0, max(0.0, 1 - abs(density - 5) / 10))  # 理想密度为5
        score += 0.2 * density_score
        
        # 形状复杂度权重 20%
        if 'shape_complexity' in metrics:
            complexity_mean = metrics['shape_complexity']['mean']
            complexity_score = min(1.0, max(0.0, 1 - abs(complexity_mean - 1.5) / 2))
            score += 0.2 * complexity_score
        
        # 连通性权重 20%
        connectivity = metrics.get('connectivity_score', 0)
        score += 0.2 * connectivity
        
        return min(1.0, max(0.0, score))
    
    def _check_feature_completeness(self, features: Dict) -> float:
        """检查特征完整性"""
        if not features:
            return 0.0
        
        required_features = [
            'geometric', 'shape', 'ink', 'color', 'position', 'comprehensive_score'
        ]
        
        total_completeness = 0
        for stroke_id, feature_dict in features.items():
            completeness = sum(1 for feat in required_features 
                             if feat in feature_dict) / len(required_features)
            total_completeness += completeness
        
        return total_completeness / len(features)
    
    def _evaluate_feature_distributions(self, features: Dict) -> Dict:
        """评估特征分布合理性"""
        scores = {}
        
        # 提取各类特征值
        feature_values = defaultdict(list)
        for stroke_id, feature_dict in features.items():
            if 'comprehensive_score' in feature_dict:
                feature_values['comprehensive_score'].append(
                    feature_dict['comprehensive_score'])
        
        # 评估分布
        for feature_name, values in feature_values.items():
            if len(values) > 1:
                # 使用正态性检验
                _, p_value = stats.normaltest(values)
                normality_score = min(1.0, p_value * 10)  # p值越大越接近正态分布
                
                # 使用变异系数评估分散度
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                dispersion_score = min(1.0, cv)  # 适度分散
                
                scores[feature_name] = {
                    'normality_score': normality_score,
                    'dispersion_score': dispersion_score,
                    'overall_score': (normality_score + dispersion_score) / 2
                }
        
        return scores
    
    def _analyze_feature_correlations(self, features: Dict) -> Dict:
        """分析特征相关性"""
        # 构建特征矩阵
        feature_matrix = []
        for stroke_id, feature_dict in features.items():
            if 'comprehensive_score' in feature_dict:
                feature_matrix.append(feature_dict['comprehensive_score'])
        
        if len(feature_matrix) < 2:
            return {'error': '特征数据不足'}
        
        # 计算自相关性（这里简化处理）
        autocorr = np.corrcoef(feature_matrix[:-1], feature_matrix[1:])[0, 1]
        
        return {
            'autocorrelation': autocorr,
            'independence_score': 1 - abs(autocorr)  # 独立性得分
        }
    
    def _calculate_feature_discrimination(self, features: Dict) -> float:
        """计算特征区分度"""
        scores = [features[sid]['comprehensive_score'] for sid in features.keys()]
        
        if len(scores) < 2:
            return 0.0
        
        # 使用标准差衡量区分度
        std_score = np.std(scores)
        max_possible_std = 0.5  # 假设最大标准差
        
        return min(1.0, std_score / max_possible_std)
    
    def _analyze_comprehensive_scores(self, features: Dict) -> Dict:
        """分析综合得分合理性"""
        scores = [features[sid]['comprehensive_score'] for sid in features.keys()]
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'range': np.max(scores) - np.min(scores),
            'distribution_quality': min(1.0, np.std(scores) * 2)  # 标准差越大分布越好
        }
    
    def _calculate_feature_quality_score(self, metrics: Dict) -> float:
        """计算特征建模质量得分"""
        score = 0.0
        
        # 完整性权重 30%
        completeness = metrics.get('completeness', 0)
        score += 0.3 * completeness
        
        # 区分度权重 30%
        discrimination = metrics.get('discrimination_score', 0)
        score += 0.3 * discrimination
        
        # 分布合理性权重 25%
        if 'distribution_scores' in metrics:
            dist_scores = metrics['distribution_scores']
            if 'comprehensive_score' in dist_scores:
                dist_quality = dist_scores['comprehensive_score']['overall_score']
                score += 0.25 * dist_quality
        
        # 相关性权重 15%
        if 'correlation_analysis' in metrics:
            corr_analysis = metrics['correlation_analysis']
            if 'independence_score' in corr_analysis:
                score += 0.15 * corr_analysis['independence_score']
        
        return min(1.0, max(0.0, score))
    
    def _analyze_graph_properties(self, graph: nx.DiGraph) -> Dict:
        """分析图结构特性"""
        return {
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_dag': nx.is_directed_acyclic_graph(graph),
            'weakly_connected': nx.is_weakly_connected(graph)
        }
    
    def _evaluate_stage_distribution(self, stages: Dict, features: Dict) -> Dict:
        """评估阶段分布合理性"""
        stage_sizes = {stage: len(stroke_list) for stage, stroke_list in stages.items()}
        total_strokes = sum(stage_sizes.values())
        
        # 理想分布：主体结构 40%，局部细节 40%，装饰 20%
        ideal_ratios = {
            'main_structure': 0.4,
            'local_details': 0.4,
            'decorative': 0.2
        }
        
        distribution_score = 0.0
        for stage, ideal_ratio in ideal_ratios.items():
            if stage in stage_sizes and total_strokes > 0:
                actual_ratio = stage_sizes[stage] / total_strokes
                deviation = abs(actual_ratio - ideal_ratio)
                stage_score = max(0, 1 - deviation * 2)  # 偏差越小得分越高
                distribution_score += stage_score
        
        distribution_score /= len(ideal_ratios)
        
        return {
            'stage_sizes': stage_sizes,
            'distribution_score': distribution_score,
            'total_strokes': total_strokes
        }
    
    def _evaluate_partial_order_consistency(self, graph: nx.DiGraph, features: Dict) -> float:
        """评估偏序关系一致性"""
        if not graph.edges():
            return 1.0
        
        consistent_edges = 0
        total_edges = 0
        
        for u, v in graph.edges():
            if u in features and v in features:
                score_u = features[u]['comprehensive_score']
                score_v = features[v]['comprehensive_score']
                
                # 检查边的方向是否与得分一致
                if score_u >= score_v:  # u应该在v之前绘制
                    consistent_edges += 1
                total_edges += 1
        
        return consistent_edges / total_edges if total_edges > 0 else 1.0
    
    def _evaluate_hierarchy_clarity(self, graph: nx.DiGraph, stages: Dict) -> float:
        """评估层次结构清晰度"""
        # 检查阶段间的边连接
        stage_order = ['main_structure', 'local_details', 'decorative']
        stage_mapping = {}
        
        for stage, stroke_list in stages.items():
            for stroke_id in stroke_list:
                stage_mapping[stroke_id] = stage
        
        correct_connections = 0
        total_connections = 0
        
        for u, v in graph.edges():
            if u in stage_mapping and v in stage_mapping:
                stage_u = stage_mapping[u]
                stage_v = stage_mapping[v]
                
                u_level = stage_order.index(stage_u) if stage_u in stage_order else -1
                v_level = stage_order.index(stage_v) if stage_v in stage_order else -1
                
                # 检查连接是否符合层次顺序
                if u_level <= v_level:
                    correct_connections += 1
                total_connections += 1
        
        return correct_connections / total_connections if total_connections > 0 else 1.0
    
    def _evaluate_topological_validity(self, graph: nx.DiGraph) -> float:
        """评估拓扑排序有效性"""
        try:
            # 检查是否为DAG
            if not nx.is_directed_acyclic_graph(graph):
                return 0.0
            
            # 尝试拓扑排序
            topo_order = list(nx.topological_sort(graph))
            return 1.0 if len(topo_order) == graph.number_of_nodes() else 0.0
            
        except nx.NetworkXError:
            return 0.0
    
    def _calculate_structure_quality_score(self, metrics: Dict) -> float:
        """计算结构构建质量得分"""
        score = 0.0
        
        # 图结构有效性权重 25%
        graph_props = metrics.get('graph_properties', {})
        if graph_props.get('is_dag', False):
            score += 0.25
        
        # 阶段分布权重 25%
        stage_dist = metrics.get('stage_distribution', {})
        score += 0.25 * stage_dist.get('distribution_score', 0)
        
        # 偏序一致性权重 25%
        consistency = metrics.get('consistency_score', 0)
        score += 0.25 * consistency
        
        # 层次清晰度权重 25%
        hierarchy = metrics.get('hierarchy_clarity', 0)
        score += 0.25 * hierarchy
        
        return min(1.0, max(0.0, score))
    
    def _evaluate_constraint_satisfaction(self, optimized_order: List[Tuple[int, str]], 
                                        hasse_graph: nx.DiGraph) -> float:
        """评估约束满足度"""
        if not hasse_graph.edges():
            return 1.0
        
        # 创建排序位置映射
        position_map = {stroke_id: i for i, (stroke_id, _) in enumerate(optimized_order)}
        
        satisfied_constraints = 0
        total_constraints = 0
        
        for u, v in hasse_graph.edges():
            if u in position_map and v in position_map:
                # u应该在v之前
                if position_map[u] < position_map[v]:
                    satisfied_constraints += 1
                total_constraints += 1
        
        return satisfied_constraints / total_constraints if total_constraints > 0 else 1.0
    
    def _evaluate_order_rationality(self, optimized_order: List[Tuple[int, str]], 
                                  features: Dict, stages: Dict) -> Dict:
        """评估绘制顺序合理性"""
        stroke_ids = [item[0] for item in optimized_order]
        
        # 检查综合得分的单调性
        scores = [features[sid]['comprehensive_score'] for sid in stroke_ids 
                 if sid in features]
        
        if len(scores) < 2:
            return {'monotonicity_score': 1.0}
        
        # 计算Spearman相关系数（应该为负，因为高分在前）
        order_indices = list(range(len(scores)))
        correlation, _ = stats.spearmanr(order_indices, scores)
        
        # 转换为单调性得分
        monotonicity_score = max(0, -correlation)  # 负相关越强越好
        
        return {
            'monotonicity_score': monotonicity_score,
            'score_correlation': correlation
        }
    
    def _evaluate_direction_accuracy(self, optimized_order: List[Tuple[int, str]], 
                                   features: Dict) -> float:
        """评估方向选择准确性"""
        # 这里简化处理，假设方向选择基于某些特征是合理的
        # 实际实现中可以根据具体的方向选择规则来评估
        
        forward_count = sum(1 for _, direction in optimized_order if direction == 'forward')
        reverse_count = len(optimized_order) - forward_count
        
        # 假设理想的前向/反向比例为 7:3
        total = len(optimized_order)
        if total > 0:
            forward_ratio = forward_count / total
            ideal_forward_ratio = 0.7
            deviation = abs(forward_ratio - ideal_forward_ratio)
            return max(0, 1 - deviation * 2)
        
        return 1.0
    
    def _analyze_convergence(self, optimized_order: List[Tuple[int, str]], 
                           features: Dict) -> Dict:
        """分析能量函数收敛性"""
        # 简化的收敛性分析
        stroke_ids = [item[0] for item in optimized_order]
        scores = [features[sid]['comprehensive_score'] for sid in stroke_ids 
                 if sid in features]
        
        if len(scores) < 2:
            return {'convergence_score': 1.0}
        
        # 计算得分变化的平滑度
        differences = np.diff(scores)
        smoothness = 1.0 / (1.0 + np.std(differences))  # 变化越平滑越好
        
        return {
            'convergence_score': smoothness,
            'score_variance': np.var(scores)
        }
    
    def _evaluate_stage_continuity(self, optimized_order: List[Tuple[int, str]], 
                                 stages: Dict) -> float:
        """评估阶段连续性"""
        stroke_ids = [item[0] for item in optimized_order]
        
        # 创建笔触到阶段的映射
        stroke_to_stage = {}
        for stage, stroke_list in stages.items():
            for stroke_id in stroke_list:
                stroke_to_stage[stroke_id] = stage
        
        # 计算阶段切换次数
        stage_sequence = [stroke_to_stage.get(sid, 'unknown') for sid in stroke_ids]
        
        switches = 0
        for i in range(1, len(stage_sequence)):
            if stage_sequence[i] != stage_sequence[i-1]:
                switches += 1
        
        # 理想情况下，阶段切换次数应该较少
        max_possible_switches = len(stage_sequence) - 1
        if max_possible_switches > 0:
            continuity_score = 1.0 - (switches / max_possible_switches)
        else:
            continuity_score = 1.0
        
        return max(0, continuity_score)
    
    def _calculate_ordering_quality_score(self, metrics: Dict) -> float:
        """计算排序优化质量得分"""
        score = 0.0
        
        # 约束满足度权重 30%
        constraint_satisfaction = metrics.get('constraint_satisfaction', 0)
        score += 0.3 * constraint_satisfaction
        
        # 顺序合理性权重 25%
        order_rationality = metrics.get('order_rationality', {})
        score += 0.25 * order_rationality.get('monotonicity_score', 0)
        
        # 方向准确性权重 20%
        direction_accuracy = metrics.get('direction_accuracy', 0)
        score += 0.2 * direction_accuracy
        
        # 收敛性权重 15%
        convergence = metrics.get('convergence_analysis', {})
        score += 0.15 * convergence.get('convergence_score', 0)
        
        # 阶段连续性权重 10%
        stage_continuity = metrics.get('stage_continuity', 0)
        score += 0.1 * stage_continuity
        
        return min(1.0, max(0.0, score))
    
    def _get_video_info(self, video_path: Path) -> Dict:
        """获取视频基本信息"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            cap.release()
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def _evaluate_animation_smoothness(self, video_path: Path) -> float:
        """评估动画流畅性"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            frame_diffs = []
            prev_frame = None
            
            # 采样部分帧进行分析
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(1, frame_count // 50)  # 最多采样50帧
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray_frame)
                    frame_diffs.append(np.mean(diff))
                
                prev_frame = gray_frame
            
            cap.release()
            
            if frame_diffs:
                # 流畅性通过帧间差异的稳定性来衡量
                smoothness = 1.0 / (1.0 + np.std(frame_diffs) / np.mean(frame_diffs))
                return min(1.0, smoothness)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _evaluate_animation_completeness(self, video_path: Path, strokes: List[Stroke]) -> float:
        """评估动画完整性"""
        # 简化实现：检查视频时长是否合理
        try:
            video_info = self._get_video_info(video_path)
            duration = video_info.get('duration', 0)
            
            # 假设每个笔触需要1-3秒绘制
            expected_duration = len(strokes) * 2  # 平均2秒每笔触
            
            if expected_duration > 0:
                completeness = min(1.0, duration / expected_duration)
                return completeness
            else:
                return 1.0
                
        except Exception:
            return 0.0
    
    def _evaluate_temporal_consistency(self, video_path: Path, 
                                     optimized_order: List[Tuple[int, str]]) -> float:
        """评估时序一致性"""
        # 简化实现：检查视频帧数与排序长度的一致性
        try:
            video_info = self._get_video_info(video_path)
            frame_count = video_info.get('frame_count', 0)
            fps = video_info.get('fps', 24)
            
            # 估算每个笔触的帧数
            expected_frames_per_stroke = fps * 2  # 假设每笔触2秒
            expected_total_frames = len(optimized_order) * expected_frames_per_stroke
            
            if expected_total_frames > 0:
                consistency = min(1.0, frame_count / expected_total_frames)
                return consistency
            else:
                return 1.0
                
        except Exception:
            return 0.0
    
    def _evaluate_visual_quality(self, video_path: Path) -> float:
        """评估视觉质量"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # 采样几帧进行质量分析
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = [frame_count // 4, frame_count // 2, frame_count * 3 // 4]
            
            quality_scores = []
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 使用拉普拉斯算子评估清晰度
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # 归一化清晰度得分
                    clarity_score = min(1.0, laplacian_var / 1000)
                    quality_scores.append(clarity_score)
            
            cap.release()
            
            return np.mean(quality_scores) if quality_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_animation_quality_score(self, metrics: Dict) -> float:
        """计算动画质量得分"""
        score = 0.0
        
        # 流畅性权重 30%
        smoothness = metrics.get('smoothness_score', 0)
        score += 0.3 * smoothness
        
        # 完整性权重 25%
        completeness = metrics.get('completeness_score', 0)
        score += 0.25 * completeness
        
        # 时序一致性权重 25%
        temporal_consistency = metrics.get('temporal_consistency', 0)
        score += 0.25 * temporal_consistency
        
        # 视觉质量权重 20%
        visual_quality = metrics.get('visual_quality', 0)
        score += 0.2 * visual_quality
        
        return min(1.0, max(0.0, score))
    
    def _generate_recommendations(self, module_scores: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        threshold = 0.7  # 低于此分数的模块需要改进
        
        if module_scores.get('stroke_extraction', 0) < threshold:
            recommendations.append("建议优化笔触提取算法，提高覆盖率和连通性")
        
        if module_scores.get('feature_modeling', 0) < threshold:
            recommendations.append("建议改进特征提取方法，增强特征区分度")
        
        if module_scores.get('structure_construction', 0) < threshold:
            recommendations.append("建议优化结构构建算法，提高层次清晰度")
        
        if module_scores.get('stroke_ordering', 0) < threshold:
            recommendations.append("建议调整排序优化参数，提高约束满足度")
        
        if module_scores.get('animation_generation', 0) < threshold:
            recommendations.append("建议优化动画生成算法，提高流畅性和视觉质量")
        
        if not recommendations:
            recommendations.append("各模块表现良好，可考虑进一步优化细节")
        
        return recommendations

# 测试代码
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass