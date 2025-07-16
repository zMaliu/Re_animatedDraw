# -*- coding: utf-8 -*-
"""
能量函数

实现论文中的能量函数设计：
1. 一致性成本（视觉相似性）
2. 变化成本（墨湿度、厚度、尺寸变化）
3. 正则化项（与多阶段结构一致性）
4. 全局能量最小化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
import cv2


@dataclass
class EnergyComponents:
    """
    能量函数组件
    """
    consistency_cost: float = 0.0
    variation_cost: float = 0.0
    regularization_cost: float = 0.0
    total_energy: float = 0.0
    
    # 详细分解
    color_consistency: float = 0.0
    shape_consistency: float = 0.0
    spatial_consistency: float = 0.0
    wetness_variation: float = 0.0
    thickness_variation: float = 0.0
    size_variation: float = 0.0
    stage_regularization: float = 0.0


class EnergyFunction:
    """
    能量函数
    
    将笔触排序问题转化为全局能量最小化问题
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化能量函数
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 权重参数
        self.consistency_weight = config.get('consistency_weight', 0.4)
        self.variation_weight = config.get('variation_weight', 0.4)
        self.regularization_weight = config.get('regularization_weight', 0.2)
        
        # 一致性成本权重
        self.color_weight = config.get('color_consistency_weight', 0.3)
        self.shape_weight = config.get('shape_consistency_weight', 0.4)
        self.spatial_weight = config.get('spatial_consistency_weight', 0.3)
        
        # 变化成本权重
        self.wetness_weight = config.get('wetness_variation_weight', 0.4)
        self.thickness_weight = config.get('thickness_variation_weight', 0.3)
        self.size_weight = config.get('size_variation_weight', 0.3)
        
        # 距离计算参数
        self.max_spatial_distance = config.get('max_spatial_distance', 1000.0)
        self.color_distance_method = config.get('color_distance_method', 'euclidean')
        self.shape_distance_method = config.get('shape_distance_method', 'hausdorff')
        
        # 变化规则参数
        self.wetness_penalty_factor = config.get('wetness_penalty_factor', 2.0)
        self.thickness_penalty_factor = config.get('thickness_penalty_factor', 1.5)
        self.size_penalty_factor = config.get('size_penalty_factor', 1.2)
        
        # 缓存
        self._distance_cache = {}
        self._feature_cache = {}
        
    def calculate_energy(self, order: List[int], 
                        stroke_features: List[Dict[str, Any]],
                        stage_order: Optional[List[int]] = None) -> EnergyComponents:
        """
        计算给定顺序的总能量
        
        Args:
            order: 笔触绘制顺序
            stroke_features: 笔触特征列表
            stage_order: 多阶段结构顺序（可选）
            
        Returns:
            EnergyComponents: 能量组件
        """
        try:
            if len(order) != len(stroke_features):
                raise ValueError("Order length must match stroke features length")
            
            # 计算一致性成本
            consistency_components = self._calculate_consistency_cost(order, stroke_features)
            
            # 计算变化成本
            variation_components = self._calculate_variation_cost(order, stroke_features)
            
            # 计算正则化项
            regularization_cost = self._calculate_regularization_cost(order, stage_order)
            
            # 组合总能量
            total_consistency = (
                self.color_weight * consistency_components['color'] +
                self.shape_weight * consistency_components['shape'] +
                self.spatial_weight * consistency_components['spatial']
            )
            
            total_variation = (
                self.wetness_weight * variation_components['wetness'] +
                self.thickness_weight * variation_components['thickness'] +
                self.size_weight * variation_components['size']
            )
            
            total_energy = (
                self.consistency_weight * total_consistency +
                self.variation_weight * total_variation +
                self.regularization_weight * regularization_cost
            )
            
            return EnergyComponents(
                consistency_cost=total_consistency,
                variation_cost=total_variation,
                regularization_cost=regularization_cost,
                total_energy=total_energy,
                color_consistency=consistency_components['color'],
                shape_consistency=consistency_components['shape'],
                spatial_consistency=consistency_components['spatial'],
                wetness_variation=variation_components['wetness'],
                thickness_variation=variation_components['thickness'],
                size_variation=variation_components['size'],
                stage_regularization=regularization_cost
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating energy: {str(e)}")
            return EnergyComponents()
    
    def _calculate_consistency_cost(self, order: List[int], 
                                  stroke_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算一致性成本
        
        Args:
            order: 笔触绘制顺序
            stroke_features: 笔触特征列表
            
        Returns:
            Dict: 一致性成本组件
        """
        color_cost = 0.0
        shape_cost = 0.0
        spatial_cost = 0.0
        
        for i in range(len(order) - 1):
            try:
                current_idx = int(np.round(order[i]))
                next_idx = int(np.round(order[i + 1]))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid index in order: {order[i]}, {order[i + 1]}")
                continue
            
            # 边界检查
            if current_idx >= len(stroke_features) or next_idx >= len(stroke_features) or current_idx < 0 or next_idx < 0:
                continue
                
            current_features = stroke_features[current_idx]
            next_features = stroke_features[next_idx]
            
            # 颜色一致性
            color_distance = self._calculate_color_distance(current_features, next_features)
            color_cost += color_distance
            
            # 形状一致性
            shape_distance = self._calculate_shape_distance(current_features, next_features)
            shape_cost += shape_distance
            
            # 空间一致性
            spatial_distance = self._calculate_spatial_distance(current_features, next_features)
            spatial_cost += spatial_distance
        
        # 归一化
        n_transitions = len(order) - 1
        if n_transitions > 0:
            color_cost /= n_transitions
            shape_cost /= n_transitions
            spatial_cost /= n_transitions
        
        return {
            'color': color_cost,
            'shape': shape_cost,
            'spatial': spatial_cost
        }
    
    def _calculate_variation_cost(self, order: List[int], 
                                stroke_features: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算变化成本
        
        Args:
            order: 笔触绘制顺序
            stroke_features: 笔触特征列表
            
        Returns:
            Dict: 变化成本组件
        """
        wetness_cost = 0.0
        thickness_cost = 0.0
        size_cost = 0.0
        
        for i in range(len(order) - 1):
            try:
                current_idx = int(np.round(order[i]))
                next_idx = int(np.round(order[i + 1]))
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid index in order: {order[i]}, {order[i + 1]}")
                continue
            
            # 边界检查
            if current_idx >= len(stroke_features) or next_idx >= len(stroke_features) or current_idx < 0 or next_idx < 0:
                continue
                
            current_features = stroke_features[current_idx]
            next_features = stroke_features[next_idx]
            
            # 湿度变化成本（湿笔触应先于干笔触）
            wetness_penalty = self._calculate_wetness_penalty(current_features, next_features)
            wetness_cost += wetness_penalty
            
            # 厚度变化成本（粗笔触应先于细笔触）
            thickness_penalty = self._calculate_thickness_penalty(current_features, next_features)
            thickness_cost += thickness_penalty
            
            # 尺寸变化成本（大笔触应先于小笔触）
            size_penalty = self._calculate_size_penalty(current_features, next_features)
            size_cost += size_penalty
        
        # 归一化
        n_transitions = len(order) - 1
        if n_transitions > 0:
            wetness_cost /= n_transitions
            thickness_cost /= n_transitions
            size_cost /= n_transitions
        
        return {
            'wetness': wetness_cost,
            'thickness': thickness_cost,
            'size': size_cost
        }
    
    def _calculate_regularization_cost(self, order: List[int], 
                                     stage_order: Optional[List[int]]) -> float:
        """
        计算正则化项
        
        Args:
            order: 笔触绘制顺序
            stage_order: 多阶段结构顺序
            
        Returns:
            float: 正则化成本
        """
        if stage_order is None or len(stage_order) != len(order):
            return 0.0
        
        # 使用Spearman秩相关系数衡量与多阶段结构的一致性
        try:
            correlation, _ = spearmanr(order, stage_order)
            # 将相关系数转换为成本（相关性越高，成本越低）
            regularization_cost = 1.0 - abs(correlation) if not np.isnan(correlation) else 1.0
            return regularization_cost
        except Exception:
            return 1.0
    
    def _calculate_color_distance(self, features1: Dict[str, Any], 
                                features2: Dict[str, Any]) -> float:
        """
        计算颜色距离
        
        Args:
            features1: 第一个笔触特征
            features2: 第二个笔触特征
            
        Returns:
            float: 颜色距离
        """
        # 获取RGB颜色
        color1 = features1.get('rgb_mean', [0, 0, 0])
        color2 = features2.get('rgb_mean', [0, 0, 0])
        
        if self.color_distance_method == 'euclidean':
            distance = euclidean(color1, color2)
            # 归一化到[0,1]
            return min(distance / (255 * np.sqrt(3)), 1.0)
        elif self.color_distance_method == 'lab':
            # 转换为LAB颜色空间计算距离
            return self._calculate_lab_distance(color1, color2)
        else:
            return euclidean(color1, color2) / (255 * np.sqrt(3))
    
    def _calculate_lab_distance(self, rgb1: List[float], rgb2: List[float]) -> float:
        """
        计算LAB颜色空间距离
        
        Args:
            rgb1: 第一个RGB颜色
            rgb2: 第二个RGB颜色
            
        Returns:
            float: LAB距离
        """
        try:
            # 转换为LAB
            rgb1_array = np.uint8([[rgb1]])
            rgb2_array = np.uint8([[rgb2]])
            
            lab1 = cv2.cvtColor(rgb1_array, cv2.COLOR_RGB2LAB)[0, 0]
            lab2 = cv2.cvtColor(rgb2_array, cv2.COLOR_RGB2LAB)[0, 0]
            
            # 计算Delta E
            delta_e = euclidean(lab1, lab2)
            # 归一化（Delta E的最大值约为100）
            return min(delta_e / 100.0, 1.0)
        except Exception:
            # 回退到RGB欧几里得距离
            return euclidean(rgb1, rgb2) / (255 * np.sqrt(3))
    
    def _calculate_shape_distance(self, features1: Dict[str, Any], 
                                features2: Dict[str, Any]) -> float:
        """
        计算形状距离
        
        Args:
            features1: 第一个笔触特征
            features2: 第二个笔触特征
            
        Returns:
            float: 形状距离
        """
        # 获取形状特征
        elongation1 = features1.get('elongation_ratio', 1.0)
        elongation2 = features2.get('elongation_ratio', 1.0)
        
        roundness1 = features1.get('roundness', 0.0)
        roundness2 = features2.get('roundness', 0.0)
        
        # 计算特征差异
        elongation_diff = abs(elongation1 - elongation2)
        roundness_diff = abs(roundness1 - roundness2)
        
        # 组合形状距离
        shape_distance = 0.5 * elongation_diff + 0.5 * roundness_diff
        
        return min(shape_distance, 1.0)
    
    def _calculate_spatial_distance(self, features1: Dict[str, Any], 
                                  features2: Dict[str, Any]) -> float:
        """
        计算空间距离
        
        Args:
            features1: 第一个笔触特征
            features2: 第二个笔触特征
            
        Returns:
            float: 空间距离
        """
        # 获取位置信息
        pos1 = features1.get('position', [0, 0])
        pos2 = features2.get('position', [0, 0])
        
        # 计算欧几里得距离
        distance = euclidean(pos1, pos2)
        
        # 归一化
        normalized_distance = min(distance / self.max_spatial_distance, 1.0)
        
        return normalized_distance
    
    def _calculate_wetness_penalty(self, features1: Dict[str, Any], 
                                 features2: Dict[str, Any]) -> float:
        """
        计算湿度变化惩罚
        
        Args:
            features1: 当前笔触特征
            features2: 下一个笔触特征
            
        Returns:
            float: 湿度惩罚
        """
        wetness1 = features1.get('wetness', 0.5)
        wetness2 = features2.get('wetness', 0.5)
        
        # 如果从干到湿（违反规则），施加惩罚
        if wetness2 > wetness1:
            penalty = (wetness2 - wetness1) * self.wetness_penalty_factor
            return min(penalty, 1.0)
        
        return 0.0
    
    def _calculate_thickness_penalty(self, features1: Dict[str, Any], 
                                   features2: Dict[str, Any]) -> float:
        """
        计算厚度变化惩罚
        
        Args:
            features1: 当前笔触特征
            features2: 下一个笔触特征
            
        Returns:
            float: 厚度惩罚
        """
        thickness1 = features1.get('thickness', 0.5)
        thickness2 = features2.get('thickness', 0.5)
        
        # 如果从细到粗（违反规则），施加惩罚
        if thickness2 > thickness1:
            penalty = (thickness2 - thickness1) * self.thickness_penalty_factor
            return min(penalty, 1.0)
        
        return 0.0
    
    def _calculate_size_penalty(self, features1: Dict[str, Any], 
                              features2: Dict[str, Any]) -> float:
        """
        计算尺寸变化惩罚
        
        Args:
            features1: 当前笔触特征
            features2: 下一个笔触特征
            
        Returns:
            float: 尺寸惩罚
        """
        area1 = features1.get('area', 0.0)
        area2 = features2.get('area', 0.0)
        
        # 如果从小到大（违反规则），施加惩罚
        if area2 > area1 and area1 > 0:
            penalty = (area2 - area1) / area1 * self.size_penalty_factor
            return min(penalty, 1.0)
        
        return 0.0
    
    def calculate_energy_gradient(self, order: List[int], 
                                stroke_features: List[Dict[str, Any]],
                                stage_order: Optional[List[int]] = None) -> np.ndarray:
        """
        计算能量函数的梯度（用于基于梯度的优化）
        
        Args:
            order: 笔触绘制顺序
            stroke_features: 笔触特征列表
            stage_order: 多阶段结构顺序
            
        Returns:
            np.ndarray: 能量梯度
        """
        gradient = np.zeros(len(order))
        epsilon = 1e-6
        
        # 计算当前能量
        current_energy = self.calculate_energy(order, stroke_features, stage_order).total_energy
        
        # 数值梯度计算
        for i in range(len(order)):
            # 创建扰动
            perturbed_order = order.copy()
            if i < len(order) - 1:
                # 交换相邻元素
                perturbed_order[i], perturbed_order[i + 1] = perturbed_order[i + 1], perturbed_order[i]
                
                # 计算扰动后的能量
                perturbed_energy = self.calculate_energy(
                    perturbed_order, stroke_features, stage_order
                ).total_energy
                
                # 计算梯度
                gradient[i] = (perturbed_energy - current_energy) / epsilon
        
        return gradient
    
    def analyze_energy_landscape(self, orders: List[List[int]], 
                               stroke_features: List[Dict[str, Any]],
                               stage_order: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        分析能量景观
        
        Args:
            orders: 多个笔触顺序
            stroke_features: 笔触特征列表
            stage_order: 多阶段结构顺序
            
        Returns:
            Dict: 能量景观分析结果
        """
        energies = []
        components_list = []
        
        for order in orders:
            components = self.calculate_energy(order, stroke_features, stage_order)
            energies.append(components.total_energy)
            components_list.append(components)
        
        if not energies:
            return {}
        
        # 统计分析
        energies_array = np.array(energies)
        
        analysis = {
            'min_energy': float(np.min(energies_array)),
            'max_energy': float(np.max(energies_array)),
            'mean_energy': float(np.mean(energies_array)),
            'std_energy': float(np.std(energies_array)),
            'best_order_index': int(np.argmin(energies_array)),
            'worst_order_index': int(np.argmax(energies_array)),
            'energy_range': float(np.max(energies_array) - np.min(energies_array)),
            'component_analysis': self._analyze_energy_components(components_list)
        }
        
        return analysis
    
    def _analyze_energy_components(self, components_list: List[EnergyComponents]) -> Dict[str, Any]:
        """
        分析能量组件
        
        Args:
            components_list: 能量组件列表
            
        Returns:
            Dict: 组件分析结果
        """
        if not components_list:
            return {}
        
        # 提取各组件数据
        consistency_costs = [c.consistency_cost for c in components_list]
        variation_costs = [c.variation_cost for c in components_list]
        regularization_costs = [c.regularization_cost for c in components_list]
        
        return {
            'consistency_cost': {
                'mean': float(np.mean(consistency_costs)),
                'std': float(np.std(consistency_costs)),
                'min': float(np.min(consistency_costs)),
                'max': float(np.max(consistency_costs))
            },
            'variation_cost': {
                'mean': float(np.mean(variation_costs)),
                'std': float(np.std(variation_costs)),
                'min': float(np.min(variation_costs)),
                'max': float(np.max(variation_costs))
            },
            'regularization_cost': {
                'mean': float(np.mean(regularization_costs)),
                'std': float(np.std(regularization_costs)),
                'min': float(np.min(regularization_costs)),
                'max': float(np.max(regularization_costs))
            }
        }
    
    def get_energy_weights(self) -> Dict[str, float]:
        """
        获取能量权重配置
        
        Returns:
            Dict: 权重配置
        """
        return {
            'consistency_weight': self.consistency_weight,
            'variation_weight': self.variation_weight,
            'regularization_weight': self.regularization_weight,
            'color_weight': self.color_weight,
            'shape_weight': self.shape_weight,
            'spatial_weight': self.spatial_weight,
            'wetness_weight': self.wetness_weight,
            'thickness_weight': self.thickness_weight,
            'size_weight': self.size_weight
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        更新能量权重
        
        Args:
            new_weights: 新的权重配置
        """
        for key, value in new_weights.items():
            if hasattr(self, key) and isinstance(value, (int, float)):
                setattr(self, key, float(value))
                self.logger.info(f"Updated {key} to {value}")
    
    def clear_cache(self):
        """
        清空缓存
        """
        self._distance_cache.clear()
        self._feature_cache.clear()