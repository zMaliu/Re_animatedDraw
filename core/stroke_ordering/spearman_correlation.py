# -*- coding: utf-8 -*-
"""
Spearman秩相关系数计算器

实现论文中要求的Spearman秩相关系数计算
用于正则化项，保持与多阶段结构的一致性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.stats import spearmanr, rankdata
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    """
    相关性计算结果
    """
    correlation: float
    p_value: float
    rank1: np.ndarray
    rank2: np.ndarray
    n_samples: int
    is_significant: bool = False
    
    def __post_init__(self):
        # 判断是否显著（p < 0.05）
        self.is_significant = self.p_value < 0.05


class SpearmanCorrelationCalculator:
    """
    Spearman秩相关系数计算器
    
    用于计算笔触顺序与多阶段结构的一致性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Spearman相关系数计算器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 计算参数
        self.significance_level = config.get('significance_level', 0.05)
        self.handle_ties = config.get('handle_ties', 'average')
        self.min_samples = config.get('min_samples', 3)
        
        # 正则化参数
        self.regularization_strength = config.get('regularization_strength', 1.0)
        self.stage_weight_main = config.get('stage_weight_main', 1.0)
        self.stage_weight_detail = config.get('stage_weight_detail', 0.8)
        self.stage_weight_decoration = config.get('stage_weight_decoration', 0.6)
        
    def calculate_correlation(self, order1: List[int], order2: List[int]) -> CorrelationResult:
        """
        计算两个顺序之间的Spearman秩相关系数
        
        Args:
            order1: 第一个顺序
            order2: 第二个顺序
            
        Returns:
            CorrelationResult: 相关性计算结果
        """
        try:
            if len(order1) != len(order2):
                raise ValueError("Orders must have the same length")
            
            if len(order1) < self.min_samples:
                self.logger.warning(f"Sample size {len(order1)} is too small for reliable correlation")
                return CorrelationResult(
                    correlation=0.0,
                    p_value=1.0,
                    rank1=np.array([]),
                    rank2=np.array([]),
                    n_samples=len(order1)
                )
            
            # 转换为numpy数组
            arr1 = np.array(order1)
            arr2 = np.array(order2)
            
            # 计算秩
            rank1 = rankdata(arr1, method=self.handle_ties)
            rank2 = rankdata(arr2, method=self.handle_ties)
            
            # 计算Spearman相关系数
            correlation, p_value = spearmanr(arr1, arr2)
            
            # 处理NaN值
            if np.isnan(correlation):
                correlation = 0.0
            if np.isnan(p_value):
                p_value = 1.0
            
            return CorrelationResult(
                correlation=float(correlation),
                p_value=float(p_value),
                rank1=rank1,
                rank2=rank2,
                n_samples=len(order1)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Spearman correlation: {str(e)}")
            return CorrelationResult(
                correlation=0.0,
                p_value=1.0,
                rank1=np.array([]),
                rank2=np.array([]),
                n_samples=0
            )
    
    def calculate_stage_consistency(self, current_order: List[int], 
                                  stage_classifications: Dict[int, str],
                                  ideal_stage_order: List[str]) -> float:
        """
        计算当前顺序与理想阶段顺序的一致性
        
        Args:
            current_order: 当前笔触顺序
            stage_classifications: 笔触阶段分类 {stroke_id: stage}
            ideal_stage_order: 理想阶段顺序 ['main', 'detail', 'decoration']
            
        Returns:
            float: 一致性得分（0-1）
        """
        try:
            if not current_order or not stage_classifications:
                return 0.0
            
            # 构建阶段到数值的映射
            stage_to_value = {stage: i for i, stage in enumerate(ideal_stage_order)}
            
            # 将当前顺序转换为阶段值序列
            current_stage_values = []
            ideal_stage_values = []
            
            for i, stroke_id in enumerate(current_order):
                if stroke_id in stage_classifications:
                    stage = stage_classifications[stroke_id]
                    if stage in stage_to_value:
                        current_stage_values.append(stage_to_value[stage])
                        ideal_stage_values.append(i / len(current_order))  # 归一化位置
            
            if len(current_stage_values) < self.min_samples:
                return 0.0
            
            # 计算相关性
            result = self.calculate_correlation(current_stage_values, ideal_stage_values)
            
            # 转换为一致性得分（0-1）
            consistency_score = (result.correlation + 1.0) / 2.0
            
            return float(np.clip(consistency_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating stage consistency: {str(e)}")
            return 0.0
    
    def calculate_regularization_penalty(self, current_order: List[int],
                                       stage_classifications: Dict[int, str]) -> float:
        """
        计算正则化惩罚项
        
        Args:
            current_order: 当前笔触顺序
            stage_classifications: 笔触阶段分类
            
        Returns:
            float: 正则化惩罚值
        """
        try:
            if not current_order or not stage_classifications:
                return 0.0
            
            # 理想阶段顺序
            ideal_stage_order = ['main', 'detail', 'decoration']
            
            # 计算阶段一致性
            consistency = self.calculate_stage_consistency(
                current_order, stage_classifications, ideal_stage_order
            )
            
            # 计算阶段跳跃惩罚
            jump_penalty = self._calculate_stage_jump_penalty(
                current_order, stage_classifications
            )
            
            # 计算阶段分布惩罚
            distribution_penalty = self._calculate_stage_distribution_penalty(
                current_order, stage_classifications
            )
            
            # 组合惩罚
            total_penalty = (
                (1.0 - consistency) * 0.5 +
                jump_penalty * 0.3 +
                distribution_penalty * 0.2
            ) * self.regularization_strength
            
            return float(total_penalty)
            
        except Exception as e:
            self.logger.error(f"Error calculating regularization penalty: {str(e)}")
            return 0.0
    
    def _calculate_stage_jump_penalty(self, current_order: List[int],
                                    stage_classifications: Dict[int, str]) -> float:
        """
        计算阶段跳跃惩罚
        
        Args:
            current_order: 当前笔触顺序
            stage_classifications: 笔触阶段分类
            
        Returns:
            float: 跳跃惩罚值
        """
        try:
            stage_order = ['main', 'detail', 'decoration']
            stage_to_index = {stage: i for i, stage in enumerate(stage_order)}
            
            jump_count = 0
            total_transitions = 0
            
            prev_stage_index = None
            
            for stroke_id in current_order:
                if stroke_id in stage_classifications:
                    stage = stage_classifications[stroke_id]
                    if stage in stage_to_index:
                        current_stage_index = stage_to_index[stage]
                        
                        if prev_stage_index is not None:
                            # 检查是否有阶段跳跃（跳过中间阶段）
                            if abs(current_stage_index - prev_stage_index) > 1:
                                jump_count += 1
                            total_transitions += 1
                        
                        prev_stage_index = current_stage_index
            
            if total_transitions > 0:
                return jump_count / total_transitions
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating stage jump penalty: {str(e)}")
            return 0.0
    
    def _calculate_stage_distribution_penalty(self, current_order: List[int],
                                            stage_classifications: Dict[int, str]) -> float:
        """
        计算阶段分布惩罚
        
        Args:
            current_order: 当前笔触顺序
            stage_classifications: 笔触阶段分类
            
        Returns:
            float: 分布惩罚值
        """
        try:
            # 统计各阶段在顺序中的位置分布
            stage_positions = {'main': [], 'detail': [], 'decoration': []}
            
            for i, stroke_id in enumerate(current_order):
                if stroke_id in stage_classifications:
                    stage = stage_classifications[stroke_id]
                    if stage in stage_positions:
                        normalized_position = i / len(current_order)
                        stage_positions[stage].append(normalized_position)
            
            penalty = 0.0
            
            # 检查主要笔触是否主要在前面
            if stage_positions['main']:
                main_mean_pos = float(np.mean(stage_positions['main']))
                if main_mean_pos > 0.5:  # 主要笔触应该在前半部分
                    penalty += (main_mean_pos - 0.5) * self.stage_weight_main
            
            # 检查装饰笔触是否主要在后面
            if stage_positions['decoration']:
                decoration_mean_pos = float(np.mean(stage_positions['decoration']))
                if decoration_mean_pos < 0.5:  # 装饰笔触应该在后半部分
                    penalty += (0.5 - decoration_mean_pos) * self.stage_weight_decoration
            
            # 检查细节笔触是否在中间
            if stage_positions['detail']:
                detail_mean_pos = float(np.mean(stage_positions['detail']))
                ideal_detail_pos = 0.5
                penalty += abs(detail_mean_pos - ideal_detail_pos) * self.stage_weight_detail
            
            return float(penalty)
            
        except Exception as e:
            self.logger.error(f"Error calculating stage distribution penalty: {str(e)}")
            return 0.0
    
    def analyze_order_quality(self, current_order: List[int],
                            stage_classifications: Dict[int, str]) -> Dict[str, Any]:
        """
        分析顺序质量
        
        Args:
            current_order: 当前笔触顺序
            stage_classifications: 笔触阶段分类
            
        Returns:
            Dict: 质量分析结果
        """
        try:
            ideal_stage_order = ['main', 'detail', 'decoration']
            
            # 计算各种指标
            consistency = self.calculate_stage_consistency(
                current_order, stage_classifications, ideal_stage_order
            )
            
            jump_penalty = self._calculate_stage_jump_penalty(
                current_order, stage_classifications
            )
            
            distribution_penalty = self._calculate_stage_distribution_penalty(
                current_order, stage_classifications
            )
            
            regularization_penalty = self.calculate_regularization_penalty(
                current_order, stage_classifications
            )
            
            # 计算总体质量得分
            quality_score = consistency * (1.0 - jump_penalty) * (1.0 - distribution_penalty)
            
            return {
                'consistency': consistency,
                'jump_penalty': jump_penalty,
                'distribution_penalty': distribution_penalty,
                'regularization_penalty': regularization_penalty,
                'quality_score': quality_score,
                'is_high_quality': quality_score > 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order quality: {str(e)}")
            return {
                'consistency': 0.0,
                'jump_penalty': 1.0,
                'distribution_penalty': 1.0,
                'regularization_penalty': 1.0,
                'quality_score': 0.0,
                'is_high_quality': False
            }