#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块4: 笔触排序优化
通过自然演化策略（NES）优化笔触顺序，最小化能量函数，生成最终绘制序列

主要功能:
1. 能量函数构建（一致性成本、变化成本、正则化项）
2. NES优化算法
3. 笔触方向确定
4. 硬约束处理
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class StrokeOrderOptimizer:
    """笔触排序优化器"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.features_df = None
        self.hasse_graph = None
        self.stages = None
        
        # NES参数（根据论文表1）
        self.lambda_param = None  # 2*(4+log(M))，M为笔触组数
        self.eta_mu = None  # 学习率 1/(9+3log(M))
        self.eta_sigma = None  # 协方差学习率
        self.eta_B = None  # 旋转矩阵学习率
        
        # 能量函数权重
        self.w_cons = 1.0  # 一致性成本权重
        self.w_var = 1.0   # 变化成本权重
        self.w_reg = 0.5   # 正则化权重
    
    def optimize_order(self, hasse_graph: nx.DiGraph, features_df: pd.DataFrame, 
                      stages: Dict[str, List[int]]) -> List[Tuple[int, str]]:
        """优化笔触顺序"""
        self.hasse_graph = hasse_graph
        self.features_df = features_df
        self.stages = stages
        
        # 获取初始拓扑排序
        initial_order = self.get_topological_order(hasse_graph)
        
        if len(initial_order) == 0:
            return []
        
        # 设置NES参数
        M = len(stages)  # 阶段数
        self.lambda_param = max(4, int(2 * (4 + np.log(M))))
        self.eta_mu = 1.0 / (9 + 3 * np.log(M))
        self.eta_sigma = 0.1
        self.eta_B = 0.1
        
        if self.debug:
            print(f"NES参数: λ={self.lambda_param}, η_μ={self.eta_mu:.3f}")
            print(f"初始顺序长度: {len(initial_order)}")
        
        # 执行NES优化
        optimized_order = self._nes_optimization(initial_order)
        
        # 确定笔触方向
        order_with_direction = self._determine_stroke_directions(optimized_order)
        
        return order_with_direction
    
    def get_topological_order(self, hasse_graph: nx.DiGraph) -> List[int]:
        """获取拓扑排序结果"""
        try:
            return list(nx.topological_sort(hasse_graph))
        except nx.NetworkXError:
            # 如果有环，返回节点列表
            return list(hasse_graph.nodes())
    
    def _nes_optimization(self, initial_order: List[int]) -> List[int]:
        """NES优化算法"""
        n = len(initial_order)
        if n <= 1:
            return initial_order
        
        # 初始化分布参数
        mu = np.arange(n, dtype=float)  # 均值向量
        sigma = np.ones(n)  # 标准差向量
        B = np.eye(n)  # 旋转矩阵
        
        best_order = initial_order.copy()
        best_energy = self._compute_energy(initial_order)
        
        # 优化迭代
        max_iterations = 50
        no_improvement = 0
        
        for iteration in range(max_iterations):
            # 生成候选解
            candidates = []
            energies = []
            
            for _ in range(self.lambda_param):
                # 从多元正态分布采样
                z = np.random.randn(n)
                y = mu + sigma * (B @ z)
                
                # 转换为排列
                candidate_order = self._vector_to_permutation(y, initial_order)
                candidates.append(candidate_order)
                
                # 计算能量
                energy = self._compute_energy(candidate_order)
                energies.append(energy)
            
            # 选择最优解
            min_idx = np.argmin(energies)
            if energies[min_idx] < best_energy:
                best_energy = energies[min_idx]
                best_order = candidates[min_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1
            
            # 更新分布参数（简化版NES更新）
            if len(energies) > 1:
                # 选择前25%的解
                elite_size = max(1, self.lambda_param // 4)
                elite_indices = np.argsort(energies)[:elite_size]
                
                # 更新均值
                elite_vectors = []
                for idx in elite_indices:
                    vector = self._permutation_to_vector(candidates[idx], initial_order)
                    elite_vectors.append(vector)
                
                if elite_vectors:
                    new_mu = np.mean(elite_vectors, axis=0)
                    mu = (1 - self.eta_mu) * mu + self.eta_mu * new_mu
            
            if self.debug and iteration % 10 == 0:
                print(f"迭代 {iteration}: 最佳能量 = {best_energy:.4f}")
            
            # 早停条件
            if no_improvement > 10:
                break
        
        if self.debug:
            print(f"NES优化完成，最终能量: {best_energy:.4f}")
        
        return best_order
    
    def _vector_to_permutation(self, vector: np.ndarray, reference_order: List[int]) -> List[int]:
        """将实数向量转换为排列"""
        # 根据向量值排序得到排列
        sorted_indices = np.argsort(vector)
        return [reference_order[i] for i in sorted_indices]
    
    def _permutation_to_vector(self, permutation: List[int], reference_order: List[int]) -> np.ndarray:
        """将排列转换为实数向量"""
        n = len(reference_order)
        vector = np.zeros(n)
        
        for i, stroke_id in enumerate(permutation):
            if stroke_id in reference_order:
                ref_idx = reference_order.index(stroke_id)
                vector[ref_idx] = i
        
        return vector
    
    def _compute_energy(self, order: List[int]) -> float:
        """计算能量函数"""
        if len(order) <= 1:
            return 0.0
        
        # 一致性成本
        consistency_cost = self._compute_consistency_cost(order)
        
        # 变化成本
        variation_cost = self._compute_variation_cost(order)
        
        # 正则化项
        regularization = self._compute_regularization(order)
        
        total_energy = (self.w_cons * consistency_cost + 
                       self.w_var * variation_cost + 
                       self.w_reg * regularization)
        
        return total_energy
    
    def _compute_consistency_cost(self, order: List[int]) -> float:
        """计算一致性成本"""
        if self.features_df is None or len(order) <= 1:
            return 0.0
        
        cost = 0.0
        
        for i in range(len(order) - 1):
            stroke_i = order[i]
            stroke_j = order[i + 1]
            
            # 获取笔触特征
            features_i = self._get_stroke_features(stroke_i)
            features_j = self._get_stroke_features(stroke_j)
            
            if features_i is not None and features_j is not None:
                # 颜色相似性
                color_diff = abs(features_i.get('avg_gray', 0) - features_j.get('avg_gray', 0))
                
                # 形状相似性
                shape_diff = abs(features_i.get('circularity', 0) - features_j.get('circularity', 0))
                
                # 空间距离
                pos_i = (features_i.get('centroid_x', 0), features_i.get('centroid_y', 0))
                pos_j = (features_j.get('centroid_x', 0), features_j.get('centroid_y', 0))
                spatial_dist = euclidean(pos_i, pos_j) / 1000.0  # 归一化
                
                # 综合一致性成本
                cost += color_diff + shape_diff + spatial_dist
        
        return cost / (len(order) - 1) if len(order) > 1 else 0.0
    
    def _compute_variation_cost(self, order: List[int]) -> float:
        """计算变化成本"""
        if self.features_df is None or len(order) <= 1:
            return 0.0
        
        cost = 0.0
        
        for i in range(len(order) - 1):
            stroke_i = order[i]
            stroke_j = order[i + 1]
            
            features_i = self._get_stroke_features(stroke_i)
            features_j = self._get_stroke_features(stroke_j)
            
            if features_i is not None and features_j is not None:
                # 湿度变化
                wetness_change = abs(features_i.get('wetness', 0) - features_j.get('wetness', 0))
                
                # 厚度变化
                thickness_change = abs(features_i.get('thickness', 0) - features_j.get('thickness', 0))
                
                # 形状变化
                shape_change = abs(features_i.get('aspect_ratio', 1) - features_j.get('aspect_ratio', 1))
                
                # 鼓励平滑过渡（变化成本越小越好）
                cost += wetness_change + thickness_change + shape_change * 0.5
        
        return cost / (len(order) - 1) if len(order) > 1 else 0.0
    
    def _compute_regularization(self, order: List[int]) -> float:
        """计算正则化项（与拓扑排序的一致性）"""
        if self.hasse_graph is None:
            return 0.0
        
        topo_order = self.get_topological_order(self.hasse_graph)
        
        if len(topo_order) != len(order):
            return 1.0  # 惩罚不一致的长度
        
        # 计算Spearman秩相关系数
        try:
            # 创建排名映射
            topo_ranks = {stroke: i for i, stroke in enumerate(topo_order)}
            order_ranks = {stroke: i for i, stroke in enumerate(order)}
            
            # 获取共同的笔触
            common_strokes = set(topo_ranks.keys()) & set(order_ranks.keys())
            
            if len(common_strokes) < 2:
                return 0.0
            
            topo_rank_values = [topo_ranks[stroke] for stroke in common_strokes]
            order_rank_values = [order_ranks[stroke] for stroke in common_strokes]
            
            correlation, _ = spearmanr(topo_rank_values, order_rank_values)
            
            # 返回负相关性作为惩罚（相关性越高，惩罚越小）
            return 1.0 - correlation if not np.isnan(correlation) else 1.0
            
        except Exception:
            return 1.0
    
    def _get_stroke_features(self, stroke_id: int) -> Optional[Dict]:
        """获取笔触特征"""
        if self.features_df is None:
            return None
        
        stroke_row = self.features_df[self.features_df['stroke_id'] == stroke_id]
        if stroke_row.empty:
            return None
        
        return stroke_row.iloc[0].to_dict()
    
    def _determine_stroke_directions(self, order: List[int]) -> List[Tuple[int, str]]:
        """确定笔触绘制方向"""
        order_with_direction = []
        
        for stroke_id in order:
            features = self._get_stroke_features(stroke_id)
            
            if features is None:
                direction = 'forward'
            else:
                # 根据湿度和厚度规则确定方向
                wetness = features.get('wetness', 0.5)
                thickness = features.get('thickness', 0.5)
                
                # D1规则：从湿端到干端
                # D2规则：从粗端到细端
                # 这里简化为基于特征值的启发式规则
                if wetness > 0.6 or thickness > 0.6:
                    direction = 'forward'  # 从起点到终点
                else:
                    direction = 'reverse'  # 从终点到起点
            
            order_with_direction.append((stroke_id, direction))
        
        return order_with_direction
    
    def save_optimization_analysis(self, optimized_order: List[Tuple[int, str]], output_dir: Path):
        """保存优化分析结果"""
        if not self.debug:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存优化后的顺序
        order_data = {
            'optimized_order': optimized_order,
            'total_strokes': len(optimized_order)
        }
        
        with open(output_dir / "optimized_order.json", 'w', encoding='utf-8') as f:
            json.dump(order_data, f, indent=2, ensure_ascii=False)
        
        # 计算并保存能量分析
        stroke_order = [stroke_id for stroke_id, _ in optimized_order]
        final_energy = self._compute_energy(stroke_order)
        
        energy_analysis = {
            'final_energy': final_energy,
            'consistency_cost': self._compute_consistency_cost(stroke_order),
            'variation_cost': self._compute_variation_cost(stroke_order),
            'regularization': self._compute_regularization(stroke_order)
        }
        
        with open(output_dir / "energy_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(energy_analysis, f, indent=2, ensure_ascii=False)
        
        # 绘制顺序可视化
        if len(optimized_order) > 0:
            plt.figure(figsize=(12, 8))
            
            stroke_ids = [stroke_id for stroke_id, _ in optimized_order]
            directions = [direction for _, direction in optimized_order]
            
            # 绘制顺序图
            plt.subplot(2, 1, 1)
            plt.plot(range(len(stroke_ids)), stroke_ids, 'bo-', alpha=0.7)
            plt.xlabel('Drawing Order')
            plt.ylabel('Stroke ID')
            plt.title('Optimized Stroke Drawing Order')
            plt.grid(True, alpha=0.3)
            
            # 绘制方向分布
            plt.subplot(2, 1, 2)
            direction_counts = {'forward': directions.count('forward'), 
                              'reverse': directions.count('reverse')}
            plt.bar(direction_counts.keys(), direction_counts.values(), 
                   color=['blue', 'red'], alpha=0.7)
            plt.xlabel('Drawing Direction')
            plt.ylabel('Count')
            plt.title('Stroke Direction Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / "optimization_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"优化分析结果已保存至: {output_dir}")

# 测试代码
if __name__ == "__main__":
    from stroke_extraction import StrokeExtractor
    from feature_modeling import FeatureModeler
    from structure_construction import StructureConstructor
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python stroke_ordering.py <image_path>")
        sys.exit(1)
    
    # 完整流程测试
    extractor = StrokeExtractor(debug=True)
    strokes = extractor.extract_strokes(sys.argv[1])
    
    modeler = FeatureModeler(debug=True)
    features = modeler.extract_features(strokes)
    
    constructor = StructureConstructor(debug=True)
    hasse_graph, stages = constructor.build_structure(features)
    
    optimizer = StrokeOrderOptimizer(debug=True)
    optimized_order = optimizer.optimize_order(hasse_graph, features, stages)
    
    print(f"\n排序优化结果:")
    print(f"优化后顺序长度: {len(optimized_order)}")
    print(f"前10个笔触: {optimized_order[:10]}")
    
    # 计算最终能量
    stroke_order = [stroke_id for stroke_id, _ in optimized_order]
    final_energy = optimizer._compute_energy(stroke_order)
    print(f"最终能量: {final_energy:.4f}")