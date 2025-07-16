# -*- coding: utf-8 -*-
"""
排序优化器

实现论文中的自然演化策略(Natural Evolution Strategy)优化算法
用于优化笔画绘制顺序
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import random
import math
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from collections import defaultdict


@dataclass
class OptimizationResult:
    """
    优化结果数据结构
    
    Attributes:
        optimized_order (List[int]): 优化后的笔画顺序
        fitness_score (float): 适应度分数
        iterations (int): 迭代次数
        convergence_history (List[float]): 收敛历史
        optimization_time (float): 优化时间
        method (str): 优化方法
        metadata (Dict): 元数据
    """
    optimized_order: List[int]
    fitness_score: float
    iterations: int
    convergence_history: List[float]
    optimization_time: float
    method: str
    metadata: Dict[str, Any]


@dataclass
class FitnessWeights:
    """
    适应度权重配置
    
    Attributes:
        spatial_continuity (float): 空间连续性权重
        stroke_direction (float): 笔画方向权重
        semantic_order (float): 语义顺序权重
        temporal_flow (float): 时间流畅性权重
        aesthetic_appeal (float): 美学吸引力权重
    """
    spatial_continuity: float = 0.3
    stroke_direction: float = 0.2
    semantic_order: float = 0.2
    temporal_flow: float = 0.15
    aesthetic_appeal: float = 0.15


class OrderingOptimizer:
    """
    排序优化器
    
    使用多种优化算法来改进笔画绘制顺序
    """
    
    def __init__(self, config):
        """
        初始化排序优化器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 优化参数
        self.optimization_method = config['stroke_ordering'].get('optimization_method', 'evolution')
        self.max_iterations = config['stroke_ordering'].get('optimization_iterations', 100)
        self.population_size = config['stroke_ordering'].get('population_size', 50)
        self.mutation_rate = config['stroke_ordering'].get('mutation_rate', 0.1)
        self.crossover_rate = config['stroke_ordering'].get('crossover_rate', 0.8)
        self.convergence_threshold = config['stroke_ordering'].get('convergence_threshold', 1e-6)
        
        # 适应度权重
        weights_config = config['stroke_ordering'].get('fitness_weights', {})
        self.fitness_weights = FitnessWeights(
            spatial_continuity=weights_config.get('spatial_continuity', 0.3),
            stroke_direction=weights_config.get('stroke_direction', 0.2),
            semantic_order=weights_config.get('semantic_order', 0.2),
            temporal_flow=weights_config.get('temporal_flow', 0.15),
            aesthetic_appeal=weights_config.get('aesthetic_appeal', 0.15)
        )
        
        # 缓存
        self._distance_cache = {}
        self._fitness_cache = {}
    
    def _safe_compare(self, a, b, operator):
        """
        安全比较函数，避免数组真值歧义
        
        Args:
            a: 第一个值
            b: 第二个值
            operator (str): 比较操作符 ('>', '<', '>=', '<=', '==', '!=')
            
        Returns:
            bool: 比较结果
        """
        try:
            from utils.math_utils import ensure_scalar
            
            # 使用通用的标量转换函数
            a_val = ensure_scalar(a)
            b_val = ensure_scalar(b)
            
            # 执行比较
            if operator == '>':
                return a_val > b_val
            elif operator == '<':
                return a_val < b_val
            elif operator == '>=':
                return a_val >= b_val
            elif operator == '<=':
                return a_val <= b_val
            elif operator == '==':
                return a_val == b_val
            elif operator == '!=':
                return a_val != b_val
            else:
                raise ValueError(f"Unsupported operator: {operator}")
                
        except Exception as e:
            self.logger.warning(f"Error in safe comparison: {str(e)}")
            return False
    
    def optimize_order(self, strokes: List[Dict[str, Any]], 
                      initial_order: Optional[List[int]] = None) -> OptimizationResult:
        """
        优化笔画顺序
        
        Args:
            strokes (List[Dict]): 笔画列表
            initial_order (Optional[List[int]]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Optimizing order for {len(strokes)} strokes using {self.optimization_method}")
            
            # 初始化
            if initial_order is None:
                initial_order = list(range(len(strokes)))
            
            self.strokes = strokes
            self._clear_cache()
            
            # 选择优化方法
            if self.optimization_method == 'evolution':
                result = self._evolutionary_optimization(initial_order)
            elif self.optimization_method == 'simulated_annealing':
                result = self._simulated_annealing_optimization(initial_order)
            elif self.optimization_method == 'genetic':
                result = self._genetic_algorithm_optimization(initial_order)
            elif self.optimization_method == 'local_search':
                result = self._local_search_optimization(initial_order)
            elif self.optimization_method == 'hybrid':
                result = self._hybrid_optimization(initial_order)
            else:
                self.logger.warning(f"Unknown optimization method: {self.optimization_method}")
                result = self._evolutionary_optimization(initial_order)
            
            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time
            
            self.logger.info(f"Optimization completed in {optimization_time:.2f}s, "
                           f"fitness improved from {self._calculate_fitness(initial_order):.4f} "
                           f"to {result.fitness_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in order optimization: {str(e)}")
            return self._create_default_result(initial_order or list(range(len(strokes))))
    
    def _evolutionary_optimization(self, initial_order: List[int]) -> OptimizationResult:
        """
        自然演化策略优化
        
        Args:
            initial_order (List[int]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            n_strokes = len(initial_order)
            
            # 初始化种群
            population = self._initialize_population(initial_order, self.population_size)
            
            # 评估初始种群
            fitness_scores = [self._calculate_fitness(individual) for individual in population]
            
            convergence_history = []
            best_fitness = max(fitness_scores)
            best_individual = population[np.argmax(fitness_scores)].copy()
            
            convergence_history.append(best_fitness)
            
            # 演化循环
            for iteration in range(self.max_iterations):
                # 选择
                selected_population = self._tournament_selection(population, fitness_scores)
                
                # 交叉
                offspring = self._crossover(selected_population)
                
                # 变异
                offspring = self._mutate(offspring)
                
                # 评估后代
                offspring_fitness = [self._calculate_fitness(individual) for individual in offspring]
                
                # 环境选择（精英保留）
                combined_population = population + offspring
                combined_fitness = fitness_scores + offspring_fitness
                
                # 选择最好的个体
                sorted_indices = np.argsort(combined_fitness)[::-1]
                population = [combined_population[i] for i in sorted_indices[:self.population_size]]
                fitness_scores = [combined_fitness[i] for i in sorted_indices[:self.population_size]]
                
                # 更新最佳解
                current_best_fitness = fitness_scores[0]
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_individual = population[0].copy()
                
                convergence_history.append(best_fitness)
                
                # 收敛检查
                if iteration > 10:
                    recent_improvement = convergence_history[-1] - convergence_history[-10]
                    if recent_improvement < self.convergence_threshold:
                        self.logger.info(f"Converged at iteration {iteration}")
                        break
            
            return OptimizationResult(
                optimized_order=best_individual,
                fitness_score=best_fitness,
                iterations=iteration + 1,
                convergence_history=convergence_history,
                optimization_time=0.0,  # 将在外部设置
                method='evolution',
                metadata={
                    'population_size': self.population_size,
                    'final_population_diversity': self._calculate_population_diversity(population)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in evolutionary optimization: {str(e)}")
            return self._create_default_result(initial_order)
    
    def _simulated_annealing_optimization(self, initial_order: List[int]) -> OptimizationResult:
        """
        模拟退火优化
        
        Args:
            initial_order (List[int]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            current_order = initial_order.copy()
            current_fitness = self._calculate_fitness(current_order)
            
            best_order = current_order.copy()
            best_fitness = current_fitness
            
            convergence_history = [current_fitness]
            
            # 退火参数
            initial_temp = 100.0
            final_temp = 0.01
            cooling_rate = (final_temp / initial_temp) ** (1.0 / self.max_iterations)
            
            temperature = initial_temp
            
            for iteration in range(self.max_iterations):
                # 生成邻居解
                neighbor_order = self._generate_neighbor(current_order)
                neighbor_fitness = self._calculate_fitness(neighbor_order)
                
                # 接受准则
                delta = neighbor_fitness - current_fitness
                if delta > 0 or random.random() < math.exp(delta / temperature):
                    current_order = neighbor_order
                    current_fitness = neighbor_fitness
                    
                    # 更新最佳解
                    if current_fitness > best_fitness:
                        best_order = current_order.copy()
                        best_fitness = current_fitness
                
                convergence_history.append(best_fitness)
                
                # 降温
                temperature *= cooling_rate
                
                # 收敛检查
                if temperature < final_temp:
                    break
            
            return OptimizationResult(
                optimized_order=best_order,
                fitness_score=best_fitness,
                iterations=iteration + 1,
                convergence_history=convergence_history,
                optimization_time=0.0,
                method='simulated_annealing',
                metadata={
                    'final_temperature': temperature,
                    'acceptance_ratio': len([h for h in convergence_history if h > convergence_history[0]]) / len(convergence_history)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in simulated annealing optimization: {str(e)}")
            return self._create_default_result(initial_order)
    
    def _genetic_algorithm_optimization(self, initial_order: List[int]) -> OptimizationResult:
        """
        遗传算法优化
        
        Args:
            initial_order (List[int]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            # 初始化种群
            population = self._initialize_population(initial_order, self.population_size)
            convergence_history = []
            
            for generation in range(self.max_iterations):
                # 评估适应度
                fitness_scores = [self._calculate_fitness(individual) for individual in population]
                
                # 记录最佳适应度
                best_fitness = max(fitness_scores)
                convergence_history.append(best_fitness)
                
                # 选择父代
                parents = self._roulette_wheel_selection(population, fitness_scores)
                
                # 生成后代
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        if random.random() < self.crossover_rate:
                            child1, child2 = self._order_crossover(parents[i], parents[i + 1])
                        else:
                            child1, child2 = parents[i].copy(), parents[i + 1].copy()
                        
                        # 变异
                        if random.random() < self.mutation_rate:
                            child1 = self._swap_mutation(child1)
                        if random.random() < self.mutation_rate:
                            child2 = self._swap_mutation(child2)
                        
                        offspring.extend([child1, child2])
                
                # 精英保留
                elite_size = max(1, self.population_size // 10)
                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                elite = [population[i] for i in elite_indices]
                
                # 新种群
                population = elite + offspring[:self.population_size - elite_size]
            
            # 最终评估
            final_fitness = [self._calculate_fitness(individual) for individual in population]
            best_index = np.argmax(final_fitness)
            
            return OptimizationResult(
                optimized_order=population[best_index],
                fitness_score=final_fitness[best_index],
                iterations=self.max_iterations,
                convergence_history=convergence_history,
                optimization_time=0.0,
                method='genetic_algorithm',
                metadata={
                    'population_size': self.population_size,
                    'elite_size': elite_size
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in genetic algorithm optimization: {str(e)}")
            return self._create_default_result(initial_order)
    
    def _local_search_optimization(self, initial_order: List[int]) -> OptimizationResult:
        """
        局部搜索优化
        
        Args:
            initial_order (List[int]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            current_order = initial_order.copy()
            current_fitness = self._calculate_fitness(current_order)
            
            convergence_history = [current_fitness]
            improved = True
            iteration = 0
            
            while improved and iteration < self.max_iterations:
                improved = False
                iteration += 1
                
                # 尝试所有可能的2-opt交换
                for i in range(len(current_order)):
                    for j in range(i + 2, len(current_order)):
                        # 2-opt交换
                        new_order = current_order.copy()
                        new_order[i:j] = reversed(new_order[i:j])
                        
                        new_fitness = self._calculate_fitness(new_order)
                        
                        if new_fitness > current_fitness:
                            current_order = new_order
                            current_fitness = new_fitness
                            improved = True
                            break
                    
                    if improved:
                        break
                
                convergence_history.append(current_fitness)
            
            return OptimizationResult(
                optimized_order=current_order,
                fitness_score=current_fitness,
                iterations=iteration,
                convergence_history=convergence_history,
                optimization_time=0.0,
                method='local_search',
                metadata={
                    'improvement_steps': len([i for i in range(1, len(convergence_history)) 
                                            if self._safe_compare(convergence_history[i], convergence_history[i-1], '>')])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in local search optimization: {str(e)}")
            return self._create_default_result(initial_order)
    
    def _hybrid_optimization(self, initial_order: List[int]) -> OptimizationResult:
        """
        混合优化方法
        
        Args:
            initial_order (List[int]): 初始顺序
            
        Returns:
            OptimizationResult: 优化结果
        """
        try:
            # 第一阶段：遗传算法全局搜索
            ga_iterations = self.max_iterations // 2
            self.max_iterations = ga_iterations
            ga_result = self._genetic_algorithm_optimization(initial_order)
            
            # 第二阶段：局部搜索精细优化
            self.max_iterations = self.max_iterations
            ls_result = self._local_search_optimization(ga_result.optimized_order)
            
            # 合并结果
            combined_history = ga_result.convergence_history + ls_result.convergence_history
            
            return OptimizationResult(
                optimized_order=ls_result.optimized_order,
                fitness_score=ls_result.fitness_score,
                iterations=ga_result.iterations + ls_result.iterations,
                convergence_history=combined_history,
                optimization_time=0.0,
                method='hybrid',
                metadata={
                    'ga_iterations': ga_result.iterations,
                    'ls_iterations': ls_result.iterations,
                    'ga_final_fitness': ga_result.fitness_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in hybrid optimization: {str(e)}")
            return self._create_default_result(initial_order)
    
    def _calculate_fitness(self, order: List[int]) -> float:
        """
        计算适应度分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 适应度分数
        """
        try:
            order_key = tuple(order)
            if order_key in self._fitness_cache:
                return self._fitness_cache[order_key]
            
            # 计算各项适应度分量
            spatial_score = self._calculate_spatial_continuity(order)
            direction_score = self._calculate_stroke_direction_score(order)
            semantic_score = self._calculate_semantic_order_score(order)
            temporal_score = self._calculate_temporal_flow_score(order)
            aesthetic_score = self._calculate_aesthetic_appeal_score(order)
            
            # 加权综合
            total_fitness = (
                self.fitness_weights.spatial_continuity * spatial_score +
                self.fitness_weights.stroke_direction * direction_score +
                self.fitness_weights.semantic_order * semantic_score +
                self.fitness_weights.temporal_flow * temporal_score +
                self.fitness_weights.aesthetic_appeal * aesthetic_score
            )
            
            self._fitness_cache[order_key] = total_fitness
            return total_fitness
            
        except Exception as e:
            self.logger.error(f"Error calculating fitness: {str(e)}")
            return 0.0
    
    def _calculate_spatial_continuity(self, order: List[int]) -> float:
        """
        计算空间连续性分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 空间连续性分数
        """
        try:
            if len(order) <= 1:
                return 1.0
            
            from utils.math_utils import ensure_scalar
            
            total_distance = 0.0
            for i in range(len(order) - 1):
                stroke1 = self.strokes[order[i]]
                stroke2 = self.strokes[order[i + 1]]
                
                centroid1 = stroke1.get('centroid', (0, 0))
                centroid2 = stroke2.get('centroid', (0, 0))
                
                # 确保质心坐标是标量值
                centroid1 = [ensure_scalar(x) for x in centroid1]
                centroid2 = [ensure_scalar(x) for x in centroid2]
                
                # 计算欧氏距离
                centroid1_array = np.array(centroid1)
                centroid2_array = np.array(centroid2)
                distance = ensure_scalar(np.linalg.norm(centroid1_array - centroid2_array))
                total_distance += distance
            
            # 归一化：距离越小，分数越高
            avg_distance = total_distance / (len(order) - 1)
            score = 1.0 / (1.0 + avg_distance / 100.0)  # 100是归一化参数
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating spatial continuity: {str(e)}")
            return 0.5
    
    def _calculate_stroke_direction_score(self, order: List[int]) -> float:
        """
        计算笔画方向分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 方向分数
        """
        try:
            if len(order) <= 1:
                return 1.0
            
            direction_consistency = 0.0
            valid_pairs = 0
            
            for i in range(len(order) - 1):
                stroke1 = self.strokes[order[i]]
                stroke2 = self.strokes[order[i + 1]]
                
                from utils.math_utils import ensure_scalar
                
                orientation1 = ensure_scalar(stroke1.get('orientation', 0))
                orientation2 = ensure_scalar(stroke2.get('orientation', 0))
                
                # 计算方向差异
                angle_diff = ensure_scalar(abs(orientation1 - orientation2))
                angle_diff = ensure_scalar(min(angle_diff, 2 * np.pi - angle_diff))  # 取较小角度
                
                # 方向一致性：角度差异越小越好
                consistency = ensure_scalar(1.0 - (angle_diff / np.pi))
                direction_consistency = ensure_scalar(direction_consistency + consistency)
                valid_pairs += 1
            
            return direction_consistency / valid_pairs if valid_pairs > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating stroke direction score: {str(e)}")
            return 0.5
    
    def _calculate_semantic_order_score(self, order: List[int]) -> float:
        """
        计算语义顺序分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 语义分数
        """
        try:
            # 中国书法笔画顺序规则权重
            stroke_priority = {
                'horizontal': 1,    # 横
                'vertical': 2,      # 竖
                'left_falling': 3,  # 撇
                'right_falling': 4, # 捺
                'dot': 5,          # 点
                'hook': 6,         # 钩
                'turning': 7,      # 折
                'curve': 8,        # 弯
                'complex': 9       # 复合
            }
            
            violations = 0
            total_pairs = 0
            
            for i in range(len(order) - 1):
                stroke1 = self.strokes[order[i]]
                stroke2 = self.strokes[order[i + 1]]
                
                class1 = stroke1.get('stroke_class', 'complex')
                class2 = stroke2.get('stroke_class', 'complex')
                
                priority1 = stroke_priority.get(class1, 9)
                priority2 = stroke_priority.get(class2, 9)
                
                # 检查是否违反顺序规则
                if priority1 > priority2:
                    violations += 1
                
                total_pairs += 1
            
            from utils.math_utils import ensure_scalar
            
            # 违反规则越少，分数越高
            violations = ensure_scalar(violations)
            total_pairs = ensure_scalar(total_pairs)
            score = ensure_scalar(1.0 - (violations / total_pairs)) if total_pairs > 0 else 1.0
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic order score: {str(e)}")
            return 0.5
    
    def _calculate_temporal_flow_score(self, order: List[int]) -> float:
        """
        计算时间流畅性分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 时间流畅性分数
        """
        try:
            if len(order) <= 1:
                return 1.0
            
            # 计算笔画间的时间间隔变化
            time_intervals = []
            
            for i in range(len(order) - 1):
                stroke1 = self.strokes[order[i]]
                stroke2 = self.strokes[order[i + 1]]
                
                # 基于笔画复杂度估算绘制时间
                complexity1 = stroke1.get('complexity', 1.0)
                complexity2 = stroke2.get('complexity', 1.0)
                
                # 基于距离估算移动时间
                from utils.math_utils import ensure_scalar
                
                centroid1 = stroke1.get('centroid', (0, 0))
                centroid2 = stroke2.get('centroid', (0, 0))
                
                # 确保质心坐标是标量值
                centroid1 = [ensure_scalar(x) for x in centroid1]
                centroid2 = [ensure_scalar(x) for x in centroid2]
                
                # 计算欧氏距离
                centroid1_array = np.array(centroid1)
                centroid2_array = np.array(centroid2)
                distance = ensure_scalar(np.linalg.norm(centroid1_array - centroid2_array))
                
                # 综合时间间隔
                interval = ensure_scalar(complexity1 + distance / 100.0 + complexity2)
                time_intervals.append(interval)
            
            from utils.math_utils import ensure_scalar
            
            # 计算时间间隔的变异系数（越小越流畅）
            if len(time_intervals) > 1:
                mean_interval = ensure_scalar(np.mean(time_intervals))
                std_interval = ensure_scalar(np.std(time_intervals))
                cv = ensure_scalar(std_interval / mean_interval if mean_interval > 0 else 1.0)
                score = ensure_scalar(1.0 / (1.0 + cv))
            else:
                score = 1.0
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal flow score: {str(e)}")
            return 0.5
    
    def _calculate_aesthetic_appeal_score(self, order: List[int]) -> float:
        """
        计算美学吸引力分数
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            float: 美学分数
        """
        try:
            # 基于黄金比例和对称性的美学评估
            if len(order) <= 1:
                return 1.0
            
            from utils.math_utils import ensure_scalar
            
            # 计算笔画分布的均匀性
            centroids = [self.strokes[i].get('centroid', (0, 0)) for i in order]
            centroids = [[ensure_scalar(x) for x in c] for c in centroids]
            x_coords = [c[0] for c in centroids]
            y_coords = [c[1] for c in centroids]
            
            # 分布均匀性
            x_std = ensure_scalar(np.std(x_coords))
            x_mean = ensure_scalar(np.mean(x_coords))
            y_std = ensure_scalar(np.std(y_coords))
            y_mean = ensure_scalar(np.mean(y_coords))
            
            x_uniformity = 1.0 - (x_std / (x_mean + 1e-6))
            y_uniformity = 1.0 - (y_std / (y_mean + 1e-6))
            
            uniformity_score = ensure_scalar((x_uniformity + y_uniformity) / 2.0)
            uniformity_score = max(0.0, min(1.0, uniformity_score))
            
            # 对称性评估
            center_x = ensure_scalar(np.mean(x_coords))
            center_y = ensure_scalar(np.mean(y_coords))
            
            symmetry_score = 0.0
            for i, (x, y) in enumerate(centroids):
                # 寻找对称点
                symmetric_x = ensure_scalar(2 * center_x - x)
                symmetric_y = ensure_scalar(2 * center_y - y)
                
                min_distance = float('inf')
                for j, (sx, sy) in enumerate(centroids):
                    if i != j:
                        distance = ensure_scalar(np.sqrt((sx - symmetric_x)**2 + (sy - symmetric_y)**2))
                        min_distance = min(min_distance, distance)
                
                # 对称性分数：距离越小越对称
                symmetry_score += ensure_scalar(1.0 / (1.0 + min_distance / 50.0))
            
            symmetry_score = ensure_scalar(symmetry_score / len(centroids))
            
            # 综合美学分数
            aesthetic_score = (uniformity_score + symmetry_score) / 2.0
            
            return aesthetic_score
            
        except Exception as e:
            self.logger.error(f"Error calculating aesthetic appeal score: {str(e)}")
            return 0.5
    
    def _initialize_population(self, initial_order: List[int], 
                             population_size: int) -> List[List[int]]:
        """
        初始化种群
        
        Args:
            initial_order (List[int]): 初始顺序
            population_size (int): 种群大小
            
        Returns:
            List[List[int]]: 种群
        """
        population = [initial_order.copy()]
        
        for _ in range(population_size - 1):
            individual = initial_order.copy()
            # 随机打乱
            random.shuffle(individual)
            population.append(individual)
        
        return population
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[List[int]]:
        """
        锦标赛选择
        
        Args:
            population (List[List[int]]): 种群
            fitness_scores (List[float]): 适应度分数
            tournament_size (int): 锦标赛大小
            
        Returns:
            List[List[int]]: 选择的个体
        """
        selected = []
        
        for _ in range(len(population)):
            # 随机选择参赛者
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            
            # 选择最佳个体
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_index].copy())
        
        return selected
    
    def _crossover(self, population: List[List[int]]) -> List[List[int]]:
        """
        交叉操作
        
        Args:
            population (List[List[int]]): 种群
            
        Returns:
            List[List[int]]: 后代
        """
        offspring = []
        
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                if random.random() < self.crossover_rate:
                    child1, child2 = self._order_crossover(population[i], population[i + 1])
                else:
                    child1, child2 = population[i].copy(), population[i + 1].copy()
                
                offspring.extend([child1, child2])
            else:
                offspring.append(population[i].copy())
        
        return offspring
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        顺序交叉(OX)
        
        Args:
            parent1 (List[int]): 父代1
            parent2 (List[int]): 父代2
            
        Returns:
            Tuple[List[int], List[int]]: 两个子代
        """
        size = len(parent1)
        
        # 选择交叉点
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        
        # 创建子代
        child1 = [-1] * size
        child2 = [-1] * size
        
        # 复制选定区间
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # 填充剩余位置
        self._fill_remaining_ox(child1, parent2, start, end)
        self._fill_remaining_ox(child2, parent1, start, end)
        
        return child1, child2
    
    def _fill_remaining_ox(self, child: List[int], parent: List[int], 
                          start: int, end: int) -> None:
        """
        填充OX交叉的剩余位置
        
        Args:
            child (List[int]): 子代
            parent (List[int]): 父代
            start (int): 开始位置
            end (int): 结束位置
        """
        parent_idx = end
        child_idx = end
        
        while -1 in child:
            parent_element = parent[parent_idx % len(parent)]
            # 安全检查元素是否在child中
            if isinstance(parent_element, np.ndarray):
                element_in_child = any(np.array_equal(parent_element, c) if isinstance(c, np.ndarray) else False for c in child)
            else:
                element_in_child = parent_element in child
            
            if not element_in_child:
                child[child_idx % len(child)] = parent_element
                child_idx += 1
            parent_idx += 1
    
    def _mutate(self, population: List[List[int]]) -> List[List[int]]:
        """
        变异操作
        
        Args:
            population (List[List[int]]): 种群
            
        Returns:
            List[List[int]]: 变异后的种群
        """
        for individual in population:
            if random.random() < self.mutation_rate:
                # 随机选择变异类型
                mutation_type = random.choice(['swap', 'insert', 'invert'])
                
                if mutation_type == 'swap':
                    self._swap_mutation(individual)
                elif mutation_type == 'insert':
                    self._insert_mutation(individual)
                elif mutation_type == 'invert':
                    self._invert_mutation(individual)
        
        return population
    
    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """
        交换变异
        
        Args:
            individual (List[int]): 个体
            
        Returns:
            List[int]: 变异后的个体
        """
        if len(individual) > 1:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def _insert_mutation(self, individual: List[int]) -> List[int]:
        """
        插入变异
        
        Args:
            individual (List[int]): 个体
            
        Returns:
            List[int]: 变异后的个体
        """
        if len(individual) > 1:
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            
            element = individual.pop(i)
            individual.insert(j, element)
        
        return individual
    
    def _invert_mutation(self, individual: List[int]) -> List[int]:
        """
        倒置变异
        
        Args:
            individual (List[int]): 个体
            
        Returns:
            List[int]: 变异后的个体
        """
        if len(individual) > 1:
            i = random.randint(0, len(individual) - 1)
            j = random.randint(i + 1, len(individual))
            individual[i:j] = reversed(individual[i:j])
        
        return individual
    
    def _generate_neighbor(self, order: List[int]) -> List[int]:
        """
        生成邻居解
        
        Args:
            order (List[int]): 当前顺序
            
        Returns:
            List[int]: 邻居顺序
        """
        neighbor = order.copy()
        
        # 随机选择邻居生成方法
        method = random.choice(['swap', '2opt', 'insert'])
        
        if method == 'swap':
            self._swap_mutation(neighbor)
        elif method == '2opt':
            if len(neighbor) > 3:
                i = random.randint(0, len(neighbor) - 3)
                j = random.randint(i + 2, len(neighbor))
                neighbor[i:j] = reversed(neighbor[i:j])
        elif method == 'insert':
            self._insert_mutation(neighbor)
        
        return neighbor
    
    def _roulette_wheel_selection(self, population: List[List[int]], 
                                fitness_scores: List[float]) -> List[List[int]]:
        """
        轮盘赌选择
        
        Args:
            population (List[List[int]]): 种群
            fitness_scores (List[float]): 适应度分数
            
        Returns:
            List[List[int]]: 选择的个体
        """
        # 确保适应度为正数
        min_fitness = min(fitness_scores)
        adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        
        total_fitness = sum(adjusted_fitness)
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        selected = []
        for _ in range(len(population)):
            r = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(population[i].copy())
                    break
        
        return selected
    
    def _calculate_population_diversity(self, population: List[List[int]]) -> float:
        """
        计算种群多样性
        
        Args:
            population (List[List[int]]): 种群
            
        Returns:
            float: 多样性分数
        """
        try:
            if len(population) <= 1:
                return 0.0
            
            total_distance = 0.0
            count = 0
            
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    # 计算两个个体的差异
                    differences = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                    distance = differences / len(population[i])
                    total_distance += distance
                    count += 1
            
            return total_distance / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating population diversity: {str(e)}")
            return 0.0
    
    def _clear_cache(self) -> None:
        """
        清空缓存
        """
        self._distance_cache.clear()
        self._fitness_cache.clear()
    
    def _create_default_result(self, order: List[int]) -> OptimizationResult:
        """
        创建默认优化结果
        
        Args:
            order (List[int]): 顺序
            
        Returns:
            OptimizationResult: 默认结果
        """
        return OptimizationResult(
            optimized_order=order,
            fitness_score=0.5,
            iterations=0,
            convergence_history=[0.5],
            optimization_time=0.0,
            method='default',
            metadata={}
        )