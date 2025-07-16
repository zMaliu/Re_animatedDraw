# -*- coding: utf-8 -*-
"""
排序优化器模块

提供笔触排序的优化算法和策略
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import time
from ..stroke_extraction.stroke_detector import Stroke
from .energy_function import EnergyFunction
from .nes_optimizer import NESOptimizer
from .constraint_handler import ConstraintHandler
from .order_evaluator import OrderEvaluator
from .spearman_correlation import SpearmanCorrelationCalculator


@dataclass
class OptimizationConfig:
    """优化配置"""
    max_iterations: int = 1000
    population_size: int = 50
    convergence_threshold: float = 1e-6
    early_stopping_patience: int = 50
    use_constraints: bool = True
    constraint_penalty: float = 1000.0
    optimization_method: str = 'nes'  # 'nes', 'genetic', 'simulated_annealing'
    

@dataclass
class OptimizationResult:
    """优化结果"""
    optimized_order: List[int]
    optimized_strokes: List[Stroke]
    best_energy: float
    convergence_history: List[float]
    iterations_used: int
    execution_time: float
    success: bool
    optimization_method: str
    metadata: Dict[str, Any]


class OrderingOptimizer:
    """排序优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化排序优化器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 优化配置
        self.opt_config = OptimizationConfig(
            max_iterations=config.get('max_iterations', 1000),
            population_size=config.get('population_size', 50),
            convergence_threshold=config.get('convergence_threshold', 1e-6),
            early_stopping_patience=config.get('early_stopping_patience', 50),
            use_constraints=config.get('use_constraints', True),
            constraint_penalty=config.get('constraint_penalty', 1000.0),
            optimization_method=config.get('optimization_method', 'nes')
        )
        
        # 初始化组件
        self.energy_function = EnergyFunction(config)
        self.nes_optimizer = NESOptimizer(config)
        self.constraint_handler = ConstraintHandler(config) if self.opt_config.use_constraints else None
        self.order_evaluator = OrderEvaluator(config)
        self.spearman_calculator = SpearmanCorrelationCalculator(config)
        
        # 优化历史
        self.optimization_history = []
        
    def optimize_stroke_order(self, strokes: List[Stroke], 
                             initial_order: List[int] = None,
                             method: str = None) -> OptimizationResult:
        """
        优化笔触排序
        
        Args:
            strokes: 笔触列表
            initial_order: 初始排序（可选）
            method: 优化方法（可选）
            
        Returns:
            优化结果
        """
        if not strokes:
            return OptimizationResult(
                optimized_order=[],
                optimized_strokes=[],
                best_energy=0.0,
                convergence_history=[],
                iterations_used=0,
                execution_time=0.0,
                success=True,
                optimization_method='none',
                metadata={}
            )
        
        start_time = time.time()
        method = method or self.opt_config.optimization_method
        
        try:
            if method == 'nes':
                result = self._optimize_with_nes(strokes, initial_order)
            elif method == 'genetic':
                result = self._optimize_with_genetic_algorithm(strokes, initial_order)
            elif method == 'simulated_annealing':
                result = self._optimize_with_simulated_annealing(strokes, initial_order)
            elif method == 'local_search':
                result = self._optimize_with_local_search(strokes, initial_order)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.optimization_method = method
            
            # 记录优化历史
            self.optimization_history.append({
                'timestamp': time.time(),
                'method': method,
                'num_strokes': len(strokes),
                'best_energy': result.best_energy,
                'iterations': result.iterations_used,
                'execution_time': execution_time,
                'success': result.success
            })
            
            self.logger.info(f"Optimization completed: method={method}, "
                           f"energy={result.best_energy:.4f}, time={execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in stroke order optimization: {e}")
            execution_time = time.time() - start_time
            
            # 返回原始顺序作为后备
            initial_order = initial_order or list(range(len(strokes)))
            return OptimizationResult(
                optimized_order=initial_order,
                optimized_strokes=strokes,
                best_energy=float('inf'),
                convergence_history=[],
                iterations_used=0,
                execution_time=execution_time,
                success=False,
                optimization_method=f'{method}_failed',
                metadata={'error': str(e)}
            )
    
    def multi_start_optimization(self, strokes: List[Stroke], 
                               num_starts: int = 5) -> OptimizationResult:
        """
        多起点优化
        
        Args:
            strokes: 笔触列表
            num_starts: 起点数量
            
        Returns:
            最佳优化结果
        """
        best_result = None
        best_energy = float('inf')
        
        for i in range(num_starts):
            # 生成随机初始顺序
            initial_order = list(range(len(strokes)))
            np.random.shuffle(initial_order)
            
            # 执行优化
            result = self.optimize_stroke_order(strokes, initial_order)
            
            # 更新最佳结果
            if result.success and result.best_energy < best_energy:
                best_energy = result.best_energy
                best_result = result
                best_result.metadata['multi_start_info'] = {
                    'start_index': i,
                    'total_starts': num_starts
                }
        
        if best_result is None:
            # 如果所有优化都失败，返回原始顺序
            return OptimizationResult(
                optimized_order=list(range(len(strokes))),
                optimized_strokes=strokes,
                best_energy=float('inf'),
                convergence_history=[],
                iterations_used=0,
                execution_time=0.0,
                success=False,
                optimization_method='multi_start_failed',
                metadata={'num_starts': num_starts}
            )
        
        return best_result
    
    def _optimize_with_nes(self, strokes: List[Stroke], 
                          initial_order: List[int] = None) -> OptimizationResult:
        """
        使用自然进化策略优化
        
        Args:
            strokes: 笔触列表
            initial_order: 初始排序
            
        Returns:
            优化结果
        """
        n = len(strokes)
        initial_order = initial_order or list(range(n))
        
        # 定义目标函数
        def objective_function(order_vector):
            try:
                # 将连续向量转换为排序
                order = self._vector_to_permutation(order_vector, n)
                # 确保order中的所有元素都是整数
                order = [int(i) for i in order]
                ordered_strokes = [strokes[i] for i in order]
            except Exception as e:
                self.logger.warning(f"Error in objective_function vector conversion: {str(e)}")
                return float('inf')
            
            # 计算能量
            # 准备笔触特征
            stroke_features = []
            for stroke in ordered_strokes:
                features = {
                    'id': getattr(stroke, 'id', 0),
                    'area': getattr(stroke, 'area', 0.0),
                    'length': getattr(stroke, 'length', 0.0),
                    'center': getattr(stroke, 'center', (0.0, 0.0)),
                    'color': getattr(stroke, 'color', (0, 0, 0)),
                    'width': getattr(stroke, 'width', 1.0)
                }
                stroke_features.append(features)
            
            # 使用整数顺序而不是order_vector
            order_indices = list(range(len(ordered_strokes)))
            energy_components = self.energy_function.calculate_energy(order_indices, stroke_features)
            energy = energy_components.total_energy
            
            # 添加约束惩罚
            if self.constraint_handler:
                try:
                    constraint_result = self.constraint_handler.check_constraints(order)
                    if isinstance(constraint_result, dict):
                        penalty = constraint_result.get('total_penalty', 0.0)
                    else:
                        # 如果constraint_result是字符串或其他类型，使用默认惩罚
                        penalty = 1000.0
                    energy += self.opt_config.constraint_penalty * penalty
                except Exception as e:
                    self.logger.warning(f"Error in constraint checking: {str(e)}")
                    energy += self.opt_config.constraint_penalty * 1000.0
            
            return energy
        
        # 执行NES优化
        # 准备笔触特征
        stroke_features = []
        for stroke in strokes:
            features = {
                'id': getattr(stroke, 'id', 0),
                'area': getattr(stroke, 'area', 0.0),
                'length': getattr(stroke, 'length', 0.0),
                'center': getattr(stroke, 'center', (0.0, 0.0))
            }
            stroke_features.append(features)
        
        nes_result = self.nes_optimizer.optimize(
            energy_function=objective_function,
            stroke_features=stroke_features,
            initial_order=initial_order
        )
        
        # 转换结果
        if nes_result.success:
            optimized_order = nes_result.best_order
            optimized_strokes = [strokes[i] for i in optimized_order]
        else:
            optimized_order = initial_order
            optimized_strokes = strokes
        
        return OptimizationResult(
            optimized_order=optimized_order,
            optimized_strokes=optimized_strokes,
            best_energy=nes_result.best_energy,
            convergence_history=nes_result.convergence_history,
            iterations_used=nes_result.iteration_count,
            execution_time=0.0,  # 将在外部设置
            success=nes_result.success,
            optimization_method='nes',
            metadata={
                'nes_result': nes_result,
                'population_size': self.opt_config.population_size
            }
        )
    
    def _optimize_with_genetic_algorithm(self, strokes: List[Stroke], 
                                       initial_order: List[int] = None) -> OptimizationResult:
        """
        使用遗传算法优化
        
        Args:
            strokes: 笔触列表
            initial_order: 初始排序
            
        Returns:
            优化结果
        """
        n = len(strokes)
        initial_order = initial_order or list(range(n))
        
        # 遗传算法参数
        population_size = self.opt_config.population_size
        mutation_rate = 0.1
        crossover_rate = 0.8
        elite_size = max(1, population_size // 10)
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = list(range(n))
            np.random.shuffle(individual)
            population.append(individual)
        
        # 确保初始顺序在种群中
        if initial_order not in population:
            population[0] = initial_order.copy()
        
        best_fitness_history = []
        best_individual = None
        best_fitness = float('inf')
        
        for iteration in range(self.opt_config.max_iterations):
            # 评估适应度
            fitness_scores = []
            for individual in population:
                ordered_strokes = [strokes[i] for i in individual]
                # 准备笔触特征
                stroke_features = []
                for stroke in ordered_strokes:
                    features = {
                        'id': getattr(stroke, 'id', 0),
                        'area': getattr(stroke, 'area', 0.0),
                        'length': getattr(stroke, 'length', 0.0),
                        'center': getattr(stroke, 'center', (0.0, 0.0)),
                        'color': getattr(stroke, 'color', (0, 0, 0)),
                        'width': getattr(stroke, 'width', 1.0)
                    }
                    stroke_features.append(features)
                
                order_indices = list(range(len(ordered_strokes)))
                energy_components = self.energy_function.calculate_energy(order_indices, stroke_features)
                energy = energy_components.total_energy
                
                # 添加约束惩罚
                if self.constraint_handler:
                    try:
                        constraint_result = self.constraint_handler.check_constraints(individual)
                        if isinstance(constraint_result, dict):
                            penalty = constraint_result.get('total_penalty', 0.0)
                        else:
                            # 如果constraint_result是字符串或其他类型，使用默认惩罚
                            penalty = 1000.0
                        energy += self.opt_config.constraint_penalty * penalty
                    except Exception as e:
                        self.logger.warning(f"Error in constraint checking: {str(e)}")
                        energy += self.opt_config.constraint_penalty * 1000.0
                
                fitness_scores.append(energy)
            
            # 更新最佳个体
            min_fitness_idx = int(np.argmin(fitness_scores))
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_individual = population[min_fitness_idx].copy()
            
            best_fitness_history.append(best_fitness)
            
            # 检查收敛
            if len(best_fitness_history) > self.opt_config.early_stopping_patience:
                recent_improvement = (best_fitness_history[-self.opt_config.early_stopping_patience] - 
                                    best_fitness_history[-1])
                if recent_improvement < self.opt_config.convergence_threshold:
                    break
            
            # 选择
            selected_indices = self._tournament_selection(fitness_scores, population_size)
            selected_population = [population[i] for i in selected_indices]
            
            # 交叉和变异
            new_population = []
            
            # 保留精英
            elite_indices = np.argsort(fitness_scores)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 生成新个体
            while len(new_population) < population_size:
                if np.random.random() < crossover_rate and len(selected_population) >= 2:
                    # 交叉
                    parent1, parent2 = np.random.choice(len(selected_population), 2, replace=False)
                    child = self._order_crossover(selected_population[parent1], selected_population[parent2])
                else:
                    # 复制
                    parent_idx = np.random.randint(len(selected_population))
                    child = selected_population[parent_idx].copy()
                
                # 变异
                if np.random.random() < mutation_rate:
                    child = self._swap_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        return OptimizationResult(
            optimized_order=best_individual or initial_order,
            optimized_strokes=[strokes[i] for i in (best_individual or initial_order)],
            best_energy=best_fitness,
            convergence_history=best_fitness_history,
            iterations_used=iteration + 1,
            execution_time=0.0,
            success=best_individual is not None,
            optimization_method='genetic',
            metadata={
                'population_size': population_size,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'elite_size': elite_size
            }
        )
    
    def _optimize_with_simulated_annealing(self, strokes: List[Stroke], 
                                         initial_order: List[int] = None) -> OptimizationResult:
        """
        使用模拟退火优化
        
        Args:
            strokes: 笔触列表
            initial_order: 初始排序
            
        Returns:
            优化结果
        """
        n = len(strokes)
        current_order = (initial_order or list(range(n))).copy()
        
        # 计算初始能量
        current_strokes = [strokes[i] for i in current_order]
        # 准备笔触特征
        stroke_features = []
        for stroke in current_strokes:
            features = {
                'id': getattr(stroke, 'id', 0),
                'area': getattr(stroke, 'area', 0.0),
                'length': getattr(stroke, 'length', 0.0),
                'center': getattr(stroke, 'center', (0.0, 0.0)),
                'color': getattr(stroke, 'color', (0, 0, 0)),
                'width': getattr(stroke, 'width', 1.0)
            }
            stroke_features.append(features)
        
        order_indices = list(range(len(current_strokes)))
        energy_components = self.energy_function.calculate_energy(order_indices, stroke_features)
        current_energy = energy_components.total_energy
        
        if self.constraint_handler:
            try:
                constraint_result = self.constraint_handler.check_constraints(current_order)
                if isinstance(constraint_result, dict):
                    penalty = constraint_result.get('total_penalty', 0.0)
                else:
                    # 如果constraint_result是字符串或其他类型，使用默认惩罚
                    penalty = 1000.0
                current_energy += self.opt_config.constraint_penalty * penalty
            except Exception as e:
                self.logger.warning(f"Error in constraint checking: {str(e)}")
                current_energy += self.opt_config.constraint_penalty * 1000.0
        
        best_order = current_order.copy()
        best_energy = current_energy
        
        # 模拟退火参数
        initial_temperature = 1000.0
        final_temperature = 0.01
        cooling_rate = 0.95
        
        temperature = initial_temperature
        energy_history = [current_energy]
        
        for iteration in range(self.opt_config.max_iterations):
            # 生成邻居解
            neighbor_order = self._generate_neighbor(current_order)
            neighbor_strokes = [strokes[i] for i in neighbor_order]
            # 准备邻居笔触特征
            neighbor_stroke_features = []
            for stroke in neighbor_strokes:
                features = {
                    'id': getattr(stroke, 'id', 0),
                    'area': getattr(stroke, 'area', 0.0),
                    'length': getattr(stroke, 'length', 0.0),
                    'center': getattr(stroke, 'center', (0.0, 0.0)),
                    'color': getattr(stroke, 'color', (0, 0, 0)),
                    'width': getattr(stroke, 'width', 1.0)
                }
                neighbor_stroke_features.append(features)
            
            neighbor_order_indices = list(range(len(neighbor_strokes)))
            neighbor_energy_components = self.energy_function.calculate_energy(neighbor_order_indices, neighbor_stroke_features)
            neighbor_energy = neighbor_energy_components.total_energy
            
            if self.constraint_handler:
                try:
                    constraint_result = self.constraint_handler.check_constraints(neighbor_order)
                    if isinstance(constraint_result, dict):
                        penalty = constraint_result.get('total_penalty', 0.0)
                    else:
                        # 如果constraint_result是字符串或其他类型，使用默认惩罚
                        penalty = 1000.0
                    neighbor_energy += self.opt_config.constraint_penalty * penalty
                except Exception as e:
                    self.logger.warning(f"Error in constraint checking: {str(e)}")
                    neighbor_energy += self.opt_config.constraint_penalty * 1000.0
            
            # 接受准则
            delta_energy = neighbor_energy - current_energy
            
            # 安全计算接受概率，避免numpy数组布尔运算错误
            accept_probability = np.exp(-delta_energy / temperature) if delta_energy > 0 else 1.0
            random_value = np.random.random()
            
            if delta_energy < 0 or random_value < accept_probability:
                current_order = neighbor_order
                current_energy = neighbor_energy
                
                # 更新最佳解
                if current_energy < best_energy:
                    best_order = current_order.copy()
                    best_energy = current_energy
            
            energy_history.append(best_energy)
            
            # 降温
            temperature *= cooling_rate
            
            # 检查终止条件
            if temperature < final_temperature:
                break
            
            # 检查收敛
            if len(energy_history) > self.opt_config.early_stopping_patience:
                recent_improvement = (energy_history[-self.opt_config.early_stopping_patience] - 
                                    energy_history[-1])
                if recent_improvement < self.opt_config.convergence_threshold:
                    break
        
        return OptimizationResult(
            optimized_order=best_order,
            optimized_strokes=[strokes[i] for i in best_order],
            best_energy=best_energy,
            convergence_history=energy_history,
            iterations_used=iteration + 1,
            execution_time=0.0,
            success=True,
            optimization_method='simulated_annealing',
            metadata={
                'initial_temperature': initial_temperature,
                'final_temperature': final_temperature,
                'cooling_rate': cooling_rate
            }
        )
    
    def _optimize_with_local_search(self, strokes: List[Stroke], 
                                  initial_order: List[int] = None) -> OptimizationResult:
        """
        使用局部搜索优化
        
        Args:
            strokes: 笔触列表
            initial_order: 初始排序
            
        Returns:
            优化结果
        """
        n = len(strokes)
        current_order = (initial_order or list(range(n))).copy()
        
        # 计算初始能量
        current_strokes = [strokes[i] for i in current_order]
        # 准备笔触特征
        stroke_features = []
        for stroke in current_strokes:
            features = {
                'id': getattr(stroke, 'id', 0),
                'area': getattr(stroke, 'area', 0.0),
                'length': getattr(stroke, 'length', 0.0),
                'center': getattr(stroke, 'center', (0.0, 0.0)),
                'color': getattr(stroke, 'color', (0, 0, 0)),
                'width': getattr(stroke, 'width', 1.0)
            }
            stroke_features.append(features)
        
        order_indices = list(range(len(current_strokes)))
        energy_components = self.energy_function.calculate_energy(order_indices, stroke_features)
        current_energy = energy_components.total_energy
        
        if self.constraint_handler:
            try:
                constraint_result = self.constraint_handler.check_constraints(current_order)
                if isinstance(constraint_result, dict):
                    penalty = constraint_result.get('total_penalty', 0.0)
                else:
                    # 如果constraint_result是字符串或其他类型，使用默认惩罚
                    penalty = 1000.0
                current_energy += self.opt_config.constraint_penalty * penalty
            except Exception as e:
                self.logger.warning(f"Error in constraint checking: {str(e)}")
                current_energy += self.opt_config.constraint_penalty * 1000.0
        
        best_order = current_order.copy()
        best_energy = current_energy
        energy_history = [current_energy]
        
        improved = True
        iteration = 0
        
        while improved and iteration < self.opt_config.max_iterations:
            improved = False
            
            # 尝试所有可能的2-opt交换
            for i in range(n):
                for j in range(i + 1, n):
                    # 创建邻居解
                    neighbor_order = current_order.copy()
                    neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
                    
                    # 计算邻居能量
                    neighbor_strokes = [strokes[k] for k in neighbor_order]
                    # 准备邻居笔触特征
                    neighbor_stroke_features = []
                    for stroke in neighbor_strokes:
                        features = {
                            'id': getattr(stroke, 'id', 0),
                            'area': getattr(stroke, 'area', 0.0),
                            'length': getattr(stroke, 'length', 0.0),
                            'center': getattr(stroke, 'center', (0.0, 0.0)),
                            'color': getattr(stroke, 'color', (0, 0, 0)),
                            'width': getattr(stroke, 'width', 1.0)
                        }
                        neighbor_stroke_features.append(features)
                    
                    neighbor_order_indices = list(range(len(neighbor_strokes)))
                    neighbor_energy_components = self.energy_function.calculate_energy(neighbor_order_indices, neighbor_stroke_features)
                    neighbor_energy = neighbor_energy_components.total_energy
                    
                    if self.constraint_handler:
                        try:
                            constraint_result = self.constraint_handler.check_constraints(neighbor_order)
                            if isinstance(constraint_result, dict):
                                penalty = constraint_result.get('total_penalty', 0.0)
                            else:
                                # 如果constraint_result是字符串或其他类型，使用默认惩罚
                                penalty = 1000.0
                            neighbor_energy += self.opt_config.constraint_penalty * penalty
                        except Exception as e:
                            self.logger.warning(f"Error in constraint checking: {str(e)}")
                            neighbor_energy += self.opt_config.constraint_penalty * 1000.0
                    
                    # 如果找到更好的解
                    if neighbor_energy < current_energy:
                        current_order = neighbor_order
                        current_energy = neighbor_energy
                        improved = True
                        
                        # 更新最佳解
                        if current_energy < best_energy:
                            best_order = current_order.copy()
                            best_energy = current_energy
                        
                        break
                
                if improved:
                    break
            
            energy_history.append(best_energy)
            iteration += 1
        
        return OptimizationResult(
            optimized_order=best_order,
            optimized_strokes=[strokes[i] for i in best_order],
            best_energy=best_energy,
            convergence_history=energy_history,
            iterations_used=iteration,
            execution_time=0.0,
            success=True,
            optimization_method='local_search',
            metadata={
                'search_type': '2-opt',
                'total_swaps_tried': iteration * n * (n - 1) // 2
            }
        )
    
    def _vector_to_permutation(self, vector: np.ndarray, n: int) -> List[int]:
        """
        将连续向量转换为排列
        
        Args:
            vector: 连续向量
            n: 排列长度
            
        Returns:
            排列
        """
        # 使用排序索引方法
        sorted_indices = np.argsort(vector)
        # 确保返回的是整数列表
        return [int(idx) for idx in sorted_indices]
    
    def _tournament_selection(self, fitness_scores: List[float], 
                            population_size: int, tournament_size: int = 3) -> List[int]:
        """
        锦标赛选择
        
        Args:
            fitness_scores: 适应度分数
            population_size: 种群大小
            tournament_size: 锦标赛大小
            
        Returns:
            选中的个体索引
        """
        selected_indices = []
        
        for _ in range(population_size):
            # 随机选择锦标赛参与者
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择最佳个体（最小适应度）
            winner_idx = tournament_indices[int(np.argmin(tournament_fitness))]
            selected_indices.append(int(winner_idx))
        
        return selected_indices
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        顺序交叉（OX）
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            子代
        """
        n = len(parent1)
        
        # 选择交叉点
        start = np.random.randint(0, n)
        end = np.random.randint(start + 1, n + 1)
        
        # 初始化子代
        child = [-1] * n
        
        # 复制父代1的片段
        child[start:end] = parent1[start:end]
        
        # 从父代2填充剩余位置
        remaining_positions = [i for i in range(n) if child[i] == -1]
        remaining_values = [val for val in parent2 if val not in child]
        
        for pos, val in zip(remaining_positions, remaining_values):
            child[pos] = val
        
        return child
    
    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """
        交换变异
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        mutated = individual.copy()
        n = len(mutated)
        
        if n > 1:
            i, j = np.random.choice(n, 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def _generate_neighbor(self, order: List[int]) -> List[int]:
        """
        生成邻居解
        
        Args:
            order: 当前排序
            
        Returns:
            邻居排序
        """
        neighbor = order.copy()
        n = len(neighbor)
        
        if n > 1:
            # 随机选择操作类型
            operation = np.random.choice(['swap', 'insert', 'reverse'])
            
            if operation == 'swap':
                # 交换两个元素
                i, j = np.random.choice(n, 2, replace=False)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
            elif operation == 'insert':
                # 插入操作
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    element = neighbor.pop(i)
                    neighbor.insert(j, element)
            
            elif operation == 'reverse':
                # 反转片段
                i = np.random.randint(n)
                j = np.random.randint(i, n)
                neighbor[i:j+1] = neighbor[i:j+1][::-1]
        
        return neighbor
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        获取优化统计信息
        
        Returns:
            统计信息
        """
        if not self.optimization_history:
            return {}
        
        history = self.optimization_history
        
        stats = {
            'total_optimizations': len(history),
            'methods_used': list(set(h['method'] for h in history)),
            'success_rate': sum(1 for h in history if h['success']) / len(history),
            'average_execution_time': np.mean([h['execution_time'] for h in history]),
            'average_iterations': np.mean([h['iterations'] for h in history]),
            'best_energy_achieved': min(h['best_energy'] for h in history if h['success']),
            'average_energy': np.mean([h['best_energy'] for h in history if h['success']])
        }
        
        # 按方法统计
        method_stats = {}
        for method in stats['methods_used']:
            method_history = [h for h in history if h['method'] == method]
            method_stats[method] = {
                'count': len(method_history),
                'success_rate': sum(1 for h in method_history if h['success']) / len(method_history),
                'average_time': np.mean([h['execution_time'] for h in method_history]),
                'average_energy': np.mean([h['best_energy'] for h in method_history if h['success']])
            }
        
        stats['method_statistics'] = method_stats
        
        return stats
    
    def clear_history(self):
        """清除优化历史"""
        self.optimization_history.clear()
        self.logger.info("Optimization history cleared")