# -*- coding: utf-8 -*-
"""
自然进化策略(NES)优化器

实现论文中的自然进化策略优化：
1. 实数空间编码
2. 多正态分布采样
3. 适应性参数调整
4. 收敛性检测
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from scipy.stats import rankdata
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class OptimizationResult:
    """
    优化结果数据结构
    """
    best_order: List[int]
    best_energy: float
    convergence_history: List[float]
    iteration_count: int
    execution_time: float
    success: bool
    termination_reason: str
    
    # 详细统计
    energy_evaluations: int = 0
    population_diversity: List[float] = None
    parameter_evolution: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.population_diversity is None:
            self.population_diversity = []
        if self.parameter_evolution is None:
            self.parameter_evolution = {}


class NESOptimizer:
    """
    自然进化策略优化器
    
    使用NES算法优化笔触绘制顺序
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化NES优化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 基本参数
        self.population_size = config.get('population_size', 50)
        self.max_iterations = config.get('max_iterations', 1000)
        self.convergence_threshold = config.get('convergence_threshold', 1e-6)
        self.stagnation_limit = config.get('stagnation_limit', 50)
        
        # NES参数
        self.learning_rate = config.get('learning_rate', 0.1)
        self.sigma_init = config.get('sigma_init', 1.0)
        self.sigma_min = config.get('sigma_min', 0.01)
        self.sigma_decay = config.get('sigma_decay', 0.999)
        
        # 选择参数
        self.elite_ratio = config.get('elite_ratio', 0.2)
        self.selection_pressure = config.get('selection_pressure', 2.0)
        
        # 并行计算
        self.use_parallel = config.get('use_parallel', True)
        self.max_workers = config.get('max_workers', 4)
        
        # 自适应参数
        self.adaptive_sigma = config.get('adaptive_sigma', True)
        self.adaptive_learning_rate = config.get('adaptive_learning_rate', True)
        
        # 状态变量
        self.current_mean = None
        self.current_sigma = self.sigma_init
        self.current_lr = self.learning_rate
        self.best_fitness_history = []
        self.diversity_history = []
        
    def optimize(self, energy_function: Callable, 
                stroke_features: List[Dict[str, Any]],
                initial_order: Optional[List[int]] = None,
                constraints: Optional[List[Any]] = None) -> OptimizationResult:
        """
        执行NES优化
        
        Args:
            energy_function: 能量函数
            stroke_features: 笔触特征列表
            initial_order: 初始顺序（可选）
            constraints: 约束条件（可选）
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()
        
        try:
            n_strokes = len(stroke_features)
            
            # 初始化
            self._initialize_optimization(n_strokes, initial_order)
            
            # 优化循环
            best_order, best_energy, history = self._optimization_loop(
                energy_function, stroke_features, constraints
            )
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                best_order=best_order,
                best_energy=best_energy,
                convergence_history=history,
                iteration_count=len(history),
                execution_time=execution_time,
                success=True,
                termination_reason="Optimization completed",
                energy_evaluations=len(history) * self.population_size,
                population_diversity=self.diversity_history.copy(),
                parameter_evolution={
                    'sigma': [self.sigma_init * (self.sigma_decay ** i) for i in range(len(history))],
                    'learning_rate': [self.learning_rate] * len(history)  # 简化版本
                }
            )
            
            self.logger.info(
                f"NES optimization completed: {len(history)} iterations, "
                f"best energy: {best_energy:.6f}, time: {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"NES optimization failed: {str(e)}")
            
            return OptimizationResult(
                best_order=list(range(len(stroke_features))),
                best_energy=float('inf'),
                convergence_history=[],
                iteration_count=0,
                execution_time=execution_time,
                success=False,
                termination_reason=f"Error: {str(e)}"
            )
    
    def _initialize_optimization(self, n_strokes: int, initial_order: Optional[List[int]]):
        """
        初始化优化过程
        
        Args:
            n_strokes: 笔触数量
            initial_order: 初始顺序
        """
        # 初始化均值（使用实数编码）
        if initial_order is not None:
            self.current_mean = np.array(initial_order, dtype=float)
        else:
            self.current_mean = np.arange(n_strokes, dtype=float)
        
        # 重置参数
        self.current_sigma = self.sigma_init
        self.current_lr = self.learning_rate
        self.best_fitness_history.clear()
        self.diversity_history.clear()
    
    def _optimization_loop(self, energy_function: Callable,
                          stroke_features: List[Dict[str, Any]],
                          constraints: Optional[List[Any]]) -> Tuple[List[int], float, List[float]]:
        """
        主优化循环
        
        Args:
            energy_function: 能量函数
            stroke_features: 笔触特征列表
            constraints: 约束条件
            
        Returns:
            Tuple: (最佳顺序, 最佳能量, 收敛历史)
        """
        best_order = None
        best_energy = float('inf')
        convergence_history = []
        stagnation_count = 0
        
        for iteration in range(self.max_iterations):
            # 生成候选解
            population = self._generate_population()
            
            # 评估适应度
            fitness_values = self._evaluate_population(
                population, energy_function, stroke_features, constraints
            )
            
            # 更新最佳解
            current_best_idx = int(np.argmin(fitness_values))
            current_best_energy = fitness_values[current_best_idx]
            
            if current_best_energy < best_energy:
                best_energy = current_best_energy
                best_order = self._decode_solution(population[current_best_idx])
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            convergence_history.append(best_energy)
            
            # 计算种群多样性
            diversity = self._calculate_population_diversity(population)
            self.diversity_history.append(diversity)
            
            # 更新分布参数
            self._update_distribution(population, fitness_values)
            
            # 检查收敛条件
            if self._check_convergence(convergence_history, stagnation_count):
                self.logger.info(f"Convergence achieved at iteration {iteration}")
                break
            
            # 自适应参数调整
            if self.adaptive_sigma:
                self._adapt_sigma(diversity, iteration)
            
            if self.adaptive_learning_rate:
                self._adapt_learning_rate(convergence_history, iteration)
        
        return best_order, best_energy, convergence_history
    
    def _generate_population(self) -> np.ndarray:
        """
        生成候选解种群
        
        Returns:
            np.ndarray: 种群矩阵
        """
        n_dims = len(self.current_mean)
        population = np.zeros((self.population_size, n_dims))
        
        for i in range(self.population_size):
            # 从多元正态分布采样
            noise = np.random.normal(0, self.current_sigma, n_dims)
            population[i] = self.current_mean + noise
        
        return population
    
    def _evaluate_population(self, population: np.ndarray,
                           energy_function: Callable,
                           stroke_features: List[Dict[str, Any]],
                           constraints: Optional[List[Any]]) -> np.ndarray:
        """
        评估种群适应度
        
        Args:
            population: 种群
            energy_function: 能量函数
            stroke_features: 笔触特征
            constraints: 约束条件
            
        Returns:
            np.ndarray: 适应度值
        """
        fitness_values = np.zeros(self.population_size)
        
        if self.use_parallel and self.max_workers > 1:
            # 并行评估
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for i, individual in enumerate(population):
                    future = executor.submit(
                        self._evaluate_individual,
                        individual, energy_function, stroke_features, constraints
                    )
                    futures.append((i, future))
                
                for i, future in futures:
                    fitness_values[i] = future.result()
        else:
            # 串行评估
            for i, individual in enumerate(population):
                fitness_values[i] = self._evaluate_individual(
                    individual, energy_function, stroke_features, constraints
                )
        
        return fitness_values
    
    def _evaluate_individual(self, individual: np.ndarray, 
                            energy_function: Callable,
                            stroke_features: List[Dict[str, Any]],
                            constraints: Optional[List[Any]] = None) -> float:
        """
        评估单个个体的适应度
        
        Args:
            individual: 个体编码
            energy_function: 能量函数
            stroke_features: 笔触特征
            constraints: 约束条件
            
        Returns:
            float: 适应度值
        """
        try:
            # 解码为整数顺序
            order = self._decode_solution(individual)
            
            # 应用约束
            if constraints:
                order = self._apply_constraints(order, constraints)
            
            # 计算能量
            if callable(energy_function):
                if hasattr(energy_function, 'calculate_energy'):
                    # 如果是对象，调用calculate_energy方法
                    energy_components = energy_function.calculate_energy(order, stroke_features)
                    return energy_components.total_energy
                else:
                    # 如果是函数，传递解码后的整数顺序而不是原始individual
                    energy = energy_function(order)  # 传递解码后的整数顺序
                    return energy
            else:
                # 如果不是可调用对象，返回无穷大
                return float('inf')
            
        except Exception as e:
            self.logger.warning(f"Error evaluating individual: {str(e)}")
            return float('inf')
    
    def _decode_solution(self, encoded_solution: np.ndarray) -> List[int]:
        """
        将实数编码解码为整数顺序
        
        Args:
            encoded_solution: 实数编码解
            
        Returns:
            List[int]: 整数顺序
        """
        # 使用排序索引将实数转换为排列
        sorted_indices = np.argsort(encoded_solution)
        # 确保返回的是整数列表
        return [int(idx) for idx in sorted_indices]
    
    def _apply_constraints(self, order: List[int], constraints: List[Any]) -> List[int]:
        """
        应用约束条件
        
        Args:
            order: 原始顺序
            constraints: 约束条件列表
            
        Returns:
            List[int]: 应用约束后的顺序
        """
        # 这里可以实现各种约束处理逻辑
        # 例如：圆形物体的成对笔触必须连续
        constrained_order = order.copy()
        
        for constraint in constraints:
            if hasattr(constraint, 'apply'):
                constrained_order = constraint.apply(constrained_order)
        
        return constrained_order
    
    def _update_distribution(self, population: np.ndarray, fitness_values: np.ndarray):
        """
        更新分布参数
        
        Args:
            population: 当前种群
            fitness_values: 适应度值
        """
        # 选择精英个体
        elite_size = max(1, int(self.population_size * self.elite_ratio))
        elite_indices = np.argsort(fitness_values)[:elite_size]
        elite_population = population[elite_indices]
        
        # 计算权重（基于排名）
        ranks = rankdata(fitness_values)
        weights = np.exp(-self.selection_pressure * (ranks - 1) / len(ranks))
        weights = weights / np.sum(weights)
        
        # 更新均值
        new_mean = np.average(population, axis=0, weights=weights)
        
        # 使用学习率进行平滑更新
        self.current_mean = (1 - self.current_lr) * self.current_mean + self.current_lr * new_mean
    
    def _calculate_population_diversity(self, population: np.ndarray) -> float:
        """
        计算种群多样性
        
        Args:
            population: 种群
            
        Returns:
            float: 多样性指标
        """
        if len(population) < 2:
            return 0.0
        
        # 计算种群中个体间的平均距离
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _check_convergence(self, history: List[float], stagnation_count: int) -> bool:
        """
        检查收敛条件
        
        Args:
            history: 收敛历史
            stagnation_count: 停滞计数
            
        Returns:
            bool: 是否收敛
        """
        # 检查停滞
        if stagnation_count >= self.stagnation_limit:
            return True
        
        # 检查改进幅度
        if len(history) >= 10:
            recent_improvement = abs(history[-10] - history[-1])
            if recent_improvement < self.convergence_threshold:
                return True
        
        # 检查标准差
        if len(history) >= 20:
            recent_std = np.std(history[-20:])
            if recent_std < self.convergence_threshold:
                return True
        
        return False
    
    def _adapt_sigma(self, diversity: float, iteration: int):
        """
        自适应调整sigma参数
        
        Args:
            diversity: 当前多样性
            iteration: 当前迭代次数
        """
        # 基于多样性调整sigma
        if len(self.diversity_history) > 5:
            diversity_trend = np.mean(self.diversity_history[-5:]) - np.mean(self.diversity_history[-10:-5]) if len(self.diversity_history) >= 10 else 0
            
            if diversity_trend < 0:  # 多样性下降，增加sigma
                self.current_sigma = min(self.current_sigma * 1.05, self.sigma_init)
            else:  # 多样性稳定或增加，减少sigma
                self.current_sigma = max(self.current_sigma * self.sigma_decay, self.sigma_min)
    
    def _adapt_learning_rate(self, history: List[float], iteration: int):
        """
        自适应调整学习率
        
        Args:
            history: 收敛历史
            iteration: 当前迭代次数
        """
        if len(history) > 10:
            # 基于改进速度调整学习率
            recent_improvement = history[-10] - history[-1]
            
            if recent_improvement > 0:  # 有改进，保持或略微增加学习率
                self.current_lr = min(self.current_lr * 1.01, self.learning_rate * 2)
            else:  # 无改进，减少学习率
                self.current_lr = max(self.current_lr * 0.99, self.learning_rate * 0.1)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        获取优化统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'population_size': self.population_size,
            'max_iterations': self.max_iterations,
            'current_sigma': self.current_sigma,
            'current_learning_rate': self.current_lr,
            'convergence_threshold': self.convergence_threshold,
            'stagnation_limit': self.stagnation_limit,
            'elite_ratio': self.elite_ratio,
            'selection_pressure': self.selection_pressure,
            'best_fitness_count': len(self.best_fitness_history),
            'diversity_trend': np.mean(self.diversity_history[-10:]) if len(self.diversity_history) >= 10 else 0.0
        }
    
    def reset_optimizer(self):
        """
        重置优化器状态
        """
        self.current_mean = None
        self.current_sigma = self.sigma_init
        self.current_lr = self.learning_rate
        self.best_fitness_history.clear()
        self.diversity_history.clear()
        
        self.logger.info("NES optimizer reset")
    
    def save_state(self) -> Dict[str, Any]:
        """
        保存优化器状态
        
        Returns:
            Dict: 状态字典
        """
        return {
            'current_mean': self.current_mean.tolist() if self.current_mean is not None else None,
            'current_sigma': self.current_sigma,
            'current_lr': self.current_lr,
            'best_fitness_history': self.best_fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
            'config': self.config.copy()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """
        加载优化器状态
        
        Args:
            state: 状态字典
        """
        if state.get('current_mean') is not None:
            self.current_mean = np.array(state['current_mean'])
        
        self.current_sigma = state.get('current_sigma', self.sigma_init)
        self.current_lr = state.get('current_lr', self.learning_rate)
        self.best_fitness_history = state.get('best_fitness_history', [])
        self.diversity_history = state.get('diversity_history', [])
        
        # 更新配置
        if 'config' in state:
            self.config.update(state['config'])
        
        self.logger.info("NES optimizer state loaded")