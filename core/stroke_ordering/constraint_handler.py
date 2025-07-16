# -*- coding: utf-8 -*-
"""
约束处理器

实现论文中的硬约束处理：
1. 圆形物体成对笔触约束
2. 阶段顺序约束
3. 空间邻近约束
4. 自定义约束支持
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class ConstraintType(Enum):
    """
    约束类型枚举
    """
    PAIR_CONSECUTIVE = "pair_consecutive"  # 成对连续约束
    STAGE_ORDER = "stage_order"  # 阶段顺序约束
    SPATIAL_PROXIMITY = "spatial_proximity"  # 空间邻近约束
    TEMPORAL_SEQUENCE = "temporal_sequence"  # 时间序列约束
    CUSTOM = "custom"  # 自定义约束


@dataclass
class StrokeConstraint:
    """
    笔触约束数据结构
    """
    constraint_type: ConstraintType
    stroke_ids: List[int]
    priority: int = 1  # 约束优先级（1-10，10最高）
    is_hard: bool = True  # 是否为硬约束
    penalty_weight: float = 1.0  # 软约束惩罚权重
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseConstraint(ABC):
    """
    约束基类
    """
    
    def __init__(self, constraint: StrokeConstraint):
        self.constraint = constraint
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def is_satisfied(self, order: List[int]) -> bool:
        """
        检查约束是否满足
        
        Args:
            order: 笔触顺序
            
        Returns:
            bool: 是否满足约束
        """
        pass
    
    @abstractmethod
    def apply(self, order: List[int]) -> List[int]:
        """
        应用约束修正顺序
        
        Args:
            order: 原始顺序
            
        Returns:
            List[int]: 修正后的顺序
        """
        pass
    
    @abstractmethod
    def calculate_violation_penalty(self, order: List[int]) -> float:
        """
        计算违反约束的惩罚
        
        Args:
            order: 笔触顺序
            
        Returns:
            float: 惩罚值
        """
        pass


class PairConsecutiveConstraint(BaseConstraint):
    """
    成对连续约束
    
    确保指定的笔触对在绘制顺序中连续出现
    """
    
    def is_satisfied(self, order: List[int]) -> bool:
        """
        检查成对笔触是否连续
        
        Args:
            order: 笔触顺序
            
        Returns:
            bool: 是否满足约束
        """
        stroke_ids = self.constraint.stroke_ids
        if len(stroke_ids) != 2:
            return True  # 无效约束，视为满足
        
        try:
            pos1 = order.index(stroke_ids[0])
            pos2 = order.index(stroke_ids[1])
            return abs(pos1 - pos2) == 1
        except ValueError:
            return False  # 笔触不在顺序中
    
    def apply(self, order: List[int]) -> List[int]:
        """
        应用成对连续约束
        
        Args:
            order: 原始顺序
            
        Returns:
            List[int]: 修正后的顺序
        """
        stroke_ids = self.constraint.stroke_ids
        if len(stroke_ids) != 2:
            return order
        
        modified_order = order.copy()
        
        try:
            pos1 = modified_order.index(stroke_ids[0])
            pos2 = modified_order.index(stroke_ids[1])
            
            if abs(pos1 - pos2) != 1:
                # 移动第二个笔触到第一个笔触旁边
                modified_order.remove(stroke_ids[1])
                insert_pos = modified_order.index(stroke_ids[0]) + 1
                modified_order.insert(insert_pos, stroke_ids[1])
            
        except ValueError:
            pass  # 笔触不在顺序中，无法修正
        
        return modified_order
    
    def calculate_violation_penalty(self, order: List[int]) -> float:
        """
        计算违反成对连续约束的惩罚
        
        Args:
            order: 笔触顺序
            
        Returns:
            float: 惩罚值
        """
        if self.is_satisfied(order):
            return 0.0
        
        stroke_ids = self.constraint.stroke_ids
        if len(stroke_ids) != 2:
            return 0.0
        
        try:
            pos1 = order.index(stroke_ids[0])
            pos2 = order.index(stroke_ids[1])
            distance = abs(pos1 - pos2) - 1
            return distance * self.constraint.penalty_weight
        except ValueError:
            return 10.0 * self.constraint.penalty_weight  # 高惩罚


class StageOrderConstraint(BaseConstraint):
    """
    阶段顺序约束
    
    确保不同阶段的笔触按照指定顺序绘制
    """
    
    def is_satisfied(self, order: List[int]) -> bool:
        """
        检查阶段顺序是否正确
        
        Args:
            order: 笔触顺序
            
        Returns:
            bool: 是否满足约束
        """
        stage_mapping = self.constraint.metadata.get('stage_mapping', {})
        if not stage_mapping:
            return True
        
        # 检查每个阶段内的笔触是否按照正确顺序
        last_stage = -1
        for stroke_id in order:
            current_stage = stage_mapping.get(stroke_id, 0)
            if current_stage < last_stage:
                return False
            last_stage = current_stage
        
        return True
    
    def apply(self, order: List[int]) -> List[int]:
        """
        应用阶段顺序约束
        
        Args:
            order: 原始顺序
            
        Returns:
            List[int]: 修正后的顺序
        """
        stage_mapping = self.constraint.metadata.get('stage_mapping', {})
        if not stage_mapping:
            return order
        
        # 按阶段分组
        stage_groups = {}
        for stroke_id in order:
            stage = stage_mapping.get(stroke_id, 0)
            if stage not in stage_groups:
                stage_groups[stage] = []
            stage_groups[stage].append(stroke_id)
        
        # 按阶段顺序重新排列
        modified_order = []
        for stage in sorted(stage_groups.keys()):
            modified_order.extend(stage_groups[stage])
        
        return modified_order
    
    def calculate_violation_penalty(self, order: List[int]) -> float:
        """
        计算违反阶段顺序约束的惩罚
        
        Args:
            order: 笔触顺序
            
        Returns:
            float: 惩罚值
        """
        if self.is_satisfied(order):
            return 0.0
        
        stage_mapping = self.constraint.metadata.get('stage_mapping', {})
        if not stage_mapping:
            return 0.0
        
        penalty = 0.0
        last_stage = -1
        
        for stroke_id in order:
            current_stage = stage_mapping.get(stroke_id, 0)
            if current_stage < last_stage:
                penalty += (last_stage - current_stage) * self.constraint.penalty_weight
            last_stage = max(last_stage, current_stage)
        
        return penalty


class SpatialProximityConstraint(BaseConstraint):
    """
    空间邻近约束
    
    鼓励空间上邻近的笔触连续绘制
    """
    
    def is_satisfied(self, order: List[int]) -> bool:
        """
        检查空间邻近约束是否满足
        
        Args:
            order: 笔触顺序
            
        Returns:
            bool: 是否满足约束
        """
        # 空间邻近约束通常是软约束，总是返回True
        return True
    
    def apply(self, order: List[int]) -> List[int]:
        """
        应用空间邻近约束
        
        Args:
            order: 原始顺序
            
        Returns:
            List[int]: 修正后的顺序
        """
        stroke_positions = self.constraint.metadata.get('stroke_positions', {})
        max_distance = self.constraint.metadata.get('max_distance', 100.0)
        
        if not stroke_positions:
            return order
        
        # 使用贪心算法重新排列以最小化空间跳跃
        modified_order = []
        remaining_strokes = set(order)
        
        # 从第一个笔触开始
        if order:
            current_stroke = int(np.round(order[0]))
            modified_order.append(current_stroke)
            remaining_strokes.remove(current_stroke)
            
            while remaining_strokes:
                current_pos = stroke_positions.get(current_stroke, [0, 0])
                
                # 找到最近的未处理笔触
                min_distance = float('inf')
                next_stroke = None
                
                for stroke_id in remaining_strokes:
                    stroke_pos = stroke_positions.get(stroke_id, [0, 0])
                    distance = np.linalg.norm(np.array(current_pos) - np.array(stroke_pos))
                    
                    if distance < min_distance:
                        min_distance = distance
                        next_stroke = stroke_id
                
                if next_stroke is not None:
                    modified_order.append(next_stroke)
                    remaining_strokes.remove(next_stroke)
                    current_stroke = next_stroke
                else:
                    # 如果没有找到，添加剩余的第一个
                    next_stroke = next(iter(remaining_strokes))
                    modified_order.append(next_stroke)
                    remaining_strokes.remove(next_stroke)
                    current_stroke = next_stroke
        
        return modified_order
    
    def calculate_violation_penalty(self, order: List[int]) -> float:
        """
        计算空间跳跃惩罚
        
        Args:
            order: 笔触顺序
            
        Returns:
            float: 惩罚值
        """
        stroke_positions = self.constraint.metadata.get('stroke_positions', {})
        if not stroke_positions or len(order) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(order) - 1):
            pos1 = stroke_positions.get(int(np.round(order[i])), [0, 0])
            pos2 = stroke_positions.get(int(np.round(order[i + 1])), [0, 0])
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            total_distance += distance
        
        # 归一化惩罚
        avg_distance = total_distance / (len(order) - 1)
        return avg_distance * self.constraint.penalty_weight / 1000.0  # 缩放因子


class ConstraintHandler:
    """
    约束处理器
    
    管理和应用各种笔触排序约束
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化约束处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 约束处理参数
        self.max_iterations = config.get('max_constraint_iterations', 10)
        self.convergence_threshold = config.get('constraint_convergence_threshold', 1e-3)
        self.soft_constraint_weight = config.get('soft_constraint_weight', 0.1)
        
        # 约束列表
        self.constraints = []
        self.constraint_objects = []
        
        # 约束类型映射
        self.constraint_classes = {
            ConstraintType.PAIR_CONSECUTIVE: PairConsecutiveConstraint,
            ConstraintType.STAGE_ORDER: StageOrderConstraint,
            ConstraintType.SPATIAL_PROXIMITY: SpatialProximityConstraint,
        }
    
    def add_constraint(self, constraint: StrokeConstraint):
        """
        添加约束
        
        Args:
            constraint: 笔触约束
        """
        self.constraints.append(constraint)
        
        # 创建约束对象
        constraint_class = self.constraint_classes.get(constraint.constraint_type)
        if constraint_class:
            constraint_obj = constraint_class(constraint)
            self.constraint_objects.append(constraint_obj)
            self.logger.info(f"Added constraint: {constraint.constraint_type.value}")
        else:
            self.logger.warning(f"Unknown constraint type: {constraint.constraint_type}")
    
    def remove_constraint(self, constraint_index: int):
        """
        移除约束
        
        Args:
            constraint_index: 约束索引
        """
        if 0 <= constraint_index < len(self.constraints):
            removed_constraint = self.constraints.pop(constraint_index)
            self.constraint_objects.pop(constraint_index)
            self.logger.info(f"Removed constraint: {removed_constraint.constraint_type.value}")
    
    def apply_constraints(self, order: List[int]) -> List[int]:
        """
        应用所有约束
        
        Args:
            order: 原始顺序
            
        Returns:
            List[int]: 应用约束后的顺序
        """
        if not self.constraint_objects:
            return order
        
        modified_order = order.copy()
        
        # 按优先级排序约束
        sorted_constraints = sorted(
            self.constraint_objects,
            key=lambda c: c.constraint.priority,
            reverse=True
        )
        
        # 迭代应用硬约束
        for iteration in range(self.max_iterations):
            previous_order = modified_order.copy()
            
            for constraint_obj in sorted_constraints:
                if constraint_obj.constraint.is_hard:
                    modified_order = constraint_obj.apply(modified_order)
            
            # 检查收敛 - 安全比较列表
            if len(modified_order) == len(previous_order) and all(a == b for a, b in zip(modified_order, previous_order)):
                break
        
        return modified_order
    
    def check_constraints(self, order: List[int]) -> Dict[str, Any]:
        """
        检查约束满足情况
        
        Args:
            order: 笔触顺序
            
        Returns:
            Dict: 约束检查结果
        """
        results = {
            'all_satisfied': True,
            'hard_constraints_satisfied': True,
            'soft_constraints_satisfied': True,
            'constraint_details': [],
            'total_penalty': 0.0
        }
        
        for i, constraint_obj in enumerate(self.constraint_objects):
            is_satisfied = constraint_obj.is_satisfied(order)
            penalty = constraint_obj.calculate_violation_penalty(order)
            
            constraint_result = {
                'index': i,
                'type': constraint_obj.constraint.constraint_type.value,
                'is_hard': constraint_obj.constraint.is_hard,
                'is_satisfied': is_satisfied,
                'penalty': penalty,
                'description': constraint_obj.constraint.description
            }
            
            results['constraint_details'].append(constraint_result)
            results['total_penalty'] += penalty
            
            if not is_satisfied:
                results['all_satisfied'] = False
                if constraint_obj.constraint.is_hard:
                    results['hard_constraints_satisfied'] = False
                else:
                    results['soft_constraints_satisfied'] = False
        
        return results
    
    def calculate_constraint_penalty(self, order: List[int]) -> float:
        """
        计算总约束惩罚
        
        Args:
            order: 笔触顺序
            
        Returns:
            float: 总惩罚值
        """
        total_penalty = 0.0
        
        for constraint_obj in self.constraint_objects:
            penalty = constraint_obj.calculate_violation_penalty(order)
            
            if constraint_obj.constraint.is_hard:
                total_penalty += penalty * 10.0  # 硬约束高权重
            else:
                total_penalty += penalty * self.soft_constraint_weight
        
        return total_penalty
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """
        获取约束统计信息
        
        Returns:
            Dict: 统计信息
        """
        hard_count = sum(1 for c in self.constraints if c.is_hard)
        soft_count = len(self.constraints) - hard_count
        
        constraint_types = {}
        for constraint in self.constraints:
            constraint_type = constraint.constraint_type.value
            constraint_types[constraint_type] = constraint_types.get(constraint_type, 0) + 1
        
        return {
            'total_constraints': len(self.constraints),
            'hard_constraints': hard_count,
            'soft_constraints': soft_count,
            'constraint_types': constraint_types,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold
        }
    
    def create_pair_constraint(self, stroke_id1: int, stroke_id2: int, 
                             priority: int = 5, description: str = "") -> StrokeConstraint:
        """
        创建成对连续约束
        
        Args:
            stroke_id1: 第一个笔触ID
            stroke_id2: 第二个笔触ID
            priority: 优先级
            description: 描述
            
        Returns:
            StrokeConstraint: 约束对象
        """
        return StrokeConstraint(
            constraint_type=ConstraintType.PAIR_CONSECUTIVE,
            stroke_ids=[stroke_id1, stroke_id2],
            priority=priority,
            is_hard=True,
            description=description or f"Strokes {stroke_id1} and {stroke_id2} must be consecutive"
        )
    
    def create_stage_constraint(self, stage_mapping: Dict[int, int], 
                              priority: int = 3, description: str = "") -> StrokeConstraint:
        """
        创建阶段顺序约束
        
        Args:
            stage_mapping: 笔触到阶段的映射
            priority: 优先级
            description: 描述
            
        Returns:
            StrokeConstraint: 约束对象
        """
        return StrokeConstraint(
            constraint_type=ConstraintType.STAGE_ORDER,
            stroke_ids=list(stage_mapping.keys()),
            priority=priority,
            is_hard=True,
            description=description or "Strokes must follow stage order",
            metadata={'stage_mapping': stage_mapping}
        )
    
    def create_spatial_constraint(self, stroke_positions: Dict[int, List[float]], 
                                max_distance: float = 100.0,
                                priority: int = 1, description: str = "") -> StrokeConstraint:
        """
        创建空间邻近约束
        
        Args:
            stroke_positions: 笔触位置映射
            max_distance: 最大距离
            priority: 优先级
            description: 描述
            
        Returns:
            StrokeConstraint: 约束对象
        """
        return StrokeConstraint(
            constraint_type=ConstraintType.SPATIAL_PROXIMITY,
            stroke_ids=list(stroke_positions.keys()),
            priority=priority,
            is_hard=False,  # 通常是软约束
            penalty_weight=0.1,
            description=description or "Spatially close strokes should be drawn consecutively",
            metadata={'stroke_positions': stroke_positions, 'max_distance': max_distance}
        )
    
    def clear_constraints(self):
        """
        清空所有约束
        """
        self.constraints.clear()
        self.constraint_objects.clear()
        self.logger.info("All constraints cleared")
    
    def export_constraints(self) -> List[Dict[str, Any]]:
        """
        导出约束配置
        
        Returns:
            List[Dict]: 约束配置列表
        """
        exported = []
        for constraint in self.constraints:
            exported.append({
                'constraint_type': constraint.constraint_type.value,
                'stroke_ids': constraint.stroke_ids,
                'priority': constraint.priority,
                'is_hard': constraint.is_hard,
                'penalty_weight': constraint.penalty_weight,
                'description': constraint.description,
                'metadata': constraint.metadata
            })
        return exported
    
    def import_constraints(self, constraint_configs: List[Dict[str, Any]]):
        """
        导入约束配置
        
        Args:
            constraint_configs: 约束配置列表
        """
        self.clear_constraints()
        
        for config in constraint_configs:
            try:
                constraint_type = ConstraintType(config['constraint_type'])
                constraint = StrokeConstraint(
                    constraint_type=constraint_type,
                    stroke_ids=config['stroke_ids'],
                    priority=config.get('priority', 1),
                    is_hard=config.get('is_hard', True),
                    penalty_weight=config.get('penalty_weight', 1.0),
                    description=config.get('description', ''),
                    metadata=config.get('metadata', {})
                )
                self.add_constraint(constraint)
            except Exception as e:
                self.logger.error(f"Error importing constraint: {str(e)}")
        
        self.logger.info(f"Imported {len(self.constraints)} constraints")