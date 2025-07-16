# -*- coding: utf-8 -*-
"""
Hasse图简化器

实现论文中的Hasse图简化：
1. 从DAG构建Hasse图
2. 移除传递性冗余边
3. 保持偏序关系的最小表示
4. 优化图结构
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
from copy import deepcopy


@dataclass
class HasseNode:
    """
    Hasse图节点数据结构
    """
    id: int
    stroke_id: int
    level: int
    direct_predecessors: Set[int] = None
    direct_successors: Set[int] = None
    all_predecessors: Set[int] = None
    all_successors: Set[int] = None
    
    def __post_init__(self):
        if self.direct_predecessors is None:
            self.direct_predecessors = set()
        if self.direct_successors is None:
            self.direct_successors = set()
        if self.all_predecessors is None:
            self.all_predecessors = set()
        if self.all_successors is None:
            self.all_successors = set()


@dataclass
class HasseEdge:
    """
    Hasse图边数据结构
    """
    source: int
    target: int
    is_direct: bool
    weight: float
    relation_type: str


class HasseSimplifier:
    """
    Hasse图简化器
    
    将DAG简化为Hasse图，移除传递性冗余
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Hasse图简化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 简化参数
        self.preserve_critical_edges = config.get('preserve_critical_edges', True)
        self.min_direct_edge_weight = config.get('min_direct_edge_weight', 0.2)
        self.transitivity_threshold = config.get('transitivity_threshold', 0.8)
        
        # Hasse图结构
        self.nodes = {}
        self.direct_edges = []
        self.removed_edges = []
        
    def simplify_dag(self, dag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        简化DAG为Hasse图
        
        Args:
            dag_result: DAG构建结果
            
        Returns:
            Dict: Hasse图简化结果
        """
        try:
            if not dag_result or not dag_result.get('nodes'):
                return self._create_empty_hasse()
            
            # 初始化Hasse节点
            self._initialize_hasse_nodes(dag_result)
            
            # 计算传递闭包
            transitive_closure = self._compute_transitive_closure(dag_result)
            
            # 识别直接关系
            direct_relations = self._identify_direct_relations(
                dag_result, transitive_closure
            )
            
            # 构建Hasse图边
            self._build_hasse_edges(dag_result, direct_relations)
            
            # 验证Hasse图
            is_valid = self._validate_hasse_diagram()
            
            # 计算简化统计
            simplification_stats = self._calculate_simplification_stats(dag_result)
            
            result = {
                'nodes': dict(self.nodes),
                'direct_edges': self.direct_edges.copy(),
                'removed_edges': self.removed_edges.copy(),
                'transitive_closure': transitive_closure,
                'is_valid': is_valid,
                'simplification_stats': simplification_stats,
                'hasse_properties': self._calculate_hasse_properties()
            }
            
            self.logger.info(
                f"Hasse diagram simplified: {len(self.direct_edges)} direct edges "
                f"from {len(dag_result.get('edges', []))} original edges"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error simplifying DAG to Hasse diagram: {str(e)}")
            return self._create_empty_hasse()
    
    def _create_empty_hasse(self) -> Dict[str, Any]:
        """
        创建空Hasse图
        
        Returns:
            Dict: 空Hasse图结构
        """
        return {
            'nodes': {},
            'direct_edges': [],
            'removed_edges': [],
            'transitive_closure': {},
            'is_valid': True,
            'simplification_stats': {
                'original_edges': 0,
                'direct_edges': 0,
                'removed_edges': 0,
                'reduction_ratio': 0.0
            },
            'hasse_properties': {
                'node_count': 0,
                'edge_count': 0,
                'max_level': 0,
                'level_distribution': {}
            }
        }
    
    def _initialize_hasse_nodes(self, dag_result: Dict[str, Any]):
        """
        初始化Hasse节点
        
        Args:
            dag_result: DAG构建结果
        """
        self.nodes.clear()
        dag_nodes = dag_result.get('nodes', {})
        
        for node_id, dag_node in dag_nodes.items():
            hasse_node = HasseNode(
                id=node_id,
                stroke_id=getattr(dag_node, 'stroke_id', node_id),
                level=getattr(dag_node, 'level', 0),
                direct_predecessors=set(),
                direct_successors=set(),
                all_predecessors=getattr(dag_node, 'predecessors', set()).copy(),
                all_successors=getattr(dag_node, 'successors', set()).copy()
            )
            self.nodes[node_id] = hasse_node
    
    def _compute_transitive_closure(self, dag_result: Dict[str, Any]) -> Dict[Tuple[int, int], bool]:
        """
        计算传递闭包
        
        Args:
            dag_result: DAG构建结果
            
        Returns:
            Dict: 传递闭包矩阵
        """
        adjacency_list = dag_result.get('adjacency_list', {})
        nodes = list(self.nodes.keys())
        
        # 初始化传递闭包矩阵
        closure = {}
        for i in nodes:
            for j in nodes:
                closure[(i, j)] = False
        
        # 设置直接连接
        for source, targets in adjacency_list.items():
            for target in targets:
                if source in nodes and target in nodes:
                    closure[(source, target)] = True
        
        # Floyd-Warshall算法计算传递闭包
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if closure[(i, k)] and closure[(k, j)]:
                        closure[(i, j)] = True
        
        return closure
    
    def _identify_direct_relations(self, dag_result: Dict[str, Any], 
                                 transitive_closure: Dict[Tuple[int, int], bool]) -> Set[Tuple[int, int]]:
        """
        识别直接关系（非传递性关系）
        
        Args:
            dag_result: DAG构建结果
            transitive_closure: 传递闭包
            
        Returns:
            Set: 直接关系集合
        """
        adjacency_list = dag_result.get('adjacency_list', {})
        direct_relations = set()
        
        for source, targets in adjacency_list.items():
            for target in targets:
                if source in self.nodes and target in self.nodes:
                    # 检查是否为直接关系
                    if self._is_direct_relation(source, target, transitive_closure):
                        direct_relations.add((source, target))
                    else:
                        # 记录被移除的传递性边
                        self._record_removed_edge(source, target, dag_result)
        
        return direct_relations
    
    def _is_direct_relation(self, source: int, target: int, 
                          transitive_closure: Dict[Tuple[int, int], bool]) -> bool:
        """
        判断是否为直接关系
        
        Args:
            source: 源节点
            target: 目标节点
            transitive_closure: 传递闭包
            
        Returns:
            bool: 是否为直接关系
        """
        # 检查是否存在中间节点使得source -> intermediate -> target
        for intermediate in self.nodes:
            if (intermediate != source and intermediate != target and
                transitive_closure.get((source, intermediate), False) and
                transitive_closure.get((intermediate, target), False)):
                return False
        
        return True
    
    def _record_removed_edge(self, source: int, target: int, dag_result: Dict[str, Any]):
        """
        记录被移除的边
        
        Args:
            source: 源节点
            target: 目标节点
            dag_result: DAG构建结果
        """
        # 查找原始边信息
        original_edges = dag_result.get('edges', [])
        for edge in original_edges:
            if (hasattr(edge, 'source') and hasattr(edge, 'target') and
                edge.source == source and edge.target == target):
                
                removed_edge = HasseEdge(
                    source=source,
                    target=target,
                    is_direct=False,
                    weight=getattr(edge, 'weight', 0.0),
                    relation_type=getattr(edge, 'relation_type', 'transitive')
                )
                self.removed_edges.append(removed_edge)
                break
    
    def _build_hasse_edges(self, dag_result: Dict[str, Any], 
                          direct_relations: Set[Tuple[int, int]]):
        """
        构建Hasse图边
        
        Args:
            dag_result: DAG构建结果
            direct_relations: 直接关系集合
        """
        self.direct_edges.clear()
        original_edges = dag_result.get('edges', [])
        
        for source, target in direct_relations:
            # 查找原始边信息
            edge_info = None
            for edge in original_edges:
                if (hasattr(edge, 'source') and hasattr(edge, 'target') and
                    edge.source == source and edge.target == target):
                    edge_info = edge
                    break
            
            # 创建Hasse边
            hasse_edge = HasseEdge(
                source=source,
                target=target,
                is_direct=True,
                weight=getattr(edge_info, 'weight', 1.0) if edge_info else 1.0,
                relation_type=getattr(edge_info, 'relation_type', 'direct') if edge_info else 'direct'
            )
            self.direct_edges.append(hasse_edge)
            
            # 更新节点的直接关系
            self.nodes[source].direct_successors.add(target)
            self.nodes[target].direct_predecessors.add(source)
    
    def _validate_hasse_diagram(self) -> bool:
        """
        验证Hasse图有效性
        
        Returns:
            bool: 是否为有效Hasse图
        """
        try:
            # 检查1: 无自环
            for edge in self.direct_edges:
                if edge.source == edge.target:
                    self.logger.warning("Hasse diagram validation failed: self-loop detected")
                    return False
            
            # 检查2: 传递性简化正确性
            if not self._verify_transitivity_reduction():
                self.logger.warning("Hasse diagram validation failed: transitivity reduction incorrect")
                return False
            
            # 检查3: 层次一致性
            if not self._verify_level_consistency():
                self.logger.warning("Hasse diagram validation failed: level inconsistency")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Hasse diagram: {str(e)}")
            return False
    
    def _verify_transitivity_reduction(self) -> bool:
        """
        验证传递性简化的正确性
        
        Returns:
            bool: 传递性简化是否正确
        """
        # 构建直接关系的邻接表
        direct_adjacency = defaultdict(set)
        for edge in self.direct_edges:
            direct_adjacency[edge.source].add(edge.target)
        
        # 检查是否存在传递性边
        for source in self.nodes:
            for intermediate in direct_adjacency[source]:
                for target in direct_adjacency[intermediate]:
                    # 如果存在source -> intermediate -> target，
                    # 则不应该存在直接边source -> target
                    if target in direct_adjacency[source]:
                        return False
        
        return True
    
    def _verify_level_consistency(self) -> bool:
        """
        验证层次一致性
        
        Returns:
            bool: 层次是否一致
        """
        for edge in self.direct_edges:
            source_level = self.nodes[edge.source].level
            target_level = self.nodes[edge.target].level
            
            # 在Hasse图中，边应该连接相邻层次或跨越层次
            if source_level >= target_level:
                return False
        
        return True
    
    def _calculate_simplification_stats(self, dag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算简化统计信息
        
        Args:
            dag_result: DAG构建结果
            
        Returns:
            Dict: 简化统计信息
        """
        original_edges = len(dag_result.get('edges', []))
        direct_edges = len(self.direct_edges)
        removed_edges = len(self.removed_edges)
        
        reduction_ratio = (removed_edges / original_edges) if original_edges > 0 else 0.0
        
        return {
            'original_edges': original_edges,
            'direct_edges': direct_edges,
            'removed_edges': removed_edges,
            'reduction_ratio': reduction_ratio,
            'compression_efficiency': 1.0 - (direct_edges / original_edges) if original_edges > 0 else 0.0
        }
    
    def _calculate_hasse_properties(self) -> Dict[str, Any]:
        """
        计算Hasse图属性
        
        Returns:
            Dict: Hasse图属性
        """
        if not self.nodes:
            return {
                'node_count': 0,
                'edge_count': 0,
                'max_level': 0,
                'level_distribution': {},
                'width': 0,
                'height': 0,
                'density': 0.0
            }
        
        # 基本属性
        node_count = len(self.nodes)
        edge_count = len(self.direct_edges)
        
        # 层次属性
        levels = [node.level for node in self.nodes.values()]
        max_level = max(levels) if levels else 0
        
        # 层次分布
        level_distribution = defaultdict(int)
        for level in levels:
            level_distribution[level] += 1
        
        # 宽度（最大层次的节点数）
        width = max(level_distribution.values()) if level_distribution else 0
        
        # 高度（层次数）
        height = len(level_distribution)
        
        # 密度
        max_edges = node_count * (node_count - 1) // 2  # 无向图最大边数的近似
        density = edge_count / max_edges if max_edges > 0 else 0.0
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'max_level': max_level,
            'level_distribution': dict(level_distribution),
            'width': width,
            'height': height,
            'density': density,
            'average_degree': self._calculate_average_degree(),
            'critical_path_length': self._calculate_critical_path_length()
        }
    
    def _calculate_average_degree(self) -> float:
        """
        计算平均度数
        
        Returns:
            float: 平均度数
        """
        if not self.nodes:
            return 0.0
        
        total_degree = 0
        for node in self.nodes.values():
            total_degree += len(node.direct_predecessors) + len(node.direct_successors)
        
        return total_degree / len(self.nodes)
    
    def _calculate_critical_path_length(self) -> int:
        """
        计算关键路径长度
        
        Returns:
            int: 关键路径长度
        """
        if not self.nodes:
            return 0
        
        # 使用动态规划计算最长路径
        distances = {node_id: 0 for node_id in self.nodes}
        
        # 按层次顺序处理节点
        nodes_by_level = defaultdict(list)
        for node_id, node in self.nodes.items():
            nodes_by_level[node.level].append(node_id)
        
        for level in sorted(nodes_by_level.keys()):
            for node_id in nodes_by_level[level]:
                for successor in self.nodes[node_id].direct_successors:
                    distances[successor] = max(
                        distances[successor],
                        distances[node_id] + 1
                    )
        
        return max(distances.values()) if distances else 0
    
    def get_minimal_elements(self) -> List[int]:
        """
        获取最小元素（无前驱的节点）
        
        Returns:
            List[int]: 最小元素列表
        """
        return [node_id for node_id, node in self.nodes.items() 
                if not node.direct_predecessors]
    
    def get_maximal_elements(self) -> List[int]:
        """
        获取最大元素（无后继的节点）
        
        Returns:
            List[int]: 最大元素列表
        """
        return [node_id for node_id, node in self.nodes.items() 
                if not node.direct_successors]
    
    def get_comparable_pairs(self) -> List[Tuple[int, int]]:
        """
        获取可比较的节点对
        
        Returns:
            List[Tuple[int, int]]: 可比较节点对列表
        """
        comparable_pairs = []
        
        for node_id, node in self.nodes.items():
            # 添加所有后继节点作为可比较对
            for successor in node.all_successors:
                comparable_pairs.append((node_id, successor))
        
        return comparable_pairs
    
    def get_incomparable_pairs(self) -> List[Tuple[int, int]]:
        """
        获取不可比较的节点对
        
        Returns:
            List[Tuple[int, int]]: 不可比较节点对列表
        """
        incomparable_pairs = []
        node_ids = list(self.nodes.keys())
        
        for i, node1 in enumerate(node_ids):
            for j, node2 in enumerate(node_ids[i+1:], i+1):
                # 检查两个节点是否可比较
                if (node2 not in self.nodes[node1].all_successors and
                    node1 not in self.nodes[node2].all_successors):
                    incomparable_pairs.append((node1, node2))
        
        return incomparable_pairs
    
    def export_hasse_dot(self) -> str:
        """
        导出Hasse图的DOT格式
        
        Returns:
            str: DOT格式字符串
        """
        dot_lines = ['digraph HasseDiagram {']
        dot_lines.append('  rankdir=BT;')  # 从下到上布局
        dot_lines.append('  node [shape=circle];')
        
        # 按层次分组节点
        nodes_by_level = defaultdict(list)
        for node_id, node in self.nodes.items():
            nodes_by_level[node.level].append(node_id)
        
        # 添加层次约束
        for level in sorted(nodes_by_level.keys()):
            if len(nodes_by_level[level]) > 1:
                dot_lines.append(f'  {{rank=same; {"; ".join(map(str, nodes_by_level[level]))};}}}')
        
        # 添加节点
        for node_id, node in self.nodes.items():
            label = f"S{node.stroke_id}"
            dot_lines.append(f'  {node_id} [label="{label}"];')
        
        # 添加边
        for edge in self.direct_edges:
            dot_lines.append(f'  {edge.source} -> {edge.target};')
        
        dot_lines.append('}')
        return '\n'.join(dot_lines)
    
    def analyze_partial_order(self) -> Dict[str, Any]:
        """
        分析偏序关系的性质
        
        Returns:
            Dict: 偏序分析结果
        """
        minimal_elements = self.get_minimal_elements()
        maximal_elements = self.get_maximal_elements()
        comparable_pairs = self.get_comparable_pairs()
        incomparable_pairs = self.get_incomparable_pairs()
        
        total_pairs = len(self.nodes) * (len(self.nodes) - 1) // 2
        comparability_ratio = len(comparable_pairs) / total_pairs if total_pairs > 0 else 0.0
        
        return {
            'minimal_elements': minimal_elements,
            'maximal_elements': maximal_elements,
            'comparable_pairs_count': len(comparable_pairs),
            'incomparable_pairs_count': len(incomparable_pairs),
            'comparability_ratio': comparability_ratio,
            'is_total_order': len(incomparable_pairs) == 0,
            'is_chain': self._is_chain(),
            'is_antichain': self._is_antichain(),
            'width': self._calculate_width(),
            'height': self._calculate_height()
        }
    
    def _is_chain(self) -> bool:
        """
        判断是否为链（全序）
        
        Returns:
            bool: 是否为链
        """
        return len(self.get_incomparable_pairs()) == 0
    
    def _is_antichain(self) -> bool:
        """
        判断是否为反链（所有元素都不可比较）
        
        Returns:
            bool: 是否为反链
        """
        return len(self.get_comparable_pairs()) == 0
    
    def _calculate_width(self) -> int:
        """
        计算偏序集的宽度（最大反链大小）
        
        Returns:
            int: 宽度
        """
        # 使用Dilworth定理：宽度等于最小链分解的链数
        # 这里简化为最大层次的节点数
        level_counts = defaultdict(int)
        for node in self.nodes.values():
            level_counts[node.level] += 1
        
        return max(level_counts.values()) if level_counts else 0
    
    def _calculate_height(self) -> int:
        """
        计算偏序集的高度（最大链长度）
        
        Returns:
            int: 高度
        """
        return self._calculate_critical_path_length() + 1