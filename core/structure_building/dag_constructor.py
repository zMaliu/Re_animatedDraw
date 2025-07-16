# -*- coding: utf-8 -*-
"""
DAG构造器

实现论文中的有向无环图(DAG)构建：
1. 将笔触关系转换为DAG
2. 检测和解决环路
3. 拓扑排序
4. 层次分析
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum


@dataclass
class DAGNode:
    """
    DAG节点数据结构
    """
    id: int
    stroke_id: int
    in_degree: int = 0
    out_degree: int = 0
    level: int = -1
    predecessors: Set[int] = None
    successors: Set[int] = None
    
    def __post_init__(self):
        if self.predecessors is None:
            self.predecessors = set()
        if self.successors is None:
            self.successors = set()


@dataclass
class DAGEdge:
    """
    DAG边数据结构
    """
    source: int
    target: int
    weight: float
    relation_type: str
    confidence: float


class DAGConstructor:
    """
    DAG构造器
    
    将笔触关系转换为有向无环图
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化DAG构造器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 构建参数
        self.min_edge_weight = config.get('min_edge_weight', 0.1)
        self.max_iterations = config.get('max_cycle_resolution_iterations', 100)
        self.cycle_detection_method = config.get('cycle_detection_method', 'dfs')
        
        # 权重调整参数
        self.weight_decay_factor = config.get('weight_decay_factor', 0.9)
        self.confidence_threshold = config.get('confidence_threshold', 0.3)
        
        # DAG结构
        self.nodes = {}
        self.edges = []
        self.adjacency_list = defaultdict(list)
        self.reverse_adjacency_list = defaultdict(list)
        
    def construct_dag(self, relations: List[Any], 
                     stroke_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        构建DAG
        
        Args:
            relations: 笔触关系列表
            stroke_features: 笔触特征列表
            
        Returns:
            Dict: DAG构建结果
        """
        try:
            if not relations or not stroke_features:
                return self._create_empty_dag()
            
            # 初始化节点
            self._initialize_nodes(stroke_features)
            
            # 添加边
            self._add_edges_from_relations(relations)
            
            # 检测和解决环路
            cycles_resolved = self._resolve_cycles()
            
            # 计算拓扑排序
            topological_order = self._topological_sort()
            
            # 计算层次结构
            levels = self._calculate_levels()
            
            # 验证DAG有效性
            is_valid = self._validate_dag()
            
            result = {
                'nodes': dict(self.nodes),
                'edges': self.edges.copy(),
                'adjacency_list': dict(self.adjacency_list),
                'topological_order': topological_order,
                'levels': levels,
                'cycles_resolved': cycles_resolved,
                'is_valid': is_valid,
                'statistics': self._calculate_dag_statistics()
            }
            
            self.logger.info(f"DAG constructed with {len(self.nodes)} nodes and {len(self.edges)} edges")
            return result
            
        except Exception as e:
            self.logger.error(f"Error constructing DAG: {str(e)}")
            return self._create_empty_dag()
    
    def _create_empty_dag(self) -> Dict[str, Any]:
        """
        创建空DAG
        
        Returns:
            Dict: 空DAG结构
        """
        return {
            'nodes': {},
            'edges': [],
            'adjacency_list': {},
            'topological_order': [],
            'levels': {},
            'cycles_resolved': 0,
            'is_valid': True,
            'statistics': {
                'node_count': 0,
                'edge_count': 0,
                'max_level': 0,
                'average_degree': 0.0
            }
        }
    
    def _initialize_nodes(self, stroke_features: List[Dict[str, Any]]):
        """
        初始化节点
        
        Args:
            stroke_features: 笔触特征列表
        """
        self.nodes.clear()
        
        for i, features in enumerate(stroke_features):
            node = DAGNode(
                id=i,
                stroke_id=i,
                in_degree=0,
                out_degree=0,
                level=-1,
                predecessors=set(),
                successors=set()
            )
            self.nodes[i] = node
    
    def _add_edges_from_relations(self, relations: List[Any]):
        """
        从关系添加边
        
        Args:
            relations: 关系列表
        """
        self.edges.clear()
        self.adjacency_list.clear()
        self.reverse_adjacency_list.clear()
        
        for relation in relations:
            # 根据关系类型确定边的方向
            if hasattr(relation, 'relation_type') and hasattr(relation, 'source_id'):
                source_id = relation.source_id
                target_id = relation.target_id
                relation_type = str(relation.relation_type)
                confidence = getattr(relation, 'confidence', 1.0)
                
                # 只处理有向关系
                if 'before' in relation_type.lower():
                    self._add_edge(source_id, target_id, confidence, relation_type)
                elif 'after' in relation_type.lower():
                    self._add_edge(target_id, source_id, confidence, relation_type)
                # 并发关系不添加有向边
    
    def _add_edge(self, source: int, target: int, confidence: float, relation_type: str):
        """
        添加边
        
        Args:
            source: 源节点
            target: 目标节点
            confidence: 置信度
            relation_type: 关系类型
        """
        if source == target:  # 避免自环
            return
        
        if source not in self.nodes or target not in self.nodes:
            return
        
        # 检查边是否已存在
        existing_edge = None
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                existing_edge = edge
                break
        
        if existing_edge:
            # 更新现有边的权重（取最大值）
            if confidence > existing_edge.confidence:
                existing_edge.weight = confidence
                existing_edge.confidence = confidence
                existing_edge.relation_type = relation_type
        else:
            # 添加新边
            if confidence >= self.min_edge_weight:
                edge = DAGEdge(
                    source=source,
                    target=target,
                    weight=confidence,
                    relation_type=relation_type,
                    confidence=confidence
                )
                self.edges.append(edge)
                
                # 更新邻接表
                self.adjacency_list[source].append(target)
                self.reverse_adjacency_list[target].append(source)
                
                # 更新节点度数
                self.nodes[source].out_degree += 1
                self.nodes[source].successors.add(target)
                self.nodes[target].in_degree += 1
                self.nodes[target].predecessors.add(source)
    
    def _resolve_cycles(self) -> int:
        """
        解决环路
        
        Returns:
            int: 解决的环路数量
        """
        cycles_resolved = 0
        iteration = 0
        
        while iteration < self.max_iterations:
            # 检测环路
            cycles = self._detect_cycles()
            
            if not cycles:
                break
            
            # 解决环路
            for cycle in cycles:
                if self._resolve_single_cycle(cycle):
                    cycles_resolved += 1
            
            iteration += 1
        
        if iteration >= self.max_iterations:
            self.logger.warning(f"Cycle resolution reached max iterations: {self.max_iterations}")
        
        return cycles_resolved
    
    def _detect_cycles(self) -> List[List[int]]:
        """
        检测环路
        
        Returns:
            List[List[int]]: 环路列表
        """
        if self.cycle_detection_method == 'dfs':
            return self._detect_cycles_dfs()
        else:
            return self._detect_cycles_tarjan()
    
    def _detect_cycles_dfs(self) -> List[List[int]]:
        """
        使用DFS检测环路
        
        Returns:
            List[List[int]]: 环路列表
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            if node in rec_stack:
                # 找到环路
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.adjacency_list.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for node in self.nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def _detect_cycles_tarjan(self) -> List[List[int]]:
        """
        使用Tarjan算法检测强连通分量（环路）
        
        Returns:
            List[List[int]]: 环路列表
        """
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        cycles = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)
            
            for neighbor in self.adjacency_list.get(node, []):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                
                if len(component) > 1:  # 强连通分量包含多个节点，即为环路
                    cycles.append(component)
        
        for node in self.nodes:
            if node not in index:
                strongconnect(node)
        
        return cycles
    
    def _resolve_single_cycle(self, cycle: List[int]) -> bool:
        """
        解决单个环路
        
        Args:
            cycle: 环路节点列表
            
        Returns:
            bool: 是否成功解决
        """
        if len(cycle) < 2:
            return False
        
        # 找到环路中权重最小的边
        min_weight = float('inf')
        min_edge_idx = -1
        
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            
            # 查找对应的边
            for idx, edge in enumerate(self.edges):
                if edge.source == source and edge.target == target:
                    if edge.weight < min_weight:
                        min_weight = edge.weight
                        min_edge_idx = idx
                    break
        
        # 移除权重最小的边
        if min_edge_idx >= 0:
            edge_to_remove = self.edges[min_edge_idx]
            self._remove_edge(edge_to_remove)
            self.logger.debug(f"Removed edge {edge_to_remove.source} -> {edge_to_remove.target} to resolve cycle")
            return True
        
        return False
    
    def _remove_edge(self, edge: DAGEdge):
        """
        移除边
        
        Args:
            edge: 要移除的边
        """
        # 从边列表中移除
        if edge in self.edges:
            self.edges.remove(edge)
        
        # 从邻接表中移除
        if edge.target in self.adjacency_list[edge.source]:
            self.adjacency_list[edge.source].remove(edge.target)
        
        if edge.source in self.reverse_adjacency_list[edge.target]:
            self.reverse_adjacency_list[edge.target].remove(edge.source)
        
        # 更新节点度数
        if edge.source in self.nodes:
            self.nodes[edge.source].out_degree -= 1
            self.nodes[edge.source].successors.discard(edge.target)
        
        if edge.target in self.nodes:
            self.nodes[edge.target].in_degree -= 1
            self.nodes[edge.target].predecessors.discard(edge.source)
    
    def _topological_sort(self) -> List[int]:
        """
        拓扑排序
        
        Returns:
            List[int]: 拓扑排序结果
        """
        # 使用Kahn算法
        in_degree = {node_id: node.in_degree for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # 更新邻居节点的入度
            for neighbor in self.adjacency_list.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否所有节点都被访问（即是否存在环路）
        if len(result) != len(self.nodes):
            self.logger.warning("Topological sort incomplete - cycles may still exist")
        
        return result
    
    def _calculate_levels(self) -> Dict[int, int]:
        """
        计算节点层次
        
        Returns:
            Dict[int, int]: 节点到层次的映射
        """
        levels = {}
        
        # 初始化所有节点层次为-1
        for node_id in self.nodes:
            levels[node_id] = -1
        
        # 使用拓扑排序计算层次
        topological_order = self._topological_sort()
        
        for node_id in topological_order:
            # 如果节点没有前驱，层次为0
            if self.nodes[node_id].in_degree == 0:
                levels[node_id] = 0
            else:
                # 层次为所有前驱节点层次的最大值+1
                max_predecessor_level = -1
                for predecessor in self.nodes[node_id].predecessors:
                    if levels[predecessor] > max_predecessor_level:
                        max_predecessor_level = levels[predecessor]
                levels[node_id] = max_predecessor_level + 1
            
            # 更新节点对象中的层次信息
            self.nodes[node_id].level = levels[node_id]
        
        return levels
    
    def _validate_dag(self) -> bool:
        """
        验证DAG有效性
        
        Returns:
            bool: 是否为有效DAG
        """
        # 检查是否存在环路
        cycles = self._detect_cycles()
        if cycles:
            self.logger.warning(f"DAG validation failed: {len(cycles)} cycles detected")
            return False
        
        # 检查拓扑排序是否包含所有节点
        topological_order = self._topological_sort()
        if len(topological_order) != len(self.nodes):
            self.logger.warning("DAG validation failed: topological sort incomplete")
            return False
        
        # 检查边的一致性
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                self.logger.warning(f"DAG validation failed: edge references non-existent node")
                return False
        
        return True
    
    def _calculate_dag_statistics(self) -> Dict[str, Any]:
        """
        计算DAG统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.nodes:
            return {
                'node_count': 0,
                'edge_count': 0,
                'max_level': 0,
                'average_degree': 0.0,
                'density': 0.0,
                'source_nodes': 0,
                'sink_nodes': 0
            }
        
        # 基本统计
        node_count = len(self.nodes)
        edge_count = len(self.edges)
        
        # 层次统计
        levels = [node.level for node in self.nodes.values() if node.level >= 0]
        max_level = max(levels) if levels else 0
        
        # 度数统计
        degrees = [node.in_degree + node.out_degree for node in self.nodes.values()]
        average_degree = np.mean(degrees) if degrees else 0.0
        
        # 密度
        max_edges = node_count * (node_count - 1)  # 有向图最大边数
        density = edge_count / max_edges if max_edges > 0 else 0.0
        
        # 源节点和汇节点
        source_nodes = sum(1 for node in self.nodes.values() if node.in_degree == 0)
        sink_nodes = sum(1 for node in self.nodes.values() if node.out_degree == 0)
        
        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'max_level': max_level,
            'average_degree': average_degree,
            'density': density,
            'source_nodes': source_nodes,
            'sink_nodes': sink_nodes,
            'level_distribution': self._calculate_level_distribution()
        }
    
    def _calculate_level_distribution(self) -> Dict[int, int]:
        """
        计算层次分布
        
        Returns:
            Dict[int, int]: 层次到节点数量的映射
        """
        level_distribution = defaultdict(int)
        
        for node in self.nodes.values():
            if node.level >= 0:
                level_distribution[node.level] += 1
        
        return dict(level_distribution)
    
    def get_critical_path(self) -> List[int]:
        """
        获取关键路径（最长路径）
        
        Returns:
            List[int]: 关键路径节点列表
        """
        if not self.nodes:
            return []
        
        # 计算每个节点的最长路径长度
        distances = {node_id: 0 for node_id in self.nodes}
        predecessors = {node_id: None for node_id in self.nodes}
        
        # 按拓扑顺序处理节点
        topological_order = self._topological_sort()
        
        for node_id in topological_order:
            for successor in self.adjacency_list.get(node_id, []):
                # 找到对应边的权重
                edge_weight = 1  # 默认权重
                for edge in self.edges:
                    if edge.source == node_id and edge.target == successor:
                        edge_weight = edge.weight
                        break
                
                new_distance = distances[node_id] + edge_weight
                if new_distance > distances[successor]:
                    distances[successor] = new_distance
                    predecessors[successor] = node_id
        
        # 找到距离最大的节点
        max_distance_node = max(distances.keys(), key=lambda k: distances[k])
        
        # 重构路径
        path = []
        current = max_distance_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()
        return path
    
    def get_parallel_groups(self) -> List[List[int]]:
        """
        获取可并行执行的节点组
        
        Returns:
            List[List[int]]: 并行组列表
        """
        levels = self._calculate_levels()
        level_groups = defaultdict(list)
        
        for node_id, level in levels.items():
            level_groups[level].append(node_id)
        
        # 按层次排序
        sorted_levels = sorted(level_groups.keys())
        parallel_groups = [level_groups[level] for level in sorted_levels]
        
        return parallel_groups
    
    def export_dot_format(self) -> str:
        """
        导出DOT格式用于可视化
        
        Returns:
            str: DOT格式字符串
        """
        dot_lines = ['digraph DAG {']
        dot_lines.append('  rankdir=TB;')
        dot_lines.append('  node [shape=box];')
        
        # 添加节点
        for node_id, node in self.nodes.items():
            label = f"S{node.stroke_id}\\nL{node.level}"
            dot_lines.append(f'  {node_id} [label="{label}"];')
        
        # 添加边
        for edge in self.edges:
            label = f"{edge.confidence:.2f}"
            dot_lines.append(f'  {edge.source} -> {edge.target} [label="{label}"];')
        
        dot_lines.append('}')
        return '\n'.join(dot_lines)