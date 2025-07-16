# -*- coding: utf-8 -*-
"""
笔画组织器

实现论文中的三阶段笔画组织方法：
1. 笔画分组 - 将相关笔画组织成组
2. 组内排序 - 优化组内笔画顺序
3. 组间排序 - 优化组间绘制顺序
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans
import networkx as nx
from collections import defaultdict
import math


@dataclass
class StrokeGroup:
    """
    笔画组数据结构
    
    Attributes:
        group_id (int): 组ID
        strokes (List[Dict]): 组内笔画列表
        centroid (Tuple[float, float]): 组质心
        bounding_box (Tuple[int, int, int, int]): 组边界框
        group_type (str): 组类型（字符、部首、笔画簇等）
        internal_order (List[int]): 组内笔画顺序
        confidence (float): 分组置信度
        metadata (Dict): 元数据
    """
    group_id: int
    strokes: List[Dict[str, Any]]
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    group_type: str
    internal_order: List[int]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class OrganizationResult:
    """
    组织结果数据结构
    
    Attributes:
        stroke_groups (List[StrokeGroup]): 笔画组列表
        group_order (List[int]): 组间顺序
        final_stroke_order (List[int]): 最终笔画顺序
        organization_score (float): 组织质量分数
        processing_time (float): 处理时间
        metadata (Dict): 元数据
    """
    stroke_groups: List[StrokeGroup]
    group_order: List[int]
    final_stroke_order: List[int]
    organization_score: float
    processing_time: float
    metadata: Dict[str, Any]


class StrokeOrganizer:
    """
    笔画组织器
    
    实现三阶段笔画组织算法
    """
    
    def __init__(self, config):
        """
        初始化笔画组织器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分组参数
        self.grouping_method = config['stroke_ordering'].get('grouping_method', 'hierarchical')
        self.max_groups = config['stroke_ordering'].get('max_groups', 20)
        self.min_group_size = config['stroke_ordering'].get('min_group_size', 1)
        self.distance_threshold = config['stroke_ordering'].get('distance_threshold', 50)
        
        # 排序参数
        self.ordering_method = config['stroke_ordering'].get('ordering_method', 'hybrid')
        self.spatial_weight = config['stroke_ordering'].get('spatial_weight', 0.4)
        self.semantic_weight = config['stroke_ordering'].get('semantic_weight', 0.3)
        self.temporal_weight = config['stroke_ordering'].get('temporal_weight', 0.3)
        
        # 优化参数
        self.use_optimization = config['stroke_ordering'].get('use_optimization', True)
        self.optimization_iterations = config['stroke_ordering'].get('optimization_iterations', 100)
    
    def organize_strokes(self, strokes: List[Dict[str, Any]]) -> OrganizationResult:
        """
        组织笔画顺序
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            OrganizationResult: 组织结果
        """
        try:
            import time
            start_time = time.time()
            
            self.logger.info(f"Organizing {len(strokes)} strokes")
            
            # 阶段1：笔画分组
            stroke_groups = self._group_strokes(strokes)
            self.logger.info(f"Created {len(stroke_groups)} stroke groups")
            
            # 阶段2：组内排序
            for group in stroke_groups:
                group.internal_order = self._order_strokes_within_group(group)
            
            # 阶段3：组间排序
            group_order = self._order_groups(stroke_groups)
            
            # 生成最终笔画顺序
            final_stroke_order = self._generate_final_order(stroke_groups, group_order)
            
            # 计算组织质量分数
            organization_score = self._evaluate_organization(stroke_groups, group_order, strokes)
            
            processing_time = time.time() - start_time
            
            result = OrganizationResult(
                stroke_groups=stroke_groups,
                group_order=group_order,
                final_stroke_order=final_stroke_order,
                organization_score=organization_score,
                processing_time=processing_time,
                metadata={
                    'grouping_method': self.grouping_method,
                    'ordering_method': self.ordering_method,
                    'num_groups': len(stroke_groups),
                    'avg_group_size': np.mean([len(g.strokes) for g in stroke_groups])
                }
            )
            
            self.logger.info(f"Organization completed in {processing_time:.2f}s, score: {organization_score:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error organizing strokes: {str(e)}")
            # 返回默认顺序
            return self._create_default_organization(strokes)
    
    def _group_strokes(self, strokes: List[Dict[str, Any]]) -> List[StrokeGroup]:
        """
        第一阶段：笔画分组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            if self.grouping_method == 'hierarchical':
                return self._hierarchical_grouping(strokes)
            elif self.grouping_method == 'dbscan':
                return self._dbscan_grouping(strokes)
            elif self.grouping_method == 'kmeans':
                return self._kmeans_grouping(strokes)
            elif self.grouping_method == 'semantic':
                return self._semantic_grouping(strokes)
            else:
                self.logger.warning(f"Unknown grouping method: {self.grouping_method}")
                return self._hierarchical_grouping(strokes)
                
        except Exception as e:
            self.logger.error(f"Error in stroke grouping: {str(e)}")
            # 返回单个组
            return [self._create_single_group(strokes)]
    
    def _hierarchical_grouping(self, strokes: List[Dict[str, Any]]) -> List[StrokeGroup]:
        """
        层次聚类分组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            if len(strokes) <= 1:
                return [self._create_single_group(strokes)]
            
            # 提取笔画特征
            features = self._extract_grouping_features(strokes)
            
            # 层次聚类
            linkage_matrix = linkage(features, method='ward')
            
            # 确定聚类数量
            num_clusters = min(self.max_groups, max(1, len(strokes) // 3))
            cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
            
            # 创建笔画组
            groups = self._create_groups_from_labels(strokes, cluster_labels)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical grouping: {str(e)}")
            return [self._create_single_group(strokes)]
    
    def _dbscan_grouping(self, strokes: List[Dict[str, Any]]) -> List[StrokeGroup]:
        """
        DBSCAN聚类分组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            if len(strokes) <= 1:
                return [self._create_single_group(strokes)]
            
            # 提取空间特征（质心坐标）
            centroids = np.array([stroke.get('centroid', (0, 0)) for stroke in strokes])
            
            # DBSCAN聚类
            dbscan = DBSCAN(eps=self.distance_threshold, min_samples=self.min_group_size)
            cluster_labels = dbscan.fit_predict(centroids)
            
            # 处理噪声点（标签为-1）
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            # 创建笔画组
            groups = []
            for label in unique_labels:
                group_strokes = [strokes[i] for i, l in enumerate(cluster_labels) if l == label]
                if len(group_strokes) >= self.min_group_size:
                    group = self._create_stroke_group(group_strokes, len(groups), 'spatial_cluster')
                    groups.append(group)
            
            # 处理噪声点
            noise_strokes = [strokes[i] for i, l in enumerate(cluster_labels) if l == -1]
            if noise_strokes:
                noise_group = self._create_stroke_group(noise_strokes, len(groups), 'noise')
                groups.append(noise_group)
            
            return groups if groups else [self._create_single_group(strokes)]
            
        except Exception as e:
            self.logger.error(f"Error in DBSCAN grouping: {str(e)}")
            return [self._create_single_group(strokes)]
    
    def _kmeans_grouping(self, strokes: List[Dict[str, Any]]) -> List[StrokeGroup]:
        """
        K-means聚类分组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            if len(strokes) <= 1:
                return [self._create_single_group(strokes)]
            
            # 提取特征
            features = self._extract_grouping_features(strokes)
            
            # 确定聚类数量
            num_clusters = min(self.max_groups, max(1, len(strokes) // 2))
            
            # K-means聚类
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # 创建笔画组
            groups = self._create_groups_from_labels(strokes, cluster_labels)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error in K-means grouping: {str(e)}")
            return [self._create_single_group(strokes)]
    
    def _semantic_grouping(self, strokes: List[Dict[str, Any]]) -> List[StrokeGroup]:
        """
        基于语义的分组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            # 按笔画类型分组
            type_groups = defaultdict(list)
            
            for i, stroke in enumerate(strokes):
                stroke_type = stroke.get('stroke_class', 'unknown')
                type_groups[stroke_type].append((i, stroke))
            
            # 创建笔画组
            groups = []
            for stroke_type, stroke_list in type_groups.items():
                if len(stroke_list) >= self.min_group_size:
                    group_strokes = [stroke for _, stroke in stroke_list]
                    group = self._create_stroke_group(group_strokes, len(groups), f'semantic_{stroke_type}')
                    groups.append(group)
            
            return groups if groups else [self._create_single_group(strokes)]
            
        except Exception as e:
            self.logger.error(f"Error in semantic grouping: {str(e)}")
            return [self._create_single_group(strokes)]
    
    def _order_strokes_within_group(self, group: StrokeGroup) -> List[int]:
        """
        第二阶段：组内笔画排序
        
        Args:
            group (StrokeGroup): 笔画组
            
        Returns:
            List[int]: 组内笔画顺序（索引）
        """
        try:
            if len(group.strokes) <= 1:
                return list(range(len(group.strokes)))
            
            if self.ordering_method == 'spatial':
                return self._spatial_ordering(group.strokes)
            elif self.ordering_method == 'rule_based':
                return self._rule_based_ordering(group.strokes)
            elif self.ordering_method == 'hybrid':
                return self._hybrid_ordering(group.strokes)
            else:
                return self._spatial_ordering(group.strokes)
                
        except Exception as e:
            self.logger.error(f"Error ordering strokes within group: {str(e)}")
            return list(range(len(group.strokes)))
    
    def _spatial_ordering(self, strokes: List[Dict[str, Any]]) -> List[int]:
        """
        基于空间位置的排序
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[int]: 排序索引
        """
        try:
            # 提取质心坐标
            centroids = np.array([stroke.get('centroid', (0, 0)) for stroke in strokes])
            
            # 计算从左上到右下的排序
            # 使用加权坐标：y坐标权重更高（从上到下优先）
            weighted_coords = centroids[:, 1] * 2 + centroids[:, 0]  # 2*y + x
            
            # 排序
            order = np.argsort(weighted_coords)
            
            return order.tolist()
            
        except Exception as e:
            self.logger.error(f"Error in spatial ordering: {str(e)}")
            return list(range(len(strokes)))
    
    def _rule_based_ordering(self, strokes: List[Dict[str, Any]]) -> List[int]:
        """
        基于规则的排序
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[int]: 排序索引
        """
        try:
            # 中国书法笔画顺序规则
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
            
            # 计算排序分数
            scores = []
            for i, stroke in enumerate(strokes):
                stroke_class = stroke.get('stroke_class', 'complex')
                priority = stroke_priority.get(stroke_class, 9)
                
                # 结合空间位置
                centroid = stroke.get('centroid', (0, 0))
                spatial_score = centroid[1] * 1000 + centroid[0]  # y优先
                
                # 综合分数
                total_score = priority * 10000 + spatial_score
                scores.append((total_score, i))
            
            # 排序
            scores.sort()
            order = [idx for _, idx in scores]
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error in rule-based ordering: {str(e)}")
            return list(range(len(strokes)))
    
    def _hybrid_ordering(self, strokes: List[Dict[str, Any]]) -> List[int]:
        """
        混合排序方法
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[int]: 排序索引
        """
        try:
            # 获取不同方法的排序结果
            spatial_order = self._spatial_ordering(strokes)
            rule_order = self._rule_based_ordering(strokes)
            
            # 计算综合排序分数
            final_scores = []
            
            for i in range(len(strokes)):
                spatial_rank = spatial_order.index(i)
                rule_rank = rule_order.index(i)
                
                # 加权综合
                combined_score = (
                    self.spatial_weight * spatial_rank +
                    (1 - self.spatial_weight) * rule_rank
                )
                
                final_scores.append((combined_score, i))
            
            # 排序
            final_scores.sort()
            order = [idx for _, idx in final_scores]
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error in hybrid ordering: {str(e)}")
            return list(range(len(strokes)))
    
    def _order_groups(self, stroke_groups: List[StrokeGroup]) -> List[int]:
        """
        第三阶段：组间排序
        
        Args:
            stroke_groups (List[StrokeGroup]): 笔画组列表
            
        Returns:
            List[int]: 组顺序
        """
        try:
            if len(stroke_groups) <= 1:
                return list(range(len(stroke_groups)))
            
            # 提取组特征
            group_features = []
            for group in stroke_groups:
                # 使用组质心作为主要特征
                centroid = group.centroid
                bbox = group.bounding_box
                
                features = [
                    centroid[0],  # x坐标
                    centroid[1],  # y坐标
                    bbox[2] * bbox[3],  # 面积
                    len(group.strokes),  # 笔画数量
                ]
                group_features.append(features)
            
            # 基于空间位置排序（从左上到右下）
            centroids = np.array([group.centroid for group in stroke_groups])
            weighted_coords = centroids[:, 1] * 2 + centroids[:, 0]  # 2*y + x
            
            order = np.argsort(weighted_coords)
            
            return order.tolist()
            
        except Exception as e:
            self.logger.error(f"Error ordering groups: {str(e)}")
            return list(range(len(stroke_groups)))
    
    def _generate_final_order(self, stroke_groups: List[StrokeGroup], 
                            group_order: List[int]) -> List[int]:
        """
        生成最终笔画顺序
        
        Args:
            stroke_groups (List[StrokeGroup]): 笔画组列表
            group_order (List[int]): 组顺序
            
        Returns:
            List[int]: 最终笔画顺序
        """
        try:
            final_order = []
            
            for group_idx in group_order:
                group = stroke_groups[group_idx]
                
                # 获取组内笔画的原始索引
                for stroke_idx in group.internal_order:
                    if stroke_idx < len(group.strokes):
                        stroke = group.strokes[stroke_idx]
                        original_idx = stroke.get('original_index', len(final_order))
                        final_order.append(original_idx)
            
            return final_order
            
        except Exception as e:
            self.logger.error(f"Error generating final order: {str(e)}")
            return list(range(sum(len(g.strokes) for g in stroke_groups)))
    
    def _extract_grouping_features(self, strokes: List[Dict[str, Any]]) -> np.ndarray:
        """
        提取用于分组的特征
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            np.ndarray: 特征矩阵
        """
        try:
            features = []
            
            for stroke in strokes:
                # 空间特征
                centroid = stroke.get('centroid', (0, 0))
                bbox = stroke.get('bounding_rect', (0, 0, 1, 1))
                
                # 几何特征
                area = stroke.get('area', 0)
                aspect_ratio = stroke.get('aspect_ratio', 1)
                orientation = stroke.get('orientation', 0)
                
                feature_vector = [
                    centroid[0],
                    centroid[1],
                    bbox[2],  # width
                    bbox[3],  # height
                    area,
                    aspect_ratio,
                    np.cos(orientation),  # 方向的cos值
                    np.sin(orientation),  # 方向的sin值
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting grouping features: {str(e)}")
            return np.zeros((len(strokes), 8))
    
    def _create_groups_from_labels(self, strokes: List[Dict[str, Any]], 
                                 labels: np.ndarray) -> List[StrokeGroup]:
        """
        从聚类标签创建笔画组
        
        Args:
            strokes (List[Dict]): 笔画列表
            labels (np.ndarray): 聚类标签
            
        Returns:
            List[StrokeGroup]: 笔画组列表
        """
        try:
            groups = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                group_strokes = [strokes[i] for i, l in enumerate(labels) if l == label]
                
                if len(group_strokes) >= self.min_group_size:
                    group = self._create_stroke_group(group_strokes, len(groups), 'cluster')
                    groups.append(group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error creating groups from labels: {str(e)}")
            return [self._create_single_group(strokes)]
    
    def _create_stroke_group(self, strokes: List[Dict[str, Any]], 
                           group_id: int, group_type: str) -> StrokeGroup:
        """
        创建笔画组
        
        Args:
            strokes (List[Dict]): 笔画列表
            group_id (int): 组ID
            group_type (str): 组类型
            
        Returns:
            StrokeGroup: 笔画组
        """
        try:
            # 添加原始索引
            for i, stroke in enumerate(strokes):
                if 'original_index' not in stroke:
                    stroke['original_index'] = i
            
            # 计算组质心
            centroids = [stroke.get('centroid', (0, 0)) for stroke in strokes]
            centroid = (np.mean([c[0] for c in centroids]), np.mean([c[1] for c in centroids]))
            
            # 计算组边界框
            bboxes = [stroke.get('bounding_rect', (0, 0, 1, 1)) for stroke in strokes]
            min_x = min(bbox[0] for bbox in bboxes)
            min_y = min(bbox[1] for bbox in bboxes)
            max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
            max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
            bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # 初始顺序
            internal_order = list(range(len(strokes)))
            
            return StrokeGroup(
                group_id=group_id,
                strokes=strokes,
                centroid=centroid,
                bounding_box=bounding_box,
                group_type=group_type,
                internal_order=internal_order,
                confidence=0.8,  # 默认置信度
                metadata={'creation_method': group_type}
            )
            
        except Exception as e:
            self.logger.error(f"Error creating stroke group: {str(e)}")
            return StrokeGroup(
                group_id=group_id,
                strokes=strokes,
                centroid=(0, 0),
                bounding_box=(0, 0, 1, 1),
                group_type='error',
                internal_order=list(range(len(strokes))),
                confidence=0.1,
                metadata={}
            )
    
    def _create_single_group(self, strokes: List[Dict[str, Any]]) -> StrokeGroup:
        """
        创建包含所有笔画的单个组
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            StrokeGroup: 单个笔画组
        """
        return self._create_stroke_group(strokes, 0, 'single_group')
    
    def _evaluate_organization(self, stroke_groups: List[StrokeGroup], 
                             group_order: List[int], 
                             original_strokes: List[Dict[str, Any]]) -> float:
        """
        评估组织质量
        
        Args:
            stroke_groups (List[StrokeGroup]): 笔画组列表
            group_order (List[int]): 组顺序
            original_strokes (List[Dict]): 原始笔画列表
            
        Returns:
            float: 组织质量分数
        """
        try:
            scores = []
            
            # 1. 组内紧密度
            for group in stroke_groups:
                if len(group.strokes) > 1:
                    centroids = np.array([s.get('centroid', (0, 0)) for s in group.strokes])
                    distances = cdist(centroids, centroids)
                    avg_distance = np.mean(distances[distances > 0])
                    compactness = 1 / (1 + avg_distance / 100)  # 归一化
                    scores.append(compactness)
            
            # 2. 组间分离度
            if len(stroke_groups) > 1:
                group_centroids = np.array([g.centroid for g in stroke_groups])
                group_distances = cdist(group_centroids, group_centroids)
                avg_separation = np.mean(group_distances[group_distances > 0])
                separation = min(1.0, avg_separation / 200)  # 归一化
                scores.append(separation)
            
            # 3. 顺序合理性（基于空间位置）
            final_order = self._generate_final_order(stroke_groups, group_order)
            if len(final_order) > 1:
                order_score = self._evaluate_order_quality(final_order, original_strokes)
                scores.append(order_score)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception as e:
            self.logger.error(f"Error evaluating organization: {str(e)}")
            return 0.5
    
    def _evaluate_order_quality(self, order: List[int], strokes: List[Dict[str, Any]]) -> float:
        """
        评估顺序质量
        
        Args:
            order (List[int]): 笔画顺序
            strokes (List[Dict]): 笔画列表
            
        Returns:
            float: 顺序质量分数
        """
        try:
            if len(order) <= 1:
                return 1.0
            
            # 计算相邻笔画间的距离
            distances = []
            for i in range(len(order) - 1):
                stroke1 = strokes[order[i]]
                stroke2 = strokes[order[i + 1]]
                
                centroid1 = stroke1.get('centroid', (0, 0))
                centroid2 = stroke2.get('centroid', (0, 0))
                
                centroid1_array = np.array(centroid1)
                centroid2_array = np.array(centroid2)
                distance = np.linalg.norm(centroid1_array - centroid2_array)
                distances.append(distance)
            
            # 距离越小，顺序质量越高
            avg_distance = np.mean(distances)
            quality = 1 / (1 + avg_distance / 100)  # 归一化
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Error evaluating order quality: {str(e)}")
            return 0.5
    
    def _create_default_organization(self, strokes: List[Dict[str, Any]]) -> OrganizationResult:
        """
        创建默认组织结果
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            OrganizationResult: 默认组织结果
        """
        single_group = self._create_single_group(strokes)
        
        return OrganizationResult(
            stroke_groups=[single_group],
            group_order=[0],
            final_stroke_order=list(range(len(strokes))),
            organization_score=0.5,
            processing_time=0.0,
            metadata={'method': 'default'}
        )