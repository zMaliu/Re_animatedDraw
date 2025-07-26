#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块3: 多阶段结构构建
将笔触分为"主要结构-局部细节-装饰笔触"三个阶段，构建Hasse图表示层级关系

主要功能:
1. 特征归一化
2. 偏序关系定义
3. Hasse图构建
4. 拓扑排序
5. 阶段分组
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class StructureConstructor:
    """多阶段结构构建器"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.hasse_graph = None
        self.stages = None
        self.feature_weights = {
            'skeleton_length': 1.0,
            'area': 1.0,
            'scale': 1.0,
            'wetness': 1.0,
            'thickness': 1.0,
            'position_saliency': 1.0
        }
    
    def build_structure(self, features_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict]:
        """构建多阶段结构"""
        # 1. 计算综合特征得分
        comprehensive_scores = self._compute_comprehensive_scores(features_df)
        
        # 2. 构建偏序关系
        partial_order = self._build_partial_order(features_df, comprehensive_scores)
        
        # 3. 构建Hasse图
        self.hasse_graph = self._build_hasse_graph(partial_order)
        
        # 4. 阶段分组
        self.stages = self._create_stages(features_df, comprehensive_scores)
        
        # 5. 拓扑排序验证
        topo_order = self._get_topological_order()
        
        if self.debug:
            print(f"构建Hasse图完成，节点数: {self.hasse_graph.number_of_nodes()}")
            print(f"边数: {self.hasse_graph.number_of_edges()}")
            print(f"阶段分组: {[(stage, len(strokes)) for stage, strokes in self.stages.items()]}")
            print(f"拓扑排序长度: {len(topo_order)}")
        
        return self.hasse_graph, self.stages
    
    def _compute_comprehensive_scores(self, features_df: pd.DataFrame) -> Dict[int, float]:
        """计算每个笔触的综合特征得分"""
        scores = {}
        
        for _, row in features_df.iterrows():
            stroke_id = int(row['stroke_id'])
            score = 0.0
            
            # 加权求和
            for feature_name, weight in self.feature_weights.items():
                if feature_name in row:
                    score += weight * row[feature_name]
            
            scores[stroke_id] = score
        
        return scores
    
    def _build_partial_order(self, features_df: pd.DataFrame, 
                           comprehensive_scores: Dict[int, float]) -> List[Tuple[int, int]]:
        """构建偏序关系"""
        partial_order = []
        stroke_ids = features_df['stroke_id'].tolist()
        
        # 对于任意两个笔触si、sj，如果si的综合特征优于sj，则si ≤ sj（si优先于sj绘制）
        for i, stroke_i in enumerate(stroke_ids):
            for j, stroke_j in enumerate(stroke_ids):
                if i != j:
                    score_i = comprehensive_scores[stroke_i]
                    score_j = comprehensive_scores[stroke_j]
                    
                    # 如果stroke_i的得分更高，则stroke_i应该优先绘制
                    if score_i > score_j:
                        partial_order.append((stroke_i, stroke_j))
                    
                    # 添加额外的规则约束
                    if self._should_precede(features_df.iloc[i], features_df.iloc[j]):
                        if (stroke_i, stroke_j) not in partial_order:
                            partial_order.append((stroke_i, stroke_j))
        
        if self.debug:
            print(f"构建偏序关系，共 {len(partial_order)} 个约束")
        
        return partial_order
    
    def _should_precede(self, stroke_i: pd.Series, stroke_j: pd.Series) -> bool:
        """判断笔触i是否应该优先于笔触j绘制"""
        # 规则1: 面积大的优先（主要结构）
        if stroke_i['area'] > stroke_j['area'] * 1.5:
            return True
        
        # 规则2: 位置显著性高的优先
        if stroke_i['position_saliency'] > stroke_j['position_saliency'] * 1.2:
            return True
        
        # 规则3: 厚度大的优先（粗笔触先画）
        if stroke_i['thickness'] > stroke_j['thickness'] * 1.3:
            return True
        
        # 规则4: 长度长的优先（主要笔画）
        if stroke_i['skeleton_length'] > stroke_j['skeleton_length'] * 1.4:
            return True
        
        return False
    
    def _build_hasse_graph(self, partial_order: List[Tuple[int, int]]) -> nx.DiGraph:
        """构建Hasse图"""
        # 创建有向图
        dag = nx.DiGraph()
        
        # 添加边
        dag.add_edges_from(partial_order)
        
        # 移除传递依赖边，得到Hasse图
        hasse_graph = self._reduce_to_hasse_diagram(dag)
        
        return hasse_graph
    
    def _reduce_to_hasse_diagram(self, dag: nx.DiGraph) -> nx.DiGraph:
        """将DAG简化为Hasse图（移除传递依赖边）"""
        hasse = dag.copy()
        
        # 找到所有传递边并移除
        edges_to_remove = []
        
        for u, v in dag.edges():
            # 检查是否存在u -> w -> v的路径
            for w in dag.nodes():
                if w != u and w != v:
                    if dag.has_edge(u, w) and dag.has_edge(w, v):
                        # 找到传递边u -> v
                        edges_to_remove.append((u, v))
                        break
        
        # 移除传递边
        for edge in edges_to_remove:
            if hasse.has_edge(*edge):
                hasse.remove_edge(*edge)
        
        if self.debug:
            print(f"移除 {len(edges_to_remove)} 条传递边")
        
        return hasse
    
    def _create_stages(self, features_df: pd.DataFrame, 
                      comprehensive_scores: Dict[int, float]) -> Dict[str, List[int]]:
        """创建三阶段分组"""
        # 根据综合得分排序
        sorted_strokes = sorted(comprehensive_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_strokes = len(sorted_strokes)
        
        # 三阶段分割点（可调整）
        stage1_end = int(total_strokes * 0.3)  # 前30%为主要结构
        stage2_end = int(total_strokes * 0.7)  # 中间40%为局部细节
        
        stages = {
            'main_structure': [stroke_id for stroke_id, _ in sorted_strokes[:stage1_end]],
            'local_details': [stroke_id for stroke_id, _ in sorted_strokes[stage1_end:stage2_end]],
            'decorative': [stroke_id for stroke_id, _ in sorted_strokes[stage2_end:]]
        }
        
        # 确保每个阶段至少有一个笔触
        if not stages['main_structure'] and sorted_strokes:
            stages['main_structure'] = [sorted_strokes[0][0]]
        
        if not stages['local_details'] and len(sorted_strokes) > 1:
            stages['local_details'] = [sorted_strokes[1][0]]
        
        if not stages['decorative'] and len(sorted_strokes) > 2:
            stages['decorative'] = [sorted_strokes[-1][0]]
        
        return stages
    
    def _get_topological_order(self) -> List[int]:
        """获取拓扑排序结果"""
        if self.hasse_graph is None:
            return []
        
        try:
            return list(nx.topological_sort(self.hasse_graph))
        except nx.NetworkXError:
            # 如果图中有环，使用近似拓扑排序
            if self.debug:
                print("警告: 图中存在环，使用近似排序")
            return list(self.hasse_graph.nodes())
    
    def get_stage_order(self) -> List[int]:
        """获取按阶段排序的笔触顺序"""
        if self.stages is None:
            return []
        
        order = []
        
        # 按阶段顺序添加笔触
        for stage_name in ['main_structure', 'local_details', 'decorative']:
            if stage_name in self.stages:
                order.extend(self.stages[stage_name])
        
        return order
    
    def get_precedence_constraints(self) -> List[Tuple[int, int]]:
        """获取优先级约束"""
        if self.hasse_graph is None:
            return []
        
        return list(self.hasse_graph.edges())
    
    def save_structure_analysis(self, hasse_graph: nx.DiGraph, stages: Dict, output_dir: Path):
        """保存结构分析结果"""
        if not self.debug:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存Hasse图信息
        graph_info = {
            'nodes': list(hasse_graph.nodes()),
            'edges': list(hasse_graph.edges()),
            'num_nodes': hasse_graph.number_of_nodes(),
            'num_edges': hasse_graph.number_of_edges()
        }
        
        with open(output_dir / "hasse_graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_info, f, indent=2, ensure_ascii=False)
        
        # 保存阶段信息
        with open(output_dir / "stages.json", 'w', encoding='utf-8') as f:
            json.dump(stages, f, indent=2, ensure_ascii=False)
        
        # 绘制Hasse图
        if hasse_graph.number_of_nodes() > 0:
            plt.figure(figsize=(12, 8))
            
            # 使用层次布局
            try:
                pos = nx.spring_layout(hasse_graph, k=1, iterations=50)
            except:
                pos = nx.random_layout(hasse_graph)
            
            # 根据阶段着色
            node_colors = []
            for node in hasse_graph.nodes():
                if node in stages.get('main_structure', []):
                    node_colors.append('red')
                elif node in stages.get('local_details', []):
                    node_colors.append('blue')
                elif node in stages.get('decorative', []):
                    node_colors.append('green')
                else:
                    node_colors.append('gray')
            
            nx.draw(hasse_graph, pos, 
                   node_color=node_colors,
                   node_size=300,
                   with_labels=True,
                   font_size=8,
                   font_weight='bold',
                   arrows=True,
                   arrowsize=20,
                   edge_color='gray',
                   alpha=0.7)
            
            plt.title('Hasse Diagram\n(Red: Main Structure, Blue: Local Details, Green: Decorative)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "hasse_diagram.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 绘制阶段分布
        stage_sizes = [len(strokes) for strokes in stages.values()]
        stage_names = list(stages.keys())
        
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green']
        plt.pie(stage_sizes, labels=stage_names, colors=colors, autopct='%1.1f%%')
        plt.title('Stroke Distribution by Stages')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_dir / "stage_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存拓扑排序结果
        topo_order = self._get_topological_order()
        with open(output_dir / "topological_order.json", 'w', encoding='utf-8') as f:
            json.dump(topo_order, f, indent=2, ensure_ascii=False)
        
        print(f"结构分析结果已保存至: {output_dir}")

# 测试代码
if __name__ == "__main__":
    from stroke_extraction import StrokeExtractor
    from feature_modeling import FeatureModeler
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python structure_construction.py <image_path>")
        sys.exit(1)
    
    # 先提取笔触和特征
    extractor = StrokeExtractor(debug=True)
    strokes = extractor.extract_strokes(sys.argv[1])
    
    modeler = FeatureModeler(debug=True)
    features = modeler.extract_features(strokes)
    
    # 构建结构
    constructor = StructureConstructor(debug=True)
    hasse_graph, stages = constructor.build_structure(features)
    
    print(f"\n结构构建结果:")
    print(f"Hasse图节点数: {hasse_graph.number_of_nodes()}")
    print(f"Hasse图边数: {hasse_graph.number_of_edges()}")
    print(f"阶段分组:")
    for stage_name, stroke_list in stages.items():
        print(f"  {stage_name}: {len(stroke_list)} 个笔触")
    
    # 获取拓扑排序
    topo_order = constructor._get_topological_order()
    print(f"拓扑排序: {topo_order[:10]}...")  # 显示前10个