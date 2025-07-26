#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化工具模块
用于调试和结果展示

主要功能:
1. 笔触可视化
2. 特征分析可视化
3. 结构图可视化
4. 排序结果可视化
5. 动画预览
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

from .stroke_extraction import Stroke

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 颜色配置
        self.colors = {
            'stroke': '#2E86AB',
            'skeleton': '#A23B72',
            'centroid': '#F18F01',
            'bbox': '#C73E1D',
            'main_structure': '#2E8B57',
            'local_details': '#4169E1',
            'decorative': '#FF6347'
        }
    
    def visualize_strokes(self, strokes: List[Stroke], output_path: Path, 
                         show_details: bool = True):
        """可视化笔触提取结果"""
        if not strokes:
            return
        
        # 计算画布大小
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for stroke in strokes:
            x, y, w, h = stroke.bbox
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'笔触提取结果 (共{len(strokes)}个笔触)', fontsize=16)
        
        # 1. 原始笔触轮廓
        ax1 = axes[0, 0]
        ax1.set_title('笔触轮廓')
        ax1.set_xlim(min_x - 50, max_x + 50)
        ax1.set_ylim(max_y + 50, min_y - 50)  # 翻转Y轴
        
        for i, stroke in enumerate(strokes):
            contour = stroke.contour.reshape(-1, 2)
            ax1.plot(contour[:, 0], contour[:, 1], 
                    color=plt.cm.tab20(i % 20), linewidth=2, label=f'笔触{stroke.id}')
        
        if len(strokes) <= 10:
            ax1.legend()
        
        # 2. 笔触掩码
        ax2 = axes[0, 1]
        ax2.set_title('笔触掩码')
        
        # 创建合成掩码
        canvas_width = int(max_x - min_x + 100)
        canvas_height = int(max_y - min_y + 100)
        combined_mask = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        for i, stroke in enumerate(strokes):
            x, y, w, h = stroke.bbox
            canvas_x = int(x - min_x + 50)
            canvas_y = int(y - min_y + 50)
            
            color = np.array(plt.cm.tab20(i % 20)[:3]) * 255
            mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
            mask_colored[stroke.mask > 0] = color
            
            # 复制到画布
            end_y = min(canvas_y + h, canvas_height)
            end_x = min(canvas_x + w, canvas_width)
            mask_h = end_y - canvas_y
            mask_w = end_x - canvas_x
            
            if mask_h > 0 and mask_w > 0:
                combined_mask[canvas_y:end_y, canvas_x:end_x] = mask_colored[:mask_h, :mask_w]
        
        ax2.imshow(combined_mask)
        ax2.axis('off')
        
        # 3. 骨架和质心
        if show_details:
            ax3 = axes[1, 0]
            ax3.set_title('骨架和质心')
            ax3.set_xlim(min_x - 50, max_x + 50)
            ax3.set_ylim(max_y + 50, min_y - 50)
            
            for i, stroke in enumerate(strokes):
                # 绘制轮廓
                contour = stroke.contour.reshape(-1, 2)
                ax3.plot(contour[:, 0], contour[:, 1], 
                        color=plt.cm.tab20(i % 20), alpha=0.3, linewidth=1)
                
                # 绘制质心
                ax3.plot(stroke.centroid[0], stroke.centroid[1], 
                        'o', color=self.colors['centroid'], markersize=8)
                
                # 绘制骨架点
                skeleton_points = stroke.get_skeleton_points()
                if skeleton_points:
                    skeleton_x = [p[0] for p in skeleton_points]
                    skeleton_y = [p[1] for p in skeleton_points]
                    ax3.plot(skeleton_x, skeleton_y, 
                            color=self.colors['skeleton'], linewidth=2, marker='.')
        
        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.set_title('笔触统计')
        
        # 面积分布
        areas = [stroke.area for stroke in strokes]
        ax4.hist(areas, bins=min(20, len(strokes)), alpha=0.7, color=self.colors['stroke'])
        ax4.set_xlabel('面积')
        ax4.set_ylabel('数量')
        
        plt.tight_layout()
        
        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.debug:
            print(f"笔触可视化已保存至: {output_path}")
    
    def visualize_features(self, features: Dict, output_path: Path):
        """可视化特征分析结果"""
        if not features:
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特征分析结果', fontsize=16)
        
        stroke_ids = list(features.keys())
        
        # 1. 几何特征
        ax1 = axes[0, 0]
        ax1.set_title('几何特征')
        
        skeleton_lengths = [features[sid]['geometric']['skeleton_length'] for sid in stroke_ids]
        scales = [features[sid]['geometric']['scale'] for sid in stroke_ids]
        
        scatter = ax1.scatter(skeleton_lengths, scales, 
                            c=range(len(stroke_ids)), cmap='viridis', alpha=0.7)
        ax1.set_xlabel('骨架长度')
        ax1.set_ylabel('尺度')
        plt.colorbar(scatter, ax=ax1, label='笔触ID')
        
        # 2. 形状特征
        ax2 = axes[0, 1]
        ax2.set_title('形状特征')
        
        circularities = [features[sid]['shape']['circularity'] for sid in stroke_ids]
        aspect_ratios = [features[sid]['shape']['aspect_ratio'] for sid in stroke_ids]
        
        ax2.scatter(circularities, aspect_ratios, alpha=0.7, color=self.colors['stroke'])
        ax2.set_xlabel('圆形度')
        ax2.set_ylabel('长宽比')
        
        # 3. 墨水特征
        ax3 = axes[0, 2]
        ax3.set_title('墨水特征')
        
        wetness = [features[sid]['ink']['wetness'] for sid in stroke_ids]
        thickness = [features[sid]['ink']['thickness'] for sid in stroke_ids]
        
        ax3.scatter(wetness, thickness, alpha=0.7, color=self.colors['skeleton'])
        ax3.set_xlabel('湿润度')
        ax3.set_ylabel('厚度')
        
        # 4. 位置显著性
        ax4 = axes[1, 0]
        ax4.set_title('位置显著性分布')
        
        saliencies = [features[sid]['position']['saliency'] for sid in stroke_ids]
        ax4.hist(saliencies, bins=min(20, len(stroke_ids)), 
                alpha=0.7, color=self.colors['centroid'])
        ax4.set_xlabel('显著性')
        ax4.set_ylabel('数量')
        
        # 5. 综合得分
        ax5 = axes[1, 1]
        ax5.set_title('综合得分排序')
        
        scores = [features[sid]['comprehensive_score'] for sid in stroke_ids]
        sorted_indices = np.argsort(scores)[::-1]
        
        ax5.bar(range(len(scores)), [scores[i] for i in sorted_indices], 
               color=self.colors['stroke'], alpha=0.7)
        ax5.set_xlabel('笔触排序')
        ax5.set_ylabel('综合得分')
        
        # 6. 特征相关性
        ax6 = axes[1, 2]
        ax6.set_title('特征相关性热图')
        
        # 构建特征矩阵
        feature_matrix = []
        feature_names = ['骨架长度', '尺度', '圆形度', '长宽比', '湿润度', '厚度', '显著性', '综合得分']
        
        for sid in stroke_ids:
            f = features[sid]
            row = [
                f['geometric']['skeleton_length'],
                f['geometric']['scale'],
                f['shape']['circularity'],
                f['shape']['aspect_ratio'],
                f['ink']['wetness'],
                f['ink']['thickness'],
                f['position']['saliency'],
                f['comprehensive_score']
            ]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        im = ax6.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(feature_names)))
        ax6.set_yticks(range(len(feature_names)))
        ax6.set_xticklabels(feature_names, rotation=45, ha='right')
        ax6.set_yticklabels(feature_names)
        
        # 添加数值标注
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax6, label='相关系数')
        
        plt.tight_layout()
        
        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.debug:
            print(f"特征可视化已保存至: {output_path}")
    
    def visualize_structure(self, hasse_graph: nx.DiGraph, stages: Dict, 
                          features: Dict, output_path: Path):
        """可视化结构图"""
        if not hasse_graph.nodes():
            return
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle('结构分析结果', fontsize=16)
        
        # 1. Hasse图可视化
        ax1 = axes[0]
        ax1.set_title('Hasse图结构')
        
        # 使用层次布局
        pos = self._create_hierarchical_layout(hasse_graph, stages)
        
        # 绘制节点
        stage_colors = {
            'main_structure': self.colors['main_structure'],
            'local_details': self.colors['local_details'],
            'decorative': self.colors['decorative']
        }
        
        for stage, stroke_ids in stages.items():
            node_list = [sid for sid in stroke_ids if sid in hasse_graph.nodes()]
            if node_list:
                nx.draw_networkx_nodes(hasse_graph, pos, nodelist=node_list,
                                     node_color=stage_colors.get(stage, 'gray'),
                                     node_size=300, alpha=0.8, ax=ax1)
        
        # 绘制边
        nx.draw_networkx_edges(hasse_graph, pos, edge_color='gray', 
                             arrows=True, arrowsize=20, alpha=0.6, ax=ax1)
        
        # 绘制标签
        nx.draw_networkx_labels(hasse_graph, pos, font_size=8, ax=ax1)
        
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # 添加图例
        legend_elements = [patches.Patch(color=color, label=stage) 
                          for stage, color in stage_colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 2. 阶段分布
        ax2 = axes[1]
        ax2.set_title('阶段分布统计')
        
        stage_names = list(stages.keys())
        stage_counts = [len(stages[stage]) for stage in stage_names]
        stage_labels = ['主体结构', '局部细节', '装饰元素']
        
        bars = ax2.bar(stage_labels, stage_counts, 
                      color=[stage_colors[stage] for stage in stage_names],
                      alpha=0.8)
        
        # 添加数值标签
        for bar, count in zip(bars, stage_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        ax2.set_ylabel('笔触数量')
        ax2.set_xlabel('绘制阶段')
        
        plt.tight_layout()
        
        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.debug:
            print(f"结构可视化已保存至: {output_path}")
    
    def visualize_ordering(self, optimized_order: List[Tuple[int, str]], 
                         features: Dict, stages: Dict, output_path: Path):
        """可视化排序结果"""
        if not optimized_order:
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('笔触排序结果', fontsize=16)
        
        stroke_ids = [item[0] for item in optimized_order]
        directions = [item[1] for item in optimized_order]
        
        # 1. 排序序列
        ax1 = axes[0, 0]
        ax1.set_title('笔触绘制顺序')
        
        # 按阶段着色
        colors = []
        stage_colors = {
            'main_structure': self.colors['main_structure'],
            'local_details': self.colors['local_details'],
            'decorative': self.colors['decorative']
        }
        
        for sid in stroke_ids:
            for stage, stroke_list in stages.items():
                if sid in stroke_list:
                    colors.append(stage_colors[stage])
                    break
            else:
                colors.append('gray')
        
        ax1.scatter(range(len(stroke_ids)), stroke_ids, c=colors, alpha=0.7)
        ax1.set_xlabel('绘制顺序')
        ax1.set_ylabel('笔触ID')
        
        # 2. 方向分布
        ax2 = axes[0, 1]
        ax2.set_title('绘制方向分布')
        
        direction_counts = {'forward': directions.count('forward'), 
                          'reverse': directions.count('reverse')}
        
        ax2.pie(direction_counts.values(), labels=['正向', '反向'], 
               autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        
        # 3. 综合得分变化
        ax3 = axes[1, 0]
        ax3.set_title('综合得分变化趋势')
        
        scores = [features[sid]['comprehensive_score'] for sid in stroke_ids]
        ax3.plot(range(len(scores)), scores, 'o-', color=self.colors['stroke'], alpha=0.7)
        ax3.set_xlabel('绘制顺序')
        ax3.set_ylabel('综合得分')
        
        # 添加趋势线
        z = np.polyfit(range(len(scores)), scores, 1)
        p = np.poly1d(z)
        ax3.plot(range(len(scores)), p(range(len(scores))), "r--", alpha=0.8)
        
        # 4. 阶段时序分布
        ax4 = axes[1, 1]
        ax4.set_title('阶段时序分布')
        
        stage_positions = defaultdict(list)
        for i, sid in enumerate(stroke_ids):
            for stage, stroke_list in stages.items():
                if sid in stroke_list:
                    stage_positions[stage].append(i)
                    break
        
        for stage, positions in stage_positions.items():
            if positions:
                ax4.scatter(positions, [stage] * len(positions), 
                          c=stage_colors[stage], alpha=0.7, s=50)
        
        ax4.set_xlabel('绘制顺序')
        ax4.set_ylabel('绘制阶段')
        ax4.set_yticks(list(stage_positions.keys()))
        ax4.set_yticklabels(['主体结构', '局部细节', '装饰元素'])
        
        plt.tight_layout()
        
        # 保存图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.debug:
            print(f"排序可视化已保存至: {output_path}")
    
    def _create_hierarchical_layout(self, graph: nx.DiGraph, stages: Dict) -> Dict:
        """创建层次化布局"""
        pos = {}
        
        # 按阶段分层
        stage_order = ['main_structure', 'local_details', 'decorative']
        y_positions = {stage: i for i, stage in enumerate(stage_order)}
        
        for stage, stroke_ids in stages.items():
            y = y_positions.get(stage, 0)
            
            # 在同一层内水平排列
            stroke_list = [sid for sid in stroke_ids if sid in graph.nodes()]
            if stroke_list:
                x_positions = np.linspace(-1, 1, len(stroke_list))
                for i, sid in enumerate(stroke_list):
                    pos[sid] = (x_positions[i], y)
        
        return pos
    
    def create_summary_report(self, strokes: List[Stroke], features: Dict, 
                            stages: Dict, optimized_order: List[Tuple[int, str]], 
                            output_path: Path):
        """创建总结报告"""
        report = {
            'summary': {
                'total_strokes': len(strokes),
                'total_features': len(features),
                'stages': {stage: len(stroke_list) for stage, stroke_list in stages.items()},
                'optimization_length': len(optimized_order)
            },
            'statistics': {
                'stroke_areas': [stroke.area for stroke in strokes],
                'stroke_perimeters': [stroke.perimeter for stroke in strokes],
                'comprehensive_scores': [features[sid]['comprehensive_score'] 
                                       for sid in features.keys()],
                'direction_distribution': {
                    'forward': sum(1 for _, direction in optimized_order if direction == 'forward'),
                    'reverse': sum(1 for _, direction in optimized_order if direction == 'reverse')
                }
            },
            'processing_info': {
                'timestamp': str(Path().cwd()),
                'debug_mode': self.debug
            }
        }
        
        # 保存JSON报告
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        if self.debug:
            print(f"总结报告已保存至: {output_path}")
        
        return report

# 测试代码
if __name__ == "__main__":
    # 这里可以添加测试代码
    pass