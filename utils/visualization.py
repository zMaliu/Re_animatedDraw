# -*- coding: utf-8 -*-
"""
可视化工具

提供绘图、颜色处理和可视化功能
包括图表绘制、动画可视化、颜色空间转换等
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
import colorsys
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ColorUtils:
    """
    颜色工具类
    
    提供颜色空间转换和颜色处理功能
    """
    
    @staticmethod
    def rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        RGB转HSV
        
        Args:
            rgb (Tuple[int, int, int]): RGB颜色值 (0-255)
            
        Returns:
            Tuple[float, float, float]: HSV颜色值 (H: 0-360, S: 0-1, V: 0-1)
        """
        try:
            r, g, b = [x / 255.0 for x in rgb]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return (h * 360, s, v)
            
        except Exception as e:
            logging.error(f"Error converting RGB to HSV: {str(e)}")
            return (0.0, 0.0, 0.0)
    
    @staticmethod
    def hsv_to_rgb(hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        HSV转RGB
        
        Args:
            hsv (Tuple[float, float, float]): HSV颜色值 (H: 0-360, S: 0-1, V: 0-1)
            
        Returns:
            Tuple[int, int, int]: RGB颜色值 (0-255)
        """
        try:
            h, s, v = hsv[0] / 360.0, hsv[1], hsv[2]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return (int(r * 255), int(g * 255), int(b * 255))
            
        except Exception as e:
            logging.error(f"Error converting HSV to RGB: {str(e)}")
            return (0, 0, 0)
    
    @staticmethod
    def generate_color_palette(base_color: Tuple[int, int, int], 
                              num_colors: int = 5, 
                              variation_type: str = 'hue') -> List[Tuple[int, int, int]]:
        """
        生成调色板
        
        Args:
            base_color (Tuple[int, int, int]): 基础颜色
            num_colors (int): 颜色数量
            variation_type (str): 变化类型 ('hue', 'saturation', 'brightness')
            
        Returns:
            List[Tuple[int, int, int]]: 颜色列表
        """
        try:
            h, s, v = ColorUtils.rgb_to_hsv(base_color)
            colors = []
            
            for i in range(num_colors):
                if variation_type == 'hue':
                    new_h = (h + i * 360 / num_colors) % 360
                    new_color = ColorUtils.hsv_to_rgb((new_h, s, v))
                
                elif variation_type == 'saturation':
                    new_s = max(0.1, s - i * 0.8 / (num_colors - 1))
                    new_color = ColorUtils.hsv_to_rgb((h, new_s, v))
                
                elif variation_type == 'brightness':
                    new_v = max(0.1, v - i * 0.8 / (num_colors - 1))
                    new_color = ColorUtils.hsv_to_rgb((h, s, new_v))
                
                else:
                    new_color = base_color
                
                colors.append(new_color)
            
            return colors
            
        except Exception as e:
            logging.error(f"Error generating color palette: {str(e)}")
            return [base_color] * num_colors
    
    @staticmethod
    def interpolate_colors(color1: Tuple[int, int, int], 
                          color2: Tuple[int, int, int], 
                          steps: int = 10) -> List[Tuple[int, int, int]]:
        """
        颜色插值
        
        Args:
            color1 (Tuple[int, int, int]): 起始颜色
            color2 (Tuple[int, int, int]): 结束颜色
            steps (int): 插值步数
            
        Returns:
            List[Tuple[int, int, int]]: 插值颜色列表
        """
        try:
            colors = []
            
            for i in range(steps):
                t = i / (steps - 1) if steps > 1 else 0
                
                r = int(color1[0] + t * (color2[0] - color1[0]))
                g = int(color1[1] + t * (color2[1] - color1[1]))
                b = int(color1[2] + t * (color2[2] - color1[2]))
                
                colors.append((r, g, b))
            
            return colors
            
        except Exception as e:
            logging.error(f"Error interpolating colors: {str(e)}")
            return [color1, color2]
    
    @staticmethod
    def get_contrasting_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        获取对比色
        
        Args:
            color (Tuple[int, int, int]): 输入颜色
            
        Returns:
            Tuple[int, int, int]: 对比色
        """
        try:
            # 计算亮度
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            
            # 根据亮度选择黑色或白色作为对比色
            if luminance > 128:
                return (0, 0, 0)  # 黑色
            else:
                return (255, 255, 255)  # 白色
            
        except Exception as e:
            logging.error(f"Error getting contrasting color: {str(e)}")
            return (255, 255, 255)


class Visualizer:
    """
    可视化器
    
    提供各种可视化功能
    """
    
    def __init__(self, config=None):
        """
        初始化可视化器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 默认颜色
        self.default_colors = [
            (31, 119, 180),   # 蓝色
            (255, 127, 14),   # 橙色
            (44, 160, 44),    # 绿色
            (214, 39, 40),    # 红色
            (148, 103, 189),  # 紫色
            (140, 86, 75),    # 棕色
            (227, 119, 194),  # 粉色
            (127, 127, 127),  # 灰色
            (188, 189, 34),   # 橄榄色
            (23, 190, 207)    # 青色
        ]
    
    def plot_stroke_detection_result(self, image: np.ndarray, 
                                   strokes: List[Dict], 
                                   output_path: Optional[str] = None) -> bool:
        """
        可视化笔画检测结果
        
        Args:
            image (np.ndarray): 原始图像
            strokes (List[Dict]): 检测到的笔画列表
            output_path (Optional[str]): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 原始图像
            axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 笔画轮廓
            axes[0, 1].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            for i, stroke in enumerate(strokes):
                if 'contour' in stroke:
                    contour = stroke['contour']
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    if len(contour) > 0:
                        contour_points = contour.reshape(-1, 2)
                        axes[0, 1].plot(contour_points[:, 0], contour_points[:, 1], 
                                       color=color_norm, linewidth=2, label=f'Stroke {i+1}')
            
            axes[0, 1].set_title('Detected Strokes')
            axes[0, 1].axis('off')
            axes[0, 1].legend()
            
            # 笔画骨架
            axes[1, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
            for i, stroke in enumerate(strokes):
                if 'skeleton' in stroke:
                    skeleton = stroke['skeleton']
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    if len(skeleton) > 0:
                        axes[1, 0].plot(skeleton[:, 0], skeleton[:, 1], 
                                       color=color_norm, linewidth=2, marker='o', markersize=2)
            
            axes[1, 0].set_title('Stroke Skeletons')
            axes[1, 0].axis('off')
            
            # 笔画特征统计
            if strokes:
                features = ['area', 'perimeter', 'aspect_ratio', 'solidity']
                feature_data = {feature: [] for feature in features}
                
                for stroke in strokes:
                    for feature in features:
                        if feature in stroke:
                            feature_data[feature].append(stroke[feature])
                        else:
                            feature_data[feature].append(0)
                
                x_pos = np.arange(len(strokes))
                width = 0.2
                
                for i, feature in enumerate(features):
                    # 检查feature_data[feature]是否为空或全零
                    feature_values = feature_data[feature]
                    if isinstance(feature_values, np.ndarray):
                        has_data = feature_values.size > 0 and np.any(feature_values)
                    elif isinstance(feature_values, list):
                        has_data = len(feature_values) > 0 and any(v != 0 for v in feature_values)
                    else:
                        has_data = bool(feature_values)
                    
                    if has_data:
                        color = self.default_colors[i % len(self.default_colors)]
                        color_norm = tuple(c / 255.0 for c in color)
                        axes[1, 1].bar(x_pos + i * width, feature_data[feature], 
                                      width, label=feature, color=color_norm)
                
                axes[1, 1].set_xlabel('Stroke Index')
                axes[1, 1].set_ylabel('Feature Value')
                axes[1, 1].set_title('Stroke Features')
                axes[1, 1].set_xticks(x_pos + width * 1.5)
                axes[1, 1].set_xticklabels([f'S{i+1}' for i in range(len(strokes))])
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Stroke detection result saved to {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting stroke detection result: {str(e)}")
            return False
    
    def plot_stroke_ordering(self, strokes: List[Dict], 
                           ordering: List[int], 
                           output_path: Optional[str] = None) -> bool:
        """
        可视化笔画排序结果
        
        Args:
            strokes (List[Dict]): 笔画列表
            ordering (List[int]): 排序结果
            output_path (Optional[str]): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 原始顺序
            axes[0].set_title('Original Order')
            for i, stroke in enumerate(strokes):
                if 'skeleton' in stroke and len(stroke['skeleton']) > 0:
                    skeleton = stroke['skeleton']
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    axes[0].plot(skeleton[:, 0], skeleton[:, 1], 
                               color=color_norm, linewidth=3, label=f'Stroke {i+1}')
                    
                    # 标记起点
                    axes[0].plot(skeleton[0, 0], skeleton[0, 1], 
                               'o', color=color_norm, markersize=8)
            
            axes[0].legend()
            axes[0].set_aspect('equal')
            axes[0].invert_yaxis()
            
            # 优化后的顺序
            axes[1].set_title('Optimized Order')
            for i, stroke_idx in enumerate(ordering):
                if stroke_idx < len(strokes):
                    stroke = strokes[stroke_idx]
                    if 'skeleton' in stroke and len(stroke['skeleton']) > 0:
                        skeleton = stroke['skeleton']
                        color = self.default_colors[i % len(self.default_colors)]
                        color_norm = tuple(c / 255.0 for c in color)
                        
                        axes[1].plot(skeleton[:, 0], skeleton[:, 1], 
                                   color=color_norm, linewidth=3, label=f'Order {i+1}')
                        
                        # 标记起点
                        axes[1].plot(skeleton[0, 0], skeleton[0, 1], 
                                   'o', color=color_norm, markersize=8)
                        
                        # 添加顺序标号
                        centroid = np.mean(skeleton, axis=0)
                        axes[1].text(centroid[0], centroid[1], str(i+1), 
                                   fontsize=12, ha='center', va='center', 
                                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
            
            axes[1].legend()
            axes[1].set_aspect('equal')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Stroke ordering result saved to {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting stroke ordering: {str(e)}")
            return False
    
    def plot_animation_timeline(self, animation_data: Dict, 
                              output_path: Optional[str] = None) -> bool:
        """
        可视化动画时间轴
        
        Args:
            animation_data (Dict): 动画数据
            output_path (Optional[str]): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # 提取时间轴数据
            timestamps = animation_data.get('timestamps', [])
            stroke_durations = animation_data.get('stroke_durations', [])
            pressure_curves = animation_data.get('pressure_curves', [])
            speed_curves = animation_data.get('speed_curves', [])
            
            # 笔画时间轴
            axes[0].set_title('Stroke Timeline')
            current_time = 0
            for i, duration in enumerate(stroke_durations):
                color = self.default_colors[i % len(self.default_colors)]
                color_norm = tuple(c / 255.0 for c in color)
                
                axes[0].barh(0, duration, left=current_time, height=0.5, 
                           color=color_norm, alpha=0.7, label=f'Stroke {i+1}')
                
                # 添加标签
                axes[0].text(current_time + duration/2, 0, f'S{i+1}', 
                           ha='center', va='center', fontweight='bold')
                
                current_time += duration
            
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Strokes')
            axes[0].set_ylim(-0.5, 0.5)
            
            # 压力曲线
            axes[1].set_title('Pressure Curves')
            for i, pressure_curve in enumerate(pressure_curves):
                if len(pressure_curve) > 0:
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    time_points = np.linspace(0, stroke_durations[i] if i < len(stroke_durations) else 1, 
                                            len(pressure_curve))
                    axes[1].plot(time_points, pressure_curve, color=color_norm, 
                               linewidth=2, label=f'Stroke {i+1}')
            
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Pressure')
            axes[1].legend()
            
            # 速度曲线
            axes[2].set_title('Speed Curves')
            for i, speed_curve in enumerate(speed_curves):
                if len(speed_curve) > 0:
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    time_points = np.linspace(0, stroke_durations[i] if i < len(stroke_durations) else 1, 
                                            len(speed_curve))
                    axes[2].plot(time_points, speed_curve, color=color_norm, 
                               linewidth=2, label=f'Stroke {i+1}')
            
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Speed')
            axes[2].legend()
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Animation timeline saved to {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting animation timeline: {str(e)}")
            return False
    
    def create_stroke_animation(self, strokes: List[Dict], 
                              canvas_size: Tuple[int, int] = (800, 600),
                              fps: int = 30,
                              output_path: Optional[str] = None) -> bool:
        """
        创建笔画动画
        
        Args:
            strokes (List[Dict]): 笔画列表
            canvas_size (Tuple[int, int]): 画布大小
            fps (int): 帧率
            output_path (Optional[str]): 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            fig, ax = plt.subplots(figsize=(canvas_size[0]/100, canvas_size[1]/100))
            ax.set_xlim(0, canvas_size[0])
            ax.set_ylim(0, canvas_size[1])
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 准备动画数据
            all_points = []
            stroke_colors = []
            
            for i, stroke in enumerate(strokes):
                if 'skeleton' in stroke and len(stroke['skeleton']) > 0:
                    skeleton = stroke['skeleton']
                    color = self.default_colors[i % len(self.default_colors)]
                    color_norm = tuple(c / 255.0 for c in color)
                    
                    all_points.extend(skeleton)
                    stroke_colors.extend([color_norm] * len(skeleton))
            
            # 动画函数
            def animate(frame):
                ax.clear()
                ax.set_xlim(0, canvas_size[0])
                ax.set_ylim(0, canvas_size[1])
                ax.set_aspect('equal')
                ax.axis('off')
                
                # 绘制到当前帧的所有点
                if frame < len(all_points):
                    points_to_draw = all_points[:frame+1]
                    colors_to_draw = stroke_colors[:frame+1]
                    
                    for i, (point, color) in enumerate(zip(points_to_draw, colors_to_draw)):
                        ax.plot(point[0], point[1], 'o', color=color, markersize=3)
                
                return []
            
            # 创建动画
            anim = animation.FuncAnimation(fig, animate, frames=len(all_points), 
                                         interval=1000/fps, blit=False, repeat=True)
            
            if output_path:
                # 保存动画
                if output_path.endswith('.gif'):
                    anim.save(output_path, writer='pillow', fps=fps)
                elif output_path.endswith('.mp4'):
                    anim.save(output_path, writer='ffmpeg', fps=fps)
                else:
                    anim.save(output_path + '.gif', writer='pillow', fps=fps)
                
                self.logger.info(f"Stroke animation saved to {output_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating stroke animation: {str(e)}")
            return False


class PlotManager:
    """
    绘图管理器
    
    管理多个图表和子图
    """
    
    def __init__(self, config=None):
        """
        初始化绘图管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.figures = {}
        self.current_figure = None
    
    def create_figure(self, figure_id: str, figsize: Tuple[int, int] = (10, 8), 
                     nrows: int = 1, ncols: int = 1) -> bool:
        """
        创建图表
        
        Args:
            figure_id (str): 图表ID
            figsize (Tuple[int, int]): 图表大小
            nrows (int): 行数
            ncols (int): 列数
            
        Returns:
            bool: 是否成功
        """
        try:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            
            self.figures[figure_id] = {
                'figure': fig,
                'axes': axes,
                'nrows': nrows,
                'ncols': ncols
            }
            
            self.current_figure = figure_id
            self.logger.info(f"Figure '{figure_id}' created")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating figure: {str(e)}")
            return False
    
    def select_figure(self, figure_id: str) -> bool:
        """
        选择当前图表
        
        Args:
            figure_id (str): 图表ID
            
        Returns:
            bool: 是否成功
        """
        try:
            if figure_id in self.figures:
                self.current_figure = figure_id
                plt.figure(self.figures[figure_id]['figure'].number)
                return True
            else:
                self.logger.warning(f"Figure '{figure_id}' not found")
                return False
            
        except Exception as e:
            self.logger.error(f"Error selecting figure: {str(e)}")
            return False
    
    def plot_data(self, data: Dict[str, Any], plot_type: str = 'line', 
                 subplot_index: Tuple[int, int] = (0, 0), **kwargs) -> bool:
        """
        绘制数据
        
        Args:
            data (Dict[str, Any]): 数据字典
            plot_type (str): 绘图类型
            subplot_index (Tuple[int, int]): 子图索引
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功
        """
        try:
            if not self.current_figure or self.current_figure not in self.figures:
                self.logger.warning("No current figure selected")
                return False
            
            figure_info = self.figures[self.current_figure]
            axes = figure_info['axes']
            
            # 获取子图
            if figure_info['nrows'] == 1 and figure_info['ncols'] == 1:
                ax = axes
            else:
                ax = axes[subplot_index[0], subplot_index[1]]
            
            # 绘制数据
            if plot_type == 'line':
                x = data.get('x', range(len(data.get('y', []))))
                y = data.get('y', [])
                ax.plot(x, y, **kwargs)
            
            elif plot_type == 'scatter':
                x = data.get('x', [])
                y = data.get('y', [])
                ax.scatter(x, y, **kwargs)
            
            elif plot_type == 'bar':
                x = data.get('x', range(len(data.get('y', []))))
                y = data.get('y', [])
                ax.bar(x, y, **kwargs)
            
            elif plot_type == 'histogram':
                values = data.get('values', [])
                ax.hist(values, **kwargs)
            
            elif plot_type == 'heatmap':
                matrix = data.get('matrix', [])
                im = ax.imshow(matrix, **kwargs)
                plt.colorbar(im, ax=ax)
            
            else:
                self.logger.warning(f"Unknown plot type: {plot_type}")
                return False
            
            # 设置标题和标签
            if 'title' in data:
                ax.set_title(data['title'])
            if 'xlabel' in data:
                ax.set_xlabel(data['xlabel'])
            if 'ylabel' in data:
                ax.set_ylabel(data['ylabel'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting data: {str(e)}")
            return False
    
    def save_figure(self, figure_id: str, output_path: str, **kwargs) -> bool:
        """
        保存图表
        
        Args:
            figure_id (str): 图表ID
            output_path (str): 输出路径
            **kwargs: 其他参数
            
        Returns:
            bool: 是否成功
        """
        try:
            if figure_id not in self.figures:
                self.logger.warning(f"Figure '{figure_id}' not found")
                return False
            
            figure = self.figures[figure_id]['figure']
            figure.savefig(output_path, **kwargs)
            
            self.logger.info(f"Figure '{figure_id}' saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving figure: {str(e)}")
            return False
    
    def show_figure(self, figure_id: Optional[str] = None) -> bool:
        """
        显示图表
        
        Args:
            figure_id (Optional[str]): 图表ID，如果为None则显示当前图表
            
        Returns:
            bool: 是否成功
        """
        try:
            if figure_id is None:
                figure_id = self.current_figure
            
            if figure_id and figure_id in self.figures:
                plt.figure(self.figures[figure_id]['figure'].number)
                plt.show()
                return True
            else:
                self.logger.warning(f"Figure '{figure_id}' not found")
                return False
            
        except Exception as e:
            self.logger.error(f"Error showing figure: {str(e)}")
            return False
    
    def close_figure(self, figure_id: str) -> bool:
        """
        关闭图表
        
        Args:
            figure_id (str): 图表ID
            
        Returns:
            bool: 是否成功
        """
        try:
            if figure_id in self.figures:
                plt.close(self.figures[figure_id]['figure'])
                del self.figures[figure_id]
                
                if self.current_figure == figure_id:
                    self.current_figure = None
                
                self.logger.info(f"Figure '{figure_id}' closed")
                return True
            else:
                self.logger.warning(f"Figure '{figure_id}' not found")
                return False
            
        except Exception as e:
            self.logger.error(f"Error closing figure: {str(e)}")
            return False
    
    def close_all_figures(self) -> bool:
        """
        关闭所有图表
        
        Returns:
            bool: 是否成功
        """
        try:
            plt.close('all')
            self.figures.clear()
            self.current_figure = None
            
            self.logger.info("All figures closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing all figures: {str(e)}")
            return False