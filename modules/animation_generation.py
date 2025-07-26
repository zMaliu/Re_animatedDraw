#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块5: 动态绘制与动画生成
根据优化后的笔触顺序和方向，逐步绘制笔触，生成动画

主要功能:
1. 笔触绘制算法（洪水填充策略）
2. 动画合成
3. 绘制速度控制
4. 视频输出
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from tqdm import tqdm
import time

from .stroke_extraction import Stroke

class AnimationGenerator:
    """动画生成器"""
    
    def __init__(self, fps: int = 24, debug: bool = False):
        self.fps = fps
        self.debug = debug
        self.canvas = None
        self.canvas_size = None
        self.video_writer = None
        
        # 绘制参数
        self.base_speed = 5  # 基础绘制速度（像素/帧）
        self.highlight_color = (0, 255, 0)  # 当前笔触高亮颜色
        self.drawn_color = (128, 128, 128)  # 已绘制笔触颜色
        self.undrawn_alpha = 0.3  # 未绘制笔触透明度
        
    def generate_animation(self, strokes: List[Stroke], 
                         optimized_order: List[Tuple[int, str]], 
                         output_path: Path) -> Path:
        """生成动画"""
        if not strokes or not optimized_order:
            raise ValueError("笔触列表或排序结果为空")
        
        # 初始化画布
        self._initialize_canvas(strokes)
        
        # 初始化视频写入器
        self._initialize_video_writer(output_path)
        
        try:
            # 生成动画帧
            self._generate_animation_frames(strokes, optimized_order)
            
        finally:
            # 释放资源
            if self.video_writer:
                self.video_writer.release()
        
        if self.debug:
            print(f"动画已保存至: {output_path}")
        
        return output_path
    
    def _initialize_canvas(self, strokes: List[Stroke]):
        """初始化画布"""
        # 计算所有笔触的边界框
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for stroke in strokes:
            x, y, w, h = stroke.bbox
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        # 添加边距
        margin = 50
        canvas_width = int(max_x - min_x + 2 * margin)
        canvas_height = int(max_y - min_y + 2 * margin)
        
        self.canvas_size = (canvas_width, canvas_height)
        self.canvas_offset = (int(min_x - margin), int(min_y - margin))
        
        # 创建白色画布
        self.canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        if self.debug:
            print(f"画布尺寸: {canvas_width} x {canvas_height}")
            print(f"画布偏移: {self.canvas_offset}")
    
    def _initialize_video_writer(self, output_path: Path):
        """初始化视频写入器"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用MP4编码
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, self.canvas_size
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
    
    def _generate_animation_frames(self, strokes: List[Stroke], 
                                 optimized_order: List[Tuple[int, str]]):
        """生成动画帧"""
        # 创建笔触ID到笔触对象的映射
        stroke_dict = {stroke.id: stroke for stroke in strokes}
        
        # 初始状态：显示所有未绘制的笔触（半透明）
        current_canvas = self.canvas.copy()
        self._draw_all_strokes_preview(current_canvas, strokes)
        
        # 写入初始帧（持续1秒）
        for _ in range(self.fps):
            self.video_writer.write(current_canvas)
        
        # 逐个绘制笔触
        drawn_strokes = set()
        
        progress_bar = tqdm(optimized_order, desc="生成动画帧") if self.debug else optimized_order
        
        for stroke_id, direction in progress_bar:
            if stroke_id not in stroke_dict:
                continue
            
            stroke = stroke_dict[stroke_id]
            
            # 绘制当前笔触的动画
            self._animate_single_stroke(current_canvas, stroke, direction, drawn_strokes)
            
            # 将当前笔触标记为已绘制
            drawn_strokes.add(stroke_id)
        
        # 最终帧（持续2秒）
        final_canvas = self._create_final_frame(strokes)
        for _ in range(self.fps * 2):
            self.video_writer.write(final_canvas)
    
    def _draw_all_strokes_preview(self, canvas: np.ndarray, strokes: List[Stroke]):
        """绘制所有笔触的预览（半透明）"""
        overlay = canvas.copy()
        
        for stroke in strokes:
            self._draw_stroke_on_canvas(overlay, stroke, color=(200, 200, 200))
        
        # 混合透明效果
        cv2.addWeighted(overlay, self.undrawn_alpha, canvas, 1 - self.undrawn_alpha, 0, canvas)
    
    def _animate_single_stroke(self, canvas: np.ndarray, stroke: Stroke, 
                             direction: str, drawn_strokes: set):
        """为单个笔触生成动画"""
        # 获取笔触的绘制路径
        drawing_path = self._get_stroke_drawing_path(stroke, direction)
        
        if not drawing_path:
            return
        
        # 计算绘制速度（根据笔触宽度调整）
        stroke_width = self._estimate_stroke_width(stroke)
        speed = max(1, int(self.base_speed * (1.0 / max(1, stroke_width / 10))))
        
        # 逐步绘制笔触
        drawn_pixels = set()
        
        for i in range(0, len(drawing_path), speed):
            # 更新画布
            frame_canvas = canvas.copy()
            
            # 绘制已完成的笔触
            self._draw_completed_strokes(frame_canvas, drawn_strokes)
            
            # 绘制当前笔触的已绘制部分
            current_pixels = drawing_path[:i+speed]
            self._draw_pixels_on_canvas(frame_canvas, current_pixels, self.highlight_color)
            
            # 高亮当前绘制点
            if i < len(drawing_path):
                current_point = drawing_path[i]
                cv2.circle(frame_canvas, current_point, 3, (0, 0, 255), -1)
            
            # 写入帧
            self.video_writer.write(frame_canvas)
        
        # 最终状态：笔触完全绘制
        final_frame = canvas.copy()
        self._draw_completed_strokes(final_frame, drawn_strokes)
        self._draw_stroke_on_canvas(final_frame, stroke, self.drawn_color)
        
        # 更新主画布
        canvas[:] = final_frame
        
        # 写入几帧以显示完成状态
        for _ in range(3):
            self.video_writer.write(canvas)
    
    def _get_stroke_drawing_path(self, stroke: Stroke, direction: str) -> List[Tuple[int, int]]:
        """获取笔触的绘制路径（洪水填充策略）"""
        # 获取笔触骨架点
        skeleton_points = stroke.get_skeleton_points()
        
        if not skeleton_points:
            # 如果没有骨架点，使用质心
            skeleton_points = [stroke.centroid]
        
        # 确定起点和终点
        if len(skeleton_points) >= 2:
            if direction == 'forward':
                start_point = skeleton_points[0]
                end_point = skeleton_points[-1]
            else:  # reverse
                start_point = skeleton_points[-1]
                end_point = skeleton_points[0]
        else:
            start_point = end_point = skeleton_points[0]
        
        # 调整坐标到画布坐标系
        start_point = self._adjust_coordinates(start_point)
        end_point = self._adjust_coordinates(end_point)
        
        # 生成绘制路径
        path = self._generate_flood_fill_path(stroke, start_point, end_point)
        
        return path
    
    def _generate_flood_fill_path(self, stroke: Stroke, start_point: Tuple[int, int], 
                                end_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """生成洪水填充路径"""
        # 获取笔触掩码
        mask = stroke.mask.copy()
        
        # 调整掩码到画布坐标系
        x, y, w, h = stroke.bbox
        canvas_x = x - self.canvas_offset[0]
        canvas_y = y - self.canvas_offset[1]
        
        # 获取笔触内的所有像素点
        stroke_pixels = []
        mask_coords = np.where(mask > 0)
        
        for py, px in zip(mask_coords[0], mask_coords[1]):
            canvas_px = canvas_x + px
            canvas_py = canvas_y + py
            
            # 确保坐标在画布范围内
            if (0 <= canvas_px < self.canvas_size[0] and 
                0 <= canvas_py < self.canvas_size[1]):
                stroke_pixels.append((canvas_px, canvas_py))
        
        if not stroke_pixels:
            return []
        
        # 从起点开始，按距离排序像素点
        def distance_to_start(point):
            return np.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
        
        # 简化的路径生成：按到起点的距离排序
        stroke_pixels.sort(key=distance_to_start)
        
        return stroke_pixels
    
    def _estimate_stroke_width(self, stroke: Stroke) -> float:
        """估算笔触宽度"""
        # 使用面积和周长估算平均宽度
        if stroke.perimeter > 0:
            return stroke.area / stroke.perimeter * 4
        else:
            return 5.0  # 默认宽度
    
    def _adjust_coordinates(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """调整坐标到画布坐标系"""
        x, y = point
        canvas_x = x - self.canvas_offset[0]
        canvas_y = y - self.canvas_offset[1]
        
        # 确保坐标在画布范围内
        canvas_x = np.clip(canvas_x, 0, self.canvas_size[0] - 1)
        canvas_y = np.clip(canvas_y, 0, self.canvas_size[1] - 1)
        
        return (int(canvas_x), int(canvas_y))
    
    def _draw_completed_strokes(self, canvas: np.ndarray, drawn_strokes: set):
        """绘制已完成的笔触"""
        # 这里可以根据需要实现已绘制笔触的显示
        pass
    
    def _draw_stroke_on_canvas(self, canvas: np.ndarray, stroke: Stroke, 
                             color: Tuple[int, int, int]):
        """在画布上绘制笔触"""
        # 获取笔触轮廓
        contour = stroke.contour.copy()
        
        # 调整轮廓坐标
        contour[:, 0, 0] -= self.canvas_offset[0]
        contour[:, 0, 1] -= self.canvas_offset[1]
        
        # 绘制填充轮廓
        cv2.fillPoly(canvas, [contour], color)
    
    def _draw_pixels_on_canvas(self, canvas: np.ndarray, pixels: List[Tuple[int, int]], 
                             color: Tuple[int, int, int]):
        """在画布上绘制像素点"""
        for x, y in pixels:
            if (0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]):
                canvas[y, x] = color
    
    def _create_final_frame(self, strokes: List[Stroke]) -> np.ndarray:
        """创建最终帧"""
        final_canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255
        
        # 绘制所有笔触
        for stroke in strokes:
            self._draw_stroke_on_canvas(final_canvas, stroke, (0, 0, 0))
        
        return final_canvas
    
    def save_preview_frames(self, strokes: List[Stroke], 
                          optimized_order: List[Tuple[int, str]], 
                          output_dir: Path, num_frames: int = 10):
        """保存预览帧"""
        if not self.debug:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化画布
        self._initialize_canvas(strokes)
        
        # 创建笔触映射
        stroke_dict = {stroke.id: stroke for stroke in strokes}
        
        # 生成预览帧
        frames_to_save = np.linspace(0, len(optimized_order) - 1, num_frames, dtype=int)
        
        for i, frame_idx in enumerate(frames_to_save):
            canvas = self.canvas.copy()
            
            # 绘制到当前帧的所有笔触
            for j in range(frame_idx + 1):
                if j < len(optimized_order):
                    stroke_id, _ = optimized_order[j]
                    if stroke_id in stroke_dict:
                        stroke = stroke_dict[stroke_id]
                        color = self.highlight_color if j == frame_idx else self.drawn_color
                        self._draw_stroke_on_canvas(canvas, stroke, color)
            
            # 保存帧
            frame_path = output_dir / f"preview_frame_{i:03d}.jpg"
            cv2.imwrite(str(frame_path), canvas)
        
        print(f"预览帧已保存至: {output_dir}")

# 测试代码
if __name__ == "__main__":
    from stroke_extraction import StrokeExtractor
    from feature_modeling import FeatureModeler
    from structure_construction import StructureConstructor
    from stroke_ordering import StrokeOrderOptimizer
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python animation_generation.py <image_path>")
        sys.exit(1)
    
    # 完整流程测试
    print("提取笔触...")
    extractor = StrokeExtractor(debug=True)
    strokes = extractor.extract_strokes(sys.argv[1])
    
    print("提取特征...")
    modeler = FeatureModeler(debug=True)
    features = modeler.extract_features(strokes)
    
    print("构建结构...")
    constructor = StructureConstructor(debug=True)
    hasse_graph, stages = constructor.build_structure(features)
    
    print("优化排序...")
    optimizer = StrokeOrderOptimizer(debug=True)
    optimized_order = optimizer.optimize_order(hasse_graph, features, stages)
    
    print("生成动画...")
    generator = AnimationGenerator(fps=24, debug=True)
    
    output_path = Path("test_animation.mp4")
    generator.generate_animation(strokes, optimized_order, output_path)
    
    print(f"\n动画生成完成: {output_path}")
    print(f"动画时长: 约 {len(optimized_order) * 2 / 24:.1f} 秒")