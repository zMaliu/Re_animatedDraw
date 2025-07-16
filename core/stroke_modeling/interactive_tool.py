# -*- coding: utf-8 -*-
"""
交互式笔触分解工具

实现论文中要求的交互式工具
将画作分解为独立的2D笔触集合
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass


@dataclass
class StrokeRegion:
    """
    笔触区域数据结构
    """
    id: int
    mask: np.ndarray
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    center: Tuple[float, float]
    label: str = "unknown"
    confidence: float = 1.0


class InteractiveStrokeTool:
    """
    交互式笔触分解工具
    
    实现论文中的交互式笔触分解功能
    支持手动标注和半自动分割
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化交互式工具
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分割参数
        self.watershed_markers = config.get('watershed_markers', 'auto')
        self.min_stroke_area = config.get('min_stroke_area', 100)
        self.max_stroke_area = config.get('max_stroke_area', 10000)
        
        # 交互参数
        self.brush_size = config.get('brush_size', 5)
        self.eraser_size = config.get('eraser_size', 3)
        
        # 预处理参数
        self.blur_kernel = config.get('blur_kernel', 3)
        self.threshold_method = config.get('threshold_method', 'adaptive')
        
        # 存储状态
        self.current_image = None
        self.stroke_regions = []
        self.user_annotations = {}
        self.operation_history = []
        
    def load_image(self, image_path: str) -> bool:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # 重置状态
            self.stroke_regions = []
            self.user_annotations = {}
            self.operation_history = []
            
            self.logger.info(f"Image loaded: {image_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            return False
    
    def auto_segment_strokes(self, image: Optional[np.ndarray] = None) -> List[StrokeRegion]:
        """
        自动分割笔触
        
        Args:
            image: 输入图像，如果为None则使用当前图像
            
        Returns:
            List[StrokeRegion]: 分割得到的笔触区域列表
        """
        try:
            if image is None:
                image = self.current_image
            
            if image is None:
                self.logger.error("No image available for segmentation")
                return []
            
            # 预处理
            preprocessed = self._preprocess_for_segmentation(image)
            
            # 分水岭分割
            stroke_masks = self._watershed_segmentation(preprocessed)
            
            # 创建笔触区域
            stroke_regions = self._create_stroke_regions(stroke_masks, image)
            
            # 过滤无效区域
            filtered_regions = self._filter_stroke_regions(stroke_regions)
            
            self.stroke_regions = filtered_regions
            self.logger.info(f"Auto-segmented {len(filtered_regions)} stroke regions")
            
            return filtered_regions
            
        except Exception as e:
            self.logger.error(f"Error in auto segmentation: {str(e)}")
            return []
    
    def interactive_segment(self, seed_points: List[Tuple[int, int]], 
                           labels: List[int]) -> List[StrokeRegion]:
        """
        交互式分割
        
        Args:
            seed_points: 种子点列表
            labels: 对应的标签列表
            
        Returns:
            List[StrokeRegion]: 分割结果
        """
        try:
            if self.current_image is None:
                self.logger.error("No image loaded for interactive segmentation")
                return []
            
            # 预处理
            preprocessed = self._preprocess_for_segmentation(self.current_image)
            
            # 创建标记图
            markers = self._create_markers_from_seeds(seed_points, labels, preprocessed.shape)
            
            # 分水岭分割
            stroke_masks = self._watershed_with_markers(preprocessed, markers)
            
            # 创建笔触区域
            stroke_regions = self._create_stroke_regions(stroke_masks, self.current_image)
            
            # 更新状态
            self.stroke_regions.extend(stroke_regions)
            
            return stroke_regions
            
        except Exception as e:
            self.logger.error(f"Error in interactive segmentation: {str(e)}")
            return []
    
    def refine_stroke_boundary(self, stroke_id: int, 
                              refinement_points: List[Tuple[int, int]],
                              operation: str = 'add') -> bool:
        """
        细化笔触边界
        
        Args:
            stroke_id: 笔触ID
            refinement_points: 细化点列表
            operation: 操作类型 ('add', 'remove')
            
        Returns:
            bool: 是否成功
        """
        try:
            if stroke_id >= len(self.stroke_regions):
                self.logger.error(f"Invalid stroke ID: {stroke_id}")
                return False
            
            stroke_region = self.stroke_regions[stroke_id]
            
            # 保存操作历史
            self.operation_history.append({
                'type': 'refine_boundary',
                'stroke_id': stroke_id,
                'points': refinement_points.copy(),
                'operation': operation,
                'old_mask': stroke_region.mask.copy()
            })
            
            # 应用细化
            new_mask = self._apply_boundary_refinement(
                stroke_region.mask, refinement_points, operation
            )
            
            # 更新笔触区域
            self._update_stroke_region(stroke_id, new_mask)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error refining stroke boundary: {str(e)}")
            return False
    
    def merge_strokes(self, stroke_ids: List[int]) -> bool:
        """
        合并笔触
        
        Args:
            stroke_ids: 要合并的笔触ID列表
            
        Returns:
            bool: 是否成功
        """
        try:
            if len(stroke_ids) < 2:
                self.logger.error("Need at least 2 strokes to merge")
                return False
            
            # 验证ID有效性
            for stroke_id in stroke_ids:
                if stroke_id >= len(self.stroke_regions):
                    self.logger.error(f"Invalid stroke ID: {stroke_id}")
                    return False
            
            # 合并掩码
            merged_mask = np.zeros_like(self.stroke_regions[0].mask)
            for stroke_id in stroke_ids:
                merged_mask = np.logical_or(merged_mask, self.stroke_regions[stroke_id].mask)
            
            # 创建新的笔触区域
            new_stroke = self._create_stroke_region_from_mask(
                merged_mask.astype(np.uint8), len(self.stroke_regions)
            )
            
            # 保存操作历史
            self.operation_history.append({
                'type': 'merge_strokes',
                'stroke_ids': stroke_ids.copy(),
                'merged_stroke': new_stroke
            })
            
            # 移除原始笔触（从后往前删除以保持索引有效）
            for stroke_id in sorted(stroke_ids, reverse=True):
                del self.stroke_regions[stroke_id]
            
            # 添加合并后的笔触
            self.stroke_regions.append(new_stroke)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging strokes: {str(e)}")
            return False
    
    def split_stroke(self, stroke_id: int, 
                    split_line: List[Tuple[int, int]]) -> bool:
        """
        分割笔触
        
        Args:
            stroke_id: 要分割的笔触ID
            split_line: 分割线点列表
            
        Returns:
            bool: 是否成功
        """
        try:
            if stroke_id >= len(self.stroke_regions):
                self.logger.error(f"Invalid stroke ID: {stroke_id}")
                return False
            
            stroke_region = self.stroke_regions[stroke_id]
            
            # 创建分割线掩码
            split_mask = self._create_split_line_mask(split_line, stroke_region.mask.shape)
            
            # 应用分割
            split_masks = self._apply_stroke_split(stroke_region.mask, split_mask)
            
            if len(split_masks) < 2:
                self.logger.warning("Split operation did not produce multiple regions")
                return False
            
            # 保存操作历史
            self.operation_history.append({
                'type': 'split_stroke',
                'stroke_id': stroke_id,
                'split_line': split_line.copy(),
                'original_stroke': stroke_region
            })
            
            # 移除原始笔触
            del self.stroke_regions[stroke_id]
            
            # 添加分割后的笔触
            for i, mask in enumerate(split_masks):
                new_stroke = self._create_stroke_region_from_mask(
                    mask, len(self.stroke_regions)
                )
                self.stroke_regions.append(new_stroke)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error splitting stroke: {str(e)}")
            return False
    
    def undo_last_operation(self) -> bool:
        """
        撤销上一次操作
        
        Returns:
            bool: 是否成功
        """
        try:
            if not self.operation_history:
                self.logger.warning("No operations to undo")
                return False
            
            last_operation = self.operation_history.pop()
            
            # 根据操作类型进行撤销
            if last_operation['type'] == 'refine_boundary':
                stroke_id = last_operation['stroke_id']
                old_mask = last_operation['old_mask']
                self._update_stroke_region(stroke_id, old_mask)
                
            elif last_operation['type'] == 'merge_strokes':
                # 恢复原始笔触，移除合并后的笔触
                # 这里需要更复杂的逻辑来完全恢复状态
                pass
                
            elif last_operation['type'] == 'split_stroke':
                # 恢复原始笔触，移除分割后的笔触
                # 这里需要更复杂的逻辑来完全恢复状态
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error undoing operation: {str(e)}")
            return False
    
    def export_stroke_data(self, output_path: str) -> bool:
        """
        导出笔触数据
        
        Args:
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            stroke_data = {
                'image_shape': self.current_image.shape if self.current_image is not None else None,
                'strokes': []
            }
            
            for i, stroke in enumerate(self.stroke_regions):
                stroke_info = {
                    'id': stroke.id,
                    'bbox': stroke.bbox,
                    'area': stroke.area,
                    'center': stroke.center,
                    'label': stroke.label,
                    'confidence': stroke.confidence,
                    'mask_file': f"stroke_{i}_mask.png",
                    'contour': stroke.contour.tolist()
                }
                stroke_data['strokes'].append(stroke_info)
                
                # 保存掩码图像
                mask_path = output_path.replace('.json', f'_stroke_{i}_mask.png')
                cv2.imwrite(mask_path, stroke.mask * 255)
            
            # 保存JSON数据
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stroke_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Stroke data exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting stroke data: {str(e)}")
            return False
    
    def _preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        分割预处理
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # 阈值化
        if self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return binary
    
    def _watershed_segmentation(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """
        分水岭分割
        
        Args:
            binary_image: 二值图像
            
        Returns:
            List[np.ndarray]: 分割掩码列表
        """
        # 距离变换
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        
        # 寻找局部最大值作为种子
        if self.watershed_markers == 'auto':
            _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            markers = markers.astype(np.uint8)
            
            # 连通分量标记
            _, markers = cv2.connectedComponents(markers)
        else:
            # 使用预定义的标记
            markers = self.watershed_markers
        
        # 分水岭算法
        markers = cv2.watershed(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers)
        
        # 提取各个区域
        stroke_masks = []
        unique_labels = np.unique(markers)
        
        for label in unique_labels:
            if label <= 0:  # 跳过背景和边界
                continue
            
            mask = (markers == label).astype(np.uint8)
            stroke_masks.append(mask)
        
        return stroke_masks
    
    def _create_stroke_regions(self, stroke_masks: List[np.ndarray], 
                              original_image: np.ndarray) -> List[StrokeRegion]:
        """
        创建笔触区域对象
        
        Args:
            stroke_masks: 笔触掩码列表
            original_image: 原始图像
            
        Returns:
            List[StrokeRegion]: 笔触区域列表
        """
        stroke_regions = []
        
        for i, mask in enumerate(stroke_masks):
            stroke_region = self._create_stroke_region_from_mask(mask, i)
            stroke_regions.append(stroke_region)
        
        return stroke_regions
    
    def _create_stroke_region_from_mask(self, mask: np.ndarray, stroke_id: int) -> StrokeRegion:
        """
        从掩码创建笔触区域
        
        Args:
            mask: 笔触掩码
            stroke_id: 笔触ID
            
        Returns:
            StrokeRegion: 笔触区域对象
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 选择最大轮廓
            main_contour = max(contours, key=cv2.contourArea)
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 计算面积
            area = cv2.contourArea(main_contour)
            
            # 计算中心
            M = cv2.moments(main_contour)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = x + w // 2, y + h // 2
        else:
            # 如果没有找到轮廓，使用掩码信息
            coords = np.where(mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
                area = len(coords[0])
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                main_contour = np.array([[[x_min, y_min]], [[x_max, y_min]], 
                                       [[x_max, y_max]], [[x_min, y_max]]])
            else:
                x = y = w = h = 0
                area = 0
                cx = cy = 0
                main_contour = np.array([])
        
        return StrokeRegion(
            id=stroke_id,
            mask=mask,
            contour=main_contour,
            bbox=(x, y, w, h),
            area=area,
            center=(cx, cy)
        )
    
    def _filter_stroke_regions(self, stroke_regions: List[StrokeRegion]) -> List[StrokeRegion]:
        """
        过滤笔触区域
        
        Args:
            stroke_regions: 原始笔触区域列表
            
        Returns:
            List[StrokeRegion]: 过滤后的笔触区域列表
        """
        filtered_regions = []
        
        for region in stroke_regions:
            # 面积过滤
            if self.min_stroke_area <= region.area <= self.max_stroke_area:
                filtered_regions.append(region)
            else:
                self.logger.debug(f"Filtered out stroke {region.id} with area {region.area}")
        
        return filtered_regions
    
    def _create_markers_from_seeds(self, seed_points: List[Tuple[int, int]], 
                                  labels: List[int], 
                                  image_shape: Tuple[int, int]) -> np.ndarray:
        """
        从种子点创建标记图
        
        Args:
            seed_points: 种子点列表
            labels: 标签列表
            image_shape: 图像形状
            
        Returns:
            np.ndarray: 标记图
        """
        markers = np.zeros(image_shape, dtype=np.int32)
        
        for (x, y), label in zip(seed_points, labels):
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                markers[y, x] = label
        
        return markers
    
    def _watershed_with_markers(self, binary_image: np.ndarray, 
                               markers: np.ndarray) -> List[np.ndarray]:
        """
        使用标记的分水岭分割
        
        Args:
            binary_image: 二值图像
            markers: 标记图
            
        Returns:
            List[np.ndarray]: 分割掩码列表
        """
        # 分水岭算法
        result_markers = cv2.watershed(
            cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), markers
        )
        
        # 提取各个区域
        stroke_masks = []
        unique_labels = np.unique(result_markers)
        
        for label in unique_labels:
            if label <= 0:  # 跳过背景和边界
                continue
            
            mask = (result_markers == label).astype(np.uint8)
            stroke_masks.append(mask)
        
        return stroke_masks
    
    def _apply_boundary_refinement(self, mask: np.ndarray, 
                                  points: List[Tuple[int, int]], 
                                  operation: str) -> np.ndarray:
        """
        应用边界细化
        
        Args:
            mask: 原始掩码
            points: 细化点列表
            operation: 操作类型
            
        Returns:
            np.ndarray: 细化后的掩码
        """
        new_mask = mask.copy()
        
        for x, y in points:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if operation == 'add':
                    # 添加区域
                    cv2.circle(new_mask, (x, y), self.brush_size, 1, -1)
                elif operation == 'remove':
                    # 移除区域
                    cv2.circle(new_mask, (x, y), self.eraser_size, 0, -1)
        
        return new_mask
    
    def _update_stroke_region(self, stroke_id: int, new_mask: np.ndarray):
        """
        更新笔触区域
        
        Args:
            stroke_id: 笔触ID
            new_mask: 新掩码
        """
        if stroke_id < len(self.stroke_regions):
            # 重新计算笔触区域属性
            updated_region = self._create_stroke_region_from_mask(new_mask, stroke_id)
            updated_region.label = self.stroke_regions[stroke_id].label
            updated_region.confidence = self.stroke_regions[stroke_id].confidence
            
            self.stroke_regions[stroke_id] = updated_region
    
    def _create_split_line_mask(self, split_line: List[Tuple[int, int]], 
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建分割线掩码
        
        Args:
            split_line: 分割线点列表
            image_shape: 图像形状
            
        Returns:
            np.ndarray: 分割线掩码
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        if len(split_line) >= 2:
            points = np.array(split_line, dtype=np.int32)
            cv2.polylines(mask, [points], False, 1, thickness=2)
        
        return mask
    
    def _apply_stroke_split(self, stroke_mask: np.ndarray, 
                           split_mask: np.ndarray) -> List[np.ndarray]:
        """
        应用笔触分割
        
        Args:
            stroke_mask: 笔触掩码
            split_mask: 分割线掩码
            
        Returns:
            List[np.ndarray]: 分割后的掩码列表
        """
        # 从笔触掩码中减去分割线
        split_stroke = stroke_mask.copy()
        split_indices = split_mask > 0
        split_stroke[split_indices] = 0
        
        # 查找连通分量
        num_labels, labels = cv2.connectedComponents(split_stroke)
        
        # 提取各个分量
        split_masks = []
        for label in range(1, num_labels):  # 跳过背景标签0
            component_mask = (labels == label).astype(np.uint8)
            
            # 检查分量大小
            component_area = int(np.sum(component_mask))
            if component_area >= self.min_stroke_area:
                split_masks.append(component_mask)
        
        return split_masks
    
    def get_stroke_statistics(self) -> Dict[str, Any]:
        """
        获取笔触统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.stroke_regions:
            return {
                'total_strokes': 0,
                'total_area': 0,
                'average_area': 0,
                'area_std': 0
            }
        
        areas = [region.area for region in self.stroke_regions]
        
        return {
            'total_strokes': len(self.stroke_regions),
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'area_std': np.std(areas),
            'min_area': min(areas),
            'max_area': max(areas)
        }