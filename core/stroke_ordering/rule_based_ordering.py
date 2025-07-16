# -*- coding: utf-8 -*-
"""
基于规则的笔画排序

实现传统中国书法的笔画顺序规则
包括：先横后竖、先撇后捺、从上到下、从左到右等
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum
import math
from collections import defaultdict


class StrokeType(Enum):
    """
    笔画类型枚举
    """
    HORIZONTAL = "horizontal"      # 横
    VERTICAL = "vertical"          # 竖
    LEFT_FALLING = "left_falling"  # 撇
    RIGHT_FALLING = "right_falling" # 捺
    DOT = "dot"                   # 点
    HOOK = "hook"                 # 钩
    TURNING = "turning"           # 折
    CURVE = "curve"               # 弯
    COMPLEX = "complex"           # 复合笔画
    UNKNOWN = "unknown"           # 未知类型


@dataclass
class OrderingRule:
    """
    排序规则数据结构
    
    Attributes:
        rule_id (str): 规则ID
        name (str): 规则名称
        description (str): 规则描述
        priority (int): 优先级（数字越小优先级越高）
        condition (callable): 规则条件函数
        action (callable): 规则动作函数
        weight (float): 规则权重
        enabled (bool): 是否启用
    """
    rule_id: str
    name: str
    description: str
    priority: int
    condition: callable
    action: callable
    weight: float = 1.0
    enabled: bool = True


@dataclass
class RuleApplication:
    """
    规则应用结果
    
    Attributes:
        rule_id (str): 规则ID
        applied_strokes (List[int]): 应用的笔画索引
        confidence (float): 应用置信度
        effect (str): 应用效果描述
    """
    rule_id: str
    applied_strokes: List[int]
    confidence: float
    effect: str


class RuleBasedOrdering:
    """
    基于规则的笔画排序器
    
    实现传统中国书法笔画顺序规则
    """
    
    def __init__(self, config):
        """
        初始化基于规则的排序器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 规则配置
        self.use_traditional_rules = config['stroke_ordering'].get('use_traditional_rules', True)
        self.rule_strictness = config['stroke_ordering'].get('rule_strictness', 0.8)
        self.spatial_tolerance = config['stroke_ordering'].get('spatial_tolerance', 20)
        
        # 初始化规则
        self.rules = self._initialize_rules()
        
        # 笔画类型优先级
        self.stroke_priority = {
            StrokeType.HORIZONTAL: 1,
            StrokeType.VERTICAL: 2,
            StrokeType.LEFT_FALLING: 3,
            StrokeType.RIGHT_FALLING: 4,
            StrokeType.DOT: 5,
            StrokeType.HOOK: 6,
            StrokeType.TURNING: 7,
            StrokeType.CURVE: 8,
            StrokeType.COMPLEX: 9,
            StrokeType.UNKNOWN: 10
        }
    
    def order_strokes(self, strokes: List[Dict[str, Any]]) -> Tuple[List[int], List[RuleApplication]]:
        """
        基于规则排序笔画
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            Tuple[List[int], List[RuleApplication]]: 排序结果和应用的规则
        """
        try:
            self.logger.info(f"Applying rule-based ordering to {len(strokes)} strokes")
            
            # 初始化
            self.strokes = strokes
            applied_rules = []
            
            # 预处理：分类笔画
            stroke_types = self._classify_strokes(strokes)
            
            # 初始顺序
            current_order = list(range(len(strokes)))
            
            # 按优先级应用规则
            sorted_rules = sorted(self.rules, key=lambda r: r.priority)
            
            for rule in sorted_rules:
                if rule.enabled:
                    order_before = current_order.copy()
                    
                    # 应用规则
                    rule_result = self._apply_rule(rule, current_order, stroke_types)
                    
                    if rule_result:
                        current_order = rule_result['order']
                        applied_rules.append(RuleApplication(
                            rule_id=rule.rule_id,
                            applied_strokes=rule_result.get('affected_strokes', []),
                            confidence=rule_result.get('confidence', 0.5),
                            effect=rule_result.get('effect', 'Order modified')
                        ))
                        
                        self.logger.debug(f"Applied rule {rule.rule_id}: {rule.name}")
            
            self.logger.info(f"Rule-based ordering completed, applied {len(applied_rules)} rules")
            
            return current_order, applied_rules
            
        except Exception as e:
            self.logger.error(f"Error in rule-based ordering: {str(e)}")
            return list(range(len(strokes))), []
    
    def _initialize_rules(self) -> List[OrderingRule]:
        """
        初始化排序规则
        
        Returns:
            List[OrderingRule]: 规则列表
        """
        rules = []
        
        # 规则1：先横后竖
        rules.append(OrderingRule(
            rule_id="horizontal_before_vertical",
            name="先横后竖",
            description="横画应该在竖画之前",
            priority=1,
            condition=self._has_horizontal_and_vertical,
            action=self._apply_horizontal_before_vertical,
            weight=1.0
        ))
        
        # 规则2：先撇后捺
        rules.append(OrderingRule(
            rule_id="left_falling_before_right_falling",
            name="先撇后捺",
            description="撇画应该在捺画之前",
            priority=2,
            condition=self._has_left_and_right_falling,
            action=self._apply_left_before_right_falling,
            weight=1.0
        ))
        
        # 规则3：从上到下
        rules.append(OrderingRule(
            rule_id="top_to_bottom",
            name="从上到下",
            description="上方的笔画应该在下方的笔画之前",
            priority=3,
            condition=self._has_vertical_separation,
            action=self._apply_top_to_bottom,
            weight=0.8
        ))
        
        # 规则4：从左到右
        rules.append(OrderingRule(
            rule_id="left_to_right",
            name="从左到右",
            description="左侧的笔画应该在右侧的笔画之前",
            priority=4,
            condition=self._has_horizontal_separation,
            action=self._apply_left_to_right,
            weight=0.7
        ))
        
        # 规则5：先中间后两边
        rules.append(OrderingRule(
            rule_id="center_first",
            name="先中间后两边",
            description="中间的笔画应该在两边的笔画之前",
            priority=5,
            condition=self._has_center_structure,
            action=self._apply_center_first,
            weight=0.6
        ))
        
        # 规则6：先外后内
        rules.append(OrderingRule(
            rule_id="outside_before_inside",
            name="先外后内",
            description="外围的笔画应该在内部的笔画之前",
            priority=6,
            condition=self._has_enclosure_structure,
            action=self._apply_outside_before_inside,
            weight=0.7
        ))
        
        # 规则7：点画最后
        rules.append(OrderingRule(
            rule_id="dots_last",
            name="点画最后",
            description="点画通常在最后书写",
            priority=7,
            condition=self._has_dots,
            action=self._apply_dots_last,
            weight=0.8
        ))
        
        # 规则8：封口最后
        rules.append(OrderingRule(
            rule_id="closing_strokes_last",
            name="封口最后",
            description="封口的笔画应该最后书写",
            priority=8,
            condition=self._has_closing_strokes,
            action=self._apply_closing_strokes_last,
            weight=0.9
        ))
        
        return rules
    
    def _classify_strokes(self, strokes: List[Dict[str, Any]]) -> List[StrokeType]:
        """
        分类笔画类型
        
        Args:
            strokes (List[Dict]): 笔画列表
            
        Returns:
            List[StrokeType]: 笔画类型列表
        """
        try:
            stroke_types = []
            
            for stroke in strokes:
                # 从笔画数据中获取类型
                stroke_class = stroke.get('stroke_class', 'unknown')
                
                # 映射到枚举类型
                if stroke_class == 'horizontal':
                    stroke_type = StrokeType.HORIZONTAL
                elif stroke_class == 'vertical':
                    stroke_type = StrokeType.VERTICAL
                elif stroke_class == 'left_falling':
                    stroke_type = StrokeType.LEFT_FALLING
                elif stroke_class == 'right_falling':
                    stroke_type = StrokeType.RIGHT_FALLING
                elif stroke_class == 'dot':
                    stroke_type = StrokeType.DOT
                elif stroke_class == 'hook':
                    stroke_type = StrokeType.HOOK
                elif stroke_class == 'turning':
                    stroke_type = StrokeType.TURNING
                elif stroke_class == 'curve':
                    stroke_type = StrokeType.CURVE
                elif stroke_class == 'complex':
                    stroke_type = StrokeType.COMPLEX
                else:
                    # 基于几何特征推断类型
                    stroke_type = self._infer_stroke_type(stroke)
                
                stroke_types.append(stroke_type)
            
            return stroke_types
            
        except Exception as e:
            self.logger.error(f"Error classifying strokes: {str(e)}")
            return [StrokeType.UNKNOWN] * len(strokes)
    
    def _infer_stroke_type(self, stroke: Dict[str, Any]) -> StrokeType:
        """
        基于几何特征推断笔画类型
        
        Args:
            stroke (Dict): 笔画数据
            
        Returns:
            StrokeType: 推断的笔画类型
        """
        try:
            # 获取几何特征
            bbox = stroke.get('bounding_rect', (0, 0, 1, 1))
            width, height = bbox[2], bbox[3]
            aspect_ratio = stroke.get('aspect_ratio', width / height if height > 0 else 1)
            orientation = stroke.get('orientation', 0)
            area = stroke.get('area', width * height)
            
            # 基于面积判断是否为点
            if area < 100:  # 小面积认为是点
                return StrokeType.DOT
            
            # 基于长宽比和方向判断类型
            if aspect_ratio > 3:  # 很长的笔画
                # 基于方向判断横竖
                angle_deg = math.degrees(orientation) % 180
                if angle_deg < 30 or angle_deg > 150:  # 接近水平
                    return StrokeType.HORIZONTAL
                elif 60 < angle_deg < 120:  # 接近垂直
                    return StrokeType.VERTICAL
                elif 30 <= angle_deg <= 60:  # 右上到左下
                    return StrokeType.LEFT_FALLING
                elif 120 <= angle_deg <= 150:  # 左上到右下
                    return StrokeType.RIGHT_FALLING
            
            elif aspect_ratio < 0.3:  # 很高的笔画
                return StrokeType.VERTICAL
            
            # 检查是否有钩或转折
            skeleton = stroke.get('skeleton', [])
            if len(skeleton) > 2:
                # 计算转折角度
                angles = self._calculate_skeleton_angles(skeleton)
                if any(abs(angle) > 45 for angle in angles):
                    return StrokeType.TURNING
            
            # 默认为复合笔画
            return StrokeType.COMPLEX
            
        except Exception as e:
            self.logger.error(f"Error inferring stroke type: {str(e)}")
            return StrokeType.UNKNOWN
    
    def _calculate_skeleton_angles(self, skeleton: List[Tuple[int, int]]) -> List[float]:
        """
        计算骨架的转折角度
        
        Args:
            skeleton (List[Tuple[int, int]]): 骨架点列表
            
        Returns:
            List[float]: 转折角度列表
        """
        try:
            if len(skeleton) < 3:
                return []
            
            angles = []
            for i in range(1, len(skeleton) - 1):
                p1 = np.array(skeleton[i - 1])
                p2 = np.array(skeleton[i])
                p3 = np.array(skeleton[i + 1])
                
                # 计算向量
                v1 = p2 - p1
                v2 = p3 - p2
                
                # 计算角度
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = math.degrees(math.acos(cos_angle))
                    angles.append(180 - angle)  # 转折角度
            
            return angles
            
        except Exception as e:
            self.logger.error(f"Error calculating skeleton angles: {str(e)}")
            return []
    
    def _apply_rule(self, rule: OrderingRule, current_order: List[int], 
                   stroke_types: List[StrokeType]) -> Optional[Dict[str, Any]]:
        """
        应用单个规则
        
        Args:
            rule (OrderingRule): 规则
            current_order (List[int]): 当前顺序
            stroke_types (List[StrokeType]): 笔画类型
            
        Returns:
            Optional[Dict[str, Any]]: 应用结果
        """
        try:
            # 检查规则条件
            if rule.condition(current_order, stroke_types):
                # 应用规则动作
                result = rule.action(current_order, stroke_types)
                if result:
                    order_from_result = result.get('order')
                    if isinstance(order_from_result, np.ndarray) and isinstance(current_order, np.ndarray):
                        if not np.array_equal(order_from_result, current_order):
                            return result
                    elif isinstance(order_from_result, list) and isinstance(current_order, list):
                        if order_from_result != current_order:
                            return result
                    else:
                        # 如果类型不同或无法比较，假设不相等
                        return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule.rule_id}: {str(e)}")
            return None
    
    # 规则条件函数
    def _has_horizontal_and_vertical(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否同时有横画和竖画
        """
        has_horizontal = StrokeType.HORIZONTAL in types
        has_vertical = StrokeType.VERTICAL in types
        return has_horizontal and has_vertical
    
    def _has_left_and_right_falling(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否同时有撇和捺
        """
        has_left = StrokeType.LEFT_FALLING in types
        has_right = StrokeType.RIGHT_FALLING in types
        return has_left and has_right
    
    def _has_vertical_separation(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有垂直方向的分离
        """
        if len(order) < 2:
            return False
        
        centroids = [self.strokes[i].get('centroid', (0, 0)) for i in order]
        y_coords = [c[1] for c in centroids]
        
        return max(y_coords) - min(y_coords) > self.spatial_tolerance
    
    def _has_horizontal_separation(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有水平方向的分离
        """
        if len(order) < 2:
            return False
        
        centroids = [self.strokes[i].get('centroid', (0, 0)) for i in order]
        x_coords = [c[0] for c in centroids]
        
        return max(x_coords) - min(x_coords) > self.spatial_tolerance
    
    def _has_center_structure(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有中心结构
        """
        if len(order) < 3:
            return False
        
        # 简单检查：是否有笔画在中心位置
        centroids = [self.strokes[i].get('centroid', (0, 0)) for i in order]
        x_coords = [c[0] for c in centroids]
        
        center_x = (min(x_coords) + max(x_coords)) / 2
        center_tolerance = (max(x_coords) - min(x_coords)) * 0.2
        
        center_strokes = [i for i, c in enumerate(centroids) 
                         if abs(c[0] - center_x) < center_tolerance]
        
        return len(center_strokes) > 0 and len(center_strokes) < len(centroids)
    
    def _has_enclosure_structure(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有包围结构
        """
        # 简化检查：是否有形成边界的笔画
        if len(order) < 3:
            return False
        
        # 检查是否有笔画形成矩形边界
        bboxes = [self.strokes[i].get('bounding_rect', (0, 0, 1, 1)) for i in order]
        
        # 计算整体边界
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        # 检查是否有笔画接近边界
        boundary_tolerance = 10
        boundary_strokes = 0
        
        for bbox in bboxes:
            if (abs(bbox[0] - min_x) < boundary_tolerance or  # 左边界
                abs(bbox[1] - min_y) < boundary_tolerance or  # 上边界
                abs(bbox[0] + bbox[2] - max_x) < boundary_tolerance or  # 右边界
                abs(bbox[1] + bbox[3] - max_y) < boundary_tolerance):  # 下边界
                boundary_strokes += 1
        
        return boundary_strokes >= 2
    
    def _has_dots(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有点画
        """
        return StrokeType.DOT in types
    
    def _has_closing_strokes(self, order: List[int], types: List[StrokeType]) -> bool:
        """
        检查是否有封口笔画
        """
        # 简化检查：是否有可能的封口笔画
        return len(order) > 2 and (StrokeType.HORIZONTAL in types or StrokeType.TURNING in types)
    
    # 规则动作函数
    def _apply_horizontal_before_vertical(self, order: List[int], 
                                        types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用先横后竖规则
        """
        new_order = order.copy()
        affected_strokes = []
        
        # 找到所有横画和竖画
        horizontal_indices = [i for i, t in enumerate(types) if t == StrokeType.HORIZONTAL]
        vertical_indices = [i for i, t in enumerate(types) if t == StrokeType.VERTICAL]
        
        # 检查是否需要调整
        swapped = False
        for h_idx in horizontal_indices:
            for v_idx in vertical_indices:
                h_pos = new_order.index(h_idx)
                v_pos = new_order.index(v_idx)
                
                # 如果竖画在横画前面，且它们在空间上相近，则交换
                if v_pos < h_pos:
                    h_centroid = self.strokes[h_idx].get('centroid', (0, 0))
                    v_centroid = self.strokes[v_idx].get('centroid', (0, 0))
                    h_array = np.array(h_centroid)
                    v_array = np.array(v_centroid)
                    distance = np.linalg.norm(h_array - v_array)
                    
                    if distance < self.spatial_tolerance * 2:
                        # 交换位置
                        new_order[h_pos], new_order[v_pos] = new_order[v_pos], new_order[h_pos]
                        affected_strokes.extend([h_idx, v_idx])
                        swapped = True
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.9 if swapped else 0.5,
            'effect': f'Moved {len(affected_strokes)} strokes to follow horizontal-before-vertical rule'
        }
    
    def _apply_left_before_right_falling(self, order: List[int], 
                                       types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用先撇后捺规则
        """
        new_order = order.copy()
        affected_strokes = []
        
        # 找到所有撇画和捺画
        left_indices = [i for i, t in enumerate(types) if t == StrokeType.LEFT_FALLING]
        right_indices = [i for i, t in enumerate(types) if t == StrokeType.RIGHT_FALLING]
        
        # 检查是否需要调整
        swapped = False
        for l_idx in left_indices:
            for r_idx in right_indices:
                l_pos = new_order.index(l_idx)
                r_pos = new_order.index(r_idx)
                
                # 如果捺画在撇画前面，且它们在空间上相近，则交换
                if r_pos < l_pos:
                    l_centroid = self.strokes[l_idx].get('centroid', (0, 0))
                    r_centroid = self.strokes[r_idx].get('centroid', (0, 0))
                    l_array = np.array(l_centroid)
                    r_array = np.array(r_centroid)
                    distance = np.linalg.norm(l_array - r_array)
                    
                    if distance < self.spatial_tolerance * 2:
                        # 交换位置
                        new_order[l_pos], new_order[r_pos] = new_order[r_pos], new_order[l_pos]
                        affected_strokes.extend([l_idx, r_idx])
                        swapped = True
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.9 if swapped else 0.5,
            'effect': f'Applied left-falling-before-right-falling rule to {len(affected_strokes)} strokes'
        }
    
    def _apply_top_to_bottom(self, order: List[int], types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用从上到下规则
        """
        # 按y坐标排序
        stroke_positions = [(i, self.strokes[order[i]].get('centroid', (0, 0))[1]) 
                           for i in range(len(order))]
        stroke_positions.sort(key=lambda x: x[1])  # 按y坐标排序
        
        new_order = [order[pos[0]] for pos in stroke_positions]
        
        affected_strokes = []
        for i in range(len(order)):
            # 安全比较，避免数组真值歧义
            try:
                if not self._safe_compare(new_order[i], order[i]):
                    affected_strokes.append(i)
            except (ValueError, TypeError, IndexError):
                # 如果比较失败，认为不相等
                affected_strokes.append(i)
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.8,
            'effect': f'Reordered {len(affected_strokes)} strokes from top to bottom'
        }
    
    def _apply_left_to_right(self, order: List[int], types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用从左到右规则
        """
        # 按x坐标排序（在相同y坐标范围内）
        new_order = order.copy()
        affected_strokes = []
        
        # 分组：y坐标相近的笔画
        stroke_groups = self._group_strokes_by_y_position(order)
        
        for group in stroke_groups:
            if len(group) > 1:
                # 在组内按x坐标排序
                group_positions = [(i, self.strokes[order[i]].get('centroid', (0, 0))[0]) 
                                 for i in group]
                group_positions.sort(key=lambda x: x[1])  # 按x坐标排序
                
                # 更新顺序
                for j, (original_pos, _) in enumerate(group_positions):
                    new_pos = group[j]
                    # 安全比较，避免数组真值歧义
                    try:
                        old_val = new_order[new_pos]
                        new_val = order[original_pos]
                        
                        # 使用安全比较函数
                        if not self._safe_compare(old_val, new_val):
                            affected_strokes.append(order[original_pos])
                    except (ValueError, TypeError, IndexError):
                        # 如果比较失败，认为不相等
                        affected_strokes.append(order[original_pos])
                    new_order[new_pos] = order[original_pos]
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.7,
            'effect': f'Applied left-to-right rule to {len(affected_strokes)} strokes'
        }
    
    def _apply_center_first(self, order: List[int], types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用先中间后两边规则
        """
        centroids = [self.strokes[i].get('centroid', (0, 0)) for i in order]
        x_coords = [c[0] for c in centroids]
        
        center_x = (min(x_coords) + max(x_coords)) / 2
        
        # 按距离中心的距离排序
        stroke_distances = [(i, abs(centroids[i][0] - center_x)) for i in range(len(order))]
        stroke_distances.sort(key=lambda x: x[1])  # 按距离中心的距离排序
        
        new_order = [order[pos[0]] for pos in stroke_distances]
        affected_strokes = []
        for i in range(len(order)):
            # 安全比较，避免数组真值歧义
            try:
                if not self._safe_compare(new_order[i], order[i]):
                    affected_strokes.append(i)
            except (ValueError, TypeError, IndexError):
                # 如果比较失败，认为不相等
                affected_strokes.append(i)
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.6,
            'effect': f'Applied center-first rule to {len(affected_strokes)} strokes'
        }
    
    def _apply_outside_before_inside(self, order: List[int], 
                                   types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用先外后内规则
        """
        # 计算每个笔画到边界的距离
        bboxes = [self.strokes[i].get('bounding_rect', (0, 0, 1, 1)) for i in order]
        
        # 计算整体边界
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
        
        # 计算每个笔画到边界的最小距离
        stroke_boundary_distances = []
        for i, bbox in enumerate(bboxes):
            distances = [
                bbox[0] - min_x,  # 到左边界
                bbox[1] - min_y,  # 到上边界
                max_x - (bbox[0] + bbox[2]),  # 到右边界
                max_y - (bbox[1] + bbox[3])   # 到下边界
            ]
            min_distance = min(distances)
            stroke_boundary_distances.append((i, min_distance))
        
        # 按到边界的距离排序（距离小的先画）
        stroke_boundary_distances.sort(key=lambda x: x[1])
        
        new_order = [order[pos[0]] for pos in stroke_boundary_distances]
        affected_strokes = []
        for i in range(len(order)):
            # 安全比较，避免数组真值歧义
            try:
                if not self._safe_compare(new_order[i], order[i]):
                    affected_strokes.append(i)
            except (ValueError, TypeError, IndexError):
                # 如果比较失败，认为不相等
                affected_strokes.append(i)
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.7,
            'effect': f'Applied outside-before-inside rule to {len(affected_strokes)} strokes'
        }
    
    def _apply_dots_last(self, order: List[int], types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用点画最后规则
        """
        new_order = []
        dots = []
        affected_strokes = []
        
        # 分离点画和其他笔画
        for i, stroke_type in enumerate(types):
            if stroke_type == StrokeType.DOT:
                dots.append(order[i])
                affected_strokes.append(order[i])
            else:
                new_order.append(order[i])
        
        # 点画放在最后
        new_order.extend(dots)
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.8,
            'effect': f'Moved {len(dots)} dots to the end'
        }
    
    def _apply_closing_strokes_last(self, order: List[int], 
                                  types: List[StrokeType]) -> Dict[str, Any]:
        """
        应用封口最后规则
        """
        # 简化实现：将可能的封口笔画（横画和转折）移到后面
        new_order = []
        closing_candidates = []
        affected_strokes = []
        
        # 识别可能的封口笔画
        for i, stroke_type in enumerate(types):
            stroke = self.strokes[order[i]]
            bbox = stroke.get('bounding_rect', (0, 0, 1, 1))
            
            # 简单判断：底部的横画可能是封口
            if (stroke_type == StrokeType.HORIZONTAL and 
                self._is_bottom_stroke(stroke, order)):
                closing_candidates.append(order[i])
                affected_strokes.append(order[i])
            else:
                new_order.append(order[i])
        
        # 封口笔画放在最后
        new_order.extend(closing_candidates)
        
        return {
            'order': new_order,
            'affected_strokes': affected_strokes,
            'confidence': 0.6,
            'effect': f'Moved {len(closing_candidates)} potential closing strokes to the end'
        }
    
    def _group_strokes_by_y_position(self, order: List[int]) -> List[List[int]]:
        """
        按y坐标位置分组笔画
        
        Args:
            order (List[int]): 笔画顺序
            
        Returns:
            List[List[int]]: 分组结果
        """
        try:
            # 获取所有y坐标
            y_positions = [(i, self.strokes[order[i]].get('centroid', (0, 0))[1]) 
                          for i in range(len(order))]
            
            # 按y坐标排序
            y_positions.sort(key=lambda x: x[1])
            
            # 分组
            groups = []
            current_group = [y_positions[0][0]]
            current_y = y_positions[0][1]
            
            for i in range(1, len(y_positions)):
                pos, y = y_positions[i]
                
                if abs(y - current_y) <= self.spatial_tolerance:
                    current_group.append(pos)
                else:
                    groups.append(current_group)
                    current_group = [pos]
                    current_y = y
            
            groups.append(current_group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error grouping strokes by y position: {str(e)}")
            return [[i] for i in range(len(order))]
    
    def _is_bottom_stroke(self, stroke: Dict[str, Any], order: List[int]) -> bool:
        """
        判断是否为底部笔画
        
        Args:
            stroke (Dict): 笔画数据
            order (List[int]): 笔画顺序
            
        Returns:
            bool: 是否为底部笔画
        """
        try:
            stroke_y = stroke.get('centroid', (0, 0))[1]
            
            # 计算所有笔画的y坐标范围
            all_y = [self.strokes[i].get('centroid', (0, 0))[1] for i in order]
            max_y = max(all_y)
            y_range = max_y - min(all_y)
            
            # 如果笔画在底部20%范围内，认为是底部笔画
            return stroke_y > max_y - y_range * 0.2
            
        except Exception as e:
            self.logger.error(f"Error checking if bottom stroke: {str(e)}")
            return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        获取规则统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules if r.enabled]),
            'rule_priorities': {r.rule_id: r.priority for r in self.rules},
            'rule_weights': {r.rule_id: r.weight for r in self.rules},
            'stroke_type_priorities': {t.value: p for t, p in self.stroke_priority.items()}
        }
    
    def enable_rule(self, rule_id: str) -> bool:
        """
        启用规则
        
        Args:
            rule_id (str): 规则ID
            
        Returns:
            bool: 是否成功
        """
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = True
                return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """
        禁用规则
        
        Args:
            rule_id (str): 规则ID
            
        Returns:
            bool: 是否成功
        """
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.enabled = False
                return True
        return False
    
    def set_rule_weight(self, rule_id: str, weight: float) -> bool:
        """
        设置规则权重
        
        Args:
            rule_id (str): 规则ID
            weight (float): 权重值
            
        Returns:
            bool: 是否成功
        """
        for rule in self.rules:
            if rule.rule_id == rule_id:
                rule.weight = weight
                return True
        return False
    
    def _safe_compare(self, a, b) -> bool:
        """
        安全比较两个值，避免数组真值歧义
        
        Args:
            a: 第一个值
            b: 第二个值
            
        Returns:
            bool: 是否相等
        """
        # 使用通用的安全比较函数
        from utils.math_utils import safe_compare
        return safe_compare(a, b)