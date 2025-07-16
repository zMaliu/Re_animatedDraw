# -*- coding: utf-8 -*-
"""
笔画数据库

管理笔画模板库，提供笔画存储、检索和匹配功能
支持从数字化艺术家绘制的笔画中构建笔画库
"""

import os
import json
import pickle
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime


@dataclass
class StrokeTemplate:
    """
    笔画模板数据结构
    
    Attributes:
        id (str): 笔画唯一标识
        category (str): 笔画类别
        skeleton (np.ndarray): 骨架点序列
        contour (np.ndarray): 轮廓点序列
        width_profile (np.ndarray): 宽度变化曲线
        pressure_profile (np.ndarray): 压力变化曲线
        velocity_profile (np.ndarray): 速度变化曲线
        features (Dict): 特征向量
        metadata (Dict): 元数据信息
        creation_time (str): 创建时间
    """
    id: str
    category: str
    skeleton: np.ndarray
    contour: np.ndarray
    width_profile: np.ndarray
    pressure_profile: np.ndarray
    velocity_profile: np.ndarray
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    creation_time: str


class StrokeDatabase:
    """
    笔画数据库
    
    管理笔画模板的存储、检索和匹配
    """
    
    def __init__(self, config, database_path: str = None):
        """
        初始化笔画数据库
        
        Args:
            config: 配置对象
            database_path (str, optional): 数据库文件路径
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据库路径
        if database_path is None:
            database_path = config['stroke_database'].get('database_path', 'data/stroke_database.pkl')
        self.database_path = Path(database_path)
        
        # 笔画模板存储
        self.templates: Dict[str, StrokeTemplate] = {}
        self.categories: Dict[str, List[str]] = {}  # 类别到模板ID的映射
        self.feature_index: Dict[str, np.ndarray] = {}  # 特征索引
        
        # 数据库配置
        self.max_templates_per_category = config['stroke_database'].get('max_templates_per_category', 1000)
        self.feature_dimensions = config['stroke_database'].get('feature_dimensions', 128)
        self.similarity_threshold = config['stroke_database'].get('similarity_threshold', 0.8)
        
        # 加载现有数据库
        self.load_database()
    
    def add_template(self, template: StrokeTemplate) -> bool:
        """
        添加笔画模板
        
        Args:
            template (StrokeTemplate): 笔画模板
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 检查模板ID是否已存在
            if template.id in self.templates:
                self.logger.warning(f"Template {template.id} already exists, updating...")
            
            # 验证模板数据
            if not self._validate_template(template):
                self.logger.error(f"Invalid template data for {template.id}")
                return False
            
            # 添加到模板存储
            self.templates[template.id] = template
            
            # 更新类别索引
            if template.category not in self.categories:
                self.categories[template.category] = []
            
            if template.id not in self.categories[template.category]:
                self.categories[template.category].append(template.id)
            
            # 限制每个类别的模板数量
            if len(self.categories[template.category]) > self.max_templates_per_category:
                # 移除最旧的模板
                oldest_id = self.categories[template.category].pop(0)
                if oldest_id in self.templates:
                    del self.templates[oldest_id]
                self.logger.info(f"Removed oldest template {oldest_id} from category {template.category}")
            
            # 更新特征索引
            self._update_feature_index(template)
            
            self.logger.info(f"Added template {template.id} to category {template.category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding template {template.id}: {str(e)}")
            return False
    
    def get_template(self, template_id: str) -> Optional[StrokeTemplate]:
        """
        获取指定ID的笔画模板
        
        Args:
            template_id (str): 模板ID
            
        Returns:
            StrokeTemplate: 笔画模板，如果不存在则返回None
        """
        return self.templates.get(template_id)
    
    def get_templates_by_category(self, category: str) -> List[StrokeTemplate]:
        """
        获取指定类别的所有笔画模板
        
        Args:
            category (str): 笔画类别
            
        Returns:
            List[StrokeTemplate]: 笔画模板列表
        """
        template_ids = self.categories.get(category, [])
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def search_similar_templates(self, query_features: Dict[str, Any], 
                               category: str = None, 
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索相似的笔画模板
        
        Args:
            query_features (Dict): 查询特征
            category (str, optional): 限制搜索的类别
            top_k (int): 返回前k个最相似的模板
            
        Returns:
            List[Tuple[str, float]]: (模板ID, 相似度)的列表
        """
        candidates = []
        
        # 确定搜索范围
        if category:
            template_ids = self.categories.get(category, [])
        else:
            template_ids = list(self.templates.keys())
        
        # 计算相似度
        for template_id in template_ids:
            if template_id not in self.templates:
                continue
                
            template = self.templates[template_id]
            similarity = self._compute_similarity(query_features, template.features)
            
            if similarity >= self.similarity_threshold:
                candidates.append((template_id, similarity))
        
        # 按相似度排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def remove_template(self, template_id: str) -> bool:
        """
        移除笔画模板
        
        Args:
            template_id (str): 模板ID
            
        Returns:
            bool: 是否成功移除
        """
        try:
            if template_id not in self.templates:
                self.logger.warning(f"Template {template_id} not found")
                return False
            
            template = self.templates[template_id]
            
            # 从模板存储中移除
            del self.templates[template_id]
            
            # 从类别索引中移除
            if template.category in self.categories:
                if template_id in self.categories[template.category]:
                    self.categories[template.category].remove(template_id)
                
                # 如果类别为空，移除类别
                if not self.categories[template.category]:
                    del self.categories[template.category]
            
            # 更新特征索引
            self._remove_from_feature_index(template_id)
            
            self.logger.info(f"Removed template {template_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing template {template_id}: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_templates': len(self.templates),
            'categories': len(self.categories),
            'category_distribution': {cat: len(ids) for cat, ids in self.categories.items()},
            'database_size_mb': self._get_database_size(),
            'last_updated': datetime.now().isoformat()
        }
        
        return stats
    
    def save_database(self, path: str = None) -> bool:
        """
        保存数据库到文件
        
        Args:
            path (str, optional): 保存路径
            
        Returns:
            bool: 是否成功保存
        """
        try:
            if path is None:
                path = self.database_path
            else:
                path = Path(path)
            
            # 确保目录存在
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 准备保存数据
            save_data = {
                'templates': {},
                'categories': self.categories,
                'metadata': {
                    'version': '1.0',
                    'created_time': datetime.now().isoformat(),
                    'total_templates': len(self.templates)
                }
            }
            
            # 序列化模板数据
            for template_id, template in self.templates.items():
                save_data['templates'][template_id] = self._serialize_template(template)
            
            # 保存到文件
            with open(path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Database saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving database: {str(e)}")
            return False
    
    def load_database(self, path: str = None) -> bool:
        """
        从文件加载数据库
        
        Args:
            path (str, optional): 加载路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if path is None:
                path = self.database_path
            else:
                path = Path(path)
            
            if not path.exists():
                self.logger.info(f"Database file {path} not found, starting with empty database")
                return True
            
            # 从文件加载
            with open(path, 'rb') as f:
                save_data = pickle.load(f)
            
            # 恢复数据
            self.categories = save_data.get('categories', {})
            
            # 反序列化模板数据
            template_data = save_data.get('templates', {})
            for template_id, serialized_template in template_data.items():
                template = self._deserialize_template(serialized_template)
                if template:
                    self.templates[template_id] = template
            
            # 重建特征索引
            self._rebuild_feature_index()
            
            metadata = save_data.get('metadata', {})
            self.logger.info(f"Database loaded from {path}, {len(self.templates)} templates")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading database: {str(e)}")
            return False
    
    def export_templates(self, export_path: str, category: str = None) -> bool:
        """
        导出模板数据
        
        Args:
            export_path (str): 导出路径
            category (str, optional): 指定类别
            
        Returns:
            bool: 是否成功导出
        """
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 确定导出的模板
            if category:
                template_ids = self.categories.get(category, [])
            else:
                template_ids = list(self.templates.keys())
            
            # 导出每个模板
            for template_id in template_ids:
                if template_id not in self.templates:
                    continue
                
                template = self.templates[template_id]
                
                # 创建模板目录
                template_dir = export_path / template_id
                template_dir.mkdir(exist_ok=True)
                
                # 保存模板数据
                template_file = template_dir / 'template.json'
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(self._template_to_dict(template), f, indent=2, ensure_ascii=False)
                
                # 保存数组数据
                np.save(template_dir / 'skeleton.npy', template.skeleton)
                np.save(template_dir / 'contour.npy', template.contour)
                np.save(template_dir / 'width_profile.npy', template.width_profile)
                np.save(template_dir / 'pressure_profile.npy', template.pressure_profile)
                np.save(template_dir / 'velocity_profile.npy', template.velocity_profile)
            
            self.logger.info(f"Exported {len(template_ids)} templates to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting templates: {str(e)}")
            return False
    
    def _validate_template(self, template: StrokeTemplate) -> bool:
        """
        验证模板数据的有效性
        
        Args:
            template (StrokeTemplate): 笔画模板
            
        Returns:
            bool: 是否有效
        """
        try:
            # 检查必要字段
            if not template.id or not template.category:
                return False
            
            # 检查数组数据
            if (template.skeleton is None or len(template.skeleton) == 0 or
                template.contour is None or len(template.contour) == 0):
                return False
            
            # 检查特征数据
            if not isinstance(template.features, dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        计算两个特征向量的相似度
        
        Args:
            features1 (Dict): 特征向量1
            features2 (Dict): 特征向量2
            
        Returns:
            float: 相似度分数
        """
        try:
            # 提取数值特征
            numeric_features1 = self._extract_numeric_features(features1)
            numeric_features2 = self._extract_numeric_features(features2)
            
            if len(numeric_features1) == 0 or len(numeric_features2) == 0:
                return 0.0
            
            # 计算余弦相似度
            dot_product = np.dot(numeric_features1, numeric_features2)
            norm1 = np.linalg.norm(numeric_features1)
            norm2 = np.linalg.norm(numeric_features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 确保非负
            
        except Exception:
            return 0.0
    
    def _extract_numeric_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        从特征字典中提取数值特征
        
        Args:
            features (Dict): 特征字典
            
        Returns:
            np.ndarray: 数值特征向量
        """
        numeric_values = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, np.ndarray) and value.size > 0:
                if value.ndim == 1:
                    numeric_values.extend(value.flatten())
                else:
                    numeric_values.extend(value.flatten()[:10])  # 限制长度
        
        return np.array(numeric_values) if numeric_values else np.array([])
    
    def _update_feature_index(self, template: StrokeTemplate):
        """
        更新特征索引
        
        Args:
            template (StrokeTemplate): 笔画模板
        """
        try:
            feature_vector = self._extract_numeric_features(template.features)
            if len(feature_vector) > 0:
                self.feature_index[template.id] = feature_vector
        except Exception as e:
            self.logger.warning(f"Error updating feature index for {template.id}: {str(e)}")
    
    def _remove_from_feature_index(self, template_id: str):
        """
        从特征索引中移除模板
        
        Args:
            template_id (str): 模板ID
        """
        if template_id in self.feature_index:
            del self.feature_index[template_id]
    
    def _rebuild_feature_index(self):
        """
        重建特征索引
        """
        self.feature_index.clear()
        for template_id, template in self.templates.items():
            self._update_feature_index(template)
    
    def _serialize_template(self, template: StrokeTemplate) -> Dict[str, Any]:
        """
        序列化模板数据
        
        Args:
            template (StrokeTemplate): 笔画模板
            
        Returns:
            Dict: 序列化数据
        """
        data = asdict(template)
        
        # 转换numpy数组为列表
        for key in ['skeleton', 'contour', 'width_profile', 'pressure_profile', 'velocity_profile']:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        
        return data
    
    def _deserialize_template(self, data: Dict[str, Any]) -> Optional[StrokeTemplate]:
        """
        反序列化模板数据
        
        Args:
            data (Dict): 序列化数据
            
        Returns:
            StrokeTemplate: 笔画模板
        """
        try:
            # 转换列表为numpy数组
            for key in ['skeleton', 'contour', 'width_profile', 'pressure_profile', 'velocity_profile']:
                if isinstance(data[key], list):
                    data[key] = np.array(data[key])
            
            return StrokeTemplate(**data)
            
        except Exception as e:
            self.logger.error(f"Error deserializing template: {str(e)}")
            return None
    
    def _template_to_dict(self, template: StrokeTemplate) -> Dict[str, Any]:
        """
        将模板转换为字典（用于JSON导出）
        
        Args:
            template (StrokeTemplate): 笔画模板
            
        Returns:
            Dict: 字典数据
        """
        data = asdict(template)
        
        # 移除numpy数组（单独保存）
        for key in ['skeleton', 'contour', 'width_profile', 'pressure_profile', 'velocity_profile']:
            if key in data:
                data[key] = f"{key}.npy"  # 引用文件名
        
        return data
    
    def _get_database_size(self) -> float:
        """
        获取数据库大小（MB）
        
        Returns:
            float: 数据库大小
        """
        try:
            if self.database_path.exists():
                size_bytes = self.database_path.stat().st_size
                return size_bytes / (1024 * 1024)  # 转换为MB
            else:
                return 0.0
        except Exception:
            return 0.0