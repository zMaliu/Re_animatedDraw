# -*- coding: utf-8 -*-
"""
笔触数据库模块

提供笔触的存储、检索和管理功能
"""

import sqlite3
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from ..stroke_extraction.stroke_detector import Stroke


@dataclass
class StrokeTemplate:
    """笔触模板数据结构"""
    id: int
    name: str
    category: str
    features: Dict[str, Any]
    contour: np.ndarray
    skeleton: np.ndarray
    metadata: Dict[str, Any]
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """后处理初始化"""
        if self.metadata is None:
            self.metadata = {}


class StrokeDatabase:
    """笔触数据库管理器"""
    
    def __init__(self, config: Dict[str, Any], database_path: str = None):
        """
        初始化笔触数据库
        
        Args:
            config: 配置参数
            database_path: 数据库文件路径
        """
        self.config = config
        self.database_path = database_path or './data/stroke_database.db'
        self.logger = logging.getLogger(__name__)
        
        # 确保数据库目录存在
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        # 内存缓存
        self._template_cache = {}
        self._feature_cache = {}
        
    def _init_database(self):
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 创建笔触模板表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stroke_templates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        category TEXT NOT NULL,
                        features BLOB,
                        contour BLOB,
                        skeleton BLOB,
                        metadata BLOB,
                        similarity_threshold REAL DEFAULT 0.7,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 创建特征索引表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feature_index (
                        template_id INTEGER,
                        feature_name TEXT,
                        feature_value REAL,
                        FOREIGN KEY (template_id) REFERENCES stroke_templates (id)
                    )
                ''')
                
                # 创建分类索引
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_category 
                    ON stroke_templates (category)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feature_name 
                    ON feature_index (feature_name)
                ''')
                
                conn.commit()
                self.logger.info(f"Database initialized: {self.database_path}")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def add_template(self, template: StrokeTemplate) -> int:
        """
        添加笔触模板
        
        Args:
            template: 笔触模板
            
        Returns:
            模板ID
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 序列化数据
                features_blob = pickle.dumps(template.features)
                contour_blob = pickle.dumps(template.contour)
                skeleton_blob = pickle.dumps(template.skeleton)
                metadata_blob = pickle.dumps(template.metadata)
                
                # 插入模板
                cursor.execute('''
                    INSERT INTO stroke_templates 
                    (name, category, features, contour, skeleton, metadata, similarity_threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    template.name,
                    template.category,
                    features_blob,
                    contour_blob,
                    skeleton_blob,
                    metadata_blob,
                    template.similarity_threshold
                ))
                
                template_id = cursor.lastrowid
                
                # 添加特征索引
                self._add_feature_index(cursor, template_id, template.features)
                
                conn.commit()
                
                # 更新缓存
                template.id = template_id
                self._template_cache[template_id] = template
                
                self.logger.info(f"Template added: {template.name} (ID: {template_id})")
                return template_id
                
        except Exception as e:
            self.logger.error(f"Error adding template: {e}")
            raise
    
    def get_template(self, template_id: int) -> Optional[StrokeTemplate]:
        """
        获取笔触模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            笔触模板或None
        """
        # 检查缓存
        if template_id in self._template_cache:
            return self._template_cache[template_id]
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, name, category, features, contour, skeleton, 
                           metadata, similarity_threshold
                    FROM stroke_templates WHERE id = ?
                ''', (template_id,))
                
                row = cursor.fetchone()
                if row:
                    template = self._row_to_template(row)
                    self._template_cache[template_id] = template
                    return template
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting template {template_id}: {e}")
            return None
    
    def search_templates(self, category: str = None, 
                        feature_filters: Dict[str, Tuple[float, float]] = None,
                        limit: int = 100) -> List[StrokeTemplate]:
        """
        搜索笔触模板
        
        Args:
            category: 分类过滤
            feature_filters: 特征过滤 {feature_name: (min_value, max_value)}
            limit: 结果数量限制
            
        Returns:
            匹配的模板列表
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询
                query = '''
                    SELECT DISTINCT t.id, t.name, t.category, t.features, 
                           t.contour, t.skeleton, t.metadata, t.similarity_threshold
                    FROM stroke_templates t
                '''
                
                conditions = []
                params = []
                
                if category:
                    conditions.append('t.category = ?')
                    params.append(category)
                
                if feature_filters:
                    query += ' JOIN feature_index f ON t.id = f.template_id'
                    for feature_name, (min_val, max_val) in feature_filters.items():
                        conditions.append(
                            'f.feature_name = ? AND f.feature_value BETWEEN ? AND ?'
                        )
                        params.extend([feature_name, min_val, max_val])
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += f' LIMIT {limit}'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                templates = []
                for row in rows:
                    template = self._row_to_template(row)
                    templates.append(template)
                    # 更新缓存
                    self._template_cache[template.id] = template
                
                return templates
                
        except Exception as e:
            self.logger.error(f"Error searching templates: {e}")
            return []
    
    def update_template(self, template: StrokeTemplate) -> bool:
        """
        更新笔触模板
        
        Args:
            template: 笔触模板
            
        Returns:
            是否成功
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 序列化数据
                features_blob = pickle.dumps(template.features)
                contour_blob = pickle.dumps(template.contour)
                skeleton_blob = pickle.dumps(template.skeleton)
                metadata_blob = pickle.dumps(template.metadata)
                
                # 更新模板
                cursor.execute('''
                    UPDATE stroke_templates 
                    SET name = ?, category = ?, features = ?, contour = ?, 
                        skeleton = ?, metadata = ?, similarity_threshold = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    template.name,
                    template.category,
                    features_blob,
                    contour_blob,
                    skeleton_blob,
                    metadata_blob,
                    template.similarity_threshold,
                    template.id
                ))
                
                # 删除旧的特征索引
                cursor.execute('DELETE FROM feature_index WHERE template_id = ?', 
                             (template.id,))
                
                # 添加新的特征索引
                self._add_feature_index(cursor, template.id, template.features)
                
                conn.commit()
                
                # 更新缓存
                self._template_cache[template.id] = template
                
                self.logger.info(f"Template updated: {template.name} (ID: {template.id})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating template: {e}")
            return False
    
    def delete_template(self, template_id: int) -> bool:
        """
        删除笔触模板
        
        Args:
            template_id: 模板ID
            
        Returns:
            是否成功
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 删除特征索引
                cursor.execute('DELETE FROM feature_index WHERE template_id = ?', 
                             (template_id,))
                
                # 删除模板
                cursor.execute('DELETE FROM stroke_templates WHERE id = ?', 
                             (template_id,))
                
                conn.commit()
                
                # 清除缓存
                if template_id in self._template_cache:
                    del self._template_cache[template_id]
                
                self.logger.info(f"Template deleted: ID {template_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting template: {e}")
            return False
    
    def get_categories(self) -> List[str]:
        """
        获取所有分类
        
        Returns:
            分类列表
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT DISTINCT category FROM stroke_templates')
                rows = cursor.fetchall()
                
                return [row[0] for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            return []
    
    def get_template_count(self, category: str = None) -> int:
        """
        获取模板数量
        
        Args:
            category: 分类过滤
            
        Returns:
            模板数量
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                if category:
                    cursor.execute(
                        'SELECT COUNT(*) FROM stroke_templates WHERE category = ?',
                        (category,)
                    )
                else:
                    cursor.execute('SELECT COUNT(*) FROM stroke_templates')
                
                return cursor.fetchone()[0]
                
        except Exception as e:
            self.logger.error(f"Error getting template count: {e}")
            return 0
    
    def create_template_from_stroke(self, stroke: Stroke, name: str, 
                                  category: str) -> StrokeTemplate:
        """
        从笔触创建模板
        
        Args:
            stroke: 笔触对象
            name: 模板名称
            category: 分类
            
        Returns:
            笔触模板
        """
        # 提取骨架（简化实现）
        skeleton = self._extract_skeleton_from_stroke(stroke)
        
        # 创建模板
        template = StrokeTemplate(
            id=0,  # 将在添加到数据库时设置
            name=name,
            category=category,
            features=stroke.features.copy(),
            contour=stroke.contour,
            skeleton=skeleton,
            metadata={
                'source_stroke_id': stroke.id,
                'area': stroke.area,
                'perimeter': stroke.perimeter,
                'length': stroke.length,
                'width': stroke.width,
                'angle': stroke.angle,
                'confidence': stroke.confidence
            }
        )
        
        return template
    
    def batch_add_templates(self, templates: List[StrokeTemplate]) -> List[int]:
        """
        批量添加模板
        
        Args:
            templates: 模板列表
            
        Returns:
            模板ID列表
        """
        template_ids = []
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                for template in templates:
                    # 序列化数据
                    features_blob = pickle.dumps(template.features)
                    contour_blob = pickle.dumps(template.contour)
                    skeleton_blob = pickle.dumps(template.skeleton)
                    metadata_blob = pickle.dumps(template.metadata)
                    
                    # 插入模板
                    cursor.execute('''
                        INSERT INTO stroke_templates 
                        (name, category, features, contour, skeleton, metadata, similarity_threshold)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        template.name,
                        template.category,
                        features_blob,
                        contour_blob,
                        skeleton_blob,
                        metadata_blob,
                        template.similarity_threshold
                    ))
                    
                    template_id = cursor.lastrowid
                    template_ids.append(template_id)
                    
                    # 添加特征索引
                    self._add_feature_index(cursor, template_id, template.features)
                    
                    # 更新缓存
                    template.id = template_id
                    self._template_cache[template_id] = template
                
                conn.commit()
                self.logger.info(f"Batch added {len(templates)} templates")
                
        except Exception as e:
            self.logger.error(f"Error batch adding templates: {e}")
            raise
        
        return template_ids
    
    def clear_cache(self):
        """清除缓存"""
        self._template_cache.clear()
        self._feature_cache.clear()
        self.logger.info("Cache cleared")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # 总模板数
                cursor.execute('SELECT COUNT(*) FROM stroke_templates')
                total_templates = cursor.fetchone()[0]
                
                # 分类统计
                cursor.execute('''
                    SELECT category, COUNT(*) 
                    FROM stroke_templates 
                    GROUP BY category
                ''')
                category_stats = dict(cursor.fetchall())
                
                # 数据库文件大小
                db_size = Path(self.database_path).stat().st_size if Path(self.database_path).exists() else 0
                
                return {
                    'total_templates': total_templates,
                    'categories': category_stats,
                    'database_size_bytes': db_size,
                    'cache_size': len(self._template_cache)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def _add_feature_index(self, cursor, template_id: int, features: Dict[str, Any]):
        """添加特征索引"""
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)):
                cursor.execute('''
                    INSERT INTO feature_index (template_id, feature_name, feature_value)
                    VALUES (?, ?, ?)
                ''', (template_id, feature_name, float(feature_value)))
    
    def _row_to_template(self, row) -> StrokeTemplate:
        """将数据库行转换为模板对象"""
        template_id, name, category, features_blob, contour_blob, \
        skeleton_blob, metadata_blob, similarity_threshold = row
        
        # 反序列化数据
        features = pickle.loads(features_blob)
        contour = pickle.loads(contour_blob)
        skeleton = pickle.loads(skeleton_blob)
        metadata = pickle.loads(metadata_blob)
        
        return StrokeTemplate(
            id=template_id,
            name=name,
            category=category,
            features=features,
            contour=contour,
            skeleton=skeleton,
            metadata=metadata,
            similarity_threshold=similarity_threshold
        )
    
    def _extract_skeleton_from_stroke(self, stroke: Stroke) -> np.ndarray:
        """从笔触提取骨架（简化实现）"""
        # 这里应该使用更复杂的骨架提取算法
        # 简化实现：返回轮廓的中心线
        if len(stroke.contour) > 0:
            points = stroke.contour.reshape(-1, 2)
            # 简单地返回轮廓点作为骨架
            return points[::2]  # 每隔一个点取样
        else:
            return np.array([[stroke.center[0], stroke.center[1]]])