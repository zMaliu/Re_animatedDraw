# -*- coding: utf-8 -*-
"""
笔触分类模块

提供笔触的自动分类和类别管理功能
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from ..stroke_extraction.stroke_detector import Stroke
from .stroke_database import StrokeTemplate


class StrokeCategory(Enum):
    """笔触分类枚举"""
    OUTLINE = "outline"          # 轮廓线
    FILL = "fill"                # 填充
    DETAIL = "detail"            # 细节
    SHADOW = "shadow"            # 阴影
    HIGHLIGHT = "highlight"      # 高光
    TEXTURE = "texture"          # 纹理
    BACKGROUND = "background"    # 背景
    FOREGROUND = "foreground"    # 前景
    UNKNOWN = "unknown"          # 未知


@dataclass
class ClassificationResult:
    """分类结果数据结构"""
    category: StrokeCategory
    confidence: float
    probabilities: Dict[StrokeCategory, float]
    features_used: List[str]
    classification_details: Dict[str, Any]


class StrokeClassifier:
    """笔触分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化笔触分类器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分类器参数
        self.classifier_type = config.get('classifier_type', 'random_forest')
        self.model_path = config.get('model_path', './models/stroke_classifier.pkl')
        self.scaler_path = config.get('scaler_path', './models/stroke_scaler.pkl')
        
        # 特征选择
        self.feature_names = [
            'area', 'perimeter', 'length', 'width', 'aspect_ratio',
            'solidity', 'extent', 'orientation', 'eccentricity',
            'compactness', 'convexity', 'roughness', 'symmetry',
            'position_x', 'position_y', 'relative_size', 'density'
        ]
        
        # 初始化模型
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 尝试加载预训练模型
        self._load_model()
        
        # 分类规则（基于规则的后备分类器）
        self.rule_based_classifier = RuleBasedClassifier(config)
        
    def classify_stroke(self, stroke: Stroke) -> ClassificationResult:
        """
        分类单个笔触
        
        Args:
            stroke: 输入笔触
            
        Returns:
            分类结果
        """
        try:
            # 提取特征
            features = self._extract_classification_features(stroke)
            feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            if self.is_trained and self.classifier is not None:
                # 使用机器学习分类器
                result = self._ml_classify(feature_vector, features)
            else:
                # 使用基于规则的分类器
                result = self.rule_based_classifier.classify(stroke, features)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying stroke {stroke.id}: {e}")
            return ClassificationResult(
                category=StrokeCategory.UNKNOWN,
                confidence=0.0,
                probabilities={cat: 0.0 for cat in StrokeCategory},
                features_used=self.feature_names,
                classification_details={'error': str(e)}
            )
    
    def classify_strokes(self, strokes: List[Stroke]) -> List[ClassificationResult]:
        """
        批量分类笔触
        
        Args:
            strokes: 笔触列表
            
        Returns:
            分类结果列表
        """
        results = []
        
        if not strokes:
            return results
        
        try:
            # 批量提取特征
            all_features = []
            feature_dicts = []
            
            for stroke in strokes:
                features = self._extract_classification_features(stroke)
                feature_vector = [features[name] for name in self.feature_names]
                all_features.append(feature_vector)
                feature_dicts.append(features)
            
            feature_matrix = np.array(all_features)
            
            if self.is_trained and self.classifier is not None:
                # 使用机器学习分类器批量处理
                results = self._ml_classify_batch(feature_matrix, feature_dicts)
            else:
                # 使用基于规则的分类器
                for i, stroke in enumerate(strokes):
                    result = self.rule_based_classifier.classify(stroke, feature_dicts[i])
                    results.append(result)
            
        except Exception as e:
            self.logger.error(f"Error in batch classification: {e}")
            # 回退到单个分类
            for stroke in strokes:
                result = self.classify_stroke(stroke)
                results.append(result)
        
        return results
    
    def train_classifier(self, training_data: List[Tuple[Stroke, StrokeCategory]], 
                        validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练分类器
        
        Args:
            training_data: 训练数据 [(stroke, category), ...]
            validation_split: 验证集比例
            
        Returns:
            训练结果统计
        """
        if len(training_data) < 10:
            raise ValueError("Training data too small (minimum 10 samples required)")
        
        try:
            # 提取特征和标签
            X = []
            y = []
            
            for stroke, category in training_data:
                features = self._extract_classification_features(stroke)
                feature_vector = [features[name] for name in self.feature_names]
                X.append(feature_vector)
                y.append(category.value)
            
            X = np.array(X)
            y = np.array(y)
            
            # 分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # 特征标准化
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # 初始化分类器
            if self.classifier_type == 'random_forest':
                self.classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
            elif self.classifier_type == 'svm':
                self.classifier = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    class_weight='balanced'
                )
            else:
                raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
            
            # 训练模型
            self.classifier.fit(X_train_scaled, y_train)
            
            # 验证模型
            y_pred = self.classifier.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            # 生成分类报告
            report = classification_report(y_val, y_pred, output_dict=True)
            
            # 保存模型
            self._save_model()
            
            self.is_trained = True
            
            training_stats = {
                'accuracy': accuracy,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'classification_report': report,
                'feature_names': self.feature_names
            }
            
            self.logger.info(f"Classifier trained successfully. Accuracy: {accuracy:.3f}")
            return training_stats
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {e}")
            raise
    
    def update_classifier(self, new_data: List[Tuple[Stroke, StrokeCategory]]) -> bool:
        """
        增量更新分类器
        
        Args:
            new_data: 新的训练数据
            
        Returns:
            是否成功更新
        """
        try:
            if not self.is_trained:
                self.logger.warning("Classifier not trained yet. Use train_classifier first.")
                return False
            
            # 提取新特征
            X_new = []
            y_new = []
            
            for stroke, category in new_data:
                features = self._extract_classification_features(stroke)
                feature_vector = [features[name] for name in self.feature_names]
                X_new.append(feature_vector)
                y_new.append(category.value)
            
            X_new = np.array(X_new)
            y_new = np.array(y_new)
            
            # 标准化新特征
            X_new_scaled = self.scaler.transform(X_new)
            
            # 对于支持增量学习的分类器，可以使用partial_fit
            # 这里简化为重新训练（在实际应用中可以优化）
            if hasattr(self.classifier, 'partial_fit'):
                self.classifier.partial_fit(X_new_scaled, y_new)
            else:
                self.logger.info("Classifier doesn't support incremental learning. Consider retraining.")
                return False
            
            # 保存更新后的模型
            self._save_model()
            
            self.logger.info(f"Classifier updated with {len(new_data)} new samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating classifier: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_trained or self.classifier is None:
            return {}
        
        try:
            if hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                return dict(zip(self.feature_names, importances))
            else:
                self.logger.warning("Classifier doesn't support feature importance")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _extract_classification_features(self, stroke: Stroke) -> Dict[str, float]:
        """
        提取分类特征
        
        Args:
            stroke: 笔触
            
        Returns:
            特征字典
        """
        # 基本几何特征
        features = {
            'area': stroke.area,
            'perimeter': stroke.perimeter,
            'length': stroke.length,
            'width': stroke.width,
            'aspect_ratio': stroke.length / max(stroke.width, 1e-6),
            'position_x': stroke.center[0],
            'position_y': stroke.center[1]
        }
        
        # 计算额外特征
        try:
            # 紧实度 (4π*面积/周长²)
            features['compactness'] = float((4 * np.pi * stroke.area) / max(stroke.perimeter ** 2, 1e-6))
            
            # 凸性 (凸包面积/实际面积)
            if hasattr(stroke, 'convex_hull_area'):
                features['convexity'] = stroke.area / max(stroke.convex_hull_area, 1e-6)
            else:
                features['convexity'] = 0.8  # 默认值
            
            # 实体性 (面积/边界框面积)
            bbox_area = stroke.length * stroke.width
            features['solidity'] = stroke.area / max(bbox_area, 1e-6)
            
            # 范围 (面积/边界框面积的另一种计算)
            features['extent'] = features['solidity']
            
            # 方向 (主轴角度)
            features['orientation'] = stroke.angle
            
            # 偏心率 (基于椭圆拟合)
            features['eccentricity'] = self._calculate_eccentricity(stroke)
            
            # 粗糙度 (周长²/面积)
            features['roughness'] = (stroke.perimeter ** 2) / max(stroke.area, 1e-6)
            
            # 对称性 (简化计算)
            features['symmetry'] = self._calculate_symmetry(stroke)
            
            # 相对大小 (相对于图像的大小)
            if hasattr(stroke, 'image_size'):
                image_area = stroke.image_size[0] * stroke.image_size[1]
                features['relative_size'] = stroke.area / max(image_area, 1e-6)
            else:
                features['relative_size'] = 0.01  # 默认值
            
            # 密度 (面积/凸包面积)
            features['density'] = features['convexity']
            
        except Exception as e:
            self.logger.warning(f"Error calculating additional features: {e}")
            # 设置默认值
            default_features = {
                'compactness': 0.5,
                'convexity': 0.8,
                'solidity': 0.7,
                'extent': 0.7,
                'orientation': 0.0,
                'eccentricity': 0.5,
                'roughness': 1.0,
                'symmetry': 0.5,
                'relative_size': 0.01,
                'density': 0.8
            }
            features.update(default_features)
        
        return features
    
    def _calculate_eccentricity(self, stroke: Stroke) -> float:
        """
        计算偏心率
        
        Args:
            stroke: 笔触
            
        Returns:
            偏心率
        """
        try:
            # 简化计算：基于长宽比
            aspect_ratio = stroke.length / max(stroke.width, 1e-6)
            # 将长宽比转换为偏心率 (0-1)
            eccentricity = min(1.0, (aspect_ratio - 1.0) / max(aspect_ratio, 1.0))
            return eccentricity
        except:
            return 0.5
    
    def _calculate_symmetry(self, stroke: Stroke) -> float:
        """
        计算对称性
        
        Args:
            stroke: 笔触
            
        Returns:
            对称性分数
        """
        try:
            # 简化实现：基于形状的规则性
            compactness = (4 * np.pi * stroke.area) / max(stroke.perimeter ** 2, 1e-6)
            return min(1.0, compactness * 2)  # 越紧实越对称
        except:
            return 0.5
    
    def _ml_classify(self, feature_vector: np.ndarray, 
                    features: Dict[str, float]) -> ClassificationResult:
        """
        使用机器学习分类器进行分类
        
        Args:
            feature_vector: 特征向量
            features: 特征字典
            
        Returns:
            分类结果
        """
        try:
            # 标准化特征
            feature_scaled = self.scaler.transform(feature_vector)
            
            # 预测类别
            predicted_class = self.classifier.predict(feature_scaled)[0]
            
            # 获取概率
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(feature_scaled)[0]
                class_names = self.classifier.classes_
                prob_dict = {}
                for i, class_name in enumerate(class_names):
                    try:
                        category = StrokeCategory(class_name)
                        prob_dict[category] = probabilities[i]
                    except ValueError:
                        continue
                
                confidence = max(probabilities)
            else:
                prob_dict = {StrokeCategory(predicted_class): 1.0}
                confidence = 1.0
            
            # 转换预测类别
            try:
                predicted_category = StrokeCategory(predicted_class)
            except ValueError:
                predicted_category = StrokeCategory.UNKNOWN
                confidence = 0.0
            
            return ClassificationResult(
                category=predicted_category,
                confidence=confidence,
                probabilities=prob_dict,
                features_used=self.feature_names,
                classification_details={
                    'method': 'machine_learning',
                    'classifier_type': self.classifier_type,
                    'features': features
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML classification: {e}")
            return ClassificationResult(
                category=StrokeCategory.UNKNOWN,
                confidence=0.0,
                probabilities={cat: 0.0 for cat in StrokeCategory},
                features_used=self.feature_names,
                classification_details={'error': str(e)}
            )
    
    def _ml_classify_batch(self, feature_matrix: np.ndarray, 
                          feature_dicts: List[Dict[str, float]]) -> List[ClassificationResult]:
        """
        批量机器学习分类
        
        Args:
            feature_matrix: 特征矩阵
            feature_dicts: 特征字典列表
            
        Returns:
            分类结果列表
        """
        try:
            # 标准化特征
            features_scaled = self.scaler.transform(feature_matrix)
            
            # 批量预测
            predicted_classes = self.classifier.predict(features_scaled)
            
            if hasattr(self.classifier, 'predict_proba'):
                probabilities_matrix = self.classifier.predict_proba(features_scaled)
                class_names = self.classifier.classes_
            else:
                probabilities_matrix = None
                class_names = None
            
            results = []
            for i, predicted_class in enumerate(predicted_classes):
                # 处理概率
                if probabilities_matrix is not None:
                    probabilities = probabilities_matrix[i]
                    prob_dict = {}
                    for j, class_name in enumerate(class_names):
                        try:
                            category = StrokeCategory(class_name)
                            prob_dict[category] = probabilities[j]
                        except ValueError:
                            continue
                    confidence = max(probabilities)
                else:
                    prob_dict = {StrokeCategory(predicted_class): 1.0}
                    confidence = 1.0
                
                # 转换预测类别
                try:
                    predicted_category = StrokeCategory(predicted_class)
                except ValueError:
                    predicted_category = StrokeCategory.UNKNOWN
                    confidence = 0.0
                
                result = ClassificationResult(
                    category=predicted_category,
                    confidence=confidence,
                    probabilities=prob_dict,
                    features_used=self.feature_names,
                    classification_details={
                        'method': 'machine_learning_batch',
                        'classifier_type': self.classifier_type,
                        'features': feature_dicts[i]
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch ML classification: {e}")
            # 回退到单个分类
            results = []
            for i in range(len(feature_matrix)):
                feature_vector = feature_matrix[i:i+1]
                result = self._ml_classify(feature_vector, feature_dicts[i])
                results.append(result)
            return results
    
    def _save_model(self):
        """保存模型"""
        try:
            # 确保目录存在
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存分类器
            joblib.dump(self.classifier, self.model_path)
            
            # 保存标准化器
            joblib.dump(self.scaler, self.scaler_path)
            
            self.logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """加载模型"""
        try:
            if Path(self.model_path).exists() and Path(self.scaler_path).exists():
                # 加载分类器
                self.classifier = joblib.load(self.model_path)
                
                # 加载标准化器
                self.scaler = joblib.load(self.scaler_path)
                
                self.is_trained = True
                self.logger.info(f"Model loaded from {self.model_path}")
            else:
                self.logger.info("No pre-trained model found")
                
        except Exception as e:
            self.logger.warning(f"Error loading model: {e}")
            self.is_trained = False


class RuleBasedClassifier:
    """基于规则的笔触分类器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基于规则的分类器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分类规则阈值
        self.thresholds = config.get('rule_thresholds', {
            'large_area_ratio': 0.1,      # 大面积阈值
            'thin_aspect_ratio': 5.0,     # 细长比阈值
            'small_area_ratio': 0.001,    # 小面积阈值
            'edge_distance_ratio': 0.1,   # 边缘距离阈值
            'high_roughness': 2.0,        # 高粗糙度阈值
            'low_compactness': 0.3        # 低紧实度阈值
        })
    
    def classify(self, stroke: Stroke, features: Dict[str, float]) -> ClassificationResult:
        """
        基于规则分类笔触
        
        Args:
            stroke: 笔触
            features: 特征字典
            
        Returns:
            分类结果
        """
        try:
            # 应用分类规则
            category, confidence, reasoning = self._apply_rules(features)
            
            # 构建概率分布
            probabilities = {cat: 0.0 for cat in StrokeCategory}
            probabilities[category] = confidence
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                probabilities=probabilities,
                features_used=list(features.keys()),
                classification_details={
                    'method': 'rule_based',
                    'reasoning': reasoning,
                    'features': features
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in rule-based classification: {e}")
            return ClassificationResult(
                category=StrokeCategory.UNKNOWN,
                confidence=0.0,
                probabilities={cat: 0.0 for cat in StrokeCategory},
                features_used=list(features.keys()),
                classification_details={'error': str(e)}
            )
    
    def _apply_rules(self, features: Dict[str, float]) -> Tuple[StrokeCategory, float, str]:
        """
        应用分类规则
        
        Args:
            features: 特征字典
            
        Returns:
            (分类, 置信度, 推理过程)
        """
        reasoning = []
        
        # 规则1: 大面积 -> 填充或背景
        if features.get('relative_size', 0) > self.thresholds['large_area_ratio']:
            reasoning.append("Large area detected")
            if features.get('position_x', 0.5) < 0.2 or features.get('position_x', 0.5) > 0.8:
                return StrokeCategory.BACKGROUND, 0.8, "; ".join(reasoning + ["Edge position -> Background"])
            else:
                return StrokeCategory.FILL, 0.7, "; ".join(reasoning + ["Central position -> Fill"])
        
        # 规则2: 细长形状 -> 轮廓线
        if features.get('aspect_ratio', 1) > self.thresholds['thin_aspect_ratio']:
            reasoning.append("High aspect ratio detected")
            return StrokeCategory.OUTLINE, 0.8, "; ".join(reasoning + ["Thin shape -> Outline"])
        
        # 规则3: 小面积 + 高粗糙度 -> 细节或纹理
        if (features.get('relative_size', 0) < self.thresholds['small_area_ratio'] and 
            features.get('roughness', 1) > self.thresholds['high_roughness']):
            reasoning.append("Small area with high roughness")
            return StrokeCategory.DETAIL, 0.7, "; ".join(reasoning + ["Small rough -> Detail"])
        
        # 规则4: 低紧实度 -> 纹理
        if features.get('compactness', 0.5) < self.thresholds['low_compactness']:
            reasoning.append("Low compactness detected")
            return StrokeCategory.TEXTURE, 0.6, "; ".join(reasoning + ["Irregular shape -> Texture"])
        
        # 规则5: 边缘位置 -> 轮廓
        edge_distance = min(
            features.get('position_x', 0.5),
            features.get('position_y', 0.5),
            1 - features.get('position_x', 0.5),
            1 - features.get('position_y', 0.5)
        )
        if edge_distance < self.thresholds['edge_distance_ratio']:
            reasoning.append("Near edge position")
            return StrokeCategory.OUTLINE, 0.6, "; ".join(reasoning + ["Edge position -> Outline"])
        
        # 规则6: 中等大小 + 规则形状 -> 前景
        if (0.001 < features.get('relative_size', 0) < 0.05 and 
            features.get('compactness', 0) > 0.5):
            reasoning.append("Medium size with regular shape")
            return StrokeCategory.FOREGROUND, 0.6, "; ".join(reasoning + ["Regular medium -> Foreground"])
        
        # 默认分类
        reasoning.append("No specific rule matched")
        return StrokeCategory.UNKNOWN, 0.3, "; ".join(reasoning + ["Default classification"])