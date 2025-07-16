# -*- coding: utf-8 -*-
"""
笔画分类器

对检测到的笔画进行分类，识别不同类型的笔画
支持基于规则和机器学习的分类方法
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import math


@dataclass
class ClassificationResult:
    """
    分类结果数据结构
    
    Attributes:
        predicted_class (str): 预测类别
        confidence (float): 分类置信度
        class_probabilities (Dict[str, float]): 各类别概率
        features_used (List[str]): 使用的特征列表
        classification_details (Dict): 分类详细信息
    """
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    features_used: List[str]
    classification_details: Dict[str, Any]


class StrokeClassifier:
    """
    笔画分类器
    
    提供多种笔画分类方法，包括基于规则和机器学习的方法
    """
    
    def __init__(self, config):
        """
        初始化笔画分类器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 分类器配置
        self.classification_method = config['stroke_classification'].get('method', 'hybrid')
        self.model_path = config['stroke_classification'].get('model_path', 'models/stroke_classifier.pkl')
        self.scaler_path = config['stroke_classification'].get('scaler_path', 'models/feature_scaler.pkl')
        
        # 笔画类别定义
        self.stroke_classes = config['stroke_classification'].get('classes', [
            'horizontal',  # 横
            'vertical',    # 竖
            'left_falling', # 撇
            'right_falling', # 捺
            'dot',         # 点
            'hook',        # 钩
            'turning',     # 折
            'curve',       # 弯
            'complex'      # 复合笔画
        ])
        
        # 特征权重
        self.feature_weights = config['stroke_classification'].get('feature_weights', {
            'geometric': 0.4,
            'directional': 0.3,
            'curvature': 0.2,
            'texture': 0.1
        })
        
        # 分类阈值
        self.confidence_threshold = config['stroke_classification'].get('confidence_threshold', 0.7)
        
        # 机器学习模型
        self.ml_model = None
        self.feature_scaler = None
        
        # 加载预训练模型
        self._load_models()
    
    def classify_stroke(self, stroke_data: Dict[str, Any]) -> ClassificationResult:
        """
        对笔画进行分类
        
        Args:
            stroke_data (Dict): 笔画数据
            
        Returns:
            ClassificationResult: 分类结果
        """
        try:
            if self.classification_method == 'rule_based':
                return self._classify_by_rules(stroke_data)
            elif self.classification_method == 'ml_based':
                return self._classify_by_ml(stroke_data)
            elif self.classification_method == 'hybrid':
                return self._classify_hybrid(stroke_data)
            else:
                self.logger.error(f"Unknown classification method: {self.classification_method}")
                return self._create_default_result()
                
        except Exception as e:
            self.logger.error(f"Error classifying stroke: {str(e)}")
            return self._create_default_result()
    
    def _classify_by_rules(self, stroke_data: Dict[str, Any]) -> ClassificationResult:
        """
        基于规则的笔画分类
        
        Args:
            stroke_data (Dict): 笔画数据
            
        Returns:
            ClassificationResult: 分类结果
        """
        try:
            from utils.math_utils import ensure_scalar
            
            # 提取关键特征，确保都是标量值
            aspect_ratio_val = stroke_data.get('aspect_ratio', 1.0)
            aspect_ratio = ensure_scalar(aspect_ratio_val, default=1.0)
            
            orientation = ensure_scalar(stroke_data.get('orientation', 0.0), default=0.0)
            curvature_mean = ensure_scalar(stroke_data.get('curvature_mean', 0.0), default=0.0)
            curvature_std = ensure_scalar(stroke_data.get('curvature_std', 0.0), default=0.0)
            length = ensure_scalar(stroke_data.get('length', 0.0), default=0.0)
            area = ensure_scalar(stroke_data.get('area', 0.0), default=0.0)
            # 规则分类逻辑
            class_scores = {}
            # 点（小面积，接近圆形）
            if area < 100 and aspect_ratio < 1.5:
                class_scores['dot'] = 0.9
            # 横（水平方向，长宽比大）
            elif aspect_ratio > 3 and abs(orientation) < math.pi/6:
                class_scores['horizontal'] = 0.8 + min(0.2, aspect_ratio / 10)
            # 竖（垂直方向，长宽比大）
            elif aspect_ratio > 3 and abs(abs(orientation) - math.pi/2) < math.pi/6:
                class_scores['vertical'] = 0.8 + min(0.2, aspect_ratio / 10)
            # 撇（左下方向）
            elif (orientation > math.pi/4 and orientation < 3*math.pi/4 and 
                  aspect_ratio > 2):
                class_scores['left_falling'] = 0.7 + min(0.3, curvature_mean * 2)
            # 捺（右下方向）
            elif (orientation > -3*math.pi/4 and orientation < -math.pi/4 and 
                  aspect_ratio > 2):
                class_scores['right_falling'] = 0.7 + min(0.3, curvature_mean * 2)
            # 钩（高曲率变化）
            elif curvature_std > 0.5:
                class_scores['hook'] = 0.6 + min(0.4, curvature_std)
            # 折（中等曲率，角度变化）
            elif curvature_mean > 0.3 and curvature_std > 0.2:
                class_scores['turning'] = 0.6 + min(0.3, curvature_mean)
            # 弯（平滑曲线）
            elif curvature_mean > 0.2 and curvature_std < 0.3:
                class_scores['curve'] = 0.5 + min(0.4, curvature_mean)
            # 复合笔画（复杂形状）
            else:
                class_scores['complex'] = 0.5
            
            from utils.math_utils import ensure_scalar
            
            # 找到最高分类别
            if class_scores:
                predicted_class = max(class_scores.keys(), key=lambda k: class_scores[k])
                confidence = ensure_scalar(class_scores[predicted_class], default=0.5)
            else:
                predicted_class = 'complex'
                confidence = 0.5
                class_scores = {'complex': 0.5}
            
            # 归一化概率
            total_score = ensure_scalar(sum(class_scores.values()), default=1.0)
            if total_score > 0:
                class_probabilities = {k: ensure_scalar(v/total_score, default=0.0) 
                                     for k, v in class_scores.items()}
            else:
                class_probabilities = class_scores
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
                features_used=['aspect_ratio', 'orientation', 'curvature_mean', 'curvature_std'],
                classification_details={
                    'method': 'rule_based',
                    'raw_scores': class_scores
                }
            )
        except Exception as e:
            self.logger.error(f"Error in rule-based classification: {str(e)}")
            return self._create_default_result()
    
    def _classify_by_ml(self, stroke_data: Dict[str, Any]) -> ClassificationResult:
        """
        基于机器学习的笔画分类
        
        Args:
            stroke_data (Dict): 笔画数据
            
        Returns:
            ClassificationResult: 分类结果
        """
        try:
            if self.ml_model is None:
                self.logger.warning("ML model not loaded, falling back to rule-based classification")
                return self._classify_by_rules(stroke_data)
            
            # 提取特征向量
            feature_vector = self._extract_feature_vector(stroke_data)
            
            if feature_vector is None or len(feature_vector) == 0:
                self.logger.warning("Failed to extract features, using default classification")
                return self._create_default_result()
            
            # 特征标准化
            if self.feature_scaler:
                feature_vector = self.feature_scaler.transform([feature_vector])
            else:
                feature_vector = [feature_vector]
            
            from utils.math_utils import ensure_scalar
            
            # 预测
            predicted_class_idx = self.ml_model.predict(feature_vector)[0]
            predicted_class = self.stroke_classes[predicted_class_idx]
            
            # 获取概率
            if hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba(feature_vector)[0]
                # 确保概率值是标量
                class_probabilities = {self.stroke_classes[i]: ensure_scalar(prob, default=0.0) 
                                     for i, prob in enumerate(probabilities)}
                confidence = ensure_scalar(max(probabilities), default=0.5)
            else:
                class_probabilities = {predicted_class: 1.0}
                confidence = 1.0
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
                features_used=self._get_feature_names(),
                classification_details={
                    'method': 'ml_based',
                    'model_type': type(self.ml_model).__name__
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML-based classification: {str(e)}")
            return self._create_default_result()
    
    def _classify_hybrid(self, stroke_data: Dict[str, Any]) -> ClassificationResult:
        """
        混合分类方法
        
        Args:
            stroke_data (Dict): 笔画数据
            
        Returns:
            ClassificationResult: 分类结果
        """
        try:
            # 获取规则分类结果
            rule_result = self._classify_by_rules(stroke_data)
            
            # 获取ML分类结果
            ml_result = self._classify_by_ml(stroke_data)
            
            # 融合结果
            if rule_result.confidence > self.confidence_threshold:
                # 规则分类置信度高，使用规则结果
                final_result = rule_result
                final_result.classification_details['method'] = 'hybrid_rule_dominant'
            elif ml_result.confidence > self.confidence_threshold:
                # ML分类置信度高，使用ML结果
                final_result = ml_result
                final_result.classification_details['method'] = 'hybrid_ml_dominant'
            else:
                # 两者置信度都不高，加权融合
                final_result = self._fuse_classification_results(rule_result, ml_result)
                final_result.classification_details['method'] = 'hybrid_fused'
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in hybrid classification: {str(e)}")
            return self._create_default_result()
    
    def _fuse_classification_results(self, result1: ClassificationResult, 
                                   result2: ClassificationResult) -> ClassificationResult:
        """
        融合两个分类结果
        
        Args:
            result1 (ClassificationResult): 分类结果1
            result2 (ClassificationResult): 分类结果2
            
        Returns:
            ClassificationResult: 融合结果
        """
        try:
            from utils.math_utils import ensure_scalar
            
            # 加权融合概率
            weight1 = ensure_scalar(result1.confidence, default=0.0)
            weight2 = ensure_scalar(result2.confidence, default=0.0)
            total_weight = ensure_scalar(weight1 + weight2, default=1.0)
            
            if total_weight == 0:
                return self._create_default_result()
            
            # 归一化权重
            weight1 = ensure_scalar(weight1 / total_weight, default=0.5)
            weight2 = ensure_scalar(weight2 / total_weight, default=0.5)
            
            # 融合类别概率
            all_classes = set(result1.class_probabilities.keys()) | set(result2.class_probabilities.keys())
            fused_probabilities = {}
            
            for class_name in all_classes:
                prob1 = ensure_scalar(result1.class_probabilities.get(class_name, 0.0), default=0.0)
                prob2 = ensure_scalar(result2.class_probabilities.get(class_name, 0.0), default=0.0)
                fused_prob = ensure_scalar(weight1 * prob1 + weight2 * prob2, default=0.0)
                fused_probabilities[class_name] = fused_prob
            
            # 找到最高概率类别
            predicted_class = max(fused_probabilities.keys(), key=lambda k: fused_probabilities[k])
            confidence = ensure_scalar(fused_probabilities[predicted_class], default=0.5)
            
            # 合并使用的特征
            features_used = list(set(result1.features_used + result2.features_used))
            
            return ClassificationResult(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=fused_probabilities,
                features_used=features_used,
                classification_details={
                    'method': 'fused',
                    'fusion_weights': {'result1': weight1, 'result2': weight2},
                    'original_results': {
                        'result1': result1.predicted_class,
                        'result2': result2.predicted_class
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error fusing classification results: {str(e)}")
            return self._create_default_result()
    
    def train_ml_model(self, training_data: List[Tuple[Dict[str, Any], str]], 
                      model_type: str = 'random_forest') -> bool:
        """
        训练机器学习模型
        
        Args:
            training_data (List): 训练数据，(特征, 标签)对的列表
            model_type (str): 模型类型
            
        Returns:
            bool: 是否训练成功
        """
        try:
            if len(training_data) == 0:
                self.logger.error("No training data provided")
                return False
            
            from utils.math_utils import ensure_scalar
            
            # 提取特征和标签
            X = []
            y = []
            
            for features, label in training_data:
                feature_vector = self._extract_feature_vector(features)
                if feature_vector is not None:
                    # 确保特征向量中的每个值都是标量
                    feature_vector = [ensure_scalar(x, default=0.0) for x in feature_vector]
                    X.append(feature_vector)
                    y.append(label)
            
            if len(X) == 0:
                self.logger.error("No valid features extracted from training data")
                return False
            
            # 转换为numpy数组并确保数值类型
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            
            # 特征标准化
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # 分割训练和测试数据
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 创建模型
            if model_type == 'random_forest':
                self.ml_model = RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight='balanced'
                )
            elif model_type == 'svm':
                self.ml_model = SVC(
                    kernel='rbf', probability=True, random_state=42, class_weight='balanced'
                )
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return False
            
            # 训练模型
            self.ml_model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            self.logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # 保存模型
            self._save_models()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {str(e)}")
            return False
    
    def _extract_feature_vector(self, stroke_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        从笔画数据中提取特征向量
        
        Args:
            stroke_data (Dict): 笔画数据
            
        Returns:
            np.ndarray: 特征向量
        """
        try:
            features = []
            
            # 几何特征
            features.extend([
                stroke_data.get('area', 0.0),
                stroke_data.get('perimeter', 0.0),
                stroke_data.get('aspect_ratio', 1.0),
                stroke_data.get('extent', 0.0),
                stroke_data.get('solidity', 0.0),
                stroke_data.get('circularity', 0.0),
                stroke_data.get('rectangularity', 0.0)
            ])
            
            # 方向特征
            features.extend([
                stroke_data.get('orientation', 0.0),
                stroke_data.get('major_axis_length', 0.0),
                stroke_data.get('minor_axis_length', 0.0),
                stroke_data.get('eccentricity', 0.0)
            ])
            
            # 曲率特征
            features.extend([
                stroke_data.get('curvature_mean', 0.0),
                stroke_data.get('curvature_std', 0.0),
                stroke_data.get('curvature_max', 0.0),
                stroke_data.get('curvature_min', 0.0)
            ])
            
            # 纹理特征
            texture_features = stroke_data.get('texture_features', {})
            features.extend([
                texture_features.get('contrast', 0.0),
                texture_features.get('energy', 0.0),
                texture_features.get('homogeneity', 0.0),
                texture_features.get('correlation', 0.0)
            ])
            
            # 动态特征统计
            width_profile = stroke_data.get('width_profile')
            if width_profile is not None and len(width_profile) > 0:
                features.extend([
                    np.mean(width_profile),
                    np.std(width_profile),
                    np.max(width_profile),
                    np.min(width_profile)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting feature vector: {str(e)}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            List[str]: 特征名称
        """
        return [
            'area', 'perimeter', 'aspect_ratio', 'extent', 'solidity', 'circularity', 'rectangularity',
            'orientation', 'major_axis_length', 'minor_axis_length', 'eccentricity',
            'curvature_mean', 'curvature_std', 'curvature_max', 'curvature_min',
            'texture_contrast', 'texture_energy', 'texture_homogeneity', 'texture_correlation',
            'width_mean', 'width_std', 'width_max', 'width_min'
        ]
    
    def _load_models(self):
        """
        加载预训练模型
        """
        try:
            model_path = Path(self.model_path)
            scaler_path = Path(self.scaler_path)
            
            if model_path.exists():
                self.ml_model = joblib.load(model_path)
                self.logger.info(f"Loaded ML model from {model_path}")
            
            if scaler_path.exists():
                self.feature_scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded feature scaler from {scaler_path}")
                
        except Exception as e:
            self.logger.warning(f"Error loading models: {str(e)}")
    
    def _save_models(self):
        """
        保存训练好的模型
        """
        try:
            model_path = Path(self.model_path)
            scaler_path = Path(self.scaler_path)
            
            # 确保目录存在
            model_path.parent.mkdir(parents=True, exist_ok=True)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.ml_model:
                joblib.dump(self.ml_model, model_path)
                self.logger.info(f"Saved ML model to {model_path}")
            
            if self.feature_scaler:
                joblib.dump(self.feature_scaler, scaler_path)
                self.logger.info(f"Saved feature scaler to {scaler_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def _create_default_result(self) -> ClassificationResult:
        """
        创建默认分类结果
        
        Returns:
            ClassificationResult: 默认结果
        """
        return ClassificationResult(
            predicted_class='complex',
            confidence=0.5,
            class_probabilities={'complex': 1.0},
            features_used=[],
            classification_details={'method': 'default'}
        )
    
    def get_class_statistics(self, classification_results: List[ClassificationResult]) -> Dict[str, Any]:
        """
        获取分类统计信息
        
        Args:
            classification_results (List): 分类结果列表
            
        Returns:
            Dict: 统计信息
        """
        try:
            if not classification_results:
                return {}
            
            # 统计各类别数量
            class_counts = {}
            confidence_sum = {}
            
            for result in classification_results:
                class_name = result.predicted_class
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidence_sum[class_name] = confidence_sum.get(class_name, 0) + result.confidence
            
            # 计算平均置信度
            avg_confidence = {}
            for class_name, count in class_counts.items():
                avg_confidence[class_name] = confidence_sum[class_name] / count
            
            # 总体统计
            total_strokes = len(classification_results)
            overall_confidence = sum(r.confidence for r in classification_results) / total_strokes
            
            return {
                'total_strokes': total_strokes,
                'class_distribution': class_counts,
                'class_percentages': {k: v/total_strokes*100 for k, v in class_counts.items()},
                'average_confidence_by_class': avg_confidence,
                'overall_average_confidence': overall_confidence,
                'most_common_class': max(class_counts.keys(), key=lambda k: class_counts[k]),
                'least_common_class': min(class_counts.keys(), key=lambda k: class_counts[k])
            }
            
        except Exception as e:
            self.logger.error(f"Error computing class statistics: {str(e)}")
            return {}