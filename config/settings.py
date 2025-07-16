# -*- coding: utf-8 -*-
"""
配置设置模块

定义中国画动画构建算法的各种参数和配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    配置管理类
    
    管理算法的所有参数配置，支持从文件加载和默认值
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path (str, optional): 配置文件路径
        """
        # 设置默认配置
        self._set_default_config()
        
        # 如果提供了配置文件路径，则加载配置
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
    
    def _set_default_config(self):
        """
        设置默认配置参数
        """
        # 图像处理参数
        self.image = {
            'max_size': 1024,  # 最大图像尺寸
            'min_size': 256,   # 最小图像尺寸
            'gaussian_blur_kernel': 3,  # 高斯模糊核大小
            'bilateral_filter_d': 9,    # 双边滤波参数
            'bilateral_filter_sigma_color': 75,
            'bilateral_filter_sigma_space': 75,
        }
        
        # 笔画检测参数
        self.stroke_detection = {
            'adaptive_block_size': 25,    # 自适应阈值分割块大小
            'adaptive_c': 10,            # 自适应阈值常数
            'region_growing_threshold': 2.0,  # 区域生长阈值
            'min_region_size': 100,      # 最小区域大小
            'min_stroke_length': 20,     # 最小笔画长度
            'max_stroke_width': 50,      # 最大笔画宽度
            'skeleton_method': 'zhang_suen',  # 骨架提取方法
            'contour_approximation_epsilon': 0.02,  # 轮廓近似精度
        }
        
        # 笔画库参数
        self.stroke_library = {
            'database_path': 'data/stroke_library/',
            'feature_vector_size': 128,   # 特征向量维度
            'similarity_threshold': 0.8,  # 相似度阈值
            'max_matches_per_stroke': 5,  # 每个笔画的最大匹配数
            'stroke_categories': [        # 笔画类别
                'horizontal',  # 横
                'vertical',    # 竖
                'dot',         # 点
                'hook',        # 钩
                'rising',      # 提
                'falling',     # 撇
                'turning',     # 折
                'curve'        # 弯
            ]
        }
        
        # 笔画排序参数
        self.stroke_ordering = {
            'composition_stages': 3,      # 构图阶段数
            'evolution_population_size': 50,  # 进化算法种群大小
            'evolution_generations': 100,     # 进化代数
            'mutation_rate': 0.1,            # 变异率
            'crossover_rate': 0.8,           # 交叉率
            'fitness_weights': {             # 适应度权重
                'artistic_rules': 0.4,       # 艺术规则权重
                'spatial_coherence': 0.3,    # 空间连贯性权重
                'temporal_smoothness': 0.3   # 时间平滑性权重
            },
            'artistic_rules': {              # 艺术规则
                'top_to_bottom': True,       # 从上到下
                'left_to_right': True,       # 从左到右
                'outside_to_inside': True,   # 从外到内
                'main_before_detail': True   # 主体先于细节
            }
        }
        
        # 动画生成参数
        self.animation = {
            'fps': 24,                    # 帧率
            'stroke_duration': 1.0,       # 单个笔画持续时间(秒)
            'pause_between_strokes': 0.2, # 笔画间暂停时间(秒)
            'brush_size_variation': 0.3,  # 笔刷大小变化
            'opacity_variation': 0.2,     # 透明度变化
            'speed_variation': 0.4,       # 速度变化
            'output_format': 'mp4',       # 输出格式
            'quality': 'high',            # 输出质量
            'background_color': (255, 255, 255),  # 背景颜色
            'stroke_color_mode': 'original',       # 笔画颜色模式
        }
        
        # 可视化参数
        self.visualization = {
            'stroke_color_map': 'viridis',  # 笔画颜色映射
            'show_stroke_order': True,      # 显示笔画顺序
            'show_skeleton': False,         # 显示骨架
            'show_contour': True,          # 显示轮廓
            'line_thickness': 2,           # 线条粗细
            'font_size': 12,               # 字体大小
        }
        
        # 性能参数
        self.performance = {
            'use_multiprocessing': True,   # 使用多进程
            'num_workers': 4,              # 工作进程数
            'chunk_size': 10,              # 数据块大小
            'memory_limit_mb': 2048,       # 内存限制(MB)
        }
        
        # 调试参数
        self.debug = {
            'save_intermediate_results': True,  # 保存中间结果
            'verbose_logging': False,           # 详细日志
            'profile_performance': False,       # 性能分析
            'visualize_steps': False,          # 可视化步骤
        }
    
    def _load_config_file(self, config_path: str):
        """
        从文件加载配置
        
        Args:
            config_path (str): 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    # 假设是Python文件
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config_data = {k: v for k, v in vars(config_module).items() 
                                 if not k.startswith('_')}
                
                # 更新配置
                self._update_config(config_data)
                
        except Exception as e:
            print(f"警告: 加载配置文件失败 {config_path}: {str(e)}")
            print("使用默认配置")
    
    def _update_config(self, config_data: Dict[str, Any]):
        """
        更新配置数据
        
        Args:
            config_data (dict): 新的配置数据
        """
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(getattr(self, section), dict):
                getattr(self, section).update(values)
            else:
                setattr(self, section, values)
    
    def get(self, section: str, key: str = None, default=None):
        """
        获取配置值
        
        Args:
            section (str): 配置段名
            key (str, optional): 配置键名
            default: 默认值
            
        Returns:
            配置值
        """
        if not hasattr(self, section):
            return default
        
        section_config = getattr(self, section)
        
        if key is None:
            return section_config
        
        if isinstance(section_config, dict):
            return section_config.get(key, default)
        else:
            return default
    
    def set(self, section: str, key: str, value):
        """
        设置配置值
        
        Args:
            section (str): 配置段名
            key (str): 配置键名
            value: 配置值
        """
        if not hasattr(self, section):
            setattr(self, section, {})
        
        section_config = getattr(self, section)
        if isinstance(section_config, dict):
            section_config[key] = value
        else:
            setattr(self, section, {key: value})
    
    def save_config(self, output_path: str):
        """
        保存配置到文件
        
        Args:
            output_path (str): 输出文件路径
        """
        config_data = {
            'image': self.image,
            'stroke_detection': self.stroke_detection,
            'stroke_library': self.stroke_library,
            'stroke_ordering': self.stroke_ordering,
            'animation': self.animation,
            'visualization': self.visualization,
            'performance': self.performance,
            'debug': self.debug,
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            print(f"配置已保存到: {output_path}")
        except Exception as e:
            print(f"保存配置失败: {str(e)}")
    
    def __str__(self):
        """
        返回配置的字符串表示
        """
        config_str = "Configuration Settings:\n"
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, dict):
                    config_str += f"\n{attr_name}:\n"
                    for key, value in attr_value.items():
                        config_str += f"  {key}: {value}\n"
                else:
                    config_str += f"{attr_name}: {attr_value}\n"
        return config_str


# 创建默认配置实例
default_config = Config()