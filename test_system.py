#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
用于验证中国画笔刷动画构建系统的基本功能
"""

import os
import sys
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """
    测试所有模块的导入
    """
    print("测试模块导入...")
    
    try:
        # 测试核心模块导入
        print("  - 导入核心模块...")
        from core.stroke_extraction import StrokeDetector, StrokeExtractor, StrokeSegmenter
        from core.stroke_database import StrokeClassifier, StrokeDatabase, StrokeMatcher
        from core.stroke_ordering import StrokeOrganizer, OrderingOptimizer, RuleBasedOrdering
        from core.animation import PaintingAnimator, BrushSimulator, AnimationRenderer, TimelineManager
        print("    ✓ 核心模块导入成功")
        
        # 测试工具模块导入
        print("  - 导入工具模块...")
        from utils.image_utils import ImageProcessor, ImageEnhancer, ImageAnalyzer
        from utils.math_utils import MathUtils, GeometryUtils, StatisticsUtils
        from utils.visualization import ColorUtils, Visualizer, PlotManager
        from utils.file_utils import FileManager, DataSerializer, ArchiveManager, ConfigManager
        from utils.logging_utils import LogManager, ContextLogger, LogAnalyzer, setup_logging
        from utils.performance import PerformanceProfiler, SystemMonitor, Timer, MemoryMonitor
        print("    ✓ 工具模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"    ✗ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"    ✗ 导入时发生未知错误: {e}")
        return False

def test_dependencies():
    """
    测试依赖包
    """
    print("\n测试依赖包...")
    
    required_packages = [
        'numpy', 'cv2', 'PIL', 'scipy', 'sklearn', 
        'matplotlib', 'pandas', 'tqdm', 'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True

def test_main_class():
    """
    测试主类初始化
    """
    print("\n测试主类初始化...")
    
    try:
        from main import ChinesePaintingAnimator
        import json
        
        # 加载配置文件
        config_path = project_root / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 使用配置文件初始化
        animator = ChinesePaintingAnimator(config)
        print("  ✓ ChinesePaintingAnimator 初始化成功")
        
        # 测试配置获取
        default_config = animator._get_default_config()
        print(f"  ✓ 配置加载成功 ({len(config)} 个配置项)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 主类初始化失败: {e}")
        traceback.print_exc()
        return False

def test_config_file():
    """
    测试配置文件
    """
    print("\n测试配置文件...")
    
    config_path = project_root / "config.json"
    
    if not config_path.exists():
        print("  ✗ config.json 文件不存在")
        return False
    
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_sections = [
            'input', 'output', 'stroke_detection', 'stroke_database',
            'stroke_ordering', 'animation', 'rendering', 'performance'
        ]
        
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section} 配置节存在")
            else:
                print(f"  ✗ {section} 配置节缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ 配置文件解析失败: {e}")
        return False

def test_directory_structure():
    """
    测试目录结构
    """
    print("\n测试目录结构...")
    
    required_dirs = [
        'core',
        'core/stroke_extraction',
        'core/stroke_database', 
        'core/stroke_ordering',
        'core/animation',
        'utils'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (缺失)")
            return False
    
    return True

def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("中国画笔刷动画构建系统 - 系统测试")
    print("=" * 60)
    
    tests = [
        ("目录结构", test_directory_structure),
        ("依赖包", test_dependencies),
        ("模块导入", test_imports),
        ("配置文件", test_config_file),
        ("主类初始化", test_main_class)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} 测试失败")
        except Exception as e:
            print(f"\n❌ {test_name} 测试出错: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())