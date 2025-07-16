#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中国画笔刷动画构建系统 - 运行示例

这个脚本演示如何使用系统处理图像并生成动画
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import ChinesePaintingAnimator
from utils.logging_utils import setup_logging

def create_directories():
    """
    创建必要的目录结构
    """
    directories = [
        'data',
        'output',
        'temp',
        'logs',
        'models'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✓ 创建目录: {directory}/")

def create_sample_image():
    """
    创建一个示例中国画图像用于测试
    """
    # 创建一个简单的中国画风格图像
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 绘制一些简单的笔画
    # 横笔
    draw.line([(50, 100), (200, 100)], fill='black', width=3)
    
    # 竖笔
    draw.line([(125, 50), (125, 150)], fill='black', width=3)
    
    # 撇笔
    draw.line([(80, 180), (150, 220)], fill='black', width=2)
    
    # 捺笔
    draw.line([(150, 180), (220, 220)], fill='black', width=2)
    
    # 点
    draw.ellipse([(170, 120), (180, 130)], fill='black')
    
    # 保存示例图像
    sample_path = project_root / 'data' / 'test.jpg'
    image.save(sample_path)
    print(f"✓ 创建示例图像: {sample_path}")
    
    return str(sample_path)

def load_config():
    """
    加载配置文件
    """
    config_path = project_root / 'config.json'
    
    if not config_path.exists():
        print("❌ 配置文件不存在: config.json")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✓ 配置文件加载成功")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def run_example():
    """
    运行示例
    """
    print("=" * 60)
    print("中国画笔刷动画构建系统 - 运行示例")
    print("=" * 60)
    
    try:
        # 1. 创建必要目录
        print("\n1. 创建目录结构...")
        create_directories()
        
        # 2. 创建示例图像
        print("\n2. 创建示例图像...")
        sample_image_path = create_sample_image()
        
        # 3. 加载配置
        print("\n3. 加载配置文件...")
        config = load_config()
        if config is None:
            return False
        
        # 4. 设置日志
        print("\n4. 设置日志系统...")
        setup_logging(config.get('logging', {}))
        
        # 5. 初始化动画构建器
        print("\n5. 初始化系统...")
        animator = ChinesePaintingAnimator(config)
        print("✓ 系统初始化成功")
        
        # 6. 处理示例图像
        print("\n6. 处理示例图像...")
        output_dir = project_root / 'output' / 'example_output'
        
        print(f"输入图像: {sample_image_path}")
        print(f"输出目录: {output_dir}")
        
        # 注意：这里只是演示初始化，实际的图像处理需要完整的算法实现
        print("\n⚠️  注意: 当前版本仅演示系统初始化")
        print("完整的图像处理功能需要进一步的算法实现")
        
        # 可以在这里添加简单的图像处理演示
        result = {
            'success': True,
            'input_image': sample_image_path,
            'output_dir': str(output_dir),
            'message': '系统初始化成功，准备进行图像处理'
        }
        
        print("\n=" * 60)
        print("✅ 示例运行完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 示例图像: {sample_image_path}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 运行示例时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_usage_guide():
    """
    打印使用指南
    """
    print("\n" + "=" * 60)
    print("使用指南")
    print("=" * 60)
    print("\n1. 运行示例:")
    print("   python run_example.py")
    print("\n2. 测试系统:")
    print("   python test_system.py")
    print("\n3. 处理自定义图像:")
    print("   # 将您的中国画图像放入 data/ 目录")
    print("   # 修改 run_example.py 中的图像路径")
    print("   # 运行处理脚本")
    print("\n4. 配置调整:")
    print("   # 编辑 config.json 文件")
    print("   # 调整各种算法参数")
    print("\n5. 查看结果:")
    print("   # 处理结果保存在 output/ 目录")
    print("   # 日志文件保存在 logs/ 目录")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    success = run_example()
    
    if success:
        print_usage_guide()
    else:
        print("\n请检查错误信息并重试")
        sys.exit(1)