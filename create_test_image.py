#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建测试图像
"""

import numpy as np
from PIL import Image, ImageDraw
import os

def create_test_image():
    """创建一个简单的测试图像，包含基本的笔画元素"""
    # 创建白色背景图像
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 绘制一些基本的笔画
    # 横笔
    draw.line([(50, 100), (200, 100)], fill='black', width=8)
    
    # 竖笔
    draw.line([(125, 50), (125, 150)], fill='black', width=8)
    
    # 撇笔
    draw.line([(250, 80), (200, 130)], fill='black', width=6)
    
    # 捺笔
    draw.line([(250, 130), (300, 180)], fill='black', width=6)
    
    # 点笔
    draw.ellipse([(314, 94), (326, 106)], fill='black')
    
    # 弯钩笔画（简化为折线）
    points = [(80, 200), (120, 220), (160, 200), (180, 180), (200, 200)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill='black', width=6)
    
    # 折笔
    draw.line([(250, 200), (300, 200)], fill='black', width=6)
    draw.line([(300, 200), (300, 250)], fill='black', width=6)
    
    # 保存图像
    output_path = 'data/test.png'
    os.makedirs('data', exist_ok=True)
    image.save(output_path)
    print(f"测试图像已创建: {output_path}")
    
    return output_path

if __name__ == '__main__':
    create_test_image()