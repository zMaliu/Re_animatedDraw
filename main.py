#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国画笔触动画构建复现
主程序入口

基于论文: "Animated Construction of Chinese Brush Paintings"
实现五个核心模块的完整流程
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.stroke_extraction import StrokeExtractor
from modules.feature_modeling import FeatureModeler
from modules.structure_construction import StructureConstructor
from modules.stroke_ordering import StrokeOrderOptimizer
from modules.animation_generation import AnimationGenerator
from modules.visualizer import Visualizer
from modules.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='中国画笔触动画构建复现')
    parser.add_argument('input_image', help='输入图像路径')
    parser.add_argument('--output_dir', '-o', default='output', help='输出目录')
    parser.add_argument('--fps', type=int, default=24, help='动画帧率')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--skip_nes', action='store_true', help='跳过NES优化（使用简单排序）')
    parser.add_argument('--evaluate', action='store_true', help='启用质量评估')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"错误: 输入文件不存在 {input_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("中国画笔触动画构建复现")
    print("=" * 60)
    print(f"输入图像: {args.input_image}")
    print(f"输出目录: {args.output_dir}")
    print(f"动画帧率: {args.fps} fps")
    print(f"调试模式: {args.debug}")
    print(f"质量评估: {args.evaluate}")
    print("=" * 60)
    
    # 初始化评估器
    evaluator = None
    if args.evaluate:
        evaluator = Evaluator(debug=args.debug)
    
    try:
        # 模块1: 数据准备与笔触提取
        print("\n[1/5] 笔触提取中...")
        start_time = time.time()
        extractor = StrokeExtractor(debug=args.debug)
        strokes = extractor.extract_strokes(args.input_image)
        extraction_time = time.time() - start_time
        print(f"提取到 {len(strokes)} 个笔触，耗时 {extraction_time:.2f}s")
        
        if len(strokes) == 0:
            print("错误: 未提取到任何笔触")
            sys.exit(1)
            
        if args.debug:
            extractor.save_debug_images(output_dir / 'debug' / 'stroke_extraction')
        
        # 评估笔触提取质量
        stroke_metrics = None
        if evaluator:
            stroke_metrics = evaluator.evaluate_stroke_extraction(strokes, args.input_image)
            print(f"笔触提取质量得分: {stroke_metrics.get('quality_score', 0):.3f}")
        
        # 模块2: 特征建模
        print("\n[2/5] 特征建模...")
        start_time = time.time()
        
        modeler = FeatureModeler(debug=args.debug)
        features = modeler.extract_features(strokes)
        
        if args.debug:
            modeler.save_feature_analysis(features, output_dir / 'debug' / 'feature_modeling')
        
        feature_time = time.time() - start_time
        print(f"提取特征完成 (耗时: {feature_time:.2f}s)")
        
        # 评估特征建模质量
        feature_metrics = None
        if evaluator:
            feature_metrics = evaluator.evaluate_feature_modeling(features)
            print(f"特征建模质量得分: {feature_metrics.get('quality_score', 0):.3f}")
        
        # 模块3: 结构构建
        print("\n[3/5] 结构构建...")
        start_time = time.time()
        
        constructor = StructureConstructor(debug=args.debug)
        hasse_graph, stages = constructor.build_structure(features)
        
        if args.debug:
            constructor.save_structure_analysis(hasse_graph, stages, 
                                               output_dir / 'debug' / 'structure_construction')
        
        structure_time = time.time() - start_time
        print(f"构建Hasse图完成 (耗时: {structure_time:.2f}s)")
        print(f"阶段分布: {[(stage, len(stroke_list)) for stage, stroke_list in stages.items()]}")
        
        # 评估结构构建质量
        structure_metrics = None
        if evaluator:
            structure_metrics = evaluator.evaluate_structure_construction(hasse_graph, stages, features)
            print(f"结构构建质量得分: {structure_metrics.get('quality_score', 0):.3f}")
        
        # 模块4: 笔触排序优化
        print("\n[4/5] 笔触排序优化...")
        start_time = time.time()
        
        optimizer = StrokeOrderOptimizer(debug=args.debug)
        if args.skip_nes:
            print("使用简单拓扑排序...")
            optimized_order = optimizer.simple_topological_order(hasse_graph, features)
        else:
            print("使用NES优化算法...")
            optimized_order = optimizer.optimize_order(hasse_graph, features, stages)
        
        if args.debug:
            optimizer.save_optimization_analysis(optimized_order, 
                                                output_dir / 'debug' / 'stroke_ordering')
        
        ordering_time = time.time() - start_time
        print(f"排序优化完成 (耗时: {ordering_time:.2f}s)")
        print(f"优化后顺序长度: {len(optimized_order)}")
        
        # 评估排序优化质量
        ordering_metrics = None
        if evaluator:
            ordering_metrics = evaluator.evaluate_stroke_ordering(optimized_order, hasse_graph, features, stages)
            print(f"排序优化质量得分: {ordering_metrics.get('quality_score', 0):.3f}")
        
        # 模块5: 动画生成
        print("\n[5/5] 动画生成...")
        start_time = time.time()
        
        generator = AnimationGenerator(fps=args.fps, debug=args.debug)
        animation_path = output_dir / f"{input_path.stem}_animation.mp4"
        
        generator.generate_animation(strokes, optimized_order, animation_path)
        
        if args.debug:
            generator.save_preview_frames(strokes, optimized_order, 
                                        output_dir / 'debug' / 'animation_preview')
        
        animation_time = time.time() - start_time
        print(f"动画生成完成 (耗时: {animation_time:.2f}s)")
        print(f"动画文件: {animation_path}")
        
        # 评估动画质量
        animation_metrics = None
        if evaluator:
            animation_metrics = evaluator.evaluate_animation_quality(animation_path, strokes, optimized_order)
            print(f"动画质量得分: {animation_metrics.get('quality_score', 0):.3f}")
        
        # 可视化结果
        if args.debug:
            print("\n生成可视化结果...")
            visualizer = Visualizer(debug=True)
            
            # 笔触可视化
            visualizer.visualize_strokes(strokes, output_dir / 'visualizations' / 'strokes.png')
            
            # 特征可视化
            visualizer.visualize_features(features, output_dir / 'visualizations' / 'features.png')
            
            # 结构可视化
            visualizer.visualize_structure(hasse_graph, stages, features, 
                                         output_dir / 'visualizations' / 'structure.png')
            
            # 排序可视化
            visualizer.visualize_ordering(optimized_order, features, stages, 
                                        output_dir / 'visualizations' / 'ordering.png')
            
            # 生成总结报告
            visualizer.create_summary_report(strokes, features, stages, optimized_order,
                                           output_dir / 'summary_report.json')
        
        # 生成综合评估报告
        if evaluator and all(m is not None for m in [stroke_metrics, feature_metrics, 
                                                    structure_metrics, ordering_metrics, 
                                                    animation_metrics]):
            print("\n生成综合评估报告...")
            comprehensive_report = evaluator.generate_comprehensive_report(
                stroke_metrics, feature_metrics, structure_metrics, 
                ordering_metrics, animation_metrics,
                output_dir / 'evaluation_report.json'
            )
            
            overall_score = comprehensive_report['overall_assessment']['score']
            grade = comprehensive_report['overall_assessment']['grade']
            description = comprehensive_report['overall_assessment']['description']
            
            print(f"\n综合评估结果: {grade} ({description}) - {overall_score:.3f}")
        
        # 保存性能统计
        performance_stats = {
            'processing_times': {
                'stroke_extraction': extraction_time,
                'feature_modeling': feature_time,
                'structure_construction': structure_time,
                'stroke_ordering': ordering_time,
                'animation_generation': animation_time,
                'total': extraction_time + feature_time + structure_time + ordering_time + animation_time
            },
            'data_statistics': {
                'stroke_count': len(strokes),
                'feature_count': len(features),
                'stage_distribution': {str(stage): len(stroke_list) for stage, stroke_list in stages.items()},
                'optimization_length': len(optimized_order),
                'animation_fps': args.fps
            },
            'configuration': {
                'input_image': str(input_path),
                'output_directory': str(output_dir),
                'debug_mode': args.debug,
                'skip_nes': args.skip_nes,
                'evaluation_enabled': args.evaluate
            }
        }
        
        with open(output_dir / 'performance_stats.json', 'w', encoding='utf-8') as f:
            json.dump(performance_stats, f, ensure_ascii=False, indent=2)
        
        print("\n=" * 60)
        print("复现完成！")
        print(f"总耗时: {performance_stats['processing_times']['total']:.2f}s")
        print(f"输出目录: {output_dir}")
        print(f"动画文件: {animation_path}")
        if args.debug:
            print(f"调试信息: {output_dir / 'debug'}")
        if args.evaluate:
            print(f"评估报告: {output_dir / 'evaluation_report.json'}")
        print(f"性能统计: {output_dir / 'performance_stats.json'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()