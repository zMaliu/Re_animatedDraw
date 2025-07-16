# -*- coding: utf-8 -*-
"""
中国画笔刷动画构建系统 - 主程序入口

基于论文《Animated Construction of Chinese Brush Paintings》实现
提供完整的中国画笔刷动画生成流程
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 核心模块
from core.stroke_extraction import StrokeDetector, StrokeExtractor, StrokeSegmenter
from core.stroke_database import StrokeDatabase, StrokeMatcher, StrokeClassifier
from core.stroke_ordering import StrokeOrganizer, OrderingOptimizer, RuleBasedOrdering
from core.animation import PaintingAnimator, BrushSimulator, AnimationRenderer, TimelineManager

# 导入工具模块
from utils.image_utils import ImageProcessor, ImageEnhancer, ImageAnalyzer
from utils.math_utils import MathUtils, GeometryUtils, StatisticsUtils
from utils.visualization import ColorUtils, Visualizer, PlotManager
from utils.file_utils import FileManager, DataSerializer, ArchiveManager, ConfigManager
from utils.logging_utils import LogManager, ContextLogger, LogAnalyzer, setup_logging
from utils.performance import PerformanceProfiler, SystemMonitor, Timer, MemoryMonitor, profile


class ChinesePaintingAnimator:
    """
    中国画动画生成器主类
    
    整合所有功能模块，提供完整的动画生成流程
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化动画生成器
        
        Args:
            config (Dict[str, Any]): 配置参数
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化性能监控
        self.profiler = PerformanceProfiler("ChinesePaintingAnimator")
        self.system_monitor = SystemMonitor()
        
        # 初始化核心组件
        self._init_components()
        
        self.logger.info("Chinese Painting Animator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            Dict[str, Any]: 默认配置
        """
        return {
            # 输入输出配置
            'input_image': None,
            'output_dir': './output',
            'temp_dir': './temp',
            
            # 笔画检测配置
            'stroke_detection': {
                'method': 'hybrid',
                'edge_threshold': 50,
                'min_stroke_length': 10,
                'max_stroke_width': 50
            },
            
            # 笔画数据库配置
            'stroke_database': {
                'database_path': './data/stroke_database.db',
                'similarity_threshold': 0.7,
                'max_templates': 1000
            },
            
            # 笔画排序配置
            'stroke_ordering': {
                'method': 'hybrid',
                'optimization_iterations': 100,
                'use_traditional_rules': True
            },
            
            # 动画生成配置
            'animation': {
                'fps': 30,
                'duration': 10.0,
                'resolution': (1920, 1080),
                'background_color': (255, 255, 255),
                'brush_simulation': True
            },
            
            # 渲染配置
            'rendering': {
                'output_format': 'mp4',
                'quality': 'high',
                'show_progress': True,
                'enable_effects': True
            },
            
            # 性能配置
            'performance': {
                'enable_profiling': True,
                'enable_monitoring': True,
                'max_memory_usage': 4096  # MB
            }
        }
    
    def _init_components(self):
        """
        初始化各个组件
        """
        # 图像处理组件
        self.image_processor = ImageProcessor()
        self.image_enhancer = ImageEnhancer()
        self.image_analyzer = ImageAnalyzer()
        
        # 笔画检测组件
        self.stroke_detector = StrokeDetector(self.config)
        self.stroke_extractor = StrokeExtractor()
        self.stroke_segmenter = StrokeSegmenter()
        
        # 笔画数据库组件
        self.stroke_database = StrokeDatabase(
            self.config,
            database_path=self.config['stroke_database']['database_path']
        )
        self.stroke_matcher = StrokeMatcher(self.config)
        self.stroke_classifier = StrokeClassifier(self.config)
        
        # 笔画排序组件
        self.stroke_organizer = StrokeOrganizer(self.config)
        self.ordering_optimizer = OrderingOptimizer(self.config)
        self.rule_based_ordering = RuleBasedOrdering(self.config)
        
        # 动画生成组件
        self.painting_animator = PaintingAnimator(self.config)
        self.brush_simulator = BrushSimulator(self.config)
        self.animation_renderer = AnimationRenderer(self.config)
        self.timeline_manager = TimelineManager(self.config)
        
        # 工具组件
        self.visualizer = Visualizer()
        self.plot_manager = PlotManager()
        self.file_manager = FileManager()
        self.data_serializer = DataSerializer()
    
    @profile
    def process_image(self, image_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        处理单张图像，生成动画
        
        Args:
            image_path (str): 输入图像路径
            output_dir (str): 输出目录
            
        Returns:
            Dict[str, Any]: 处理结果
        """
        try:
            # 设置输出目录
            if output_dir is None:
                output_dir = self.config['output_dir']
            
            self.file_manager.create_directory(output_dir)
            
            self.logger.info(f"Processing image: {image_path}")
            
            # 启动系统监控
            if self.config['performance']['enable_monitoring']:
                self.system_monitor.start_monitoring()
            
            # 步骤1: 加载和预处理图像
            with self.profiler.profile_context("image_loading"):
                image = self._load_and_preprocess_image(image_path)
            
            # 步骤2: 笔画检测
            with self.profiler.profile_context("stroke_detection"):
                strokes = self._detect_strokes(image)
            
            # 步骤3: 笔画分类和匹配
            with self.profiler.profile_context("stroke_classification"):
                classified_strokes = self._classify_strokes(strokes)
            
            # 步骤4: 笔画排序
            with self.profiler.profile_context("stroke_ordering"):
                ordered_strokes = self._order_strokes(classified_strokes)
            
            # 步骤5: 动画生成
            with self.profiler.profile_context("animation_generation"):
                animation = self._generate_animation(ordered_strokes, image)
            
            # 步骤6: 渲染输出
            with self.profiler.profile_context("animation_rendering"):
                output_files = self._render_animation(animation, output_dir)
            
            # 步骤7: 生成可视化和报告
            with self.profiler.profile_context("visualization"):
                visualization_files = self._generate_visualizations(
                    image, strokes, classified_strokes, ordered_strokes, output_dir
                )
            
            # 停止系统监控
            if self.config['performance']['enable_monitoring']:
                self.system_monitor.stop_monitoring()
            
            # 生成处理报告
            report = self._generate_report(
                image_path, output_files, visualization_files
            )
            
            self.logger.info(f"Image processing completed: {image_path}")
            
            return {
                'success': True,
                'input_image': image_path,
                'output_dir': output_dir,
                'output_files': output_files,
                'visualization_files': visualization_files,
                'report': report,
                'performance_metrics': self.profiler.get_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'input_image': image_path
            }
    
    def _load_and_preprocess_image(self, image_path: str):
        """
        加载和预处理图像
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            预处理后的图像
        """
        # 加载图像
        image = self.image_processor.load_image(image_path)
        
        # 图像增强
        enhanced_image = self.image_enhancer.adjust_contrast(image, factor=1.2)
        enhanced_image = self.image_enhancer.unsharp_masking(enhanced_image, sigma=1.0, strength=0.5)
        
        # 转换为灰度图像（如果需要）
        if len(enhanced_image.shape) == 3:
            gray_image = self.image_processor.convert_to_grayscale(enhanced_image)
        else:
            gray_image = enhanced_image
        
        return {
            'original': image,
            'enhanced': enhanced_image,
            'grayscale': gray_image
        }
    
    def _detect_strokes(self, image_data: Dict):
        """
        检测笔画
        
        Args:
            image_data (Dict): 图像数据
            
        Returns:
            检测到的笔画列表
        """
        gray_image = image_data['grayscale']
        
        # 笔画检测
        detected_strokes = self.stroke_detector.detect_strokes(gray_image)
        
        # 笔画提取（如果需要进一步特征提取）
        extracted_strokes = []
        for stroke in detected_strokes:
            # 使用已检测到的笔画，或进行进一步特征提取
            extracted_strokes.append(stroke)
        
        # 笔画分割（如果需要）
        segmented_strokes = []
        for stroke in extracted_strokes:
            segments = self.stroke_segmenter.segment_stroke(stroke)
            segmented_strokes.extend(segments)
        
        return segmented_strokes
    
    def _classify_strokes(self, strokes: list):
        """
        分类笔画
        
        Args:
            strokes (list): 笔画列表
            
        Returns:
            分类后的笔画列表
        """
        classified_strokes = []
        
        for stroke in strokes:
            # 笔画分类
            classification_result = self.stroke_classifier.classify_stroke(stroke)
            
            # 在数据库中查找相似笔画
            matching_result = self.stroke_matcher.find_best_match(
                stroke,
                threshold=self.config['stroke_database']['similarity_threshold']
            )
            
            # 合并分类和匹配结果
            stroke_info = {
                'stroke_data': stroke,
                'classification': classification_result,
                'matching': matching_result
            }
            
            classified_strokes.append(stroke_info)
        
        return classified_strokes
    
    def _order_strokes(self, classified_strokes: list):
        """
        排序笔画
        
        Args:
            classified_strokes (list): 分类后的笔画列表
            
        Returns:
            排序后的笔画列表
        """
        # 提取笔画数据
        strokes = [s['stroke_data'] for s in classified_strokes]
        
        # 笔画组织
        organization_result = self.stroke_organizer.organize_strokes(
            strokes,
            method=self.config['stroke_ordering']['method']
        )
        
        # 优化排序
        if self.config['stroke_ordering']['method'] in ['hybrid', 'optimization']:
            optimization_result = self.ordering_optimizer.optimize_ordering(
                organization_result.ordered_strokes,
                max_iterations=self.config['stroke_ordering']['optimization_iterations']
            )
            final_order = optimization_result.best_ordering
        else:
            final_order = organization_result.ordered_strokes
        
        # 应用传统规则（如果启用）
        if self.config['stroke_ordering']['use_traditional_rules']:
            final_order = self.rule_based_ordering.apply_traditional_rules(final_order)
        
        # 重新组织分类信息
        ordered_classified_strokes = []
        for stroke in final_order:
            # 找到对应的分类信息
            for classified_stroke in classified_strokes:
                if self._strokes_equal(stroke, classified_stroke['stroke_data']):
                    ordered_classified_strokes.append(classified_stroke)
                    break
        
        return ordered_classified_strokes
    
    def _strokes_equal(self, stroke1, stroke2) -> bool:
        """
        判断两个笔画是否相等
        
        Args:
            stroke1: 笔画1
            stroke2: 笔画2
            
        Returns:
            bool: 是否相等
        """
        try:
            from utils.math_utils import safe_compare, ensure_scalar
            
            # 首先检查是否是同一个对象
            if id(stroke1) == id(stroke2):
                return True
                
            # 检查基本属性
            if not isinstance(stroke1, dict) or not isinstance(stroke2, dict):
                return False
            
            # 比较关键字段
            key_fields = ['id', 'centroid', 'bounding_rect', 'contour']
            for field in key_fields:
                val1 = stroke1.get(field)
                val2 = stroke2.get(field)
                
                # 如果两个值都不存在，继续检查下一个字段
                if val1 is None and val2 is None:
                    continue
                    
                # 如果只有一个值不存在，则笔画不相等
                if val1 is None or val2 is None:
                    return False
                
                # 使用safe_compare进行安全比较
                if not safe_compare(val1, val2):
                    return False
            
            return True
                
        except (ValueError, TypeError) as e:
            # 如果比较失败，记录日志并返回False
            self.logger.warning(f"Error comparing strokes: {e}")
            return False
    
    def _generate_animation(self, ordered_strokes: list, image_data: Dict):
        """
        生成动画
        
        Args:
            ordered_strokes (list): 排序后的笔画列表
            image_data (Dict): 图像数据
            
        Returns:
            动画数据
        """
        # 提取笔画数据
        strokes = [s['stroke_data'] for s in ordered_strokes]
        
        # 创建动画
        animation = self.painting_animator.create_animation(
            strokes,
            duration=self.config['animation']['duration'],
            fps=self.config['animation']['fps']
        )
        
        # 如果启用笔刷模拟
        if self.config['animation']['brush_simulation']:
            # 为每个笔画添加笔刷效果
            for stroke_animation in animation.stroke_animations:
                self.brush_simulator.simulate_brush_stroke(stroke_animation)
        
        return animation
    
    def _render_animation(self, animation, output_dir: str) -> Dict[str, str]:
        """
        渲染动画
        
        Args:
            animation: 动画数据
            output_dir (str): 输出目录
            
        Returns:
            Dict[str, str]: 输出文件路径
        """
        output_files = {}
        
        # 设置渲染参数
        render_settings = {
            'resolution': self.config['animation']['resolution'],
            'fps': self.config['animation']['fps'],
            'background_color': self.config['animation']['background_color'],
            'quality': self.config['rendering']['quality'],
            'show_progress': self.config['rendering']['show_progress']
        }
        
        # 渲染视频
        if self.config['rendering']['output_format'] in ['mp4', 'avi']:
            video_path = os.path.join(output_dir, f"animation.{self.config['rendering']['output_format']}")
            self.animation_renderer.render_video(animation, video_path, **render_settings)
            output_files['video'] = video_path
        
        # 渲染GIF
        if self.config['rendering']['output_format'] == 'gif':
            gif_path = os.path.join(output_dir, "animation.gif")
            self.animation_renderer.render_gif(animation, gif_path, **render_settings)
            output_files['gif'] = gif_path
        
        # 渲染图像序列
        frames_dir = os.path.join(output_dir, "frames")
        self.file_manager.create_directory(frames_dir)
        frame_paths = self.animation_renderer.render_image_sequence(
            animation, frames_dir, **render_settings
        )
        output_files['frames'] = frame_paths
        
        # 生成缩略图
        thumbnail_path = os.path.join(output_dir, "thumbnail.png")
        self.animation_renderer.generate_thumbnail(animation, thumbnail_path)
        output_files['thumbnail'] = thumbnail_path
        
        return output_files
    
    def _generate_visualizations(self, image_data: Dict, strokes: list, 
                               classified_strokes: list, ordered_strokes: list, 
                               output_dir: str) -> Dict[str, str]:
        """
        生成可视化图像
        
        Args:
            image_data (Dict): 图像数据
            strokes (list): 原始笔画
            classified_strokes (list): 分类后的笔画
            ordered_strokes (list): 排序后的笔画
            output_dir (str): 输出目录
            
        Returns:
            Dict[str, str]: 可视化文件路径
        """
        viz_files = {}
        
        # 笔画检测结果可视化
        detection_viz_path = os.path.join(output_dir, "stroke_detection.png")
        self.visualizer.visualize_stroke_detection(
            image_data['original'], strokes, detection_viz_path
        )
        viz_files['detection'] = detection_viz_path
        
        # 笔画分类结果可视化
        classification_viz_path = os.path.join(output_dir, "stroke_classification.png")
        self.visualizer.visualize_stroke_classification(
            image_data['original'], classified_strokes, classification_viz_path
        )
        viz_files['classification'] = classification_viz_path
        
        # 笔画排序结果可视化
        ordering_viz_path = os.path.join(output_dir, "stroke_ordering.png")
        self.visualizer.visualize_stroke_ordering(
            image_data['original'], ordered_strokes, ordering_viz_path
        )
        viz_files['ordering'] = ordering_viz_path
        
        return viz_files
    
    def _generate_report(self, input_image: str, output_files: Dict, 
                        visualization_files: Dict) -> Dict[str, Any]:
        """
        生成处理报告
        
        Args:
            input_image (str): 输入图像路径
            output_files (Dict): 输出文件
            visualization_files (Dict): 可视化文件
            
        Returns:
            Dict[str, Any]: 处理报告
        """
        import time
        report = {
            'input_image': input_image,
            'processing_time': time.time(),
            'output_files': output_files,
            'visualization_files': visualization_files,
            'performance_summary': self.profiler.get_summary(),
            'system_metrics': self.system_monitor.get_metrics_summary() if self.config['performance']['enable_monitoring'] else None
        }
        
        return report
    
    def _save_intermediate_results(self, input_path, output_dir, image_data, strokes, ordered_strokes):
        """
        保存中间处理结果
        
        Args:
            input_path (str): 输入路径
            output_dir (str): 输出目录
            image_data (Dict): 图像数据
            strokes (list): 检测到的笔画
            ordered_strokes (list): 排序后的笔画
        """
        try:
            # 创建结果目录
            base_name = Path(input_path).stem
            result_dir = Path(output_dir) / f"{base_name}_results"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存原始图像副本
            original_path = result_dir / "original.png"
            self.image_processor.save_image(image_data['original'], str(original_path))
            
            # 保存增强图像
            enhanced_path = result_dir / "enhanced.png"
            self.image_processor.save_image(image_data['enhanced'], str(enhanced_path))
            
            # 保存灰度图像
            grayscale_path = result_dir / "grayscale.png"
            self.image_processor.save_image(image_data['grayscale'], str(grayscale_path))
            
            # 保存笔画数据
            strokes_path = result_dir / "strokes_data.json"
            stroke_data = {
                'input_image': input_path,
                'num_strokes': len(strokes),
                'num_ordered_strokes': len(ordered_strokes),
                'processing_timestamp': time.time()
            }
            self.data_serializer.save_json(stroke_data, str(strokes_path))
            
            print(f"中间结果已保存到: {result_dir}")
            
        except Exception as e:
            print(f"保存中间结果时发生错误: {str(e)}")
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.png") -> Dict[str, Any]:
        """
        批量处理图像
        
        Args:
            input_dir (str): 输入目录
            output_dir (str): 输出目录
            file_pattern (str): 文件模式
            
        Returns:
            Dict[str, Any]: 批量处理结果
        """
        self.logger.info(f"开始批量处理: {input_dir} -> {output_dir}")
        
        # 获取输入文件列表
        input_files = self.file_manager.list_files(input_dir, pattern=file_pattern)
        
        if not input_files:
            self.logger.warning(f"在目录 {input_dir} 中未找到匹配 {file_pattern} 的文件")
            return {'success': False, 'message': '未找到输入文件'}
        
        # 创建输出目录
        self.file_manager.create_directory(output_dir)
        
        # 批量处理结果
        batch_results = {
            'total_files': len(input_files),
            'processed_files': 0,
            'failed_files': 0,
            'results': [],
            'errors': []
        }
        
        # 开始性能监控
        if self.config['performance']['enable_monitoring']:
            self.system_monitor.start_monitoring()
        
        try:
            for i, input_file in enumerate(input_files):
                self.logger.info(f"处理文件 {i+1}/{len(input_files)}: {input_file}")
                
                try:
                    # 为每个文件创建单独的输出目录
                    file_name = os.path.splitext(os.path.basename(input_file))[0]
                    file_output_dir = os.path.join(output_dir, file_name)
                    
                    # 处理单个图像
                    result = self.process_image(input_file, file_output_dir)
                    
                    if result['success']:
                        batch_results['processed_files'] += 1
                        batch_results['results'].append({
                            'input_file': input_file,
                            'output_dir': file_output_dir,
                            'result': result
                        })
                        self.logger.info(f"成功处理: {input_file}")
                    else:
                        batch_results['failed_files'] += 1
                        batch_results['errors'].append({
                            'input_file': input_file,
                            'error': result.get('error', '未知错误')
                        })
                        self.logger.error(f"处理失败: {input_file} - {result.get('error', '未知错误')}")
                
                except Exception as e:
                    batch_results['failed_files'] += 1
                    error_msg = f"处理文件 {input_file} 时发生异常: {str(e)}"
                    batch_results['errors'].append({
                        'input_file': input_file,
                        'error': error_msg
                    })
                    self.logger.error(error_msg, exc_info=True)
        
        finally:
            # 停止性能监控
            if self.config['performance']['enable_monitoring']:
                self.system_monitor.stop_monitoring()
        
        # 保存批量处理报告
        batch_report_path = os.path.join(output_dir, "batch_report.json")
        batch_results['performance_summary'] = self.profiler.get_summary()
        if self.config['performance']['enable_monitoring']:
            batch_results['system_metrics'] = self.system_monitor.get_metrics_summary()
        
        self.data_serializer.save_json(batch_results, batch_report_path)
        
        self.logger.info(f"批量处理完成: 成功 {batch_results['processed_files']}, 失败 {batch_results['failed_files']}")
        
        return batch_results


def main():
    """
    主函数
    """
    import time
    
    parser = argparse.ArgumentParser(
        description="中国画笔刷动画构建算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --input data/input_images/painting.jpg --output data/output_animations/
  python main.py --input data/input_images/ --output data/output_animations/ --batch
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入图像文件路径或包含图像的目录路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出动画的目录路径"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径 (可选)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批处理模式，处理输入目录中的所有图像"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出模式"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="*.jpg,*.jpeg,*.png,*.bmp,*.tiff,*.tif",
        help="支持的图像格式 (默认: *.jpg,*.jpeg,*.png,*.bmp,*.tiff,*.tif)"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(console_level=log_level, file_level='DEBUG')
    
    # 加载配置
    config = None
    config_file = args.config or 'config.json'  # 默认使用config.json
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            import json
            config = json.load(f)
        print(f"已加载配置文件: {config_file}")
    except FileNotFoundError:
        if args.config:  # 如果用户指定了配置文件但找不到
            print(f"配置文件不存在: {config_file}")
            return 1
        else:  # 如果是默认配置文件不存在，给出提示但继续运行
            print(f"默认配置文件 {config_file} 不存在，使用内置默认配置")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化动画构建器
    print("初始化中国画动画构建器...")
    try:
        animator = ChinesePaintingAnimator(config)
    except Exception as e:
        print(f"初始化失败: {e}")
        return 1
    
    # 处理输入
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 批处理模式
        print(f"批处理模式: 处理目录 {input_path} 中的所有图像")
        
        # 解析支持的图像格式
        image_patterns = [pattern.strip() for pattern in args.format.split(',')]
        
        try:
            # 使用批量处理方法
            batch_result = animator.process_batch(
                str(input_path), 
                str(output_dir),
                file_pattern=image_patterns[0]  # 使用第一个模式
            )
            
            if batch_result['success'] is False:
                print(f"批处理失败: {batch_result.get('message', '未知错误')}")
                return 1
            
            print(f"\n批处理完成: {batch_result['processed_files']}/{batch_result['total_files']} 个文件处理成功")
            
            if batch_result['failed_files'] > 0:
                print(f"失败文件数: {batch_result['failed_files']}")
                for error in batch_result['errors']:
                    print(f"  - {error['input_file']}: {error['error']}")
        
        except Exception as e:
            print(f"批处理过程中发生错误: {e}")
            return 1
        
    else:
        # 单文件处理模式
        if not input_path.exists():
            print(f"错误: 输入文件 {input_path} 不存在")
            return 1
        
        print(f"单文件处理模式: {input_path}")
        try:
            result = animator.process_image(str(input_path), str(output_dir))
            
            if result['success']:
                print("处理成功!")
                print(f"输出文件: {result['output_files']}")
                print(f"可视化文件: {result['visualization_files']}")
            else:
                print(f"处理失败: {result.get('error', '未知错误')}")
                return 1
        
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            return 1
    
    print("\n处理完成!")
    return 0


if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")
    sys.exit(exit_code)