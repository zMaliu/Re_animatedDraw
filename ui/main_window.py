# -*- coding: utf-8 -*-
"""
主窗口界面

提供交互式的绘画动画生成界面
"""

import sys
import os
from pathlib import Path
from typing import Optional
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class AnimationWorker(QThread):
    """
    动画生成工作线程
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, animator, input_path: str, output_path: str):
        super().__init__()
        self.animator = animator
        self.input_path = input_path
        self.output_path = output_path
    
    def run(self):
        try:
            # 生成动画
            success = self.animator.process_image(
                self.input_path,
                self.output_path,
                progress_callback=self.progress.emit
            )
            
            # 发送完成信号
            if success:
                self.finished.emit(True, "动画生成成功！")
            else:
                self.finished.emit(False, "动画生成失败，请检查日志。")
                
        except Exception as e:
            self.finished.emit(False, f"错误：{str(e)}")

class MainWindow(QMainWindow):
    """
    主窗口
    """
    
    def __init__(self, animator):
        super().__init__()
        self.animator = animator
        self.config = animator.config or {}
        self.logger = logging.getLogger(__name__)
        
        # 界面状态
        self.input_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.worker: Optional[AnimationWorker] = None
        
        self._init_ui()
    
    def _init_ui(self):
        """
        初始化界面
        """
        # 设置窗口
        self.setWindowTitle("中国画笔刷动画生成器")
        self.setMinimumSize(800, 600)
        
        # 创建主部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建布局
        layout = QVBoxLayout(main_widget)
        
        # 图像预览区域
        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")
        preview_layout.addWidget(self.image_label)
        
        layout.addWidget(preview_group)
        
        # 参数设置区域
        params_group = QGroupBox("参数设置")
        params_layout = QHBoxLayout(params_group)
        
        # 笔画参数
        stroke_group = QGroupBox("笔画参数")
        stroke_layout = QVBoxLayout(stroke_group)
        
        # 最小笔画长度
        min_length_layout = QHBoxLayout()
        min_length_layout.addWidget(QLabel("最小笔画长度："))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 100)
        self.min_length_spin.setValue(self.config.get('stroke_detection', {}).get('min_stroke_length', 10))
        min_length_layout.addWidget(self.min_length_spin)
        stroke_layout.addLayout(min_length_layout)
        
        # 最大笔画宽度
        max_width_layout = QHBoxLayout()
        max_width_layout.addWidget(QLabel("最大笔画宽度："))
        self.max_width_spin = QSpinBox()
        self.max_width_spin.setRange(1, 100)
        self.max_width_spin.setValue(self.config.get('stroke_detection', {}).get('max_stroke_width', 30))
        max_width_layout.addWidget(self.max_width_spin)
        stroke_layout.addLayout(max_width_layout)
        
        params_layout.addWidget(stroke_group)
        
        # 动画参数
        animation_group = QGroupBox("动画参数")
        animation_layout = QVBoxLayout(animation_group)
        
        # 帧率
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率："))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.config.get('animation', {}).get('fps', 30))
        fps_layout.addWidget(self.fps_spin)
        animation_layout.addLayout(fps_layout)
        
        # 笔画持续时间
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("笔画持续时间："))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setValue(self.config.get('animation', {}).get('duration', 1.0))
        duration_layout.addWidget(self.duration_spin)
        animation_layout.addLayout(duration_layout)
        
        params_layout.addWidget(animation_group)
        
        # 效果参数
        effect_group = QGroupBox("效果参数")
        effect_layout = QVBoxLayout(effect_group)
        
        # 墨晕效果
        ink_layout = QHBoxLayout()
        ink_layout.addWidget(QLabel("墨晕效果："))
        self.ink_spin = QDoubleSpinBox()
        self.ink_spin.setRange(0.0, 1.0)
        self.ink_spin.setSingleStep(0.1)
        self.ink_spin.setValue(self.config.get('animation', {}).get('ink_diffusion', 0.3))
        ink_layout.addWidget(self.ink_spin)
        effect_layout.addLayout(ink_layout)
        
        # 运动模糊
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("运动模糊："))
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 1.0)
        self.blur_spin.setSingleStep(0.1)
        self.blur_spin.setValue(self.config.get('animation', {}).get('motion_blur', 0.2))
        blur_layout.addWidget(self.blur_spin)
        effect_layout.addLayout(blur_layout)
        
        params_layout.addWidget(effect_group)
        
        layout.addWidget(params_group)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        # 选择输入文件
        self.input_btn = QPushButton("选择输入图像")
        self.input_btn.clicked.connect(self._select_input)
        control_layout.addWidget(self.input_btn)
        
        # 选择输出目录
        self.output_btn = QPushButton("选择输出目录")
        self.output_btn.clicked.connect(self._select_output)
        control_layout.addWidget(self.output_btn)
        
        # 开始生成
        self.start_btn = QPushButton("开始生成")
        self.start_btn.clicked.connect(self._start_generation)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        layout.addLayout(control_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
    
    def _select_input(self):
        """
        选择输入图像
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.input_path = file_path
            self._update_preview()
            self._check_start_button()
    
    def _select_output(self):
        """
        选择输出目录
        """
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )
        
        if dir_path:
            self.output_path = dir_path
            self._check_start_button()
    
    def _update_preview(self):
        """
        更新图像预览
        """
        try:
            # 读取图像
            image = cv2.imread(self.input_path)
            if image is None:
                return
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应预览区域
            height, width = image.shape[:2]
            max_size = 400
            if width > height:
                if width > max_size:
                    scale = max_size / width
                    width = max_size
                    height = int(height * scale)
            else:
                if height > max_size:
                    scale = max_size / height
                    height = max_size
                    width = int(width * scale)
            
            image = cv2.resize(image, (width, height))
            
            # 转换为QImage
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 显示图像
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")
    
    def _check_start_button(self):
        """
        检查是否可以开始生成
        """
        self.start_btn.setEnabled(
            self.input_path is not None and
            self.output_path is not None
        )
    
    def _start_generation(self):
        """
        开始生成动画
        """
        try:
            # 更新配置
            if 'stroke_detection' not in self.config:
                self.config['stroke_detection'] = {}
            if 'animation' not in self.config:
                self.config['animation'] = {}
                
            self.config['stroke_detection']['min_stroke_length'] = self.min_length_spin.value()
            self.config['stroke_detection']['max_stroke_width'] = self.max_width_spin.value()
            self.config['animation']['fps'] = self.fps_spin.value()
            self.config['animation']['duration'] = self.duration_spin.value()
            self.config['animation']['ink_diffusion'] = self.ink_spin.value()
            self.config['animation']['motion_blur'] = self.blur_spin.value()
            
            # 禁用控件
            self._set_controls_enabled(False)
            
            # 创建输出路径
            output_file = os.path.join(
                self.output_path,
                f"animation_{Path(self.input_path).stem}.mp4"
            )
            
            # 创建工作线程
            self.worker = AnimationWorker(
                self.animator,
                self.input_path,
                output_file
            )
            
            # 连接信号
            self.worker.progress.connect(self._update_progress)
            self.worker.finished.connect(self._generation_finished)
            
            # 启动线程
            self.worker.start()
            
        except Exception as e:
            self.logger.error(f"Error starting generation: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动生成失败：{str(e)}")
            self._set_controls_enabled(True)
    
    def _update_progress(self, value: int):
        """
        更新进度条
        
        Args:
            value (int): 进度值 [0-100]
        """
        self.progress_bar.setValue(value)
    
    def _generation_finished(self, success: bool, message: str):
        """
        动画生成完成
        
        Args:
            success (bool): 是否成功
            message (str): 完成消息
        """
        # 启用控件
        self._set_controls_enabled(True)
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 显示结果
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "警告", message)
        
        # 清理工作线程
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def _set_controls_enabled(self, enabled: bool):
        """
        设置控件启用状态
        
        Args:
            enabled (bool): 是否启用
        """
        self.input_btn.setEnabled(enabled)
        self.output_btn.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        self.min_length_spin.setEnabled(enabled)
        self.max_width_spin.setEnabled(enabled)
        self.fps_spin.setEnabled(enabled)
        self.duration_spin.setEnabled(enabled)
        self.ink_spin.setEnabled(enabled)
        self.blur_spin.setEnabled(enabled)