# 中国水墨画动态构建系统

## 项目简介

本项目基于论文《Animated Construction of Chinese Brush Paintings》，实现了一个完整的从静态中国水墨画重建绘画过程的技术框架。系统通过计算符合艺术规则的笔触绘制顺序，实现自动化的绘画动态生成。

## 核心技术实现

根据论文技术框架，系统实现了以下四个核心步骤：

### 1. 输入与笔触建模
- **输入要求**：处理笔触清晰的小写意水墨画（排除工笔或泼墨风格）
- **交互式分解**：通过交互式工具将画作分解为独立的2D笔触集合
- **多维度特征提取**：
  - **几何特征**：骨架点（Canny边缘检测+Harris角点）、长度、面积、尺度、形状（傅里叶描述子）
  - **墨色特征**：湿度、厚度、颜色
  - **位置特征**：基于三分法则的显著性得分（专用显著性计算器）

### 2. 多阶段结构构建
- **艺术原则映射**：依据"先整体后局部再装饰"的创作逻辑
- **三阶段分类**：
  - 主要笔触：决定构图的大块深色笔触
  - 局部细节笔触：细化主体的中等复杂度笔触
  - 装饰笔触：干燥或细点缀的修饰性笔触
- **层级关系建模**：构建有向无环图（DAG）并简化为Hasse图

### 3. 笔触排序优化
- **能量函数设计**：
  - 一致性成本：颜色、形状、空间距离相似性
  - 变化成本：湿度、厚度、尺寸变化规则
  - 正则化项：与多阶段结构的一致性
- **硬约束处理**：圆形物体成对笔触连续绘制
- **自然进化策略（NES）**：实数空间编码和多正态分布采样优化
- **Spearman秩相关系数**：量化笔触顺序与理想阶段顺序的一致性

### 4. 动态绘制与动画生成
- **笔触方向确定**：基于墨湿度和厚度梯度
- **Flood-filling渲染**：8/12邻域种子扩散，椭圆足迹模型
- **动态速度调整**：根据笔触宽度调整绘制速度

## 关键创新点

- 将传统绘画艺术规则转化为可计算的特征和能量函数
- 结合多阶段结构和自然进化策略的无监督笔触顺序优化
- 动态渲染过程模拟真实绘画的节奏和墨色变化
- 适用于教学、动画制作等多种应用场景

## 快速开始

### 1. 环境要求
- Python 3.8+
- CUDA支持（可选，用于GPU加速）

### 2. 安装步骤
```bash
# 克隆项目
git clone https://github.com/zMaliu/Re_animatedDraw.git
cd Re_animatedDraw

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 使用方法

#### 命令行方式
```bash
# 运行示例（自动创建必要目录和示例图像）
python run_example.py

# 处理自定义图像
python main.py --input data/test2.jpg --output output/ --config config.json
```

#### 图形界面方式
```bash
# 启动GUI界面
python main.py --gui
```

## 项目结构

```
Re_animatedDraw/
├── core/                      # 核心算法模块
│   ├── stroke_modeling/       # 笔触建模与特征提取
│   │   ├── feature_extractor.py    # 多维度特征提取
│   │   ├── edge_detector.py        # Canny边缘检测
│   │   ├── corner_detector.py      # Harris角点检测
│   │   └── interactive_tool.py     # 交互式分解工具
│   ├── structure_building/    # 多阶段结构构建
│   │   ├── stage_classifier.py     # 三阶段笔触分类
│   │   ├── hierarchy_builder.py    # 层级关系建模
│   │   ├── dag_constructor.py       # 有向无环图构建
│   │   └── hasse_simplifier.py     # Hasse图简化
│   ├── stroke_ordering/       # 笔触排序优化
│   │   ├── energy_function.py      # 能量函数设计
│   │   ├── nes_optimizer.py        # 自然进化策略
│   │   ├── constraint_handler.py   # 硬约束处理
│   │   ├── order_evaluator.py      # 排序结果评估
│   │   └── spearman_correlation.py # Spearman秩相关系数
│   ├── animation/             # 动态绘制与动画生成
│   │   ├── direction_detector.py   # 笔触方向确定
│   │   ├── stroke_animator.py      # 笔触动画生成
│   │   ├── flood_renderer.py       # Flood-filling渲染
│   │   └── animation_controller.py # 动画控制管理
│   ├── stroke_extraction/     # 笔触提取
│   │   ├── stroke_extractor.py     # 笔触提取算法
│   │   ├── skeleton_extractor.py   # 骨架提取
│   │   └── contour_analyzer.py     # 轮廓分析
│   └── image_processing/      # 图像处理
│       ├── image_processor.py      # 基础图像处理
│       ├── fourier_descriptor.py   # 傅里叶描述子形状分析
│       └── saliency_calculator.py  # 三分法则显著性计算
├── config/                    # 配置管理
├── utils/                     # 工具函数
├── ui/                        # 用户界面
├── data/                      # 输入数据
├── examples/                  # 示例代码
└── tests/                     # 测试代码
```

## 技术栈

- **核心框架**
  - Python 3.8+
  - NumPy (数值计算)
  - SciPy (科学计算)
  - OpenCV (图像处理)

- **机器学习**
  - scikit-learn (特征分析)
  - scikit-image (图像分析)

- **优化算法**
  - 自然进化策略（NES）
  - Spearman秩相关系数
  - 约束满足优化

- **可视化**
  - Matplotlib (数据可视化)
  - PyQt5 (用户界面)

## 核心模块详解

### 1. 笔触建模模块 (stroke_modeling)
- **特征提取器**：几何、墨色、位置特征的多维度提取
- **边缘检测器**：基于Canny算法的边缘检测
- **角点检测器**：Harris角点检测算法
- **交互工具**：笔触分解的交互式界面

### 2. 结构构建模块 (structure_building)
- **阶段分类器**：基于艺术原则的三阶段笔触分类
- **层级构建器**：笔触间偏序关系建模
- **DAG构造器**：有向无环图构建和环路检测
- **Hasse简化器**：传递性冗余边移除

### 3. 笔触排序模块 (stroke_ordering)
- **能量函数**：一致性成本、变化成本、正则化项
- **NES优化器**：自然进化策略全局优化
- **约束处理器**：硬约束的定义和处理
- **排序评估器**：多指标排序质量评估
- **Spearman相关系数**：笔触顺序一致性量化分析

### 4. 动画生成模块 (animation)
- **方向检测器**：基于梯度的笔触方向确定
- **笔触动画器**：笔触级别的动画生成
- **Flood渲染器**：种子扩散的绘制模拟
- **动画控制器**：播放控制和交互管理

## 配置说明

系统支持通过 `config.json` 自定义各模块参数：

```json
{
  "stroke_modeling": {
    "canny_low_threshold": 50,
    "canny_high_threshold": 150,
    "harris_k": 0.04,
    "harris_threshold": 0.01
  },
  "structure_building": {
    "stage_weights": {
      "geometric": 0.4,
      "ink": 0.4,
      "position": 0.2
    },
    "main_stage_threshold": 0.7,
    "decoration_stage_threshold": 0.3
  },
  "stroke_ordering": {
    "energy_weights": {
      "consistency": 0.4,
      "variation": 0.4,
      "regularization": 0.2
    },
    "nes_population_size": 50,
    "nes_generations": 100
  },
  "animation": {
    "fps": 30,
    "flood_neighborhood": "8-connected",
    "elliptical_footprint": true
  }
}
```

## 实现状态

### 已完成模块
- [x] **笔触建模**：多维度特征提取、边缘检测、角点检测
- [x] **结构构建**：三阶段分类、层级关系、DAG构建、Hasse简化
- [x] **笔触排序**：能量函数、NES优化、约束处理、排序评估、Spearman相关系数
- [x] **动画生成**：方向检测、笔触动画、Flood渲染、动画控制
- [x] **笔触提取**：骨架提取、轮廓分析、笔触分割
- [x] **图像处理**：傅里叶描述子形状分析、三分法则显著性计算
- [x] **基础设施**：配置管理、工具函数、日志系统

### 待完善功能
- [ ] 交互式笔触分解工具的GUI实现
- [ ] 更多艺术风格的适配
- [ ] 性能优化和GPU加速
- [ ] 用户研究和效果验证

## 使用示例

### 1. 完整流程示例
```python
from core.stroke_modeling import FeatureExtractor
from core.structure_building import StageClassifier, HierarchyBuilder
from core.stroke_ordering import EnergyFunction, NESOptimizer, SpearmanCorrelationCalculator
from core.animation import StrokeAnimator, AnimationController
from core.image_processing import FourierDescriptor, SaliencyCalculator

# 1. 笔触特征提取（使用新的图像处理模块）
feature_extractor = FeatureExtractor(config)
strokes = feature_extractor.extract_strokes(image_path)

# 2. 多阶段结构构建
stage_classifier = StageClassifier(config)
stage_results = stage_classifier.classify_strokes(strokes)

hierarchy_builder = HierarchyBuilder(config)
hierarchy = hierarchy_builder.build_hierarchy(strokes, stage_results)

# 3. 笔触排序优化（包含Spearman相关性分析）
energy_function = EnergyFunction(config)
nes_optimizer = NESOptimizer(config)
optimal_order = nes_optimizer.optimize(strokes, energy_function)

# 评估排序质量
correlation_calc = SpearmanCorrelationCalculator()
quality_result = correlation_calc.analyze_stroke_order_quality(
    strokes, optimal_order, stage_results
)
print(f"排序质量得分: {quality_result.overall_score}")

# 4. 动画生成
stroke_animator = StrokeAnimator(config)
animation_frames = stroke_animator.create_animation(strokes, optimal_order)

controller = AnimationController(config)
controller.load_animation(strokes, optimal_order)
controller.play()
```

### 2. 模块化使用
```python
# 单独使用阶段分类器
from core.structure_building import StageClassifier

classifier = StageClassifier({
    'weights': {'geometric': 0.4, 'ink': 0.4, 'position': 0.2},
    'main_threshold': 0.7,
    'decoration_threshold': 0.3
})

stage_results = classifier.classify_strokes(strokes)
print(f"主要笔触: {len(stage_results.main_strokes)}")
print(f"细节笔触: {len(stage_results.detail_strokes)}")
print(f"装饰笔触: {len(stage_results.decoration_strokes)}")
```

### 3. 图像处理模块使用
```python
from core.image_processing import FourierDescriptor, SaliencyCalculator
import cv2

# 使用傅里叶描述子进行形状分析
contour = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
fourier_desc = FourierDescriptor()
shape_features = fourier_desc.calculate_shape_features(contour)
print(f"圆度: {shape_features.circularity}, 细长比: {shape_features.elongation}")

# 使用显著性计算器
saliency_calc = SaliencyCalculator()
saliency_score = saliency_calc.calculate_saliency(stroke_center, image_shape)
print(f"显著性得分: {saliency_score}")
```

### 4. 自定义能量函数
```python
from core.stroke_ordering import EnergyFunction

# 自定义权重配置
energy_config = {
    'consistency_weight': 0.5,
    'variation_weight': 0.3,
    'regularization_weight': 0.2,
    'color_similarity_sigma': 0.1,
    'spatial_distance_sigma': 50.0
}

energy_function = EnergyFunction(energy_config)
total_energy = energy_function.calculate_total_energy(strokes, order)
print(f"总能量: {total_energy}")
```

## 故障排除

### 常见问题

1. **安装依赖失败**
   ```bash
   # 使用清华源安装
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **CUDA相关错误**
   - 确保CUDA版本与PyTorch版本匹配
   - 使用 `nvidia-smi` 检查GPU状态

3. **内存不足**
   - 减小处理图像的尺寸
   - 调整批处理大小

### 调试模式

在 `config.json` 中启用详细日志：
```json
{
  "logging": {
    "level": "DEBUG",
    "file_path": "logs/debug.log",
    "console_output": true
  }
}
```


## 参考文献

- Animated Construction of Chinese Brush Paintings (IEEE)
- Animating Chinese paintings through stroke-based decomposition
- Virtual brush: a model-based synthesis of Chinese calligraphy

