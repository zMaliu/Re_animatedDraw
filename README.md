# 中国画笔触动画系统

基于论文《Animating Chinese Paintings through Stroke-based Decomposition》的完整实现，采用先进的分层架构设计，实现从静态中国画到动态绘制动画的智能转换。

## ✨ 核心特性

### 🎨 双系统架构
- **原始系统**: 严格按照论文五阶段流程实现
- **新架构系统**: 分层设计，增强笔触建模和物理效果

### 🖌️ 先进笔触建模
- **物理属性模拟**: 压力、湿度、厚度、速度
- **笔触类型识别**: 横、竖、点、钩、弯、撇捺等
- **真实效果**: 墨水渗透、纸张纹理、笔触抖动

### 🎬 智能动画生成
- **笔触排序优化**: NES算法 + 拓扑排序
- **渐进式绘制**: 平滑过渡和自然动画
- **多格式输出**: MP4视频、静态图像、数据分析

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 立即体验
```bash
# 🎨 新架构演示 (推荐)
python main_new_architecture.py --mode demo --debug

# 🖼️ 处理现有图像
python main.py test.jpg --debug

# 🖌️ 笔触功能测试
# python test_brush_strokes.py  # 已移除毛笔生成功能
```

## 📁 项目结构

```
Re_animatedDraw/
├── core/                    # 🏗️ 核心层
│   ├── stroke_model.py     # 笔触数据模型
│   └── canvas.py          # 画布管理
├── modules/                # ⚙️ 处理层
│   ├── stroke_extraction.py
│   ├── feature_modeling.py
│   ├── structure_construction.py
│   ├── stroke_ordering.py
│   └── animation_generation.py
├── main.py                 # 📜 原始系统
├── main_new_architecture.py # 🆕 新架构系统
# └── test_brush_strokes.py   # 🧪 功能测试 (已移除)
```

## 🎯 使用方式

### 原始系统 (论文实现)
```bash
# 基础运行
python main.py test.jpg

# 调试模式 + 性能评估
python main.py test.jpg --debug --evaluate
```

### 新架构系统 (增强版)
```bash
# 演示模式 - 生成山水画
python main_new_architecture.py --mode demo --debug

# 处理图像 - 转换为动画
python main_new_architecture.py --mode process --input test.jpg

# 创建作品 - 自定义艺术
python main_new_architecture.py --mode create --canvas_size 800,600
```

## 📊 输出内容

### 🎬 动画视频
- `test_animation.mp4`: 完整绘制过程
- `artwork_creation.mp4`: 自定义作品动画

### 🖼️ 静态图像
- `custom_artwork.jpg`: 生成的艺术作品
- 关键帧序列和调试可视化

### 📈 数据分析
- `performance_stats.json`: 性能统计
- `artwork_strokes.json`: 笔触数据
- 特征分析和结构信息

## 🔬 技术亮点

### 笔触建模算法
```python
# 物理属性建模
stroke.set_stroke_properties(
    pressure=0.7,    # 70% 压力
    wetness=0.6,     # 60% 湿度
    thickness=0.8,   # 80% 厚度
    speed=0.5        # 50% 速度
)

# 效果应用
canvas.apply_ink_bleeding(bleeding_radius=2)
canvas.add_paper_texture(intensity=0.2)
```

### 智能排序优化
- **NES算法**: 自然进化策略优化
- **拓扑排序**: 保持绘制逻辑
- **层次分析**: Hasse图构建

### 动画生成引擎
- **帧级控制**: 精确时间管理
- **路径插值**: 贝塞尔曲线平滑
- **渐进渲染**: 自然绘制效果

## 📚 文档资源

- 📖 [架构设计文档](ARCHITECTURE.md) - 详细技术架构
- 📋 [项目结构说明](PROJECT_STRUCTURE.md) - 文件组织
- 🚀 [快速开始指南](QUICK_START.md) - 上手教程
- 📄 [参考论文](Animated_Construction_of_Chinese_Brush_Paintings.pdf)

## 🛠️ 开发状态

✅ **已完成功能**:
- [x] 完整五阶段处理流程
- [x] 分层架构设计
- [x] 笔触物理建模
- [x] 智能动画生成
- [x] 调试和可视化工具
- [x] 性能优化和错误修复

🚀 **核心优势**:
- 🎨 真实笔触效果模拟
- 🧠 智能笔触识别和排序
- 🎬 高质量动画生成
- 🔧 模块化可扩展设计
- 📊 完整的分析和调试工具

## 🎨 示例效果

### 输入
- 静态中国画图像 (test.jpg, test2.jpeg)

### 输出
- 动态绘制动画 (MP4格式)
- 笔触分解和分析
- 绘制顺序优化
- 物理效果模拟

## 🤝 技术栈

- **图像处理**: OpenCV, scikit-image
- **数值计算**: NumPy, SciPy
- **机器学习**: scikit-learn
- **图算法**: NetworkX
- **可视化**: Matplotlib
- **视频处理**: OpenCV VideoWriter

---

🎨 **将静态的中国画转化为动态的艺术创作过程，体验传统艺术与现代技术的完美结合！**

## 文件结构

```
Re_animatedDraw/
├── main.py                          # 主程序入口
├── test_system.py                   # 系统测试脚本
├── requirements.txt                 # 依赖包列表
├── modules/                         # 核心模块
│   ├── __init__.py
│   ├── stroke_extraction.py         # 笔触提取模块
│   ├── feature_modeling.py          # 特征建模模块
│   ├── structure_construction.py    # 结构构建模块
│   ├── stroke_ordering.py           # 笔触排序模块
│   ├── animation_generation.py      # 动画生成模块
│   ├── visualizer.py               # 可视化工具
│   └── evaluator.py                # 质量评估工具
├── test.jpg                        # 测试图像1
├── test2.jpeg                      # 测试图像2
├── Animated_Construction_of_Chinese_Brush_Paintings.pdf  # 原论文
└── 复现思路.txt                    # 复现计划
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python main.py test.jpg --output_dir output
```

### 完整参数

```bash
python main.py <输入图像> --output_dir <输出目录> [选项]
```

**参数说明：**
- `input_image`: 输入的中国画图像路径
- `--output_dir`: 输出目录路径 (默认: output)
- `--fps`: 动画帧率 (默认: 24)
- `--debug`: 启用调试模式，保存中间结果
- `--skip_nes`: 跳过NES优化，使用简单拓扑排序
- `--evaluate`: 启用质量评估

### 示例

```bash
# 基本运行
python main.py test.jpg

# 调试模式
python main.py test.jpg --debug --evaluate

# 自定义参数
python main.py test2.jpeg --output_dir results --fps 30 --debug
```

## 输出结果

运行后会在输出目录生成：

- `<图像名>_animation.mp4` - 生成的动画文件
- `debug/` - 调试信息 (如果启用 --debug)
  - `stroke_extraction/` - 笔触提取结果
  - `feature_modeling/` - 特征分析结果
  - `structure_construction/` - 结构构建结果
  - `stroke_ordering/` - 排序优化结果
  - `animation_preview/` - 动画预览帧
- `visualizations/` - 可视化结果 (如果启用 --debug)
  - `strokes.png` - 笔触可视化
  - `features.png` - 特征可视化
  - `structure.png` - 结构可视化
  - `ordering.png` - 排序可视化
- `summary_report.json` - 处理总结报告
- `evaluation_report.json` - 质量评估报告 (如果启用 --evaluate)

## 系统测试

运行测试脚本验证系统功能：

```bash
python test_system.py
```

## 技术特点

### 笔触提取
- 自适应阈值分割
- 形态学操作优化
- 连通组件分析
- 基于面积、圆形度、长宽比的过滤

### 特征建模
- **几何特征**: 骨架长度、尺度、形状描述符
- **墨色特征**: 湿润度、厚度
- **颜色特征**: HSV颜色空间分析
- **位置显著性**: 基于图像中心的重要性

### 结构构建
- 多阶段分层 (主体结构、局部细节、装饰元素)
- Hasse图构建
- 偏序关系建立
- 传递依赖消除

### 笔触排序
- Natural Evolution Strategy (NES) 优化
- 能量函数设计 (一致性、变化性、正则化)
- 绘制方向启发式确定
- 拓扑约束保持

### 动画生成
- 逐步绘制模拟
- 动态速度调整
- 洪水填充路径生成
- 高质量视频输出

## 质量评估

系统包含全面的质量评估机制：

- **笔触提取质量**: 覆盖率、连通性
- **特征建模质量**: 完整性、分布合理性
- **结构构建质量**: 图属性、一致性
- **排序优化质量**: 约束满足、合理性
- **动画质量**: 平滑度、完整性

## 注意事项

1. 输入图像应为清晰的中国画作品
2. 建议图像分辨率不超过2000x2000以确保处理速度
3. 调试模式会生成大量中间文件，注意磁盘空间
4. NES优化可能需要较长时间，可使用 --skip_nes 跳过

## 论文参考

本项目基于以下论文实现：
- **标题**: Animated Construction of Chinese Brush Paintings
- **核心算法**: 多阶段笔触分析、Hasse图构建、NES优化

## 开发信息

- **开发语言**: Python 3.7+
- **主要依赖**: OpenCV, NumPy, SciPy, NetworkX, Matplotlib
- **支持平台**: Windows, Linux, macOS