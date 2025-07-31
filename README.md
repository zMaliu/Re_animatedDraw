# 中国画笔触动画系统

基于论文《Animating Chinese Paintings through Stroke-based Decomposition》实现，将静态中国画转化为动态绘制动画。

## 核心特性

### 系统架构
- **系统**: 严格按照论文五阶段流程实现


### 笔触建模
- **物理属性模拟**: 压力、湿度、厚度、速度
- **笔触类型识别**: 横、竖、点、钩、弯、撇捺等
- **真实效果**: 墨水渗透、纸张纹理、笔触抖动

### 智能动画生成
- **笔触排序优化**: NES算法 + 拓扑排序
- **渐进式绘制**: 平滑过渡和自然动画
- **多格式输出**: MP4视频、静态图像、数据分析

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 程序运行
```bash
# 原始系统演示
python main.py test.jpg --debug
```

## 项目结构

```
Re_animatedDraw/
├── core/                    # 核心层
│   ├── stroke_model.py     # 笔触数据模型
│   └── canvas.py          # 画布管理
├── modules/                # 处理层
│   ├── stroke_extraction.py      # 笔触提取
│   ├── feature_modeling.py       # 特征建模
│   ├── structure_construction.py # 结构构建
│   ├── stroke_ordering.py        # 笔触排序
│   ├── animation_generation.py   # 动画生成
│   ├── visualizer.py             # 可视化工具
│   └── evaluator.py              # 质量评估
├── output/                  # 输出目录
│   ├── debug/               # 调试信息
│   ├── performance_stats.json # 性能统计
│   └── summary_report.json  # 汇总报告
├── main.py                 # 主程序入口
└── README.md               # 说明文档
```

## 系统工作流程

1. **笔触提取**: 从输入图像中分割出独立的笔触
2. **特征建模**: 分析每个笔触的几何、颜色、纹理等特征
3. **结构构建**: 建立笔触间的偏序关系和层级结构
4. **笔触排序**: 使用NES算法和拓扑排序优化绘制顺序
5. **动画生成**: 根据优化顺序逐步绘制笔触，生成动画

## 使用方式

### 基础运行
```bash
python main.py input_image.jpg
```

### 调试模式
```bash
python main.py input_image.jpg --debug
```

### 性能评估
```bash
python main.py input_image.jpg --evaluate
```

## 输出内容

### 动画视频
- `*_animation.mp4`: 完整绘制过程动画

### 调试信息
- `output/debug/`: 包含各阶段的详细数据和可视化结果

### 性能统计
- `performance_stats.json`: 处理时间等性能数据
- `summary_report.json`: 处理结果汇总报告



