# 中国画笔刷动画构建算法复现

## 项目简介

本项目旨在复现论文《Animated Construction of Chinese Brush Paintings》中的算法，实现从静态中国画图像重建绘画过程的动画效果。

## 算法核心特点

基于论文研究，该算法主要包含以下核心技术：

1. **笔画分解与提取**：从静态画作中提取假设的笔画描述
2. **笔画库构建**：通过数字化经验艺术家的单个笔画建立笔画库
3. **笔画建模**：使用骨架和轮廓对单个笔画建模
4. **绘画顺序重建**：计算符合艺术规则的合理绘画顺序
5. **动画生成**：在笔画级别生成绘画动画

## 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/zMaliu/Re_animatedDraw.git
cd Re_animatedDraw
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行示例
```bash
# 自动创建必要目录和示例图像
python run_example.py

# 或处理自定义图像
python main.py --input data/your_painting.jpg --output output/
```

## 项目结构

```
Re_animatedDraw/
├── core/                   # 核心算法模块
│   ├── animation/         # 动画生成相关
│   ├── stroke_database/   # 笔画数据库
│   ├── stroke_extraction/ # 笔画提取
│   └── stroke_ordering/   # 笔画排序
├── utils/                 # 工具函数
├── config/                # 配置文件
├── data/                  # 输入数据
├── output/               # 输出结果
└── logs/                 # 日志文件
```

## 技术栈

- Python 3.8+
- OpenCV (图像处理)
- NumPy (数值计算)
- Matplotlib (可视化)
- Pillow (图像操作)
- scikit-image (图像分析)

## 模块说明

### 1. 笔画提取模块 (stroke_extraction)
- 从输入图像中检测和提取笔画
- 计算笔画的骨架和轮廓
- 分析笔画的纹理变化

### 2. 笔画库模块 (stroke_database)
- 管理预建的笔画库
- 匹配提取的笔画与库中的标准笔画
- 提供笔画特征描述

### 3. 顺序优化模块 (stroke_ordering)
- 分析画作的构图原则
- 组织笔画的三阶段结构
- 使用自然进化策略优化笔画顺序

### 4. 动画生成模块 (animation)
- 生成笔画级别的动画
- 渲染绘画过程
- 输出动画文件

## 故障排除

### 常见问题

1. **KeyError: 'canny_low_threshold'**
   - 确保 `config.json` 文件完整且格式正确
   - 运行 `python run_example.py` 会自动创建必要的配置

2. **TypeError: unexpected keyword argument**
   - 检查Python版本是否为3.8+
   - 确保所有依赖正确安装

3. **缺少data文件夹**
   - 运行 `python run_example.py` 会自动创建所有必要目录
   - 或手动创建：`mkdir data output logs temp models`

4. **导入模块错误**
   - 确保在项目根目录运行脚本
   - 检查Python路径设置

### 调试模式

在 `config.json` 中启用详细日志：
```json
{
  "logging": {
    "level": "DEBUG",
    "file_path": "logs/debug.log"
  }
}
```


## 参考文献

- Animated Construction of Chinese Brush Paintings (IEEE)
- Animating Chinese paintings through stroke-based decomposition
- Virtual brush: a model-based synthesis of Chinese calligraphy
