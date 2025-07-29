# 简单GAN框架

这是一个采用分层架构的简单GAN实现，用于训练图像生成模型。

## 目录结构

```
GAN/
├── datasets/           # 数据处理模块
│   └── data_loader.py  # 数据加载和预处理
├── models/             # 模型定义
│   ├── generator.py    # 生成器模型
│   └── discriminator.py # 判别器模型
├── trainers/           # 训练逻辑
│   └── gan_trainer.py  # GAN训练器
├── train.py            # 主训练脚本
└── README.md           # 说明文档
```

## 架构说明

### 1. 数据层 (datasets/)
负责数据加载和预处理：
- [data_loader.py](file:///F:/AI_draw/Re_animatedDraw/GAN/datasets/data_loader.py): 提供数据加载功能，支持图像数据集的加载和预处理

### 2. 模型层 (models/)
定义网络结构：
- [generator.py](file:///F:/AI_draw/Re_animatedDraw/GAN/models/generator.py): 生成器模型，将随机噪声转换为图像
- [discriminator.py](file:///F:/AI_draw/Re_animatedDraw/GAN/models/discriminator.py): 判别器模型，区分真实图像和生成图像

### 3. 训练层 (trainers/)
实现训练逻辑：
- [gan_trainer.py](file:///F:/AI_draw/Re_animatedDraw/GAN/trainers/gan_trainer.py): 管理训练过程，包括损失计算和参数更新

## 使用方法

### 训练模型

```bash
cd Re_animatedDraw/GAN
python train.py --data_dir path/to/your/dataset --epochs 50
```

### 参数说明

- `--data_dir`: 数据集路径（必需）
- `--batch_size`: 批处理大小（默认：64）
- `--image_size`: 图像尺寸（默认：64）
- `--nz`: 噪声向量维度（默认：100）
- `--ngf`: 生成器特征图数量（默认：64）
- `--ndf`: 判别器特征图数量（默认：64）
- `--epochs`: 训练轮数（默认：25）
- `--lr`: 学习率（默认：0.0002）
- `--beta1`: Adam优化器参数（默认：0.5）
- `--output_dir`: 结果保存目录（默认：./results）
- `--checkpoint_dir`: 检查点保存目录（默认：./checkpoints）

## 模型架构

### 生成器

生成器采用转置卷积网络结构：
- 输入：随机噪声向量（默认100维）
- 输出：64x64图像
- 网络结构：包含多个转置卷积层和批归一化层

### 判别器

判别器采用卷积网络结构：
- 输入：64x64图像
- 输出：0-1之间的标量，表示图像是真实的概率
- 网络结构：包含多个卷积层、批归一化层和LeakyReLU激活函数

## 训练过程

1. 判别器首先在真实数据上训练，目标是正确识别真实图像
2. 判别器然后在生成数据上训练，目标是正确识别生成图像
3. 生成器训练，目标是让判别器将生成图像误判为真实图像

## 依赖项

- PyTorch
- torchvision

## 注意事项

1. 数据集应按照以下结构组织：
```
dataset/
└── class_name/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

2. 建议使用GPU进行训练以提高速度
3. 可根据需要调整网络参数和训练超参数