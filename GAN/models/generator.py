"""
生成器模型
负责将随机噪声转换为图像
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    GAN生成器网络
    将随机噪声向量转换为图像
    """
    def __init__(self, nz=100, ngf=64, nc=1):
        """
        初始化生成器
        
        Args:
            nz: 噪声向量的长度
            ngf: 生成器中的特征图数量
            nc: 输出图像的通道数
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        # 生成器网络结构
        #除了输出层 每层使用BatchNorm与ReLU激活函数
        self.main = nn.Sequential(
            # 输入是Z，进入卷积层前的预处理
            #转置卷积
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态大小: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态大小: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态大小: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态大小: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出大小: (nc) x 64 x 64
        )
        
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 输入噪声张量
            
        Returns:
            生成的图像
        """
        return self.main(input)