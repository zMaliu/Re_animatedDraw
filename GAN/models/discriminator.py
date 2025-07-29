"""
判别器模型
负责区分真实图像和生成图像
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    GAN判别器网络
    区分输入图像是真实的还是生成的
    """
    def __init__(self, nc=1, ndf=64):
        """
        初始化判别器
        
        Args:
            nc: 输入图像的通道数：灰度图为1 RGB图为3
            ndf: 判别器中的特征图数量
        """
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        
        # 判别器网络结构
        #除了输入层 每层都是用Batch Normalization 使用LeakyReLU激活函数（a=0.2）
        self.main = nn.Sequential(
            # （输入层）输入大小: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出大小: 1 x 1 x 1
        )
        
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 输入图像张量
            
        Returns:
            图像是真实的概率
        """
        return self.main(input).view(-1, 1).squeeze(1)