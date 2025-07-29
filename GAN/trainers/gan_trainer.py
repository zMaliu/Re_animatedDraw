"""
GAN训练器
负责训练生成器和判别器
"""

import torch
import torch.nn as nn
import os

class GANTrainer:
    """
    GAN训练器类
    管理GAN模型的训练过程
    """
    def __init__(self, generator, discriminator, device, nz=100, lr=0.0002, beta1=0.5):
        """
        初始化训练器
        
        Args:
            generator: 生成器模型
            discriminator: 判别器模型
            device: 训练设备 (CPU 或 GPU)
            nz: 噪声向量维度
            lr: 学习率
            beta1: Adam优化器的beta1参数
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.nz = nz
        self.lr = lr
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 优化器
        self.optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # 固定噪声，用于可视化训练过程
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        
    def train_step(self, dataloader):
        """
        执行一个epoch的训练
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            tuple: (生成器损失, 判别器损失)
        """
        # 初始化损失
        G_losses = []
        D_losses = []
        
        for i, data in enumerate(dataloader, 0):
            # ---------------------
            # 1. 更新判别器
            # ---------------------
            self.discriminator.zero_grad()
            
            # 使用真实数据训练
            real_data = data[0].to(self.device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), 1., dtype=torch.float, device=self.device)
            
            output = self.discriminator(real_data)
            errD_real = self.criterion(output, label)
            errD_real.backward()
            
            # 使用生成数据训练
            noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
            fake_data = self.generator(noise)
            label.fill_(0.)
            
            output = self.discriminator(fake_data.detach())
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            
            # 更新判别器参数
            self.optimizerD.step()
            
            # 计算判别器总损失
            D_loss = errD_real + errD_fake
            D_losses.append(D_loss.item())
            
            # ---------------------
            # 2. 更新生成器
            # ---------------------
            self.generator.zero_grad()
            
            # 生成新数据并训练生成器
            label.fill_(1.)  # 希望生成器能让判别器认为是真实数据
            output = self.discriminator(fake_data)
            errG = self.criterion(output, label)
            errG.backward()
            
            # 更新生成器参数
            self.optimizerG.step()
            
            # 记录生成器损失
            G_losses.append(errG.item())
            
        # 返回平均损失
        return sum(G_losses)/len(G_losses), sum(D_losses)/len(D_losses)
    
    def save_model(self, filepath, epoch):
        """
        保存模型检查点
        
        Args:
            filepath: 保存路径
            epoch: 当前epoch数
        """
        os.makedirs(filepath, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
        }, os.path.join(filepath, f'checkpoint_epoch_{epoch}.pth'))