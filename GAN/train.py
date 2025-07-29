"""
GAN主训练脚本
"""

import torch
import torch.nn as nn
import os
import argparse

from datasets.data_loader import get_data_loader
from models.generator import Generator
from models.discriminator import Discriminator
from trainers.gan_trainer import GANTrainer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a simple GAN')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training (default: 64)')
    parser.add_argument('--image_size', type=int, default=64, 
                        help='Image size (default: 64)')
    parser.add_argument('--nz', type=int, default=100, 
                        help='Size of the latent z vector (default: 100)')
    parser.add_argument('--ngf', type=int, default=64, 
                        help='Size of feature maps in generator (default: 64)')
    parser.add_argument('--ndf', type=int, default=64, 
                        help='Size of feature maps in discriminator (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, 
                        help='Number of training epochs (default: 25)')
    parser.add_argument('--lr', type=float, default=0.0002, 
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, 
                        help='Beta1 hyperparameter for Adam optimizers (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='./results', 
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory to save checkpoints (default: ./checkpoints)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据加载器
    print("Loading data...")
    dataloader = get_data_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # 获取图像通道数
    nc = 1 if 'mnist' in args.data_dir.lower() else 3  # 简单判断灰度图或彩色图
    
    # 创建生成器和判别器
    print("Creating models...")
    generator = Generator(nz=args.nz, ngf=args.ngf, nc=nc).to(device)
    discriminator = Discriminator(nc=nc, ndf=args.ndf).to(device)
    
    # 打印模型结构
    print("Generator:")
    print(generator)
    print("\nDiscriminator:")
    print(discriminator)
    
    # 创建训练器
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        nz=args.nz,
        lr=args.lr,
        beta1=args.beta1
    )
    
    # 开始训练
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 执行一个epoch的训练
        G_loss, D_loss = trainer.train_step(dataloader)
        
        print(f"Generator Loss: {G_loss:.4f}, Discriminator Loss: {D_loss:.4f}")
        
        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            trainer.save_model(args.checkpoint_dir, epoch+1)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("\nTraining completed!")
    print(f"Final Generator Loss: {G_loss:.4f}")
    print(f"Final Discriminator Loss: {D_loss:.4f}")

if __name__ == '__main__':
    main()