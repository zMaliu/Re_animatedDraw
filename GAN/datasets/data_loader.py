"""
数据处理模块
负责加载和预处理训练数据
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataManager:
    """
    数据管理器
    处理数据加载和预处理
    """
    def __init__(self, batch_size=64, image_size=64):
        self.batch_size = batch_size
        self.image_size = image_size
        
        # 定义数据预处理流程
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
        ])
    
    def create_dataloader(self, data_dir):
        """
        创建数据加载器
        
        Args:
            data_dir: 数据集目录路径
            
        Returns:
            DataLoader: PyTorch数据加载器
        """
        dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        return dataloader

def get_data_loader(data_dir, batch_size=64, image_size=64):
    """
    获取数据加载器的便捷函数
    
    Args:
        data_dir: 数据集目录路径
        batch_size: 批处理大小
        image_size: 图像尺寸
        
    Returns:
        DataLoader: 数据加载器
    """
    data_manager = DataManager(batch_size, image_size)
    return data_manager.create_dataloader(data_dir)