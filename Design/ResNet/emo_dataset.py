import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_emotion_dataloader(data_dir, batch_size=64):
    data_transforms={
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.RandomHorizontalFlip(),           
            transforms.RandomRotation(10),               
            transforms.ToTensor(),                       
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]),
    }
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(os.path.join(data_dir, 'val')):
        val_dir_name = 'val'
    elif os.path.exists(os.path.join(data_dir, 'test')):
        val_dir_name = 'test'
    else:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 'val' 或 'test' 文件夹")
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, val_dir_name), data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes  
    
    return dataloaders, dataset_sizes, class_names
if __name__ == '__main__':
    data_path = 'D:/Code/pytorch/Design/ResNet/Fer2013' 
    
    try:
        loaders, sizes, classes = get_emotion_dataloader(data_path, batch_size=4)
        print(f"数据加载成功！")
        print(f"类别: {classes}")
        print(f"训练集数量: {sizes['train']}, 验证集数量: {sizes['val']}")
        inputs, labels = next(iter(loaders['train']))
        print(f"输入 Batch 形状: {inputs.shape}") 
        print(f"标签 Batch: {labels}")
        img = inputs[0].permute(1, 2, 0).numpy()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Label: {classes[labels[0]]}")
        plt.show()
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查 data_path 是否正确，以及文件夹结构是否为 train/val 形式。")