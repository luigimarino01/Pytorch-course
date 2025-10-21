import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(10)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = SimpleDataset()
print("Dataset length:", len(dataset))
print("Sample [3]:", dataset[3])

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    print(batch)

from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


images, labels = next(iter(train_loader))
print(images.shape, labels.shape)


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx, 0]}"
        image = Image.open(img_path)
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


import matplotlib.pyplot as plt

def show_images(images, labels, n=6):
    plt.figure(figsize=(10, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

show_images(images, labels)