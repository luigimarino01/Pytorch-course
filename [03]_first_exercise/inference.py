import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from PIL import Image

if(torch.accelerator.is_available()):
    device = torch.accelerator.current_accelerator()

print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)



        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64*6*6, 128)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

TEST_PATH = "test_img"
for img in os.listdir(TEST_PATH):
    img_path = os.path.join(TEST_PATH, img)

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
        class_names = ['cat', 'dog']
        print(f"{img}: {class_names[pred]}")






