import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data", transform=transform)

dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

images, label = next(iter(dataloader))
print("Img shape:", images.shape)
print("Label shape: ", label.shape)


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
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters())

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
                    loss_val, current = loss.item(), (batch+1)*len(X)
                    print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")






os.makedirs("model/", exist_ok=True)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dataloader, model, loss_fn, optimizer)
print("Training completed!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")