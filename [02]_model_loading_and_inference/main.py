import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = NeuralNetwork().to(device)


model.load_state_dict(torch.load("../[01]_model_creation_and_train/model/model.pth"))
model.eval()
print("Model loaded successfully!")

images, labels = next(iter(test_dataloader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    predictions = outputs.argmax(dim=1)

print("Predictions:", predictions[:10])
print("True labels:", labels[:10])


def show_images(images, labels, preds, n=6):
    plt.figure(figsize=(12, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f"P: {preds[i].item()}\nT: {labels[i].item()}")
        plt.axis('off')
    plt.show()

show_images(images, labels, predictions)
