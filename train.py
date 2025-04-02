from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models.segmentation as segmentation
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def download_data(pth: str, split: str):
    assert split in ["trainval", "test"]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long) - 1)
    ])
    dataset = OxfordIIITPet(root=pth, split=split, target_types=['segmentation'], download=True, 
                            transform=transform, target_transform=target_transform)
    return dataset

# Load datasets
train_dataset = download_data('./data', 'trainval')
test_dataset = download_data('./data', 'test')

# Data loaders with num_workers
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Verify loader sizes
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of testing batches: {len(test_loader)}")

# Initialize model
model = segmentation.deeplabv3_resnet50(weights=None, num_classes=3)
device = torch.device('cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with tqdm
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), 'deeplabv3_resnet50_fully_supervised.pth')