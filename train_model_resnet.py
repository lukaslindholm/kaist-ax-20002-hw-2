# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# 1. Configuration
DATA = Path("data")
N_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 16
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 2. Dataset Definition
class SteelDefectDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy array of shape (N, 64, 64)
        labels: numpy array of shape (N,)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images in .npz are [0, 255] float64. Convert to uint8 for PIL
        img_np = self.images[idx].astype(np.uint8) 
        
        # Convert grayscale (64x64) to RGB (64x64x3) because ResNet expects 3 channels
        img_rgb = np.stack((img_np,)*3, axis=-1)
        
        # Convert numpy array to PIL Image so torchvision transforms work
        img_pil = Image.fromarray(img_rgb)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label

# 3. Data Augmentation (Tackling Covariate Shift)
# Training transform includes augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # Randomly change brightness and contrast to simulate future stages
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    # Standard ImageNet normalization
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Validation transform is deterministic (no random augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_data_loaders():
    print("Loading data...")
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"] # Shape: (1079, 64, 64)

    labels = train_df['label'].values
    stages = train_df['stage'].values

    # Validation Strategy: Train on Stage 1 & 2, Validate on Stage 3
    train_mask = (stages == 1) | (stages == 2)
    val_mask = (stages == 3)

    X_train, y_train = Xtr[train_mask], labels[train_mask]
    X_val, y_val = Xtr[val_mask], labels[val_mask]

    train_dataset = SteelDefectDataset(X_train, y_train, transform=train_transform)
    val_dataset = SteelDefectDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# 4. Model Definition (Pre-trained ResNet18)
def get_model():
    # Load pre-trained ResNet18
    # Use weights="DEFAULT" instead of pretrained=True for modern torchvision
    model = models.resnet34(weights="DEFAULT") 
    
    # Modify the final Fully Connected (FC) layer for 6 classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, N_CLASSES)
    
    return model.to(DEVICE)

# 5. Training Loop
def train_model():
    train_loader, val_loader = get_data_loaders()
    model = get_model()
    
    criterion = nn.CrossEntropyLoss()
    # We fine-tune the whole model, but we could choose to freeze early layers later
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("\nStarting Training...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc (Stage 3): {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> Model saved! (Validation Accuracy improved)")

    print(f"\nTraining complete. Best Stage 3 Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()