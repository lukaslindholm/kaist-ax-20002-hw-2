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

DATA = Path("data")
N_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 16
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class SteelDefectDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np = self.images[idx].astype(np.uint8) 
        img_rgb = np.stack((img_np,)*3, axis=-1)
        img_pil = Image.fromarray(img_rgb)
        
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img_tensor, label

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_with_scheduler():
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"]
    labels = train_df['label'].values
    stages = train_df['stage'].values

    # Träna på Steg 1 & 2, Validera på Steg 3
    train_mask = (stages == 1) | (stages == 2)
    val_mask = (stages == 3)
    
    train_dataset = SteelDefectDataset(Xtr[train_mask], labels[train_mask], transform=train_transform)
    val_dataset = SteelDefectDataset(Xtr[val_mask], labels[val_mask], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing ResNet34...")
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- NYTT: LEARNING RATE SCHEDULER ---
    # Denna drar ner learning raten mjukt som en cosinus-kurva
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    print(f"Training for {EPOCHS} epochs with CosineAnnealingLR...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        # Uppdatera schedulern efter varje epok
        scheduler.step()
        
        # Validering
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
        val_acc = val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_dataset):.4f} | LR: {current_lr:.6f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("  -> New best model!")

    print(f"\nTraining complete. Best Validation Accuracy (Stage 3): {best_val_acc:.4f}")

if __name__ == "__main__":
    train_with_scheduler()