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
LEARNING_RATE = 0.0001
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

def train_and_test_tta():
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"]
    labels = train_df['label'].values
    stages = train_df['stage'].values

    train_mask = (stages == 1) | (stages == 2)
    val_mask = (stages == 3)
    
    train_dataset = SteelDefectDataset(Xtr[train_mask], labels[train_mask], transform=train_transform)
    val_dataset = SteelDefectDataset(Xtr[val_mask], labels[val_mask], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing ResNet34...")
    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} done.")

    print("\n--- Running Validation (Standard vs TTA) ---")
    model.eval()
    
    correct_standard = 0
    correct_tta = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            total += targets.size(0)
            
            # 1. STANDARD GISSNING
            probs_standard = torch.softmax(model(inputs), dim=1)
            _, pred_standard = torch.max(probs_standard, 1)
            correct_standard += (pred_standard == targets).sum().item()
            
            # 2. TTA GISSNING
            # Flip horisontellt (dimension -1 för bildtensor)
            inputs_hflip = torch.flip(inputs, [-1])
            probs_hflip = torch.softmax(model(inputs_hflip), dim=1)
            
            # Flip vertikalt (dimension -2 för bildtensor)
            inputs_vflip = torch.flip(inputs, [-2])
            probs_vflip = torch.softmax(model(inputs_vflip), dim=1)
            
            # Ta genomsnittet av alla tre sannolikheter!
            probs_tta_avg = (probs_standard + probs_hflip + probs_vflip) / 3.0
            _, pred_tta = torch.max(probs_tta_avg, 1)
            correct_tta += (pred_tta == targets).sum().item()

    print(f"Standard Val Acc: {correct_standard/total:.4f}")
    print(f"TTA Val Acc:      {correct_tta/total:.4f}")

if __name__ == "__main__":
    train_and_test_tta()