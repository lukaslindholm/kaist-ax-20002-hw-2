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

# --- INSTÄLLNINGAR ---
DATA = Path("data")
N_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 16

# ÄNDRA DESSA TVÅ FÖR VARJE MODELL I DIN ENSEMBLE!
LEARNING_RATE = 0.0001
SAVE_NAME = "model_lr1_v3.pth"

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

# Data Augmentation för träningsdatan
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def train_and_save():
    print(f"Loading ALL training data (Stages 1-3)...")
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    
    X_train_all = z["X_train"] 
    y_train_all = train_df['label'].values

    train_dataset = SteelDefectDataset(X_train_all, y_train_all, transform=train_transform)
    # Shuffle=True är kritiskt här för att få variation mellan ensemble-modellerna!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 

    print("Initializing ResNet34 from scratch...")
    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Vi använder den vanliga, robusta Adam-optimeraren
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Training for {EPOCHS} epochs (LR = {LEARNING_RATE}) ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f}")

    # Spara endast vikterna
    torch.save(model.state_dict(), SAVE_NAME)
    print(f"\nTraining complete! Weights successfully saved to: {SAVE_NAME}")

if __name__ == "__main__":
    train_and_save()