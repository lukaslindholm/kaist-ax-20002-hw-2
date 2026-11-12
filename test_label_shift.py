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
EPOCHS = 16  # Vi vet att ResNet34 peakar bra här med låg LR
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def test_label_shift():
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"]
    labels = train_df['label'].values
    stages = train_df['stage'].values

    # Träna 1 & 2, Validera på 3
    train_mask = (stages == 1) | (stages == 2)
    val_mask = (stages == 3)
    
    y_train = labels[train_mask]
    
    # 1. RÄKNA UT FÖRDELNINGEN I TRÄNINGSDATAT (Steg 1 & 2)
    train_counts = np.bincount(y_train, minlength=N_CLASSES)
    p_train = train_counts / train_counts.sum()
    print("Prior i Träningsdata (Steg 1&2):", np.round(p_train, 3))
    
    # 2. DEFINIERA FÖRVÄNTAD FÖRDELNING I TESTDATAT (Steg 3)
    # Vi vet från vår dataanalys att Steg 3 är nästan exakt jämnt fördelat (1/6)
    p_test_uniform = np.array([1/6] * N_CLASSES)

    train_dataset = SteelDefectDataset(Xtr[train_mask], y_train, transform=train_transform)
    val_dataset = SteelDefectDataset(Xtr[val_mask], labels[val_mask], transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet34(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    print("\nTraining ResNet34...")
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} done.")

    # --- VALIDERINGS-FAS MED LABEL SHIFT ---
    model.eval()
    
    correct_standard = 0
    correct_adjusted = 0
    total = 0

    # Omvandla priors till tensors
    p_train_tensor = torch.tensor(p_train, dtype=torch.float32).to(DEVICE)
    p_test_uniform_tensor = torch.tensor(p_test_uniform, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Modellens råa utdata (Logits)
            logits = model(inputs)
            
            # 1. Omvandla logits till sannolikheter (0 till 1)
            probs = torch.softmax(logits, dim=1)
            
            # --- STANDARD GISSNING ---
            _, pred_standard = torch.max(probs, 1)
            correct_standard += (pred_standard == targets).sum().item()
            
            # --- JUSTERAD GISSNING (Uniform) ---
            # Dividera med tränings-prior och multiplicera med test-prior
            adjusted_probs = probs * (p_test_uniform_tensor / p_train_tensor)
            _, pred_adjusted = torch.max(adjusted_probs, 1)
            correct_adjusted += (pred_adjusted == targets).sum().item()
            
            total += targets.size(0)

    print("\n--- RESULTAT PÅ STEG 3 ---")
    print(f"Standard Val Acc:           {correct_standard/total:.4f}")
    print(f"Justerad Val Acc (Uniform): {correct_adjusted/total:.4f}")

if __name__ == "__main__":
    test_label_shift()