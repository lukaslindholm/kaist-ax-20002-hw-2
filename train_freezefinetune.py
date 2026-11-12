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
# Vi delar upp träningen i två faser
EPOCHS_FROZEN = 3   # Fas 1: Bara sista lagret tränas
EPOCHS_UNFROZEN = 30 # Fas 2: Hela modellen tränas
LEARNING_RATE = 0.0005 # Lite högre för fas 1
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

def get_data_loaders():
    train_df = pd.read_csv(DATA / "train.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    Xtr = z["X_train"]
    labels = train_df['label'].values
    stages = train_df['stage'].values

    train_mask = (stages == 1) | (stages == 2)
    val_mask = (stages == 3)
    X_train, y_train = Xtr[train_mask], labels[train_mask]
    X_val, y_val = Xtr[val_mask], labels[val_mask]

    train_dataset = SteelDefectDataset(X_train, y_train, transform=train_transform)
    val_dataset = SteelDefectDataset(X_val, y_val, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def train_model():
    train_loader, val_loader = get_data_loaders()
    
    print("Initializing ResNet34...")
    model = models.resnet34(weights="DEFAULT")
    
    # FAS 1: Frys alla lager utom sista
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, N_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Använder AdamW (med Weight Decay) och skickar ENDAST in det okrysta lagret
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print(f"\n--- PHASE 1: Training only the final layer for {EPOCHS_FROZEN} epochs ---")
    best_val_acc = 0.0

    for epoch in range(EPOCHS_FROZEN):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, predicted = torch.max(model(inputs), 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        print(f"Phase 1 - Epoch {epoch+1} | Val Acc: {val_correct/val_total:.4f}")

    print(f"\n--- PHASE 2: Unfreezing all layers and training for {EPOCHS_UNFROZEN} epochs ---")
    # TINA upp alla lager
    for param in model.parameters():
        param.requires_grad = True
        
    # Skapa en ny optimerare för hela nätverket, med en MYCKET lägre learning rate!
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)

    for epoch in range(EPOCHS_UNFROZEN):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                _, predicted = torch.max(model(inputs), 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Phase 2 - Epoch {epoch+1}/{EPOCHS_UNFROZEN} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_finetuned.pth')
            print(f"  -> Model saved!")

    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train_model()