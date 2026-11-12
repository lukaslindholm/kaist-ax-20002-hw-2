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
EPOCHS_FROZEN = 3
EPOCHS_UNFROZEN = 20
LEARNING_RATE_F1 = 0.0005
LEARNING_RATE_F2 = 0.00005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class SteelDefectDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
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
            
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, label
        else:
            return img_tensor

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_submission():
    print("Loading data...")
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    
    # TRÄNA PÅ ALL DATA (Steg 1, 2, 3)
    X_train_all = z["X_train"] 
    y_train_all = train_df['label'].values
    X_test = z["X_test"]

    train_dataset = SteelDefectDataset(X_train_all, y_train_all, transform=train_transform)
    test_dataset = SteelDefectDataset(X_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing ResNet34...")
    model = models.resnet34(weights="DEFAULT")
    
    # FAS 1: Frys lager
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE_F1, weight_decay=0.01)

    print(f"\n--- PHASE 1: Training only the final layer for {EPOCHS_FROZEN} epochs ---")
    for epoch in range(EPOCHS_FROZEN):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Phase 1 - Epoch {epoch+1} | Train Loss: {running_loss/len(train_dataset):.4f}")

    # FAS 2: Tina upp allt
    print(f"\n--- PHASE 2: Unfreezing all layers and training for {EPOCHS_UNFROZEN} epochs ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_F2, weight_decay=0.01)

    for epoch in range(EPOCHS_UNFROZEN):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Phase 2 - Epoch {epoch+1}/{EPOCHS_UNFROZEN} | Train Loss: {running_loss/len(train_dataset):.4f}")

    print("\nGenerating predictions for Stages 4 & 5...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # --- HÄR KOMMER VI LÄGGA IN LABEL SHIFT (Just nu bara ren argmax) ---
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    sub = pd.DataFrame({
        "id": test_df.image_id, 
        "label": predictions
    })
    sub.to_csv("submission_resnet34_base.csv", index=False)
    
    print("\nSaved submission_resnet34_base.csv!")
    print("Predicted class counts:", np.bincount(predictions, minlength=N_CLASSES))

if __name__ == "__main__":
    create_submission()