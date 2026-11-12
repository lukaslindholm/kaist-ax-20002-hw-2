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
LEARNING_RATE = 0.001 # Högre startvärde för schedulern!
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
    print("Loading data for final training...")
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    
    X_train_all = z["X_train"] 
    y_train_all = train_df['label'].values
    X_test = z["X_test"]

    train_dataset = SteelDefectDataset(X_train_all, y_train_all, transform=train_transform)
    test_dataset = SteelDefectDataset(X_test, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing Model...")
    model = models.resnet34(weights="DEFAULT") # TESTA ATT BYTA MELLAN resnet18 OCH resnet34 HÄR
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("Training on ALL Stages (1-3) combined...")
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
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_dataset):.4f} | LR: {current_lr:.6f}")

    # Spara vikterna för eventuell ensembling senare!
    torch.save(model.state_dict(), 'final_model_weights.pth')

    print("\nGenerating predictions for Stages 4 & 5...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    sub = pd.DataFrame({
        "id": test_df.image_id, 
        "label": predictions
    })
    sub.to_csv("submission_lr_scheduler34.csv", index=False)
    
    pred_counts = np.bincount(predictions, minlength=N_CLASSES)
    print("\nSaved submission_lr_scheduler34.csv!")
    print("Predicted class counts:", pred_counts)

if __name__ == "__main__":
    create_submission()