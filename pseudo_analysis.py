# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import random

# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- SETTINGS ---
# Read data from the current working directory
DATA = Path(".")
N_CLASSES = 6
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SELECT YOUR BEST MODEL (The one used to analyze the test data)
MODEL_PATH = "best_model_all_data.pth" 

class SteelDefectDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
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
        return img_tensor

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_pseudo_analysis():
    print(f"Loading test data from Stages 4 and 5...")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    X_test = z["X_test"]

    # We know from the data description that each stage has approx 360 images.
    # test.csv has 720 images. Thus, the first 360 = Stage 4, and the last 360 = Stage 5.
    X_stage_4 = X_test[:360]
    X_stage_5 = X_test[360:]

    dataset_s4 = SteelDefectDataset(X_stage_4, transform=test_transform)
    dataset_s5 = SteelDefectDataset(X_stage_5, transform=test_transform)

    loader_s4 = DataLoader(dataset_s4, batch_size=BATCH_SIZE, shuffle=False)
    loader_s5 = DataLoader(dataset_s5, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model weights from {MODEL_PATH}...")
    model = models.resnet34() # ENSURE THIS MATCHES YOUR SAVED MODEL ARCHITECTURE
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    def get_predictions(loader):
        preds = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
        return np.array(preds)

    print("\nAnalyzing Stage 4...")
    preds_s4 = get_predictions(loader_s4)
    counts_s4 = np.bincount(preds_s4, minlength=N_CLASSES)
    dist_s4 = (counts_s4 / len(preds_s4)) * 100
    print(f"Prediction counts: {counts_s4}")
    print(f"Percentage distribution: {np.round(dist_s4, 1)}%")

    print("\nAnalyzing Stage 5...")
    preds_s5 = get_predictions(loader_s5)
    counts_s5 = np.bincount(preds_s5, minlength=N_CLASSES)
    dist_s5 = (counts_s5 / len(preds_s5)) * 100
    print(f"Prediction counts: {counts_s5}")
    print(f"Percentage distribution: {np.round(dist_s5, 1)}%")

if __name__ == "__main__":
    run_pseudo_analysis()