# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# --- INSTÄLLNINGAR ---
DATA = Path("data")
N_CLASSES = 6
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Skriv in de exakta namnen på dina sparade modellvikter här!
MODEL_WEIGHTS = [
    "model_lr1_v1.pth",      
    "model_lr1_v2.pth",      
    "model_lr1_v3.pth"    
]

print(f"Using device: {DEVICE}")

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

# Testdatan behöver bara göras om till tensorer (ingen data augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_ensemble_submission():
    print("Loading test data...")
    test_df = pd.read_csv(DATA / "test.csv")
    z = np.load(DATA / "images.npz", allow_pickle=True)
    X_test = z["X_test"]

    test_dataset = SteelDefectDataset(X_test, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ensemble_models = []
    
    # Ladda in alla modeller
    for weight_path in MODEL_WEIGHTS:
        print(f"Loading weights from {weight_path}...")
        model = models.resnet34(weights=None) # Vi behöver inte ladda ner DEFAULT igen, vi skriver över dem direkt
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ensemble_models.append(model)

    print("\nGenerating ENSEMBLE predictions for Stages 4 & 5...")
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            
            # Skapa en tom tensor för att samla ihop sannolikheterna
            batch_probs_sum = torch.zeros(inputs.size(0), N_CLASSES).to(DEVICE)
            
            # Låt varje modell gissa
            for model in ensemble_models:
                logits = model(inputs)
                # Vi måste göra om logits till procent (softmax) INNAN vi snittar
                probs = torch.softmax(logits, dim=1)
                batch_probs_sum += probs
                
            # Ta snittet av sannolikheterna
            batch_probs_avg = batch_probs_sum / len(ensemble_models)
            
            # Välj den klass med högst genomsnittlig sannolikhet
            _, predicted = torch.max(batch_probs_avg, 1)
            predictions.extend(predicted.cpu().numpy())

    sub = pd.DataFrame({
        "id": test_df.image_id, 
        "label": predictions
    })
    sub.to_csv("submission_ensemble.csv", index=False)
    
    pred_counts = np.bincount(predictions, minlength=N_CLASSES)
    print("\nSaved submission_ensemble.csv!")
    print(f"Predicted class counts: {pred_counts}")

if __name__ == "__main__":
    create_ensemble_submission()