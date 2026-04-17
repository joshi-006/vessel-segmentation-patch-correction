"""
train_ensemble.py
-----------------
Trains a 5-model FR-UNet ensemble for retinal vessel segmentation.
Each model is trained with a different random seed for MC-Dropout diversity.

Usage:
    python train_ensemble.py

Requirements:
    pip install torch torchvision albumentations opencv-python tqdm scikit-learn scikit-image
    git clone https://github.com/berenslab/MIDL24-segmentation_quality_control.git
"""

import os
import sys
import random
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from sklearn.model_selection import train_test_split

# ── FR-UNet import ────────────────────────────────────────────────────────────
REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "MIDL24-segmentation_quality_control")
if not os.path.isdir(REPO_PATH):
    raise RuntimeError(
        f"FR-UNet repo not found at {REPO_PATH}.\n"
        "Run: git clone https://github.com/berenslab/MIDL24-segmentation_quality_control.git"
    )
sys.path.insert(0, REPO_PATH)
from models.frunet import FR_UNet


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these directly
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT   = "/path/to/fives"        # ← change this
SAVE_DIR    = "./models"
N_MODELS    = 5
EPOCHS      = 100
LR          = 1e-4
BATCH_SIZE  = 8
NUM_WORKERS = 4
VAL_SPLIT   = 0.15
IMG_SIZE    = 512
DROPOUT     = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.CLAHE(clip_limit=4.0, p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.ElasticTransform(alpha=80, sigma=8, p=0.3),
    A.GridDistortion(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
])

class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, file_list=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = file_list or sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name  = self.images[idx]
        img   = cv2.imread(os.path.join(self.img_dir, name))
        green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img[:, :, 1])
        img   = cv2.resize(np.stack([green]*3, axis=-1), (IMG_SIZE, IMG_SIZE))
        mask  = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask  = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        img  = torch.tensor((img / 255.0 - 0.5) / 0.5, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)
        return img, mask

def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred).view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)

def bce_dice_loss(pred, target, alpha=0.25, gamma=2.0):
    prob  = torch.sigmoid(pred)
    focal = (
        -alpha * (1 - prob) ** gamma * target * torch.log(prob + 1e-6)
        - (1 - alpha) * prob ** gamma * (1 - target) * torch.log(1 - prob + 1e-6)
    ).mean()
    return 0.5 * focal + 0.5 * dice_loss(pred, target)

def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred   = (torch.sigmoid(pred) > threshold).float().view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir  = os.path.join(DATA_ROOT, "train", "Original")
mask_dir = os.path.join(DATA_ROOT, "train", "Ground truth")

all_files = sorted([
    f for f in os.listdir(img_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
])
train_files, val_files = train_test_split(all_files, test_size=VAL_SPLIT, random_state=42)
print(f"Device: {device} | Train: {len(train_files)} | Val: {len(val_files)}")


# ─────────────────────────────────────────────────────────────────────────────
# Train each ensemble member
# ─────────────────────────────────────────────────────────────────────────────

for model_id in range(N_MODELS):
    set_seed(model_id * 100)

    model = FR_UNet(num_classes=1, num_channels=3, dropout=DROPOUT).to(device)

    train_loader = DataLoader(
        VesselDataset(img_dir, mask_dir, train_transform, train_files),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        VesselDataset(img_dir, mask_dir, None, val_files),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    optimizer     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler        = GradScaler()
    best_val_dice = 0.0
    save_path     = os.path.join(SAVE_DIR, f"FRUNet_MC_{model_id}.pth")

    print(f"\n{'='*60}\n  Model {model_id+1}/{N_MODELS}  (seed={model_id*100})\n{'='*60}")

    for epoch in range(EPOCHS):

        # Train
        model.train()
        total_loss = total_dice = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"[{model_id}] Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = bce_dice_loss(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            with torch.no_grad():
                total_dice += dice_score(model(imgs), masks).item()
        scheduler.step()

        # Validate
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                val_dice += dice_score(model(imgs.to(device)), masks.to(device)).item()
        val_dice /= len(val_loader)

        # Log & save best
        saved = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            saved = "  *** saved ***"

        print(
            f"[{model_id}] Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Dice: {total_dice/len(train_loader):.4f} | "
            f"Val Dice: {val_dice:.4f}{saved}"
        )

    print(f"\n[{model_id}] Best val Dice: {best_val_dice:.4f} → {save_path}")

print(f"\n{'='*60}\n  Done. Models saved to: {SAVE_DIR}\n{'='*60}")
