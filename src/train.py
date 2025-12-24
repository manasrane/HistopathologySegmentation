import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import kagglehub

from dataset import GLaSDataset
from model import UNet

# ======================
# Config
# ======================
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4

MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "unet.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Download Dataset
# ======================
dataset_path = kagglehub.dataset_download(
    "sani84/glasmiccai2015-gland-segmentation"
)
dataset_root = os.path.join(
    dataset_path,
    "Warwick_QU_Dataset"
)
# ======================
# Dataset & Loader
# ======================
dataset = GLaSDataset(
    root_dir=dataset_root,
    split="train"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# ======================
# Model
# ======================
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
bce_loss = torch.nn.BCELoss()

# ======================
# Dice Loss
# ======================
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)

# ======================
# Training Loop
# ======================
os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for imgs, masks in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

# ======================
# Save Model
# ======================
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
