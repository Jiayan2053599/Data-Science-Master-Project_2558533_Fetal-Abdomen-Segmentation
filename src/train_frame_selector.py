# train_frame_selector.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from frame_select_model import ResNet18_1ch
from frame_dataset import FrameDataset
from torchvision import transforms
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# Device configuration
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 20
# Path to save best model
MODEL_DIR  = Path("models")
MODEL_PATH = MODEL_DIR / "frame_selector_best.pth"


def main():
    # 1) Prepare data paths and labels
    INPUT_PATH = Path("../test/input/filtered/images/stacked-fetal-ultrasound")
    MASK_PATH  = Path(r"D:\Data_Science_project-data\acouslic-ai-train-set\masks\stacked_fetal_abdomen")
    img_paths  = sorted(INPUT_PATH.glob("*.mha")) + sorted(INPUT_PATH.glob("*.tiff"))
    labels     = [1 if (MASK_PATH / p.name).exists() else 0 for p in img_paths]

    # 2) Split into train/validation sets
    split = int(0.8 * len(img_paths))
    train_ds = FrameDataset(
        img_paths[:split], labels[:split],
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ])
    )
    val_ds = FrameDataset(
        img_paths[split:], labels[split:], transform=None
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # 3) Initialize model, loss, optimizer
    model = ResNet18_1ch(pretrained=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4) Training and validation loop
    best_val_acc = 0.0
    MODEL_DIR.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labs).sum().item()
            train_total += labs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labs)

                val_loss += loss.item() * labs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labs).sum().item()
                val_total += labs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Print metrics
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Saved best model to {MODEL_PATH} (Val Acc={val_acc:.4f})")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
