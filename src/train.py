"""
src/train.py
학습 실행 진입점 — `python src/train.py` 로 실행
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset import get_dataloaders
from models.efficientnet_model import build_efficientnet_b3, unfreeze_backbone

# ── 설정 ─────────────────────────────────────────────────────
DATA_DIR   = "./data/tomato"          # ← 본인 데이터 경로로 수정
SAVE_PATH  = "./weights/efficientnet_b3_tomato.pth"
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 3e-4
SEED       = 42
# ─────────────────────────────────────────────────────────────


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience  = patience
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def main():
    import os
    os.makedirs("./weights", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 데이터
    train_loader, val_loader, _ = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, seed=SEED
    )

    # 모델
    model = build_efficientnet_b3(num_classes=10, freeze_backbone=True).to(device)

    # 손실 / 옵티마이저 / 스케줄러
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler     = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stop    = EarlyStopping(patience=5)
    best_val_acc  = 0.0

    print("\n[START] 학습 시작\n" + "=" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Epoch 10 → backbone unfreeze
        if epoch == 10:
            unfreeze_backbone(model)
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
            print("[INFO] Epoch 10: Backbone Unfreeze (Fine-tuning 시작)")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Best 저장 (Val Acc: {best_val_acc:.4f})")

        early_stop(val_loss)
        if early_stop.stop:
            print(f"[EarlyStopping] Epoch {epoch} 조기 종료")
            break

    print(f"\n[DONE] Best Val Accuracy: {best_val_acc:.4f}")
    print(f"[DONE] 저장 경로: {SAVE_PATH}")


if __name__ == "__main__":
    main()