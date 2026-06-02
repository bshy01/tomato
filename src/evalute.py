"""
src/evaluate.py
저장된 가중치로 test셋 정확도 + confusion matrix 출력
실행: python src/evaluate.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from dataset import get_dataloaders, CLASS_NAMES
from models.efficientnet_model import build_efficientnet_b3

WEIGHT_PATH = "./weights/efficientnet_b3_tomato.pth"
DATA_DIR    = "./data/tomato"
BATCH_SIZE  = 32


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total      += images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — EfficientNet-B3")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=150)
    print("[SAVED] confusion_matrix.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    model = build_efficientnet_b3(num_classes=10, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    print(f"[INFO] 가중치 로드: {WEIGHT_PATH}")

    criterion = nn.CrossEntropyLoss()
    test_loss, preds, labels = evaluate(model, test_loader, criterion, device)

    acc = (preds == labels).mean()
    print(f"\n[TEST] Loss: {test_loss:.4f} | Accuracy: {acc:.4f}")
    print("\n[Classification Report]")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    plot_confusion_matrix(preds, labels, CLASS_NAMES)


if __name__ == "__main__":
    main()