"""
src/dataset.py
데이터 경로 설정 / 전처리 / DataLoader 반환
"""

import os
import copy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── 클래스 정의 (폴더명과 순서 일치시킬 것) ──────────────────
CLASS_NAMES = [
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Spider_Mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Mosaic_Virus",
    "Tomato_Healthy",
]
NUM_CLASSES = len(CLASS_NAMES)   # 10

# ── Transform 정의 ────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    data_dir 구조:
        data_dir/
          train/
            Tomato_Bacterial_Spot/  ← 클래스 폴더
            Tomato_Early_Blight/
            ...
          test/
            Tomato_Bacterial_Spot/
            ...

    Returns:
        train_loader, val_loader, test_loader
    """
    # train 전체 로드
    full_train = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=TRAIN_TRANSFORM,
    )

    # train / val 분리
    val_size   = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # val에는 augmentation 미적용
    val_ds.dataset = copy.deepcopy(full_train)
    val_ds.dataset.transform = VAL_TRANSFORM

    # test 로드
    test_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=VAL_TRANSFORM,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[Dataset] Train: {train_size} | Val: {val_size} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader