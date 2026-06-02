"""
src/models/efficientnet_model.py
EfficientNet-B3 모델 정의 — 클래스 수만 바꾸면 재사용 가능
"""

import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_efficientnet_b3(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    """
    EfficientNet-B3 pretrained 로드 후 classifier 교체

    Args:
        num_classes    : 출력 클래스 수 (토마토 10클래스)
        freeze_backbone: True면 feature extractor 동결 (초반 학습용)

    Returns:
        nn.Module
    """
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Epoch 10 이후 backbone unfreeze (fine-tuning 전환)"""
    for param in model.features.parameters():
        param.requires_grad = True