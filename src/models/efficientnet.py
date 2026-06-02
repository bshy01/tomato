"""
src/models/efficientnet.py
EfficientNet-B0 모델 정의
팀: 멋쟁이토마토 | 담당: 모아사랑
"""

import torch
import torch.nn as nn
import torchvision.models as models


def efficientnet_b0(pretrained=False, num_classes=10, **kwargs):
    """EfficientNet-B0 모델 생성"""
    model = EfficientNetB0(num_classes=num_classes, **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(model.model_path))

    return model


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, model_path="efficientnet_b0.pth"):
        super().__init__()

        self.model_path = model_path

        backbone = models.efficientnet_b0(weights='DEFAULT')

        self.features   = backbone.features
        self.avgpool    = backbone.avgpool

        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 모델 정상 작동 테스트용
if __name__ == "__main__":
    model = efficientnet_b0(num_classes=10)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("EfficientNet-B0 모델 생성 및 연산 성공")
    print("출력 텐서 크기:", output.shape)