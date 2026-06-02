import torch
import timm


def get_tomato_resnet50(num_classes: int, pretrained: bool = True):
    """
    토마토 질병 분류를 위한 ResNet50 모델을 생성하여 반환.
    """
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    return model


if __name__ == '__main__':
    # 코드 단독 실행 테스트용
    mock_num_classes = 11
    model = get_tomato_resnet50(num_classes=mock_num_classes, pretrained=False)

    # 8장 배치, 3채널, 224x224 가짜 이미지 입력 테스트
    mock_input = torch.randn(8, 3, 224, 224)
    output = model(mock_input)

    print("ResNet50 모델 빌드 성공!")
    print(f"입력 크기: {mock_input.shape}")
    print(f"출력 크기 (배치 사이즈, 클래스 개수): {output.shape}")