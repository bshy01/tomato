"""
src/predict.py
단일 이미지 → Top-3 예측 반환
PyQt5 GUI에서 import해서 사용

사용 예:
    from predict import TomatoPredictor
    predictor = TomatoPredictor("./weights/efficientnet_b3_tomato.pth")
    results = predictor.predict("leaf.jpg")
    # [("Tomato_Early_Blight", 0.92), ("Tomato_Late_Blight", 0.05), ...]
"""

import torch
from torchvision import transforms
from PIL import Image

from dataset import CLASS_NAMES
from models.efficientnet_model import build_efficientnet_b3

TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TomatoPredictor:
    def __init__(self, weight_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_efficientnet_b3(num_classes=10, freeze_backbone=False)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[Predictor] 모델 로드 완료 | Device: {self.device}")

    def predict(self, image_path: str, top_k: int = 3) -> list[tuple[str, float]]:
        """
        Args:
            image_path : 이미지 파일 경로
            top_k      : 반환할 상위 예측 수 (기본 3)

        Returns:
            [(클래스명, 확률), ...]  ex) [("Tomato_Early_Blight", 0.923), ...]
        """
        img    = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs  = torch.softmax(output, dim=1)[0]
            top    = torch.topk(probs, k=top_k)

        return [
            (CLASS_NAMES[idx.item()], round(prob.item(), 4))
            for idx, prob in zip(top.indices, top.values)
        ]


# 단독 실행 테스트용
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python src/predict.py <이미지경로>")
        sys.exit(1)

    predictor = TomatoPredictor("./weights/efficientnet_b3_tomato.pth")
    results   = predictor.predict(sys.argv[1])

    print("\n[예측 결과 Top-3]")
    for rank, (cls, prob) in enumerate(results, 1):
        print(f"  {rank}. {cls:<40} {prob*100:.2f}%")