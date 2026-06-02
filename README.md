# 🍅 토마토 잎 질병 분류

> 딥러닝(EfficientNet-B3)을 활용한 토마토 잎 질병 10종 분류 시스템

**과목:** 인공지능 캡스톤 디자인 | **팀명:** 멋쟁이토마토  
**팀원:** 윤수환 · 임재무 · 류혜원 · 모아사랑

---

## 📁 폴더 구조

```
tomato-disease-classification/
├── src/
│   ├── models/
│   │   └── efficientnet_model.py  # 모델 정의
│   ├── dataset.py                 # 데이터 로드/전처리
│   ├── train.py                   # 학습 실행
│   ├── evaluate.py                # 성능 평가
│   └── predict.py                 # 단일 이미지 추론 (PyQt 연동)
├── data/
│   └── tomato/
│       ├── train/                 # 클래스별 학습 이미지
│       └── test/                  # 클래스별 테스트 이미지
├── weights/                       # 학습된 가중치 (.pth)
├── requirements.txt
└── README.md
```

## 🌿 분류 클래스 (10종)

| # | 클래스 |
|---|--------|
| 0 | Tomato_Bacterial_Spot |
| 1 | Tomato_Early_Blight |
| 2 | Tomato_Late_Blight |
| 3 | Tomato_Leaf_Mold |
| 4 | Tomato_Septoria_Leaf_Spot |
| 5 | Tomato_Spider_Mites |
| 6 | Tomato_Target_Spot |
| 7 | Tomato_Yellow_Leaf_Curl_Virus |
| 8 | Tomato_Mosaic_Virus |
| 9 | Tomato_Healthy |

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

## 🚀 실행

```bash
# 학습
python src/train.py

# 평가
python src/evaluate.py

# 단일 이미지 추론
python src/predict.py ./data/tomato/test/Tomato_Early_Blight/sample.jpg
```

## 🏗️ 모델

- **Architecture:** EfficientNet-B3 (Transfer Learning)
- **Pretrained:** ImageNet
- **Optimizer:** Adam (lr=3e-4)
- **Scheduler:** CosineAnnealingWarmRestarts
- **Input Size:** 300×300
