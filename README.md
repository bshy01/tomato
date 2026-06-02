# tomato-disease-classification

tomato-disease-classification/
│
├── src/
│   ├── models/
│   │   └── efficientnet_model.py   ← 모델 정의만
│   ├── dataset.py                  ← 데이터 로드/전처리
│   ├── train.py                    ← 학습 루프 실행
│   ├── evaluate.py                 ← 테스트/성능 평가
│   └── predict.py                  ← 단일 이미지 추론 (PyQt 연동)
│
├── data/
│   └── tomato/
│       ├── train/
│       │   ├── Tomato_Bacterial_Spot/
│       │   ├── Tomato_Early_Blight/
│       │   └── ... (10개 폴더)
│       └── test/
│           └── ... (동일 구조)
│
├── weights/
│   └── efficientnet_b3_tomato.pth  ← 학습된 가중치
│
├── requirements.txt
└── README.md
