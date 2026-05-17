import sys
import os
import torch
import torch.nn as nn
import numpy as np

# PySide6 핵심 모듈 임포트
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtGui import QPixmap
from PIL import Image
from torchvision import transforms, models

class TomatoResNetApp(QWidget):
    def __init__(self):
        super().__init__()
        # 로컬 테스트용이므로 CPU를 기본으로 잡습니다.
        self.device = torch.device("cpu")

        # 토마토 질병 11개 클래스명 정의
        self.class_names = [
            'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
            'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
            'Tomato_Yellow_Leaf_Curl', 'Tomato_mosaic_virus', 'powdery_mildew', 'healthy'
        ]
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🍅 Tomato Leaf Disease - ResNet50 Test (AI CELL)')
        self.resize(500, 600)

        layout = QVBoxLayout()

        # 1. 이미지 업로드 버튼
        self.btn_upload = QPushButton('📸 Upload Tomato Leaf Image', self)
        self.btn_upload.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        self.btn_upload.clicked.connect(self.upload_image)
        layout.addWidget(self.btn_upload)

        # 2. 이미지 미리보기 영역 (정렬 오타 수정 완료)
        self.lbl_img = QLabel('Upload an image to start', self)
        self.lbl_img.setStyleSheet("border: 2px dashed #aaa; background-color: #fafafa; color: #777;")
        self.lbl_img.setFixedSize(350, 350)
        self.lbl_img.setScaledContents(True)
        layout.addWidget(self.lbl_img, alignment=Qt.AlignCenter)

        # 3. 결과 출력 레이블 (정렬 오타 수정 완료)
        self.lbl_result = QLabel('Prediction Result: Waiting for image...', self)
        self.lbl_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-top: 15px;")
        layout.addWidget(self.lbl_result, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', './', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            pixmap = QPixmap(fname)
            self.lbl_img.setPixmap(pixmap)
            self.predict_resnet(fname)

    def predict_resnet(self, img_path):
        # ResNet50 스펙 고정
        input_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # 1. 전처리
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # 2. 모델 정의 (가상의 빈 모델 생성)
        num_classes = len(self.class_names)
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # 로컬 results/ 폴더에 가중치 파일이 있다면 로드
        weight_path = "./results/resnet50.pth"
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            status = "Loaded trained weights"
        else:
            status = "Using Random weights (No .pth file found)"

        model.eval()

        # 3. 추론
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            pred_idx = np.argmax(probabilities)
            confidence = probabilities[pred_idx] * 100

        self.lbl_result.setText(f"Result: {self.class_names[pred_idx]} ({confidence:.2f}%)\n[{status}]")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TomatoResNetApp()
    ex.show()
    sys.exit(app.exec())