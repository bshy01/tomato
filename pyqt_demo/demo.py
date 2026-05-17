import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtGui import QPixmap
from PIL import Image
from torchvision import transforms

# Matplotlib을 PyQt5에 임베딩하기 위한 모듈
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 기존에 작성하신 factory 및 스펙 구조 임포트
from src.models.model_factory import get_model, get_model_specs


class TomatoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_list = ['resnet50', 'efficientnet_b0', 'vit']
        # 이미지 가공 시 확인한 11개 클래스명을 순서대로 적어주세요.
        self.class_names = [
            'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
            'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
            'Tomato_Yellow_Leaf_Curl', 'Tomato_mosaic_virus', 'powdery_mildew', 'healthy'
        ]
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🍅 Tomato Leaf Disease Multi-Model Benchmark (AI CELL)')
        self.resize(1200, 700)

        # 메인 레이아웃 (좌우 분할)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # --- [좌측 영역]: 이미지 및 테이블 ---
        self.btn_upload = QPushButton('📸 Upload Tomato Leaf Image', self)
        self.btn_upload.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        self.btn_upload.clicked.connect(self.upload_image)
        left_layout.addWidget(self.btn_upload)

        self.lbl_img = QLabel('Upload an image to start benchmark', self)
        self.lbl_img.setStyleSheet("border: 2px dashed #aaa; background-color: #fafafa; color: #777;")
        self.lbl_img.setFixedSize(350, 350)
        self.lbl_img.setScaledContents(True)
        left_layout.addWidget(self.lbl_img, alignment=torch.Qt.AlignCenter)

        # 3개 모델 결과를 보여줄 테이블
        self.table_result = QTableWidget(3, 2)
        self.table_result.setHorizontalHeaderLabels(['Model', 'Predicted Class'])
        self.table_result.verticalHeader().setVisible(False)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, m_name in enumerate(self.models_list):
            self.table_result.setItem(i, 0, QTableWidgetItem(m_name))
            self.table_result.setItem(i, 1, QTableWidgetItem('-'))
        left_layout.addWidget(self.table_result)

        # --- [우측 영역]: Matplotlib 확률 그래프 ---
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)

        # 레이아웃 병합
        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(right_layout, stretch=6)
        self.setLayout(main_layout)

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', './', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            pixmap = QPixmap(fname)
            self.lbl_img.setPixmap(pixmap)
            self.predict_all(fname)

    def predict_all(self, img_path):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # 바 차트를 그리기 위한 변수 설정
        x = np.arange(len(self.class_names))
        width = 0.25

        # 3개 모델을 돌면서 예측 진행
        for idx, model_name in enumerate(self.models_list):
            specs = get_model_specs(model_name)

            # 1. 모델별 맞춤 전처리
            transform = transforms.Compose([
                transforms.Resize((specs['input_size'], specs['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=specs['mean'], std=specs['std'])
            ])

            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # 2. 모델 로드 및 가중치 매핑
            weight_path = f"./results/{model_name}.pth"
            if not os.path.exists(weight_path):
                self.table_result.setItem(idx, 1, QTableWidgetItem("Weight missing"))
                continue

            model = get_model(model_name, num_classes=len(self.class_names))
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()

            # 3. 추론 및 Softmax로 확률 변환
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probabilities)

            # 테이블 업데이트
            self.table_result.setItem(idx, 1, QTableWidgetItem(self.class_names[pred_idx]))

            # 그래프에 막대 추가
            ax.bar(x + (idx - 1) * width, probabilities, width, label=model_name)

        # 4. 그래프 디자인 데코레이션
        ax.set_ylabel('Confidence (Probability)')
        ax.set_title('Tomato Disease Softmax Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TomatoApp()
    ex.show()
    sys.exit(app.exec_())