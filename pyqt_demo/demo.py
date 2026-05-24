import sys
import os
import torch
import torch.nn as nn
import numpy as np

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal  # PyQt5 전용 pyqtSignal 임포트
from PyQt5.QtGui import QPixmap, QColor
from PIL import Image
from torchvision import transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.models.model_factory import get_model, get_model_specs


class InferenceWorker(QThread):
    """
    대용량 모델 추론 시 GUI 프레임 드랍(Freezing)을 방지하기 위한 백그라운드 스레드입니다.
    """
    # PyQt5 전용 시그널 정의: (모델인덱스, 예측클래스명, 예측확률, 전체확률배열)
    finished = pyqtSignal(int, str, float, np.ndarray)

    def __init__(self, idx, model_name, img_path, class_names, device):
        super().__init__()
        self.idx = idx
        self.model_name = model_name
        self.img_path = img_path
        self.class_names = class_names
        self.device = device

    def run(self):
        try:
            # 실행 경로에 무관하도록 스크립트 위치 기준 절대 경로로 가중치 탐색
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            weight_path = os.path.join(project_root, "results", f"{self.model_name}.pth")

            if not os.path.exists(weight_path):
                self.finished.emit(self.idx, "Weight missing", 0.0, np.array([]))
                return

            # 모델별 전처리 고유 스펙 매핑 (Input Size, Mean, Std)
            specs = get_model_specs(self.model_name)
            transform = transforms.Compose([
                transforms.Resize((specs['input_size'], specs['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=specs['mean'], std=specs['std'])
            ])

            image = Image.open(self.img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            # 팩토리 패턴 기반 모델 아키텍처 동적 로드 및 가중치 매핑
            model = get_model(self.model_name, num_classes=len(self.class_names))
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()

            # 피드포워드 및 Softmax 연산으로 클래스별 신뢰도 계산
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probabilities)

            pred_class = self.class_names[pred_idx]
            pred_conf = float(probabilities[pred_idx])

            # 결과를 메인 GUI 스레드로 송신
            self.finished.emit(self.idx, pred_class, pred_conf, probabilities)
        except Exception as e:
            self.finished.emit(self.idx, f"Error: {str(e)}", 0.0, np.array([]))


class TomatoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_list = ['resnet50', 'efficientnet_b0', 'vit']
        self.class_names = [
            'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
            'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot',
            'Tomato_Yellow_Leaf_Curl', 'Tomato_mosaic_virus', 'powdery_mildew', 'healthy'
        ]
        self.current_img_path = None
        self.active_workers = []  # 실행 중인 비동기 스레드 추적 제어 레이어
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tomato Leaf Disease Benchmarking System')
        self.resize(1300, 700)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 모델 탐색 및 개별 검증을 위한 콤보박스 세팅
        combo_layout = QHBoxLayout()
        lbl_combo = QLabel('Select Model:', self)
        lbl_combo.setStyleSheet("font-weight: bold;")

        self.combo_model = QComboBox(self)
        self.combo_model.addItem('All Models (Benchmark)')
        self.combo_model.addItems(self.models_list)
        self.combo_model.currentIndexChanged.connect(self.on_model_changed)

        combo_layout.addWidget(lbl_combo)
        combo_layout.addWidget(self.combo_model, stretch=1)
        left_layout.addLayout(combo_layout)

        # 파일 탐색기 기반 업로드 버튼
        self.btn_upload = QPushButton('Select Image File...', self)
        self.btn_upload.clicked.connect(self.upload_image)
        left_layout.addWidget(self.btn_upload)

        # 드래그 앤 드롭 입력을 수용하는 이미지 프리뷰 레이블
        self.lbl_img = QLabel('\n\n\nDrag & Drop Image Here\n(or click the button above)', self)
        self.lbl_img.setStyleSheet(
            "border: 2px dashed #cccccc; border-radius: 6px; background-color: #f9f9f9; color: #888888; font-size: 13px;")
        self.lbl_img.setFixedSize(380, 350)
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setAlignment(Qt.AlignCenter)

        # 드롭 이벤트 활성화
        self.setAcceptDrops(True)
        left_layout.addWidget(self.lbl_img, alignment=Qt.AlignCenter)

        # 모델별 정량 결과 매핑 테이블 인터페이스
        self.table_result = QTableWidget(3, 2)
        self.table_result.setHorizontalHeaderLabels(['Model Architecture', 'Diagnostic Result'])
        self.table_result.verticalHeader().setVisible(False)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reset_table()
        left_layout.addWidget(self.table_result)

        # 확률 분포 시각화를 위한 Matplotlib 캔버스 임베딩
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)

        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(right_layout, stretch=6)
        self.setLayout(main_layout)

    def reset_table(self):
        for i, m_name in enumerate(self.models_list):
            self.table_result.setItem(i, 0, QTableWidgetItem(m_name))
            self.table_result.setItem(i, 1, QTableWidgetItem('-'))

    # OS 드래그 이벤트 인터페이스 정의
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    # 드롭 발생 시 확장자 검증 후 파이프라인 트리거
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            fname = url.toLocalFile()
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_new_image(fname)
                break

    def upload_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', './', 'Image files (*.jpg *.png *.jpeg)')
        if fname:
            self.process_new_image(fname)

    def process_new_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.lbl_img.setPixmap(pixmap)
        self.current_img_path = img_path
        self.start_async_prediction(img_path)

    def on_model_changed(self):
        if self.current_img_path:
            self.start_async_prediction(self.current_img_path)

    def start_async_prediction(self, img_path):
        """
        이전 호출 스레드를 강제 자원 해제(Terminate)하고 조건에 맞는 모델의 비동기 연산을 새로 시작합니다.
        """
        for worker in self.active_workers:
            worker.terminate()
            worker.wait()
        self.active_workers.clear()

        self.fig.clear()
        self.canvas.draw()

        selected_option = self.combo_model.currentText()
        self.all_probabilities = {}

        for idx, model_name in enumerate(self.models_list):
            # 콤보박스 선택 사양에 맞춰 타겟 모델 외 스킵 처리
            if 'All Models' not in selected_option and model_name != selected_option:
                self.table_result.setItem(idx, 1, QTableWidgetItem('Skipped'))
                continue

            self.table_result.setItem(idx, 1, QTableWidgetItem('Analyzing...'))

            # 백그라운드 추론 스레드 가동
            worker = InferenceWorker(idx, model_name, img_path, self.class_names, self.device)
            worker.finished.connect(self.on_prediction_finished)
            self.active_workers.append(worker)
            worker.start()

    def on_prediction_finished(self, idx, pred_class, pred_conf, probabilities):
        """
        추론 스레드가 종료되면 호출되는 콜백 함수로, 테이블 UI 갱신 및 데이터 수집을 수행합니다.
        """
        model_name = self.models_list[idx]

        if len(probabilities) == 0:
            self.table_result.setItem(idx, 1, QTableWidgetItem(pred_class))
            return

        result_text = f"{pred_class} ({pred_conf * 100:.1f}%)"
        item = QTableWidgetItem(result_text)

        # 진단 결과(정상/질병)에 따른 가시성 최적화 컬러 매핑 분기
        if pred_class == 'healthy':
            item.setBackground(QColor('#e2f0d9'))  # 소프트 그린 테마
            item.setForeground(QColor('#385723'))
        else:
            item.setBackground(QColor('#fce4d6'))  # 소프트 오렌지 테마
            item.setForeground(QColor('#c65911'))

        self.table_result.setItem(idx, 1, item)
        self.all_probabilities[model_name] = probabilities

        # 선택 조건에 맞는 스레드의 데이터 집적이 끝나면 통합 시각화 차트 트리거
        selected_option = self.combo_model.currentText()
        expected_count = 3 if 'All Models' in selected_option else 1
        if len(self.all_probabilities) == expected_count:
            self.draw_horizontal_chart()

    def draw_horizontal_chart(self):
        """
        긴 클래스명의 폰트 깨짐 및 가독성을 방지하기 위해 Seaborn 스타일의 가로형 막대그래프를 인쇄합니다.
        """
        self.fig.clear()

        plt.style.use('seaborn-v0_8-whitegrid')
        ax = self.fig.add_subplot(111)

        y = np.arange(len(self.class_names))
        height = 0.25

        selected_option = self.combo_model.currentText()

        for idx, model_name in enumerate(self.models_list):
            if model_name not in self.all_probabilities:
                continue

            probs = self.all_probabilities[model_name]

            # 벤치마크 모드와 단독 모드 스케일 격차 분기 처리
            if 'All Models' in selected_option:
                ax.barh(y + (idx - 1) * height, probs, height, label=model_name)
            else:
                ax.barh(y, probs, height * 1.5, label=model_name)

        ax.set_xlabel('Confidence Probability')
        ax.set_title('Model Softmax Statistical Analysis')
        ax.set_yticks(y)
        ax.set_yticklabels(self.class_names)
        ax.set_xlim(0, 1.05)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 💡 PyQt5의 크로스플랫폼 오피스 룩 스타일 고정 적용
    app.setStyle('Fusion')

    ex = TomatoApp()
    ex.show()
    sys.exit(app.exec())