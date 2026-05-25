import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor
from PIL import Image
from torchvision import transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2  # Grad-CAM 결과 오버레이 및 핫맵 생성용

from src.models.model_factory import get_model, get_model_specs


class InferenceWorker(QThread):
    """
    백그라운드에서 추론, 시간 측정, Grad-CAM 연산을 수행하는 스레드입니다.
    """
    # 전송 데이터: (모델인덱스, 예측클래스명, 예측확률, 전체확률배열, 추론시간_ms, GradCAM_이미지배열)
    finished = pyqtSignal(int, str, float, np.ndarray, float, np.ndarray)

    def __init__(self, idx, model_name, img_path, class_names, device):
        super().__init__()
        self.idx = idx
        self.model_name = model_name
        self.img_path = img_path
        self.class_names = class_names
        self.device = device

    def run(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            weight_path = os.path.join(project_root, "results", f"{self.model_name}.pth")

            if not os.path.exists(weight_path):
                self.finished.emit(self.idx, "Weight missing", 0.0, np.array([]), 0.0, np.array([]))
                return

            specs = get_model_specs(self.model_name)
            transform = transforms.Compose([
                transforms.Resize((specs['input_size'], specs['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=specs['mean'], std=specs['std'])
            ])

            # 원본 이미지 로드 (Grad-CAM 합성용 배경)
            orig_img = cv2.imread(self.img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_img = cv2.resize(orig_img, (specs['input_size'], specs['input_size']))

            image = Image.open(self.img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            model = get_model(self.model_name, num_classes=len(self.class_names))
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()

            # ----------------------------------------------------
            # 1. 추론 시간 측정 및 피드포워드
            # ----------------------------------------------------
            start_time = time.perf_counter()

            # Grad-CAM을 위한 특성 맵 및 그래디언트 저장을 위한 훅(Hook) 설정
            feature_maps = []
            gradients = []

            def forward_hook(module, input, output):
                feature_maps.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            # 모델별 타겟 레이어 탐색 (일반적인 파이토치 컨벤션 기준 부모 레이어 추적)
            target_layer = None
            if 'resnet' in self.model_name.lower():
                target_layer = model.layer4[-1]
            elif 'efficientnet' in self.model_name.lower():
                target_layer = model.features[-1]
            elif 'vit' in self.model_name.lower():
                target_layer = model.encoder.layers[-1].ln_1  # ViT 특성에 따른 예시 구조

            if target_layer is not None:
                f_hook = target_layer.register_forward_hook(forward_hook)
                b_hook = target_layer.register_backward_hook(backward_hook)

            # 연산 수행
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            pred_idx = np.argmax(probabilities)

            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            # ----------------------------------------------------
            # 2. Grad-CAM 계산 파이프라인
            # ----------------------------------------------------
            cam_rgb = np.array([])
            if target_layer is not None:
                model.zero_grad()
                class_loss = outputs[0, pred_idx]
                class_loss.backward()

                if len(gradients) > 0 and len(feature_maps) > 0:
                    grads = gradients[0].cpu().data.numpy()[0]
                    f_maps = feature_maps[0].cpu().data.numpy()[0]

                    # GAP(Global Average Pooling) 연산으로 채널별 가중치 산출
                    weights = np.mean(grads, axis=(1, 2)) if grads.ndim == 3 else np.mean(grads, axis=0)
                    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

                    for i, w in enumerate(weights):
                        if i < f_maps.shape[0]:
                            cam += w * f_maps[i]

                    # ReLU 및 정규화
                    cam = np.maximum(cam, 0)
                    if np.max(cam) > 0:
                        cam = cam / np.max(cam)
                    cam = cv2.resize(cam, (specs['input_size'], specs['input_size']))

                    # 컬러 맵 입히기 및 오버레이
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    cam_rgb = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

                # 훅 해제 (메모리 누수 방지)
                f_hook.remove()
                b_hook.remove()

            pred_class = self.class_names[pred_idx]
            pred_conf = float(probabilities[pred_idx])

            self.finished.emit(self.idx, pred_class, pred_conf, probabilities, inference_time_ms, cam_rgb)
        except Exception as e:
            self.finished.emit(self.idx, f"Error: {str(e)}", 0.0, np.array([]), 0.0, np.array([]))


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

        # 📝 [추가] 농약 및 방제 가이드 데이터베이스 데이터셋 정의
        self.prevention_guides = {
            'Bacterial_spot': "병징 초기 구리 수화제 혹은 가스미민 수화제 살포. 작업 전 기구 소독 필수.",
            'Early_blight': "작물 패부 환기 철저히 유도. 다코닐 수화제 또는 플루아지남 분무 적용.",
            'Late_blight': "고온다습 환경 유의. 메탈락실 침투성 살균제 및 전용 방제 유제 교독 처방.",
            'Leaf_Mold': "하우스 내부 습도를 낮추고 통풍 유도. 폴리옥신 수화제 주기적 살포.",
            'Septoria_leaf_spot': "병든 잎은 즉시 제거 후 소각. 만코제브 수화제 등을 통한 조기 예방.",
            'Spider_mites': "초기 발견이 중요. 아바메크린 유제 등 응애 전용 약제를 잎 뒷면에 집중 살포.",
            'Target_Spot': "이프로디온 수화제 도포. 배수성 개선 및 토양 멀칭 상태 점검.",
            'Tomato_Yellow_Leaf_Curl': "매개충인 '담배가루이' 차단이 핵심. 살충제 격주 방제 및 방충망 설치.",
            'Tomato_mosaic_virus': "바이러스 치료제 없음. 진딧물 철저 방제 및 감염 식물체 전량 제거.",
            'powdery_mildew': "황산 가스 훈증법 활용 가능. 트리데모르프 등 흰가루병 전용 약제 도포.",
            'healthy': "정상 상태입니다. 정기적인 예찰과 적정 시비, 관수 제어를 유지하십시오."
        }

        self.current_img_path = None
        self.active_workers = []
        self.all_probabilities = {}
        self.all_cam_images = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tomato Leaf Disease Advanced Benchmarking System')
        self.resize(1450, 800)  # 시각화 공간 확장을 위해 메인 해상도 증설

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 모델 탑색 및 개별 검증용 콤보박스
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

        # 이미지 업로드 버튼
        self.btn_upload = QPushButton('Select Image File...', self)
        self.btn_upload.clicked.connect(self.upload_image)
        left_layout.addWidget(self.btn_upload)

        # 이미지 드래그앤드롭 프리뷰
        self.lbl_img = QLabel('\n\n\nDrag & Drop Image Here\n(or click the button above)', self)
        self.lbl_img.setStyleSheet(
            "border: 2px dashed #cccccc; border-radius: 6px; background-color: #f9f9f9; color: #888888; font-size: 13px;")
        self.lbl_img.setFixedSize(380, 260)
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        left_layout.addWidget(self.lbl_img, alignment=Qt.AlignCenter)

        # [변경] 추론 시간 항목이 추가된 결과 매핑 테이블 인터페이스
        self.table_result = QTableWidget(3, 3)
        self.table_result.setHorizontalHeaderLabels(['Model Architecture', 'Diagnostic Result', 'Inference Time'])
        self.table_result.verticalHeader().setVisible(False)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reset_table()
        left_layout.addWidget(self.table_result)

        # 🎯 [추가] 농약/방제 가이드 연동 전용 텍스트 필드
        lbl_guide = QLabel('💡 Crop Disease Control & Pesticide Guide:', self)
        lbl_guide.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 5px;")
        left_layout.addWidget(lbl_guide)

        self.txt_guide = QTextEdit(self)
        self.txt_guide.setReadOnly(True)
        self.txt_guide.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ced4da; border-radius: 4px;")
        self.txt_guide.setFixedHeight(120)
        left_layout.addWidget(self.txt_guide)

        # [변경] 1행 2열 멀티플롯 구조를 위한 Matplotlib 캔버스 임베딩 (왼쪽: XAI, 오른쪽: Top-3)
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)

        main_layout.addLayout(left_layout, stretch=4)
        main_layout.addLayout(right_layout, stretch=6)
        self.setLayout(main_layout)

    def reset_table(self):
        for i, m_name in enumerate(self.models_list):
            self.table_result.setItem(i, 0, QTableWidgetItem(m_name))
            self.table_result.setItem(i, 1, QTableWidgetItem('-'))
            self.table_result.setItem(i, 2, QTableWidgetItem('-'))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

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
        for worker in self.active_workers:
            worker.terminate()
            worker.wait()
        self.active_workers.clear()

        self.fig.clear()
        self.canvas.draw()
        self.txt_guide.clear()

        selected_option = self.combo_model.currentText()
        self.all_probabilities = {}
        self.all_cam_images = {}

        for idx, model_name in enumerate(self.models_list):
            if 'All Models' not in selected_option and model_name != selected_option:
                self.table_result.setItem(idx, 1, QTableWidgetItem('Skipped'))
                self.table_result.setItem(idx, 2, QTableWidgetItem('Skipped'))
                continue

            self.table_result.setItem(idx, 1, QTableWidgetItem('Analyzing...'))
            self.table_result.setItem(idx, 2, QTableWidgetItem('Running...'))

            worker = InferenceWorker(idx, model_name, img_path, self.class_names, self.device)
            worker.finished.connect(self.on_prediction_finished)
            self.active_workers.append(worker)
            worker.start()

    def on_prediction_finished(self, idx, pred_class, pred_conf, probabilities, inference_time_ms, cam_rgb):
        model_name = self.models_list[idx]

        if len(probabilities) == 0:
            self.table_result.setItem(idx, 1, QTableWidgetItem(pred_class))
            self.table_result.setItem(idx, 2, QTableWidgetItem('Error'))
            return

        # 1. 결과 테이블 출력 및 컬러 하이라이팅
        result_text = f"{pred_class} ({pred_conf * 100:.1f}%)"
        item_res = QTableWidgetItem(result_text)

        # ⏱️ [추가] 정밀 측정된 추론 시간 UI 바인딩
        item_time = QTableWidgetItem(f"{inference_time_ms:.1f} ms")

        if pred_class == 'healthy':
            item_res.setBackground(QColor('#e2f0d9'))
            item_res.setForeground(QColor('#385723'))
        else:
            item_res.setBackground(QColor('#fce4d6'))
            item_res.setForeground(QColor('#c65911'))

        self.table_result.setItem(idx, 1, item_res)
        self.table_result.setItem(idx, 2, item_time)

        self.all_probabilities[model_name] = probabilities
        self.all_cam_images[model_name] = cam_rgb

        # 2. 메인 모델 방제 가이드 업데이트 (선택된 옵션 기반)
        selected_option = self.combo_model.currentText()

        # All Models 일때는 첫 번째 연산이 끝난 모델 혹은 대표 결과 바인딩, 단독일 땐 해당 모델 바인딩
        if 'All Models' not in selected_option and model_name == selected_option:
            self.update_guide_text(pred_class)
        elif 'All Models' in selected_option and len(self.all_probabilities) == 1:
            # 벤치마크 모드에서는 우선 가장 먼저 분석이 끝난 모델의 질병 가이드를 띄워줌
            self.update_guide_text(pred_class)

        # 3. 캔버스 통합 렌더링 검증 트리거
        expected_count = 3 if 'All Models' in selected_option else 1
        if len(self.all_probabilities) == expected_count:
            self.draw_advanced_plots()

    def update_guide_text(self, disease_name):
        """방제 가이드 창에 텍스트 서식을 지정하여 연동합니다."""
        guide_msg = self.prevention_guides.get(disease_name, "등록된 방제 정보가 없습니다.")
        self.txt_guide.setHtml(f"<b>[진단 정보: {disease_name}]</b><br><br>{guide_msg}")

    def draw_advanced_plots(self):
        """
        [변경 및 추가] 왼쪽 레이아웃엔 Grad-CAM 시각화, 오른쪽 레이아웃엔 Top-3 가로 막대 그래프 배치
        """
        self.fig.clear()
        plt.style.use('seaborn-v0_8-whitegrid')

        selected_option = self.combo_model.currentText()

        # 시각화할 메인 타겟 모델 설정 (All Models 모드인 경우 성능 지표 모니터링을 위해 첫번째 모델 대표 출력)
        target_model = self.models_list[0] if 'All Models' in selected_option else selected_option

        # ----------------------------------------------------
        # 좌측 Subplot: Grad-CAM XAI 이미지 출력
        # ----------------------------------------------------
        ax_cam = self.fig.add_subplot(121)
        if target_model in self.all_cam_images and self.all_cam_images[target_model].size > 0:
            ax_cam.imshow(self.all_cam_images[target_model])
            ax_cam.set_title(f"Grad-CAM XAI Layer ({target_model})", fontsize=11, fontweight='bold')
        else:
            ax_cam.text(0.5, 0.5, "Grad-CAM Layer\nNot Supported for This Architecture",
                        ha='center', va='center', fontsize=10)
        ax_cam.axis('off')

        # ----------------------------------------------------
        # 우측 Subplot: 질병 신뢰도 Top-3 가로 막대 차트 연산
        # ----------------------------------------------------
        ax_chart = self.fig.add_subplot(122)

        if 'All Models' in selected_option:
            # 벤치마크 모드: 모든 모델의 전체 클래스를 보면 복잡하므로, 대표성 있는 타겟 모델의 Top-3 클래스 통계 스냅샷 출력
            probs = self.all_probabilities[target_model]
            top3_idx = np.argsort(probs)[-3:]  # 하위 값 순서이므로 끝에서 3개 슬라이싱

            y_pos = np.arange(3)
            ax_chart.barh(y_pos, probs[top3_idx], color='#4a90e2', height=0.5, edgecolor='black')
            ax_chart.set_yticks(y_pos)
            ax_chart.set_yticklabels([self.class_names[i] for i in top3_idx], fontsize=10, fontweight='bold')
            ax_chart.set_title(f"Top-3 Class Probability ({target_model})", fontsize=11, fontweight='bold')
        else:
            # 단독 모델 모드: 선택 모델의 명확한 상위 Top-3 확률 분포 분석
            probs = self.all_probabilities[target_model]
            top3_idx = np.argsort(probs)[-3:]

            y_pos = np.arange(3)
            ax_chart.barh(y_pos, probs[top3_idx], color='#e67e22', height=0.5, edgecolor='black')
            ax_chart.set_yticks(y_pos)
            ax_chart.set_yticklabels([self.class_names[i] for i in top3_idx], fontsize=10, fontweight='bold')
            ax_chart.set_title(f"Top-3 Class Statistical Analysis", fontsize=11, fontweight='bold')

        ax_chart.set_xlabel('Confidence Probability Values')
        ax_chart.set_xlim(0, 1.05)
        ax_chart.grid(True, linestyle='--', alpha=0.6, axis='x')

        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = TomatoApp()
    ex.show()
    sys.exit(app.exec())