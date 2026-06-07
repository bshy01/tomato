import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
import requests

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox, QTextEdit, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor
from PIL import Image
from torchvision import transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_factory import get_model, get_model_specs


class LLMNetworkWorker(QThread):
    reply_ready = pyqtSignal(str)

    def __init__(self, server_url, chosen_class, question):
        super().__init__()
        self.server_url = server_url
        self.chosen_class = chosen_class
        self.question = question

    def run(self):
        try:
            payload = {
                'chosen_class': self.chosen_class,
                'question': self.question
            }
            response = requests.post(self.server_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json().get('response', '답변을 파싱할 수 없습니다.')
                self.reply_ready.emit(result)
            else:
                self.reply_ready.emit(f"서버 응답 에러 (코드: {response.status_code})")
        except Exception as e:
            self.reply_ready.emit(f"리눅스 AI 서버와 통신 실패: {str(e)}")


class InferenceWorker(QThread):
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
            # 기본 경로: outputs/{model_name}.pth
            weight_path = os.path.join(project_root, "outputs", f"{self.model_name}.pth")
            
            # 만약 위 경로에 파일이 없고, train.py가 생성한 'best_experiment_vX.pth' 형식을 따를 경우를 위한 매핑
            if not os.path.exists(weight_path):
                exp_mapping = {
                    'resnet50': 'best_experiment_v1.pth',
                    'vit': 'best_experiment_v2.pth',
                    'efficientnet_b0': 'best_experiment_v3.pth'
                }
                if self.model_name in exp_mapping:
                    weight_path = os.path.join(project_root, "outputs", exp_mapping[self.model_name])
            
            specs = get_model_specs(self.model_name)
            transform = transforms.Compose([
                transforms.Resize((specs['input_size'], specs['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=specs['mean'], std=specs['std'])
            ])

            orig_img = cv2.imread(self.img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_img = cv2.resize(orig_img, (specs['input_size'], specs['input_size']))

            image = Image.open(self.img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)

            model = get_model(self.model_name, num_classes=len(self.class_names))
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()

            start_time = time.perf_counter()
            feature_maps = []
            gradients = []

            def forward_hook(module, input, output):
                feature_maps.append(output)

            def backward_hook(module, grad_in, grad_out):
                gradients.append(grad_out[0])

            target_layer = None
            if 'resnet' in self.model_name.lower():
                target_layer = model.layer4[-1]
            elif 'efficientnet' in self.model_name.lower():
                target_layer = model.features[-1]
            elif 'vit' in self.model_name.lower():
                target_layer = model.encoder.layers[-1].ln_1

            if target_layer is not None:
                f_hook = target_layer.register_forward_hook(forward_hook)
                b_hook = target_layer.register_full_backward_hook(backward_hook)

            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            pred_idx = np.argmax(probabilities)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            cam_rgb = np.array([])
            if target_layer is not None:
                model.zero_grad()
                class_loss = outputs[0, pred_idx]
                class_loss.backward()

                if len(gradients) > 0 and len(feature_maps) > 0:
                    grads = gradients[0].cpu().data.numpy()[0]
                    f_maps = feature_maps[0].cpu().data.numpy()[0]
                    weights = np.mean(grads, axis=(1, 2)) if grads.ndim == 3 else np.mean(grads, axis=0)
                    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

                    for i, w in enumerate(weights):
                        if i < f_maps.shape[0]:
                            cam += w * f_maps[i]

                    cam = np.maximum(cam, 0)
                    if np.max(cam) > 0:
                        cam = cam / np.max(cam)
                    cam = cv2.resize(cam, (specs['input_size'], specs['input_size']))
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    cam_rgb = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

                f_hook.remove()
                b_hook.remove()

            self.finished.emit(self.idx, self.class_names[pred_idx], float(probabilities[pred_idx]), probabilities,
                               inference_time_ms, cam_rgb)
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

        self.server_url = "http://192.168.0.21:5000/predict"

        self.current_img_path = None
        self.active_workers = []
        self.all_probabilities = {}
        self.all_cam_images = {}
        self.detected_disease = "healthy"
        self.llm_worker = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Tomato Leaf Disease Hybrid Cloud AI Consultation System')
        self.resize(1500, 850)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

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

        self.btn_upload = QPushButton('Select Image File...', self)
        self.btn_upload.clicked.connect(self.upload_image)
        left_layout.addWidget(self.btn_upload)

        self.lbl_img = QLabel('\n\n\nDrag & Drop Image Here\n(or click the button above)', self)
        self.lbl_img.setStyleSheet(
            "border: 2px dashed #cccccc; border-radius: 6px; background-color: #f9f9f9; color: #888888; font-size: 13px;")
        self.lbl_img.setFixedSize(380, 240)
        self.lbl_img.setScaledContents(True)
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        left_layout.addWidget(self.lbl_img, alignment=Qt.AlignCenter)

        self.table_result = QTableWidget(3, 3)
        self.table_result.setHorizontalHeaderLabels(['Model Architecture', 'Diagnostic Result', 'Inference Time'])
        self.table_result.verticalHeader().setVisible(False)
        self.table_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.reset_table()
        left_layout.addWidget(self.table_result)

        lbl_chat = QLabel('리눅스 분산 연산 기반 전문가 실시간 처방 Q&A:', self)
        lbl_chat.setStyleSheet("font-weight: bold; color: #2c3e50; margin-top: 5px;")
        left_layout.addWidget(lbl_chat)

        self.txt_chat_history = QTextEdit(self)
        self.txt_chat_history.setReadOnly(True)
        self.txt_chat_history.setStyleSheet("background-color: #ffffff; border: 1px solid #ced4da; border-radius: 4px;")
        self.txt_chat_history.setFixedHeight(160)
        left_layout.addWidget(self.txt_chat_history)
        self.txt_chat_history.append("<font color='green'>시스템: 원격 리눅스 GPU 추론 연산 노드 바인딩 완료. 이미지를 로드해 주세요.</font><br>")

        self.btn_layout_recommend = QHBoxLayout()
        self.btn_rec_1 = QPushButton("추천 약제 문의", self)
        self.btn_rec_2 = QPushButton("하우스 환경 관리", self)
        self.btn_rec_1.clicked.connect(lambda: self.send_recommended_question(1))
        self.btn_rec_2.clicked.connect(lambda: self.send_recommended_question(2))
        self.btn_rec_1.setEnabled(False)
        self.btn_rec_2.setEnabled(False)
        self.btn_layout_recommend.addWidget(self.btn_rec_1)
        self.btn_layout_recommend.addWidget(self.btn_rec_2)
        left_layout.addLayout(self.btn_layout_recommend)

        input_layout = QHBoxLayout()
        self.input_question = QLineEdit(self)
        self.input_question.setPlaceholderText("토마토 방제에 대해 궁금한 점을 입력하세요...")
        self.input_question.returnPressed.connect(self.send_custom_question)
        self.btn_send_chat = QPushButton("질문하기", self)
        self.btn_send_chat.clicked.connect(self.send_custom_question)
        input_layout.addWidget(self.input_question, stretch=8)
        input_layout.addWidget(self.btn_send_chat, stretch=2)
        left_layout.addLayout(input_layout)

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
        self.txt_chat_history.clear()
        self.txt_chat_history.append("<font color='green'>시스템: 원격 리눅스 GPU 추론 연산 노드 바인딩 완료. 이미지를 로드해 주세요.</font><br>")
        self.btn_rec_1.setEnabled(False)
        self.btn_rec_2.setEnabled(False)

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
            return

        result_text = f"{pred_class} ({pred_conf * 100:.1f}%)"
        item_res = QTableWidgetItem(result_text)
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

        selected_option = self.combo_model.currentText()
        if 'All Models' not in selected_option and model_name == selected_option:
            self.lock_disease_and_enable_chat(pred_class)
        elif 'All Models' in selected_option and len(self.all_probabilities) == 1:
            self.lock_disease_and_enable_chat(pred_class)

        if len(self.all_probabilities) == (3 if 'All Models' in selected_option else 1):
            self.draw_advanced_plots()

    def lock_disease_and_enable_chat(self, disease_name):
        self.detected_disease = disease_name
        self.btn_rec_1.setEnabled(True)
        self.btn_rec_2.setEnabled(True)
        self.btn_rec_1.setText(f"{disease_name} 추천 약제")
        self.btn_rec_2.setText(f"{disease_name} 환경 방제법")
        self.txt_chat_history.append(f"<b>[진단 동기화 완료]</b> 상단 이미지 상태: <b>{disease_name}</b><br>")

    def send_recommended_question(self, type_idx):
        if type_idx == 1:
            question = f"해당 토마토 하우스에 발생한 {self.detected_disease}를 방제하기 위해 농진청에서 권장하는 전용 약제와 살포 주기는 어떻게 되나요?"
        else:
            question = f"현재 발생한 {self.detected_disease}의 확산을 막기 위해 하우스 내부의 온도, 습도 및 조기 전염원 관리를 어떻게 해야 하나요?"
        self.execute_llm_inference(question)

    def send_custom_question(self):
        question = self.input_question.text().strip()
        if not question:
            return
        self.input_question.clear()
        self.execute_llm_inference(question)

    def execute_llm_inference(self, question):
        self.txt_chat_history.append(f"<b>농민:</b> {question}")
        self.txt_chat_history.append("<font color='gray'>원격 리눅스 GPU 서버에서 실시간 가중치 연산 중...</font>")

        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.terminate()
            self.llm_worker.wait()

        self.llm_worker = LLMNetworkWorker(self.server_url, self.detected_disease, question)
        self.llm_worker.reply_ready.connect(self.on_llm_reply_ready)
        self.llm_worker.start()

    def on_llm_reply_ready(self, response):
        cursor = self.txt_chat_history.textCursor()
        cursor.movePosition(cursor.End)
        self.txt_chat_history.setTextCursor(cursor)
        self.txt_chat_history.append(f"<b>[AI 전문가 처방]:</b> {response}<br>")
        self.txt_chat_history.ensureCursorVisible()

    def draw_advanced_plots(self):
        self.fig.clear()
        plt.style.use('seaborn-v0_8-whitegrid')
        selected_option = self.combo_model.currentText()
        target_model = self.models_list[0] if 'All Models' in selected_option else selected_option

        ax_cam = self.fig.add_subplot(121)
        if target_model in self.all_cam_images and self.all_cam_images[target_model].size > 0:
            ax_cam.imshow(self.all_cam_images[target_model])
            ax_cam.set_title(f"Grad-CAM XAI Layer ({target_model})", fontsize=11, fontweight='bold')
        ax_cam.axis('off')

        ax_chart = self.fig.add_subplot(122)
        probs = self.all_probabilities[target_model]
        top3_idx = np.argsort(probs)[-3:]
        y_pos = np.arange(3)
        ax_chart.barh(y_pos, probs[top3_idx], color='#4a90e2', height=0.5, edgecolor='black')
        ax_chart.set_yticks(y_pos)
        ax_chart.set_yticklabels([self.class_names[i] for i in top3_idx], fontsize=10, fontweight='bold')
        ax_chart.set_title(f"Top-3 Class Probability ({target_model})", fontsize=11, fontweight='bold')
        ax_chart.set_xlim(0, 1.05)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = TomatoApp()
    ex.show()
    sys.exit(app.exec())