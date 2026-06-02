import base64
import json
import os
import time
import requests


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_tomato_qa(image_path):
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"   [실패] 이미지 인코딩 에러 ({image_path}): {str(e)}")
        return None

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    parent_dir = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.basename(image_path)

    prompt = f"""
        당신은 농촌진흥청 가이드라인을 준수하는 대한민국 최고의 식물병리학자이자 토마토 방제 전문 지도사입니다.
        제공된 토마토 잎 사진(파일명: {file_name} / 추정 클래스: {parent_dir})을 정밀 분석하고, 농민들이 질문할 법한 예상 질문과 전문적인 답변 세트를 3개 생성하세요.

        ⚠️ [필수 준수 사항: 질병명 제한]
        답변에 등장하는 질병명은 반드시 아래 정의된 11개의 공식 명칭 내에서만 선택해야 하며, 절대 새로운 질병명(탄저병, 백음병, 백리스병 등)을 지어내지 마세요.
        - Bacterial_spot (세균성점무늬병)
        - Early_blight (겹무늬병)
        - Late_blight (역병)
        - Leaf_Mold (잎곰팡이병)
        - Septoria_leaf_spot (점무늬병)
        - Spider_mites (응애 피해)
        - Target_Spot (표적형점무늬병)
        - Tomato_Yellow_Leaf_Curl (토마토황화잎말림바이러스)
        - Tomato_mosaic_virus (토마토모자이크바이러스)
        - powdery_mildew (흰가루병)
        - healthy (정상/건강함)

        ⚠️ [필수 준수 사항: 농약 처방 제한]
        존재하지 않는 가짜 약제나 금지된 살충제(린다나 등)를 언급하지 마세요. 오직 아래에 지정된 실제 농업용 약제 서식만을 인용하세요.
        - 세균성점무늬병: 구리 수화제, 가스미민 수화제
        - 겹무늬병: 다코닐 수화제, 플루아지남 분무
        - 역병: 메탈락실 침투성 살균제, 전용 방제 유제
        - 잎곰팡이병: 폴리옥신 수화제
        - 응애: 아바메크린 유제 (응애 전용 약제)
        - 바이러스계열(황화잎말림, 모자이크): 치료 약제 없음. 매개충(담배가루이, 진딧물) 차단용 살충제 방제 및 감염 식물체 즉시 제거.
        - 흰가루병: 황산 가스 훈증법, 트리데모르프 수화제

        --------------------------------------------------
        출력 서식 예시 (농민의 현실적인 정황이 담긴 질문과 정밀한 처방전 구조):
        [
            {{"instruction": "최근 하우스 가습량이 늘면서 토마토 잎에 갈색 겹무늬 원형 반점이 번지는데 약제를 어떻게 해야 하나요?", "output": "비전 분석 결과 겹무늬병(Early_blight) 증상으로 확인됩니다. 습도를 낮추는 재배 환경 개선과 함께 농진청 지침에 따른 다코닐 수화제 또는 플루아지남 분무를 7-10일 주기로 교차 살포하시는 것을 권장합니다."}}
        ]
        --------------------------------------------------

        반드시 위 규칙에 정렬된 정밀한 JSON 구조로만 출력하세요. 이모지 사용은 전면 금지하며 다른 사족은 절대 붙이지 마세요:
        [
            {{"instruction": "농민의 예상 질문 1", "output": "전문가의 방제 및 농약 추천 답변 1"}},
            {{"instruction": "농민의 예상 질문 2", "output": "전문가의 방제 및 농약 추천 답변 2"}},
            {{"instruction": "농민의 예상 질문 3", "output": "전문가의 방제 및 농약 추천 답변 3"}}
        ]
        """

    data = {
        "model": "qwen2.5:7b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_image]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            result_text = response.json()['message']['content'].strip()

            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].split("```")[0].strip()

            qa_data = json.loads(result_text)
            return qa_data
        else:
            print(f"   [실패] Ollama 서버 응답 코드 에러: {response.status_code}")
            return None

    except Exception as e:
        print(f"   [실패] 로컬 연산 및 JSON 파싱 실패: {str(e)}")
        return None


def process_target_directory(root_folder, output_json_path):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    target_images = []

    print(f"🔍 타겟 폴더 탐색 시작: {root_folder}")

    for current_dir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(current_dir, file)
                target_images.append(full_path)

    total_count = len(target_images)
    print(f"📂 탐색 완료! 발견된 총 이미지 파일 개수: {total_count}개")

    if total_count == 0:
        print("❌ 처리할 이미지 파일이 존재하지 않습니다. 경로를 다시 확인하세요.")
        return

    final_dataset = []
    success_count = 0

    for idx, img_path in enumerate(target_images, 1):
        print(f"🔄 [{idx}/{total_count}] 처리 중: {os.path.basename(img_path)}")

        qa_results = generate_tomato_qa(img_path)

        if qa_results:
            final_dataset.extend(qa_results)
            success_count += 1
            print(f"   [성공] 3개의 Q&A 세트 추출 완료.")

        if idx % 10 == 0 or idx == total_count:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_dataset, f, indent=2, ensure_ascii=False)
            print(f"   💾 [{idx}개 캐시 지점] 누적 데이터 파일 세이브 완료.")

        time.sleep(0.5)

    print(f"✨ 작업이 모두 완료되었습니다!")
    print(f"   - 총 이미지: {total_count}개 중 {success_count}개 성공")
    print(f"   - 빌드된 누적 Q&A 총 문장 개수: {len(final_dataset)}개")


if __name__ == "__main__":
    INPUT_DIR = "/shared/data/tomato/train"
    OUTPUT_FILE = "/home/jaemu/tomato/pyqt_demo/tomato_qa_dataset.json"

    process_target_directory(INPUT_DIR, OUTPUT_FILE)