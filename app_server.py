import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__)

base_model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_dir = "/home/jaemu/tomato/results/tomato_llm_lora_v3/checkpoint-4835"

print("⚙️ 리눅스 GPU 가중치 세션 가동 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
print("✅ 리눅스 실시간 LLM 처방 백엔드 준비 완료!")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    chosen_class = data.get('chosen_class', 'healthy')
    question = data.get('question', '')

    messages = [
        {
            "role": "system",
            "content": f"당신은 농촌진흥청 지침을 준수하는 토마토 방제 전문가 AI 어드바이저입니다. 현재 비전 인식 시스템을 통해 해당 토마토 잎이 [{chosen_class}] 상태인 것이 정밀 판독되었습니다. 이 진단 결과를 바탕으로 농민에게 정확한 가이드라인과 약제를 처방하세요."
        },
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)