import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_dir = "/home/jaemu/tomato/results/tomato_llm_lora_v3/final_adapter_v3"

    print("🚀 1. Qwen 베이스 모델 로드 중 (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("🧠 2. 파인튜닝된 토마토 방제 v3 LoRA 가중치 융합 중...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    model.eval()
    print("\n✨ v3 연산 엔진 준비 완료! 대화 시스템을 시작합니다.")
    print("=================================================================")
    print("질문을 입력하면 파인튜닝된 AI가 농진청 지침에 따라 답변합니다. (종료: q)")
    print("=================================================================\n")

    while True:
        user_question = input("🌾 농민 질문 입력: ").strip()
        if user_question.lower() == 'q' or not user_question:
            print("테스트 세션을 종료합니다.")
            break

        messages = [
            {"role": "system", "content": "당신은 농촌진흥청 지침을 준수하는 토마토 방제 전문가 AI 어드바이저입니다."},
            {"role": "user", "content": user_question}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        print("🤖 AI 분석 및 처방전 작성 중...", end="", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print("\r=================================================================")
        print(f"📋 [전문가 처방 결과]\n{response}")
        print("=================================================================\n")


if __name__ == "__main__":
    main()