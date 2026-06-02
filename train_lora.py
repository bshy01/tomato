import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

def main():
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = "/home/jaemu/tomato/pyqt_demo/tomato_qa_dataset.json"
    output_dir = "/home/jaemu/tomato/results/tomato_llm_lora_v3" # v3 폴더

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("📦 [v3] 데이터셋 로드 중...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # 공식 Chat Template 인코딩 (v2 규칙 유지)
    def tokenize_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        for i in range(len(examples['instruction'])):
            messages = [
                {"role": "system", "content": "당신은 농촌진흥청 지침을 준수하는 토마토 방제 전문가 AI 어드바이저입니다."},
                {"role": "user", "content": examples['instruction'][i]},
                {"role": "assistant", "content": examples['output'][i]}
            ]
            tokenized = tokenizer.apply_chat_template(
                messages, truncation=True, max_length=512, return_dict=True
            )
            batch_input_ids.append(tokenized['input_ids'])
            batch_attention_mask.append(tokenized['attention_mask'])
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("🚀 Qwen Base Model 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # 💡 [보정 1] 과적합 방지를 위해 LoRA 타겟을 가장 중요한 q, v 레이어로 압축합니다.
    peft_config = LoraConfig(
        r=8,              # 가볍고 예리하게 학습하도록 변경
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 💡 [보정 2] 학습률을 10배 낮춰(2e-5) 한국어 문장 붕괴 현상을 원천 차단합니다.
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,               # 🔥 핵심 수정 파트
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    print("🔥 토마토 특화 파인튜닝 v3 가동!")
    trainer = SFTTrainer(
        model=model, train_dataset=tokenized_dataset, peft_config=peft_config, args=training_args
    )
    trainer.train()

    print("💾 v3 가중치 저장 중...")
    trainer.model.save_pretrained(os.path.join(output_dir, "final_adapter_v3"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_adapter_v3"))
    print("✨ v3 학습 성공 종료.")

if __name__ == "__main__":
    main()