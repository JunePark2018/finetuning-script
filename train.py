"""
노지 작물 해충 진단 (서브셋) - Qwen3.5-9B LoRA 파인튜닝 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
데이터셋: /workspace/data (RunPod)
환경: 32GB+ VRAM (A5000/A6000), bf16 LoRA
"""

import json
import os
import random
import requests

from PIL import Image

# ════════════════════════════════════════
# Discord Webhook 설정
# ════════════════════════════════════════

DISCORD_BOT = {
    "username": "RunPod",
    "avatar_url": "https://i.imgur.com/0HOIh4r.png",
}
DISCORD_COLOR = 12648430
DISCORD_THUMBNAIL = "https://i.imgur.com/3ClKkzk.jpeg"


def notify_discord(message):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 메시지 전송"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message}, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def notify_discord_json(payload):
    """DISCORD_WEBHOOK_URL이 설정되어 있으면 JSON payload를 그대로 전송 (Embed 등)"""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Discord 알림 실패: {e}")


def discord_embed(description, thumbnail=False):
    """Embed payload 생성 헬퍼"""
    embed = {"description": description, "color": DISCORD_COLOR}
    if thumbnail:
        embed["thumbnail"] = {"url": DISCORD_THUMBNAIL}
    return {**DISCORD_BOT, "embeds": [embed]}


# ════════════════════════════════════════
# 하이퍼파라미터
# ════════════════════════════════════════

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 6))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 2))
                            # → Total Batch Size = BATCH_SIZE × GRAD_ACCUM × GPU 수

LORA_R = int(os.environ.get("LORA_R", 16))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", 16))

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 3))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 50))

Image.MAX_IMAGE_PIXELS = None

# ════════════════════════════════════════
# 1. 데이터셋 경로
# ════════════════════════════════════════

notify_discord_json(discord_embed("📂 [1/9] 데이터셋 경로를 확인합니다.", thumbnail=True))
try:
    DATA_DIR = "/workspace/data"

    assert os.path.exists(os.path.join(DATA_DIR, "train.jsonl")), \
        f"데이터셋이 없습니다: {DATA_DIR}/train.jsonl"
    print(f"데이터셋 경로: {DATA_DIR}")
    notify_discord_json(discord_embed("✅ [1/9] 데이터셋 경로 확인 완료. (train.jsonl, val.jsonl)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [1/9] 데이터셋 경로 확인 실패: {e}"))
    raise

# ════════════════════════════════════════
# 2. 이미지 전처리 (크롭 → 디스크 저장)
# ════════════════════════════════════════

notify_discord_json(discord_embed("🖼️ [2/9] 이미지 전처리를 시작합니다. (크롭 → 디스크 저장)"))
try:
    PROMPTS = [
        "이 사진에 있는 해충의 이름을 알려주세요.",
        "이 벌레는 무엇인가요?",
        "사진 속 해충을 식별해주세요.",
        "이 작물에 있는 해충의 종류가 무엇인가요?",
        "이 사진에서 어떤 해충이 보이나요?",
    ]

    SYSTEM_MSG = (
        "당신은 작물 해충 식별 전문가입니다. "
        '사진을 보고 해충의 이름만 한국어로 답하세요. '
        '해충이 없으면 "정상"이라고만 답하세요. '
        "부가 설명 없이 이름만 출력하세요."
    )

    BBOX_GROW_STAGE = 33


    def crop_to_bbox(img, bbox, padding_ratio=0.0):
        xtl, ytl = bbox["xtl"], bbox["ytl"]
        xbr, ybr = bbox["xbr"], bbox["ybr"]
        bw, bh = xbr - xtl, ybr - ytl
        pad_x, pad_y = int(bw * padding_ratio), int(bh * padding_ratio)
        x1 = max(0, xtl - pad_x)
        y1 = max(0, ytl - pad_y)
        x2 = min(img.width, xbr + pad_x)
        y2 = min(img.height, ybr + pad_y)
        return img.crop((x1, y1, x2, y2))


    def find_label_json(split, class_name, img_filename):
        json_path = os.path.join(DATA_DIR, split, class_name, img_filename + ".json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for obj in data["annotations"]["object"]:
            if obj["grow"] == BBOX_GROW_STAGE and obj.get("points"):
                return obj["points"][0]
        return None


    def preprocess_split(split="train"):
        """원본 이미지를 크롭하여 디스크에 저장하고 새 JSONL 생성"""
        jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        out_dir = os.path.join(DATA_DIR, f"{split}_cropped")
        out_jsonl = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")

        if os.path.exists(out_jsonl):
            with open(out_jsonl, "r") as f:
                count = sum(1 for _ in f)
            print(f"  [{split}] 이미 전처리 완료: {count}건")
            return count

        with open(jsonl_path, "r", encoding="utf-8") as f:
            total = sum(1 for _ in f)

        out_file = open(out_jsonl, "w", encoding="utf-8")
        count = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if (i + 1) % 500 == 0 or (i + 1) == total:
                    print(f"\r  [{split}] {i + 1}/{total} ({(i + 1) * 100 // total}%)", end="", flush=True)

                record = json.loads(line)
                messages = record["messages"]
                label = messages[-1]["content"][0]["text"]

                img_rel_path = None
                for msg in messages:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            img_rel_path = content["image"].replace("\\", "/")
                            break

                if img_rel_path is None:
                    continue

                parts = img_rel_path.split("/")
                class_name = parts[1]
                img_filename = parts[2]

                class_dir = os.path.join(out_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                base_name = os.path.splitext(img_filename)[0]
                out_filename = f"{base_name}.jpg"
                out_path = os.path.join(class_dir, out_filename)
                out_rel_path = f"{split}_cropped/{class_name}/{out_filename}"

                img_path = os.path.join(DATA_DIR, img_rel_path)
                img = Image.open(img_path).convert("RGB")

                if label == "정상":
                    result = img
                else:
                    bbox = find_label_json(split, class_name, img_filename)
                    if bbox:
                        r = random.random()
                        if r < 0.5:
                            result = img
                        elif r < 0.75:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.0)
                        else:
                            result = crop_to_bbox(img, bbox, padding_ratio=0.5)
                    else:
                        result = img

                result.save(out_path, "JPEG", quality=95)
                if result is not img:
                    result.close()
                img.close()

                new_record = {
                    "messages": [
                        {"role": "system", "content": [
                            {"type": "text", "text": SYSTEM_MSG}
                        ]},
                        {"role": "user", "content": [
                            {"type": "image", "image": out_rel_path},
                            {"type": "text", "text": random.choice(PROMPTS)},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": label}
                        ]},
                    ]
                }
                out_file.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                count += 1

        out_file.close()
        print(f"\n  [{split}] 완료: {count}건 → {out_dir}")
        return count


    random.seed(42)
    num_train = preprocess_split("train")
    num_val = preprocess_split("val")
    notify_discord_json(discord_embed(f"✅ [2/9] 전처리 완료! (train {num_train}건, val {num_val}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [2/9] 이미지 전처리 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 3. 데이터 로딩 (경로 기반 — RAM 절약)
# ════════════════════════════════════════

notify_discord_json(discord_embed("📊 [3/9] 데이터를 로딩합니다."))
try:
    def load_dataset_from_cropped_jsonl(split="train"):
        jsonl_path = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")
        dataset = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for msg in record["messages"]:
                    for content in msg["content"]:
                        if content["type"] == "image" and "image" in content:
                            content["image"] = os.path.join(DATA_DIR, content["image"])
                dataset.append(record)
        random.shuffle(dataset)
        return dataset

    random.seed(42)
    train_dataset = load_dataset_from_cropped_jsonl("train")
    val_dataset = load_dataset_from_cropped_jsonl("val")
    print(f"Train: {len(train_dataset)}건, Val: {len(val_dataset)}건")
    notify_discord_json(discord_embed(f"✅ [3/9] 데이터 로딩 완료! (Train {len(train_dataset)}건, Val {len(val_dataset)}건)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [3/9] 데이터 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 4. 모델 로딩
# ════════════════════════════════════════

notify_discord_json(discord_embed("🤖 [4/9] Qwen3.5-9B 모델을 로딩합니다."))
try:
    import torch
    from unsloth import FastVisionModel

    os.environ["HF_HOME"] = "/workspace/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-9B",
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    notify_discord_json(discord_embed("✅ [4/9] 모델 로딩 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [4/9] 모델 로딩 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 5. LoRA 설정
# ════════════════════════════════════════

notify_discord_json(discord_embed(f"⚙️ [5/9] LoRA 어댑터를 설정합니다. (r={LORA_R}, alpha={LORA_ALPHA})"))
try:
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    notify_discord_json(discord_embed("✅ [5/9] LoRA 설정 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [5/9] LoRA 설정 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 6. 학습
# ════════════════════════════════════════

notify_discord_json(discord_embed(f"@everyone\n🚀 [6/9] 학습을 시작합니다! ({NUM_EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM})"))
try:
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    FastVisionModel.for_training(model)

    OUTPUT_DIR = "pest-detector-qwen3.5"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=OUTPUT_DIR,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
            run_name="pest-subset-qwen3.5",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
        ),
    )

    trainer.train(resume_from_checkpoint=True)
    notify_discord_json(discord_embed("@everyone\n✅ [6/9] 학습 완료!"))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [6/9] 학습 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 7. 모델 저장
# ════════════════════════════════════════

notify_discord_json(discord_embed("💾 [7/9] LoRA 어댑터를 저장합니다."))
try:
    LORA_DIR = "pest-detector-lora"

    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    print(f"LoRA 어댑터 저장 완료: {LORA_DIR}")
    notify_discord_json(discord_embed("✅ [7/9] 모델 저장 완료! (pest-detector-lora/)"))
except Exception as e:
    notify_discord_json(discord_embed(f"❌ [7/9] 모델 저장 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 8. 추론 테스트
# ════════════════════════════════════════

notify_discord_json(discord_embed("@everyone\n🔍 [8/9] 추론 테스트를 시작합니다."))
try:
    model, tokenizer = FastVisionModel.from_pretrained(
        "pest-detector-lora",
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

    test_dir = os.path.join(DATA_DIR, "test", "썩덩나무노린재")
    test_images = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    TEST_IMAGE = os.path.join(test_dir, test_images[0])

    image = Image.open(TEST_IMAGE).convert("RGB")

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_MSG}
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "이 사진에 있는 해충의 이름을 알려주세요."},
        ]},
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
        temperature=0.1,
    )
    generated_ids = output[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    image.close()

    print(f"추론 테스트 - 이미지: {TEST_IMAGE}")
    print(f"추론 테스트 - 예측: {prediction}")
    notify_discord_json(discord_embed(f"@everyone\n✅ [8/9] 추론 테스트 완료! 예측: {prediction}"))
except Exception as e:
    notify_discord_json(discord_embed(f"@everyone\n❌ [8/9] 추론 테스트 중 에러 발생: {e}"))
    raise

# ════════════════════════════════════════
# 9. HuggingFace Hub에 업로드
# ════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_NAME = os.environ.get("HF_REPO_NAME", "")

if HF_TOKEN and HF_REPO_NAME:
    notify_discord_json(discord_embed("☁️ [9/9] HuggingFace Hub에 업로드합니다."))
    try:
        model.push_to_hub(
            HF_REPO_NAME,
            tokenizer,
            token=HF_TOKEN,
        )
        print("업로드 완료!")
        notify_discord_json(discord_embed("✅ [9/9] 업로드 완료! 모든 파이프라인이 끝났습니다! 🎉"))
    except Exception as e:
        notify_discord_json(discord_embed(f"❌ [9/9] HuggingFace Hub 업로드 중 에러 발생: {e}"))
        raise
else:
    missing = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if not HF_REPO_NAME:
        missing.append("HF_REPO_NAME")
    print(f"{', '.join(missing)} 미설정 — 업로드를 건너뜁니다.")
    notify_discord_json(discord_embed(f"✅ [9/9] Hub 업로드 건너뜀 ({', '.join(missing)} 미설정). 파이프라인 완료! 🎉"))
