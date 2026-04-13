"""
노지 작물 해충 진단 (서브셋) - Qwen3.5-9B LoRA 파인튜닝 스크립트
대상: 썩덩나무노린재 + 정상 (2클래스)
데이터셋: data/ (로컬)
환경: 32GB+ VRAM (A5000/A6000), bf16 LoRA
"""

import json
import os
import random

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ════════════════════════════════════════
# 1. 데이터셋 경로
# ════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

assert os.path.exists(os.path.join(DATA_DIR, "train.jsonl")), \
    f"데이터셋이 없습니다: {DATA_DIR}/train.jsonl"
print(f"데이터셋 경로: {DATA_DIR}")

# ════════════════════════════════════════
# 2. 이미지 전처리 (크롭 → 디스크 저장)
# ════════════════════════════════════════

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
        if obj["grow"] == 33 and obj.get("points"):
            return obj["points"][0]
    return None


def preprocess_split(split="train"):
    """원본 이미지를 크롭/리사이즈하여 디스크에 저장하고 새 JSONL 생성"""
    jsonl_path = os.path.join(DATA_DIR, f"{split}.jsonl")
    out_dir = os.path.join(DATA_DIR, f"{split}_cropped")
    out_jsonl = os.path.join(DATA_DIR, f"{split}_cropped.jsonl")

    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r") as f:
            count = sum(1 for _ in f)
        print(f"  [{split}] 이미 전처리 완료: {count}건")
        return

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


random.seed(42)
print("=== 이미지 전처리 ===")
preprocess_split("train")
preprocess_split("val")
print("전처리 완료!")

# ════════════════════════════════════════
# 3. 데이터 로딩 (경로 기반 — RAM 절약)
# ════════════════════════════════════════


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


train_dataset = load_dataset_from_cropped_jsonl("train")
val_dataset = load_dataset_from_cropped_jsonl("val")
print(f"Train: {len(train_dataset)}건, Val: {len(val_dataset)}건")

# ════════════════════════════════════════
# 4. 모델 로딩
# ════════════════════════════════════════

import torch
from unsloth import FastVisionModel

# 모델 캐시를 /workspace에 저장 (Container disk 부족 방지)
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"

print("모델 로딩 중...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-9B",
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

# ════════════════════════════════════════
# 5. LoRA 설정
# ════════════════════════════════════════

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ════════════════════════════════════════
# 6. 학습
# ════════════════════════════════════════

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
        # batch size sweet spot 찾기: max_steps=3 추가 후 batch size를 바꿔가며 테스트
        # nvidia-smi로 VRAM 70~85%, GPU-Util 90%+ 되는 값이 최적
        per_device_train_batch_size=6,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name="pest-subset-qwen3.5",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    ),
)

print("학습 시작!")
trainer.train()
print("학습 완료!")

# ════════════════════════════════════════
# 7. 모델 저장
# ════════════════════════════════════════

LORA_DIR = "pest-detector-lora"

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"LoRA 어댑터 저장 완료: {LORA_DIR}")

# ════════════════════════════════════════
# 8. HuggingFace Hub에 업로드
# ════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if HF_TOKEN:
    print("HuggingFace Hub에 업로드 중...")
    model.push_to_hub(
        "YOUR_REPO_NAME",
        tokenizer,
        token=HF_TOKEN,
    )
    print("업로드 완료!")
else:
    print("HF_TOKEN이 설정되지 않아 업로드를 건너뜁니다.")
    print("업로드하려면: HF_TOKEN=hf_xxx python train.py")
